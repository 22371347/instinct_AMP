#!/usr/bin/env python3
"""G1 本地 MuJoCo 轻量回放 / 模型检查（无需 Isaac Sim）

用途
----
1. `--dump-model`  打印 G1 MJCF 的结构（关节/执行器/限位），用于核对关节序
2. `--onnx-info`   打印 ONNX 策略的输入输出维度（核对观测/动作维度）
3. 回放            加载策略在 MuJoCo 里闭环运行，可开 viewer 或录制成 mp4

⚠️ 能力边界（诚实交代）
--------------------
本地 MuJoCo **能**验证：策略加载、IO 维度、关节序映射、PD 跟踪、步态合理性。
本地 MuJoCo **不能**验证：IsaacLab 的地形/课程/域随机化，以及 InstinctLab 的
`depth_image` / `NoisyGroupedRayCasterCamera` / `VolumePoints` / `feet_at_plane`
等专有传感器。**越障行为的真实验证必须在云端完成。**

用法
----
    # 1) 看模型
    python tools/g1_mujoco_play.py --dump-model

    # 2) 看策略 IO
    python tools/g1_mujoco_play.py --onnx-info --policy /path/to/policy.onnx

    # 3) 回放（默认开 viewer）
    python tools/g1_mujoco_play.py --policy /path/to/policy.onnx --duration 15
    python tools/g1_mujoco_play.py --policy /path/to/policy.onnx --no-view --record out.mp4

    # 4) ⭐ 键盘遥控（WASD/QE）——平地技能验证
    #    开窗后先点一下窗口让它获得焦点，然后：
    #      W/S = 前进/后退(vx)   A/D = 左移/右移(vy)   Q/E = 左转/右转(wz)
    #      [ ] = 调增量         空格/X = 停止          R   = 复位
    python tools/g1_mujoco_play.py --policy /path/to/policy.onnx --duration 0
    python tools/g1_mujoco_play.py --policy /path/to/policy.onnx --duration 0 --cmd-step 0.2
    #    指令会被夹在训练范围 `CMD_LIMITS` 内（超范围 = 分布外，摔了不代表策略差）。
    #    离线单测（不开窗）：python tools/test_keyboard_teleop.py
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

import numpy as np

# 允许直接 `python tools/xxx.py` 运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_XML = Path(__file__).resolve().parent.parent / "assets/g1_29dof_popsicle_play.xml"
#: 备用：原始 InstinctLab 资产（注意这是 23-DoF + 双手变体，nu=37，**不适合** 29-DoF 跑酷策略）
FALLBACK_XML = (
    Path(__file__).resolve().parent.parent
    / "repos/InstinctLab/source/instinctlab/instinctlab/assets/resources/unitree_g1/scene.xml"
)

def _per_joint_action_scale() -> np.ndarray:
    """逐关节动作尺度（beyondmimic_action_scale = 0.25*effort/kp）。

    ⚠️ 不是全局 0.25：实测范围 0.0745(wrist) ~ 0.5476(hip/knee 类)。
    用错会导致大关节幅度偏小、小关节幅度偏大。
    """
    try:
        import g1_params  # type: ignore
        from joint_order import MUJOCO_ORDER  # type: ignore

        return np.array([g1_params.action_scale_for(n) for n in MUJOCO_ORDER], dtype=np.float64)
    except Exception:  # noqa: BLE001
        return np.full(29, 0.25, dtype=np.float64)


ACTION_SCALE = _per_joint_action_scale()

# ---------------------------------------------------------------------------
# 策略封装
# ---------------------------------------------------------------------------


class OnnxPolicy:
    def __init__(self, path: str):
        import onnxruntime as ort

        self.path = path
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        # 输入最后一维 = 策略期望的观测维度（用来自动选择 obs builder）
        shape = self.sess.get_inputs()[0].shape
        self.input_dim = int(shape[-1]) if shape and isinstance(shape[-1], int) else None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        out = self.sess.run(None, {self.input_name: obs.astype(np.float32)})
        return np.asarray(out[0])


class TorchPolicy:
    def __init__(self, path: str):
        import torch

        self.path = path
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.nn.Module):
            obj.eval()
            self.module = obj
            # torch 分支不解析维度；如需非默认观测布局，用 `--obs-mode` 显式指定
            self.input_dim = None
        else:
            raise SystemExit(
                "checkpoint 不是可直接调用的 nn.Module。\n"
                "建议优先使用云端 `--exportonnx` 导出的 ONNX（本脚本支持得更好）。"
            )

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            t = torch.from_numpy(obs.astype(np.float32))
            out = self.module(t)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.cpu().numpy()


def load_policy(path: str, policy_type: str = "auto"):
    if policy_type == "auto":
        policy_type = "onnx" if path.endswith(".onnx") else "torch"
    return OnnxPolicy(path) if policy_type == "onnx" else TorchPolicy(path)


class ObsNormalizer:
    """复现 `instinct_rl` 的观测归一化。

    ⭐ **不加载它会直接得到垃圾动作**：`OnPolicyRunner.export_as_onnx` 在导出时
    是先把 obs 归一化、再把**归一化器单独存成 `policy_normalizer.npz`** 的：

        if "policy" in self.normalizers:
            obs = self.normalizers["policy"](obs)              # 先归一化
            self.normalizers["policy"].export(...npz)           # 统计量另存
        self.alg.actor_critic.export_as_onnx(obs, ...)

    ⇒ **ONNX 的输入是归一化后的观测**。直接喂原始观测 = OOD，
      实测会在标称站姿下输出 ±3.8 rad 的“疯狂”动作、机器人立即摔倒。

    公式（`instinct_rl/modules/normalizer.py`）：`(x - mean) / (std + eps)`
    """

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=True)
        self.mean = np.asarray(z["mean"], dtype=np.float64).reshape(1, -1)
        self.std = np.asarray(z["std"], dtype=np.float64).reshape(1, -1)
        self.eps = float(np.asarray(z["eps"]).reshape(-1)[0]) if "eps" in z.files else 1e-2
        self.dim = self.mean.shape[1]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        o = np.asarray(obs, dtype=np.float64)
        return ((o - self.mean) / (self.std + self.eps)).astype(np.float32)


def find_normalizer(policy_path: str) -> str | None:
    """在策略文件旁边找 `policy_normalizer.npz`（云端 export 的默认位置）。"""
    cand = os.path.join(os.path.dirname(os.path.abspath(policy_path)), "policy_normalizer.npz")
    return cand if os.path.isfile(cand) else None


# ---------------------------------------------------------------------------
# 四元数 / 观测拼接工具
# ---------------------------------------------------------------------------

# IsaacLab 的 `projected_gravity_b` 用的重力方向（单位向量，wxyz 约定）
GRAVITY_VEC_W = np.array([0.0, 0.0, -1.0])


def rotmat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """四元数(w,x,y,z) → 旋转矩阵 R（body→world）。"""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_apply_inverse_wxyz(quat_wxyz: np.ndarray, vec_w: np.ndarray) -> np.ndarray:
    """world 系向量 → body 系。等价于 IsaacLab 的 `math_utils.quat_apply_inverse`。"""
    return rotmat_wxyz(quat_wxyz).T @ np.asarray(vec_w, dtype=np.float64)


def projected_gravity_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """IsaacLab `asset.data.projected_gravity_b` 的等价实现。

    直立时返回 (0, 0, -1) —— 这是判对错的关键：如果得到 (0,0,+1) 就是符号错了。
    """
    return quat_apply_inverse_wxyz(quat_wxyz, GRAVITY_VEC_W)


# ---------------------------------------------------------------------------
# 观测构造
# ---------------------------------------------------------------------------

class ObsBuilder:
    """按 InstinctLab parkour 的 `policy` 观测组拼接观测。

    对的配置（repos/InstinctLab/.../tasks/parkour/config/parkour_env_cfg.py）：
        base_ang_vel       scale 0.25   history 8
        projected_gravity  scale 1.0    history 8
        velocity_commands  scale 1.0    history 8   (commands 由外部给定)
        joint_pos          (rel)        scale 1.0    history 8
        joint_vel          scale 0.05   history 8
        actions            (last_action)            history 8
        depth_image        ← 本地无法提供，见文件头“能力边界”

    ⚠️ 拼接顺序/缩放/历史长度**必须与训练配置一致**。请以 `--onnx-info` 打印的
    输入维度为准自行核对，本类只是「可工作的默认值」。
    """

    def __init__(self, model, data, default_joint_pos: np.ndarray, history: int = 8, use_depth: bool = False):
        self.model = model
        self.data = data
        self.default_joint_pos = default_joint_pos
        self.history = history
        self.use_depth = use_depth
        self.reset()

    def reset(self):
        self.hist: dict[str, list[np.ndarray]] = {}

    def _push(self, key: str, value: np.ndarray) -> np.ndarray:
        buf = self.hist.setdefault(key, [np.zeros_like(value) for _ in range(self.history)])
        buf.append(value)
        del buf[0]
        return np.concatenate(buf, axis=-1)

    def build(self, cmd: np.ndarray, last_action: np.ndarray) -> np.ndarray:
        d = self.data
        # 角速度（mujoco free joint：qvel[3:6] 已经是 **body 系** 角速度）
        ang_vel = d.qvel[3:6].copy() * 0.25
        # 投影重力
        # ⚠️ IsaacLab 的 `projected_gravity` = `asset.data.projected_gravity_b`
        #    = quat_apply_inverse(root_quat_w, (0,0,-1))  → 直立时 (0,0,-1)。
        #    这里原来算的是 R^T·ẑ = +[2(xz-wy), 2(yz+wx), 1-2(x²+y²)]，**符号反了**。
        #    （IsaacLab 源码：`projected_gravity_b` 用 GRAVITY_VEC_W=(0,0,-1)；
        #      InstinctLab 的 `instinct_mdp.projected_gravity` 就是 isaaclab 那个，
        #      见 instinctlab/envs/mdp/observations/expanded.py 的 import 头）
        grav = projected_gravity_wxyz(d.qpos[3:7])
        joint_pos_rel = d.qpos[7:] - self.default_joint_pos
        joint_vel = d.qvel[6:] * 0.05

        parts = [
            self._push("ang_vel", ang_vel),
            self._push("grav", grav),
            self._push("cmd", cmd),
            self._push("joint_pos", joint_pos_rel),
            self._push("joint_vel", joint_vel),
            self._push("action", last_action.copy()),
        ]
        if self.use_depth:
            raise NotImplementedError(
                "本地无法生成 InstinctLab 的 depth_image 观测。"
                "带深度的策略只能回放「非深度」分支，或到云端用 Isaac Sim 验证。"
            )
        return np.concatenate(parts, axis=-1)


class AmpWalk626ObsBuilder:
    """`Instinct-Amp-Walk-G1-v0` 的 **626 维** actor 特权观测拼接。

    与上面的 `ObsBuilder`（InstinctLab parkour：8 帧历史 + scale 0.25/0.05）
    完全不同：本任务是**单帧、无额外缩放、无历史**，按 `G1AmpWalkPolicyObsCfg`
    的**声明顺序**拼接：

        [ height_scan(527) | velocity_command(3) | base_lin_vel(3)
          | projected_gravity(3) | base_ang_vel(3)
          | joint_pos_rel(29) | joint_vel(29) | last_action(29) ] = 626

    每一项的语义都对着源码钉过（不是猜的）：

    | 项 | IsaacLab 实现 | MuJoCo 侧取法 |
    |---|---|---|
    | height_scan | `sensor_z - hit_z - 0.5`，clip(-1,1) | 平地 → 常数 `torso_z - 0.5` |
    | velocity_command | `vel_command_b`（body 系） | 外部给定指令 |
    | base_lin_vel | `root_lin_vel_b` | `R^T @ qvel[0:3]` |
    | projected_gravity | `R^T @ (0,0,-1)` | `projected_gravity_wxyz()`（直立=(0,0,-1)）|
    | base_ang_vel | `root_ang_vel_b`，**无缩放** | `qvel[3:6]`（free joint 角速度本就在 body 系）|
    | joint_pos_rel | `joint_pos - default_joint_pos` | 需按**关节名**重排到 IsaacLab 序 |
    | joint_vel | `joint_vel`，**无缩放** | 同上重排 |
    | last_action | `action_manager.action` | 上一步策略输出（policy 序，**不重排**）|

    来源核对：`instinctlab/envs/mdp/observations/expanded.py` 的 import 头
    显示 `projected_gravity / base_ang_vel / last_action / joint_pos_rel / joint_vel`
    全部转发自 `isaaclab.envs.mdp`；`height_scan` 的具体公式见
    `isaaclab/envs/mdp/observations.py` 的 `height_scan`（offset 默认 0.5）。

    ⚠️ **height_scan 是特权量**：MuJoCo 里没有 527 条射线的地形扫描器。
    平地近似成常数是**精确**的（每条射线都命中 z=0 地面，公式退化成
    `torso_z - 0.5`）；地形不平坦时这个近似不成立，回放结论不可信。
    """

    N_SCAN = 527          # 31 x 17：0.1 m 分辨率、3.0 x 1.6 m
    SCAN_OFFSET = 0.5     # IsaacLab `height_scan` 默认 offset（我们的 ObsTerm 没传 → 0.5）
    SCAN_CLIP = (-1.0, 1.0)

    def __init__(self, model, data, default_joint_pos_mjcf, n_joint=29,
                 torso_body_name="torso_link", obs_joint_order="npz"):
        import mujoco

        try:
            from joint_order import (ISAACLAB_ORDER, MUJOCO_ORDER, NPZ_ORDER,
                                     mjcf_to_order_index)  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "需要同目录的 joint_order.py（提供各套关节序映射）："
                f"{exc}"
            )

        self.model, self.data = model, data
        self.n_joint = n_joint
        # ⭐ 关键：观测里的 `joint_pos_rel` / `joint_vel` 用的是 `SceneEntityCfg("robot")`
        # 默认（joint_names=None → joint_ids=slice(None)）⇒ **articulation 序**，
        # 与「动作」用的 OMNI_LOCO 序 **不是**同一套！
        # 候选：articulation（= Unitree URDF 声明序）/ omiloco / mjcf
        cand = {
            "isaaclab": ISAACLAB_ORDER,
            "omniloco": ISAACLAB_ORDER,
            "npz": NPZ_ORDER,
            "articulation": NPZ_ORDER,   # 云端实测确认后固定到这个
            "mjcf": MUJOCO_ORDER,
        }
        if obs_joint_order not in cand:
            raise SystemExit(f"未知 --obs-joint-order={obs_joint_order}，可选 {list(cand)}")
        self.obs_order_name = obs_joint_order
        self.mj_to_obs = np.array(mjcf_to_order_index(cand[obs_joint_order]), dtype=int)
        # 默认站姿按**观测序**重排，供 joint_pos_rel 使用
        self.default_policy = np.asarray(default_joint_pos_mjcf,
                                         dtype=np.float64)[self.mj_to_obs]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, torso_body_name)
        if bid < 0:
            raise SystemExit(
                f"MJCF 里找不到 body '{torso_body_name}' —— height_scan 需要它的 z 坐标"
            )
        self.torso_body_id = bid
        print(f"[obs] AMP-Walk 626 维：height_scan={self.N_SCAN} "
              f"(平地常数 torso_z-{self.SCAN_OFFSET})，torso_body_id={bid}")
        print(f"[obs] 观测关节块用 **{obs_joint_order}** 序"
              f"（与动作的 OMNI_LOCO 序不同；重排={'生效' if not np.array_equal(self.mj_to_obs, np.arange(n_joint)) else 'no-op'}）")

    def build(self, cmd: np.ndarray, last_action: np.ndarray) -> np.ndarray:
        d = self.data
        # 1) height_scan(527)：平地精确解
        scan = np.full(self.N_SCAN, d.xpos[self.torso_body_id, 2] - self.SCAN_OFFSET)
        np.clip(scan, *self.SCAN_CLIP, out=scan)
        # 2) velocity_command(3)，body 系
        cmd = np.asarray(cmd, dtype=np.float64)
        # 3) base_lin_vel(3)：world → root(pelvis) 系
        quat = d.qpos[3:7]  # w, x, y, z
        lin_b = quat_apply_inverse_wxyz(quat, d.qvel[0:3])
        # 4) projected_gravity(3)
        grav = projected_gravity_wxyz(quat)
        # 5) base_ang_vel(3)：free joint 的 qvel[3:6] 已是 body 系，且**无缩放**
        ang_b = d.qvel[3:6].copy()
        # 6/7) joint_pos_rel / joint_vel：MJCF 序 → **观测序**（articulation/npz）
        jp = d.qpos[7:7 + self.n_joint][self.mj_to_obs] - self.default_policy
        jv = d.qvel[6:6 + self.n_joint][self.mj_to_obs]
        # 8) last_action（policy 序）
        return np.concatenate(
            [scan, cmd, lin_b, grav, ang_b, jp, jv,
             np.asarray(last_action, dtype=np.float64)]
        )


# ---------------------------------------------------------------------------
# 模型检查
# ---------------------------------------------------------------------------

def dump_model(xml_path: str) -> None:
    import mujoco

    m = mujoco.MjModel.from_xml_path(xml_path)
    print(f"MJCF       : {xml_path}")
    print(f"nq={m.nq}  nv={m.nv}  nu={m.nu}  nbody={m.nbody}  njnt={m.njnt}")
    print(f"timestep   : {m.opt.timestep}")

    print("\n--- 执行器 (ctrl 顺序) ---")
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
        jid = int(m.actuator_trnid[i, 0])
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"jnt{jid}"
        lo, hi = m.actuator_ctrlrange[i]
        print(f"  [{i:>2}] {name:<34} -> joint={jname:<30} ctrl_range=[{lo:+.3f}, {hi:+.3f}]")

    print("\n--- 全部关节 ---")
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"jnt{j}"
        jtype = m.jnt_type[j]
        typ = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(int(jtype), str(jtype))
        if int(jtype) == 3:
            lo, hi = m.jnt_range[j]
            print(f"  [{j:>2}] {name:<32} {typ:<6} range=[{lo:+.4f}, {hi:+.4f}]")
        else:
            print(f"  [{j:>2}] {name:<32} {typ}")

    print("\n--- 与已知关节序常量比对 ---")
    try:
        from joint_order import MUJOCO_ORDER  # type: ignore

        ctrl_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0])) for i in range(m.nu)
        ]
        if m.nu != 29:
            print(f"⚠️  执行器数量 {m.nu} != 29，与 29-DoF 假设不符，请确认用的是哪个变体")
        elif list(ctrl_names) == list(MUJOCO_ORDER):
            print("✅ ctrl 顺序与 MUJOCO_ORDER 一致")
        else:
            print("⚠️  ctrl 顺序与内置 MUJOCO_ORDER 不同，实际为：")
            for i, n in enumerate(ctrl_names):
                print(f"   [{i:>2}] {n}")
    except ImportError:
        pass


def onnx_info(path: str) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print(f"ONNX: {path}")
    print("--- 输入 ---")
    for i in sess.get_inputs():
        print(f"  {i.name:<24} shape={i.shape} dtype={i.type}")
    print("--- 输出 ---")
    for o in sess.get_outputs():
        print(f"  {o.name:<24} shape={o.shape} dtype={o.type}")
    print(
        "\n提示：请用上面的输入维度反推观测拼接量。"
        "若为 3D(含时间维) 说明策略期望历史帧堆叠，回放时需自行维护 history。"
    )


# ---------------------------------------------------------------------------
# 回放
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 键盘遥控（WASD/QE）—— 平地技能验证
# ---------------------------------------------------------------------------

#: 训练时的速度指令范围，= `instinct_xu` 的 `AMP_WALK_COMMAND_RANGES`
#: （`tasks/locomotion/config/g1/amp_walk_env_cfg.py:92`）。
#: 键盘遥控**必须**夹在这个范围内：超出范围属于分布外指令，摔了不代表策略差。
CMD_LIMITS: tuple[tuple[float, float], ...] = ((-1.0, 1.2), (-0.5, 0.5), (-0.8, 0.8))

#: 键码 → 动作名。**两种键码语义都注册**：
#:   · GLFW 键码：`glfw.KEY_W == 87`
#:   · 字符码：   `ord('W') == 87`（恰好相同）、`ord('w') == 119`
#: `mujoco.viewer.launch_passive(key_callback=...)` 只声明 `Callable[[int], None]`，
#: 回传的到底是哪一种**没有文档保证**，所以两套都填，避免「按了没反应」。
_KEY_ACTIONS: dict[int, str] = {}


def _register_keys() -> None:
    import glfw

    spec = [
        ("W", "vx+", "w"), ("S", "vx-", "s"),
        ("A", "vy+", "a"), ("D", "vy-", "d"),
        ("Q", "wz+", "q"), ("E", "wz-", "e"),
        ("SPACE", "zero", " "), ("R", "reset", "r"), ("X", "zero", "x"),
        ("LEFT_BRACKET", "step-", "["), ("RIGHT_BRACKET", "step+", "]"),
    ]
    for key_name, action, ascii_char in spec:
        code = getattr(glfw, f"KEY_{key_name}", None)
        if code is not None:
            _KEY_ACTIONS[int(code)] = action
        for ch in {ascii_char, ascii_char.upper()}:
            _KEY_ACTIONS[ord(ch)] = action


_register_keys()


def reset_state(model, data) -> None:
    """把机器人复位到 MJCF 的初始 keyframe（`R` 键与启动时**共用这一份逻辑**）。

    ⚠️ **必须跟一次 `mj_forward`**：`mj_resetDataKeyframe` 只写了 `qpos`/`qvel`，
    `data.xpos`（body 世界坐标）等派生量仍是**未初始化**的。`height_scan` 要用
    torso 的 z，不 forward 就会拿到垃圾值（实测首帧 scan=0.5 而非 0.32）→
    第一帧观测错 → 动作飙到 4 rad → 因为 `last_action` 是无裁剪反馈，
    直接指数发散到 NaN。
    """
    import mujoco

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model)
    mujoco.mj_forward(model, data)


class KeyboardCommandController:
    """WASD/QE 键盘遥控器（线程安全）。

    | 按键 | 作用 |
    |---|---|
    | `W` / `S` | vx += / -= step（前进 / 后退）|
    | `A` / `D` | vy += / -= step（左移 / 右移）|
    | `Q` / `E` | wz += / -= step（左转 / 右转）|
    | `[` / `]` | 调小 / 调大 step |
    | `空格` / `X` | 指令清零（原地站立）|
    | `R` | 复位机器人到初始 keyframe |

    设计取舍：**每次按键 = 一次指令增量**，而不是「按住才动」。因为
    `key_callback` 没有任何文档保证会同时回传 RELEASE 事件，增量模式在
    「只给 PRESS」和「PRESS + RELEASE 都给」两种情况下**都正确**。

    线程安全：回调由 viewer 线程调用，主循环在另一条线程，故用锁保护。
    """

    STEP_PRESETS: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)

    def __init__(self, cmd0, step: float = 0.1, limits=CMD_LIMITS, verbose: bool = True):
        self._lock = threading.Lock()
        self._limits = tuple(tuple(float(x) for x in pair) for pair in limits)
        self._verbose = verbose
        self._reset = False
        # 初值先夹回训练范围内：`--cmd` 给了超范围的值也不至于一上来就摔。
        self._cmd = [
            float(np.clip(float(v), self._limits[i][0], self._limits[i][1]))
            for i, v in enumerate(cmd0)
        ]
        # step 落到最近的预设上，便于 `[` / `]` 连续调整
        self._step = min(self.STEP_PRESETS, key=lambda p: abs(p - float(step)))

    # -- 由 viewer 线程调用 --------------------------------------------------
    def on_key(self, keycode: int) -> None:
        action = _KEY_ACTIONS.get(int(keycode))
        if action is None:
            return
        with self._lock:
            do_reset = action == "reset"
            if do_reset:
                self._reset = True
            elif action == "zero":
                self._cmd = [0.0, 0.0, 0.0]
            elif action in ("step+", "step-"):
                i = self.STEP_PRESETS.index(self._step)
                i = min(len(self.STEP_PRESETS) - 1, i + 1) if action == "step+" else max(0, i - 1)
                self._step = self.STEP_PRESETS[i]
            else:
                axis = {"vx": 0, "vy": 1, "wz": 2}[action[:2]]
                sign = 1.0 if action.endswith("+") else -1.0
                lo, hi = self._limits[axis]
                self._cmd[axis] = float(np.clip(self._cmd[axis] + sign * self._step, lo, hi))
            snapshot, step = tuple(self._cmd), self._step
        if self._verbose:
            if do_reset:
                print("[teleop] R -> 请求复位", flush=True)
            else:
                print(f"[teleop] vx={snapshot[0]:+.2f} vy={snapshot[1]:+.2f} "
                      f"wz={snapshot[2]:+.2f}   (step={step:.2f})", flush=True)

    # -- 由主循环调用 --------------------------------------------------------
    def poll(self):
        """返回 `(cmd[3], 复位标志)`；复位标志读取后即清零。"""
        with self._lock:
            do_reset = self._reset
            self._reset = False
            return np.array(self._cmd, dtype=np.float64), do_reset

    def hud(self):
        """给 `viewer.set_texts()` 用的两个文本框。"""
        import mujoco

        with self._lock:
            c, step, lim = tuple(self._cmd), self._step, self._limits
        top = (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            f"vx={c[0]:+.2f}  vy={c[1]:+.2f}  wz={c[2]:+.2f}",
            f"step={step:.2f}",
        )
        bottom = (
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            "W/S=vx  A/D=vy  Q/E=wz  [ ]=step  Space/X=stop  R=reset",
            f"vx{lim[0]} vy{lim[1]} wz{lim[2]}",
        )
        return [top, bottom]

    def help_text(self) -> str:
        lim = self._limits
        return (
            "=============== WASD/QE 键盘遥控（先点一下 viewer 窗口让它获得焦点）===============\n"
            "   W / S      前进 / 后退      (vx)\n"
            "   A / D      左移 / 右移      (vy)\n"
            "   Q / E      左转 / 右转      (wz)\n"
            "   [ / ]      调小 / 调大 每次按键的增量\n"
            "   空格 / X   指令清零（原地站立）\n"
            "   R          复位到初始 keyframe\n"
            f"   指令范围（= 训练范围）: vx{lim[0]}  vy{lim[1]}  wz{lim[2]}\n"
            "   关闭窗口即退出；每 2 秒打印一次状态。\n"
            "================================================================================"
        )


def play(args) -> int:
    import mujoco

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    if args.dump_model:
        dump_model(args.xml)
        return 0

    # 默认站姿：用 MJCF 的 keyframe（若有），否则用 qpos0
    if model.nkey > 0:
        default_qpos = model.key_qpos[0].copy()
        print(f"[init] 使用 keyframe 0：{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, 0)}")
    else:
        default_qpos = model.qpos0.copy()
        print("[init] 无 keyframe，使用 qpos0")
    # 复位到初始状态（含必须的 `mj_forward`，见 `reset_state` 的 docstring；
    # `R` 键走的是同一个函数，逻辑只有一份）
    reset_state(model, data)

    n_joint = model.nu
    default_joint_pos = default_qpos[7 : 7 + n_joint].copy()
    # ⚠️ action_scale 在下面（知道 obs_mode 之后）才能定 —— 两种任务的约定不同，
    #    见那里的注释（这是个能把机器人直接搞摔的坑）。

    # 策略
    policy = load_policy(args.policy, args.policy_type) if args.policy else None
    if policy is None:
        print("[warn] 未提供 --policy，仅做零动作站立（用于验证模型/PD）")

    # --- 观测归一化（⭐ 不加载会得到垃圾动作，见 ObsNormalizer 注释）---
    normalizer = None
    if args.policy and not args.no_normalizer:
        npath = args.normalizer or find_normalizer(args.policy)
        if npath and os.path.isfile(npath):
            normalizer = ObsNormalizer(npath)
            print(f"[norm] 已加载观测归一化: {npath}  (dim={normalizer.dim}, eps={normalizer.eps:g})")
        else:
            print("[norm] ⚠️ 未找到 policy_normalizer.npz —— 若该策略训练时启用了观测归一化，"
                  "现在得到的就是垃圾动作。用 --normalizer 指定路径，或 --no-normalizer 明确跳过。")

    # --- 观测 builder 选择（按策略输入维度自动判定）---
    obs_mode = args.obs_mode
    if obs_mode == "auto":
        dim = getattr(policy, "input_dim", None) if policy else None
        if dim == 626:
            obs_mode = "amp626"
        elif dim is None:
            if args.selftest_obs:
                obs_mode = "amp626"   # 自检默认校验本任务的 626 维
            else:
                raise SystemExit(
                    "--obs-mode auto 无法从策略文件推断观测维度（torch 分支 / 动态维度）。\n"
                    "请显式指定：--obs-mode amp626（本任务）或 --obs-mode legacy（parkour）"
                )
        else:
            obs_mode = "legacy"
    if policy is None and not args.selftest_obs:
        obs_builder = None
    elif obs_mode == "amp626":
        obs_builder = AmpWalk626ObsBuilder(model, data, default_joint_pos, n_joint=n_joint,
                                           obs_joint_order=args.obs_joint_order)
    else:
        obs_builder = ObsBuilder(model, data, default_joint_pos, history=args.history)
    print(f"[obs] obs_mode={obs_mode}"
          + (f"  策略期望维度={getattr(policy, 'input_dim', None)}" if policy else "")
          + f"  （自检：实际拼出 {626 if obs_mode == 'amp626' else '?'} 维）")

    # 动作重排：策略输出是 IsaacLab(policy) 序，MJCF 的 ctrl 是 MJCF 序。
    # 两套顺序不同（IsaacLab 肩优先左右交错；MJCF 按 body 遍历）→ 不重排会完全错位。
    if args.joint_order == "isaaclab":
        try:
            from joint_order import policy_to_mujoco_index  # type: ignore
            pol_to_mj = np.array(policy_to_mujoco_index(), dtype=int)
            print(f"[act] 动作按关节名从 policy 序 → MJCF 序重排"
                  f"（{'no-op' if np.array_equal(pol_to_mj, np.arange(n_joint)) else '生效'}）")
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(f"--joint-order isaaclab 需要 joint_order.py：{exc}")
    else:
        pol_to_mj = np.arange(n_joint)
        print("[act] 不做关节重排（--joint-order as-is）")

    # --- 动作尺度（⭐ 这是个能把机器人直接搞摔的坑）---
    #   本仓库（OmniLoco/AMP-Walk）的 `JointPositionAction` 是 **scale = 1.0**：
    #       target = default_joint_pos + 1.0 * action        （action 直接是弧度偏移）
    #   而 `beyondmimic` 那套（parkour/vault）用的是 per-joint
    #       action_scale = 0.25 * effort / kp                 （例：0.0745 ~ 0.5476）
    #   用错会把目标角缩小 2~13 倍 → 腿推不动 → 一启动就摔。
    #   契约校验会打印 `scale=1.0`，与我们的 cfg 一致。
    scale_mode = args.action_scale_mode
    if scale_mode == "auto":
        scale_mode = "unity" if obs_mode == "amp626" else "beyondmimic"
    if scale_mode == "unity":
        action_scale = np.ones(n_joint, dtype=np.float64)
        print(f"[ctrl] action_scale = 1.0（unity，与 IsaacLab 的 scale=1.0 一致）")
    elif ACTION_SCALE.shape[0] != n_joint:
        action_scale = np.full(n_joint, args.action_scale, dtype=np.float64)
        print(f"[warn] MJCF 有 {n_joint} 个执行器（期望 29），回退到全局 action_scale={args.action_scale}")
    else:
        action_scale = ACTION_SCALE.copy()
        print(f"[ctrl] 逐关节 action_scale（beyondmimic）: "
              f"{action_scale.min():.5f} ~ {action_scale.max():.5f}")

    cmd = np.array(args.cmd, dtype=np.float64)
    last_action = np.zeros(n_joint, dtype=np.float64)  # 始终是 **policy 序**

    # --- 观测自检：只拼一帧，打印各段维度/取值（不跑策略，不需要 --policy）---
    if args.selftest_obs:
        mujoco.mj_forward(model, data)
        obs = obs_builder.build(cmd, last_action)
        segs = [
            ("height_scan", 527), ("velocity_command", 3), ("base_lin_vel", 3),
            ("projected_gravity", 3), ("base_ang_vel", 3), ("joint_pos_rel", 29),
            ("joint_vel", 29), ("last_action", 29),
        ]
        print(f"\n[selftest] 总维度 = {len(obs)}（本任务期望 626）")
        off = 0
        for name, n in segs:
            chunk = obs[off:off + n]
            print(f"  {name:18s} [{off:3d}:{off + n:3d}] n={n:3d}  "
                  f"min={chunk.min(): .4f} max={chunk.max(): .4f}  "
                  f"head={np.array2string(chunk[:3], precision=4)}")
            off += n
        g = obs[527 + 3 + 3: 527 + 3 + 3 + 3]
        print(f"\n[selftest] 符号关键检查：")
        print(f"  projected_gravity = {np.array2string(g, precision=4)}"
              f"   → 直立时应接近 (0, 0, -1)；若是 (0,0,+1) 则符号错了")
        print(f"  height_scan[0]    = {obs[0]:.4f}   → 平地应 ≈ torso_z - {0.5}")
        print(f"  joint_pos_rel[:3] = {np.array2string(obs[527 + 3 + 3 + 3 + 3:][:3], precision=4)}"
              f"   → 按 IsaacLab(肩优先交错) 序，即 left/right shoulder pitch, waist pitch")
        print(f"  base_ang_vel      = {np.array2string(obs[527 + 3 + 3 + 3:527 + 3 + 3 + 3 + 3], precision=4)}")
        return 0

    # PD 增益：优先用 MJCF 的位置执行器增益，否则用给定默认值
    if model.nu > 0 and np.all(model.actuator_gainprm[:, 0] > 0):
        kp = model.actuator_gainprm[:, 0].copy()
        kd = -model.actuator_biasprm[:, 2].copy() if model.actuator_biastype[0] == 1 else np.full(n_joint, args.kd)
        use_position_actuator = model.actuator_biastype[0] == 1
        print(f"[ctrl] 位置执行器: kp~{kp[:3]} kd~{kd[:3]}")
    else:
        kp = np.full(n_joint, args.kp)
        kd = np.full(n_joint, args.kd)
        use_position_actuator = False
        print(f"[ctrl] 手动 PD: kp={args.kp} kd={args.kd}")

    dt = model.opt.timestep
    decimation = max(1, int(round((1.0 / args.policy_hz) / dt)))
    # `--duration 0`（或负数）= **无限运行**，直到手动关闭 viewer 窗口。
    infinite = args.duration <= 0
    n_steps = 10**9 if infinite else int(args.duration / (dt * decimation))

    renderer = None
    frames: list[np.ndarray] = []
    if args.record:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    viewer_ctx = None
    teleop: KeyboardCommandController | None = None
    if args.view:
        import mujoco.viewer

        if args.teleop:
            teleop = KeyboardCommandController(cmd, step=args.cmd_step, limits=CMD_LIMITS)
            print("\n" + teleop.help_text())
        viewer_ctx = mujoco.viewer.launch_passive(
            model, data, key_callback=teleop.on_key if teleop is not None else None
        )

    total_txt = "inf" if infinite else str(n_steps)
    print(f"[run] {total_txt} policy steps @ {args.policy_hz} Hz (decimation={decimation}, dt={dt})")
    try:
        for step in range(n_steps):
            # --- 键盘遥控：每步覆盖 cmd / 处理复位 / 关窗即退出 ---
            if teleop is not None:
                if viewer_ctx is not None and not viewer_ctx.is_running():
                    print("\n[teleop] viewer 窗口已关闭，退出")
                    break
                cmd, do_reset = teleop.poll()
                if do_reset:
                    # ⚠️ 复位必须同时清 `last_action`：不清的话策略会把「上一段生命
                    #    末期的大动作」当作第一帧输入，直接指数发散（见 §13 坑 2）。
                    reset_state(model, data)
                    last_action = np.zeros(n_joint, dtype=np.float64)
                    print("[teleop] 已复位到初始 keyframe（last_action 已清零）")
                if step % 25 == 0 and viewer_ctx is not None:
                    viewer_ctx.set_texts(teleop.hud())
                if step % 100 == 0:
                    print(f"  [teleop] step {step:>6}  h={data.qpos[2]:.3f}  "
                          f"cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})")
            # --- policy 推理 ---
            if policy is not None:
                obs = obs_builder.build(cmd, last_action)
                if obs.ndim == 1:
                    obs = obs[None, :]
                if step == 0:
                    print(f"[obs] 第 1 帧拼出维度 = {obs.shape[-1]}"
                          + ("  ✅ 与策略输入一致" if int(obs.shape[-1]) == int(getattr(policy, 'input_dim', -1) or -1) else "  ⚠️ 与策略输入维度不一致！"))
                # 非有限值会把 ctrl 变成 NaN、让仿真彻底崩 —— 这里定位到**具体哪一段**
                o1 = obs.reshape(-1)
                if not np.isfinite(o1).all():
                    print(f"  ❌ step {step}: obs 出现非有限值，分段排查：")
                    for nm, a, b in [("height_scan", 0, 527), ("velocity_command", 527, 530),
                                     ("base_lin_vel", 530, 533), ("projected_gravity", 533, 536),
                                     ("base_ang_vel", 536, 539), ("joint_pos_rel", 539, 568),
                                     ("joint_vel", 568, 597), ("last_action", 597, 626)]:
                        seg = o1[a:b]
                        print(f"     {nm:18s} |max|={np.abs(seg).max():.4g} "
                              f"finite={np.isfinite(seg).all()}")
                    break
                obs_in = normalizer(obs) if normalizer is not None else obs
                act = policy(obs_in)
                act = np.asarray(act).reshape(-1)[:n_joint]
                # 可选安全裁剪：IsaacLab 的 action 项 clip=None（不裁），所以默认也不裁。
                # 但 MuJoCo 里一个 NaN 就会污染整个仿真（ctct 变 NaN），调试时用
                # --act-clip 限住更好定位。
                if args.act_clip is not None:
                    act = np.clip(act, -args.act_clip, args.act_clip)
                if not np.isfinite(act).all():
                    idx = np.nonzero(~np.isfinite(act))[0]
                    print(f"  ❌ step {step}: 策略输出出现非有限值，关节下标 {idx.tolist()}"
                          f"（policy 序）→ obs 幅值 |max|={np.abs(o1).max():.4g}")
                    print(f"     obs 分段 |max|: scan={np.abs(o1[:527]).max():.4g} "
                          f"cmd={np.abs(o1[527:530]).max():.4g} "
                          f"lin_b={np.abs(o1[530:533]).max():.4g} "
                          f"grav={np.abs(o1[533:536]).max():.4g} "
                          f"ang_b={np.abs(o1[536:539]).max():.4g} "
                          f"jp={np.abs(o1[539:568]).max():.4g} "
                          f"jv={np.abs(o1[568:597]).max():.4g} "
                          f"la={np.abs(o1[597:626]).max():.4g}")
                    break
                if step % 50 == 0:
                    print(f"    [obs] |max|={np.abs(o1).max():6.3f} (scan={np.abs(o1[:527]).max():.3f} "
                          f"lin_b={np.abs(o1[530:533]).max():.3f} ang_b={np.abs(o1[536:539]).max():.3f} "
                          f"jp={np.abs(o1[539:568]).max():.3f} jv={np.abs(o1[568:597]).max():.3f})"
                          f"  [act] min={act.min(): .3f} max={act.max(): .3f}")
            else:
                act = np.zeros(n_joint)
            last_action = act.copy()          # IsaacLab 的 last_action 是 policy 序

            # 应用到 MJCF：目标角 = MJCF 序默认站姿 + 逐关节 scale × MJCF 序动作
            target = default_joint_pos + action_scale * act[pol_to_mj]

            # --- 物理子步 ---
            for _ in range(decimation):
                if use_position_actuator:
                    data.ctrl[:] = target
                else:
                    q = data.qpos[7 : 7 + n_joint]
                    dq = data.qvel[6 : 6 + n_joint]
                    data.ctrl[:] = kp * (target - q) - kd * dq
                mujoco.mj_step(model, data)

            if viewer_ctx is not None:
                viewer_ctx.sync()
            if renderer is not None and step % args.record_every == 0:
                renderer.update_scene(data)
                frames.append(renderer.render().copy())

            if step % max(1, n_steps // 10) == 0:
                print(f"  step {step:>5}/{n_steps}  height={data.qpos[2]:.3f}  |v|={np.linalg.norm(data.qvel[:3]):.3f}")
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()
        if renderer is not None:
            renderer.close()

    if args.record and frames:
        out = Path(args.record)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v3 as iio

            iio.imwrite(out, frames, fps=args.policy_hz / max(1, args.record_every))
            print(f"[record] 已保存 {len(frames)} 帧 -> {out}")
        except ImportError:
            np.save(out.with_suffix(".npy"), np.array(frames))
            print(f"[record] 缺少 imageio，已存为 {out.with_suffix('.npy')}（pip install imageio imageio-ffmpeg 可出 mp4）")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", default=str(DEFAULT_XML), help="G1 MJCF/scene XML 路径")
    ap.add_argument("--dump-model", action="store_true", help="打印模型结构后退出")
    ap.add_argument("--onnx-info", action="store_true", help="打印 ONNX 策略 IO 后退出")
    ap.add_argument(
        "--obs-mode", choices=["auto", "amp626", "legacy"], default="auto",
        help="观测拼接方式：auto=按策略输入维度自动判定（626→amp626）；"
             "amp626=本任务 Instinct-Amp-Walk（单帧 626 维）；legacy=InstinctLab parkour（8 帧堆叠）",
    )
    ap.add_argument(
        "--joint-order", choices=["isaaclab", "as-is"], default="isaaclab",
        help="策略向量 ↔ MJCF ctrl 的关节序转换：isaaclab=按关节名重排（正确）；as-is=不重排",
    )
    ap.add_argument(
        "--action-scale-mode", choices=["auto", "unity", "beyondmimic"], default="auto",
        help="动作尺度约定：auto=amp626 用 unity / legacy 用 beyondmimic；"
             "unity=target=default+1.0*action（本仓库 OmniLoco/AMP-Walk 的 scale=1.0）；"
             "beyondmimic=per-joint 0.25*effort/kp",
    )
    ap.add_argument(
        "--act-clip", type=float, default=None,
        help="动作裁剪上限（默认 None = 不裁，与 IsaacLab 的 clip=None 一致）；"
             "调试发散问题时建议给个 5.0",
    )
    ap.add_argument(
        "--obs-joint-order", choices=["npz", "articulation", "omniloco", "isaaclab", "mjcf"],
        default="isaaclab",
        help="观测里 joint_pos_rel/joint_vel 用的关节序。**云端实测：InstinctLab 的"
             "articulation 序 == ACTION_TERM_JOINTS == OMNI_LOCO（肩优先交错）**，"
             "所以默认 isaaclab/omniloco 才是对的；npz 是 AMP 参考动作的序（腿优先），"
             "仅在排查 AMP 观测对齐时才有用",
    )
    ap.add_argument(
        "--normalizer", default=None,
        help="观测归一化 npz（默认：在策略文件旁边找 policy_normalizer.npz）。"
             "⭐ 训练启用了观测归一化时必须给，否则策略输入 OOD、输出疯狂动作",
    )
    ap.add_argument(
        "--no-normalizer", action="store_true",
        help="明确跳过观测归一化（默认会检查同目录下的 policy_normalizer.npz）",
    )
    ap.add_argument(
        "--selftest-obs", action="store_true",
        help="只初始化环境并打印第一帧观测各段维度/取值后退出（不跑策略，无需 --policy）",
    )
    ap.add_argument("--policy", help="策略文件 (.onnx / .pt)")
    ap.add_argument("--policy-type", choices=["auto", "onnx", "torch"], default="auto")
    ap.add_argument("--cmd", nargs=3, type=float, default=[0.5, 0.0, 0.0], metavar=("VX", "VY", "WZ"),
                    help="速度指令 (默认 0.5 0.0 0.0)")
    ap.add_argument("--duration", type=float, default=10.0,
                    help="回放时长（秒）；**0 = 无限运行直到关闭窗口**（键盘遥控时用）")
    ap.add_argument("--policy-hz", type=float, default=50.0, help="策略频率（InstinctLab=50Hz）")
    ap.add_argument("--history", type=int, default=8, help="观测历史帧数（对齐训练配置）")
    ap.add_argument("--action-scale", type=float, default=0.25,
                    help="回退用的全局 joint_pos 动作尺度（仅在执行器数 != 29 时生效）")
    ap.add_argument("--kp", type=float, default=100.0, help="手动 PD 的 kp")
    ap.add_argument("--kd", type=float, default=2.0, help="手动 PD 的 kd")
    ap.add_argument("--view", dest="view", action="store_true", default=True, help="打开 viewer（默认开）")
    ap.add_argument("--no-view", dest="view", action="store_false", help="不开 viewer（headless）")
    ap.add_argument("--teleop", dest="teleop", action="store_true", default=True,
                    help="开 viewer 时启用 WASD/QE 键盘遥控（默认开）")
    ap.add_argument("--no-teleop", dest="teleop", action="store_false",
                    help="关掉键盘遥控（指令固定为 --cmd）")
    ap.add_argument("--cmd-step", type=float, default=0.1, help="键盘每次按键的指令增量（默认 0.1）")
    ap.add_argument("--record", help="录制到 mp4（需 imageio）")
    ap.add_argument("--record-every", type=int, default=1, help="每 N 个策略步录一帧")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args(argv)

    if not Path(args.xml).is_file():
        raise SystemExit(
            f"找不到 MJCF: {args.xml}\n"
            "先执行 `python tools/build_g1_mjcf.py` 生成可回放模型（注入执行器 + 默认站姿）。"
        )

    if args.onnx_info:
        if not args.policy:
            raise SystemExit("--onnx-info 需要同时提供 --policy")
        onnx_info(args.policy)
        return 0

    return play(args)


if __name__ == "__main__":
    sys.exit(main())
