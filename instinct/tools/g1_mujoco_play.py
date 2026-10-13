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

    # 5) ⭐ 高亮 height_scan 的扫描区域（**默认就开**，随机器人平移 + 偏航）
    #    橙色框 = 地面矩形，即全部射线的落点范围；内部淡橙线 = 0.5 m 网格；
    #    前边缘向外伸出的短段 = “这一侧是前方”。
    #    ⚠️ 几何随**布局**变（`--obs-mode` 或根据策略输入维度自动选）：
    #      · amp626（v1~v8）= 3.0×1.6 m，31×17 = 527 条，**居中在躯干正下方**；
    #      · amp378（v9）  = 3.0×0.8 m，31×9  = 279 条，**中心前移 1.5 m**（落点 x∈[0,3]）。
    #    来源：`omni_loco_env_cfg.py:86` 与 `instinct_xu/.../amp_walk_v9_env_cfg.py`。
    python tools/g1_mujoco_play.py --policy /path/to/policy.onnx --duration 0 --no-show-scan
    #    离线几何单测（不开窗）：python tools/test_scan_visual.py
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# AMP-Walk 观测布局表 —— **版本间唯一会变的就是 height_scan 的宽度与前移量**
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AmpWalkLayout:
    """一个版本的 actor 观测布局。

    ⚠️ 除 `height_scan` 外，其余 7 项的**项名/顺序/维数在各版本完全一致**：
    `velocity_command(3) + base_lin_vel(3) + projected_gravity(3) + base_ang_vel(3)
     + joint_pos_rel(29) + joint_vel(29) + last_action(29) = 99`
    ⇒ `obs_dim = scan_dim + 99`。
    """

    name: str
    scan_ny: int                     # 侧向条数（左右范围/分辨率 + 1）
    scan_nx: int                     # 前向条数
    scan_size: tuple[float, float]   # (前后, 左右) m —— GridPatternCfg 的 size
    scan_forward: float              # ⭐ 扫描中心相对躯干的前移量 m（0.0 = 居中）
    with_lin_vel: bool = True        # ⭐ 非扫描尾段里是否含 base_lin_vel(3)
    #   AMP-Walk（626/378）含 base_lin_vel ⇒ 尾段 99；
    #   OmniLoco 基座（`LocoTeacherPolicyObsCfg`，623）**不含** ⇒ 尾段 96。
    n_frames: int = 1                # ⭐ 非扫描尾段的**时序堆叠帧数**（= env cfg 的 `history_length`）
    #   T1o（663）用 **4**；626/378/623 都是 1 ⇒ 零回归。
    #   ⚠️ 堆叠是**逐项**做的、每项内部"最旧在前、最新在后" —— 对齐 isaaclab
    #   `CircularBuffer.buffer`（docstring 明写 "oldest entry at the beginning"，
    #   实现是 `torch.roll(shifts=max_length - _pointer - 1)`）+ **每项一个独立
    #   history buffer**（`_group_obs_term_history_buffer[group][term]`）
    #   ⇒ 展平后是 `[term1_f0..f_{N-1}, term2_f0..f_{N-1}, ...]`（**项优先，不是帧优先**）。
    scan_frames: int = 1             # ⭐ **height_scan 的堆叠帧数**
    #   626/378/663 都是 **1**（那份 env cfg 里 height_scan 没设 `history_length`）；
    #   ⭐⭐ **T2（1500）是 4** —— 用户明确要求 "height_scan 也是 4 帧"。

    @property
    def tail_dim(self) -> int:
        return (99 if self.with_lin_vel else 96) * self.n_frames

    @property
    def scan_dim(self) -> int:
        """**每帧**的射线数（不是展平后的总维数）。"""
        return self.scan_ny * self.scan_nx

    @property
    def scan_block_dim(self) -> int:
        return self.scan_dim * self.scan_frames

    @property
    def obs_dim(self) -> int:
        return self.scan_block_dim + self.tail_dim


#: 版本 → 布局。**环境侧改了扫描几何就必须同步这里**，否则切片/叠加框都会错。
AMP_WALK_LAYOUTS: dict[str, AmpWalkLayout] = {
    # v1~v8：3.0×1.6 / res 0.1 → 31×17 = 527，居中（offset.pos=(0,0,20)）
    "amp626": AmpWalkLayout("amp626", scan_ny=17, scan_nx=31,
                            scan_size=(3.0, 1.6), scan_forward=0.0),
    # v9(MVP)：3.0×0.8 / res 0.1 → 31×9 = 279，**中心前移 1.5 m**（落点 x∈[0,3]）
    "amp378": AmpWalkLayout("amp378", scan_ny=9, scan_nx=31,
                            scan_size=(3.0, 0.8), scan_forward=1.5),
    # ⭐ OmniLoco 基座（`Instinct-Unified-OmniLoco-G1-v0`）：527+96 = **623**。
    #   扫描几何与 amp626 相同，但**尾段没有 base_lin_vel**（基座是纯速度跟踪任务，
    #   没有这个特权量）。从零重训的 T0 实验要用它做 MuJoCo 验收。
    "loco623": AmpWalkLayout("loco623", scan_ny=17, scan_nx=31,
                             scan_size=(3.0, 1.6), scan_forward=0.0,
                             with_lin_vel=False),
    # ⭐⭐ T1o（`Instinct-Amp-Walk-T1o-G1-v0`）：279 + 96×4 = **663**。
    #   扫描几何与 amp378 **完全相同**（3.0×0.8 / 31×9 / 前移 1.5 m）；
    #   区别是尾段：**删掉 base_lin_vel**（对齐 AMP_mjlab：它只在 critic；
    #   真机也没有这个量）且本体感受 6 项 **各堆叠 4 帧**（= mjlab 的 `history_length=4`）。
    #   依据：`AMP_mjlab/.../amp_env_cfg.py:148-169` 的 actor 观测组
    #   `history_length=4, history_ordering="time"` ⇒ mjlab actor = 96×4 = 384。
    #   ❗ 注意：**163+384** 是"每项内部堆叠、项优先"，**不是**帧优先。
    "amp663": AmpWalkLayout("amp663", scan_ny=9, scan_nx=31,
                            scan_size=(3.0, 0.8), scan_forward=1.5,
                            with_lin_vel=False, n_frames=4),
    # ⭐⭐ T2（`Instinct-Amp-Walk-T2-G1-v0`）：**279×4 + 96×4 = 1500**。
    #   与 amp663 的唯一区别：`height_scan` **也堆叠 4 帧**（用户："height_scan 也是 4 帧"）。
    #   扫描几何与 amp378/amp663 **完全相同**（3.0×0.8 / 31×9 / 前移 1.5 m）。
    #   ⇒ 分段 = `scan[0:1116] + 12+12+12+116+116+116`，**无 base_lin_vel**。
    "amp1500": AmpWalkLayout("amp1500", scan_ny=9, scan_nx=31,
                             scan_size=(3.0, 0.8), scan_forward=1.5,
                             with_lin_vel=False, n_frames=4, scan_frames=4),
}


class AmpWalkObsBuilder:
    """AMP-Walk 的 actor 特权观测拼接（**布局可变**：626/527 或 378/279）。

    布局由 `layout`（`AmpWalkLayout`）决定 —— 见上面的 `AMP_WALK_LAYOUTS`。
    除 `height_scan` 的宽度外，各项的语义/顺序/取法与 v1~v8 完全一致，
    按 `G1AmpWalkPolicyObsCfg` 的**声明顺序**拼接：

        [ height_scan(527 或 279) | velocity_command(3) | base_lin_vel(3)
          | projected_gravity(3) | base_ang_vel(3)
          | joint_pos_rel(29) | joint_vel(29) | last_action(29) ] = 626 或 378

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

    ⚠️ **height_scan 是特权量**：MuJoCo 里没有地形扫描器。平地近似成常数是
    **精确**的（每条射线都命中 z=0 地面，公式退化成 `torso_z - 0.5`，
    **与扫描尺寸/前移量无关**）；地形不平坦时这个近似不成立，回放结论不可信。
    """

    #: 旧默认（v1~v8 的 527）；实例化时会被 `self.layout.scan_dim` 覆盖。
    N_SCAN = 527          # 31 x 17：0.1 m 分辨率、3.0 x 1.6 m
    SCAN_OFFSET = 0.5     # IsaacLab `height_scan` 默认 offset（我们的 ObsTerm 没传 → 0.5）
    SCAN_CLIP = (-1.0, 1.0)

    def __init__(self, model, data, default_joint_pos_mjcf, n_joint=29,
                 torso_body_name="torso_link", obs_joint_order="npz",
                 layout: "AmpWalkLayout | None" = None):
        import mujoco

        #: 观测布局（默认 v1~v8 的 626/527，保持旧调用行为不变）
        self.layout = layout if layout is not None else AMP_WALK_LAYOUTS["amp626"]
        self.N_SCAN = self.layout.scan_dim
        #: ⭐ 时序堆叠缓冲（`layout.n_frames > 1` 时启用；见 `build`）。
        #: `{项名: deque(maxlen=n_frames)}`，**逐项**维护（因为展平是"项优先"）。
        self._hist = None
        #: ⭐ `height_scan` 的堆叠缓冲（`layout.scan_frames > 1` 时启用；T2 的 1500）。
        self._scan_hist = None
        #: 用于**自动检测 reset**：仿真时间倒流 ⇒ 清空时序缓冲（见 `build`）。
        #: 这样就不用在两个 `reset_state()` 调用点手工插 `reset_history()`（易漏）。
        self._last_time: float | None = None

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
        print(f"[obs] AMP-Walk {self.layout.obs_dim} 维（布局 {self.layout.name}）："
              f"height_scan={self.N_SCAN} "
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
        #    ⭐ 尾段是否含 base_lin_vel 由布局决定（基座 623 没有这一项）
        #: ⭐ 非扫描尾段：**按 `G1AmpWalkPolicyObsCfg` 的声明顺序**（与 env cfg 一一对应）
        #:   `base_lin_vel` 只在含它的布局里出现（626/378 ⇒ 99；623/663 ⇒ 96）。
        terms: dict[str, np.ndarray] = {"velocity_command": cmd}
        if self.layout.with_lin_vel:
            terms["base_lin_vel"] = lin_b
        terms["projected_gravity"] = grav
        terms["base_ang_vel"] = ang_b
        terms["joint_pos_rel"] = jp
        terms["joint_vel"] = jv
        terms["last_action"] = np.asarray(last_action, dtype=np.float64)

        #: ⭐ 时序堆叠（`n_frames > 1`，即 T1o 的 663）：**逐项**堆叠，每项内部
        #:   **最旧在前、最新在后**（依据见 `AmpWalkLayout.n_frames` 的注释）。
        #:   T2（amp1500）里 **`height_scan` 也参与**（`scan_frames=4`），单独一个 deque。
        _t = float(self.data.time)
        if self._last_time is not None and _t < self._last_time:
            self._hist = None            # 仿真时间倒流 ⇒ `reset_state()` 被调过
            self._scan_hist = None
        self._last_time = _t
        _sfc = int(self.layout.scan_frames)
        if self.layout.n_frames > 1 or _sfc > 1:
            from collections import deque

            if self._hist is None:
                self._hist = {_nm: deque(maxlen=self.layout.n_frames) for _nm in terms}
            if _sfc > 1:
                if self._scan_hist is None:
                    self._scan_hist = deque(maxlen=_sfc)
                if len(self._scan_hist) == 0:              # 首帧 / 回合开始：用当前帧填满
                    self._scan_hist.extend([scan] * _sfc)
                else:
                    self._scan_hist.append(scan)
            for _nm, _v in terms.items():
                _hb = self._hist[_nm]
                if len(_hb) == 0:                      # 首帧 / 回合开始：用当前帧填满
                    _hb.extend([_v] * self.layout.n_frames)
                else:
                    _hb.append(_v)

        parts = [np.concatenate(self._scan_hist) if self._scan_hist is not None else scan]
        for _nm, _v in terms.items():
            parts.append(np.concatenate(self._hist[_nm]) if self._hist is not None else _v)
        return np.concatenate(parts)

    def reset_history(self) -> None:
        """手工清空时序缓冲（回合开始 / `reset_state()` 后）。

        ⭐ 通常**不需要**手动调：`build()` 会看 `data.time` 倒流自动清。这里只留给
        单测/脚本显式使用。

        ⚠️ isaaclab 在 reset 时只把 `CircularBuffer._num_pushes` 归零、**不清内容**
        ⇒ 真实行为是"上一回合的尾巴残留几帧"。这里改成"下一帧用当前帧重新填满"
        ⇒ 4 帧后收敛到同一结果；只有前 3 帧有差异，**不影响稳态回放结论**。
        """
        self._hist = None
        self._scan_hist = None


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
# 高程图（height_scan）扫描区域可视化
# ---------------------------------------------------------------------------
# 几何**完全取自源码**，不是估的：`instinct_xu` 的
# `tasks/locomotion/config/g1/omni_loco_env_cfg.py:86`（`G1OmniLocoSceneCfg.height_scanner`）：
#
#     height_scanner = MultiMeshRayCasterCfg(
#         prim_path = "{ENV_REGEX_NS}/Robot/torso_link",   # 挂在 torso 上
#         offset    = OffsetCfg(pos=(0.0, 0.0, 20.0)),     # 从 torso 上方 20 m 垂直向下打
#         ray_alignment = "yaw",                            # ⭐ 只跟航向，不随俯仰/横滚倾斜
#         pattern_cfg = GridPatternCfg(resolution=0.1, size=(3.0, 1.6)),
#     )
#
# 自洽校验：3.0/0.1 + 1 = 31、1.6/0.1 + 1 = 17 → 31 × 17 = **527**
#           ✓ = 契约里的 `AMP_WALK_SCAN_RAYS`（`amp_walk_env_cfg.py:87`）
#
# 平地时扫描区域就是**地面上一个 3.0（前后）× 1.6（左右）m 的矩形**，
# 中心随机器人 (x, y) 平移、随航向（yaw）旋转。下面画的正是它。
SCAN_SIZE_XY = (3.0, 1.6)       # (前后, 左右) m —— GridPatternCfg 的 size（默认=v1~v8）
SCAN_RESOLUTION = 0.1           # m —— GridPatternCfg 的 resolution
SCAN_FORWARD = 0.0              # ⭐ 扫描中心相对躯干的前移量 m（v9 = 1.5）
SCAN_TORSO_BODY = "torso_link"  # 挂载点
SCAN_ALIGNMENT = "yaw"          # 与 IsaacLab 的 ray_alignment 对齐


def apply_scan_layout(layout) -> None:
    """把可视化用的扫描几何切到某个布局（`AMP_WALK_LAYOUTS[...]`）。

    ⚠️ 下面几个是**模块级全局**（绘图函数与离线单测 `test_scan_visual.py` 都读它们），
    所以启动时按 `--obs-mode` 调一次；单测可对不同布局各调一次再断言。
    """
    global SCAN_SIZE_XY, SCAN_FORWARD, SCAN_NY, SCAN_NX, SCAN_GRID_RAYS
    SCAN_SIZE_XY = tuple(float(v) for v in layout.scan_size)
    SCAN_FORWARD = float(layout.scan_forward)
    SCAN_NY, SCAN_NX = int(layout.scan_ny), int(layout.scan_nx)
    SCAN_GRID_RAYS = SCAN_NY * SCAN_NX

SCAN_COLOR_EDGE = np.array([1.00, 0.35, 0.05, 0.90], dtype=np.float32)  # 边框（亮橙）
SCAN_COLOR_GRID = np.array([1.00, 0.60, 0.30, 0.35], dtype=np.float32)  # 内部网格（淡）
SCAN_LINE_RADIUS = 0.012        # 胶囊半径（m）：3 m 外也看得清，又不遮住机器人
SCAN_DRAW_Z = 0.012             # 抬离地面 12 mm，避免与地面 z-fighting


def yaw_frame_basis(quat_wxyz: np.ndarray) -> np.ndarray:
    """把机体姿态投影到水平面 → 3x3 矩阵，列依次是 yaw 系的 x / y / z 轴。

    ⭐ 为什么不用完整四元数：`ray_alignment="yaw"` 让扫描网格**不随躯干的
    roll/pitch 倾斜**（永远水平扣在地面上）。用完整姿态画就会跟着躯干翻转 ——
    那就把几何画错了。做法：取机体 x 轴投到水平面当 yaw 系 x 轴，z 取世界 z。
    """
    fwd = rotmat_wxyz(quat_wxyz)[:, 0].copy()   # 机体 x 轴 → 世界坐标
    fwd[2] = 0.0
    norm = float(np.linalg.norm(fwd))
    if norm < 1e-6:                             # 机体 x 轴几乎竖直（极端姿态）→ 退回单位阵
        return np.eye(3)
    x_axis = fwd / norm
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def scan_footprint_segments(data, torso_id: int, *, z: float = SCAN_DRAW_Z):
    """算出 height_scan 扫描区域的全部线段，返回 `[(起点, 终点, rgba), ...]`（世界坐标）。

    单独拆出来是为了**能被离线单测直接验证几何**（`test_scan_visual.py`）：
    若从画好的 `mjvGeom` 反推端点，要处理 capsule 的轴向（MuJoCo 里是局部 **z** 轴，
    不是 x）与半径补偿，容易把测试自己写错 —— 那测的就不是几何了。

    画三样东西：边框（亮橙）、每 0.5 m 的内部网格（淡）、以及前边缘中点向外
    0.3 m 的短线段（提示「这一侧是前方」，因为 GridPatternCfg 的 x 是前后方向）。
    """
    p = data.xpos[torso_id]
    R = yaw_frame_basis(data.xquat[torso_id])
    half_x, half_y = SCAN_SIZE_XY[0] / 2.0, SCAN_SIZE_XY[1] / 2.0

    def to_world(ux: float, uy: float) -> np.ndarray:
        """扫描系 (x=前后, y=左右) → 世界坐标（落在地面上）

        ⚠️ `SCAN_FORWARD` = 扫描中心相对躯干的**前移量**（v9 = 1.5 m）：
        IsaacLab 的 `offset.pos=(forward, 0, 20)` 意味着射线束整体前移，
        所以这里在前向分量上加上它。镜像损失不受影响（左右仍关于躯干对称）。
        """
        return np.array([p[0], p[1], z]) + R[:, 0] * (ux + SCAN_FORWARD) + R[:, 1] * uy

    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # 1) 边框 4 条边
    corners = [(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)]
    for i in range(4):
        segments.append((to_world(*corners[i]), to_world(*corners[(i + 1) % 4]), SCAN_COLOR_EDGE))

    # 2) 内部网格（0.5 m 一条，避免把 31x17 条射线全画出来遮住机器人）
    grid_step = 0.5
    for i in range(1, int(SCAN_SIZE_XY[0] / grid_step)):
        ux = -half_x + i * grid_step
        segments.append((to_world(ux, -half_y), to_world(ux, half_y), SCAN_COLOR_GRID))
    for i in range(1, int(SCAN_SIZE_XY[1] / grid_step)):
        uy = -half_y + i * grid_step
        segments.append((to_world(-half_x, uy), to_world(half_x, uy), SCAN_COLOR_GRID))

    # 3) 前向标记：前边缘中点再往前 0.3 m
    segments.append((to_world(half_x, 0.0), to_world(half_x + 0.3, 0.0), SCAN_COLOR_EDGE))
    return segments


def draw_scan_footprint(scn, data, torso_id: int, *, z: float = SCAN_DRAW_Z) -> int:
    """把扫描区域画进 `mjvScene`（**只画线，不动物理**）。返回画出的线条数。"""
    import mujoco

    drawn = 0
    for a, b, rgba in scan_footprint_segments(data, torso_id, z=z):
        if scn.ngeom >= scn.maxgeom:
            break
        geom = scn.geoms[scn.ngeom]
        # `mjv_connector` 的前置要求：先 mjv_initGeom 把其余属性（rgba 等）设好
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
            np.eye(3).flatten(), rgba,
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, SCAN_LINE_RADIUS, a, b)
        scn.ngeom += 1
        drawn += 1
    return drawn


def scan_geometry_text() -> str:
    """启动时打印扫描区域几何（便于和云端配置对账）。"""
    fwd = (
        "居中在躯干正下方"
        if abs(SCAN_FORWARD) < 1e-9
        else f"中心**前移 {SCAN_FORWARD:.2f} m**（落点 x∈[{SCAN_FORWARD - SCAN_SIZE_XY[0] / 2:+.2f}, {SCAN_FORWARD + SCAN_SIZE_XY[0] / 2:+.2f}] m）"
    )
    return (
        f"[scan] 高程图扫描区域 = {SCAN_SIZE_XY[0]:.1f}(前后) x {SCAN_SIZE_XY[1]:.1f}(左右) m，"
        f"res {SCAN_RESOLUTION} m → "
        f"{SCAN_NY} x {SCAN_NX} = {SCAN_GRID_RAYS} 条射线；{fwd}\n"
        f"       挂载 {SCAN_TORSO_BODY}，ray_alignment='{SCAN_ALIGNMENT}'"
        f"（只跟航向，不随俯仰/横滚倾斜）\n"
        f"       平地时 = 地面上的矩形，中心随机器人平移、随航向旋转（橙色框=边界，前边缘外伸段=前方）"
    )


#: 射线条数：默认布局（v1~v8 的 527）；`apply_scan_layout()` 会按布局重算
SCAN_NY, SCAN_NX = 17, 31
SCAN_GRID_RAYS = SCAN_NY * SCAN_NX


# ---------------------------------------------------------------------------
# 键盘遥控（WASD/QE）—— 平地技能验证
# ---------------------------------------------------------------------------

#: 训练时的速度指令范围，= `instinct_xu` 的 `AMP_WALK_COMMAND_RANGES`
#: （`tasks/locomotion/config/g1/amp_walk_env_cfg.py` 的 `(-1.0, 1.2) / (-0.5, 0.5) / (-0.8, 0.8)`）。
#: 键盘遥控**必须**夹在这个范围内：超出范围属于分布外指令，摔了不代表策略差。
#:
#: ⚠️ 2026-10-06 修正：这里原先是 `((-2.0, 2.0), (-1.5, 1.5), (-2.8, 2.8))`
#:    （OmniLoco 基线的旧口径），**与 AMP-Walk 的训练范围不一致** —— 键盘按几下
#:    就能把 vx 拧到 2.0、wz 拧到 2.8（训练上限的 1.7~3.5 倍），回放现象会被
#:    误判成「策略差」。注释早就写着「= 训练范围」，数值却是旧的。
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


#: 给地面补的棋盘材质（
#: 重要：MJCF 的 `<default>` 里写了 `<geom group="3" .../>`，而
#: `mjvOption.geomgroup` 默认是 `[1,1,1,0,0,0]` —— **group 3 根本不渲染**，
#: 所以 floor 在 viewer 里完全看不见，机器人像站在空气上（碰撞照常发生，
#: 因为 floor 的 `contype=1`，仿真本身没错，只是看不见）。
_FLOOR_ASSETS = """
    <texture name="llp_grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.30 0.34 0.38" rgb2="0.18 0.21 0.24"/>
    <material name="llp_groundplane" texture="llp_grid" texrepeat="8 8" reflectance="0.05"/>
"""


def _patch_xml_for_floor(xml_text: str) -> str:
    """给 MJCF 文本补上「可见的地面」：棋盘材质 + `group="2"`。

    改的是**加载时的文本副本**，不碰磁盘上的 XML（那文件由
    `build_g1_mjcf.py` 生成，手改会被下次重生覆盖）。
    棋盘化不是好看而已 —— 纯色地面**看不出机器人有没有位移/打滑**，
    而这两件事恰好是评估行走策略最关心的。
    """
    if "<asset>" not in xml_text:
        raise ValueError("MJCF 里没有 <asset>，无法注入材质")
    xml_text = xml_text.replace("<asset>", "<asset>" + _FLOOR_ASSETS, 1)
    lines = xml_text.split("\n")
    for i, ln in enumerate(lines):
        if 'name="floor"' in ln and ln.rstrip().endswith("/>"):
            # 追加 `group="2"`（可见）与棋盘材质；只改渲染，不动碰撞参数
            lines[i] = ln.rstrip()[: -len("/>")] + ' group="2" material="llp_groundplane"/>'
            return "\n".join(lines)
    raise ValueError("找不到 floor 的 geom 行")


def load_model(xml_path, checkered_floor: bool = True):
    """加载 MJCF；默认把地面做成**可见的棋盘**。

    ⭐ 不补的话，viewer 里看不到地面（原因见 `_FLOOR_ASSETS` 的注释）。
    补丁失败时回退到直接改 `model.geom_group[floor] = 2`（物理不受影响）。
    """
    import mujoco

    xml_path = Path(xml_path)
    if checkered_floor:
        try:
            patched = _patch_xml_for_floor(xml_path.read_text(encoding="utf-8"))
            # meshdir 在本 XML 里是**绝对路径**，所以 from_xml_string 能直接找到网格
            model = mujoco.MjModel.from_xml_string(patched)
            print("[scene] 地面已补棋盘材质并设为可见 group=2（便于看出位移/打滑）")
            return model
        except Exception as exc:  # noqa: BLE001
            print(f"[scene] 地面补丁失败（{type(exc).__name__}: {exc}），回退到只改渲染组")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    for i in range(model.ngeom):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) == "floor":
            model.geom_group[i] = 2  # 只影响渲染，不影响碰撞
            print("[scene] 已把 floor 的渲染组改为 2（可见；仅渲染，不影响物理）")
            break
    return model


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


def apply_init_vel(model, data, init_vx: float) -> None:
    """诊断用（P1 探针）：给根节点注入初始**前向**速度（世界系 x，m/s）。

    为什么要它：`cmd=+0.8` 时策略实际只跑 0.014 m/s（原地踏步）。要区分两件事：
      (a) 策略的**固定点里根本没有**"前进"这个解（一给速度就刹停）；
      (b) 固定点**存在**，但策略学不会从静止加速上去（自举/探索问题）。
    办法就是把初速直接置成指令值，看它是"维持"还是"刹车"。

    ⚠️ 必须在 `reset_state()`（内含 `mj_forward`）**之后**调用，并再 forward 一次，
    否则 `data.xpos` 等派生量仍是旧值（与 `reset_state` 的 docstring 同一个坑）。
    """
    if abs(float(init_vx)) < 1e-9:
        return
    import mujoco   # 模块级没有 mujoco（`play()` 内是局部导入），这里必须自己导

    data.qvel[0] = float(init_vx)
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

    model = load_model(args.xml, checkered_floor=not args.no_checkered_floor)
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
    apply_init_vel(model, data, args.init_vx)
    if abs(args.init_vx) > 1e-9:
        print(f"[init] P1 探针：已注入初始前向速度 v0={args.init_vx:+.2f} m/s（世界系 x）")

    n_joint = model.nu
    default_joint_pos = default_qpos[7 : 7 + n_joint].copy()
    # 逐关节动作尺度（在下面知道 obs_mode 之后才能定 —— 两种任务的约定不同），
    # 见那里的注释（这是个能把机器人直接搞摔的坑）。

    # --- ⭐ 取证：逐步 trace（
    #     膝/腰到底摆在哪、躯干有多僵、每一项在此轨迹上的拉力 —— 都从这份 CSV 算）---
    trace_names: list[str] | None = None
    trace_rows: list[list[float]] | None = None
    trace_header: list[str] = []
    if args.trace_csv:
        # 按**执行器序**取关节名（= `default_joint_pos`/`ctrl` 的序，也是 qpos[7:] 的序）
        trace_names = []
        for i in range(n_joint):
            jid = int(model.actuator_trnid[i][0])
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"jnt{jid}"
            trace_names.append(nm)
        trace_header = (["t", "cmd_vx", "cmd_vy", "cmd_wz", "cmd_speed",
                         "root_x", "root_y", "root_z", "qw", "qx", "qy", "qz",
                         "act_absmax"]
                        + [f"j_{n}" for n in trace_names]
                        + [f"d_{n}" for n in trace_names])
        trace_rows = []
        print(f"[trace] 将记录 {n_joint} 关节（名序前 3：{trace_names[:3]}）-> {args.trace_csv}")

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
        # ⭐ 按策略输入维度自动选布局：626 → v1~v8 的 527 扫描；378 → v9 的 279 扫描
        by_dim = {lay.obs_dim: name for name, lay in AMP_WALK_LAYOUTS.items()}
        if dim in by_dim:
            obs_mode = by_dim[dim]
        elif dim is None:
            if args.selftest_obs:
                obs_mode = "amp626"   # 自检默认校验 v1~v8 的 626 维
            else:
                raise SystemExit(
                    "--obs-mode auto 无法从策略文件推断观测维度（torch 分支 / 动态维度）。\n"
                    f"请显式指定：--obs-mode {' 或 '.join(AMP_WALK_LAYOUTS)}（本仓库 AMP-Walk），"
                    "或 --obs-mode legacy（InstinctLab parkour）"
                )
        else:
            obs_mode = "legacy"
    if policy is None and not args.selftest_obs:
        obs_builder = None
    elif obs_mode in AMP_WALK_LAYOUTS:
        # ⭐ AMP-Walk：先把可视化几何切到同一布局（否则叠加的橙框会误导），再建 builder
        amp_layout = AMP_WALK_LAYOUTS[obs_mode]
        apply_scan_layout(amp_layout)
        obs_builder = AmpWalkObsBuilder(model, data, default_joint_pos, n_joint=n_joint,
                                        obs_joint_order=args.obs_joint_order, layout=amp_layout)
    else:
        obs_builder = ObsBuilder(model, data, default_joint_pos, history=args.history)
    _dim_txt = (f"{obs_builder.layout.obs_dim}"
                if (obs_builder is not None and obs_mode in AMP_WALK_LAYOUTS) else "?")
    print(f"[obs] obs_mode={obs_mode}"
          + (f"  策略期望维度={getattr(policy, 'input_dim', None)}" if policy else "")
          + f"  （自检：实际拼出 {_dim_txt} 维）")

    # ⭐ 观测分段表：**按当前布局推导**（v1~v8: 527 / v9: 279）——
    #    下面所有“分段排查/周期打印”都用它，不再写死 527/626。
    _n0 = int(getattr(obs_builder, "N_SCAN", 527))
    #: ⭐ 时序堆叠（`history_length=N`）：**每一项**内部都是 `N × 单帧维`，
    #:    所以除首部的 `height_scan`（默认单帧）之外，后面每段都要整体乘 N。
    #:    顺序仍是"项优先"：同一项的 N 帧连续（不是帧优先）。
    _nf = int(getattr(getattr(obs_builder, "layout", None), "n_frames", 1) or 1)
    #: ⭐⭐ T2（1500）：`height_scan` **也堆叠**（`scan_frames=4`）⇒ 首段也要乘。
    _sf = int(getattr(getattr(obs_builder, "layout", None), "scan_frames", 1) or 1)
    _ns = _n0 * _sf
    _o = _ns + 3 * _nf                             # 扫描块 + 指令（两者都可堆叠）
    SEG: list[tuple[str, int, int]] = [
        ("height_scan", 0, _ns),
        ("velocity_command", _ns, _o),
    ]
    if bool(getattr(getattr(obs_builder, "layout", None), "with_lin_vel", True)):
        SEG.append(("base_lin_vel", _o, _o + 3 * _nf))
        _o += 3 * _nf
    SEG += [
        ("projected_gravity", _o, _o + 3 * _nf),
        ("base_ang_vel", _o + 3 * _nf, _o + 6 * _nf),
        ("joint_pos_rel", _o + 6 * _nf, _o + 35 * _nf),
        ("joint_vel", _o + 35 * _nf, _o + 64 * _nf),
        ("last_action", _o + 64 * _nf, _o + 93 * _nf),
    ]
    SEGD: dict[str, tuple[int, int]] = {nm: (a, b) for nm, a, b in SEG}

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
        scale_mode = "unity" if obs_mode in AMP_WALK_LAYOUTS else "beyondmimic"
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
        print(f"\n[selftest] 总维度 = {len(obs)}（布局 {obs_builder.layout.name}，期望 {obs_builder.layout.obs_dim}）")
        # ⭐ 直接用上面的 `SEG`（唯一真相源，按布局推导）—— 原来这里又写死了一遍
        #    "尾段必含 base_lin_vel"，对 623 的基座会把 projected_gravity 标成
        #    base_lin_vel，误导排查。不许再出现第二份分段表。
        for name, a, b in SEG:
            chunk = obs[a:b]
            print(f"  {name:18s} [{a:3d}:{b:3d}] n={b - a:3d}  "
                  f"min={chunk.min(): .4f} max={chunk.max(): .4f}  "
                  f"head={np.array2string(chunk[:3], precision=4)}")
        g = obs[SEGD["projected_gravity"][0]: SEGD["projected_gravity"][1]]
        print(f"\n[selftest] 符号关键检查：")
        print(f"  projected_gravity = {np.array2string(g, precision=4)}"
              f"   → 直立时应接近 (0, 0, -1)；若是 (0,0,+1) 则符号错了")
        print(f"  height_scan[0]    = {obs[0]:.4f}   → 平地应 ≈ torso_z - {0.5}"
              f"（与扫描尺寸/前移量无关）")
        print(f"  joint_pos_rel[:3] = {np.array2string(obs[SEGD['joint_pos_rel'][0]:][:3], precision=4)}"
              f"   → 按 IsaacLab(肩优先交错) 序，即 left/right shoulder pitch, waist pitch")
        print(f"  base_ang_vel      = {np.array2string(obs[SEGD['base_ang_vel'][0]: SEGD['base_ang_vel'][1]], precision=4)}")
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

    # --- 高程图扫描区域可视化（默认开）---
    scan_torso_id = -1
    if args.view and args.show_scan:
        scan_torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SCAN_TORSO_BODY)
        if scan_torso_id < 0:
            print(f"[scan] ⚠️ MJCF 里找不到 body '{SCAN_TORSO_BODY}'，关闭扫描区域可视化")
        else:
            print(scan_geometry_text())
            print(f"       torso_body_id = {scan_torso_id}")

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
        # 初始相机：否则默认视角偏高偏远，第一眼容易看不到地面/脚
        try:
            with viewer_ctx.lock():
                viewer_ctx.cam.lookat[:] = [0.0, 0.0, 0.5]
                viewer_ctx.cam.distance = float(args.cam_distance)
                viewer_ctx.cam.azimuth = 145.0
                viewer_ctx.cam.elevation = -18.0
                if args.show_collision:
                    # group 3 = floor 原本的组 + 29 个碰撞胶囊（默认隐藏）
                    viewer_ctx.opt.geomgroup[3] = 1
                    print("[scene] 已开启 group 3 渲染（碰撞胶囊等非可视 geom）")
            print(f"[scene] 初始相机: lookat=(0,0,0.5) distance={args.cam_distance} "
                  f"azimuth=145 elevation=-18（鼠标可自由调）")
        except Exception as exc:  # noqa: BLE001
            print(f"[scene] 设置初始相机失败（不影响仿真）: {type(exc).__name__}: {exc}")

    # --- 实时节拍（⭐ 不加的话仿真会以 CPU 全速跑，看起来像 50x 快进）---
    wall_dt = dt * decimation / max(1e-6, float(args.speed))
    realtime = not args.no_realtime
    t_start = time.perf_counter()
    t_next = t_start
    lag_reports = 0

    # --- 指令跟踪统计：判「原地踏步」的客观量（指令 vs 实际速度/位移）---
    tr_n = 0
    tr_cmd = np.zeros(2, dtype=np.float64)
    tr_vx = tr_vy = 0.0
    tr_x0, tr_y0 = float(data.qpos[0]), float(data.qpos[1])
    tr_skipped = args.track_skip <= 0.0
    tr_skip_step = max(0, int(round(args.track_skip * float(args.policy_hz))))

    total_txt = "inf" if infinite else str(n_steps)
    print(f"[run] {total_txt} policy steps @ {args.policy_hz} Hz (decimation={decimation}, dt={dt})")
    print("[track] 判据：给 +0.8 前进指令时，实际 vx 均值应接近 +0.8；若≈0 即「原地踏步」")
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
                    apply_init_vel(model, data, args.init_vx)
                    last_action = np.zeros(n_joint, dtype=np.float64)
                    # 跟踪统计也要重新起算，否则位移会跨生命期累加
                    tr_n = 0
                    tr_cmd = np.zeros(2, dtype=np.float64)
                    tr_vx = tr_vy = 0.0
                    tr_x0, tr_y0 = float(data.qpos[0]), float(data.qpos[1])
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
                    print(f"  ❌ step {step}: obs 出现非有限值，分段排查（布局 {obs_builder.layout.name}）：")
                    for nm, a, b in SEG:
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
                    print("     obs 分段 |max|: " + "  ".join(
                        f"{nm}={np.abs(o1[a:b]).max():.4g}" for nm, a, b in SEG))
                    break
                if step % 50 == 0:
                    # ⭐ 不能写死 base_lin_vel：基座布局（loco623）没有这一段
                    _ang = SEGD["base_ang_vel"]
                    _jp = SEGD["joint_pos_rel"]
                    _jv = SEGD["joint_vel"]
                    _lin_txt = ""
                    if "base_lin_vel" in SEGD:
                        _l = SEGD["base_lin_vel"]
                        _lin_txt = f"lin_b={np.abs(o1[_l[0]:_l[1]]).max():.3f} "
                    print(f"    [obs] |max|={np.abs(o1).max():6.3f} (scan={np.abs(o1[:_n0]).max():.3f} "
                          f"{_lin_txt}"
                          f"ang_b={np.abs(o1[_ang[0]:_ang[1]]).max():.3f} "
                          f"jp={np.abs(o1[_jp[0]:_jp[1]]).max():.3f} "
                          f"jv={np.abs(o1[_jv[0]:_jv[1]]).max():.3f})"
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

            # --- 取证 trace：记录**物理子步之后**的状态（与 obs 下一帧一致）---
            if trace_rows is not None:
                trace_rows.append(
                    [step * dt * decimation, float(cmd[0]), float(cmd[1]), float(cmd[2]),
                     float(np.hypot(cmd[0], cmd[1]) + abs(cmd[2])),
                     float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2]),
                     float(data.qpos[3]), float(data.qpos[4]), float(data.qpos[5]),
                     float(data.qpos[6]), float(np.abs(act).max())]
                    + data.qpos[7 : 7 + n_joint].tolist()
                    + data.qvel[6 : 6 + n_joint].tolist()
                )

            # --- 累计「指令 vs 实际」：世界系 pelvis 线速度 + 位移 ---
            # `--track-skip`：跳过起步瞬态（注入初速后前几秒是过渡段，会污染均值）
            if (not tr_skipped) and step >= tr_skip_step:
                tr_skipped = True
                tr_n = 0
                tr_cmd = np.zeros(2, dtype=np.float64)
                tr_vx = tr_vy = 0.0
                tr_x0, tr_y0 = float(data.qpos[0]), float(data.qpos[1])
            tr_n += 1
            tr_cmd += np.asarray(cmd[:2], dtype=np.float64)
            tr_vx += float(data.qvel[0])
            tr_vy += float(data.qvel[1])

            if viewer_ctx is not None:
                # 每帧重画扫描区域（user_scn 是“只读物理”的叠加层，不影响碰撞）
                if scan_torso_id >= 0:
                    with viewer_ctx.lock():
                        viewer_ctx.user_scn.ngeom = 0
                        draw_scan_footprint(viewer_ctx.user_scn, data, scan_torso_id)
                viewer_ctx.sync()
            if renderer is not None and step % args.record_every == 0:
                renderer.update_scene(data)
                frames.append(renderer.render().copy())

            # --- 实时节拍：把每个策略步的墙上耗时对齐到 wall_dt ---
            if realtime:
                t_next += wall_dt
                now = time.perf_counter()
                slack = t_next - now
                if slack > 0:
                    time.sleep(slack)
                elif -slack > 5.0 * wall_dt:
                    # 落后超过 5 步 → 重置基准，不要"追赶式暴走"
                    if lag_reports < 3:
                        print(f"  [pace] 第 {step} 步落后 {-slack * 1000:.0f} ms，重置节拍基准")
                        lag_reports += 1
                    t_next = now

            if step % (100 if infinite else max(1, n_steps // 10)) == 0:
                _mvx = tr_vx / tr_n if tr_n else 0.0
                _mvy = tr_vy / tr_n if tr_n else 0.0
                print(f"  step {step:>6}/{total_txt}  height={data.qpos[2]:.3f}  "
                      f"|v|={np.linalg.norm(data.qvel[:3]):.3f}  "
                      f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f}) 实际均值=({_mvx:+.3f},{_mvy:+.3f})  "
                      f"Δx={data.qpos[0] - tr_x0:+.3f}m")
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()
        if renderer is not None:
            renderer.close()

    # 实时性汇总：客观确认"不再是快进"（目标 1.00x）
    wall_s = time.perf_counter() - t_start
    sim_s = (step + 1) * dt * decimation
    print(f"[pace] 仿真 {sim_s:.2f} s / 墙上 {wall_s:.2f} s = {sim_s / max(1e-9, wall_s):.2f}x 实时"
          + ("（目标 1.00x）" if realtime else "（--no-realtime，全速跑）"))

    # --- 指令跟踪汇总（回放判据：指标好 ≠ 真的走了）---
    if tr_n > 0:
        mcmd, mvx, mvy = tr_cmd / tr_n, tr_vx / tr_n, tr_vy / tr_n
        dx, dy = float(data.qpos[0]) - tr_x0, float(data.qpos[1]) - tr_y0
        eff = f"，跟踪效率 = {mvx / mcmd[0] * 100:.0f}%" if abs(mcmd[0]) > 1e-6 else ""
        print("[track] ===== 指令跟踪汇总 =====")
        print(f"        指令均值 vx={mcmd[0]:+.3f} vy={mcmd[1]:+.3f} m/s")
        print(f"        实际均值 vx={mvx:+.3f} vy={mvy:+.3f} m/s{eff}")
        print(f"        净位移   Δx={dx:+.3f} m  Δy={dy:+.3f} m  （窗口 {max(0.0, sim_s - args.track_skip):.2f} s）")
        print(f"        稳态跟踪误差 |cmd-actual|_xy = {np.hypot(mcmd[0] - mvx, mcmd[1] - mvy):.3f} m/s")
        print("        ⇒ 指令 ≈ +0.8 而实际 ≈ 0 即「原地踏步」：策略问题，不是回放问题")

    # --- 取证 trace 落盘（CSV）---
    if args.trace_csv and trace_rows is not None:
        import csv as _csv

        tpath = Path(args.trace_csv)
        tpath.parent.mkdir(parents=True, exist_ok=True)
        with open(tpath, "w", newline="") as fh:
            wr = _csv.writer(fh)
            wr.writerow(trace_header)
            wr.writerows(trace_rows)
        print(f"[trace] 已写入 {len(trace_rows)} 行 × {len(trace_header)} 列 -> {tpath}")

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
        "--obs-mode", choices=["auto", *AMP_WALK_LAYOUTS, "legacy"], default="auto",
        help="观测拼接方式：auto=按策略输入维度自动判定（626→amp626 / 378→amp378）；"
             "amp626=v1~v8（单帧 626 维、527 条射线、扫描居中）；"
             "amp378=v9（单帧 378 维、279 条射线、扫描中心前移 1.5 m）；"
             "legacy=InstinctLab parkour（8 帧堆叠）",
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
    ap.add_argument("--cmd", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "WZ"),
                    help="速度指令 (默认 0.0 0.0 0.0；训练范围 vx[-1.0,1.2] vy[-0.5,0.5] wz[-0.8,0.8])")
    ap.add_argument("--duration", type=float, default=10.0,
                    help="回放时长（秒）；**0 = 无限运行直到关闭窗口**（键盘遥控时用）")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="实时倍速（1.0 = 真机同速；0.5 = 慢放便于看细节）")
    ap.add_argument("--no-realtime", action="store_true",
                    help="关掉实时节拍（全速跑，用于快速扫参数；默认是实时）")
    ap.add_argument("--cam-distance", type=float, default=3.0, help="初始相机距离（米）")
    ap.add_argument("--show-collision", action="store_true",
                    help="显示 group 3 的碰撞 geom（胶囊体等，排查接触时用）")
    ap.add_argument("--show-scan", dest="show_scan", action="store_true", default=True,
                    help="高亮 height_scan 的扫描区域（默认开；橙色框=3.0x1.6 m）")
    ap.add_argument("--no-show-scan", dest="show_scan", action="store_false",
                    help="关掉扫描区域高亮")
    ap.add_argument("--no-checkered-floor", action="store_true",
                    help="不给地面补棋盘材质（保持 MJCF 原样；地面会是纯色）")
    ap.add_argument("--policy-hz", type=float, default=50.0, help="策略频率（InstinctLab=50Hz）")
    ap.add_argument("--init-vx", type=float, default=0.0,
                    help="P1 探针：复位后给根节点注入初始前向速度 (m/s，世界系 x)。"
                         "用于区分'策略不会前进'与'策略不会从静止起步加速'")
    ap.add_argument("--track-skip", type=float, default=0.0,
                    help="跟踪统计跳过开头 N 秒（默认 0 = 全程）；配合 --init-vx 看稳态。")
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
    ap.add_argument("--cmd-step", type=float, default=0.5, help="键盘每次按键的指令增量（默认 0.5）")
    ap.add_argument("--record", help="录制到 mp4（需 imageio）")
    ap.add_argument("--record-every", type=int, default=1, help="每 N 个策略步录一帧")
    ap.add_argument(
        "--trace-csv", default=None,
        help="把逐步状态写成 CSV（取证用）：t / cmd / root 位姿 / 29 关节位置与速度 / |act|max。"
             "列名形如 `j_waist_pitch_joint`（按关节名，下游不靠下标猜）")
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
