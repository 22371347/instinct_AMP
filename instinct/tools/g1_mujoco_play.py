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
"""
from __future__ import annotations

import argparse
import os
import sys
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
        # 角速度（mujoco 的 free joint 角速度在前 3，线性在后 3）
        ang_vel = d.qvel[3:6].copy() * 0.25
        # 投影重力
        quat = d.qpos[3:7]  # w, x, y, z
        w, x, y, z = quat
        grav = np.array(
            [
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ]
        )
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

def play(args) -> int:
    import mujoco

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    if args.dump_model:
        dump_model(args.xml)
        return 0

    # 默认站姿：用 MJCF 的 keyframe（若有），否则用 qpos0
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        default_qpos = model.key_qpos[0].copy()
        print(f"[init] 使用 keyframe 0：{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, 0)}")
    else:
        default_qpos = model.qpos0.copy()
        print("[init] 无 keyframe，使用 qpos0")

    n_joint = model.nu
    default_joint_pos = default_qpos[7 : 7 + n_joint].copy()
    if ACTION_SCALE.shape[0] != n_joint:
        action_scale = np.full(n_joint, args.action_scale, dtype=np.float64)
        print(f"[warn] MJCF 有 {n_joint} 个执行器（期望 29），回退到全局 action_scale={args.action_scale}")
    else:
        action_scale = ACTION_SCALE.copy()
        print(f"[ctrl] 逐关节 action_scale: {action_scale.min():.5f} ~ {action_scale.max():.5f}")

    # 策略
    policy = load_policy(args.policy, args.policy_type) if args.policy else None
    if policy is None:
        print("[warn] 未提供 --policy，仅做零动作站立（用于验证模型/PD）")

    obs_builder = ObsBuilder(model, data, default_joint_pos, history=args.history) if policy else None
    cmd = np.array(args.cmd, dtype=np.float64)
    last_action = np.zeros(n_joint, dtype=np.float64)

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
    n_steps = int(args.duration / (dt * decimation))

    renderer = None
    frames: list[np.ndarray] = []
    if args.record:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    viewer_ctx = None
    if args.view:
        import mujoco.viewer

        viewer_ctx = mujoco.viewer.launch_passive(model, data)

    print(f"[run] {n_steps} policy steps @ {args.policy_hz} Hz (decimation={decimation}, dt={dt})")
    try:
        for step in range(n_steps):
            # --- policy 推理 ---
            if policy is not None:
                obs = obs_builder.build(cmd, last_action)
                if obs.ndim == 1:
                    obs = obs[None, :]
                act = policy(obs)
                act = np.asarray(act).reshape(-1)[:n_joint]
            else:
                act = np.zeros(n_joint)
            last_action = act.copy()

            target = default_joint_pos + action_scale * act

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
    ap.add_argument("--policy", help="策略文件 (.onnx / .pt)")
    ap.add_argument("--policy-type", choices=["auto", "onnx", "torch"], default="auto")
    ap.add_argument("--cmd", nargs=3, type=float, default=[0.5, 0.0, 0.0], metavar=("VX", "VY", "WZ"),
                    help="速度指令 (默认 0.5 0.0 0.0)")
    ap.add_argument("--duration", type=float, default=10.0, help="回放时长（秒）")
    ap.add_argument("--policy-hz", type=float, default=50.0, help="策略频率（InstinctLab=50Hz）")
    ap.add_argument("--history", type=int, default=8, help="观测历史帧数（对齐训练配置）")
    ap.add_argument("--action-scale", type=float, default=0.25,
                    help="回退用的全局 joint_pos 动作尺度（仅在执行器数 != 29 时生效）")
    ap.add_argument("--kp", type=float, default=100.0, help="手动 PD 的 kp")
    ap.add_argument("--kd", type=float, default=2.0, help="手动 PD 的 kd")
    ap.add_argument("--view", dest="view", action="store_true", default=True, help="打开 viewer（默认开）")
    ap.add_argument("--no-view", dest="view", action="store_false", help="不开 viewer（headless）")
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
