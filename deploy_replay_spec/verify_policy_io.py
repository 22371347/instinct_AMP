#!/usr/bin/env python3
"""在**任意**项目里自检 P1 策略的观测/动作接口（只依赖 numpy + onnxruntime，不需要 MuJoCo）。

用途：把训练好的 P1 策略接进你那个 MuJoCo 部署项目后，先用这个脚本确认
"观测拼对了、归一化对了、动作量级正常"，再去跑闭环。

三种用法
--------
1) 出生帧参考（不发你项目里的任何数据，纯本地算参考向量）：
       python verify_policy_io.py --birth
   打印：1500 维参考观测各段取值 + 归一化后的 |z| + 策略输出动作。
   你可以把自己在**默认站姿**下拼出的 1500 维存成 .npy 再对比：

2) 对比你自己拼的观测：
       python verify_policy_io.py --obs my_obs.npy            # (1500,) 或 (1,1500)
   会打印：逐段 |z|max（>3 就是那一段定义/缩放错了）+ 与你出生帧参考的逐段差异。

3) 只看策略对某个观测的输出：
       python verify_policy_io.py --obs my_obs.npy --no-reference

判定标准（本仓库实测）：
   · 出生帧：height_scan 全 = pelvis_z-0.5 = 0.2585；projected_gravity ≈ (0,0,-1)；
     joint_pos_rel / joint_vel / last_action 全 0；|z|max ≈ 1.5、|z|>3 的维度 0 个；
     策略输出 |action| ≲ 0.3。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

#: 观测分段（与训练契约 `T2_OBSERVATION_DIMS` 逐项一致）
SEGMENTS = [
    ("height_scan", 0, 1116, 279, 4),
    ("velocity_command", 1116, 1128, 3, 4),
    ("projected_gravity", 1128, 1140, 3, 4),
    ("base_ang_vel", 1140, 1152, 3, 4),
    ("joint_pos_rel", 1152, 1268, 29, 4),
    ("joint_vel", 1268, 1384, 29, 4),
    ("last_action", 1384, 1500, 29, 4),
]
OBS_DIM = 1500
ACTION_DIM = 29
PELVIS_Z_DEFAULT = 0.7585      # 默认站姿下的骨盆高度
SCAN_OFFSET = 0.5             # height_scan = pelvis_z - hit_z - 0.5


def birth_obs(cmd=(0.0, 0.0, 0.0)) -> np.ndarray:
    """默认站姿 + 平地 的参考观测（1500 维）。"""
    obs = np.zeros(OBS_DIM, dtype=np.float64)
    # ① height_scan：平地 ⇒ 每条射线都 = pelvis_z - 0.5，4 帧相同
    obs[0:SEGMENTS[0][2]] = PELVIS_Z_DEFAULT - SCAN_OFFSET
    # ② velocity_command：4 帧相同
    for f in range(4):
        obs[1116 + 3 * f: 1116 + 3 * f + 3] = cmd
    # ③ projected_gravity：直立 ⇒ (0,0,-1)，4 帧相同
    for f in range(4):
        obs[1128 + 3 * f: 1128 + 3 * f + 3] = (0.0, 0.0, -1.0)
    # ④ base_ang_vel / ⑤ joint_pos_rel / ⑥ joint_vel / ⑦ last_action 全 0
    return obs


class Normalizer:
    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=True)
        self.mean = np.asarray(z["mean"], dtype=np.float64).reshape(1, -1)
        self.std = np.asarray(z["std"], dtype=np.float64).reshape(1, -1)
        self.eps = float(np.asarray(z["eps"]).reshape(-1)[0]) if "eps" in z.files else 1e-2

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        return ((x - self.mean) / (self.std + self.eps)).astype(np.float32)


def run_policy(onnx_path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: x.astype(np.float32)})[0]
    return np.asarray(out).reshape(-1)


def report(tag: str, obs: np.ndarray, z: np.ndarray | None, act: np.ndarray) -> None:
    print(f"\n===== {tag} =====")
    print(f"观测维度 = {len(obs)}（期望 {OBS_DIM}）   动作 |max| = {np.abs(act).max():.3f}")
    print(f"{'段':20s} {'维度':>5s} {'取值范围':>26s} {'|z|max':>8s}")
    for nm, a, b, per, fr in SEGMENTS:
        seg = obs[a:b]
        zt = "n/a" if z is None else f"{np.abs(z[a:b]).max():.3f}"
        print(f"{nm:20s} {b - a:5d} [{seg.min():+.4f}, {seg.max():+.4f}]".ljust(60)
              + f" {zt:>8s}")
    if z is not None:
        n_bad = int((np.abs(z) > 3).sum())
        print(f"  |z|max(全体) = {np.abs(z).max():.3f}   |z|>3 的维度 = {n_bad}/{OBS_DIM}"
              + ("   ✅ 落在训练分布内" if n_bad == 0 else "   ❌ 有 OOD 段，按上表定位"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default=str(HERE / "actor.onnx"))
    ap.add_argument("--norm", default=str(HERE / "policy_normalizer.npz"))
    ap.add_argument("--obs", help="你自己拼的 1500 维观测 .npy")
    ap.add_argument("--birth", action="store_true", help="只看出生帧参考（用本脚本造）")
    ap.add_argument("--cmd", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
    ap.add_argument("--no-reference", action="store_true", help="不与出生帧参考做逐段对比")
    args = ap.parse_args(argv)

    onnx_path, norm_path = Path(args.onnx), Path(args.norm)
    for p in (onnx_path, norm_path):
        if not p.is_file():
            raise SystemExit(f"找不到 {p}")
    norm = Normalizer(norm_path)

    ref = birth_obs(tuple(args.cmd))
    if args.birth or args.obs is None:
        act = run_policy(onnx_path, norm(ref))
        report("出生帧参考（默认站姿 + 平地，cmd=%s）" % (tuple(args.cmd),), ref, norm(ref).reshape(-1), act)
        if args.obs is None and not args.birth:
            print("\n提示：要检查你自己拼的观测，用 --obs my_obs.npy")
        return 0

    obs = np.load(args.obs).reshape(-1).astype(np.float64)
    if len(obs) != OBS_DIM:
        raise SystemExit(f"观测维度 {len(obs)} != {OBS_DIM}")
    z = norm(obs).reshape(-1)
    act = run_policy(onnx_path, z)
    report(f"你的观测 {args.obs}", obs, z, act)

    if not args.no_reference:
        zref = norm(ref).reshape(-1)
        print("\n--- 与你出生帧参考的逐段差异（只在出生帧状态才有意义）---")
        for nm, a, b, _, _ in SEGMENTS:
            d = float(np.abs(obs[a:b] - ref[a:b]).max())
            dz = float(np.abs(z[a:b] - zref[a:b]).max())
            flag = "  <<< 差得多" if dz > 1.0 else ""
            print(f"  {nm:20s} Δobs max = {d:+.5f}   Δ|z| max = {dz:.3f}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
