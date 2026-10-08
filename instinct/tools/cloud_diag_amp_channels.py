#!/usr/bin/env python3
"""**专家集 vs 参考 npz 的朝向通道自洽检验**（零 GPU，只需 numpy/torch）。

动机
----
用户 2026-10-08 观察：**"AMP 影响大的前进模态都严重后倾"** ⇒ 怀疑数据管线把朝向搞错了。
本脚本用两条独立证据交叉验证：

  A. 参考 npz 侧：从 `base_quat_w`（= torso_link 的世界位姿）算躯干倾斜
     `tilt = arccos(up_z)` 的分布（按帧加权）。
  B. 专家集侧：从 `anchor_gravity_b`（`x[:,390:396]`）反推同一个量
     `tilt = arccos(-g_z)`（因为 g = R^T·ĝ，ĝ=(0,0,-1) ⇒ -g = R^T·(0,0,1) = 本体系里的"上"）
  ⇒ **若两侧分布一致 ⇒ 导出管线的朝向/重力通道自洽**；若差很多（如专家集≈0 或≈180）
     ⇒ 朝向或重力通道的口径错了。

同时做内部自洽检查：
  * body0（anchor）的 `pos` 通道应 ≈ 0；`ori` 通道应 = **Identity 的 rot6d**
    （即 `[1,0,0,0,1,0]`；因为我们记的语义是"各刚体相对 anchor"）
  * `‖grav‖` 应为 1

用法（云端）：
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_diag_amp_channels.py
（不需要 isaaclab ⇒ 不需要 AppLauncher，也不需要 source .llp_env.sh）
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

MOTION_DIR = "/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/motions/amp_walk"
EXPERT = "/root/amp_expert_states_396.pt"
SEL = "amp_walk_v13_motion_selection.yaml"
NB = 13


def q2R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def rot6(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (a * b).sum(-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], axis=-1)


def main() -> int:
    # ---------- A. 参考 npz ----------
    try:
        import yaml
        sel = yaml.safe_load(open(os.path.join(MOTION_DIR, SEL)))["selected_files"]
        files = [s["file"] if isinstance(s, dict) else s for s in sel]
    except Exception:
        files = [os.path.basename(p) for p in sorted(glob.glob(MOTION_DIR + "/*_retargeted.npz"))]

    print("=" * 78)
    print("A. 参考 npz：躯干（torso_link）倾斜分布")
    print("=" * 78)
    tilts = []
    for f in files:
        p = os.path.join(MOTION_DIR, f)
        if not os.path.exists(p):
            print(f"  (缺) {f}")
            continue
        z = np.load(p, allow_pickle=True)
        q = z["base_quat_w"]
        up = q2R(q) @ np.array([0.0, 0.0, 1.0])
        t = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
        tilts.append(t)
        print(f"  {f[:40]:40s} T={len(t):4d}  |tilt| p5/p50/p95 = "
              f"{np.percentile(t,5):5.1f}/{np.percentile(t,50):5.1f}/{np.percentile(t,95):5.1f} deg")
    ref = np.concatenate(tilts)
    print(f"  >>> （按帧加权）p5/p50/p95 = {np.percentile(ref,5):.1f}/"
          f"{np.percentile(ref,50):.1f}/{np.percentile(ref,95):.1f} deg   mean={ref.mean():.1f}")

    # ---------- B. 专家集 ----------
    print()
    print("=" * 78)
    print("B. 专家集：anchor_gravity_b 反推的倾斜 + 内部自洽")
    print("=" * 78)
    raw = torch.load(EXPERT, map_location="cpu")
    x = (next(v for v in raw.values() if isinstance(v, torch.Tensor)) if isinstance(raw, dict) else raw)
    x = x.float().numpy()
    pos = x[:, 0:78].reshape(-1, NB, 3)
    ori = x[:, 78:234].reshape(-1, NB, 6)
    lin = x[:, 234:312].reshape(-1, NB, 3)
    ang = x[:, 312:390].reshape(-1, NB, 3)
    grav = x[:, 390:396]
    print(f"  shape={x.shape}（396 = {NB}body×(pos3+ori6+lin3+ang3) + grav3）")

    gz = np.clip(-grav[:, 2], -1, 1)
    t_exp = np.degrees(np.arccos(gz))
    print(f"  由 grav 反推 |tilt| p5/p50/p95 = {np.percentile(t_exp,5):.1f}/"
          f"{np.percentile(t_exp,50):.1f}/{np.percentile(t_exp,95):.1f} deg   mean={t_exp.mean():.1f}")
    print(f"  ‖grav‖ mean={np.linalg.norm(grav,axis=1).mean():.4f}（应=1.0）")

    print("\n  body0（anchor）通道：")
    print(f"    pos[:,0] mean={np.round(pos[:,0].mean(0),4)} std={np.round(pos[:,0].std(0),4)}（应≈0）")
    print(f"    ori[:,0] mean={np.round(ori[:,0].mean(0),4)}（应=Identity rot6d [1,0,0,0,1,0]）")
    R0 = rot6(ori[:, 0, :3], ori[:, 0, 3:])
    print(f"    ‖mean(R) − I‖_F = {np.linalg.norm(R0.mean(0) - np.eye(3)):.4f}")
    print(f"    lin[:,0] mean={np.round(lin[:,0].mean(0),4)}  ang[:,0] mean={np.round(ang[:,0].mean(0),4)}")

    print("\n  逐块统计（|均值| 与 std 的均值）：")
    for nm, blk in (("pos", pos), ("ori", ori), ("lin", lin), ("ang", ang)):
        m = blk.reshape(len(x), -1).mean(0)
        s = blk.reshape(len(x), -1).std(0)
        print(f"    {nm:4s} mean|μ|={np.abs(m).mean():.4f}  mean σ={s.mean():.4f}  max|μ|={np.abs(m).max():.4f}")

    # ---------- 结论 ----------
    print()
    print("=" * 78)
    d50 = abs(np.percentile(ref, 50) - np.percentile(t_exp, 50))
    print(f"结论：参考 npz 中位 |tilt| = {np.percentile(ref,50):.1f}°  vs  "
          f"专家集反推 {np.percentile(t_exp,50):.1f}°  （差 {d50:.1f}°）")
    if d50 <= 5.0:
        print("  ✅ 两侧一致 ⇒ 导出管线的**朝向/重力通道自洽**，后倾不是这里造成的")
    else:
        print("  ❌ 两侧相差过大 ⇒ **朝向/重力通道口径可疑**，需查导出脚本与环境的实现")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
