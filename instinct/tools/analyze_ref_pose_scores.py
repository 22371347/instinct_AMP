"""参考动作在**两项 mjlab 姿态项**上能拿几分？（回答"谁在逼腰立直"）

    python3 instinct/tools/analyze_ref_pose_scores.py

背景：本仓库 root = `torso_link`（mjlab 的 root = `pelvis`）⇒ 两项**以 root 为靶**的
mjlab 项在我们这里都变成了"躯干预度量"，可能与"参考走路时腰前倾/躯干摆动"冲突：
  * `track_base_height`（std 0.3，target = default_root_state z）：见
    `analyze_root_height.py` ⇒ 参考自身 **0.995**（很松、无冲突）
  * `body_ang_vel_xy`（std 3.14）：罚 **torso 的 ‖ω_xy^b‖** ⇒ 本脚本量它。
另量参考的 `waist_pitch` / `knee` 与我们的 default 之差（"立腰"的第二个候选机制）。
"""

from __future__ import annotations

import sys

import numpy as np
import yaml

sys.path.insert(0, "instinct/tools")
from joint_order import NPZ_ORDER  # type: ignore  # noqa: E402

BASE = (
    "instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1"
    "/motions/amp_walk/"
)
SEL = BASE + "amp_walk_v13_motion_selection.yaml"


def quat_mul(a, b):
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_conj(q):
    w, x, y, z = q.T
    return np.stack([w, -x, -y, -z], axis=-1)


def body_ang_vel(q, dt):
    """由四元数差分得到**本体系**角速度 ω_b ≈ 2·vec(q_{t+1} ⊗ q_t⁻¹)/dt。"""
    dq = quat_mul(q[1:], quat_conj(q[:-1]))
    sign = np.sign(dq[:, 0:1])
    sign[sign == 0] = 1.0
    dq = dq * sign                       # 取短弧
    vec = dq[:, 1:]
    return 2.0 * vec / dt


def main() -> int:
    sel = yaml.safe_load(open(SEL))["selected_files"]
    W, WP, KN, WY = [], [], [], []
    for s in sel:
        d = np.load(BASE + s["file"])
        q = np.asarray(d["base_quat_w"], dtype=np.float64)
        fps = float(d["framerate"])
        w = body_ang_vel(q, 1.0 / fps)
        W.append(w)
        jp = np.asarray(d["joint_pos"], dtype=np.float64)
        WP.append(jp[:, NPZ_ORDER.index("waist_pitch_joint")])
        KN.append(jp[:, NPZ_ORDER.index("left_knee_joint")])
        WY.append(jp[:, NPZ_ORDER.index("waist_yaw_joint")])
    W = np.concatenate(W)
    wxy = np.linalg.norm(W[:, :2], axis=1)
    wz = W[:, 2]
    print(f"参考动作的 **torso** 角速度（{len(W)} 帧，50 Hz）")
    print(f"  ‖ω_xy^b‖ : p50 {np.median(wxy):.3f}  p95 {np.percentile(wxy,95):.3f}  "
          f"max {wxy.max():.3f} rad/s")
    print(f"  ωz^b     : p50 {np.median(np.abs(wz)):.3f}  p95 {np.percentile(np.abs(wz),95):.3f}")

    for std in (3.14,):
        r = np.exp(-np.square(wxy) / std**2)
        print(f"\n代入 `body_ang_vel_xy`（std={std}）⇒ 参考自身得分："
              f"mean {r.mean():.4f}  p5 {np.percentile(r,5):.4f}  min {r.min():.4f}")
        print(f"  ⇒ 参考丢掉 {100*(1-r.mean()):.2f}% 的该项收益"
              f"（权重 0.7 ⇒ 最多 {0.7*(1-r.mean()):.4f}/s 的差距）")
    print("\n参考的腰/膝（对照我们的 default：waist_pitch 0.000 / knee 0.669）")
    WP, KN, WY = np.concatenate(WP), np.concatenate(KN), np.concatenate(WY)
    print(f"  waist_pitch: p1 {np.percentile(WP,1):+.3f}  p50 {np.median(WP):+.3f}  "
          f"p99 {np.percentile(WP,99):+.3f}   ⇒ 与 default 相差 {np.median(WP):+.3f} rad")
    print(f"  knee       : p1 {np.percentile(KN,1):+.3f}  p50 {np.median(KN):+.3f}  "
          f"p99 {np.percentile(KN,99):+.3f}   ⇒ 与 default 相差 {np.median(KN)-0.669:+.3f} rad")
    print(f"  waist_yaw  : p1 {np.percentile(WY,1):+.3f}  p50 {np.median(WY):+.3f}  "
          f"p99 {np.percentile(WY,99):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
