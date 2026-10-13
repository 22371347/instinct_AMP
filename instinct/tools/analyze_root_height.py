"""`track_base_height` 是否与参考动作冲突？—— 量它三个数。

    python3 instinct/tools/analyze_root_height.py

本仓库 root = `torso_link` ⇒ `track_base_height_exp` 量的是**躯干高度**
（目标 = `default_root_state[:, 2]` = 0.82 m，std = 0.3）。
而 mjlab 的 root = `pelvis` ⇒ 它的 `track_root_height` 量的是**骨盆高度**
（= "别塌"，与腰 pitch 几乎解耦）。同样 1.5/0.3 在两边**语义不同**。

本脚本量：
  ① 参考动作（npz 的 `base_pos_w[:, 2]`）的躯干高度分布 —— 与 0.82 / std 0.3 对照；
  ② 参考动作里 waist_pitch / knee 的变化（解释高度起伏来自哪）；
  ③ 把参考帧的高度代进 `exp(-e²/std²)`，看参考自身能拿几分（<1 就说明与本项冲突）。
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
TARGET = 0.82          # = `init_state.pos_z`（也是 `default_root_state[:,2]`）
STD = 0.3              # = 本项生效的 std（T1l/T1o/T2 都是 0.3）


def main() -> int:
    sel = yaml.safe_load(open(SEL))["selected_files"]
    zs, wp, kn = [], [], []
    print(f"{'clip':44s} {'|z| min':>8s} {'p50':>8s} {'max':>8s} {'起伏(max-min)':>13s}")
    print("-" * 86)
    for s in sel:
        d = np.load(BASE + s["file"])
        z = np.asarray(d["base_pos_w"], dtype=np.float64)[:, 2]
        q = np.asarray(d["joint_pos"], dtype=np.float64)
        i_wp = NPZ_ORDER.index("waist_pitch_joint")
        i_kn = NPZ_ORDER.index("left_knee_joint")
        zs.append(z)
        wp.append(q[:, i_wp])
        kn.append(q[:, i_kn])
        print(f"{s['file'][:42]:44s} {z.min():8.3f} {np.median(z):8.3f} {z.max():8.3f} "
              f"{z.max()-z.min():13.3f}")
    Z = np.concatenate(zs)
    WP = np.concatenate(wp)
    KN = np.concatenate(kn)

    print(f"\n=== ① 参考动作的躯干高度（{Z.size} 帧） ===")
    print(f"  min {Z.min():.3f}  p1 {np.percentile(Z,1):.3f}  p50 {np.median(Z):.3f}  "
          f"p99 {np.percentile(Z,99):.3f}  max {Z.max():.3f}")
    print(f"  目标 {TARGET:.3f}（= 站姿）   参考中位 − 目标 = {np.median(Z)-TARGET:+.3f} m")
    print(f"  |z − 目标| 的分布：p50 {np.median(np.abs(Z-TARGET)):.3f}  "
          f"p95 {np.percentile(np.abs(Z-TARGET),95):.3f}  max {np.abs(Z-TARGET).max():.3f} m")

    r = np.exp(-np.square(Z - TARGET) / STD**2)
    print(f"\n=== ② 把参考帧代进 exp(−e²/{STD}²) ===")
    print(f"  参考自身得分：p50 {np.median(r):.3f}  p5 {np.percentile(r,5):.3f}  "
          f"min {r.min():.3f}  mean {r.mean():.3f}")
    print(f"  ⇒ 满分 1.0 意味着\"躯干高度恒定\"；参考自身只拿 {np.mean(r):.3f} "
          f"⇒ **与本项存在系统性冲突**（{100*(1-np.mean(r).mean()):.1f}% 的收益拿不到）")

    print(f"\n=== ③ 参考动作的 waist_pitch / knee（解释高度起伏）===")
    print(f"  waist_pitch: p1 {np.percentile(WP,1):+.3f} p50 {np.median(WP):+.3f} "
          f"p99 {np.percentile(WP,99):+.3f}（我们的 default = 0.000）")
    print(f"  knee      : p1 {np.percentile(KN,1):+.3f} p50 {np.median(KN):+.3f} "
          f"p99 {np.percentile(KN,99):+.3f}（我们的 default = 0.669）")
    print("\n如果政策把 torso 高度锁在 0.82（现在 `Episode_Reward/track_base_height` 已 99%），")
    print("它就必须压住 waist_pitch(0) 与 knee(0.669) 的可动范围 ⇒ 视觉上就是\"腰被立直/锁死\"。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
