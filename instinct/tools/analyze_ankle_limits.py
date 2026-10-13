"""踝部限位问题的**数据取证**：参考动作里各关节离限位有多近？

    cd /home/leo/rsl_project/llp && source .../conda.sh && conda activate instinct
    python3 instinct/tools/analyze_ankle_limits.py

背景：T1 全程"踝部贴限位"。要区分两种可能：
  (a) 策略自己作出来的（奖励/尺度问题）—— 参考动作离限位很远；
  (b) **参考动作本身就在限位附近** —— 那"贴限位"是学参考的必然结果，
      需要的是"参考/限位标定"层面的修正，而不是再加惩罚项。
"""

from __future__ import annotations

import sys

import numpy as np
import yaml

sys.path.insert(0, "instinct/tools")
from joint_order import NPZ_ORDER  # type: ignore  # noqa: E402

MJCF = "/home/leo/rsl_project/AMP_mjlab/deploy/xmls/g1.xml"
SEL = (
    "instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1"
    "/motions/amp_walk/amp_walk_v13_motion_selection.yaml"
)
SOFT_FACTOR = 0.9        # 我们资产里的 `soft_joint_pos_limit_factor`（isaaclab：软限位 = 0.9×硬限位）
DEFAULT_ANKLE_PITCH = -0.363


def parse_mjcf_limits(path: str) -> dict[str, tuple[float, float]]:
    """从 MJCF 里抓 `<joint name=... range="lo hi">`（同一台机器人的硬限位）。"""
    import re

    txt = open(path).read()
    out: dict[str, tuple[float, float]] = {}
    for m in re.finditer(r'<joint\s+name="([^"]+)"[^>]*?range="([^"]+)"', txt):
        name, rng = m.group(1), m.group(2)
        lo, hi = (float(v) for v in rng.split())
        out[name] = (lo, hi)
    return out


def main() -> int:
    limits = parse_mjcf_limits(MJCF)
    print(f"从 MJCF 读到 {len(limits)} 个关节限位：{MJCF}")

    sel = yaml.safe_load(open(SEL))["selected_files"]
    base = "instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/motions/amp_walk/"
    qs = []
    for s in sel:
        d = np.load(base + s["file"])
        qs.append(np.asarray(d["joint_pos"], dtype=np.float64))
    Q = np.concatenate(qs, axis=0)
    print(f"参考集：{len(sel)} 个 clip、共 {Q.shape[0]} 帧、{Q.shape[1]} 关节\n")

    assert Q.shape[1] == len(NPZ_ORDER), f"npz 关节数 {Q.shape[1]} != NPZ_ORDER {len(NPZ_ORDER)}"

    rows = []
    for j, name in enumerate(NPZ_ORDER):
        if name not in limits:
            continue
        lo, hi = limits[name]
        slo, shi = lo * SOFT_FACTOR, hi * SOFT_FACTOR     # isaaclab：两侧各乘 factor
        col = Q[:, j]
        # 离**软限位**还有多少余量（负值 = 已越界）
        margin_lo = col - slo          # >0 = 在软限位内侧
        margin_hi = shi - col
        m = np.minimum(margin_lo, margin_hi)
        over = float((m < 0).mean())    # 越软限位的帧占比
        rows.append(
            dict(
                name=name,
                lo=lo,
                hi=hi,
                slo=slo,
                shi=shi,
                qmin=float(col.min()),
                qmax=float(col.max()),
                qmed=float(np.median(col)),
                min_margin=float(m.min()),
                med_margin=float(np.median(m)),
                over_frac=over,
            )
        )

    rows.sort(key=lambda r: r["med_margin"])
    hdr = f"{'joint':32s} {'硬限位':>17s} {'软限位':>17s} {'参考 [min, med, max]':>26s} {'中位余量':>8s} {'越软限位帧%':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        flag = " ⚠️" if r["med_margin"] < 0.05 else ""
        print(
            f"{r['name']:32s} [{r['lo']:+6.3f},{r['hi']:+6.3f}] "
            f"[{r['slo']:+6.3f},{r['shi']:+6.3f}] "
            f"[{r['qmin']:+6.3f},{r['qmed']:+6.3f},{r['qmax']:+6.3f}] "
            f"{r['med_margin']:8.3f} {100*r['over_frac']:9.1f}%{flag}"
        )

    print("\n=== 踝部专项 ===")
    for nm in ("left_ankle_pitch_joint", "right_ankle_pitch_joint",
               "left_ankle_roll_joint", "right_ankle_roll_joint"):
        j = NPZ_ORDER.index(nm)
        col = Q[:, j]
        lo, hi = limits[nm]
        print(
            f"{nm:26s} 参考 median={np.median(col):+.3f} p1={np.percentile(col,1):+.3f} "
            f"p99={np.percentile(col,99):+.3f} min={col.min():+.3f} max={col.max():+.3f}  "
            f"硬限位=[{lo:+.3f},{hi:+.3f}]  软限位=[{lo*SOFT_FACTOR:+.3f},{hi*SOFT_FACTOR:+.3f}]"
        )
    print(
        f"\n默认站姿 ankle_pitch = {DEFAULT_ANKLE_PITCH:+.3f}；"
        f"软下限 = {limits['left_ankle_pitch_joint'][0]*SOFT_FACTOR:+.3f}；"
        f"硬下限 = {limits['left_ankle_pitch_joint'][0]:+.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
