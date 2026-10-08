#!/usr/bin/env python3
"""AMP 风格**中间量**的分段对比：把 TB 标量按**迭代窗口**取均值，多个 run 并排。

用途：回答"V15 在 9000/9500/10000 轮之间，风格相关量到底怎么变的？V14/V11 的对应值是多少？"

用法（云端）：
    python /root/cloud_amp_compare.py v15 v14 v11 v12 \
        --window 250 --last 6 --style-scale v15=0.25 v14=0.0625 v11=0.25 v12=0.25

`--style-scale` 给每个 run 的**有效风格权重** (1−λ)c（可选）：给了就多打一行
"风格入账贡献 = scale × amp_style_reward"（每步）。
"""

from __future__ import annotations

import argparse
import glob
import os
from statistics import fmean

LOGROOT = "/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk"

TAGS = (
    "Step/amp_d",
    "Step/amp_style_reward",
    "Step/amp_style_std",
    "Step/amp_style_zero_frac",
    "Step/amp_stand_frac",
    "Step/discriminator_reward",
    "Loss/discriminator_loss",
    "Loss/discriminator_weight_decay",
    "Loss/discriminator_gradient_penalty",
    "Train/discriminator_actor",
    "Train/discriminator_reference",
    "Loss/mirror",
    "Step/tilt_deg_mean",
    "Step/tilt_p95_deg",
    "Step/tilt_gt30_frac",
    "Step/tilt_gt45_frac",
    "Episode_Reward/flat_orientation_l2",
    "Episode_Reward/feet_air_time",
    "Episode_Reward/action_rate_l2",
    "Episode_Reward/track_lin_vel_xy_exp",
    "Episode_Reward/feet_slide",
    "Episode_Reward/feet_swing_time_penalty",
)


def resolve(arg: str) -> str | None:
    if os.path.isdir(arg):
        return arg
    c = [d for d in glob.glob(os.path.join(LOGROOT, "*")) if arg.lower() in os.path.basename(d).lower()]
    return sorted(c, key=os.path.getmtime)[-1] if c else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--window", type=int, default=250, help="窗口宽度（迭代）")
    ap.add_argument("--last", type=int, default=6, help="只看最后 N 个窗口")
    ap.add_argument("--style-scale", nargs="*", default=[],
                    help="每个 run 的有效风格权重 (1−λ)c，形如 v15=0.25")
    args = ap.parse_args()

    scale = {}
    for kv in args.style_scale:
        k, v = kv.split("=")
        scale[k.lower()] = float(v)

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    for arg in args.runs:
        d = resolve(arg)
        print("=" * 100)
        if d is None:
            print(f"❌ 找不到 run: {arg}")
            continue
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        avail = set(ea.Tags().get("scalars", []))
        series = {}
        for t in TAGS:
            if t in avail:
                s = ea.Scalars(t)
                if s:
                    series[t] = [(int(e.step), float(e.value)) for e in s]
        if not series:
            print("❌ 无标量")
            continue
        last_step = max(v[-1][0] for v in series.values())
        first_step = min(v[0][0] for v in series.values())
        print(f"run: {os.path.basename(d)}")
        print(f"迭代范围: {first_step} → {last_step}（窗口宽 {args.window}）")
        # 窗口边界：以 last_step 为右端往左取最多 args.last 个**非空**窗口
        # ⚠️ 之前的写法 `max(first_step, right-(i+1)*w)` 会把最左边那个窗口压成
        #    [9000, 8881) 这种**反向空窗口**（run 不够 6×w 宽时），标签也是错的
        #    ⇒ 现在显式跳过完全落在数据左侧的窗口。
        w = args.window
        right = last_step + 1
        windows = []
        for i in range(args.last):
            hi = right - i * w
            lo = max(first_step, hi - w)
            if hi <= first_step:
                break
            windows.append((lo, hi))
        windows = windows[::-1]
        header = "".join(f"{f'{lo}-{hi}':>15}" for lo, hi in windows)
        print(f"\n{'量':<44}{header}")
        print("-" * (44 + 15 * len(windows)))
        for t in TAGS:
            if t not in series:
                continue
            ss = series[t]
            cells = ""
            for lo, hi in windows:
                vals = [v for st, v in ss if lo <= st < hi]
                cells += f"{(fmean(vals) if vals else float('nan')):>15.4f}"
            print(f"{t:<44}{cells}")
        sc = scale.get(arg.lower())
        if sc is not None and "Step/amp_style_reward" in series:
            ss = series["Step/amp_style_reward"]
            cells = ""
            for lo, hi in windows:
                vals = [v for st, v in ss if lo <= st < hi]
                cells += f"{(sc * fmean(vals) if vals else float('nan')):>15.4f}"
            print(f"{f'⭐ 风格入账贡献 = {sc}×style（每步）':<44}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
