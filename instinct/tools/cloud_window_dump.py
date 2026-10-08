#!/usr/bin/env python3
"""按**指定迭代窗口**并排打印多个 run 的 TB 标量均值（用于同轮次公平 A/B）。

与 `cloud_amp_compare.py` 的区别：那个只能取"最后 N 个窗口"，做不了"两个 run 在**同一批
早期窗口**上对齐比较"（例如 V17 vs V18 都比 500-1000 / 1000-1500 轮）。本脚本专门干这个。

用法（云端）：
    /opt/conda/envs/isaaclab/bin/python /root/cloud_window_dump.py \
        v17 v18 --windows 250:1000,1000:1750 --tags auto

    # 也可显式给标签
    /opt/conda/envs/isaaclab/bin/python /root/cloud_window_dump.py v18 \
        --windows 0:500 --tags Episode_Reward/feet_air_time,Step/amp_style_reward

`--windows a:b`（左闭右开，可多个逗号分隔）。`--tags auto` 用下面的 AUTO_TAGS 表，
并**自动补上 raw 值**（`Episode_Reward/x` 是"每步加权均值"，TB 值 = 权重 × 原始量
⇒ 要跨版本比较行为时看 raw = TB/|weight|；本脚本按 KNOWN_WEIGHTS 折算）。
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics

LOGROOT = "/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk"

AUTO_TAGS = (
    "Step/amp_d",
    "Step/amp_d_ref",
    "Step/amp_style_reward",
    "Step/amp_style_std",
    "Step/discriminator_reward",
    "Loss/discriminator_loss",
    "Step/tilt_deg_mean",
    "Step/air_time_single_stance",
    "Step/air_time_in_band_frac",
    "Step/action_abs_mean",
    "Episode_Reward/track_lin_vel_xy_exp",
    "Episode_Reward/track_ang_vel_z_exp",
    "Episode_Reward/track_base_height",
    "Episode_Reward/feet_air_time",
    "Episode_Reward/action_rate_l2",
    "Episode_Reward/flat_orientation_l2",
    "Episode_Reward/feet_slide",
    "Episode_Reward/dof_acc_l2",
    "Episode_Reward/termination_penalty",
)

#: 版本 → 关键项的**权重**（用于把 TB 值折回"原始量"，跨版本比较行为时必须折算）
KNOWN_WEIGHTS = {
    "v17": {
        "Episode_Reward/action_rate_l2": -0.06,
        "Episode_Reward/feet_air_time": 4.0,
        "Episode_Reward/flat_orientation_l2": -2.0,
    },
    "v18": {
        "Episode_Reward/action_rate_l2": -0.10,
        "Episode_Reward/feet_air_time": 4.0,
        "Episode_Reward/flat_orientation_l2": -2.0,
    },
}


def resolve_run(arg: str) -> str | None:
    if os.path.isdir(arg):
        return arg
    cands = sorted(
        (d for d in glob.glob(os.path.join(LOGROOT, "*")) if arg.lower() in os.path.basename(d).lower()),
        key=lambda d: os.path.getmtime(d),
    )
    return cands[-1] if cands else None


def load_scalars(run_dir: str) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    return {t: [(int(e.step), float(e.value)) for e in ea.Scalars(t)] for t in ea.Tags().get("scalars", [])}


def main() -> int:
    ap = argparse.ArgumentParser(description="按指定迭代窗口并排打印多个 run 的 TB 均值")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--windows", required=True, help="如 250:1000,1000:1750（左闭右开，逗号分隔）")
    ap.add_argument("--tags", default="auto", help="auto 或逗号分隔的 tag 列表")
    args = ap.parse_args()

    windows = []
    for w in args.windows.split(","):
        lo, hi = w.split(":")
        windows.append((int(lo), int(hi)))
    tags = list(AUTO_TAGS) if args.tags == "auto" else args.tags.split(",")

    for arg in args.runs:
        run_dir = resolve_run(arg)
        print("=" * 100)
        if run_dir is None:
            print(f"❌ 找不到 run: {arg}")
            continue
        scalars = load_scalars(run_dir)
        # ⚠️ 不能对**所有** tag 取 `s[-1][0]` 的最大值：`Train/time/*` 那一组的 x 轴是
        #    **挂钟秒数**（实测 V18 报 8640 s ≈ 2.4 h，而真末轮是 2220），会把"最后轮号"
        #    顶成一个无意义的数。⇒ 只按**本表要打印的 tag** 求最大值。
        last = max((scalars[t][-1][0] for t in tags if t in scalars and scalars[t]), default=0)
        odd = sorted({t for t in scalars if scalars[t] and scalars[t][-1][0] > last * 1.2})
        print(f"run: {os.path.basename(run_dir)[:110]}\n最后轮号 = {last}（仅按本表 tag 计）")
        if odd:
            print(f"  ⚠️ 另有 {len(odd)} 个 tag 的 x 轴单位不同（如挂钟秒数），已排除："
                  f"{', '.join(odd[:4])}{' …' if len(odd) > 4 else ''}")
        header = "".join(f"{f'{lo}-{hi}':>15}" for lo, hi in windows)
        print(f"\n{'量':<40}{header}")
        print("-" * (40 + 15 * len(windows)))
        for t in tags:
            if t not in scalars:
                continue
            cells = ""
            for lo, hi in windows:
                vals = [v for s, v in scalars[t] if lo <= s < hi]
                cells += f"{(statistics.fmean(vals) if vals else float('nan')):>15.4f}"
            print(f"{t:<40}{cells}")
        # raw 折算（跨版本比行为时必须看这个）
        wmap = KNOWN_WEIGHTS.get(arg.lower().split("_")[0], {})
        if wmap:
            print(f"\n--- 折回原始量（TB / |权重|）---")
            for t, w in wmap.items():
                if t not in scalars or w == 0:
                    continue
                cells = ""
                for lo, hi in windows:
                    vals = [v for s, v in scalars[t] if lo <= s < hi]
                    cells += f"{(statistics.fmean(vals) / abs(w) if vals else float('nan')):>15.4f}"
                print(f"{t.split('/')[-1] + ' / ' + str(abs(w)):<40}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
