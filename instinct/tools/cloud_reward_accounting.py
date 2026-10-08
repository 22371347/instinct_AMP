#!/usr/bin/env python3
"""奖励项**会计 + 波动（常数项）分析**：从 TB event 文件里读 `Episode_Reward/*` 等序列。

回答的问题
----------
"AMP 风格奖励的影响是不是被一堆**几乎不变的常数项**淹没了？"
⇒ 对每一项算出：尾段均值(/s)、尾段标准差、**波动率 std/|mean|**、占 Σ|全部项| 的比例，
   并汇总"常数项"（波动率 < 5%）占总回报的比例。

⚠️ 口径提醒（别混）
------------------
* TB 里的 `Episode_Reward/x` 是**每步的（加权）时间平均** ——
  IsaacLab 的 `RewardManager` 累计 `Σ_steps (value × weight × dt)`，日志里除以
  `max_episode_length_s` ⇒ `= Σ(value·weight)·dt/N = 每步加权均值`。
  **不是"每秒速率"**（早先按 ÷50 理解是错的 ⇒ 曾把 34° 后仰误算成 4.35°）。
  ⇒ 每步贡献 = mean；**每回合贡献 = ×20**（因为 episode = 20 s 且累计时乘了 dt：
     Σ_steps(v·w·dt) = 0.02 × 1000 × mean(v·w) = 20 × mean）。
  且它**未乘 λ**（λ 混合发生在算法侧的 `auxiliary_reward` 里）；乘 λ 后才是策略真正优化的量。
  🔎 自检：`flat_orientation_l2 = −0.288` ⇒ Σg_xy² = 0.288 ⇒ 倾角 ≈ 30.8°
     （与 policy normalizer 的 `projected_gravity` 均值反推一致 ✅）。
* AMP 相关量在算法侧记，键名含 `amp`/`discriminator`/`style`（本脚本一并打印）。

用法（云端）
------------
    /opt/conda/envs/isaaclab/bin/python /root/cloud_reward_accounting.py <run 名或路径> [...] \
        [--tail 0.3] [--top 40] [--all-tags]

例
--
    /opt/conda/envs/isaaclab/bin/python /root/cloud_reward_accounting.py v14 --all-tags
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys

LOGROOT = "/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk"

#: `--trend` 关注的量（AMP 贡献相关 + 两个关键奖励）
TREND_TAGS = (
    "Step/amp_d",
    "Step/amp_style_reward",
    "Step/amp_style_std",
    "Step/discriminator_reward",
    "Loss/mirror",
    "Episode_Reward/flat_orientation_l2",
    "Episode_Reward/feet_air_time",
    "Episode_Reward/track_lin_vel_xy_exp",
)


def resolve_run(arg: str) -> str | None:
    """把「run 名/子串/绝对路径」解析成 run 目录（取匹配中最新的那个）。"""
    if os.path.isdir(arg):
        return arg
    cands = sorted(
        (d for d in glob.glob(os.path.join(LOGROOT, "*")) if arg.lower() in os.path.basename(d).lower()),
        key=lambda d: os.path.getmtime(d),
    )
    return cands[-1] if cands else None


def load_scalars(run_dir: str) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})  # 0 = 全部保留
    ea.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [(int(e.step), float(e.value)) for e in ea.Scalars(tag)]
    return out


def tail_stats(series: list[tuple[int, float]], frac: float) -> tuple[float, float, float, float, int]:
    """返回 (尾段均值, 尾段标准差, 尾段最小值, 尾段最大值, 尾段点数)。"""
    n = len(series)
    k = max(20, int(n * frac))
    vals = [v for _, v in series[-k:]]
    if not vals:
        return (float("nan"),) * 4 + (0,)
    return (statistics.fmean(vals), statistics.pstdev(vals), min(vals), max(vals), len(vals))


def main() -> int:
    ap = argparse.ArgumentParser(description="奖励会计 + 常数项占比分析")
    ap.add_argument("runs", nargs="+", help="run 名/子串/路径")
    ap.add_argument("--tail", type=float, default=0.3, help="尾段比例（默认 0.3）")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--all-tags", action="store_true", help="也打印非 Episode_Reward 的标量")
    ap.add_argument("--const-threshold", type=float, default=0.05,
                    help="波动率阈值：std/|mean| 低于它视为'几乎不变'（默认 5%%）")
    ap.add_argument("--trend", action="store_true",
                    help="额外打印 AMP/关键奖励的**时间趋势**与尾段斜率（看'变化率'）")
    args = ap.parse_args()

    for arg in args.runs:
        run_dir = resolve_run(arg)
        print("=" * 92)
        if run_dir is None:
            print(f"❌ 找不到 run: {arg}")
            continue
        print(f"run: {os.path.basename(run_dir)}")
        scalars = load_scalars(run_dir)
        if not scalars:
            print("❌ TB 里没有标量")
            continue

        rew_tags = [t for t in scalars if t.startswith("Episode_Reward/")]
        rows = []
        for t in rew_tags:
            mean, std, lo, hi, k = tail_stats(scalars[t], args.tail)
            if k == 0:
                continue
            rows.append({"tag": t.split("/", 1)[1], "mean": mean, "std": std, "lo": lo, "hi": hi,
                         "const": abs(std) / max(abs(mean), 1e-9), "n": k})
        rows.sort(key=lambda r: -abs(r["mean"]))

        sum_pos = sum(r["mean"] for r in rows if r["mean"] > 0)
        sum_neg = sum(r["mean"] for r in rows if r["mean"] < 0)
        sum_abs = sum(abs(r["mean"]) for r in rows)
        const_rows = [r for r in rows if r["const"] < args.const_threshold]
        const_abs = sum(abs(r["mean"]) for r in const_rows)

        last_iter = max(series[-1][0] for series in scalars.values() if series)
        print(f"尾段：最后 {rows[0]['n'] if rows else 0} 个记录点（约 tail={args.tail:.0%}）｜最后轮号≈{last_iter}")
        print(f"\n{'奖励项':<34}{'每步均值':>10}{'×20=每回合':>12}{'标准差':>10}{'波动率':>9}{'占比':>8}")
        print("-" * 92)
        for r in rows[: args.top]:
            share = abs(r["mean"]) / max(sum_abs, 1e-9)
            flag = "  ← 常数" if r["const"] < args.const_threshold else ""
            print(f"{r['tag']:<34}{r['mean']:>10.4f}{r['mean'] * 20:>12.3f}"
                  f"{r['std']:>10.4f}{r['const']:>9.3f}{share:>8.1%}{flag}")
        print("-" * 92)
        print(f"{'Σ 正 / Σ 负 / Σ|·|':<34}{sum_pos:>10.4f}{sum_neg:>12.4f}"
              f"{'':>10}{'':>9}{1.0:>8.1%}")
        print(f"{'（×20 = 每回合；episode 20 s @ dt 0.02）':<34}{sum_pos * 20:>10.2f}{sum_neg * 20:>12.2f}")
        print(f"\n⭐ 常数项（波动率 < {args.const_threshold:.0%}）占 Σ|全部项| 的 "
              f"{const_abs / max(sum_abs, 1e-9):.1%}"
              f"（{len(const_rows)}/{len(rows)} 项）")
        if const_rows:
            print("   其中最大的几项：", ", ".join(
                f"{r['tag']}({r['mean']:+.3f}/s, 波动率{r['const']:.3f})" for r in const_rows[:6]))

        amp_tags = [t for t in scalars if any(s in t.lower() for s in ("amp", "discrim", "style", "mirror"))]
        if amp_tags:
            print(f"\n--- AMP / 判别器 / 镜像 相关量（尾段均值 ± 标准差）---")
            for t in amp_tags[: args.top]:
                mean, std, lo, hi, k = tail_stats(scalars[t], args.tail)
                if k == 0:
                    continue
                print(f"  {t:<52}{mean:>10.4f} ± {std:.4f}   [{lo:.4f}, {hi:.4f}]")

        if args.all_tags:
            others = [t for t in scalars if not t.startswith("Episode_Reward/") and t not in amp_tags]
            print(f"\n--- 其它标量（尾段均值）---")
            for t in sorted(others)[: args.top]:
                mean, std, *_ = tail_stats(scalars[t], args.tail)
                print(f"  {t:<60}{mean:>12.4f} ± {std:.4f}")

        if args.trend:
            print("\n--- 时间趋势（等距采样 12 点 + 尾段斜率/1000 轮）---")
            for t in TREND_TAGS:
                if t not in scalars or not scalars[t]:
                    print(f"  {t:<44}<无>")
                    continue
                series = scalars[t]
                pts = series[:: max(1, len(series) // 12)]
                line = " ".join(f"{s}:{v:+.3f}" for s, v in pts)
                # 尾段 60% 的最小二乘斜率（单位：每 1000 轮）
                tail = series[-max(10, int(len(series) * 0.6)):]
                xs = [float(s) for s, _ in tail]
                ys = [float(v) for _, v in tail]
                mx, my = statistics.fmean(xs), statistics.fmean(ys)
                den = sum((x - mx) ** 2 for x in xs)
                slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den > 0 else 0.0
                print(f"  {t:<44}斜率/1000轮 = {slope * 1000:+.5f}")
                print(f"      {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
