#!/usr/bin/env python3
"""**AMP 风格奖励核算**：把「判别器输出 → quad(D) → 风格奖励 → 进总奖励的那一份」逐级算出来，
并做**跨 run 窗口对比**（回答"改了 GP/wd/logit_wd/eps 之后，风格奖励具体变了什么"）。

为什么要专门算
--------------
线上能直接看到的是 `Step/amp_style_reward = −log(1−D)`（已是非线性后的值）与 `Step/amp_d`。
只看均值会漏掉三件关键的事：

1. **策略侧的"亏损深度"**：`style_actor` 越低 ⇒ 离参考越远、可提升空间越大；
2. ⭐ **奖励对判别器的灵敏度 `dr/dd = 2/(1−d)`** —— 这才是"风格项有没有梯度"的直接量。
   `d` 从 −0.62 走到 −0.82 ⇒ 灵敏度 1.23 → 2.19（×1.8）；
3. **参考/策略的风格差 `gap_style`** = 策略还欠多少 —— 也就是"风格项能给出的总激励"。

总奖励口径（`WasabiPPOAmpMix` + lerp）：
    R = λ·r_task + (1−λ)·c·style        （λ=`style_lerp`，c=`discriminator_reward_coef`）
    ⇒ 风格项每步贡献 = (1−λ)·c·style

用法（云端）：
    python cloud_amp_style_accounting.py --runs v18,v19a --last 40
    python cloud_amp_style_accounting.py --runs v18,v19a --window 500   # 按固定窗口分块
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

LOG_ROOT = "/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk"

STYLE_TAGS = [
    "Step/amp_d",
    "Step/amp_d_ref",
    "Step/amp_style_reward",
    "Step/amp_style_std",
    "Step/amp_style_zero_frac",
    "Step/discriminator_reward",
    "Loss/discriminator_loss",
    "Loss/discriminator_gradient_penalty",
    "Loss/discriminator_weight_decay",
    "Loss/logit_weight_decay",
    "Loss/amp_total_loss",
    "Episode_Reward/feet_air_time",
    "Episode_Reward/track_lin_vel_xy_exp",
    "Episode_Reward/action_rate_l2",
]


def load_run(sub: str) -> tuple[str, dict[str, np.ndarray]]:
    dirs = sorted(glob.glob(os.path.join(LOG_ROOT, f"*{sub}*")), key=os.path.getmtime)
    if not dirs:
        raise SystemExit(f"[X] 没有 run 匹配 {sub!r}")
    run = dirs[-1]
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(run, size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    out: dict[str, np.ndarray] = {}
    for t in STYLE_TAGS + [x for x in sorted(tags) if x.startswith("Episode_Reward/")]:
        if t in tags:
            ev = ea.Scalars(t)
            out[t] = np.array([e.value for e in ev], dtype=float)
    # 步号（用任意一个量的 step 数组；TB 每 10 轮记一次）
    any_tag = next(iter(out))
    out["__steps__"] = np.array([e.step for e in ea.Scalars(any_tag)], dtype=float)
    return run, out


def quad(d: float, eps: float, k: float = 0.25) -> float:
    """`D = clamp(1 − k·(d−1)², 0, 1−eps)`（`log_quad` 用的就是它）。

    ⚠️ `k` **必须与 run 的 `agent.yaml::amp_quad_k` 一致**：默认 0.25 是历史值，
    V19b 起改成 **0.125**（下端 clamp 从 `d ≤ −1.0` 推到 `d ≤ −1.83`）。
    上一版工具把 k 写死成 0.25 ⇒ 对 V19b/V19c 算出的 `style_actor` 偏差约 60 倍
    （0.0118 vs 实测 0.7075），**派生列会完全误导**。
    """
    return float(np.clip(1.0 - k * (d - 1.0) ** 2, 0.0, 1.0 - eps))


def style_log_quad(d: float, eps: float, k: float = 0.25) -> float:
    """`r = −log(1 − D)`；未触限幅时闭式 `−log k − 2·ln|d−1|`。"""
    D = quad(d, eps, k)
    return float(-np.log(max(1.0 - D, 1e-12)))


def slope(d: float) -> float:
    """`dr/dd = 2/(1−d)`（未触限幅时的闭式导数）。"""
    return float(2.0 / (1.0 - d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="v18,v19a")
    ap.add_argument("--last", type=int, default=40, help="取每个 run **最后 N 个 TB 采样点** 做平均")
    ap.add_argument("--style-lerp", type=float, default=0.75)
    ap.add_argument("--reward-coef", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.05, help="amp_reward_eps（V18 是 1e-6）")
    ap.add_argument("--quad-k", type=float, default=0.125,
                    help="amp_quad_k（⭐ 必须与 run 的 agent.yaml 一致；默认 0.125 = V19b 起的值）")
    ap.add_argument("--air-weight", type=float, default=4.0)
    ap.add_argument("--air-ceiling", type=float, default=0.6,
                    help="真·步行的单支撑时间占比（用来估 feet_air_time 的**上限**）")
    args = ap.parse_args()

    lam, c = args.style_lerp, args.reward_coef
    k = args.quad_k

    print(f"总奖励口径：R = λ·r_task + (1−λ)·c·style，λ={lam}、c={c}"
          f" ⇒ **风格项每步权重 (1−λ)c = {(1-lam)*c:g}**；quad k = {k}")

    rows = []
    for sub in args.runs.split(","):
        run, tb = load_run(sub)
        n = args.last
        mean = {k: float(v[-n:].mean()) for k, v in tb.items() if k != "__steps__"}
        step = int(tb["__steps__"][-1])
        d = mean.get("Step/amp_d", float("nan"))
        d_ref = mean.get("Step/amp_d_ref", float("nan"))
        # `amp_reward_eps` 按**版本**选：V18 及更早是 1e-6（上端限幅几乎不起作用）、
        # V19 系列是 0.05（把 `−log(1−D)` 上界压到 3.0）。
        # ⚠️ 之前写成 `if "v19a" in sub` ⇒ V19b/V19c/V19d 全落回 1e-6，
        #    `style_ref` 会算成 10.8 而不是实际生效的 3.0。
        eps = args.eps if "v19" in sub else 1e-6
        rows.append((sub, os.path.basename(run), step, d, d_ref, eps, mean))

    hdr = f"{'量':<44}" + "".join(f"{s:>14}" for s, *_ in rows)
    print(hdr)
    print("-" * len(hdr))

    def line(label: str, fn) -> None:
        print(f"{label:<44}" + "".join(f"{fn(r):>14.4g}" for r in rows))

    line("最后轮号", lambda r: r[2])
    print("\n--- 判别器输出 ---")
    line("d（策略侧，越低越差）", lambda r: r[3])
    line("d_ref（参考侧）", lambda r: r[4])
    line("gap = d_ref − d", lambda r: r[4] - r[3])
    print("\n--- quad / 风格奖励（用 d 的均值代入闭式，非线性量的近似）---")
    line(f"D = clamp(1−{k:g}(d−1)², 0, 1−eps)", lambda r: quad(r[3], r[5], k))
    line("D_ref", lambda r: quad(r[4], r[5], k))
    line("style_actor = −log(1−D)", lambda r: style_log_quad(r[3], r[5], k))
    line("style_ref", lambda r: style_log_quad(r[4], r[5], k))
    line("gap_style = style_ref − style_actor", lambda r: style_log_quad(r[4], r[5], k) - style_log_quad(r[3], r[5], k))
    line("⭐ 灵敏度 dr/dd = 2/(1−d)", lambda r: slope(r[3]))
    line("（TB 实测）Step/amp_style_reward", lambda r: r[6].get("Step/amp_style_reward", float("nan")))
    line("（TB 实测）Step/amp_style_std", lambda r: r[6].get("Step/amp_style_std", float("nan")))
    line("（TB 实测）amp_style_zero_frac", lambda r: r[6].get("Step/amp_style_zero_frac", float("nan")))
    print("\n--- 进总奖励的那一份 ---")
    line(f"风格项/步 = (1−λ)c·style", lambda r: (1 - lam) * c * style_log_quad(r[3], r[5], k))
    line("（TB 实测）Step/discriminator_reward", lambda r: r[6].get("Step/discriminator_reward", float("nan")))
    line("r_task/步（Σ Episode_Reward）", lambda r: sum(v for k, v in r[6].items() if k.startswith("Episode_Reward/")))
    line("⇒ 风格项占 r_task 的比例", lambda r: (1 - lam) * c * style_log_quad(r[3], r[5], k)
         / max(sum(v for k2, v in r[6].items() if k2.startswith("Episode_Reward/")), 1e-9))

    print("\n--- 步态项（feet_air_time）---")
    line("Episode_Reward/feet_air_time（加权）", lambda r: r[6].get("Episode_Reward/feet_air_time", float("nan")))
    line("⇒ 原始 score 均值 = 值/权重", lambda r: r[6].get("Episode_Reward/feet_air_time", float("nan")) / args.air_weight)
    line(f"⭐ 上限（占比{args.air_ceiling}·权重）", lambda r: args.air_weight * args.air_ceiling)
    line("⇒ 已拿到上限的百分比", lambda r: 100.0 * r[6].get("Episode_Reward/feet_air_time", 0.0)
         / (args.air_weight * args.air_ceiling))
    line("track_lin_vel_xy_exp/步", lambda r: r[6].get("Episode_Reward/track_lin_vel_xy_exp", float("nan")))

    print("\n--- 判别器损失（占比）---")
    line("discriminator_loss", lambda r: r[6].get("Loss/discriminator_loss", float("nan")))
    line("gradient_penalty（未乘系数）", lambda r: r[6].get("Loss/discriminator_gradient_penalty", float("nan")))
    line("amp_total_loss", lambda r: r[6].get("Loss/amp_total_loss", float("nan")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
