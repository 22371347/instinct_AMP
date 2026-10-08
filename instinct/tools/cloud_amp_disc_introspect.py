#!/usr/bin/env python3
"""**判别器体检**：读 TB 里的 AMP/损失标量 + 直接解析 checkpoint 里的判别器权重。

目的（V19a 前的一次性核查）：
1. 线上 `Step/amp_d` 到底存不存在（决定 `amp_reward_form` 实际是哪一支）；
2. 判别器损失里各分项的真实占比（`wd` 到底占多少 —— 之前我把**未乘系数的**
   `Loss/discriminator_weight_decay` 直接和 `Loss/discriminator_loss` 比，是错的）；
3. `Σw²`、最后一层 `Σw_logit²`、各层权重范数 ⇒ 用证据选 `discriminator_weight_decay_coef`。

用法（云端，isaaclab python）：
    python cloud_amp_disc_introspect.py --run v18 --ckpt latest
"""

from __future__ import annotations

import argparse
import glob
import os

import torch

LOG_ROOT = "/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk"

#: 与 `wasabi.py::update` 里 `getattr(self, k + "_coef", 1.0)` 逐字对应
#: ⚠️ `logit_weight_decay` 的**真实**系数取到的是 `logit_weight_decay_coef`（不存在）⇒ **1.0**，
#:    而配置里的 `discriminator_logit_weight_decay_coef = 0.04` **根本没被用上**。
COEFS = {
    "discriminator_loss": 1.0,
    "discriminator_gradient_penalty": 5.0,
    "discriminator_weight_decay": 3.0e-4,
    "logit_weight_decay": 1.0,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v18")
    ap.add_argument("--ckpt", default="latest", help="latest | 轮号")
    args = ap.parse_args()

    runs = sorted(glob.glob(os.path.join(LOG_ROOT, f"*{args.run}*")), key=os.path.getmtime)
    if not runs:
        print(f"[X] 没找到 run 匹配 {args.run!r}")
        return 1
    run = runs[-1]
    print(f"[RUN] {run}")

    # --- 1) TB 标量 -------------------------------------------------------------
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        ea = EventAccumulator(run, size_guidance={"scalars": 0})
        ea.Reload()
        tags = sorted(ea.Tags()["scalars"])
        keys = [t for t in tags if ("amp" in t or "discriminator" in t or "logit" in t or "mirror" in t)]
        print(f"\n[TB] 相关标量（{len(keys)} 个）：")
        vals = {}
        for t in keys:
            ev = ea.Scalars(t)
            vals[t] = ev[-1].value
            print(f"  {t:<52} 末值 = {ev[-1].value:+.5f}   (n={len(ev)})")
        print("\n[TB] 判别器损失预算（用 COEFS 还原真实占比）：")
        total = 0.0
        for k, coef in COEFS.items():
            tag = f"Loss/{k}"
            if tag in vals:
                c = coef * vals[tag]
                total += c
                print(f"  {tag:<42} × {coef:<8g} = {c:+.5f}")
        print(f"  {'(以上四项之和)':<42}   {'':<8} = {total:+.5f}"
              f"   [TB Loss/amp_total_loss = {vals.get('Loss/amp_total_loss', float('nan')):+.5f}]")
        if not any(f"Step/{k}" == "Step/amp_d" for k in ()) and "Step/amp_d" not in vals:
            print("  ⚠️ `Step/amp_d` **不存在** ⇒ `amp_reward_form` 走的是 'raw'，"
                  "`discriminator_reward_type='quad'` 的返回值就是风格奖励本身。")
    except Exception as e:
        print(f"[TB] 读取失败：{e!r}")

    # --- 2) 判别器权重 ----------------------------------------------------------
    ckpts = sorted(glob.glob(os.path.join(run, "**", "model_*.pt"), recursive=True))
    if not ckpts:
        ckpts = sorted(glob.glob(os.path.join(run, "*.pt")))
    if not ckpts:
        print("[X] 没找到 checkpoint")
        return 1
    if args.ckpt != "latest":
        want = f"model_{args.ckpt}.pt"
        ck = [c for c in ckpts if c.endswith(want)] or ckpts[-1]
    else:
        ck = max(ckpts, key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0])
                 if os.path.basename(p).split("_")[-1].split(".")[0].isdigit() else 0)
    print(f"\n[CKPT] {ck}")
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    disc = sd.get("discriminator")
    if disc is None:
        print("[X] checkpoint 里没有判别器（被 drop 过？）")
        return 1
    tot_sq, nparam = 0.0, 0
    for k, v in disc.items():
        if hasattr(v, "numel") and v.dtype.is_floating_point:
            tot_sq += float(v.pow(2).sum())
            nparam += v.numel()
    print(f"[CKPT] 判别器参数量 = {nparam}   Σw² = {tot_sq:.3f}   RMS(w) = {(tot_sq/max(nparam,1))**0.5:.4f}")
    print(f"[CKPT] ⇒ `discriminator_weight_decay` 项实际贡献 = 3e-4 × {tot_sq:.1f} = {3e-4*tot_sq:+.5f}")
    # 最后一层（logit layer）
    last = [(k, v) for k, v in disc.items() if hasattr(v, "dim") and v.dim() == 2]
    last.sort(key=lambda kv: kv[1].numel())
    if last:
        k, v = last[-1]
        print(f"[CKPT] 输出层 {k} shape={tuple(v.shape)}  Σw²={float(v.pow(2).sum()):.4f} "
              f"‖w‖∞={float(v.abs().max()):.4f}  RMS={float(v.pow(2).mean()**0.5):.4f}")
    print("[CKPT] 各层权重范数：")
    for k, v in disc.items():
        if hasattr(v, "dim") and v.dim() >= 1:
            print(f"  {k:<34} shape={tuple(v.shape)}  ‖w‖₂={float(v.norm()):.4f}  "
                  f"‖w‖∞={float(v.abs().max()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
