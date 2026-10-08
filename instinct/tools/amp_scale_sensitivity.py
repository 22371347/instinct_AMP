#!/usr/bin/env python3
"""**判别器输入尺度敏感性探针**：验证"没白化 ⇒ 判别器只认大尺度维度"这一假说。

背景（用户 2026-10-03 的问题）
------------------------------
我们刚体 195 维的 4 个项量纲差 1~2 个数量级：

| 项 | 每体 | 典型尺度 |
|---|---|---|
| `body_pos_b` | 3 | ~0.3 m |
| `body_ori_b` | 6 | 6D 旋转矩阵前两列 ∈ [−1, 1] |
| `body_lin_vel_b` | 3 | ~0.1~2 m/s |
| `body_ang_vel_b` | 3 | ~0.5~5 rad/s |

而 AMP_mjlab **有** `amp_normalizer`（对判别器输入做 running mean-std 白化），我们**没有**。
假说：不白化 ⇒ 判别器的决策边界被大尺度分量（速度类）主导，**最想让它看的位姿类被淹没**
⇒ 风格奖励变成"别动/别抖"而不是"像人一样动"（V20 的 `hop_frac 0.92` 与此一致）。

本探针量三件事
--------------
1. **逐块敏感度**：`∂d/∂x` 的逐维 RMS，按 4 个项（pos/ori/lv/av）汇总成占比 —— 与它们在
   195 维里的**面积占比**（36/78/39/39 = 18.5%/40%/20%/20%）对比；
2. **白化前后的判别力**：同一协议（参考 + 时间结构类轻微退化）下的 `AUC` 与 `style_std`；
3. **白化前后的敏感度再平衡**：看 pos/ori 的占比是否回到面积占比附近。

白化实现（对齐 mjlab 的口径）
----------------------------
`x_norm = (x − μ) / (σ + eps)`，**μ/σ 只用参考（expert）训练split统计**、且**两侧用同一组** ——
这正是 AMP normalizer 的语义（它累积 expert 统计量，对 policy/expert 施加同一变换）。
⚠️ 注意政策侧的输入不会被"重新标准化"，尺度错配依旧存在 —— 那正是判据的一部分。

用法（本地，CPU）：
    python instinct/tools/amp_scale_sensitivity.py \
        --dump instinct/eval/amp_state_dump_body.npz \
        --outdir instinct/plots/amp_scale_sensitivity
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

PER_BODY_DIM = {"pos": 3, "ori": 6, "lv": 3, "av": 3}
LEVELS = {"N3": (2, 3), "N4": (1.0, 1.0), "N5": (1.0, 1.0)}      # 只取时间结构类（轻微档 = idx 0）


def blocks_of(n_bodies: int) -> dict[str, slice]:
    blocks, off = {}, 0
    for k, p in PER_BODY_DIM.items():
        blocks[k] = slice(off, off + p * n_bodies)
        off += p * n_bodies
    return blocks


def build_windows(stream, h, stride, n, rng, dim):
    T, N, D = stream.shape
    span = (h - 1) * stride + 1
    starts = np.arange(0, T - span + 1)
    envs = rng.choice(N, size=n, replace=True)
    t0s = rng.choice(starts, size=n, replace=True)
    idx = t0s[:, None] + stride * np.arange(h)[None, :]
    return stream[idx, envs[:, None], :].reshape(n, h * dim).astype(np.float32)


def perturb(w, family, level, rng, blocks, dim, n_bodies, leg_bodies):
    x = w.reshape(w.shape[0], -1, dim).copy()
    if family == "N4":      # 冻结 6 条腿刚体（hip/knee/ankle）的全部 15 维，与 sweep v2 的 N4 一致
        freeze = []
        for k, p in PER_BODY_DIM.items():
            for b in leg_bodies:
                lo = blocks[k].start + b * p
                freeze.extend(range(lo, lo + p))
        freeze = np.array(sorted(freeze))
        x[:, :, freeze] = x[:, :1, freeze]
    elif family == "N5":
        x = x[:, rng.permutation(x.shape[1]), :]
    else:
        raise ValueError(family)
    return x.reshape(w.shape[0], -1)


class Disc:
    def __init__(self, dim, seed=0):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.torch = torch
        self.net = nn.Sequential(nn.Linear(dim, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=1e-4, weight_decay=0.0)
        self.loss = nn.MSELoss()

    def fit(self, pos, neg, epochs=5, bs=256):
        import torch

        x = np.concatenate([pos, neg]).astype(np.float32)
        y = np.concatenate([np.ones(len(pos)), -np.ones(len(neg))]).astype(np.float32)
        xt, yt = torch.from_numpy(x), torch.from_numpy(y)[:, None]
        for _ in range(epochs):
            perm = torch.randperm(len(x))
            for i in range(0, len(x), bs):
                idx = perm[i : i + bs]
                self.opt.zero_grad()
                self.loss(self.net(xt[idx]), yt[idx]).backward()
                self.opt.step()
        return self

    def logits(self, x) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.net(torch.from_numpy(np.asarray(x, dtype=np.float32))).numpy().ravel()

    def input_grad_rms(self, x, bs=512) -> np.ndarray:
        """逐维 RMS(∂d/∂x)（对样本取均方根）。"""
        import torch

        xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
        acc = torch.zeros(xt.shape[1])
        n = 0
        for i in range(0, len(xt), bs):
            xb = xt[i : i + bs].clone().requires_grad_(True)
            out = self.net(xb).sum()
            g = torch.autograd.grad(out, xb)[0]
            acc += (g**2).sum(dim=0)
            n += len(xb)
        return (acc / max(n, 1)).sqrt().numpy()


def auc(pos, neg) -> float:
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    r = ranks[: len(pos)].sum()
    n1, n0 = len(pos), len(neg)
    return float((r - n1 * (n1 + 1) / 2) / (n1 * n0))


def style_of(d, k=0.125):
    lo = 1.0 - np.sqrt(1.0 / k)
    return np.clip(1.0 - k * (d - 1.0) ** 2, lo, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="instinct/eval/amp_state_dump_body.npz")
    ap.add_argument("--outdir", default="instinct/plots/amp_scale_sensitivity")
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--n-per-class", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    d = np.load(args.dump, allow_pickle=True)
    ref = d["ref_frame"]
    names = [str(x) for x in d["body_names"]]
    n_bodies = len(names)
    dim = ref.shape[-1]
    blocks = blocks_of(n_bodies)
    leg_bodies = [i for i, n in enumerate(names) if any(k in n for k in ("hip", "knee", "ankle"))]
    h, st = args.h, args.stride
    rng = np.random.default_rng(0)
    print(f"[probe] N4 冻结刚体 = {[names[i] for i in leg_bodies]}")

    print(f"[probe] dump={args.dump} ref={ref.shape} h={h} stride={st} "
          f"判别器输入维 = {h * dim}")
    # 单帧分量尺度
    for k, sl in blocks.items():
        blk = ref[..., sl]
        print(f"  单帧 {k:<4} std={blk.std():.3f}  p1={np.percentile(blk, 1):+.3f} "
              f"p99={np.percentile(blk, 99):+.3f}  面积占比={blk.shape[-1] / dim:.1%}")

    pos = build_windows(ref, h, st, args.n_per_class, rng, dim)
    ntr = int(0.8 * len(pos))
    # 白化参数：只用参考训练split（= expert 统计量，对齐 amp_normalizer 语义）
    mu = pos[:ntr].mean(axis=0)
    sd = pos[:ntr].std(axis=0) + 1e-6

    results = {}
    for tag, xform in (("raw", lambda x: x), ("whitened", lambda x: (x - mu) / sd)):
        pos_w = xform(pos)
        per_fam: dict[str, list[float]] = {}
        grads = []
        styles = []
        for fam, levels in LEVELS.items():
            lvl = levels[0]
            if fam == "N3":
                neg = build_windows(ref[:: int(lvl)], h, st, len(pos), rng, dim)
            else:
                neg = perturb(pos, fam, float(lvl), rng, blocks, dim, n_bodies, leg_bodies)
            neg_w = xform(neg)
            clf = Disc(h * dim, seed=0).fit(pos_w[:ntr], neg_w[:ntr], epochs=args.epochs)
            d_pos, d_neg = clf.logits(pos_w[ntr:]), clf.logits(neg_w[ntr:])
            a = auc(d_pos, d_neg)
            mix = np.concatenate([d_pos, d_neg])
            s = style_of(mix)
            per_fam[fam] = a
            styles.append(float(s.std()))
            grads.append(clf.input_grad_rms(pos_w[ntr:]))
            print(f"  [{tag}] {fam}@{lvl}: AUC={a:.3f} style_std={s.std():.4f}")
        # 逐块敏感度占比（对窗口取均值：同一维在所有帧上的 RMS 合并）
        g = np.mean(grads, axis=0).reshape(h, dim)          # (h, dim) —— 注意 term-major 帧序
        shares = {}
        for k, sl in blocks.items():
            # 每一项在每一帧里都占同样位置：按帧切片后取该块
            idx = np.arange(sl.start, sl.stop)
            shares[k] = float((g[:, idx] ** 2).sum())
        tot = sum(shares.values())
        shares = {k: v / tot for k, v in shares.items()}
        results[tag] = dict(auc=per_fam, style_std=float(np.mean(styles)), grad_share=shares)
        area_str = " ".join(f"{k}={(blocks[k].stop - blocks[k].start) / dim:.1%}" for k in blocks)
        print(f"  [{tag}] 敏感度占比 " + " ".join(f"{k}={v:.1%}" for k, v in shares.items())
              + "   (面积占比 " + area_str + ")")

    # ---- 汇总 ----
    area = {k: (blocks[k].stop - blocks[k].start) / dim for k in blocks}
    lines = ["# 判别器输入尺度敏感性探针（刚体 195 维）\n",
             f"- 数据：`{args.dump}`，h={h}，stride={st}，判别器输入维 {h * dim}",
             "- 白化 = `(x − μ_ref)/σ_ref`，μ/σ **只用参考训练split统计**、两侧同一变换（对齐 `amp_normalizer`）\n",
             "## 敏感度占比 vs 面积占比\n",
             "| 项 | 面积占比 | **raw** 敏感度占比 | **whitened** 敏感度占比 |",
             "|---|---|---|---|"]
    for k in blocks:
        lines.append(f"| {k} | {area[k]:.1%} | **{results['raw']['grad_share'][k]:.1%}** | "
                     f"{results['whitened']['grad_share'][k]:.1%} |")
    lines += ["\n## 判别力与排序信号（时间结构类三族平均，轻微档）\n",
              "| | raw | whitened |", "|---|---|---|",
              f"| AUC | {np.mean(list(results['raw']['auc'].values())):.3f} | "
              f"{np.mean(list(results['whitened']['auc'].values())):.3f} |",
              f"| style_std | {results['raw']['style_std']:.4f} | {results['whitened']['style_std']:.4f} |"]
    lines += ["\n## 各族 AUC\n", "| 族 | raw | whitened |", "|---|---|---|"]
    for fam in LEVELS:
        lines.append(f"| {fam} | {results['raw']['auc'][fam]:.3f} | {results['whitened']['auc'][fam]:.3f} |")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.outdir, "result.json"), "w") as f:
        json.dump({"area": area, **results}, f, indent=2)
    print(f"[probe] 报告 → {os.path.join(args.outdir, 'REPORT.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
