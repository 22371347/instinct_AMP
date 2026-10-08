#!/usr/bin/env python3
"""**白化 × h 联合扫描**：回答"开白化后，帧数 h 能不能降下来"，并顺带量出 GP 需要的新标定。

复用 `amp_scale_sensitivity.py` 的构件（同一判别器/同一扰动/同一指标），只把"h"变成扫描维度，
并对每种 (h, 白化) 组合记录：

* `AUC`：时间结构类三族（N3 时间倍速2× / N4 局部冻结 / N5 相位打乱）轻微档的平均可分性
* `style_std`：排序信号宽度
* **`‖∇d‖`（网络输入空间）与 `‖∇d‖`（原始空间）** —— 后者 = `‖∇_{x̃}d / σ‖`
  ⇒ 这两个数**直接决定 GP 系数该怎么重标定**（见 `RESULTS_amp_vs_mjlab_params.md` §E/§F）

为什么 GP 会被牵连
------------------
白化把判别器的**前向输入**换成 `x̃=(x−μ)/σ`。若 GP 仍在**原始空间**算 `‖∂d/∂x‖`，
由于 `∂d/∂x = (∂d/∂x̃)/σ`，惩罚会被 `1/σ` 加权 ⇒ **偏向小尺度维**（与白化想修的偏差反向）。
mjlab 恰好就是这个写法（`disc = self.amp_linear(self.trunk(expert_data))` 用的是**原始** expert，
主损失用归一化状态）⇒ 它的 GP 约束的是"另一个函数"。本脚本把两个空间的 `‖∇d‖` 都量出来，
供我们**有依据地**选择"GP 在哪个空间算 + coef 取多少"。

用法（本地，CPU）：
    python instinct/tools/amp_whiten_h_scan.py --h 1,2,3,4,6,8,10
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from amp_scale_sensitivity import (  # 同一目录，直接复用（无副作用）
    PER_BODY_DIM,
    Disc,
    auc,
    blocks_of,
    build_windows,
    perturb,
    style_of,
)

LEVELS = {"N3": 2, "N4": 1.0, "N5": 1.0}      # 只取"轻微档"


def run_one(h: int, stride: int, xform, pos, ref, rng, blocks, dim, n_bodies, leg_bodies,
            n_train: int, epochs: int):
    """训练一个判别器，返回 (平均AUC, 平均style_std, 网络输入空间逐维梯度RMS)。"""
    pos_w = xform(pos)
    aucs, stds, grads = [], [], []
    for fam, lvl in LEVELS.items():
        if fam == "N3":
            neg = build_windows(ref[:: int(lvl)], h, stride, len(pos), rng, dim)
        else:
            neg = perturb(pos, fam, float(lvl), rng, blocks, dim, n_bodies, leg_bodies)
        neg_w = xform(neg)
        clf = Disc(h * dim, seed=0).fit(pos_w[:n_train], neg_w[:n_train], epochs=epochs)
        d_pos, d_neg = clf.logits(pos_w[n_train:]), clf.logits(neg_w[n_train:])
        aucs.append(auc(d_pos, d_neg))
        stds.append(float(style_of(np.concatenate([d_pos, d_neg])).std()))
        grads.append(clf.input_grad_rms(pos_w[n_train:]))      # 网络输入空间
        del clf
    return float(np.mean(aucs)), float(np.mean(stds)), np.mean(grads, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="instinct/eval/amp_state_dump_body.npz")
    ap.add_argument("--outdir", default="instinct/plots/amp_whiten_h_scan")
    ap.add_argument("--h", default="1,2,3,4,6,8,10,15")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--n-per-class", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    hs = [int(x) for x in args.h.split(",")]
    d = np.load(args.dump, allow_pickle=True)
    ref = d["ref_frame"]
    names = [str(x) for x in d["body_names"]]
    n_bodies, dim = len(names), ref.shape[-1]
    blocks = blocks_of(n_bodies)
    leg_bodies = [i for i, n in enumerate(names) if any(k in n for k in ("hip", "knee", "ankle"))]
    rng = np.random.default_rng(0)
    st = args.stride
    print(f"[scan] dump={args.dump} ref={ref.shape} stride={st} horizons={hs}")
    del names

    rows = []
    t0 = time.time()
    for h in hs:
        rng = np.random.default_rng(0)          # 每个 h 用同一种子，保证 raw/wh 可比
        pos = build_windows(ref, h, st, args.n_per_class, rng, dim)
        n_train = int(0.8 * len(pos))
        mu = pos[:n_train].mean(axis=0)
        sd = pos[:n_train].std(axis=0) + 1e-6
        # ---- raw ----
        rng = np.random.default_rng(0)
        auc_r, std_r, g_r = run_one(h, st, lambda x: x, pos, ref, rng, blocks, dim,
                                    n_bodies, leg_bodies, n_train, args.epochs)
        # ---- whitened ----
        rng = np.random.default_rng(0)
        auc_w, std_w, g_w = run_one(h, st, lambda x: (x - mu) / sd, pos, ref, rng, blocks, dim,
                                    n_bodies, leg_bodies, n_train, args.epochs)
        # ‖∇d‖：网络输入空间 vs 原始空间（后者 = ‖g/σ‖）
        gnorm_norm = float(np.sqrt((g_w**2).mean()))
        gnorm_raw = float(np.sqrt((((g_w / sd)) ** 2).mean()))
        gnorm_raw_raw = float(np.sqrt((g_r**2).mean()))     # 未白化判别器在原始空间的 ‖∇d‖
        rows.append(dict(h=h, dim=h * dim,
                         auc_raw=auc_r, auc_wh=auc_w,
                         style_std_raw=std_r, style_std_wh=std_w,
                         gnorm_net_wh=gnorm_norm, gnorm_raw_wh=gnorm_raw,
                         gnorm_raw_raw=gnorm_raw_raw))
        print(f"  h={h:<3} AUС raw={auc_r:.3f} → wh={auc_w:.3f} | style_std {std_r:.4f} → {std_w:.4f} "
              f"| ‖∇d‖ 原始空间: raw判别器={gnorm_raw_raw:.4f} / 白化判别器={gnorm_raw:.4f}"
              f"  网络空间(白化)={gnorm_norm:.4f}", flush=True)
    print(f"[scan] {len(rows)} 组，用时 {time.time() - t0:.0f}s")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "scan.json"), "w") as f:
        json.dump(rows, f, indent=2)

    def need_h(key: str, target: float) -> str:
        for r in rows:
            if r[key] >= target:
                return str(r["h"])
        return f">{max(hs)}"

    lines = ["# 白化 × 帧数 h 联合扫描（刚体 195 维，时间结构类三族轻微档）\n",
             f"- 数据：`{args.dump}`，stride={st}，判别器同构（MLP[1024,512]+ReLU / MSE(±1) / AdamW 1e-4）",
             f"- 每格 = 三族（N3 时间倍速2× / N4 局部冻结(6腿) / N5 相位打乱）平均",
             "- 白化 = `(x−μ_ref)/σ_ref`（只用参考训练split统计，两侧同一变换）\n",
             "| h | 输入维 | AUC raw | **AUC 白化** | style_std raw | **style_std 白化** | ‖∇d‖(原始空间) raw判别器 | ‖∇d‖(原始空间) 白化判别器 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['h']} | {r['dim']} | {r['auc_raw']:.3f} | **{r['auc_wh']:.3f}** | "
                     f"{r['style_std_raw']:.4f} | **{r['style_std_wh']:.4f}** | {r['gnorm_raw_raw']:.4f} | "
                     f"{r['gnorm_raw_wh']:.4f} |")
    lines += ["\n## 达到目标 AUC 所需的 h\n", "| 目标 AUC | raw | **白化** |", "|---|---|---|"]
    for t in (0.70, 0.80, 0.85, 0.90):
        lines.append(f"| {t:.2f} | {need_h('auc_raw', t)} | **{need_h('auc_wh', t)}** |")
    lines += ["\n## GP 标定提示\n",
              "`‖∇d‖`(原始空间) 一列可比出\"白化前后判别器在**原始输入度量**下的陡峭度\"：",
              "- 若白化判别器的该值显著小于 raw 判别器 ⇒ 同样 `coef=1.0` 的 GP 惩罚会变弱 ⇒ 需按比例上调；",
              "- 若我们在**归一化空间**算 GP，则应该用 `‖∇d‖(网络空间)` 那一列的尺度来定 `coef`。"]
    with open(os.path.join(args.outdir, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[scan] 报告 → {os.path.join(args.outdir, 'REPORT.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
