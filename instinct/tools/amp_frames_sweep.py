#!/usr/bin/env python3
"""**离线判别性研究**：扫"判别器输入帧数 h"，看哪种表示对"人形度退化"最敏感、且给出最强排序信号。

背景（为什么要做）
------------------
线上只做过**一次** 2 帧 vs 10 帧的消融（V15/V16，各 1 个种子），而且风格项长期"存在但不说话"
（`amp_style_std ≈ 0.15`）。与其烧两轮 RL，不如**先离线量化**：
"给定 h 帧输入，判别器能多大程度上分辨'参考 vs 轻微退化'，以及它给出的**排序信号**有多强？"

设计要点（用户 2026-10-01 的两条修正意见已吸收）
----------------------------------------------
1. **不用当前策略采样的数据当主判据**（策略本身是碎步 ⇒ 离参考流形太远 ⇒ 任何 h 都能分开，
   结果饱和、答不出"哪个 h 更好"）。主判据 = **参考 + 受控退化**（与策略质量无关）。
2. **扫 h，而不是只比 2/10**（最优值大概率不在端点）。用户要求**不要**加 h≥30（成本）；
   本脚本默认 `h ∈ {1,2,3,4,6,8,10,15,20}` × `stride ∈ {1,2}`。

负样本族（都对应我们真实观察到的病象）
--------------------------------------
| 族 | 做法 | 对应病象 |
|---|---|---|
| N1 关节抖动 | `jpos += N(0,σ)`、`jvel += N(0,σ')` | 高频抖动（`action_rate_l2` 大） |
| N2 幅度压缩 | `jpos *= α`（相对默认位姿的偏差被压小） | 小碎步 |
| N3 时间倍速 | 建窗时按 `k×` 抽帧（同一物理时长更短） | 高频碎步（我们最想治的） |
| N4 局部冻结 | 把关节维的一段连续区间冻结成首帧值 | 单腿拖/单脚跳（近似） |
| N5 相位打乱 | 窗口内帧顺序随机重排 | 非物理转移 |

每个 (h, stride, wd, 族, 强度) 训练一个**与线上同构**的判别器（MLP[1024,512]+ReLU、
`MSELoss`(actor→−1, ref→+1)、AdamW lr 1e-4、wd ∈ {0, 3e-4}），然后报：
  * `AUC`（可分性）——"能不能分开"
  * ⭐ `std(style)` 在"参考 vs 轻微退化"混合批上的值 —— **排序信号**（我们真正缺的）
  * `gap = mean(style_ref) − mean(style_neg)` 与 `style_zero_frac`（quad 死区占比 = 没梯度）
  * 输入维度 D=67h（成本/容量）

用法（本地，CPU 即可）
--------------------
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate instinct
    python instinct/tools/amp_frames_sweep.py --dump /tmp/amp_state_dump.npz \
        --outdir instinct/plots/amp_frames [--quick] [--families N1,N2,N3,N5]

输出：终端表格 + `REPORT.md`（含 h* 建议）+ 三张图（敏感度/排序信号/死区 vs h）。
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

# ---- 67 维状态布局（见 cloud_dump_amp_states.py）--------------------------------
SL = {
    "pg": slice(0, 3),
    "jpos": slice(3, 32),
    "jvel": slice(32, 61),
    "blv": slice(61, 64),
    "bav": slice(64, 67),
}
FRAME_DIM = 67


def build_windows(stream: np.ndarray, h: int, stride: int, max_windows: int, rng: np.random.Generator) -> np.ndarray:
    """把一个 (T, N, 67) 的逐帧序列切成 (M, h*67) 的窗口。

    `stream` 可以是原始序列（正样本），也可以是**抽帧后的序列**（N3 时间倍速负样本）。
    """
    T, N, _ = stream.shape
    span = (h - 1) * stride + 1
    if T < span:
        raise ValueError(f"序列太短：T={T} < span={span}")
    starts = np.arange(0, T - span + 1)
    envs = rng.choice(N, size=max_windows, replace=True)
    t0s = rng.choice(starts, size=max_windows, replace=True)
    idx = t0s[:, None] + stride * np.arange(h)[None, :]        # (M, h)
    out = stream[idx, envs[:, None], :]                        # (M, h, 67)
    return out.reshape(max_windows, h * FRAME_DIM).astype(np.float32)


def perturb(w: np.ndarray, family: str, level: float, rng: np.random.Generator) -> np.ndarray:
    """在**状态空间**对窗口做受控退化（w: (M, h*67)）。"""
    x = w.reshape(w.shape[0], -1, FRAME_DIM).copy()            # (M, h, 67)
    if family == "N1":                                          # 关节抖动
        x[..., SL["jpos"]] += rng.normal(0, level, size=x[..., SL["jpos"]].shape).astype(np.float32)
        x[..., SL["jvel"]] += rng.normal(0, level, size=x[..., SL["jvel"]].shape).astype(np.float32)
    elif family == "N2":                                        # 幅度压缩（jpos/jvel 相对默认的偏差 × lvl）
        x[..., SL["jpos"]] *= level
        x[..., SL["jvel"]] *= level
    elif family == "N4":                                        # 局部冻结：jpos/jvel 各前 15 维冻结成首帧值
        # ⚠️ 近似：参考侧关节顺序取自 npz（未导出名字），故用"连续前半段"代替"单腿"。
        half = 15
        x[:, :, 3 : 3 + half] = x[:, :1, 3 : 3 + half]           # (M,h,half) ← (M,1,half) 广播
        x[:, :, 32 : 32 + half] = x[:, :1, 32 : 32 + half]
    elif family == "N5":                                        # 相位打乱（窗口内帧顺序随机）
        perm = rng.permutation(x.shape[1])
        x = x[:, perm, :]
    else:
        raise ValueError(f"未知扰动族 {family}")
    return x.reshape(w.shape[0], -1)


class Disc:
    """与线上同构的判别器（MLP[1024,512]+ReLU / MSE(±1) / AdamW）。"""

    def __init__(self, dim: int, wd: float, seed: int = 0):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.torch = torch
        self.net = nn.Sequential(
            nn.Linear(dim, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=1e-4, weight_decay=wd)
        self.loss = nn.MSELoss()

    def fit(self, pos, neg, epochs=5, bs=256):
        import torch

        x = np.concatenate([pos, neg]).astype(np.float32)
        y = np.concatenate([np.ones(len(pos)), -np.ones(len(neg))]).astype(np.float32)
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)[:, None]
        n = len(x)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                self.opt.zero_grad()
                out = self.net(xt[idx])
                loss = self.loss(out, yt[idx])
                loss.backward()
                self.opt.step()
        return self

    def logits(self, x) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.net(torch.from_numpy(np.asarray(x, dtype=np.float32))).numpy().ravel()


def style_of(d: np.ndarray) -> np.ndarray:
    """线上 `discriminator_reward_type="quad"`：style = clamp(1 − (d−1)²/4, 0)。"""
    return np.clip(1.0 - 0.25 * (d - 1.0) ** 2, 0.0, None)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC（秩统计，无 sklearn 依赖）。"""
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    r = ranks[: len(pos)].sum()
    n1, n0 = len(pos), len(neg)
    return float((r - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="/tmp/amp_state_dump.npz")
    ap.add_argument("--outdir", default="instinct/plots/amp_frames")
    ap.add_argument("--horizons", default="1,2,3,4,6,8,10,15,20")
    ap.add_argument("--strides", default="1,2")
    ap.add_argument("--wds", default="0,3e-4")
    ap.add_argument("--families", default="N1,N2,N3,N4,N5")
    ap.add_argument("--n-per-class", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="小样本/少 epoch（先跑通链路）")
    ap.add_argument("--forward-only", action="store_true",
                    help="只用参考里 vx>0.3 且 |vy|<0.15 的窗口（与 V11/V16 的'<1 m/s 前向'可比）")
    args = ap.parse_args()
    if args.quick:
        args.n_per_class, args.epochs = 1200, 2

    d = np.load(args.dump, allow_pickle=True)
    ref = d["ref_frame"]                                   # (T, N, 67)
    T, N, D = ref.shape
    print(f"[sweep] dump={args.dump} ref={ref.shape} step_dt={float(d['step_dt']):.3f}s "
          f"clip 分布={dict(zip(*[x.tolist() for x in np.unique(d['clip_id'], return_counts=True)]))}")

    horizons = [int(x) for x in args.horizons.split(",")]
    strides = [int(x) for x in args.strides.split(",")]
    wds = [float(x) for x in args.wds.split(",")]
    families = args.families.split(",")

    # 扰动强度（"轻微/明显"两档；N2 是乘法系数）
    levels = {
        "N1": (0.02, 0.05),
        "N2": (0.50, 0.25),
        "N3": (2, 3),      # 抽帧倍率
        "N4": (1.0, 1.0),  # 只有一种
        "N5": (1.0, 1.0),
    }

    rng = np.random.default_rng(0)
    rows = []
    t_start = time.time()
    for h in horizons:
        for st in strides:
            # 正样本（可选：只取"前向"窗口，用于与 V11/V16 的 <1 m/s 前向类比）
            pos = build_windows(ref, h, st, args.n_per_class, rng)
            if args.forward_only:
                # 用窗口**最后一帧**的 base_lin_vel 判定方向（与 V11/V16 只有 <1 m/s 前向可比）
                m = pos.reshape(-1, h, D)[:, -1, SL["blv"]]
                keep = (m[:, 0] > 0.3) & (np.abs(m[:, 1]) < 0.15)
                if keep.sum() >= 200:
                    pos = pos[keep]
            for fam in families:
                for li, lvl in enumerate(levels[fam]):
                    if fam == "N3":     # 时间倍速：换"抽帧后的序列"建负样本窗口
                        fast = ref[:: int(lvl)]
                        neg = build_windows(fast, h, st, len(pos), rng)
                    else:
                        neg = perturb(pos, fam, float(lvl), rng)
                    for wd in wds:
                        ntr = int(len(pos) * 0.8)
                        clf = Disc(h * D, wd, seed=0).fit(pos[:ntr], neg[:ntr], epochs=args.epochs)
                        d_pos, d_neg = clf.logits(pos[ntr:]), clf.logits(neg[ntr:])
                        a = auc(d_pos, d_neg)
                        # ⭐ 排序信号：在"参考 vs 轻微退化"混合批上的跨样本 style 标准差
                        mix = np.concatenate([d_pos, d_neg])
                        s = style_of(mix)
                        gap = float(style_of(d_pos).mean() - style_of(d_neg).mean())
                        rows.append(
                            dict(h=h, stride=st, wd=wd, fam=fam, level=lvl, auc=a,
                                 style_std=float(s.std()), gap=gap,
                                 zero_frac=float((s <= 0).mean()), dim=h * D,
                                 d_pos_mean=float(d_pos.mean()), d_neg_mean=float(d_neg.mean()))
                        )
                        print(f"  h={h:<3} st={st} wd={wd:<6} {fam}@{lvl}: AUC={a:.3f} "
                              f"style_std={s.std():.3f} gap={gap:+.3f} zero={float((s<=0).mean()):.2f} "
                              f"d+={d_pos.mean():+.2f} d-={d_neg.mean():+.2f}", flush=True)
    print(f"[sweep] {len(rows)} 组，用时 {time.time()-t_start:.0f}s")

    # ---- 汇总：每个 h 的"轻微档"平均表现（排序信号为主指标）----
    os.makedirs(args.outdir, exist_ok=True)
    import csv
    with open(os.path.join(args.outdir, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(h, wd, lvl_idx=0):
        """取该 h/wd 下、各族**同一档位**（0=轻微,1=明显）的行。"""
        return [r for r in rows if r["h"] == h and r["wd"] == wd
                and r["level"] == levels[r["fam"]][lvl_idx]]

    lines = ["# 离线判别性研究：帧数 h 扫描\n",
             f"- 数据：`{args.dump}`，参考序列 {ref.shape}，step_dt={float(d['step_dt']):.3f}s",
             f"- 每个 (h,stride,wd,族,档) 训练一个与线上同构的判别器；主指标 = **混合批上 style 的跨样本 std**（排序信号）\n",
             "| h | 输入维 | AUC(轻微) | **style_std(轻微)** | gap | zero_frac | 明显档 AUC |",
             "|---|---|---|---|---|---|---|"]
    summary = []
    for wd in wds:
        for h in horizons:
            mild = agg(h, wd, 0)
            hard = agg(h, wd, 1)
            if not mild:
                continue
            auc_m = float(np.mean([r["auc"] for r in mild]))
            std_m = float(np.mean([r["style_std"] for r in mild]))
            gap_m = float(np.mean([r["gap"] for r in mild]))
            zf_m = float(np.mean([r["zero_frac"] for r in mild]))
            auc_h = float(np.mean([r["auc"] for r in hard])) if hard else float("nan")
            summary.append((wd, h, auc_m, std_m, gap_m, zf_m, auc_h))
            lines.append(f"| {h} | {67*h} | {auc_m:.3f} | **{std_m:.3f}** | {gap_m:+.3f} | {zf_m:.2f} | {auc_h:.3f} |")
    lines.append("\n## 结论（自动）\n")
    for wd in wds:
        cand = [s for s in summary if s[0] == wd]
        if not cand:
            continue
        best = max(cand, key=lambda s: s[3])
        lines.append(f"- wd={wd}：**排序信号最强的是 h={best[1]}**（style_std={best[3]:.3f}，"
                     f"轻微档 AUC={best[2]:.3f}，死区占比={best[5]:.2f}）。")
    lines.append("\n读法：`style_std` 高且 `zero_frac` 低 ⇒ 该 h 在参考流形附近**既敏感又有排序梯度**；"
                 "AUC 高但 `style_std` 低 ⇒ 只能分开远的东西，对'轻微不像人'没有引导力（正是线上 2/10 帧的毛病）。")
    with open(os.path.join(args.outdir, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- 画图（可选）----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for wd in wds:
            for fam in families:
                sel = [r for r in rows if r["wd"] == wd and r["fam"] == fam
                       and r["level"] == levels[fam][0]]
                if not sel:
                    continue
                xs = sorted({r["h"] for r in sel})
                axes[0].plot(xs, [np.mean([r["auc"] for r in sel if r["h"] == x]) for x in xs],
                             marker="o", label=f"{fam} wd={wd}")
                axes[1].plot(xs, [np.mean([r["style_std"] for r in sel if r["h"] == x]) for x in xs],
                             marker="o", label=f"{fam} wd={wd}")
                axes[2].plot(xs, [np.mean([r["zero_frac"] for r in sel if r["h"] == x]) for x in xs],
                             marker="o", label=f"{fam} wd={wd}")
        for ax, t in zip(axes, ["AUC (mild) vs h", "* ranking signal: std(style) vs h", "quad dead-zone frac vs h"]):
            ax.set_xlabel("h (# frames fed to discriminator)"); ax.set_title(t); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "sweep.png"), dpi=130)
        print(f"[sweep] 图 → {args.outdir}/sweep.png")
    except Exception as e:  # pragma: no cover
        print(f"[sweep] 画图跳过：{e!r}")

    print(f"[sweep] 报告 → {args.outdir}/REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
