#!/usr/bin/env python3
"""**离线判别性研究 v2**：把"帧数 h 扫描"做成**布局感知**的，从而回答用户 2026-10-02 的问题：

    "刚体空间 195 维（13 体 × pos3+ori6+lin3+ang3）需要几帧，才能达到
     关节空间 67 维在 h=8~10 的判别水平？"

与 v1（`amp_frames_sweep.py`）的关系
-----------------------------------
* **协议逐字相同**：扰动族 N1~N5、强度两档、"参考 + 受控退化"为主判据、判别器与线上同构
  （MLP[1024,512]+ReLU、MSE(±1)、AdamW lr=1e-4、wd∈{0,3e-4}）、指标
  AUC / **style_std（排序信号，主指标）** / gap / zero_frac。
* **唯一新增**：布局定义（`legacy67` = `amp_state_dump.npz`；`body195` =
  `amp_state_dump_body.npz`），以及按布局翻译的扰动实现。

两种布局的扰动对照（"同类退化、不同坐标系"）
------------------------------------------
| 族 | legacy67（关节空间） | body195（刚体空间） |
|---|---|---|
| N1 抖动 | `jpos += N(0,σ)`, `jvel += N(0,σ)` | `pos/ori/lv/av += N(0,σ)` |
| N2 幅度压缩 | `jpos × α`, `jvel × α`（→ 向默认位姿收缩） | **整帧 × α**（→ 向 0 收缩；对 ori 不保证是合法姿态，只作"静态退化"探针） |
| N3 时间倍速 | 建窗时按 k× 抽帧（布局无关） | 同左 |
| N4 局部冻结 | `jpos/jvel` 各前 15 维冻结成首帧 | **6 条腿刚体的全部 15 维**冻结成首帧（更贴合"单腿拖/单脚跳"） |
| N5 相位打乱 | 窗口内帧顺序随机重排（布局无关） | 同左 |

用法（本地，CPU 即可）
--------------------
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate instinct
    # 刚体 195 维
    python instinct/tools/amp_frames_sweep_v2.py \
        --dump instinct/eval/amp_state_dump_body.npz --tag body195 \
        --outdir instinct/plots/amp_frames_body
    # 关节 67 维（同协议复跑，便于并列）
    python instinct/tools/amp_frames_sweep_v2.py \
        --dump instinct/eval/amp_state_dump.npz --tag legacy67 \
        --outdir instinct/plots/amp_frames_legacy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

# ===========================================================================
# 布局定义
# ===========================================================================


class Layout:
    """一套 AMP 状态布局：维度 + 各"语义块"的切片 + 扰动实现所需的索引。"""

    def __init__(self, name: str, dim: int, blocks: dict[str, slice], freeze: np.ndarray, desc: str):
        self.name = name
        self.dim = dim
        self.blocks = blocks                    # 语义名 → 该块在单帧内的切片
        self.freeze = freeze                    # N4 冻结用的维度索引（一维 int 数组）
        self.desc = desc


def layout_legacy67() -> Layout:
    """关节空间 5 项（V1~V19d 全谱系的判别器输入）。"""
    blocks = {
        "pg": slice(0, 3),
        "jpos": slice(3, 32),
        "jvel": slice(32, 61),
        "blv": slice(61, 64),
        "bav": slice(64, 67),
    }
    # N4：jpos / jvel 各前 15 维（与 v1 逐字相同）
    freeze = np.concatenate([np.arange(3, 3 + 15), np.arange(32, 32 + 15)])
    return Layout("legacy67", 67, blocks, freeze, "projected_gravity|joint_pos_rel|joint_vel(0.05)|base_lin_vel|base_ang_vel")


def layout_body195(body_names: tuple[str, ...]) -> Layout:
    """刚体空间 4 项（13 体 × 15）。term-major、项内逐刚体连续。"""
    nb = len(body_names)
    per = {"pos": 3, "ori": 6, "lv": 3, "av": 3}
    blocks, off = {}, 0
    for key, p in per.items():
        blocks[key] = slice(off, off + p * nb)
        off += p * nb
    # N4：6 条腿刚体（名字里含 hip/knee/ankle）的全部 15 维
    leg_body_idx = [i for i, n in enumerate(body_names) if any(k in n for k in ("hip", "knee", "ankle"))]
    idx = []
    for key in ("pos", "ori", "lv", "av"):
        p = per[key]
        for b in leg_body_idx:
            lo = blocks[key].start + b * p
            idx.extend(range(lo, lo + p))
    return Layout("body195", 15 * nb, blocks, np.array(sorted(idx), dtype=int),
                  "body_pos_b|body_ori_b|body_lin_vel_b|body_ang_vel_b")


def detect_layout(d: np.lib.npyio.NpzFile) -> Layout:
    names = tuple(str(x) for x in d["body_names"]) if "body_names" in d.files else ()
    dim = int(d["frame_dim"]) if "frame_dim" in d.files else int(d["ref_frame"].shape[-1])
    if "layout" in d.files and str(d["layout"]) == "body" and names:
        return layout_body195(names)
    if dim == 195 and names:
        return layout_body195(names)
    if dim != 67:
        raise ValueError(f"无法识别布局：dim={dim}，body_names={names}")
    return layout_legacy67()


# ===========================================================================
# 建窗 / 扰动 / 判别器 / 指标（与 v1 同构）
# ===========================================================================


def build_windows(stream: np.ndarray, h: int, stride: int, n: int, rng: np.random.Generator, dim: int):
    """→ (windows (n, h*dim), envs (n,), t0s (n,))，索引同时返回供 `--forward-only` 分档用。"""
    T, N, D = stream.shape
    assert D == dim
    span = (h - 1) * stride + 1
    if T < span:
        raise ValueError(f"序列太短：T={T} < span={span}")
    starts = np.arange(0, T - span + 1)
    envs = rng.choice(N, size=n, replace=True)
    t0s = rng.choice(starts, size=n, replace=True)
    idx = t0s[:, None] + stride * np.arange(h)[None, :]
    out = stream[idx, envs[:, None], :]
    return out.reshape(n, h * dim).astype(np.float32), envs, t0s


def perturb(w: np.ndarray, lay: Layout, family: str, level: float, rng: np.random.Generator) -> np.ndarray:
    x = w.reshape(w.shape[0], -1, lay.dim).copy()          # (M, h, D)
    if family == "N1":                                     # 抖动
        if lay.name == "legacy67":
            for k in ("jpos", "jvel"):
                x[..., lay.blocks[k]] += rng.normal(0, level, size=x[..., lay.blocks[k]].shape).astype(np.float32)
        else:
            for k in ("pos", "ori", "lv", "av"):
                x[..., lay.blocks[k]] += rng.normal(0, level, size=x[..., lay.blocks[k]].shape).astype(np.float32)
    elif family == "N2":                                   # 幅度压缩
        if lay.name == "legacy67":
            for k in ("jpos", "jvel"):
                x[..., lay.blocks[k]] *= level
        else:
            x *= level
    elif family == "N4":                                   # 局部冻结
        x[:, :, lay.freeze] = x[:, :1, lay.freeze]
    elif family == "N5":                                   # 相位打乱
        x = x[:, rng.permutation(x.shape[1]), :]
    else:
        raise ValueError(f"未知扰动族 {family}")
    return x.reshape(w.shape[0], -1)


class Disc:
    """与线上同构的判别器。"""

    def __init__(self, dim: int, wd: float, seed: int = 0):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.torch = torch
        self.net = nn.Sequential(nn.Linear(dim, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=1e-4, weight_decay=wd)
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
                loss = self.loss(self.net(xt[idx]), yt[idx])
                loss.backward()
                self.opt.step()
        return self

    def logits(self, x) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.net(torch.from_numpy(np.asarray(x, dtype=np.float32))).numpy().ravel()


def style_of(d: np.ndarray, k: float = 0.125) -> np.ndarray:
    """线上风格奖励：`clamp(1 − k(d−1)², min=1−sqrt(1/k))`（k 默认 0.125 = 当前线上值）。"""
    lo = 1.0 - np.sqrt(1.0 / k)
    return np.clip(1.0 - k * (d - 1.0) ** 2, lo, None)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    r = ranks[: len(pos)].sum()
    n1, n0 = len(pos), len(neg)
    return float((r - n1 * (n1 + 1) / 2) / (n1 * n0))


# ===========================================================================
# 主流程
# ===========================================================================

LEVELS = {"N1": (0.02, 0.05), "N2": (0.50, 0.25), "N3": (2, 3), "N4": (1.0, 1.0), "N5": (1.0, 1.0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tag", default="auto")
    ap.add_argument("--outdir", default="instinct/plots/amp_frames_v2")
    ap.add_argument("--horizons", default="1,2,3,4,6,8,10,15,20")
    ap.add_argument("--strides", default="1,2")
    ap.add_argument("--wds", default="0,3e-4")
    ap.add_argument("--families", default="N1,N2,N3,N4,N5")
    ap.add_argument("--n-per-class", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--quad-k", type=float, default=0.125)
    ap.add_argument("--quick", action="store_true", help="小样本/少 epoch（先跑通链路）")
    ap.add_argument("--forward-only", action="store_true", help="只用参考里 vx>0.3 且 |vy|<0.15 的窗口")
    args = ap.parse_args()
    if args.quick:
        args.n_per_class, args.epochs = 1200, 2

    d = np.load(args.dump, allow_pickle=True)
    lay = detect_layout(d)
    ref = d["ref_frame"]
    T, N, D = ref.shape
    tag = args.tag if args.tag != "auto" else lay.name
    print(f"[sweep-v2] dump={args.dump} tag={tag} 布局={lay.name} dim={lay.dim} ref={ref.shape} "
          f"step_dt={float(d['step_dt']):.3f}s")
    print(f"[sweep-v2] {lay.desc}")
    if lay.name == "body195":
        print(f"[sweep-v2] 13 体 = {list(d['body_names'])}；N4 冻结维度数 = {len(lay.freeze)}（6 条腿）")
    # 前向分档用的基座线速度
    if "ref_base_lin_vel_b" in d.files:
        blv = d["ref_base_lin_vel_b"]
    else:
        blv = ref[..., lay.blocks["blv"]]

    horizons = [int(x) for x in args.horizons.split(",")]
    strides = [int(x) for x in args.strides.split(",")]
    wds = [float(x) for x in args.wds.split(",")]
    families = args.families.split(",")

    rng = np.random.default_rng(0)
    rows: list[dict] = []
    t0 = time.time()
    for h in horizons:
        for st in strides:
            pos, envs, t0s = build_windows(ref, h, st, args.n_per_class, rng, lay.dim)
            if args.forward_only:
                # 用**窗口末帧**的参考基座线速度判方向（与 V11/V16 的"只有 <1 m/s 前向"类比）；
                # 两种布局统一用 `ref_base_lin_vel_b`（关节布局没有这一项时回退到 blv 切片）
                m = blv[t0s + (h - 1) * st, envs]
                keep = (m[:, 0] > 0.3) & (np.abs(m[:, 1]) < 0.15)
                if keep.sum() >= 200:
                    pos, envs, t0s = pos[keep], envs[keep], t0s[keep]
            for fam in families:
                for lvl in LEVELS[fam]:
                    if fam == "N3":
                        fast = ref[:: int(lvl)]
                        neg, _, _ = build_windows(fast, h, st, len(pos), rng, lay.dim)
                    else:
                        neg = perturb(pos, lay, fam, float(lvl), rng)
                    for wd in wds:
                        ntr = int(len(pos) * 0.8)
                        clf = Disc(h * lay.dim, wd, seed=0).fit(pos[:ntr], neg[:ntr], epochs=args.epochs)
                        d_pos, d_neg = clf.logits(pos[ntr:]), clf.logits(neg[ntr:])
                        a = auc(d_pos, d_neg)
                        mix = np.concatenate([d_pos, d_neg])
                        s = style_of(mix, args.quad_k)
                        gap = float(style_of(d_pos, args.quad_k).mean() - style_of(d_neg, args.quad_k).mean())
                        rows.append(dict(
                            layout=lay.name, h=h, stride=st, wd=wd, fam=fam, level=lvl, auc=a,
                            style_std=float(s.std()), gap=gap, zero_frac=float((s <= 0).mean()),
                            dim=h * lay.dim, d_pos_mean=float(d_pos.mean()), d_neg_mean=float(d_neg.mean()),
                        ))
                        print(f"  [{tag}] h={h:<3} st={st} wd={wd:<6} {fam}@{lvl}: AUC={a:.3f} "
                              f"style_std={s.std():.3f} gap={gap:+.3f} zero={float((s <= 0).mean()):.2f}", flush=True)
    print(f"[sweep-v2] {len(rows)} 组，用时 {time.time() - t0:.0f}s")

    # ---- 输出 ----
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.outdir, "sweep.json"), "w") as f:
        json.dump(rows, f)

    def agg(h, wd, lvl_idx=0):
        return [r for r in rows if r["h"] == h and r["wd"] == wd and r["level"] == LEVELS[r["fam"]][lvl_idx]]

    lines = [f"# 离线判别性研究 v2：布局 `{lay.name}`（{lay.dim} 维/帧）的帧数 h 扫描\n",
             f"- 数据：`{args.dump}`，参考序列 `{ref.shape}`，step_dt={float(d['step_dt']):.3f}s",
             f"- 布局：{lay.desc}",
             f"- 协议与 v1 逐字相同（N1~N5、两档强度、MLP[1024,512]/MSE(±1)/AdamW，主指标 = 混合批 style 跨样本 std）",
             f"- 风格函数：`k={args.quad_k:g}`\n",
             "| h | 输入维 | AUC(轻微) | **style_std(轻微)** | gap | zero_frac | 明显档 AUC |",
             "|---|---|---|---|---|---|---|"]
    summary = []
    for wd in wds:
        for h in horizons:
            mild, hard = agg(h, wd, 0), agg(h, wd, 1)
            if not mild:
                continue
            auc_m = float(np.mean([r["auc"] for r in mild]))
            std_m = float(np.mean([r["style_std"] for r in mild]))
            gap_m = float(np.mean([r["gap"] for r in mild]))
            zf_m = float(np.mean([r["zero_frac"] for r in mild]))
            auc_h = float(np.mean([r["auc"] for r in hard])) if hard else float("nan")
            summary.append((wd, h, auc_m, std_m, gap_m, zf_m, auc_h))
            lines.append(f"| {h} | {lay.dim * h} | {auc_m:.3f} | **{std_m:.3f}** | {gap_m:+.3f} | {zf_m:.2f} | {auc_h:.3f} |")
    lines.append("\n## 自动结论\n")
    for wd in wds:
        cand = [s for s in summary if s[0] == wd]
        if not cand:
            continue
        best = max(cand, key=lambda s: s[3])
        lines.append(f"- wd={wd}：排序信号最强 h={best[1]}（style_std={best[3]:.3f}，轻微档 AUC={best[2]:.3f}）。")
    with open(os.path.join(args.outdir, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[sweep-v2] 报告 → {os.path.join(args.outdir, 'REPORT.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
