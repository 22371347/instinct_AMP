#!/usr/bin/env python3
"""**离线判别器架构 bake-off**（纯 torch，本地 CPU/GPU，不占训练机 GPU、不需要 Isaac）。

回答的问题
----------
"判别器加一层（[1024,512] → mjlab 的 [1024,512,256]）到底有没有用？"

⚠️ **不能看 `disc_loss`**：我们自己的历史就是铁证 —— V27 `disc_loss = 0.0014`
（判别器**死掉**、风格项零梯度、步态不类人）vs V30 `0.086`（活着、风格项有效）。
⇒ **拟合越好 = AMP 越废**。所以本脚本比的是"**信息量**"：

| 指标 | 含义 | 越好 |
|---|---|---|
| `r_actor` / `r_ref` | 风格奖励（`bounded_quad`, k=0.25）在两分布上的均值 | 前者越高越好 |
| `style_std` | `r` 在 actor 批上的**标准差** = 策略真正能拿到的**差分信号** | ↑ |
| `zero_frac` | `r == 0` 的比例（`d ≤ −1` 死区）| ↓（死区 = 无梯度）|
| ⭐ **β 曲线可用区间** | `x(β)=(1−β)·ref+β·actor` 上 `r̄(β)` **保持有斜率**的 β 范围 | ↑（这才是"能不能当老师"）|
| ⭐ **平均斜率** | `|dr̄/dβ|` 在可用区间上的均值 | ↑ |
| `grad_attr` | `|∂d/∂x|` 按 **量块 × 部位** 的归因 | 看新容量花在**有用维**还是少数退化维 |

数据（**真实、成对、同相位**）
------------------------------
`instinct/eval/amp_state_dump_body.npz`：
    `pol_frame`/`ref_frame` (T=1995, N=16, 195)、`clip_id` (T, N)
  * `ref_frame` = **同一时刻、同一环境的相位匹配参考**（AMP 的 expert 侧）；
  * `pol_frame` = **同一时刻的策略**（AMP 的 actor 侧）—— 所以 `(pol, ref)` 是**成对**的，
    β 插值曲线才有意义（不是两个无关分布的插值）；
  * 按 `clip_id` 变化切掉"回合重置"处的假邻接帧，再把相邻两帧拼成 390 维 `(s_t, s_{t+1})`
    —— 与训练侧判别器输入**同构**（两侧用同一套拼接 ⇒ 架构对比有效）；
  * **按 env 切分 train/test**（默认 12/4）⇒ 报告里的指标全部是**留出集**上的，不是训练集。

复刻的训练侧口径（两边完全一致，只有 `hidden_sizes` 不同 ⇒ 干净的单变量）
------------------------------------------------------------------------
    trunk/head 结构     `Linear+ReLU`×L + `Linear(·,1)`（与 `instinct_rl/modules/discriminator.py` 同构）
    判别器损失          `0.5·(MSE(d_actor,−1) + MSE(d_ref,+1))`
    GP                  λ=10，**只在 expert 侧**，目标 ‖∇d‖→0（tol 0），在**白化后空间**求
    优化器              AdamW lr=1e-4, betas=(0.9,0.999)，**分组 wd**：trunk 1e-3 / head 1e-2
    风格奖励形状        `clamp(1 − 0.25·(d−1)², 0, 1)`
    白化                固定标准化（`eps=0.01`，同 `EmpiricalNormalization`），
                        统计量只由**指定一侧/two侧**拟合（默认 `expert` = V27~V31 的口径）

用法
----
    conda activate instinct
    python instinct/tools/offline_disc_bakeoff.py --steps 1200            # 完整对比
    python instinct/tools/offline_disc_bakeoff.py --steps 50 --archs L2   # 冒烟测试
    python instinct/tools/offline_disc_bakeoff.py --whiten both           # 换成 B/C 臂口径
    python instinct/tools/offline_disc_bakeoff.py --identity_insert       # 追加"无损插层"臂
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
DUMP = REPO / "instinct" / "eval" / "amp_state_dump_body.npz"
OUT_DIR = REPO / "instinct" / "probe"

#: 候选架构（名字 → hidden sizes）。前两个是我们要对比的；第三个用来把"深度"与"宽度"分开。
ARCHS: dict[str, list[int]] = {
    "L2": [1024, 512],          # 我们当前（V27~V32）
    "L3_mjlab": [1024, 512, 256],  # AMP_mjlab 原版
    "L3_wide": [1024, 512, 512],   # 同样多一层但宽度不变（便于无损插层对照）
}

#: 13 体顺序（与 `AMP_V20_BODY_NAMES` 一致）：pelvis + 双髋/膝/踝 + 双肩/肘/腕
_BODY_PARTS = {"torso": (0,), "legs": (1, 2, 3, 4, 5, 6), "arms": (7, 8, 9, 10, 11, 12)}
_TERM_BLOCKS = (("pos", 3), ("ori", 6), ("linvel", 3), ("angvel", 3))


# ----------------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------------
class Disc(nn.Module):
    """与 `instinct_rl/modules/discriminator.py` 同构：`Linear+ReLU` 主干 + 单头。"""

    def __init__(self, in_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers, cur = [], in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(cur, h), nn.ReLU()]
            cur = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(cur, 1)
        self.hidden_sizes = list(hidden_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


def build_identity_inserted(prev: Disc, in_dim: int, base_hidden: list[int], seed: int) -> tuple[Disc, bool]:
    """**无损插层**：把 `base_hidden`（如 [1024,512]）的判别器扩成 [1024,512,512]，
    并在中间插入 `Linear(512,512, W=I, b=0)+ReLU` ⇒ 主干输出 `ReLU(I·h) = h`（h≥0）
    ⇒ 与旧判别器**函数逐点相同**（head 权重逐字拷贝）。

    ⇒ 于是"多一层"带来的任何差异都只可能来自**额外容量 + 该层的新优化器状态**，
       而没有"冷启动"这个混淆源（这是在线实验里手段 A 的核心）。
    """
    if len(base_hidden) < 2:
        raise ValueError("base_hidden 至少两层")
    w = base_hidden[-1]
    new = Disc(in_dim, base_hidden + [w])
    with torch.no_grad():
        # 拷贝前 len(base_hidden) 个 Linear/ReLU 的权重
        src = [m for m in prev.trunk if isinstance(m, nn.Linear)]
        dst = [m for m in new.trunk if isinstance(m, nn.Linear)]
        for s, d in zip(src, dst):
            d.weight.copy_(s.weight)
            d.bias.copy_(s.bias)
        # 插入层 = 单位阵（ReLU 对非负输入是恒等）
        ins = dst[len(base_hidden)]
        nn.init.zeros_(ins.bias)
        ins.weight.copy_(torch.eye(w))
        # head 逐字拷贝
        new.head.weight.copy_(prev.head.weight)
        new.head.bias.copy_(prev.head.bias)
    # 函数等价性自检（随机输入上逐点比较）
    with torch.no_grad():
        x = torch.randn(64, in_dim)
        ok = torch.allclose(new(x), prev(x), atol=1e-6)
    return new, bool(ok)


class FixedWhiten(nn.Module):
    """固定标准化（离线版；比"运行统计量"更干净 —— 去掉统计量漂移这个混淆源）。"""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)


# ----------------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------------
def _layout_from_meta(d) -> tuple[int, list[tuple[str, int]], int]:
    """从 dump 的元数据推 (体数, 每体量块, 每帧维度)；解析失败则回落到 13 体固定布局。"""
    frame_dim = int(d["frame_dim"]) if "frame_dim" in d.files else 195
    n_bodies = None
    blocks: list[tuple[str, int]] = list(_TERM_BLOCKS)
    try:
        names = [str(x) for x in np.asarray(d["body_names"]).reshape(-1)]
        n_bodies = len(names) if names else None
    except Exception:
        names = []
    try:
        tpb = np.asarray(d["term_per_body"]).reshape(-1)
        parsed = []
        for item in tpb:
            s = str(item)
            for nm, dim in _TERM_BLOCKS:
                if nm.replace("_", "") in s.replace("_", "").lower().replace("body", ""):
                    parsed.append((nm, dim))
                    break
        if len(parsed) == len(_TERM_BLOCKS):
            blocks = parsed
    except Exception:
        pass
    if n_bodies is None:
        per_body = sum(dim for _, dim in blocks)
        n_bodies = frame_dim // per_body
    return int(n_bodies), blocks, frame_dim


def load_pairs(test_env_frac: float, seed: int, log: list[str]):
    if not DUMP.is_file():
        raise FileNotFoundError(f"找不到 {DUMP}（换用 --dump 指定）")
    d = np.load(DUMP, allow_pickle=True)
    pol, ref = d["pol_frame"], d["ref_frame"]          # (T, N, 195)
    clip = d["clip_id"] if "clip_id" in d.files else None
    n_bodies, blocks, frame_dim = _layout_from_meta(d)
    log.append(f"数据：{DUMP.name}  pol/ref={pol.shape}  frame_dim={frame_dim}  "
               f"n_bodies={n_bodies}  blocks={blocks}")

    T, N, D = pol.shape
    if clip is not None:
        same = clip[:-1] == clip[1:]                    # 排除回合重置处的假邻接
    else:
        same = np.ones((T - 1, N), dtype=bool)
    Xa = np.concatenate([pol[:-1], pol[1:]], axis=-1)[same]   # (M, 2D) = (s_t, s_{t+1})
    Xr = np.concatenate([ref[:-1], ref[1:]], axis=-1)[same]
    env_ids = np.broadcast_to(np.arange(N), (T - 1, N))[same]
    clip_ids = (clip[:-1][same] if clip is not None else np.zeros(len(Xa)))
    log.append(f"成对样本：{Xa.shape[0]} 条（有效邻接 {same.mean()*100:.1f}%）  两侧同维 {Xa.shape[1]}")

    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(N * test_env_frac)))
    test_envs = rng.choice(N, size=n_test, replace=False)
    te = np.isin(env_ids, test_envs)
    data = dict(
        Xa_tr=torch.tensor(Xa[~te], dtype=torch.float32),
        Xr_tr=torch.tensor(Xr[~te], dtype=torch.float32),
        Xa_te=torch.tensor(Xa[te], dtype=torch.float32),
        Xr_te=torch.tensor(Xr[te], dtype=torch.float32),
        clip_te=clip_ids[te],
        in_dim=int(Xa.shape[1]),
        n_bodies=n_bodies,
        blocks=blocks,
        frame_dim=frame_dim,
    )
    log.append(f"划分：train {int((~te).sum())} 条 / test {int(te.sum())} 条（test envs={sorted(test_envs.tolist())}）")
    return data


# ----------------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------------
def grad_pen(disc: Disc, xs: list[torch.Tensor], lam: float = 10.0) -> torch.Tensor:
    """λ·E‖∂d/∂x‖²（tol=0）。`xs` = 要施加约束的分布（专家侧 / 两侧）。

    ⚠️ 我们训练侧对应两个开关：
      * `discriminator_gp_expert_only=True`（V30/V31/V32-A，对齐 mjlab）⇒ 只传 `[xr]`；
      * `False`（V27/V29 的旧口径）⇒ 传 `[xa, xr]`。
    """
    x = torch.cat(xs, dim=0).detach().clone().requires_grad_(True)
    d = disc(x)
    g = torch.autograd.grad(d, x, torch.ones_like(d), create_graph=True, retain_graph=True)[0]
    return lam * g.norm(2, dim=1).pow(2).mean()


def make_optimizer(disc: Disc, lr: float, wd_trunk: float, wd_head: float):
    return torch.optim.AdamW(
        [
            {"params": list(disc.trunk.parameters()), "weight_decay": wd_trunk},
            {"params": list(disc.head.parameters()), "weight_decay": wd_head},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )


def train_one(name: str, hidden: list[int], data, whiten: FixedWhiten, args, log: list[str]) -> dict:
    torch.manual_seed(args.seed)
    dev = args.device
    disc = Disc(data["in_dim"], hidden).to(dev)
    opt = make_optimizer(disc, args.lr, args.wd_trunk, args.wd_head)
    Xa, Xr = data["Xa_tr"].to(dev), data["Xr_tr"].to(dev)
    mse = nn.MSELoss()
    hist = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        ia = torch.randint(0, Xa.shape[0], (args.batch,), device=dev)
        ir = torch.randint(0, Xr.shape[0], (args.batch,), device=dev)
        xa, xr = whiten(Xa[ia]), whiten(Xr[ir])
        d_a, d_r = disc(xa), disc(xr)
        loss_a = mse(d_a, -torch.ones_like(d_a))
        loss_r = mse(d_r, torch.ones_like(d_r))
        amp_loss = 0.5 * (loss_a + loss_r)
        gp = grad_pen(disc, [xr] if args.gp_sides == "expert" else [xa, xr], args.gp_coef) \
            if args.gp_coef > 0 else torch.zeros((), device=dev)
        loss = amp_loss + gp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.log_every == 0 or step == 1:
            hist.append(dict(step=step, disc_loss=float(amp_loss.detach()), gp=float(gp.detach()),
                             d_a=float(d_a.mean()), d_r=float(d_r.mean())))
    dt = time.time() - t0
    log.append(f"  [{name}] 训练完成 {args.steps} 步，用时 {dt:.1f}s（{dt/max(1,args.steps)*1000:.0f} ms/步）")
    return dict(name=name, hidden=hidden, disc=disc, whiten=whiten, history=hist, train_s=dt)


# ----------------------------------------------------------------------------
# 评估
# ----------------------------------------------------------------------------
def style_reward(d: torch.Tensor, k: float = 0.25) -> torch.Tensor:
    return torch.clamp(1.0 - k * (d - 1.0) ** 2, min=0.0, max=1.0)


@torch.no_grad()
def metrics(disc: Disc, whiten: FixedWhiten, Xa, Xr, bs: int = 8192) -> dict:
    dev = next(disc.parameters()).device
    out = {}
    for tag, X in (("actor", Xa), ("ref", Xr)):
        ds, rs = [], []
        for i in range(0, X.shape[0], bs):
            d = disc(whiten(X[i:i + bs].to(dev))).reshape(-1)
            ds.append(d)
            rs.append(style_reward(d))
        d = torch.cat(ds)
        r = torch.cat(rs)
        out[f"d_{tag}"] = float(d.mean())
        out[f"r_{tag}"] = float(r.mean())
        out[f"style_std_{tag}"] = float(r.std())
        out[f"zero_frac_{tag}"] = float((r == 0).float().mean())
    out["gap"] = out["d_ref"] - out["d_actor"]
    return out


@torch.no_grad()
def beta_curve(disc: Disc, whiten: FixedWhiten, Xa, Xr, betas=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)) -> list[dict]:
    """`x(β)=(1−β)·ref + β·actor`（**同相位成对**插值）⇒ r̄(β)。
    β=0 是纯参考、β=1 是纯策略 —— 一条**策略演化会走过的路径**的代理。"""
    dev = next(disc.parameters()).device
    n = min(Xa.shape[0], Xr.shape[0])
    Xa, Xr = Xa[:n], Xr[:n]
    rows = []
    for b in betas:
        x = whiten((1.0 - b) * Xr.to(dev) + b * Xa.to(dev))
        d = disc(x).reshape(-1)
        r = style_reward(d)
        rows.append(dict(beta=float(b), r=float(r.mean()), d=float(d.mean()),
                         zero_frac=float((r == 0).float().mean())))
    return rows


def usable_range(rows: list[dict], thr: float = 0.05) -> tuple[float, float]:
    """可用区间 = r̄(β) 仍 > thr 的最大 β；平均斜率 = 在可用区间上的 |Δr̄/Δβ|。"""
    beta_max = 0.0
    for row in rows:
        if row["r"] > thr:
            beta_max = row["beta"]
    xs, ys = [r["beta"] for r in rows if r["beta"] <= beta_max], [r["r"] for r in rows if r["beta"] <= beta_max]
    slope = 0.0
    if len(xs) >= 2 and (xs[-1] - xs[0]) > 0:
        slope = abs(ys[0] - ys[-1]) / (xs[-1] - xs[0])
    return beta_max, slope


def grad_attribution(disc: Disc, whiten: FixedWhiten, Xr, data, bs: int = 2048) -> dict:
    """`|∂d/∂x|` 按量块 / 部位 / 全体的均值（在**白化后空间**，与 GP 同空间）。

    ⚠️ 索引口径：本 dump 的每条样本 = `[frame_t(195), frame_t+1(195)]`，而每帧内部是
    **按量块分块的**：`pos(13×3) | ori(13×6) | linvel(13×3) | angvel(13×3)`。
    ⇒ 每帧维度 = `n_bodies × Σ块宽 = 13×15 = 195`（**不是 15**！写错会让所有切片变空 ⇒ NaN）。
    """
    dev = next(disc.parameters()).device
    n_bodies, blocks = data["n_bodies"], data["blocks"]
    x = whiten(Xr[:bs].to(dev)).detach().clone().requires_grad_(True)
    d = disc(x)
    g = torch.autograd.grad(d, x, torch.ones_like(d), retain_graph=False)[0]
    ag = g.detach().abs().mean(dim=0)                       # (D,)
    per_frame = n_bodies * sum(dim for _, dim in blocks)    # 13×15 = 195
    n_frames = max(1, ag.numel() // per_frame)              # 390/195 = 2
    term_sum = {nm: 0.0 for nm, _ in blocks}
    term_cnt = 0
    part_sum = {p: 0.0 for p in _BODY_PARTS}
    part_cnt = {p: 0 for p in _BODY_PARTS}
    for fi in range(n_frames):
        off = fi * per_frame
        base = 0
        for nm, dim in blocks:
            for b in range(n_bodies):
                s = off + base + b * dim
                m = float(ag[s:s + dim].mean())
                term_sum[nm] += m
                term_cnt += 1
                for p, bodies in _BODY_PARTS.items():
                    if b in bodies:
                        part_sum[p] += m
                        part_cnt[p] += 1
            base += n_bodies * dim
    return {
        **{f"term/{k}": v / max(1, term_cnt) for k, v in term_sum.items()},
        **{f"part/{k}": part_sum[k] / max(1, part_cnt[k]) for k in _BODY_PARTS},
        "all": float(ag.mean()),
    }


def live_point(rows: list[dict], target_d: float) -> tuple[float, float, float]:
    """在 β 曲线上找**当前策略的工作点**：`d̄(β) ≈ target_d` 的那个 β（默认 −0.75 = V32 实测）。

    为什么需要：dump 里的 actor 是 **V20 时代**的策略，偏离比现在的 V32 大得多 ⇒
    整条曲线上的“纯 actor 端”（β=1）比当前实际工作点更糟。
    把 **β̂ 处的 r̄ 与局部斜率**拿出来，两个架构才是在**同一个工作点**上比。
    """
    best_i = min(range(len(rows)), key=lambda i: abs(rows[i]["d"] - target_d))
    lo, hi = rows[max(0, best_i - 1)], rows[min(len(rows) - 1, best_i + 1)]
    dx = hi["beta"] - lo["beta"]
    slope = abs(hi["r"] - lo["r"]) / dx if dx > 0 else 0.0
    return rows[best_i]["beta"], rows[best_i]["r"], slope


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(DUMP))
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--wd_trunk", type=float, default=1.0e-3)
    ap.add_argument("--wd_head", type=float, default=1.0e-2)
    ap.add_argument("--gp_coef", type=float, default=10.0)
    ap.add_argument("--gp_sides", choices=["expert", "both"], default="expert",
                    help="GP 施加在哪些分布上：expert = 只专家侧（mjlab/我们 V30+）；both = 两侧（V27/V29 旧口径）")
    ap.add_argument("--whiten", choices=["none", "expert", "both"], default="expert")
    ap.add_argument("--test_env_frac", type=float, default=0.25)
    ap.add_argument("--archs", default="L2,L3_mjlab,L3_wide")
    ap.add_argument("--identity_insert", action="store_true",
                    help="追加一臂：L2 训练完 → 无损插层成 [1024,512,512] → 再训同样步数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--live_d", type=float, default=-0.75,
                    help="当前策略的实测工作点 d_actor（V32 稳态≈−0.75）；用于 β̂ 对齐")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    log: list[str] = []
    log.append("=" * 100)
    log.append("离线判别器架构 bake-off")
    log.append(f"参数：{vars(args)}")
    log.append(f"设备：{args.device}（cuda 可用={torch.cuda.is_available()}）")
    data = load_pairs(args.test_env_frac, args.seed, log)

    # ---- 白化（固定标准化；统计量只由指定一侧拟合）----
    if args.whiten == "none":
        whiten = FixedWhiten(torch.zeros(data["in_dim"]), torch.ones(data["in_dim"])).to(args.device)
        log.append("白化：关（恒等）")
    else:
        pool = [data["Xr_tr"]]
        if args.whiten == "both":
            pool.append(data["Xa_tr"])
        cat = torch.cat(pool, dim=0)
        whiten = FixedWhiten(cat.mean(0), cat.std(0).clamp_min(1e-6)).to(args.device)
        log.append(f"白化：{args.whiten}（mean/std 只由 {'+'.join(['expert'] + (['actor'] if args.whiten=='both' else []))} 侧拟合）")

    names = [s.strip() for s in args.archs.split(",") if s.strip()]
    runs = []
    for name in names:
        log.append(f"\n---- 训练 {name} hidden={ARCHS[name]} ----")
        runs.append(train_one(name, ARCHS[name], data, whiten, args, log))

    if args.identity_insert:
        base = next((r for r in runs if r["name"] == "L2"), None)
        if base is None:
            log.append("!! --identity_insert 需要先跑 L2")
        else:
            log.append("\n---- 无损插层：由训练好的 L2 构造 [1024,512,512] 并**继续**训练 ----")
            ins, ok = build_identity_inserted(base["disc"], data["in_dim"], ARCHS["L2"], args.seed)
            log.append(f"  插层函数等价性自检（逐点 max|Δd|）：{'✅ 通过' if ok else '❌ 失败'}")
            ins = ins.to(args.device)
            opt = make_optimizer(ins, args.lr, args.wd_trunk, args.wd_head)
            Xa, Xr = data["Xa_tr"].to(args.device), data["Xr_tr"].to(args.device)
            mse = nn.MSELoss()
            hist, t0 = [], time.time()
            for step in range(1, args.steps + 1):
                ia = torch.randint(0, Xa.shape[0], (args.batch,), device=args.device)
                ir = torch.randint(0, Xr.shape[0], (args.batch,), device=args.device)
                xa, xr = whiten(Xa[ia]), whiten(Xr[ir])
                da, dr = ins(xa), ins(xr)
                amp_loss = 0.5 * (mse(da, -torch.ones_like(da)) + mse(dr, torch.ones_like(dr)))
                gp = grad_pen(ins, [xr] if args.gp_sides == "expert" else [xa, xr], args.gp_coef) \
                    if args.gp_coef > 0 else torch.zeros((), device=args.device)
                loss = amp_loss + gp
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if step % args.log_every == 0 or step == 1:
                hist.append(dict(step=step, disc_loss=float(amp_loss.detach()), gp=float(gp.detach()),
                                 d_a=float(da.mean()), d_r=float(dr.mean())))
            log.append(f"  [L3_id] 继续训练 {args.steps} 步，用时 {time.time()-t0:.1f}s")
            runs.append(dict(name="L3_id(lossless)", hidden=ARCHS["L2"] + [ARCHS["L2"][-1]],
                             disc=ins, whiten=whiten, history=hist, train_s=time.time() - t0))

    # ---- 留出集评估 ----
    log.append("\n" + "=" * 100)
    log.append("留出集评估（**信息量**口径；`disc_loss` 只作参照，越低不等于越好）")
    rows = []
    for r in runs:
        m = metrics(r["disc"], whiten, data["Xa_te"], data["Xr_te"])
        bc = beta_curve(r["disc"], whiten, data["Xa_te"], data["Xr_te"])
        bmax, slope = usable_range(bc)
        bhat, rhat, shat = live_point(bc, args.live_d)
        ga = grad_attribution(r["disc"], whiten, data["Xr_te"], data)
        last = r["history"][-1]
        rows.append(dict(name=r["name"], hidden=r["hidden"], **m, beta_max=bmax, beta_slope=slope,
                         beta_hat=bhat, r_hat=rhat, slope_hat=shat,
                         disc_loss=last["disc_loss"], gp=last["gp"], **ga, beta_curve=bc,
                         history=r["history"], train_s=r["train_s"]))

    hdr = (f"{'arch':<16}{'hidden':<20}{'disc_loss':>10}{'d_actor':>9}{'d_ref':>8}{'gap':>7}"
           f"{'r_actor':>9}{'style_std':>10}{'zero_frac':>10}{'β_max':>7}{'|dr/dβ|':>9}"
           f"{'β̂':>7}{'r(β̂)':>9}{'斜率@β̂':>9}")
    log.append(hdr)
    log.append("-" * len(hdr))
    for r in rows:
        log.append(f"{r['name']:<16}{str(r['hidden']):<20}{r['disc_loss']:>10.4f}"
                   f"{r['d_actor']:>9.3f}{r['d_ref']:>8.3f}{r['gap']:>7.3f}"
                   f"{r['r_actor']:>9.4f}{r['style_std_actor']:>10.4f}{r['zero_frac_actor']:>10.3f}"
                   f"{r['beta_max']:>7.2f}{r['beta_slope']:>9.3f}"
                   f"{r['beta_hat']:>7.2f}{r['r_hat']:>9.4f}{r['slope_hat']:>9.3f}")

    log.append("\n梯度归因（`|∂d/∂x|`，白化后空间）：")
    for r in rows:
        log.append(f"  {r['name']:<16} all={r['all']:.4f} | "
                   + " ".join(f"{k}={r['term/'+k]:.3f}" for k in ("pos", "ori", "linvel", "angvel"))
                   + " | " + " ".join(f"{k}={r['part/'+k]:.3f}" for k in ("torso", "legs", "arms")))

    log.append("\nβ 曲线（策略→参考的插值路径上 r̄ 的变化；**这是「能不能当老师」的直接读数**）：")
    for r in rows:
        log.append(f"  {r['name']:<16} " + " ".join(f"β={c['beta']:.2f}:{c['r']:+.3f}" for c in r["beta_curve"]))

    # ---- 结论 ----
    log.append("\n" + "=" * 100)
    if len(rows) >= 2:
        a, b = rows[0], rows[1]
        log.append("结论（{0} vs {1}）：".format(a["name"], b["name"]))
        log.append(f"  · 差分信号 style_std：{a['style_std_actor']:.4f} → {b['style_std_actor']:.4f} "
                   f"（{(b['style_std_actor']/max(a['style_std_actor'],1e-9)-1)*100:+.1f}%）")
        log.append(f"  · 死区 zero_frac：{a['zero_frac_actor']:.3f} → {b['zero_frac_actor']:.3f}")
        log.append(f"  · 可用区间 β_max：{a['beta_max']:.2f} → {b['beta_max']:.2f}；斜率 {a['beta_slope']:.3f} → {b['beta_slope']:.3f}")
        log.append(f"  · ⭐ 同工作点 β̂：{a['beta_hat']:.2f} vs {b['beta_hat']:.2f}；"
                   f"该点 r̄ {a['r_hat']:.4f} → {b['r_hat']:.4f}；局部斜率 {a['slope_hat']:.3f} → {b['slope_hat']:.3f}")
        log.append(f"  · 拟合 disc_loss：{a['disc_loss']:.4f} → {b['disc_loss']:.4f}（⚠️ 越低≠越好）")
        better = (b["style_std_actor"] > a["style_std_actor"] * 1.05 and b["beta_max"] >= a["beta_max"]
                  and b["slope_hat"] >= a["slope_hat"] * 0.95)
        log.append(f"  ⇒ 判定：**{'加层更好（信号更强/可用区间不缩）' if better else '加层没有带来可用的信息量提升（或更差）'}**")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or time.strftime("%m%d_%H%M%S")
    out_txt = OUT_DIR / f"offline_bakeoff_{tag}.txt"
    out_json = OUT_DIR / f"offline_bakeoff_{tag}.json"
    out_txt.write_text("\n".join(log) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(dict(args=vars(args), rows=rows), ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(log))
    print(f"\n报告：{out_txt}\nJSON：{out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
