#!/usr/bin/env python3
"""**判别器体检 Phase 1（零 GPU 成本）**：先查"专家"本身是否健康。

为什么先做这一步
----------------
T1b（风格权重 0.25）与 T1c（0.10）给出**几乎相同的速度崩塌**（`vx_high` 7400 轮 →0.17/0.21，
7600 轮 →0.076/0.100，最终 0.014/0.030）。权重砍掉 60% 完全没延缓崩塌 ⇒ 不像是"风格项太强"，
更像是**风格项的梯度方向本身在奖励低能量模态**。

那就要问一个前置问题：**判别器奖励"低能量"会不会是"正确"的？** 如果这 2929/2934 帧专家集
里绝大多数是低速/静止帧，那么"模仿专家 ⇒ 站着"就是判别器的正确行为，病根在**参考数据**，
改奖励形状（`bounded_quad → log_quad`）只是治标。

本脚本只读一个静态张量，不起 Isaac、不用 GPU。

键位口径（从 `amp_expert_stats_396.txt` 的 μ 重复结构反推 + 与训练日志 obs_format 对齐）
--------------------------------------------------------------------------------------
    obs_format: body_pos_b(78) body_ori_b(156) body_lin_vel_b(78) body_ang_vel_b(78) anchor_gravity_b(6)
    78 = 39×2、156 = 78×2、6 = 3×2 ⇒ **每帧 195 维、13 个 body、term-major / frame-inner**
    偏移:  pos 0:39 / 39:78      ori 78:156 / 156:234
          lin_vel 234:273 / 273:312    ang_vel 312:351 / 351:390
          gravity 390:393 / 393:396
    body 0 = anchor（`σ[pos_y] == 0`，锚点自身在锚系里 y ≡ 0）
    ⇒ **前进速度 = 第 234 维**，侧向 235，竖直 236；偏航角速度 = 第 312 维
    脚本不硬编码偏移：由 term 尺寸推导，并在总数 != 396 时 **fail-fast**（防止口径漂移）。

用法（云端）：
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_diag_discriminator.py
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_diag_discriminator.py --buf /root/amp_expert_states.pt --dim 390
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

#: 各 term 的**每帧**尺寸（= 训练日志 obs_format 里的一半）
TERMS = (
    ("body_pos_b", 39),
    ("body_ori_b", 78),
    ("body_lin_vel_b", 39),
    ("body_ang_vel_b", 39),
    ("anchor_gravity_b", 3),
)
BODIES = 13


def offsets(dim: int) -> dict[str, tuple[int, int]]:
    """term → (t 帧起始, t+1 帧起始)。fail-fast 校验总维数。"""
    out: dict[str, tuple[int, int]] = {}
    cur = 0
    for name, per_frame in TERMS:
        if cur + per_frame * 2 > dim:
            raise SystemExit(f"[FATAL] 口径不符：{name} 需要 {per_frame*2} 维，已用 {cur}/{dim}")
        out[name] = (cur, cur + per_frame)
        cur += per_frame * 2
    if cur != dim:
        raise SystemExit(
            f"[FATAL] 口径不符：TERMS 合计 {cur} 维 != 数据 {dim} 维。"
            "（V33S 多导了 anchor_gravity 6 维；390 版应少这 6 维）"
        )
    return out


def hist(v: torch.Tensor, edges: list[float], label: str) -> None:
    n = v.numel()
    print(f"  {label}:  mean={v.mean():+.3f}  std={v.std():.3f} "
          f"p5={v.quantile(0.05):+.3f}  p50={v.quantile(0.50):+.3f}  p95={v.quantile(0.95):+.3f}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        frac = ((v >= lo) & (v < hi)).float().mean().item()
        bar = "#" * int(round(frac * 60))
        print(f"    [{lo:+.2f},{hi:+.2f}) {frac*100:5.1f}%  {bar}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", default="/root/amp_expert_states_396.pt")
    ap.add_argument("--dim", type=int, default=396)
    args = ap.parse_args()

    if not os.path.isfile(args.buf):
        print(f"[FATAL] 找不到 {args.buf}")
        return 2

    raw = torch.load(args.buf, map_location="cpu")
    if isinstance(raw, dict):
        print(f"[note] 是 dict，键={list(raw.keys())}")
        x = next(v for v in raw.values() if isinstance(v, torch.Tensor))
    else:
        x = raw
    x = x.float()
    print(f"# 专家集: {args.buf}")
    print(f"# shape={tuple(x.shape)}  dtype={x.dtype}  finite={bool(torch.isfinite(x).all())}")
    if x.dim() != 2 or x.shape[1] != args.dim:
        print(f"[FATAL] 期望 (N, {args.dim})，实得 {tuple(x.shape)}")
        return 2
    n = x.shape[0]
    print(f"# 帧数 N={n}")

    off = offsets(args.dim)
    print("\n[1] 键位口径（term → 帧 t / 帧 t+1 的列区间）")
    for k, (a, b) in off.items():
        print(f"  {k:18s} t=[{a}:{b}]  t+1=[{b}:{b + (b - a)}]")

    lv_t = off["body_lin_vel_b"][0]
    av_t = off["body_ang_vel_b"][0]
    g_t = off["anchor_gravity_b"][0]

    # ---- ② anchor 自身的速度（body 0 = 第 0..2 列）-----------------------------
    print("\n[2] ⭐ 专家集的 anchor 速度分布（body 0，帧 t）—— 决定判别器「该奖励什么」")
    vx = x[:, lv_t + 0]
    vy = x[:, lv_t + 1]
    vz = x[:, lv_t + 2]
    print("  ── 前进速度 lin_vel_x（锚系）")
    hist(vx, [-1.5, -0.5, -0.1, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 3.0], "lin_vel_x")
    for thr in (0.3, 0.5, 0.7):
        f = (vx.abs() >= thr).float().mean().item()
        print(f"    |lin_vel_x| >= {thr:.1f} m/s 占比: {f*100:.1f}%")
    stand = (x[:, lv_t + 0].abs() < 0.1) & (x[:, lv_t + 1].abs() < 0.1)
    print(f"    「近静止」(|vx|<0.1 且 |vy|<0.1) 占比: {stand.float().mean().item()*100:.1f}%")
    print("  ── 侧向 / 竖直速度")
    hist(vy, [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0], "lin_vel_y")
    hist(vz, [-0.5, -0.1, 0.1, 0.5], "lin_vel_z")

    print("\n[3] 专家集的偏航角速度（body 0，帧 t）—— 转向能力的参考")
    wz = x[:, av_t + 2]
    hist(wz, [-3.0, -1.0, -0.3, 0.3, 1.0, 3.0], "ang_vel_z")
    for thr in (0.3, 0.8):
        print(f"    |ang_vel_z| >= {thr:.1f} rad/s 占比: {(wz.abs() >= thr).float().mean().item()*100:.1f}%")

    print("\n[4] 前后帧一致性（专家批是 (t, t+1) 成对 ⇒ 相邻帧应高度相关）")
    for name in ("body_pos_b", "body_lin_vel_b"):
        a, b = off[name]
        d = (x[:, b:b + (b - a)] - x[:, a:b]).abs().mean().item()
        print(f"  Δ{name:16s} 逐元素 |Δ| 均值 = {d:.5f}"
              f"（{'✅ 相邻帧、成对' if d < 0.05 else '⚠️ 差异偏大，检查是否成对排布'}）")

    print("\n[5] 结论提示")
    hi = (vx >= 0.5).float().mean().item()
    print(f"  专家集高速前进帧（vx>=0.5 m/s）占比 = {hi*100:.1f}%")
    if hi < 0.10:
        print("  ⛔ 专家集**高速帧稀缺** ⇒ 判别器「奖励低能量」是**正确**的；"
              "病根在参考数据/子采样，改奖励形状只是治标。")
    elif hi < 0.25:
        print("  ⚠️ 高速帧偏少 ⇒ 风格项会把策略往中低速拉；建议重采专家集（按速度分层）。")
    else:
        print("  ✅ 专家集含足量高速帧 ⇒ 病根更可能在**判别器/奖励形状**（下一步做敏感性探针）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
