#!/usr/bin/env python3
"""V29 逐维诊断：**判别器到底是"靠哪几维"把策略与参考分开的？**

输入：`instinct/probe/v29_dims/amp_dim_probe_*.txt`（V29 口径，2 帧 × 13 body × 15 维 = 390）
       每文件 4 行：μ_actor / μ_ref / σ_actor / σ_ref（在 4096 env 上统计）。

布局（`amp_walk_v20_env_cfg.py` 模块 docstring §3：帧序 **term-major**）：
    每帧 195 = pos_b(13×3=39) + ori_b(13×6=78) + lin_vel_b(13×3=39) + ang_vel_b(13×3=39)
    两组帧：**老 → 新**（旧帧在前）
    ⇒ 0-38 老 pos / 39-77 新 pos / 78-155 老 ori / 156-233 新 ori /
       234-272 老 lin_vel / 273-311 新 lin_vel / 312-350 老 ang_vel / 351-389 新 ang_vel

指标：
  * `sep = |μ_a − μ_r| / σ_r`     —— 判别器"按这一维切一刀"能拿到的相对间隔
  * `d   = |μ_a − μ_r| / σ_pooled`—— 标准化均值差（Cohen's d）；**d ≳ 1 就已经"很容易分"**
"""

from __future__ import annotations

import glob
import os

import numpy as np

PROBE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probe")
BODIES = [
    "pelvis", "hip_roll_L", "hip_roll_R", "knee_L", "knee_R",
    "ankle_roll_L", "ankle_roll_R", "shoulder_roll_L", "shoulder_roll_R",
    "elbow_L", "elbow_R", "wrist_roll_L", "wrist_roll_R",
]
#: term -> (老帧起点, 新帧起点, 每 body 维数)
TERMS = [
    ("pos_b", 0, 39, 3),
    ("ori_b", 78, 156, 6),
    ("lin_vel_b", 234, 273, 3),
    ("ang_vel_b", 312, 351, 3),
]


def decode(idx: int) -> str:
    for name, s_old, s_new, per in TERMS:
        for tag, base in (("老", s_old), ("新", s_new)):
            if base <= idx < base + 13 * per:
                k = idx - base
                b, c = divmod(k, per)
                comp = {3: ["x", "y", "z"], 6: ["r0", "r1", "r2", "r3", "r4", "r5"]}[per][c]
                return f"{name}({tag})[{BODIES[b]}.{comp}]"
    return f"<未映射 #{idx}>"


def main() -> int:
    files = sorted(glob.glob(os.path.join(PROBE_DIR, "amp_dim_probe_*.txt")))
    if not files:
        print(f"❌ 没找到探针文件：{PROBE_DIR}")
        return 1
    print("=" * 92)
    print(f"V29 逐维诊断：{len(files)} 个探针文件 × 4 行（μ_actor / μ_ref / σ_actor / σ_ref）")
    print("=" * 92)

    seps, cohens = [], []
    for f in files:
        a = np.loadtxt(f)
        mu_a, mu_r, sd_a, sd_r = a[0], a[1], a[2], a[3]
        seps.append(np.abs(mu_a - mu_r) / np.maximum(sd_r, 1e-6))
        pool = np.sqrt(0.5 * (sd_a**2 + sd_r**2))
        cohens.append(np.abs(mu_a - mu_r) / np.maximum(pool, 1e-6))
    sep = np.mean(seps, axis=0)
    coh = np.mean(cohens, axis=0)
    D = sep.shape[0]
    print(f"维度 D = {D}（期望 390）")

    # ---------- 1) 按 term × 帧 聚合 ----------
    print("\n【1】按 term × 帧 聚合（均值 / 中位 / 最大）")
    print(f"  {'term':<12}{'帧':<4}{'均值 sep':>10}{'中位 sep':>10}{'最大 sep':>10}{'d>1 的维数':>12}{'不均衡份额':>12}")
    share = {}
    for name, s_old, s_new, per in TERMS:
        for tag, base in (("老", s_old), ("新", s_new)):
            sl = slice(base, base + 13 * per)
            v = sep[sl]
            share[f"{name}({tag})"] = float(np.sum(v))
            print(f"  {name:<12}{tag:<4}{v.mean():>10.3f}{np.median(v):>10.3f}{v.max():>10.3f}"
                  f"{int((coh[sl] > 1).sum()):>9}/{len(v):<3}{100 * np.sum(v) / sep.sum():>10.1f}%")

    # ---------- 2) 位置/姿态 vs 速度 ----------
    pose = np.concatenate([sep[0:39], sep[39:78], sep[78:156], sep[156:234]])
    vel = np.concatenate([sep[234:273], sep[273:312], sep[312:351], sep[351:390]])
    print("\n【2】「位置/姿态」 vs 「速度」（速度 = lin_vel_b + ang_vel_b，78 维/帧 × 2 帧 = 156 维）")
    print(f"  姿态类 {pose.size:>3} 维：均值 sep {pose.mean():.3f}  中位 {np.median(pose):.3f}  "
          f"最大 {pose.max():.3f}  占分离份额 {100 * pose.sum() / sep.sum():.1f}%")
    print(f"  速度类 {vel.size:>3} 维：均值 sep {vel.mean():.3f}  中位 {np.median(vel):.3f}  "
          f"最大 {vel.max():.3f}  占分离份额 {100 * vel.sum() / sep.sum():.1f}%")

    # ---------- 3) Top-20 单维 ----------
    print("\n【3】单维分离度 Top-20（判别器「只按这一维切一刀」的间隔 / 标准化均值差）")
    print(f"  {'#':<4}{'sep':>9}{'Cohen d':>10}  {'维度':<38}")
    order = np.argsort(-sep)[:20]
    for r, i in enumerate(order, 1):
        print(f"  {r:<4}{sep[i]:>9.3f}{coh[i]:>10.3f}  {decode(int(i)):<38}")

    # ---------- 4) 整体可分离性 ----------
    print("\n【4】整体可分离性（判别器能否「一步到位」分开）")
    n_strong = int((coh > 1).sum())
    print(f"  Cohen's d > 1 的维数：{n_strong}/{D}（d>1 ⇒ 仅凭该维就能高度可分）")
    print(f"  Cohen's d > 2 的维数：{int((coh > 2).sum())}/{D}（d>2 ⇒ 几乎是「白送」的判别依据）")
    print(f"  全体 sep：均值 {sep.mean():.3f}  中位 {np.median(sep):.3f}  最大 {sep.max():.3f}")
    print(f"  ⇒ 只要 top-{max(1, n_strong // 10)} 个维度就足以让判别器在 20 步内把 MSE 压到 {0.035:.3f}")

    # ---------- 5) 身体部位聚合 ----------
    print("\n【5】按 body 聚合（每个 body 的 30 维 = 15 维 × 2 帧）")
    rows = []
    for b, name in enumerate(BODIES):
        idx = []
        for _, s_old, s_new, per in TERMS:
            idx += list(range(s_old + b * per, s_old + (b + 1) * per))
            idx += list(range(s_new + b * per, s_new + (b + 1) * per))
        rows.append((np.mean(sep[idx]), np.max(sep[idx]), name))
    for m, mx, name in sorted(rows, reverse=True):
        print(f"  {name:<18}均值 {m:>8.3f}   最大 {mx:>8.3f}")

    # ---------- 6) 结论 ----------
    print("\n【6】结论")
    top_terms = sorted(share.items(), key=lambda kv: -kv[1])[:3]
    print("  分离份额最大的 3 个 term： " +
          "、".join(f"{k} {100 * v / sep.sum():.1f}%" for k, v in top_terms))
    if vel.mean() > 1.5 * pose.mean():
        print("  ⚠️ **速度维（lin_vel_b/ang_vel_b）是主要分离来源** ⇒ 建议把速度维从 AMP 状态里删掉")
    elif pose.mean() > 1.5 * vel.mean():
        print("  ⚠️ **位姿维（pos_b/ori_b）是主要分离来源** ⇒ 更可能是「整体偏移」（基座高度/姿态），"
              "建议只做**均值对齐**（减参考均值、不做缩放）")
    else:
        print("  ⚠️ 位姿与速度**都**在分离，且量级相当 ⇒ 判别器输入整体过大，建议降容量 + 专家侧 GP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
