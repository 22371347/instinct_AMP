#!/usr/bin/env python3
"""**V30 预检（零训练成本）**：把"专家源换成全数据集随机抽帧"是否能救判别器，先算出来。

思路：判别器的可分离性可以用**标准化均值差（Cohen's d）**预估
    d = |μ_a − μ_e| / sqrt((σ_a² + σ_e²)/2)
两个专家源各算一遍：
  * `live_reference`：现在的影子机器人（V29 实测根因：近同相位 ⇒ σ 极小 ⇒ d 巨大）
  * `expert_dataset`：导出的全数据集 390 维状态（mjlab 的 `AMPLoader` 口径）

判读：
  * 中位 d ≲ 0.5 ⇒ 两个分布**大量重叠** ⇒ 判别器无法平凡分开 ⇒ **值得做 V30（值得开训）**
  * 中位 d ≳ 2   ⇒ 仍是"零重叠" ⇒ 换专家源没用 ⇒ 直接转显式步态奖励（B 路线）

输入（从 `/root/amp_export_bundle.tgz` 解出）：
  * `amp_dim_probe_steady_005.txt`（热启动瞬态）/ `_050.txt`（**稳态**，量化以此为准）
    4 行：μ_actor / μ_live_ref / σ_actor / σ_live_ref
  * `amp_expert_stats.txt`：含 "== <name> μ ==" / "== <name> σ ==" 各一段（空格分隔的 390 个数）

用法：python instinct/tools/analyze_v30_expert_gap.py [数据目录]
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np

BODIES = [
    "pelvis", "hip_roll_L", "hip_roll_R", "knee_L", "knee_R", "ankle_roll_L", "ankle_roll_R",
    "shoulder_roll_L", "shoulder_roll_R", "elbow_L", "elbow_R", "wrist_roll_L", "wrist_roll_R",
]
TERMS = [("pos_b", 0, 39, 3), ("ori_b", 78, 156, 6),
         ("lin_vel_b", 234, 273, 3), ("ang_vel_b", 312, 351, 3)]


def decode(idx: int) -> str:
    for name, s_old, s_new, per in TERMS:
        for tag, base in (("老", s_old), ("新", s_new)):
            if base <= idx < base + 13 * per:
                k = idx - base
                b, c = divmod(k, per)
                comp = {3: ["x", "y", "z"], 6: [f"r{i}" for i in range(6)]}[per][c]
                return f"{name}({tag})[{BODIES[b]}.{comp}]"
    return f"<#{idx}>"


def parse_stats(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """解析 `amp_expert_stats.txt`：返回 {name: (μ, σ)}。"""
    if not os.path.exists(path):
        return {}
    txt = open(path, encoding="utf-8").read()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for m in re.finditer(r"==\s+(.+?)\s+(μ|σ)\s+==\s*\n([\d\.\-\+eE\s]+)", txt):
        name, kind, nums = m.group(1).strip(), m.group(2), m.group(3)
        arr = np.fromstring(nums.strip(), sep=" ")
        if arr.size == 0:
            continue
        mu, sd = out.get(name, (None, None))
        if kind == "μ":
            mu = arr
        else:
            sd = arr
        out[name] = (mu, sd)
    return {k: v for k, v in out.items() if v[0] is not None and v[1] is not None}


def cohen(mu1, sd1, mu2, sd2):
    return np.abs(mu1 - mu2) / np.maximum(np.sqrt(0.5 * (sd1**2 + sd2**2)), 1e-9)


def report(tag: str, d: np.ndarray) -> None:
    print(f"  {tag:<34} 中位 {np.median(d):>7.3f}  均值 {d.mean():>7.3f}  最大 {d.max():>8.3f}  "
          f"d>1: {int((d > 1).sum()):>3}/390   d>2: {int((d > 2).sum()):>3}")


def term_table(d: np.ndarray) -> None:
    print(f"    {'term':<12}{'均值 d':>9}{'最大 d':>9}")
    for name, s_old, s_new, per in TERMS:
        for tag, base in (("老", s_old), ("新", s_new)):
            v = d[base:base + 13 * per]
            print(f"    {name + '(' + tag + ')':<12}{v.mean():>9.3f}{v.max():>9.3f}")


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "probe")
    print("=" * 96)
    print(f"V30 预检：换专家源（影子机器人 → 全数据集随机抽帧）能否让判别器不再平凡可分")
    print(f"数据目录：{root}")
    print("=" * 96)

    probes = sorted(glob.glob(os.path.join(root, "amp_dim_probe_*.txt")))
    if not probes:
        print("❌ 没找到探针文件 amp_dim_probe_*.txt")
        return 1
    stats = parse_stats(os.path.join(root, "amp_expert_stats.txt"))
    if not stats:
        print("⚠️ 没解析到 amp_expert_stats.txt（只做活体参考的对照）")
    print(f"探针：{[os.path.basename(p) for p in probes]}")
    print(f"数据集统计段：{list(stats)}")

    for p in probes:
        a = np.loadtxt(p)
        mu_a, mu_r, sd_a, sd_r = a[0], a[1], a[2], a[3]
        label = os.path.basename(p).replace("amp_dim_probe_steady_", "").replace(".txt", "")
        note = "（热启动瞬态，仅参考）" if label in ("005", "05", "5") else "（稳态，**以此为准**）"
        print(f"\n【{label}】{note}")
        d_live = cohen(mu_a, sd_a, mu_r, sd_r)
        report("d(actor vs 影子机器人)", d_live)
        if "live_reference" in stats:
            mu_l, sd_l = stats["live_reference"]
            report("d(actor vs 活体参考[再采])", cohen(mu_a, sd_a, mu_l, sd_l))
        if "expert_dataset" in stats:
            mu_e, sd_e = stats["expert_dataset"]
            d_ds = cohen(mu_a, sd_a, mu_e, sd_e)
            report("d(actor vs **全数据集**)", d_ds)
            print(f"    ⇒ 分离度下降倍数（中位）：{np.median(d_live) / max(np.median(d_ds), 1e-9):.2f}×")
            term_table(d_ds)
            order = np.argsort(-d_ds)[:12]
            print("    **换成全数据集后**仍然最强的 12 维：")
            for r, i in enumerate(order, 1):
                print(f"      {r:<3}d={d_ds[i]:>7.3f}  {decode(int(i)):<34}"
                      f"μ_a={mu_a[i]:+.4g} μ_e={mu_e[i]:+.4g} σ_a={sd_a[i]:.3g} σ_e={sd_e[i]:.3g}")

    if "expert_dataset" in stats:
        sd_e = stats["expert_dataset"][1]
        print("\n【数据集本身的多样性】σ_dataset："
              f"中位 {np.median(sd_e):.4g}（活体参考 σ 中位 "
              f"{np.median(stats['live_reference'][1]) if 'live_reference' in stats else float('nan'):.4g}）"
              f"；σ<0.01 的维数 {int((sd_e < 0.01).sum())}/390")

        print("\n【结论】")
        d_all = []
        for p in probes:
            if "050" in p or "50" in os.path.basename(p):
                a = np.loadtxt(p)
                d_all.append(cohen(a[0], a[2], stats["expert_dataset"][0], stats["expert_dataset"][1]))
        if not d_all:
            a = np.loadtxt(probes[-1])
            d_all.append(cohen(a[0], a[2], stats["expert_dataset"][0], stats["expert_dataset"][1]))
        med = float(np.median(d_all[0]))
        if med < 0.5:
            print(f"  ✅ 稳态中位 d = {med:.3f} < 0.5 ⇒ 两个分布**大量重叠** ⇒ "
                  "换成 mjlab 的专家源**有希望**让判别器不再平凡可分 ⇒ 值得开 V30 训练")
        elif med < 2.0:
            print(f"  🟡 稳态中位 d = {med:.3f} ∈ [0.5, 2) ⇒ **部分重叠** ⇒ "
                  "值得一试（但不保证）；建议同时把速度维去掉再降一档")
        else:
            print(f"  ❌ 稳态中位 d = {med:.3f} ≥ 2 ⇒ 仍是零重叠 ⇒ **换专家源救不了** ⇒ "
                  "不要开 V30；转显式步态奖励（B 路线）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
