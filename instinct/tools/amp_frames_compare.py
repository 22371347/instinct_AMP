#!/usr/bin/env python3
"""把两次 `amp_frames_sweep_v2.py` 的结果并列对比，回答：

    "刚体空间 195 维（body195）需要几帧，才能达到关节空间 67 维（legacy67）在 h=8~10 的水平？"

输入：若干个 `sweep.json`（每个对应一套布局/数据）。
输出：终端表 + `REPORT_compare.md` + 两张图（ASCII 轴标签，避免 CJK 字体问题）。

用法（本地）：
    python instinct/tools/amp_frames_compare.py \
        --sweep instinct/plots/amp_frames_legacy/sweep.json \
        --sweep instinct/plots/amp_frames_body/sweep.json \
        --outdir instinct/plots/amp_frames_compare
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

#: 各族的"轻微档"强度（与 sweep 脚本的 LEVELS 一致），用于按档位取行
LEVELS = {"N1": (0.02, 0.05), "N2": (0.50, 0.25), "N3": (2, 3), "N4": (1.0, 1.0), "N5": (1.0, 1.0)}
TEMPORAL = ("N3", "N4", "N5")     # 时间结构类退化（离线研究证明 2 帧对它们不可分）
STATIC = ("N1", "N2")             # 静态类退化（任何 h 都能分开）


def load(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def agg_by_h(rows: list[dict], wd: float, fams: tuple[str, ...], lvl_idx: int = 0) -> dict[int, dict]:
    """{h: {auc, style_std, gap, zero_frac, dim}}，对给定族集合取平均。"""
    out: dict[int, dict] = {}
    for r in rows:
        if r["wd"] != wd or r["fam"] not in fams:
            continue
        if r["level"] != LEVELS[r["fam"]][lvl_idx]:
            continue
        e = out.setdefault(r["h"], {"auc": [], "style_std": [], "gap": [], "zero_frac": [], "dim": r["dim"]})
        e["auc"].append(r["auc"])
        e["style_std"].append(r["style_std"])
        e["gap"].append(r["gap"])
        e["zero_frac"].append(r["zero_frac"])
    return {h: dict(dim=v["dim"], auc=float(np.mean(v["auc"])), style_std=float(np.mean(v["style_std"])),
                    gap=float(np.mean(v["gap"])), zero_frac=float(np.mean(v["zero_frac"])))
            for h, v in out.items()}


def h_equivalent(curve: dict[int, dict], target_auc: float, hs: list[int]) -> str:
    """返回"达到 target_auc 所需的最小 h"。"""
    for h in hs:
        if h in curve and curve[h]["auc"] >= target_auc:
            return str(h)
    return f">{max(hs)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="append", required=True, help="sweep.json（可多次）")
    ap.add_argument("--labels", default="", help="逗号分隔的显示名（默认取 json 里的 layout 名）")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--outdir", default="instinct/plots/amp_frames_compare")
    args = ap.parse_args()

    sweeps = [load(p) for p in args.sweep]
    labels = [s for s in args.labels.split(",") if s] or [s[0]["layout"] for s in sweeps]
    assert len(labels) == len(sweeps), "标签数必须与 sweep 数一致"
    hs = sorted({int(r["h"]) for r in sweeps[0]})

    os.makedirs(args.outdir, exist_ok=True)
    lines = ["# 离线判别性研究 v2：**布局对比**（帧数 h 扫描，协议逐字相同）\n",
             f"- 主指标：轻微档下**时间结构类退化**（N3 时间倍速 / N4 局部冻结 / N5 相位打乱）的",
             f"  `AUC`（可分性）与 `style_std`（排序信号）；wd={args.wd}",
             "- 静态类退化（N1 抖动 / N2 幅度压缩）单独列出，作对照（任何 h 都能分开 ⇒ 不构成判据）\n"]

    curves_t: dict[str, dict[int, dict]] = {}
    curves_s: dict[str, dict[int, dict]] = {}
    for lab, rows in zip(labels, sweeps):
        curves_t[lab] = agg_by_h(rows, args.wd, TEMPORAL)
        curves_s[lab] = agg_by_h(rows, args.wd, STATIC)
        dim = rows[0]["dim"] // rows[0]["h"]
        print(f"[cmp] {lab}: {dim} 维/帧，h 覆盖 {sorted(curves_t[lab])}")

    # ---- 表 1：时间结构类 ----
    lines.append("## 表 1｜时间结构类退化（N3/N4/N5 平均，轻微档）\n")
    head = "| h | " + " | ".join(f"{l} AUC" for l in labels) + " | " + " | ".join(f"{l} style_std" for l in labels) + " |"
    lines.append(head)
    lines.append("|---" * (1 + 2 * len(labels)) + "|")
    for h in hs:
        a = " | ".join(f"{curves_t[l][h]['auc']:.3f}" if h in curves_t[l] else "-" for l in labels)
        s = " | ".join(f"{curves_t[l][h]['style_std']:.4f}" if h in curves_t[l] else "-" for l in labels)
        lines.append(f"| {h} | {a} | {s} |")

    # ---- 表 2：静态类（对照）----
    lines.append("\n## 表 2｜静态类退化（N1/N2 平均，对照用）\n")
    lines.append("| h | " + " | ".join(f"{l} AUC" for l in labels) + " |")
    lines.append("|---" * (1 + len(labels)) + "|")
    for h in hs:
        a = " | ".join(f"{curves_s[l][h]['auc']:.3f}" if h in curves_s[l] else "-" for l in labels)
        lines.append(f"| {h} | {a} |")

    # ---- 等价帧数 ----
    lines.append("\n## 表 3｜\"达到某个判别水平所需的 h\"（时间结构类）\n")
    lines.append("| 目标 AUC | " + " | ".join(f"{l} 所需 h" for l in labels) + " |")
    lines.append("|---" * (1 + len(labels)) + "|")
    for target in (0.60, 0.65, 0.70, 0.80):
        cells = " | ".join(h_equivalent(curves_t[l], target, hs) for l in labels)
        lines.append(f"| {target:.2f} | {cells} |")

    # ---- 自动结论 ----
    lines.append("\n## 自动结论\n")
    for l in labels:
        if not curves_t[l]:
            continue
        best = max(curves_t[l].items(), key=lambda kv: kv[1]["style_std"])
        lines.append(f"- `{l}`：排序信号最强 h={best[0]}（style_std={best[1]['style_std']:.4f}，"
                     f"AUC={best[1]['auc']:.3f}，输入维 {best[1]['dim']}）；"
                     f"h=2 的 AUC={curves_t[l].get(2, {}).get('auc', float('nan')):.3f}、"
                     f"h=10 的 AUC={curves_t[l].get(10, {}).get('auc', float('nan')):.3f}。")
    with open(os.path.join(args.outdir, "REPORT_compare.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- 图（ASCII 标签）----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for fname, curves, ylab, title in (
            ("auc_vs_h.png", curves_t, "AUC (temporal degradations, mild)", "Discriminability vs horizon h"),
            ("style_std_vs_h.png", curves_t, "std(style) (temporal degradations, mild)", "Ranking signal vs horizon h"),
        ):
            fig, ax = plt.subplots(figsize=(7, 4.2))
            for l in labels:
                ys = [curves[l].get(h, {}).get("auc" if "auc" in ylab else "style_std", np.nan) for h in hs]
                ax.plot(hs, ys, marker="o", label=l)
            ax.set_xlabel("horizon h (frames)")
            ax.set_ylabel(ylab)
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, fname), dpi=130)
            print(f"[cmp] 图 → {os.path.join(args.outdir, fname)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[cmp] 画图跳过：{exc}")

    print(f"[cmp] 报告 → {os.path.join(args.outdir, 'REPORT_compare.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
