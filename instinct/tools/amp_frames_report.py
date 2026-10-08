#!/usr/bin/env python3
"""把 `amp_frames_sweep.py` 的 `sweep.csv` 汇成可读矩阵（按 h / stride / wd 切片）。

用法：
    python instinct/tools/amp_frames_report.py --csv instinct/plots/amp_frames/all/sweep.csv
"""

from __future__ import annotations

import argparse
import csv

FAMILY_NAME = {
    ("N1", 0.02): "关节抖动 σ=0.02",
    ("N1", 0.05): "关节抖动 σ=0.05",
    ("N2", 0.50): "幅度压缩 ×0.5",
    ("N2", 0.25): "幅度压缩 ×0.25",
    ("N3", 2.0): "时间倍速 2x",
    ("N3", 3.0): "时间倍速 3x",
    ("N4", 1.0): "局部冻结",
    ("N5", 1.0): "相位打乱",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="instinct/plots/amp_frames/all/sweep.csv")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--wd", type=float, default=0.0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    for r in rows:
        r["h"] = int(r["h"]); r["stride"] = int(r["stride"])
        for k in ("wd", "level", "auc", "style_std", "gap", "zero_frac", "d_pos_mean", "d_neg_mean"):
            r[k] = float(r[k])

    hs = sorted({r["h"] for r in rows})

    def get(fam, lvl, k, h, stride=None, wd=None):
        stride = args.stride if stride is None else stride
        wd = args.wd if wd is None else wd
        sel = [r for r in rows if r["fam"] == fam and r["level"] == lvl and r["h"] == h
               and r["stride"] == stride and abs(r["wd"] - wd) < 1e-9]
        return sel[0][k] if sel else float("nan")

    keys = sorted({(r["fam"], r["level"]) for r in rows})
    for metric, fmt, title in (
        ("auc", "%6.3f", "AUC（可分性；0.5 = 分不开）"),
        ("style_std", "%6.4f", "std(style)（**排序信号**，越大越好）"),
        ("d_pos_mean", "%6.3f", "参考侧 logit d+（越接近 +1 ⇒ 判别器'敢说话'）"),
    ):
        print(f"\n== {title} — stride={args.stride}, wd={args.wd} ==")
        print("%-16s %s" % ("退化族", " ".join("%6d" % h for h in hs)))
        for fam, lvl in keys:
            name = FAMILY_NAME.get((fam, lvl), f"{fam}@{lvl}")
            vals = [get(fam, lvl, metric, h) for h in hs]
            print("%-16s %s" % (name, " ".join(fmt % v for v in vals)))

    print("\n== stride 影响（AUC）==")
    for metric_fam in keys:
        fam, lvl = metric_fam
        name = FAMILY_NAME.get((fam, lvl), f"{fam}@{lvl}")
        for st in sorted({r["stride"] for r in rows}):
            vals = [get(fam, lvl, "auc", h, stride=st) for h in hs]
            print("  %-16s st=%d %s" % (name, st, " ".join("%6.3f" % v for v in vals)))

    print("\n== wd 影响（AUC 最大差，stride=%d）==" % args.stride)
    for fam, lvl in keys:
        name = FAMILY_NAME.get((fam, lvl), f"{fam}@{lvl}")
        wds = sorted({r["wd"] for r in rows})
        if len(wds) < 2:
            continue
        d = max(abs(get(fam, lvl, "auc", h, wd=wds[0]) - get(fam, lvl, "auc", h, wd=wds[1])) for h in hs)
        print("  %-16s max|ΔAUC| = %.3f" % (name, d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
