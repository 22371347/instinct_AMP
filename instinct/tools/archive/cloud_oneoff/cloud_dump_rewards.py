#!/usr/bin/env python3
"""把 TB events 里 **全部** `Episode_Reward/*` 的实测数值打出来（在云端跑）。

用途：核对「某奖励项是否真的作用在奖励函数内 / 权重调整后实际贡献多少」。
只打印，不做任何判断。

用法（云端）：
  python cloud_dump_rewards.py --dirs <run1> [<run2> ...] [--tail-frac 0.1]

- 多目录会按 step 合并（用于「原始 run + 续训 run」拼接完整曲线）。
- 输出列：term / 末值 / 末段均值 / 全程最大 / 全程最小 / 数据点数
  「末段均值」= 最后 `--tail-frac` 比例的数据点均值，抗单点噪声。
- 数值是 **normalize 后的 per-second 值**（与训练日志打印一致）。
"""
import argparse
import glob
import os
import shutil
import tempfile

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tags(dirs):
    """返回 {tag: [(step, value), ...]}，多目录按 step 合并排序。"""
    series = {}
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "events.out.tfevents.*")))
        if not files:
            print(f"[warn] 目录内没有 events 文件: {d}")
            continue
        # 云端 events 可能在被写入，先拷到 /tmp 再读，避免读到半截。
        tmp = tempfile.mkdtemp(prefix="dumprew_")
        local = []
        for f in files:
            dst = os.path.join(tmp, os.path.basename(f))
            shutil.copy2(f, dst)
            local.append(dst)
        for f in local:
            acc = EventAccumulator(f, size_guidance={"scalars": 0})
            acc.Reload()
            for tag in acc.Tags().get("scalars", []):
                pts = [(e.step, e.value) for e in acc.Scalars(tag)]
                series.setdefault(tag, []).extend(pts)
        shutil.rmtree(tmp, ignore_errors=True)
    for tag in series:
        series[tag].sort(key=lambda p: p[0])
    return series


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", required=True, help="run 目录（可多个，按 step 合并）")
    ap.add_argument("--tail-frac", type=float, default=0.1, help="末段比例，默认 0.1")
    ap.add_argument("--prefix", default="Episode_Reward/", help="只看以此开头的 tag")
    ap.add_argument("--all", action="store_true", help="打印全部 tag，而不只是 Episode_Reward/*")
    args = ap.parse_args()

    series = load_tags(args.dirs)
    if not series:
        print("没有读到任何 scalar tag")
        return

    tags = sorted(t for t in series if args.all or t.startswith(args.prefix))
    print(f"目录数 = {len(args.dirs)}   匹配 tag 数 = {len(tags)}")
    print(f"{'tag':<42}{'末值':>12}{'末段均值':>12}{'全程max':>12}{'全程min':>12}{'点数':>7}")
    print("-" * 97)
    rows = []
    for tag in tags:
        pts = series[tag]
        if not pts:
            continue
        vals = [v for _, v in pts]
        n_tail = max(1, int(len(vals) * args.tail_frac))
        tail = vals[-n_tail:]
        rows.append(
            (
                tag,
                vals[-1],
                sum(tail) / len(tail),
                max(vals),
                min(vals),
                len(vals),
            )
        )
    # 按「末段均值的绝对值」从大到小排，方便看谁在起作用
    for tag, last, tailavg, mx, mn, n in sorted(rows, key=lambda r: -abs(r[2])):
        name = tag.replace(args.prefix, "")
        print(f"{name:<42}{last:>12.5f}{tailavg:>12.5f}{mx:>12.5f}{mn:>12.5f}{n:>7}")


if __name__ == "__main__":
    main()
