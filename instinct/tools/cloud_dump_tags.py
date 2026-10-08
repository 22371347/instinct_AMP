#!/usr/bin/env python3
"""按**正则**列出某个 run 里所有匹配的 TB 标量（末 25% 均值），用于 per-joint 之类的细粒度排查。

用法（云端）：
    /opt/conda/envs/isaaclab/bin/python /root/cloud_dump_tags.py v19d joint_error_rmse
    /opt/conda/envs/isaaclab/bin/python /root/cloud_dump_tags.py v19d 'Step/air_time_'
    /opt/conda/envs/isaaclab/bin/python /root/cloud_dump_tags.py v19d --list      # 只列 tag 名
"""

from __future__ import annotations

import glob
import os
import re
import statistics
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGROOT = "/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk"


def main() -> int:
    key, pattern = sys.argv[1], sys.argv[2]
    dirs = [d for d in sorted(glob.glob(os.path.join(LOGROOT, "*"))) if key in os.path.basename(d)]
    if not dirs:
        print(f"[warn] 找不到 run '{key}'")
        return 1
    d = dirs[-1]
    evs = sorted(glob.glob(os.path.join(d, "**", "events.out.tfevents.*"), recursive=True))
    ea = EventAccumulator(sorted(evs, key=os.path.getsize)[-1], size_guidance={"scalars": 0})
    ea.Reload()
    tags = sorted(ea.Tags().get("scalars", []))
    if pattern == "--list":
        print(f"RUN {os.path.basename(d)}  共 {len(tags)} 个 scalar tag：")
        for t in tags:
            print("   ", t)
        return 0
    rx = re.compile(pattern)
    hits = [t for t in tags if rx.search(t)]
    print(f"RUN {os.path.basename(d)}   匹配 '{pattern}' 的 tag 共 {len(hits)} 个（末 25% 均值）：")
    rows = []
    for t in hits:
        s = [(x.step, x.value) for x in ea.Scalars(t)]
        n = len(s)
        v = statistics.mean([y for _, y in s[max(int(n * 0.75), 0):]])
        rows.append((v, t, s[0][0], s[-1][0], n))
    for v, t, a0, b0, n in sorted(rows, reverse=True):
        print(f"  {v:+12.5f}   {t}   (iters {a0}..{b0}, n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
