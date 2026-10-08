#!/usr/bin/env python3
"""站立踏步探针 dump（云端跑，纯 CPU）。

目的：回答两件事
  1. V19d 加 `stand_penalty = 0.5`（站立档 `feet_air_time` 由 0 价改负价）之后，
     **站立时原地踏步**（`air_time_stand_step_frac`）有没有降下来？
  2. V20 的踏步发生在**站立档**（`cmd_speed ≤ 0.1`）还是**非零指令档**？
     ⇒ 决定下一步是"抬 stand_penalty"还是"上进度判据（投影）"。

判据（全部来自 TB，不需要 GPU）：
  * `Step/air_time_stand_step_frac`  站立 + 真抬脚（d > 0.03 s）占比 ← 主指标，目标 → 0
  * `Step/air_time_stand_pen_mean`   实际付出的站立罚均值（确认罚到东西了没）
  * `Step/air_time_motion_gate_off_frac` 被"实测速度门"关掉的比例（非零指令档的漏检率）
  * `Step/air_time_in_band_frac` / `air_time_hop_frac` / `air_time_single_stance`
  * `Step/stand_pose_pos_err` / `stand_pose_vel_err` / `stand_pose_active`
  * AMP / PPO 侧：`Step/amp_style_std` / `Step/amp_d` / `Loss/discriminator_loss` / `Policy/mean_noise_std`

用法（云端）：
    /opt/conda/envs/isaaclab/bin/python /root/cloud_dump_stand_probes.py v19c v19d v20
"""

from __future__ import annotations

import glob
import os
import statistics
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGROOT = "/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk"

TAGS = (
    "Step/air_time_stand_step_frac",
    "Step/air_time_stand_pen_mean",
    "Step/air_time_motion_gate_off_frac",
    "Step/air_time_in_band_frac",
    "Step/air_time_hop_frac",
    "Step/air_time_single_stance",
    "Step/stand_pose_pos_err",
    "Step/stand_pose_vel_err",
    "Step/stand_pose_active",
    "Step/amp_style_std",
    "Step/amp_d",
    "Step/amp_d_ref",
    "Loss/discriminator_loss",
    "Policy/mean_noise_std",
    "Episode_Reward/feet_air_time",
    "Episode_Reward/track_lin_vel_xy_exp",
)

N_SEG = 6  # 按迭代切几档打印


def find_runs(key: str):
    out = []
    for d in sorted(glob.glob(os.path.join(LOGROOT, "*"))):
        if key not in os.path.basename(d):
            continue
        evs = sorted(glob.glob(os.path.join(d, "**", "events.out.tfevents.*"), recursive=True))
        if evs:
            out.append((os.path.basename(d), sorted(evs, key=os.path.getsize)[-1]))
    return out


def series(ea, tag):
    try:
        return [(s.step, s.value) for s in ea.Scalars(tag)]
    except Exception:
        return []


def report(base: str, evpath: str) -> None:
    ea = EventAccumulator(evpath, size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    print("=" * 100)
    print(f"RUN {base}")
    ref = "Policy/mean_noise_std" if "Policy/mean_noise_std" in tags else (sorted(tags)[0] if tags else None)
    lo = hi = None
    if ref:
        s = series(ea, ref)
        if s:
            lo, hi = s[0][0], s[-1][0]
            print(f"  迭代范围 {lo}..{hi}   （共 {len(s)} 个点；scalar tag 数 {len(tags)}）")
    print(f"  {'tag':<44} {'seg1':>9} {'seg2':>9} {'seg3':>9} {'seg4':>9} {'seg5':>9} {'seg6':>9}   (按迭代均分)")
    for t in TAGS:
        s = series(ea, t)
        if not s:
            print(f"  {t:<44} {'[该 run 无此 tag]':>20}")
            continue
        n = len(s)
        cells = []
        for i in range(N_SEG):
            a = int(n * i / N_SEG)
            b = max(int(n * (i + 1) / N_SEG), a + 1)
            cells.append(statistics.mean([v for _, v in s[a:b]]))
        span = ""
        if lo is not None:
            a0, b0 = s[0][0], s[-1][0]
            span = f"  iters {a0}..{b0}"
        print(f"  {t:<44} " + " ".join(f"{c:>9.4f}" for c in cells) + span)


def main() -> int:
    keys = sys.argv[1:] or ["v19c", "v19d", "v20"]
    for k in keys:
        runs = find_runs(k)
        if not runs:
            print(f"[warn] 找不到匹配 '{k}' 的 run")
        for base, ev in runs:
            report(base, ev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
