#!/usr/bin/env python3
"""0 指令（站立档）下的**惩罚账目**：`track_default_pose_on_stand` 的两个分量 vs `feet_air_time` 的站立负价。

公式（`rewards_amp.py`）：
    track_default_pose_on_stand:  cost = (pos_scale·pos_err + vel_scale·vel_err) · active
                                  pos_scale = 4.0, vel_scale = 0.5, active = exp(−cmd_speed/0.25)
                                  term weight w_pose（V19c = 0.3, V19d/V20 = 0.5）⇒ 每步罚 = −w_pose·cost
    feet_air_time.stand_penalty:  pen = stand_penalty · min(d/good_lo, 1)  （score 空间）
                                  term weight 4.0 ⇒ 每步罚 = −4.0·pen
                                  ← 实测值就是 `Step/air_time_stand_pen_mean`

`Episode_Reward/<key>/timestep` = 该 term 的**每步加权值**（`reward_manager.py:108` 定义）⇒ 直接是"每步罚额"。

分量拆分用**比值法**（对门控口径不敏感）：
    vel_share = (0.5·vel_err) / (4·pos_err + 0.5·vel_err)

用法（云端）：
    /opt/conda/envs/isaaclab/bin/python /root/cloud_dump_stand_accounting.py v19c v19d v20
"""

from __future__ import annotations

import glob
import os
import statistics
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGROOT = "/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk"

POS_SCALE, VEL_SCALE = 4.0, 0.5
AIR_WEIGHT = 4.0
#: term 权重（来自 env cfg）
W_POSE = {"v19a": 0.5, "v19b": 0.5, "v19c": 0.3, "v19d": 0.5, "v20": 0.5, "v21": 0.5}

TAGS = (
    "Episode_Reward/track_default_pose_on_stand/timestep",
    "Episode_Reward/track_default_pose_on_stand",
    "Episode_Reward/track_default_pose_on_stand/max_episode_len_s",
    "Episode_Reward/feet_air_time/timestep",
    "Episode_Reward/feet_air_time",
    "Episode_Reward/track_lin_vel_xy_exp/timestep",
    "Episode_Reward/track_ang_vel_z_exp/timestep",
    "Episode_Reward/track_base_height/timestep",
    "Episode_Reward/body_ang_vel_xy/timestep",
    "Episode_Reward/action_rate_l2/timestep",
    "Step/stand_pose_pos_err",
    "Step/stand_pose_vel_err",
    "Step/stand_pose_active",
    "Step/air_time_stand_pen_mean",
    "Step/air_time_stand_step_frac",
)


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


def tail_mean(ser, frac=0.25):
    if not ser:
        return None
    n = len(ser)
    a = max(int(n * (1 - frac)), 0)
    return statistics.mean([v for _, v in ser[a:]]), n, ser[0][0], ser[-1][0]


def main() -> int:
    keys = sys.argv[1:] or ["v19c", "v19d", "v20"]
    for k in keys:
        for base, ev in find_runs(k):
            ea = EventAccumulator(ev, size_guidance={"scalars": 0})
            ea.Reload()
            w_pose = W_POSE.get(k, 0.5)
            print("=" * 104)
            print(f"RUN {base}")
            print(f"  （假设 w_pose = {w_pose}）")
            got = {}
            for t in TAGS:
                s = series(ea, t)
                r = tail_mean(s)
                got[t] = r
                if r is None:
                    print(f"  {t:<58} [无此 tag]")
                else:
                    v, n, a0, b0 = r
                    print(f"  {t:<58} 末25%均值 = {v:+.5f}   (n={n}, iters {a0}..{b0})")

            print("  " + "-" * 100)
            err_p = got.get("Step/stand_pose_pos_err")
            err_v = got.get("Step/stand_pose_vel_err")
            act = got.get("Step/stand_pose_active")
            pen = got.get("Step/air_time_stand_pen_mean")
            pose_term = got.get("Episode_Reward/track_default_pose_on_stand/timestep")
            if not (err_p and err_v and pose_term):
                print("  [无法拆分：缺 pos_err / vel_err / pose term 的某一项]")
                continue
            pe, ve, ac = err_p[0], err_v[0], (act[0] if act else float("nan"))
            raw_pos, raw_vel = POS_SCALE * pe, VEL_SCALE * ve
            tot_raw = raw_pos + raw_vel
            pose_val = abs(pose_term[0])  # 每步罚额（已含 term 权重与门控）
            pos_part = pose_val * raw_pos / tot_raw
            vel_part = pose_val * raw_vel / tot_raw
            print(f"  原始量（全 step 平均）：pos_err = {pe:.4f} rad   vel_err = {ve:.4f} rad/s   active = {ac:.4f}")
            print(f"  原始 cost（未含 term 权重/门控）：{POS_SCALE}×{pe:.4f} + {VEL_SCALE}×{ve:.4f} = {raw_pos:.4f} + {raw_vel:.4f} = {tot_raw:.4f}")
            print(f"  实测该项每步罚额（TB /timestep）  = −{pose_val:.4f}/步")
            print(f"    ├─ 姿态回位分量（pos, ×{POS_SCALE}） ≈ −{pos_part:.4f}/步  （占 {100*pos_part/pose_val:.0f}%）")
            print(f"    └─ 关节速度阻尼（vel, ×{VEL_SCALE}） ≈ −{vel_part:.4f}/步  （占 {100*vel_part/pose_val:.0f}%）")
            if pen:
                air_part = AIR_WEIGHT * pen[0]
                print(f"  feet_air_time 的站立负价（stand_penalty=0.5, ×{AIR_WEIGHT}） = −{air_part:.4f}/步")
                print(f"  ⇒ 站立档总罚额 = {pose_val + air_part:.4f}/步（pose 项占 {100*pose_val/(pose_val+air_part):.0f}%，"
                      f"air-time 占 {100*air_part/(pose_val+air_part):.0f}%）")
            pos_sum = 0.0
            for t in ("Episode_Reward/track_lin_vel_xy_exp/timestep", "Episode_Reward/track_ang_vel_z_exp/timestep",
                      "Episode_Reward/track_base_height/timestep", "Episode_Reward/body_ang_vel_xy/timestep"):
                r = got.get(t)
                if r:
                    pos_sum += r[0]
            print(f"  站立档正奖励（4 项主项之和）      = +{pos_sum:.4f}/步")
            if pen:
                print(f"  ⇒ 踏步的总罚 / 站立正奖励 = {(pose_val + AIR_WEIGHT * pen[0]) / pos_sum * 100:.1f}%")
            print(f"  （参考）action_rate_l2 每步 = {got.get('Episode_Reward/action_rate_l2/timestep', (float('nan'),))[0]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
