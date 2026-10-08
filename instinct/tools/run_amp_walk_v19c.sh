#!/usr/bin/env bash
# ============================================================================
# V19c 启动器（跑在云端）：**步态分红加“实测运动”门控 + 抖动降权 + 站立姿态加强**
# ============================================================================
# 基准 = V19b（从零 / 10 帧 / log_quad+wasserstein / quad k=0.125 / GP1.0 / wd1e-5 /
#               logit_wd0 / eps0.05 / 步态带 exp tau=0.08 / 5 clip 全向指令）。
# **AMP 侧一个参数都不动。**
#
# ① `feet_air_time` 加实测运动门控（motion_threshold=0.15、motion_ang_scale=0.5）
#    ⇒ 修正“指令非零但不前进、原地左右脚交替抬放刷单支撑分红”。
# ② `action_rate_l2` −0.10 → −0.06（回 V17 值）⇒ raw 已 15，罚额 −1.5/步（占正奖励 60%）。
# ③ `track_default_pose_on_stand` −0.3 → −0.5 ⇒ 站立时更彻底回默认姿态、抑制晃腿。
#
# 起点：**热启动 V19b 最新 checkpoint，且保留判别器**（AMP 口径逐字相同）。
#
# 早期监测：
#   * ⭐ `Step/air_time_motion_gate_off_frac`（新探针）> 0（门控在工作）；
#   * `Episode_Reward/action_rate_l2` 应明显变小 → 原始量 = 值/0.06，盯它是否失控（>18~20 就回调）；
#   * `Episode_Reward/track_default_pose_on_stand` 更负；
#   * 步态不退步：`air_time_in_band_frac` / `air_time_hop_frac`（注意运动门控会**压低**
#     `Episode_Reward/feet_air_time` 的绝对值，别只看它）；
#   * 守住 `track_lin_vel_xy_exp ≥ 1.0`、`tilt_deg_mean ≤ 12°`。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v19c.sh' /root/train_amp_walk_v19c_4096.log
#
# 手册：$1 num_envs $2 iters $3 device $4 tag  $5 load_run（空=从零） $6 checkpoint
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v19c_warmV19b_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V19c-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"