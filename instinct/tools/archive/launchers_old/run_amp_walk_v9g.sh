#!/usr/bin/env bash
# ============================================================================
# V9g 的**从零**训练启动器（跑在云端）：奖励重平衡 + 风格项主导
# ============================================================================
# V9g = V9f + ① 压"站着白拿"三项（base_height 1.5→0.5 / ang_vel_z 1.2→0.2 /
#        body_ang_vel_xy 0.7→0.2）+ `track_default_pose_on_stand` −1.0→0
#      + ② 抬"动起来才有"（track_lin_vel 1.5→3.0 / feet_air_time 1.0→2.0 /
#        feet_slide −0.25→−0.1）
#      + ③ 风格权重 ×4（style_lerp 0.75→0.5, discriminator_reward_coef 0.25→0.5）
#
# 判读：`Episode_Reward/feet_air_time` 从 0.0026/s 起来（>0.1/s = 真在交替支撑）、
#       `vx_*_rmse` 下降、`amp_d` 上升 ⇒ 起效；仍 ≈0 ⇒ 下一步动"探索"而非奖励。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v9g.sh' /root/train_amp_walk_v9g_4096.log
#
# 手册：
#   $1 num_envs（默认 4096）  $2 iters（默认 10^9 = 无上限）
#   $3 device（默认 cuda:0）  $4 run_name tag（默认 amp_walk_v9g_mvp_4096）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v9g_mvp_4096}"
TASK="Instinct-Amp-Walk-V9g-G1-v0"

# ⚠️ `$5`(load_run) / `$6`(checkpoint) 显式传空 ⇒ 从零
exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "" "" "$TASK"
