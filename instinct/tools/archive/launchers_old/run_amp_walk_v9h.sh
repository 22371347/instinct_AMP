#!/usr/bin/env bash
# ============================================================================
# V9h 的**从零**训练启动器（跑在云端）：抗跳 + 恢复平衡刹车 + 补回基类项
# ============================================================================
# V9h = V9g 的奖励集**修过头之后**的版本：
#   ① 恢复平衡刹车：track_ang_vel_z 0.2→0.8、body_ang_vel_xy 0.2→0.5、
#      track_base_height 0.5→1.0（std 0.3→0.2）
#   ② 保留激励：track_lin_vel 3.0、feet_air_time 2.0
#   ③ `track_default_pose_on_stand` 0 → −0.3（小权重保留静止姿态稳定）
#   ④ 补回 4 项基类被删项（16 → **20 项**）：
#      lin_vel_z_l2 −0.2 ⭐（躯干竖直速度：跳 vz≈±1，走 vz≈±0.1 ⇒ 直接给跳定价）
#      dof_torques_l2 −4e-6、joint_deviation_hip −0.1、joint_deviation_knee −0.05
#   ⑤ 风格侧回退：style_lerp 0.5→0.75、discriminator_reward_coef 0.5→0.25
#      （v9g 的 ×4 实测无效：amp_d 只动了 0.026，且同期出现崩溃振荡）
#
# 判读：`Episode_Reward/feet_air_time` > 0.05/s（终于出现单脚支撑）、
#       `Episode_Termination/base_contact` < 0.5、`Mean reward` 包络 ≥45、
#       新增 `Mean step amp_d_ref` ⇒ gap = d_ref − d_actor 是否明显（AMP 还有没有分辨力）。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v9h.sh' /root/train_amp_walk_v9h_4096.log
#
# 手册：$1 num_envs（默认 4096）  $2 iters（默认 10^9 = 无上限）
#       $3 device（默认 cuda:0）  $4 run_name tag（默认 amp_walk_v9h_mvp_4096）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v9h_mvp_4096}"
TASK="Instinct-Amp-Walk-V9h-G1-v0"

# ⚠️ `$5`(load_run) / `$6`(checkpoint) 显式传空 ⇒ 从零
exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "" "" "$TASK"
