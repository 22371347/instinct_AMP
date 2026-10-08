#!/usr/bin/env bash
# ============================================================================
# V31 启动器（跑在云端）：**只改一个数** —— `flat_orientation_l2` −2 → −4
# ============================================================================
# 起点：**V30 最新 ckpt 热启动，且保留判别器**（AMP 口径没变 ⇒ 不需要 `drop_amp_discriminator`）
#       ⇒ "权重 −2→−4" 是干净的单变量。
#
# 为什么：2026-10-06 姿态诊断（数据在案）
#   * 关节树根 = `torso_link`（URDF 实测）⇒ `flat_orientation_l2` 读的就是**躯干绝对倾角**；
#   * V27 → V30：`Step/tilt_deg_mean` 12.56° → **23.86°**、`tilt_gt30_frac` 1.2% → **36%**、
#     `Step/grav_bx_mean` −0.037 → **−0.366**（负 = **后仰** ≈21.5°）；
#   * 参考动作基线（17 clip / 7417 帧）：**9.0° 且略前倾**（`g_x=+0.147`、>30° 为 0%）；
#   * **AMP 判别器对绝对倾角是盲的**（状态只含 13 体相对 anchor 的位姿/速度）
#     ⇒ 姿态**只能**靠任务奖励定价，而 `flat_orientation_l2` 是**唯一**管它的项；
#   * 边际核算：w=−2 时 24° 处边际价格 2.97/rad < 收益梯度 ≈4.3/rad ⇒ 净梯度为正 ⇒ 继续后仰；
#     w=−4 ⇒ 5.94/rad ⇒ 净梯度转负。
#
# 监测（TB）：
#   主判据：`Step/tilt_deg_mean` ≤15°、`Step/grav_bx_mean` ≥ −0.15、`tilt_gt30_frac` ≤0.10
#   闸门（防"为站直把步态压保守"）：`air_time_single_stance ≥ 0.18`、`feet_air_time ≥ 0.95`、
#                                    `mean_episode_length ≥ 500`
#   判别器不许退化：`Loss/discriminator_loss` 仍 >0.05（V30 已 0.088）
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v31.sh' /root/train_amp_walk_v31_4096.log
#
# 手册：$1 num_envs $2 iters $3 device $4 tag $5 load_run（空=从零） $6 checkpoint
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v31_flat4_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V31-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
