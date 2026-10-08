#!/usr/bin/env bash
# ============================================================================
# V9c 的**从零**训练启动器（跑在云端）：v9b（无镜像损失）+ 降 σ + 抬 AMP 权重
# ============================================================================
# 与 V9b 的差别只在 agent/env cfg（见 `amp_walk_v9c_env_cfg.py` 顶部的对照表）：
#   entropy_coef 0.008→0.0025、clip_max_std 0.4→0.3、discriminator_reward_coef 0.05→0.25、
#   joint_deviation_arms −0.1→0（权重归零，项本身仍在 ⇒ teacher 契约不变）
#
# ⚠️ 与 V9b **并行**跑（V9b 用 4096 env，V9c 用 2048 env，4090 显存够）。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v9c.sh' /root/train_amp_walk_v9c_2048.log
#
# 手册：
#   $1 num_envs（默认 2048）  $2 iters（默认 10^9 = 无上限）
#   $3 device（默认 cuda:0）  $4 run_name tag（默认 amp_walk_v9c_mvp_2048）
# ============================================================================
set -uo pipefail

INST="${1:-2048}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v9c_mvp_2048}"
TASK="Instinct-Amp-Walk-V9c-G1-v0"

# ⚠️ `$5`(load_run) / `$6`(checkpoint) 显式传空 ⇒ 从零
exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "" "" "$TASK"
