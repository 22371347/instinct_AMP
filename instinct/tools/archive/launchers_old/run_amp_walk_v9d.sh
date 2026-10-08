#!/usr/bin/env bash
# ============================================================================
# V9d 的**从零**训练启动器（跑在云端）：V9c + 恢复镜像损失（单变量）
# ============================================================================
# V9c 的实测有效项全部保留：entropy 0.0025（σ 收敛到 0.21）/ 上限 0.3 / 初值 0.2、
# AMP 权重 c=0.25（amp_d −0.79→−0.45，r_amp 0.23→0.65）、joint_deviation_arms=0。
# ⭐ 新增：use_mirror_loss=True —— 去掉它会让策略退化成"单脚/不对称跳"。
#
# ⏱️ 判读建议：至少跑到 3000~5000 轮再比步态（V9b/V9c 只跑到 ~430 轮，太早）。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v9d.sh' /root/train_amp_walk_v9d_4096.log
#
# 手册：
#   $1 num_envs（默认 4096）  $2 iters（默认 10^9 = 无上限）
#   $3 device（默认 cuda:0）  $4 run_name tag（默认 amp_walk_v9d_mvp_4096）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v9d_mvp_4096}"
TASK="Instinct-Amp-Walk-V9d-G1-v0"

# ⚠️ `$5`(load_run) / `$6`(checkpoint) 显式传空 ⇒ 从零
exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "" "" "$TASK"
