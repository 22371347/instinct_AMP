#!/usr/bin/env bash
# ============================================================================
# V9b（消融：V9 − 镜像损失）的**从零**训练启动器（跑在云端）
# ============================================================================
# 与 `run_amp_walk_v9.sh` 的唯一区别：TASK 换成 `-V9b-`、默认 tag 换成 v9b。
# 环境配置与 V9 逐字相同（同一 `G1AmpWalkV9EnvCfg` + 同一 teacher 校验器），
# 只有 agent cfg 不同（`use_mirror_loss = False`）⇒ 单变量对照。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v9b.sh' /root/train_amp_walk_v9b_4096.log
#
# 手册：
#   $1 num_envs（默认 4096）  $2 iters（默认 10^9 = 无上限）
#   $3 device（默认 cuda:0）  $4 run_name tag（默认 amp_walk_v9b_mvp_4096）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v9b_mvp_4096}"
TASK="Instinct-Amp-Walk-V9b-G1-v0"

# ⚠️ `$5`(load_run) / `$6`(checkpoint) 显式传空 ⇒ 从零（观测维度与 V9 相同，但换 agent cfg）
exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "" "" "$TASK"
