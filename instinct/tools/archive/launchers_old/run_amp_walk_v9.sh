#!/usr/bin/env bash
# ============================================================================
# V9(MVP) 的**从零**训练启动器（跑在云端）
# ============================================================================
# 为什么单独一个文件：训练脚本 `cloud_train_amp_walk_v3.sh` 的 `$5 load_run`
# **非空即触发 --resume**，而 V9 换了观测维度（626→378）**必须从零**，于是需要传两个
# **空参数**占位。在 ssh → cloud_run_bg → bash 的多层嵌套里写 `""` 极易被吃掉或
# 变成字面量 ⇒ 把命令固化在这里，外层就只剩一层单引号，彻底消除歧义。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v9.sh' /root/train_amp_walk_v9_4096.log
#
# 手册：
#   $1 num_envs（默认 4096）  $2 iters（默认 10^9 = 无上限）
#   $3 device（默认 cuda:0）  $4 run_name tag（默认 amp_walk_v9_mvp_4096）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v9_mvp_4096}"
TASK="Instinct-Amp-Walk-V9-G1-v0"

# ⚠️ `$5`(load_run) / `$6`(checkpoint) 显式传空 ⇒ 脚本内 RESUME_FLAGS 为空 ⇒ 从零
exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "" "" "$TASK"
