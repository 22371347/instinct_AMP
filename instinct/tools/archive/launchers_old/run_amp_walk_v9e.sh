#!/usr/bin/env bash
# ============================================================================
# V9e 的**从零**训练启动器（跑在云端）
# ============================================================================
# V9e = V9d（镜像损失 ON + v9c 的 σ/AMP）+ 两项：
#   ① 复原 `feet_air_time`（奖励集 15 → 16 项）：奖励"恰好单脚支撑"的持续时间，
#      对称跳跃（双脚同时着地/离地）拿 0 分 ⇒ 直接打击"原地跳"退化解
#   ② 原地（精确 0）指令占比减半：`zero_frac_in_low` 0.6 → 0.3（18% → 9%）
#      ⇒ 前向指令覆盖 ~70% → ~79%
#
# ⏱️ 判读建议：≥3000 轮再看步态；关注 `Episode_Reward/feet_air_time` 是否 >0
#    （>0 = 真的在单脚支撑交替）、`vx_*_vxy_rmse`、`base_contact`。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v9e.sh' /root/train_amp_walk_v9e_4096.log
#
# 手册：
#   $1 num_envs（默认 4096）  $2 iters（默认 10^9 = 无上限）
#   $3 device（默认 cuda:0）  $4 run_name tag（默认 amp_walk_v9e_mvp_4096）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v9e_mvp_4096}"
TASK="Instinct-Amp-Walk-V9e-G1-v0"

# ⚠️ `$5`(load_run) / `$6`(checkpoint) 显式传空 ⇒ 从零
exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "" "" "$TASK"
