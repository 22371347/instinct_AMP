#!/usr/bin/env bash
# ============================================================================
# V9f 的**从零**训练启动器（跑在云端）
# ============================================================================
# V9f = V9e + 两处"解放"改动（针对用户假设：学习率偏低 / 探索幅度不足）
#   ① `algorithm.schedule`: "adaptive" → "fixed"（真实 LR 稳定在 1e-3）
#   ② `policy.init_noise_std` 0.2 → 0.3、`clip_max_std` 0.3 → 0.4（探索幅度 ↑）
# 诊断：本版起日志里会有 `Mean step effective_lr`（真实学习率）与
#       `Mean step action_abs_mean/action_abs_max`（动作幅度）。
#
# 预期判读：`action_abs_mean` 升到 0.2~0.4 且 `feet_air_time` > 0 ⇒ 假设成立；
#           若 `action_abs_mean` 仍 ≲0.1 ⇒ 回到"奖励形状"（压三个"站着白拿"项）。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v9f.sh' /root/train_amp_walk_v9f_4096.log
#
# 手册：
#   $1 num_envs（默认 4096）  $2 iters（默认 10^9 = 无上限）
#   $3 device（默认 cuda:0）  $4 run_name tag（默认 amp_walk_v9f_mvp_4096）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v9f_mvp_4096}"
TASK="Instinct-Amp-Walk-V9f-G1-v0"

# ⚠️ `$5`(load_run) / `$6`(checkpoint) 显式传空 ⇒ 从零
exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "" "" "$TASK"
