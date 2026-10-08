#!/usr/bin/env bash
# ============================================================================
# V9i 的**从零**训练启动器（跑在云端）：唯一变量 = 探索量级
# ============================================================================
# V9i = V9h 的奖励集/AMP 设置**逐字不动**，只把"探索"抬到参考配方的量级：
#   init_noise_std 0.3 → 0.5（±17° → ±29°）
#   clip_max_std   0.4 → 0.8
#   entropy_coef   0.0025 → 0.005
#   clip_min_std   0.0125 → 0.025
# 参照：仓库里不含 AMP 的行走配方 `G1OmniLocoPPORunnerCfg`（同机器人、同 action_scale=1.0、
#       同时序与指令契约）用的是 init_noise_std=1.0 + entropy_coef=0.008。
#
# ⚠️ **不要与 V9h 并行跑**：实测训练瓶颈在 CPU，并行会把两边速度都砍半。
#    等 V9h 停掉（或跑到 ~3000 轮拿到结论）之后再启动本脚本。
#
# 判读：`Episode_Reward/feet_air_time` > 0.1/s（真交替支撑）、`vx_high` RMSE 继续降、
#       `base_contact` < 0.5、`amp_d` 是否向 0 靠（探索有没有把策略推向参考动作那片区域）。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v9i.sh' /root/train_amp_walk_v9i_4096.log
#
# 手册：$1 num_envs（默认 4096）  $2 iters（默认 10^9 = 无上限）
#       $3 device（默认 cuda:0）  $4 run_name tag（默认 amp_walk_v9i_mvp_4096）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v9i_mvp_4096}"
TASK="Instinct-Amp-Walk-V9i-G1-v0"

# ⚠️ `$5`(load_run) / `$6`(checkpoint) 显式传空 ⇒ 从零
exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "" "" "$TASK"
