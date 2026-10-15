#!/usr/bin/env bash
# ============================================================================
# V21 启动器（跑在云端）：**刚体空间 195 维 × 8 帧 = 1560**（在 V20 基础上只改帧数）
# ============================================================================
# 基准 = V19d 的奖励/指令/数据集（**逐字不动**）+ V20 的刚体表征（13 体 × 15）。
# 本轮唯一改动：判别器输入帧数 h 2 → **8** ⇒ 390 → **1560**。
#
# 依据（离线研究 v2，`RESULTS_amp_frames_body_vs_joint.md`，同协议扫 h）：
#   h=2（V20）：AUC 0.625  ⇒ **不够**（V20 因此退回"高频小跳/原地踏步"模态、MuJoCo 不走）
#   h=4        ：AUC 0.740
#   h=6        ：AUC 0.853
#   ⭐ h=8       ：AUC **0.890**、style_std 0.0791、输入维 1560  ← 本轮采用
#   h=10       ：AUC 0.910、style_std 0.0988、输入维 1950（多 25% 算力，暂不取）
#
# 起点：**从零**（`Linear(1560, 1024)` 与 V20 的 390 / V19x 的 670 都不兼容）。
#
# 监测（对照 V19b/V19d 基线）：
#   * 启动横幅 `Linear(in_features=1560)` ⇔ 表征+帧数都换对了；`Learning iteration 0/` ⇔ 从零；
#   * `Step/amp_style_std`：**本轮的核心里程碑** —— V20 是 0.08、V19b 是 0.136；
#     离线预测"排序信号大幅增强"（body195@h8 的离线 style_std 是 legacy@h10 的 5.3×）；
#   * `Loss/discriminator_loss`：不应像 V20 那样 2000 轮就塌到 0.003（那是"没压力可学"）；
#   * 步态主指标：`Step/air_time_{in_band_frac, single_stance, hop_frac}`、
#     `Episode_Reward/feet_air_time` —— 目标是把 `hop_frac` 从 V20 的 0.92 压到 V19b 的 0.73 以下；
#   * `Episode_Reward/track_lin_vel_xy_exp` 守住 ≈1.0（4000 轮时 V19b/V20 都在 1.07~1.09）。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v21.sh' /root/train_amp_walk_v21_4096.log
#
# 手册：$1 num_envs $2 iters $3 device $4 tag  $5 load_run（空=从零） $6 checkpoint
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v21_scratch_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V21-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
