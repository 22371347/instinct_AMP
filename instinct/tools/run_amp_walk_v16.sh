#!/usr/bin/env bash
# ============================================================================
# V16 启动器（跑在云端）：**帧数消融 —— 判别器输入 2 帧(134) → 10 帧(670)**
# ============================================================================
# V16 = V15 **逐字沿用**（同环境/同奖励/同 λ=0.75、c=1.0 ⇒ 有效风格权重 0.25）：
#       唯一差异 = 判别器输入帧数（2 → 10）
#       起点同样热启动自 **V11 的 model_9000.pt**（与 V15 同一个 ckpt）
#       判别器两边都**从零重学**（都带 ckpt_manipulator 丢弃旧判别器）
#
# 为什么这么设计：V11（10 帧，会走路）与 V15（2 帧，10k 轮退化）在 AMP 侧的差异只有
#   两项 —— 帧数、判别器是否预训练。本脚本把后者对齐（都丢弃），只留帧数做变量。
#   ⚠️ V16 的 670 维输入与 V11 ckpt 里的判别器**同形**，"不丢"也能载入，但那会引入
#      第二个变量 ⇒ 故意保留 ckpt_manipulator。
#
# 判读（≥500 轮，用 cloud_amp_compare.py 按窗口并排 V15/V16）：
#   * `Step/amp_d` / `Step/amp_style_reward` 的**斜率**：10 帧是否让判别器"重新说话"；
#   * **真痛点**：`Step/tilt_deg_mean` / `tilt_p95_deg`（V15 从 23.5° 退到 30~35° 后仰）、
#     `Episode_Reward/flat_orientation_l2`、`action_rate_l2`、`feet_air_time`；
#   * 判别器链路：`Loss/discriminator_loss` vs `Loss/discriminator_weight_decay`
#     （V15 实测 0.39 vs 74 ⇒ 判别器被权重衰减支配，与帧数无关，预期仍如此）。
#
# ⚠️ 不要和其它训练同时跑。
# 用法：bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v16.sh' /root/train_amp_walk_v16_4096.log
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v16_4096}"
#: 默认热启动来源：V11 的 run 目录（**与 V15 完全同一个起点**）
LOAD_RUN="${5:-20260930_093037_G1AmpWalkV11MVPPlane_obs378_rew21_airtime8_ar0.06_bigstep_fromV10b7500_amp_walk_v11_4096_from20260929_232408}"
CKPT="${6:-model_9000.pt}"
TASK="Instinct-Amp-Walk-V16-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
