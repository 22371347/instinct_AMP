#!/usr/bin/env bash
# ============================================================================
# V26 启动器（跑在云端）：**V24 全部设置 + σ 上限 0.6→0.20（300 轮退火）**
# ============================================================================
# 实验 B：从 **V24 `model_8000.pt` 热启动**，只把动作噪声 σ 压下来。
#
# 为什么热启动 + 退火：
#   * V24 的 σ 自然平衡在 0.572（cap 0.6 并未 binding）⇒ 噪声税 −5.26/步，
#     把同样的策略从"每步 +4.92"拖到"−0.48"（D4 实测，见 /root/d4_v24.txt）；
#   * Step 0 扫描（/root/sigma_scan_v24.txt）：σ=0.20 时噪声税只剩 −0.58（−10%），
#     且**步态最优**（hop 0.762→0.626、in_band 0.385→0.443、single_stance 0.063→0.138）；
#   * 直接硬夹会让奖励分布整体上移 ~4/步 ⇒ value function 突跳 ⇒ 用 300 轮线性退火。
#
# 监测（新标签）：
#   * `Train/sigma_cap` / `Train/sigma_actual` —— 核对退火是否按 0.57→0.20 走；
#     `Policy/mean_noise_std` 应在 ~300 轮后稳定在 ≈0.20；
#   * `Episode_Reward/action_rate_l2`（−1.24 → ~−0.17）、`dof_acc_l2`（−1.46 → ~−0.16）、
#     `track_default_pose_on_stand`（−1.01 → ~−0.35）、`dof_pos_limits`（−0.42 → ~−0.04）；
#   * `Mean reward`（manager 口径 −0.48 → 期望 ≥ +3.5）；
#   * 步态：`air_time_hop_frac`（<0.65）、`air_time_in_band_frac`（>0.44）、
#     `air_time_single_stance`（>0.12）；`air_time_progress_frac` 应回落到 ~0.73；
#   * 风格项：`Step/amp_style_std`（0.008 → 是否 >0.03 = σ 是不是"杀死风格"的元凶）。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v26.sh' /root/train_amp_walk_v26_4096.log
#
# 手册：$1 num_envs $2 iters $3 device $4 tag  $5 load_run（空=从零） $6 checkpoint
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v26_hot_v24_m8000_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V26-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
