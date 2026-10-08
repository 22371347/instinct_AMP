#!/usr/bin/env bash
# ============================================================================
# V23 启动器（跑在云端）：白化 AMP + 进度判据 + 逐关节死区/权重
# ============================================================================
# 基线 = V21（刚体 195 维 × 8 帧 = 1560、V19d 的奖励/指令/数据集）。相对 V21 三处改动：
#   ① `feet_air_time`：删掉"实测瞬时幅值"门（三代关闭率 <1%），改用**进度判据**
#      （方向投影 + EMA 低通）；负价条件 `stand` → `no_progress`。
#   ② `track_default_pose_on_stand`：逐关节死区 + 内部权重（按 Σw 归一）+ weight 0.5→0.8。
#   ③ 判别器输入白化（expert-only 统计）+ GP 移到归一化空间（= V22）。
#
# 起点：**从零**（判别器多了 normalizer buffer、且输入分布定义变了）。
#
# 监测（对照 V19c/V19d/V20/V21）：
#   * 启动横幅 `Linear(in_features=1560)`；`Learning iteration 0/`；无 `[ckpt-manipulator]`
#   * ⭐ `Step/air_time_progress_frac`：应随训练**上升**（在走时才有分）；
#   * ⭐ `Step/air_time_proj_ratio` = EMA/‖v_cmd‖：>0.35 才算达标（κ 标定用）；
#   * ⭐ `Step/air_time_lift_frac` / `hop_frac`：抬脚率与小跳率，目标下降；
#   * ⭐ `Step/stand_pose_pos_cost` / `vel_cost`（新口径）与 `stand_pose_{pos,vel}_err_{legs,arms,waist}`：
#     分组贡献 —— 验证"腿的定价份额"是否真的上来了（V19d 实测腿仅 35%）；
#   * `Step/amp_style_std` 应明显高于过去三代（0.09~0.14，判别器饱和），
#     `Loss/discriminator_loss` 不应 2000 轮就塌到 0.003；
#   * `Episode_Reward/track_lin_vel_xy_exp` 守住 ≈1.0（防"任务被压住"）。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v23.sh' /root/train_amp_walk_v23_4096.log
#
# 手册：$1 num_envs $2 iters $3 device $4 tag  $5 load_run（空=从零） $6 checkpoint
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v23_scratch_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V23-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
