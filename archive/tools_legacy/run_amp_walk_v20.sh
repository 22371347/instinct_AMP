#!/usr/bin/env bash
# ============================================================================
# V20 启动器（跑在云端）：**判别器输入换成"刚体空间 195 维"（照抄 AMP_mjlab），2 帧，从零**
# ============================================================================
# 基准 = V19d（奖励/指令/数据集/AMP 奖励口径**逐字不动**）。
# 本轮唯一改动：`amp_policy` / `amp_reference` 两组观测
#   关节空间 5 项（67 维/帧） → **刚体空间 4 项**（13 体 × 15 = **195 维/帧**）：
#     body_pos_b      (3/体)  相对 anchor(`torso_link`) 的位置，anchor 系下
#     body_ori_b      (6/体)  相对姿态，旋转矩阵**前两列**（连续 6D）
#     body_lin_vel_b  (3/体)  线速度 → 自身刚体系
#     body_ang_vel_b  (3/体)  角速度 → 自身刚体系
#   帧数 10 → **2**（论文式 (s_t, s_{t+1})）⇒ 判别器输入 **670 → 390**。
#
# ⚠️ 为什么这是一次"新维度"的消融：本仓库 V1~V19d 的 AMP 状态**从来只变过帧数**
#    （10 / 2），项名与内容一字未改；`body_pos_b` 这类字符串在 `.py` 里 0 次命中。
#    刚体位姿 = 关节角的非线性函数（FK）、刚体速度 = J(q)·q̇ 的线性函数（雅可比）
#    ⇒ 判别器不必自己学 FK。离线实测：67 维状态在 h=1~2 时 AUC≈0.5（完全不可分）。
#
# 起点：**从零**（输入维度变了，判别器 `Linear(390,1024)` 与旧权重形状不兼容）。
#
# 早期监测（对照 V19d 基线）：
#   * 启动横幅 `Linear(in_features=390)` ⇔ 表征换对了；`Learning iteration 0/` ⇔ 确实从零。
#   * `Loss/discriminator_loss`：应比 V12（2 帧 × 67 维 ⇒ 判别器"不敢说话"）更容易下降。
#   * `Step/amp_d` / `amp_d_ref`：应尽快张开到 ±1 附近（V19a 从 ±0.62 → ±0.88 用了 30 轮）。
#   * `Step/amp_style_*`、`Episode_Reward/feet_air_time`、
#     `Step/air_time_{single_stance,in_band_frac,hop_frac}`：步态质量主指标。
#   * `Step/air_time_stand_step_frac`：站立原地踏步（V19d 是**为此**引入的负罚）。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v20.sh' /root/train_amp_walk_v20_4096.log
#
# 手册：$1 num_envs $2 iters $3 device $4 tag  $5 load_run（空=从零） $6 checkpoint
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v20_scratch_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V20-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
