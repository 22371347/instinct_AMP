#!/usr/bin/env bash
# ============================================================================
# V19d 启动器（跑在云端）：**站立档改成负价**（把“原地踏步”从 0 价变负价）
# ============================================================================
# 基准 = V19c（运动门控 / ar −0.06 / 站立姿态 −0.5 / AMP 侧与 V19b 相同）。
# 本轮唯一改动：`feet_air_time` 新增 `stand_penalty = 0.5` ⇒
#   `stand = 1[‖v_cmd,xy‖+|ωz| ≤ 0.1]` 时 `reward −= 0.5·min(d/good_lo,1)·1[单支撑]`
#   罚额（×weight 4.0）：0.08 s → −0.40/步、0.15 s → −0.75、0.30 s → −1.50、≥0.40 s → −2.00。
# 新探针：`Step/air_time_stand_step_frac`（站立+真抬脚占比，**主指标** → 0）、
#          `Step/air_time_stand_pen_mean`、`Step/stand_pose_{pos_err,vel_err,active}`
#          （最后三个直接回答“`track_default_pose_on_stand` 的速度阻尼项在不在工作”）。
#
# 起点：**热启动 V19c 最新 checkpoint，且保留判别器**（AMP 口径逐字相同）。
#
# 早期监测：
#   * ⭐ `Step/air_time_stand_step_frac` 应快速下降（0.0x → 接近 0）；
#   * `Step/stand_pose_pos_err` / `stand_pose_vel_err`：两者应随训练**下降**；
#     （特别盯 `vel_err`：它跌了才证明“关节速度阻尼”在起作用）
#   * `Episode_Reward/feet_air_time` 会因负罚而变小（部分站立步被扣）⇒ 判据用
#     `air_time_in_band_frac` / `air_time_hop_frac` / `track_lin_vel_xy_exp`；
#   * 不能退化为“不敢抬脚”：`air_time_in_band_frac` 不应大跌（V19c ≈ 0.52）。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v19d.sh' /root/train_amp_walk_v19d_4096.log
#
# 手册：$1 num_envs $2 iters $3 device $4 tag  $5 load_run（空=从零） $6 checkpoint
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v19d_warmV19c_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V19d-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"