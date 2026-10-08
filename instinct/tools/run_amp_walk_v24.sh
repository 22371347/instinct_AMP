#!/usr/bin/env bash
# ============================================================================
# V24 启动器（跑在云端）：V23 + `dof_acc_l2` **全关节**（权重不变 −1.0e-6）
# ============================================================================
# 为什么：加速度 L2 对频率是 4 次方（a∝A(2πf)² ⇒ 罚 a²∝A²f⁴），对"高频小跳"的选择性
# 是 action_rate_l2（f²）的两个倍频程。AMP_mjlab 的 joint_acc_l2 = −2.5e-7 无 asset_cfg
# ⇒ **全关节**；我方从 V2 起只罚髋/膝（而小跳主要是踝+腰）⇒ 罚不到病灶。
#
# 起点：**从零**（V23 在 2500 轮已确定进入高频小跳模态；从塌陷 ckpt 续训出不来自吸引子）。
#
# 监测：
#   * ⭐ `Step/hf_acc_all / hf_acc_legs / hf_acc_arms / hf_acc_waist`（未乘权重的 mean|ddq|）
#     —— 定位"谁在高频抖"，也是本轮效果的**直接读数**；
#   * `Episode_Reward/dof_acc_l2` 绝对值应变大（覆盖面扩大）；若过大再回落权重；
#   * 步态：`Step/air_time_hop_frac`（目标 <0.9）、`air_time_in_band_frac`（目标 >0.3）、
#     `air_time_single_stance`（目标 >0.04 s）；
#   * 副作用：踝被罚会同时抑制"硬着陆"（好）与"离地/摆腿"（可能过强）⇒ 盯
#     `Step/air_time_lift_frac`、`Episode_Reward/track_lin_vel_xy_exp`（守住 ≈1.0）；
#   * `Policy/mean_noise_std`（σ 仍顶在 0.6 上限；acc 罚可能连带压噪声）。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v24.sh' /root/train_amp_walk_v24_4096.log
#
# 手册：$1 num_envs $2 iters $3 device $4 tag  $5 load_run（空=从零） $6 checkpoint
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v24_scratch_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V24-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
