#!/usr/bin/env bash
# ============================================================================
# V11 的**续训**启动器（跑在云端）：从 V10b 的走路 checkpoint 热启动
# ============================================================================
# V11 = V10b（2-clip / 三档指令 / 镜像增广 / 21 项奖励）之上做"步态形状"三改：
#   ① feet_air_time 2.0→8.0、threshold 0.5→0.6   （小碎步≈0、大步伐≈满分）
#   ② action_rate_l2 −0.02→−0.06                  （高频碎步的物理特征）
#   ③ dof_acc_l2 −2.5e-7→−1.0e-6                  （同上）
#   + 风格压力 (1−λ)c 0.0625→0.25（λ0.75→0.5, c0.25→0.5）
#   + σ 上限 0.7→0.6（用户指定）
#
# ⚠️ 续训注意：`init_noise_std` 不生效（σ 用 checkpoint 里学到的 0.58）；
#    回报会先掉再涨（critic 要适应新奖励语义）；首轮指标不可信。
#
# 判读：`feet_air_time` >0.15/s、`action_rate_l2` 回到 −0.1/s 量级、
#       `dof_acc_l2` →−0.015、`amp_d` 向 0 靠（−0.78→−0.4）、`vx_*_rmse` 不退化。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v11.sh' /root/train_amp_walk_v11_4096.log
#
# 手册：$1 num_envs（默认 4096）  $2 iters（默认 10^9 = 无上限）
#       $3 device（默认 cuda:0）  $4 run_name tag（默认 amp_walk_v11_4096）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v11_4096}"
TASK="Instinct-Amp-Walk-V11-G1-v0"

#: 热启动来源：V10b 跑到 7930 轮的那次 run（最新 ckpt = model_7500.pt）
LOAD_RUN="20260929_232408_G1AmpWalkV10bMVPPlane_obs378_rew21_explore_aug_cmd3_clip2_amp_walk_v10b_mvp_4096"
CKPT="model_7500.pt"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
