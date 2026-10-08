#!/usr/bin/env bash
# ============================================================================
# V17 启动器（跑在云端）：**从零**学人类步行风格（基准 V16，唯一变量：直立惩罚 ×2）
# ============================================================================
# V17 = V16（10 帧判别器 670 维 / 21 项奖励 / 三档指令 / 2-clip / 镜像增广+损失 /
#            限位软边距 / `track_default_pose_on_stand −0.3` / λ0.75 c1.0）
#        + ① `flat_orientation_l2` 权重 **−1 → −2**
#        + ② `feet_air_time` 权重 **8.0 → 4.0（减半）**
#        + ③ **从零开始**（V16 是热启动 V11 `model_9000.pt`）
#
# 动机：V16 的倾角仍从 22.7°（起点）爬到 28.0°，而 `flat_orientation_l2 = sin²θ` 在 28° 只收
#   0.127/步（对比 `track_lin_vel_xy` +1.45/步）⇒ 定价过低 ⇒ ×2。
#   同时把反单脚的 `feet_air_time` 减半，降低“一脚踩久+另一脚抬着”的刷分收益。
#
# ⚠️ 早期监控（权重减半后健康区间也减半）：
#   `Episode_Reward/feet_air_time`：健康 **0.05~0.15**，**≥0.3 即疑似单脚跳**；
#   `Episode_Reward/feet_swing_time_penalty` 明显更负、`Step/tilt_deg_mean` 高且不降 ⇒ 止损。
#
# 用法（云端）：
#   pgrep -af 'train[.]py'        # 必须为空（CPU 只有 14 核）
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v17.sh' /root/train_amp_walk_v17_4096.log
#
# 手册：$1 num_envs（默认 4096） $2 iters（默认 10^9） $3 device $4 tag
#       $5 load_run（**默认空 = 从零**；要热启动自 V11/V16 就显式传，两者判别器都是 670 维、可直载）
#       $6 checkpoint（默认空）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v17_scratch_4096}"
#: ⚠️ 默认**从零**（不 resume）。用 `${5:-}` 而非 `${5-默认}`：后者会把显式空串当"未设置"。
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V17-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
