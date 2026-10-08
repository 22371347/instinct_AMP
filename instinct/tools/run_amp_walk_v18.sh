#!/usr/bin/env bash
# ============================================================================
# V18 启动器（跑在云端）：**步态目标带 + 抖动加大 + 方案 A（1 clip / 前进+停止）**
# ============================================================================
# V18 = V17（从零 / 10 帧判别器 670 维 / `flat_orientation_l2 −2`）
#     + ① `feet_air_time` → **目标时长带**（单支撑 ∈ [0.4,1.0] s 满分、两侧线性衰竭）
#     + ② `action_rate_l2` −0.06 → **−0.10**
#     + ③ `feet_swing_time_penalty` 权重 → **0**（与①上界重叠）
#     + ④ 数据集/指令 = **方案 B**（5 clip：直行/弧线/侧移×2/后退 + 四档全向指令）
#       ⭐ 三个跟踪项（lin_vel 3.0 / ang_vel 0.8 / base_height 1.0）**一律不动**：
#          方案 B 里 ωz 有真实指令 ⇒ `track_ang_vel_z_exp` 是有效激励。
#
# 为什么（V17 实测）：
#   * 旧 `feet_air_time` 的核 = `min(支撑接触时长, 摆动腾空时长) clamp 0.6`（**按秒计费**）
#     ⇒ V17 实测核值只有 **0.005~0.009 s**（单支撑相位仅几毫秒）⇒ 数据上就是"小幅高频跳走"；
#   * 旧形式还允许"单脚一直踩着/吊着"无限刷分（V14 从零就是这么跳起来的）；
#     新形式**上限 1.0、两侧衰竭** ⇒ 两个坑同时堵住（无需再靠压低权重防跳）。
#   * 数据集收敛到 1 个纯前进 clip（AMP_mjlab 教训：模态越多，"走"的密度越稀 ⇒ 判别器失去分辨力）。
#
# 早期监测（前 ~1.5k 轮）：
#   * `Step/air_time_single_stance` ← **新探针，单支撑时长均值**：应从 V17 的 ~0.007 s 升向 0.4~1.0 s；
#   * `Step/air_time_in_band_frac`：合规步数占比，应显著 > 0；
#   * `Episode_Reward/action_rate_l2`：不再单调恶化；
#   * `Loss/discriminator_loss` / `Step/amp_d`：判别器是否"重新有话说"；
#   * 守住 `track_lin_vel_xy_exp`（不能为步态牺牲前进能力）与 `Mean reward`。
#
# 用法（云端）：
#   pgrep -af 'train[.]py'        # 必须为空
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v18.sh' /root/train_amp_walk_v18_4096.log
#
# 手册：$1 num_envs（默认 4096） $2 iters（默认 10^9） $3 device $4 tag
#       $5 load_run（默认空 = 从零） $6 checkpoint（默认空）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v18_scratch_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V18-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
