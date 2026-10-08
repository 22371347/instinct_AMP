#!/usr/bin/env bash
# ============================================================================
# V19a 启动器（跑在云端）：
#   **让判别器"敢说话"**（AMP 三件套 + 风格奖励上端限幅）+ 步态目标带**消死区**
# ============================================================================
# 基准 = V18（10 帧 670 维判别器 / `log_quad` + `wasserstein` / λ0.75 / c1.0 /
#               5 clip 全向指令 / `flat −2` / `action_rate −0.10` / 三个跟踪项不动）。
#
# 本版**只改 5 个量**（算法 4 + 环境 1）：
#   ① `discriminator_gradient_penalty_coef` 5.0 → **1.0**
#      （V18 实测：它占判别器损失 **51%**（5×0.0419=0.2095 vs 总 0.4094），
#        且 `tolerance=0` ⇒ 惩罚任何输入梯度；实测 `mean‖∂d/∂x‖²=0.042`
#        ⇒ 判别器被练成极平的函数 ⇒ 风格项**没有梯度**，这是头号嫌疑）；
#   ② `discriminator_weight_decay_coef` 3.0e-4 → **1.0e-5**（占 8% → 0.3%）；
#      ⚠️ 更正：此前"wd≈105 ≫ loss≈0.1"是把**未乘系数**的 Σw² 与已乘系数的 loss 对比，
#         属误判；乘系数后 wd 只占 8%，**不是**主因。
#   ③ `discriminator_logit_weight_decay_coef` 0.04 → **0.0**
#      （该键真实系数是 `logit_weight_decay_coef`，**不存在** ⇒ 落回默认 **1.0**，
#        配置里的 0.04 **从未生效**；专罚输出层：实测 `‖w‖₂=0.125` ⇒ d 只能到 ±0.62）；
#   ④ `amp_reward_eps` 1e-6 → **0.05**（`−log(1−D)` 上界 13.8 → 3.0：
#      判别器修好后 d 会向 ±1 靠，不限幅会让风格项淹没任务项）；
#   ⑤ env 侧 `feet_air_time.soft` 0.3 → **0.4**
#      （V18 实测 `in_band_frac` **恒 0**、单支撑仅 0.009~0.011 s ⇒ `soft=0.3` 时
#        `d < 0.1 s` 处 score 与梯度**同时恒为 0** ⇒ 步态项从头到尾没工作，"小跳"无人定价）。
#
# **不动**：h=10 / stride=1（用户指定）、`log_quad` 形式、λ0.75、c1.0、`MSELoss`(±1)、
#           AdamW lr1e-4、5 clip 数据集、四档全向指令、三个跟踪项 3.0/0.8/1.0。
#
# 起点：**热启动 V18 最新 checkpoint，但丢掉判别器**（`drop_amp_discriminator`）
#   ⇒ actor/critic/优化器/normalizer 续用（V18@2k 已会全向走），判别器按新口径重学。
#
# 早期监测（前 ~1k 轮）：
#   * `Step/amp_d`：应从 **±0.62 展宽**（理想走向 ±1）—— 这是改判据的**主指标**；
#   * `Loss/discriminator_gradient_penalty`：应从 0.042 降下来/或占比明显下降；
#   * `Step/amp_style_std`：应显著 > V18 的 0.17（"排序信号"变强）；
#   * `Loss/discriminator_loss`：短期会**上升**（判别器终于开始分辨）—— 正常，不是坏兆头；
#   * `Step/air_time_single_stance` / `air_time_in_band_frac` / `Episode_Reward/feet_air_time`：
#     ⭐ V18 分别是 0.011 s / **0.0000** / 0.0011 ⇒ 现在应**都开始 > 0**；
#   * 守住 `Episode_Reward/track_lin_vel_xy_exp`（V18 ≈1.1~1.2/步）与 `Mean reward`；
#   * `Step/tilt_deg_mean`（V18 ≈8°）不应恶化。
#
# 用法（云端）：
#   pgrep -af 'train[.]py'        # 必须为空
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v19a.sh' /root/train_amp_walk_v19a_4096.log
#
# 手册：$1 num_envs（默认 4096） $2 iters（默认 10^9） $3 device $4 tag
#       $5 load_run（默认空 = 从零） $6 checkpoint（默认空）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v19a_warmV18_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V19a-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
