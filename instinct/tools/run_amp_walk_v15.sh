#!/usr/bin/env bash
# ============================================================================
# V15 启动器（跑在云端）：**风格分量 ×4 实验**（用户 2026-09-30 指定）
# ============================================================================
# V15 = V14 的环境/奖励（2 帧判别器 134 维 / 三档指令 / 2-clip / 站姿 −0.3 /
#        限位软边距全关节）+ **算法侧只改一个数**：
#        `discriminator_reward_coef` 0.25 → 1.0 ⇒ 风格入账 (1−λ)c：0.0625 → 0.25（×4）
#        （λ=0.75 不动 ⇒ 任务入账仍是 ×0.75）
#
# 为什么这样做：V14 实测 `Step/amp_style_reward` 5400 轮内只动 **1.7%**（`amp_d` 从 450 轮
# 起就钉在 −0.47）⇒ 风格项**饱和/平坦**（不是"占比小"：它 0.0375/步 > 任务净额 0.0314/步）。
# 本实验把风格分量单独放大 4 倍，看 `amp_style_reward`/`amp_d` 的**变化率是否恢复**。
#
# ⭐ 热启动自 **V11 的 model_9000.pt**（会走路的策略；V14 从零已跳成单脚跳，不能从它起步）。
#    判别器 670→134 ⇒ `ckpt_manipulator` 自动丢掉旧判别器。
#
# 判读（≥500 轮）：
#   * `Step/amp_style_reward` 的**斜率**（用 `cloud_reward_accounting.py --trend` 看）
#     显著非零、`amp_d` 从 −0.47 向 0 迁移 ⇒ 风格梯度回来了（实验目的达成）；
#   * 若两者**依然平坦** ⇒ 说明问题不在"权重大小"，而在判别器/优化口径
#     （下一个候选变量：`discriminator_weight_decay_coef` 3e-4 → 0，它现在
#      `Loss/discriminator_weight_decay ≈ 76` 远超 `discriminator_loss ≈ 0.27` ⇒ 判别器被它支配）；
#   * 副作用监控：`feet_air_time` 是否掉、`action_rate_l2` 是否恶化（风格权重大了容易"以任务换风格"）。
#
# ⚠️ 不要和其它训练同时跑。
# 用法：bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v15.sh' /root/train_amp_walk_v15_4096.log
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v15_4096}"
#: 默认热启动来源：V11 的 run 目录（最后一份会走路的策略）
LOAD_RUN="${5:-20260930_093037_G1AmpWalkV11MVPPlane_obs378_rew21_airtime8_ar0.06_bigstep_fromV10b7500_amp_walk_v11_4096_from20260929_232408}"
CKPT="${6:-model_9000.pt}"
TASK="Instinct-Amp-Walk-V15-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
