#!/usr/bin/env bash
# ============================================================================
# V30 启动器（跑在云端）：**判别器 = AMP_mjlab 原始配方**
# ============================================================================
# 起点：**V27 最新 ckpt 热启动**（策略 / PPO 优化器 / 观测 normalizer 保留；
#       判别器与其优化器用 `drop_amp_discriminator` 丢掉重学 —— 判别器口径变了，必须重学）。
#
# V30 = V29 逐字不动（输入 390 / `bounded_quad` / 去白化 / σ=0.20+退火 / 奖励表 / λ=0.75）
#       + **判别器侧四件事（全部对齐 mjlab）**：
#   ① `discriminator_gradient_penalty_coef` 1.0 → **10.0**（mjlab λ=10）
#   ② GP **只在 expert 侧**求梯度（`discriminator_gp_expert_only=True`）
#      —— 旧口径把 actor 侧也拉进来，等于额外要求"对策略分布也不要敏感"（自相矛盾）
#   ③ wd 从 loss 项（1e-5）→ **优化器级分组**：trunk **1e-3** / head **1e-2**
#   ④ 专家批从"活体影子机器人" → **离线运动数据集 i.i.d. 抽帧**（`/root/amp_expert_states.pt`）
#   ⑤ 风格强度**刻意不动**：`discriminator_reward_coef` 保持 **1.0**（有效权重 0.25）
#
# 为什么（V29 的读数）：判别器 **1 个迭代（20 梯度步）** 内把 MSE 压到 0.035（V27 稳态 0.0014）
#   ⇒ 风格奖励变常数 ⇒ 风格项无梯度。"拟合到完美"是核心症状，而 GP 是唯一**直接压制**
#   它的机制（惩罚 ‖∂d/∂x‖² ⇒ 逼 d 平滑 ⇒ 不会在整个 actor 云上恒为 −1）。
#   参考：V11 是唯一未饱和的版本（`disc_loss` 0.1214），也是走得最久最稳、风格最好的一版。
#
# 监测（TB，**只看判别器是否活过来**）：
#   * `Loss/discriminator_loss`：0.0014~0.035 → **期望 > 0.05**
#   * `Step/amp_d` / `amp_d_ref`：−0.92 / +0.87（贴边）→ 期望 **−0.2 ~ −0.7** / 不恒 +1
#   * `Step/amp_style_std`：0.058 → **期望 > 0.10**
#   * `Step/amp_style_zero_frac`：0.36→1 → **期望 < 0.3**
#   * `Loss/discriminator_gradient_penalty`：0.017 → 期望显著上升（λ=10）
#   * 不退化闸门：`Mean reward ≥ 30`、`episode length ≥ 300`、`action_abs_mean ≤ 0.6`
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v30.sh' /root/train_amp_walk_v30_4096.log
#
# 手册：$1 num_envs $2 iters $3 device $4 tag $5 load_run（空=从零） $6 checkpoint
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v30_mjlab_disc_recipe_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V30-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
