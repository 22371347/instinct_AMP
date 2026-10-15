#!/usr/bin/env bash
# ============================================================================
# V29 启动器（跑在云端）：**判别器向 AMP_mjlab 对齐**（系数回退到 V27）
# ============================================================================
# 起点：**V27 最新 ckpt 热启动**（策略 / PPO 优化器 / 观测 normalizer 保留；
#       判别器与其优化器用 `drop_amp_discriminator` 丢掉重学 —— 输入维度从 1560 变 390，
#       本来就**必须**重学）。
#
# V29 = V27 逐字不动（去白化 / σ=0.20 + 300 轮退火 / 奖励表 21 项 / action_scale=1.0 /
#        AMP 观测定义 13body×15）+ 判别器向 mjlab 对齐三件事：
#   ① 判别器输入 8 帧 1560 → **2 帧 390**（`(s_t, s_{t+1})` 语义）
#   ② 专家批（参考批）从"与策略同相位" → **独立随机重采 (T,B)**（全相位覆盖）
#      —— 单测已验证：旧口径 1536/1536 样本与策略批**同一格**；新口径 0/1536 重合
#   ③ 风格奖励形状 `−log(1−quad)`（∈[0,13.8]）→ **`clamp(1−0.25(d−1)²,0,1)`**（∈[0,1]，
#      `d≤−1` 恰为 0）= mjlab 形状
#   ④ 判别器系数回退：GP 5.0→**1.0**、wd 3e-4→**1.0e-5**、coef 4.0→**1.0**
#      ⇒ **有效风格权重仍是 (1−0.75)×1.0 = 0.25（刻意不动权重）**
#
# 为什么回退：V28 在 iter 10470 **崩了**（Mean reward 14.85→−7.28、episode length 489→2.7、
#   action_abs_mean 0.35→2.6、surrogate_loss 单点 12× 尖峰）。机制是"风格项噪声 ×8.3 +
#   从任务的 16% 变 ~47% + log_quad 无上界产生奖励离群"。"放大 AMP 风格权重"已被证伪两次
#   （V15 无效 / V28 有害）⇒ 改攻"让风格信号有信息"。
#
# 监测（TB）：
#   * ① 启动日志必须是 `Linear(in_features=390)`（V27 是 1560）
#   * ② `amp_d` **不再 20 轮内贴到 ±1**（判别器失去"相位/时序"捷径）
#   * ③ `amp_style_reward ∈ [0,1]`；`amp_style_zero_frac` **不要**长期 = 1.0（=1.0 ⇒ 风格项死）
#   * 不退化：`Mean reward` ≥ 30（V27 同期 34~44）、`Mean episode length` ≥ 300、
#     `action_abs_mean` ≤ 0.6（V27 是 0.354）、`hf_acc_all` ≈44
#   * 行为侧（本轮核心）：`air_time_single_stance` ↑ / `air_time_hop_frac` ↓
#   * 守护：`bash /root/v29_guard.sh` 会在崩坏时自动停训（见 v29_guard.sh 头部）
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v29.sh' /root/train_amp_walk_v29_4096.log
#
# 手册：$1 num_envs $2 iters $3 device $4 tag $5 load_run（空=从零） $6 checkpoint
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v29_mjlabalign_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V29-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
