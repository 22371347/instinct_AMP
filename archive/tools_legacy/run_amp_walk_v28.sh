#!/usr/bin/env bash
# ============================================================================
# V28 启动器（跑在云端）：**放大 AMP 风格项权重**（coef 1.0 → 4.0）
# ============================================================================
# 起点：**V27 最新 ckpt 热启动**（策略 / PPO 优化器 / 观测 normalizer 保留；
#       判别器与其优化器用 `drop_amp_discriminator` 丢掉重学）。
# 其余一律不动：去白化、σ=0.20 + 300 轮退火、奖励表（21 项）、AMP 观测 13body×8frame、
#               `action_scale=1.0`、λ=style_lerp=0.75。
#
# 为什么（诊断）：
#   `r_total = λ·r_task + (1−λ)·coef·style`，λ=0.75、coef=1.0
#   ⇒ **风格项有效权重只有 0.25**；而 V27 实测 `amp_style_std = 0.058`（风格奖励的
#   跨状态方差 = 唯一能引导策略的排序信号）⇒ 风格项作用量 ≈ 0.25×0.058 = 0.0145/步，
#   比任务项（track_lin_vel +1.2 / feet_air_time +1.1 / 罚项 O(0.1~1)）小 10~70 倍
#   ⇒ **判别器再准也推不动步态**（与"V27 练到 10000 轮无类人风格"吻合）。
#   本轮把 coef 提到 4.0 ⇒ 风格项权重 1.0（4×），作用量 0.058/步（与任务项同量级）。
#   只动 coef 不动 λ ⇒ 任务项权重仍 0.75，**不拿已学到的步态换风格**。
#
# 监测（TB 新标签）：
#   * `Step/amp_style_std`：期望**仍** ≈0.05~0.08（信号本身没变），
#     但风格项的量级 `Episode_Reward/...`/`Train/amp_*` 应 ×4；
#   * **核心可观测是行为**：`air_time_single_stance`↑、`air_time_lift_frac`↑、
#     `air_time_hop_frac`↓（<0.72）⇒ 说明风格项真的开始推动步态；
#   * 风险闸门：`Mean reward` 不得从 ~+45 掉到 <+20；`Step/hf_acc_all` 不得明显抬头
#     （若是 ⇒ 判定"风格项在带坏步态" ⇒ 回退 coef 到 2.0）。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v28.sh' /root/train_amp_walk_v28_4096.log
#
# 手册：$1 num_envs $2 iters $3 device $4 tag $5 load_run（空=从零） $6 checkpoint
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v28_ampcoef4_dropdisc_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V28-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
