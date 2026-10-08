#!/usr/bin/env bash
# ============================================================================
# V27 启动器（跑在云端）：**判别器"重启"** = 丢掉已饱和的判别器 + 去掉白化
# ============================================================================
# 起点：**V26 最新 ckpt 热启动**（策略/PPO 优化器保留；判别器与其优化器用
#       `drop_amp_discriminator` 丢掉重学）。σ=0.20 与其 300 轮退火**不变**。
#
# 为什么（用户 2026-10-04 的洞察 + 实测）：
#   * V26 的判别器是**继承 V24** 的（`ckpt_manipulator=None`）⇒ 实测 V26 第一个 log 块
#     （iter 8000）就已 `amp_d=−1.0004` / `d_ref=0.9999` / `style_std=0.0014`
#     ⇒ **V26 的风格项从第 0 轮就是死的**，它的数据只能回答 σ 的问题；
#   * 白化（`EmpiricalNormalization(eps=0.01)` + 统计量**只由 reference 侧**累积）
#     把"策略 vs 参考"的系统性偏移放大 ~100× ⇒ 判别器永久 100% 可分 ⇒ `−log(1−D)`
#     变常数 ⇒ 风格项无梯度（V23/V24/V26 的 `style_std` 0.008/0.003/0.0008）；
#   * 对照：V19b/V20 **不白化**、同源 AMP 观测 ⇒ `style_std` 0.094~0.124（健康）。
#
# 监测：
#   * `Step/amp_style_std`（0.0008 → 期望 >0.03，目标 0.09~0.12）、
#     `Loss/discriminator_loss`（1e-5 → >1e-3）、`amp_d`（−0.998 → >−0.9）、
#     `amp_d_ref`（不再恒为 +1.000）；
#   * 不能退化：`Step/hf_acc_all` 仍 ~45、`Mean reward` 仍 > +35（回合口径）、
#     `air_time_hop_frac` < 0.648、`in_band_frac` > 0.446、`single_stance` > 0.119；
#   * 期望追平/超过 V24 的任务项：`Episode_Reward/track_lin_vel_xy_exp` > 1.412。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v27.sh' /root/train_amp_walk_v27_4096.log
#
# 手册：$1 num_envs $2 iters $3 device $4 tag  $5 load_run（空=从零） $6 checkpoint
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v27_dropdisc_nodrop_white_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V27-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
