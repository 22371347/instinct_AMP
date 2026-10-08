#!/usr/bin/env bash
# ============================================================================
# V10 的**从零**训练启动器（跑在云端）：探索量级 + 数据层镜像增广 + 指令扩展 + 交替约束
# ============================================================================
# V10 = V9j（21 项奖励，含 feet_swing_time_penalty）之上一次上齐 3 处改动：
#   ① 探索量级：init_noise_std 0.3→0.6、clip_max_std 0.4→0.7、entropy 0.0025→0.005、
#      clip_min_std 0.0125→0.03   （参照：不含 AMP 的行走配方 = 1.0 / 无上限 / 0.008）
#   ② 数据层镜像增广 ON：symmetric_augmentation_{joint_mapping,joint_reverse_buf,link_mapping}
#      —— 映射用 `build_joint_mirror(OMNI_LOCO_JOINT_ORDER)` 现算（与镜像损失同一套代码）
#      ⇒ 判别器的 expert 分布变左右对称 ⇒ 单侧步态不再"像人"
#   ③ 指令扩展：(0,0.7) × (−0.35,0.35) × (−0.6,0.6) ⇒ 左前/右前对角 + 转向都要会
#
# ⚠️ 本版放弃单变量归因（目标：尽快拿到可参考的行走策略）。诊断键已留全：
#    `feet_swing_excess` / `Mean action noise std` / `Episode_Reward/dof_pos_limits` /
#    `amp_d` vs `amp_d_ref` / `vx_mid|high RMSE`。
# ⚠️ 启动后**必须看**这一行（增广标签是否可信的唯一在线证据）：
#    `[AMP-WALK INFO] 数据层镜像增广=ON；joint 顺序 vs OMNI_LOCO_JOINT_ORDER：一致`
# ⚠️ 不要与其它训练并行（CPU 瓶颈，会两边减半）。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v10.sh' /root/train_amp_walk_v10_4096.log
#
# 手册：$1 num_envs（默认 4096）  $2 iters（默认 10^9 = 无上限）
#       $3 device（默认 cuda:0）  $4 run_name tag（默认 amp_walk_v10_mvp_4096）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v10_mvp_4096}"
TASK="Instinct-Amp-Walk-V10-G1-v0"

# ⚠️ `$5`(load_run) / `$6`(checkpoint) 显式传空 ⇒ 从零
exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "" "" "$TASK"
