#!/usr/bin/env bash
# ============================================================================
# V9j 的**从零**训练启动器（跑在云端）：堵住"单脚吊腿"漏洞，强制交替支撑
# ============================================================================
# V9j = V9h 逐字不动 + **新增 1 项** `feet_swing_time_penalty`（20 → 21 项）：
#   cost = Σ_f clamp(air_time_f − 0.6 s, 0) · 1[‖cmd_xy‖ > 0.1]，权重 −0.5
#
# 为什么（V9h@4770 实测）：`lin_vel_z_l2` 只罚竖直速度（实测 −0.0034/s ⇒ 几乎没有竖直弹跳），
# 而 `feet_air_time` 只要"恰好一只脚着地"就持续给满（不区分踩了 0.3 s 还是 10 s）
# ⇒ "单脚侧身跳着走"成了既躲开抗跳项、又拿满步态项的最优解；
# 镜像损失拦不住它（0.0368 很小，因为单侧步态的镜像态从未被访问，约束在 off-distribution 上被满足）。
#
# ⚠️ 不要与其它训练并行（CPU 瓶颈，会两边减半）。
#
# 判读：新诊断 `Mean step feet_swing_excess`（单脚平均超长腾空秒数）走路应 ≈0；
#       `Episode_Reward/feet_air_time` 冲上 >0.1/s；`vx_high` RMSE 保持 <0.25；
#       `base_contact` < 0.5；`Mean reward` 若真交替则明显 > 46.6。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v9j.sh' /root/train_amp_walk_v9j_4096.log
#
# 手册：$1 num_envs（默认 4096）  $2 iters（默认 10^9 = 无上限）
#       $3 device（默认 cuda:0）  $4 run_name tag（默认 amp_walk_v9j_mvp_4096）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v9j_mvp_4096}"
TASK="Instinct-Amp-Walk-V9j-G1-v0"

# ⚠️ `$5`(load_run) / `$6`(checkpoint) 显式传空 ⇒ 从零
exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "" "" "$TASK"
