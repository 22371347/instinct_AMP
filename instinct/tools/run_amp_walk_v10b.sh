#!/usr/bin/env bash
# ============================================================================
# V10b 的**从零**训练启动器（跑在云端）：V10 + 数据集收敛到 2 clip + 三档指令
# ============================================================================
# V10b = V10（探索 σ0.6/上限0.7/ent0.005 + 数据层镜像增广 + 交替支撑约束 21 项）
#        之上按用户指令做两处收敛：
#   ① 数据集 17 clip → **2 clip**：
#        walk_forward_loop_002__A022（直行前进）
#        arc_walk_left_loop_001__A029（弧线/转向）
#      ⚠️ 之前实际生效的一直是 17-clip 那份（文档写的是 1 clip，**文档与实际不符**），
#         17 种形态把"走"这一模态的密度摊薄了。
#   ② 指令分档明确为三档（用户指定的指令集）：
#        prob_low=0.30  静止/低速（‖v‖≤0.1、wz=0，其中 30% 精确 (0,0,0)）
#        prob_lin=0.35  纯前进（vx∈(0.4,0.7)、vy=wz=0）
#        余 0.35        前向斜向/弧线（vx∈(0,0.7)、vy∈±0.35、wz∈±0.4）
#        并**删掉** prob_side / prob_back（指令集里没有，参考集里也没有对应 clip）
#
# ③ 同时装上"门控口径"探针（回答用户问题 2）：
#    桥接层（切片）  : `Mean step cmd_slice_vx / cmd_slice_vy / cmd_slice_wz / cmd_slice_speed`
#    环境层（真值）  : `Step/cmd_speed_mean` / `Step/cmd_low_frac_true`
#    ⇒ 两者吻合 ⇒ `disable_style_on_stand`（0.1 以下 xy 速度不响应）实现正确；
#      不吻合    ⇒ `_command_slice` 错位，必须修（也会解释 action_abs_max=11.8 的怪值）。
#
# ⚠️ 不要与其它训练并行（CPU 瓶颈，会两边减半）。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v10b.sh' /root/train_amp_walk_v10b_4096.log
#
# 手册：$1 num_envs（默认 4096）  $2 iters（默认 10^9 = 无上限）
#       $3 device（默认 cuda:0）  $4 run_name tag（默认 amp_walk_v10b_mvp_4096）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v10b_mvp_4096}"
TASK="Instinct-Amp-Walk-V10b-G1-v0"

# ⚠️ `$5`(load_run) / `$6`(checkpoint) 显式传空 ⇒ 从零
exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "" "" "$TASK"
