#!/usr/bin/env bash
# ============================================================================
# V14 启动器（跑在云端）：**V12 退化修正版** + 腰部限位软边距惩罚
# ============================================================================
# V14 = V12（2 帧判别器 134 维 / 三档指令 / 2-clip / 镜像增广+损失 / σ上限0.6 / λ0.5 c0.5）
#   ① 回退 `track_default_pose_on_stand` 权重：−1.0 → **−0.3**
#      （V11→V12 同口径对照：任务回报 27.7→9.28/回合，其中 −10.2/回合直接来自这项）
#   ② 限位项换成**软边距**版：踝 → **全关节**，距限位 0.15 rad 内就开始二次收费，权重 −1.5
#      （治"`waist_pitch_joint` 长期顶在 ±0.52"；此前腰/髋/膝**根本没被罚**）
#
# ⭐ 起点：**从零**（默认；用户 2026-09-30 决定）。要热启动需显式传 run 目录 + ckpt
#    （若要从 V11 的 670 维判别器 ckpt 热启动，`ckpt_manipulator` 会自动丢掉旧判别器）。
#
# ⚠️ 不要和任何其它训练同时跑（CPU 是瓶颈）。
#
# 判读（≥500 轮）：
#   ① 修正：`Mean reward` 回到 **27 附近**、`feet_air_time` 回到 **0.15+/s**、
#           `action_rate_l2` 回到 −0.45/s 量级、`vx_mid RMSE` 回到 0.24 附近；
#   ② 限位：看 `Step/waist_pitch_abs_mean`（新增探针）是否从 ~0.52 回落，
#           回放里"上身后仰 / 腰卡住"是否改善；
#   ③ 风格：`amp_d` 应保持 V12 的水平（≈ −0.4）——这是 2 帧判别器的收益，不该丢。
#
# 用法（云端）：
#   pgrep -af 'train[.]py'        # 必须为空
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v14.sh' /root/train_amp_walk_v14_4096.log
#
# 手册：$1 num_envs（默认 4096）  $2 iters（默认 10^9）  $3 device  $4 tag
#       $5 load_run（默认 V11 的 run 目录）  $6 checkpoint（默认 model_9000.pt）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v14_scratch_4096}"
#: ⚠️ 默认**从零**（不 resume）—— 2026-09-30 用户决定：V12 已退化 ⇒ 停掉后从头重训。
#:    要热启动就**显式**传 $5（run 目录）与 $6（ckpt），例：
#:      bash /root/run_amp_walk_v14.sh 4096 1000000000 cuda:0 <tag> <V11_run_dir> model_9000.pt
#:    （用 `${5:-}` 而不是 `${5-默认}` 是有意的：后者会把显式传入的空串当成“未设置”。）
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V14-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
