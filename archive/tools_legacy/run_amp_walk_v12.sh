#!/usr/bin/env bash
# ============================================================================
# V12 启动器（跑在云端）：**AMP 判别器换成论文的两帧 (s_t, s_{t+1})** + 站姿权重
# ============================================================================
# V12 = V11 逐字（21 项奖励 / 三档指令 / 2-clip 数据集 / 镜像增广+损失 / σ 上限 0.6 /
#        λ0.5 c0.5）之上，只动两处：
#   ① AMP 判别器输入 10 帧 → **2 帧**（`amp_policy`/`amp_reference` 670 → 134 维）
#      —— LLP 原论文的状态转移输入；10 帧下判别器只看"姿态轮廓/抖动统计"就能分辨，
#         没有压力学"单步转移像不像人"，对应我们 `amp_d` 长期 −0.56~−0.78 的现象。
#   ② `track_default_pose_on_stand` −0.3 → **−1.0**（静止稳定性；门控走真值指令，
#      走动时几乎不生效，不拖累行走）
#
# ⚠️ 热启动必须配 `ckpt_manipulator`：判别器**存在 checkpoint 里**
#    （`wasabi.py` override 了 state_dict/load_state_dict）⇒ 直接 --resume 会
#    `size mismatch [1024,670] vs [1024,134]` 启动即崩。
#    V12 的 agent cfg 已设 `drop_amp_discriminator`：删掉 ckpt 里的判别器（+其 AdamW 状态），
#    actor/critic 与 PPO 优化器照常加载、判别器从零重学。
#
# ⚠️ **不要和 V11 同时跑**（CPU 是瓶颈，两边都会减半）。等 V11 停了再起这个。
#
# 判读（≥500 轮，看包络）：
#   风格：`amp_d` 从 −0.78 抬向 −0.4 以内 / `r_amp` 回升  ← ①的效应
#   静止：站立档 `track_default_pose_on_stand`、`base_ang_vel`、`base_contact`  ↓  ← ②的效应
#   行走：`feet_air_time` >0.15/s、`vx_*_rmse`、中位回合长度**不退化**
#
# 用法（云端，必须先看 V11 是否已停）：
#   pgrep -af 'train[.]py'                       # 空 ⇒ V11 已停
#   ls ~/gpufree-data/instinct_xu/logs/*V11*     # 取 V11 的 run 目录名 + 最新 ckpt
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v12.sh 4096 1000000000 cuda:0 amp_walk_v12_4096 <V11_run_dir> model_8500.pt' \
#        /root/train_amp_walk_v12_4096.log
#   # 不想热启动（例如 V11 的集合已经明显退化）⇒ 第 5 个参数留空 = 从零开始：
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v12.sh 4096 1000000000 cuda:0 amp_walk_v12_4096' \
#        /root/train_amp_walk_v12_4096.log
#
# 手册：$1 num_envs（默认 4096）  $2 iters（默认 10^9 = 无上限）
#       $3 device（默认 cuda:0）  $4 run_name tag（默认 amp_walk_v12_4096）
#       $5 load_run（可选；非空即触发 --resume）  $6 checkpoint（默认 model_2000.pt）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v12_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-model_2000.pt}"
TASK="Instinct-Amp-Walk-V12-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
