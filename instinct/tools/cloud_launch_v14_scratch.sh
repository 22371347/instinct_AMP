#!/usr/bin/env bash
# ============================================================================
# 一键：**停 V12 → 起 V14（从零重训）**（跑在云端）
# ============================================================================
# V14 = V12 的环境（2 帧判别器 134 维 / 三档指令 / 2-clip / 镜像增广+损失 /
#              σ 上限 0.6 / 21 项奖励）
#       ① 回退 `track_default_pose_on_stand` −1.0 → **−0.3**（V12 退化的主要成因）
#       ② 限位项换**软边距**版：踝 → 全关节，距限位 0.15 rad 内二次收费，权重 −1.5
#       ③ 算法侧 λ/c 改回**从零验证过的 0.75 / 0.25**（V10b 配方；λ=0.5 会把任务信号砍半）
#
# ⚠️ **不 resume**（从零开始）：run_amp_walk_v14.sh 的 $5/$6 传空 ⇒ cloud_train_amp_walk_v3.sh
#    不加 --resume。从零时 `init_noise_std=0.35` 生效、期望回报从 0 开始爬。
#
# 为什么固化这个文件：命令要经 `本地 → ssh → cloud_run_bg.sh → bash -c → train.py` 五层，
# 命令行里传空字符串很容易被引号吃掉（本项目踩过多次）⇒ 写进文件只有一个 token。
# ============================================================================
set -uo pipefail

echo "=== 1) 停掉所有训练进程 ==="
pkill -9 -f 'train[.]py' || true
pgrep -af 'train[.]py' || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true

echo "=== 3) 起 V14（**从零**，不 resume）==="
# run_amp_walk_v14.sh 的 $5/$6 默认为空 ⇒ cloud_train_amp_walk_v3.sh 不加 --resume
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v14.sh 4096 1000000000 cuda:0 amp_walk_v14_scratch_4096" \
  /root/train_amp_walk_v14_4096.log
echo "=== done ==="
