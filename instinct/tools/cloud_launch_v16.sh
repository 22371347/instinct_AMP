#!/usr/bin/env bash
# ============================================================================
# 一键：**停 V15（已达 10k 轮目标）→ 起 V16（帧数消融：2 帧 → 10 帧）**（跑在云端）
# ============================================================================
# V16 与 V15 的唯一差异 = 判别器输入帧数；起点、λ/c、判别器初始化全部对齐
#   ⇒ 直接回答"V11↔V15 的 AMP 差距是不是 10 帧 vs 2 帧造成的"。
#
# ⚠️ V15 的 checkpoint 全部保留在 `logs/.../amp_walk_v15_4096_*/model_*.pt`，
#    若之后想续跑 V15，用 run_amp_walk_v15.sh 指定该 run 目录 + ckpt 即可。
#
# 命令固化成文件，避免多层级传参被引号吃掉。
# ============================================================================
set -uo pipefail

echo "=== 1) 停掉所有训练进程（V15 已到 10k 轮目标）==="
pkill -9 -f 'train[.]py' || true
pgrep -af 'train[.]py' || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true

echo "=== 3) 起 V16（10 帧判别器，热启动自 V11 model_9000.pt）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v16.sh" \
  /root/train_amp_walk_v16_4096.log
echo "=== done ==="
