#!/usr/bin/env bash
# ============================================================================
# 一键：**停掉现有训练 → 从零起 V17（直立惩罚 ×2）**（跑在云端）
# ============================================================================
# V17 = V16 基准 + `flat_orientation_l2 −1 → −2`，**从零训练**（不 resume）。
# ⚠️ 起之前务必确认没有别的训练在跑：容器 CPU 配额只有 **14 核**，两个 Isaac Sim 训练
#    会互相拖慢（这也是本文件第 1 步 `pkill` 的原因）。
# ============================================================================
set -uo pipefail

echo "=== 1) 停掉所有训练进程 ==="
pkill -9 -f 'train[.]py' || true
pgrep -af 'train[.]py' | grep -v pgrep || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU / CPU 现状 ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true
nproc

echo "=== 3) 从零起 V17 ==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v17.sh" \
  /root/train_amp_walk_v17_4096.log
echo "=== done（核验：日志里应是 Learning iteration 0/… 且 flat_orientation_l2=-2）==="
