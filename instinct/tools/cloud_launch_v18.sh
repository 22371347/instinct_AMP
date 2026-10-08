#!/usr/bin/env bash
# ============================================================================
# 一键：**停 V17 → 从零起 V18（步态目标带 + 方案 A）**（跑在云端）
# ============================================================================
# ⚠️ 容器 CPU 配额只有 14 核 ⇒ 起新的必须先停旧的（两个 Isaac Sim 训练会互相拖慢）。
# ⚠️ V17 的 checkpoint 全部保留在 `logs/.../amp_walk_v17_scratch_4096_*/model_*.pt`，
#    要续跑 V17 用 run_amp_walk_v17.sh 显式传 run 目录 + ckpt 即可。
# ============================================================================
set -uo pipefail

echo "=== 1) 停掉所有训练进程 ==="
pkill -9 -f 'train[.]py' || true
pgrep -af 'train[.]py' | grep -v pgrep || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true

echo "=== 3) 从零起 V18 ==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v18.sh" \
  /root/train_amp_walk_v18_4096.log
echo "=== done（核验：flat_orientation_l2=-2 / action_rate_l2=-0.1 / feet_air_time=4（新函数）/ feet_swing_time_penalty=0 / track_*=3.0,0.8,1.0 / cmd_ranges=((-0.5,1.0),(-0.5,0.5),(-0.8,0.8)) / 5 clip）==="
