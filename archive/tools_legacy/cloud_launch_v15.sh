#!/usr/bin/env bash
# ============================================================================
# 一键：**停 V14（已收敛到单脚跳）→ 起 V15（风格分量 ×4 实验）**（跑在云端）
# ============================================================================
# V15 的唯一变量：`discriminator_reward_coef` 0.25 → 1.0
#   ⇒ 风格入账 (1−λ)c：0.0625 → 0.25（×4）；λ=0.75 不动 ⇒ 任务权重不变。
# 目的：V14 实测风格项 5400 轮只动 1.7%（amp_d 钉在 −0.47）⇒ 验证"放大风格分量"
#       能不能把 AMP 贡献的**变化率**拉回来（风格梯度是否真的存在）。
#
# 起点：热启动自 **V11 的 model_9000.pt**（会走路；V14 从零已跳成单脚跳，不能从它起）。
# 命令固化成文件，避免多层级传参被引号吃掉。
# ============================================================================
set -uo pipefail

echo "=== 1) 停掉所有训练进程 ==="
pkill -9 -f 'train[.]py' || true
pgrep -af 'train[.]py' || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true

echo "=== 3) 起 V15（风格 ×4，热启动自 V11 model_9000.pt）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v15.sh" \
  /root/train_amp_walk_v15_4096.log
echo "=== done ==="
