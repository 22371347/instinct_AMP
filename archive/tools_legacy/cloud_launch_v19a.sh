#!/usr/bin/env bash
# ============================================================================
# 一键：**停 V18 → 热启动 V19a**（跑在云端）
# ============================================================================
# 热启动 = 载入 V18 **最新** checkpoint（actor/critic/优化器/normalizer 续用），
#          但 `ckpt_manipulator = drop_amp_discriminator` ⇒ **判别器按新口径重学**。
#
# ⚠️ 容器 CPU 配额只有 14 核 ⇒ 起新的必须先停旧的。
# ⚠️ V18 的 checkpoint 全部保留；若将来要续跑 V18：
#    bash /root/run_amp_walk_v18.sh 4096 1000000000 cuda:0 amp_walk_v18_scratch_4096 <run 目录> model_XXXX.pt
# ============================================================================
set -uo pipefail

B=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk

echo "=== 1) 停掉所有训练进程 ==="
pkill -9 -f 'train[.]py' || true
sleep 2
pgrep -af 'train[.]py' | grep -v pgrep || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true

echo "=== 3) 取 V18 最新 checkpoint ==="
V18_DIR=$(ls -1dt "$B"/*v18_scratch_4096* 2>/dev/null | head -1)
if [[ -z "$V18_DIR" ]]; then
  echo "  ❌ 没找到 V18 run 目录（$B/*v18_scratch_4096*）" >&2
  exit 1
fi
CKPT=$(ls -1 "$V18_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)
if [[ -z "$CKPT" ]]; then
  echo "  ❌ V18 目录里没有 model_*.pt" >&2
  exit 1
fi
RUN_NAME=$(basename "$V18_DIR")
CKPT_NAME=$(basename "$CKPT")
echo "  run : $RUN_NAME"
echo "  ckpt: $CKPT_NAME"

echo "=== 4) 热启动 V19a（丢掉判别器，从 V18 的 $CKPT_NAME 续） ==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v19a.sh 4096 1000000000 cuda:0 amp_walk_v19a_warmV18_4096 $RUN_NAME $CKPT_NAME" \
  /root/train_amp_walk_v19a_4096.log

echo "=== done ==="
echo "核验清单（启动日志里必须看到）："
echo "  * [ckpt-manipulator] drop_amp_discriminator: 丢弃 ...（判别器+优化器）"
echo "  * Learning iteration <V18 的轮号>/      ← 证明是续训不是从零"
echo "  * [AMP-WALK WEIGHTS] feet_air_time=4 / flat_orientation_l2=-2 / action_rate_l2=-0.1"
echo "  * cmd_ranges=((-0.5,1.0),(-0.5,0.5),(-0.8,0.8))、Linear(in_features=670)"
echo "  * params/agent.yaml: gradient_penalty_coef=1.0 / weight_decay_coef=1e-05 /"
echo "    logit_weight_decay_coef=0.0 / amp_reward_eps=0.05 / amp_reward_form=log_quad"
