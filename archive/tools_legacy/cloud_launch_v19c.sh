#!/usr/bin/env bash
# ============================================================================
# 一键：**停 V19b → 热启动 V19c**（跑在云端）
# ============================================================================
# 热启动 = 载入 V19b **最新** checkpoint（actor/critic/判别器/优化器/normalizer 全部续用），
#          因为 V19c 的 AMP 口径与 V19b **逐字相同**（同样的 670 维、同样的 quad k=0.125）。
#
# ⚠️ 容器 CPU 配额只有 14 核 ⇒ 起新的必须先停旧的。
# ⚠️ V19b 的 checkpoint 保留；要续跑 V19b：
#    bash /root/run_amp_walk_v19b.sh 4096 1000000000 cuda:0 amp_walk_v19b_scratch_4096 <run 目录> model_XXXX.pt
# ============================================================================
set -uo pipefail

B=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk

echo "=== 1) 停掉所有训练进程 ==="
pkill -9 -f 'train[.]py' || true
sleep 2
pgrep -af 'train[.]py' | grep -v pgrep || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true

echo "=== 3) 取 V19b 最新 checkpoint ==="
V19B_DIR=$(ls -1dt "$B"/*v19b_scratch_4096* 2>/dev/null | head -1)
if [[ -z "$V19B_DIR" ]]; then
  echo "  ❌ 没找到 V19b run 目录" >&2
  exit 1
fi
CKPT=$(ls -1 "$V19B_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)
if [[ -z "$CKPT" ]]; then
  echo "  ❌ V19b 目录里没有 model_*.pt" >&2
  exit 1
fi
RUN_NAME=$(basename "$V19B_DIR")
CKPT_NAME=$(basename "$CKPT")
echo "  run : $RUN_NAME"
echo "  ckpt: $CKPT_NAME"

echo "=== 4) 热启动 V19c（保留判别器）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v19c.sh 4096 1000000000 cuda:0 amp_walk_v19c_warmV19b_4096 $RUN_NAME $CKPT_NAME" \
  /root/train_amp_walk_v19c_4096.log

cat <<'EOF'
=== done ===
核验清单（启动日志 / params/*.yaml）：
  * Learning iteration <V19b 的轮号>/     ← 证明是续训
  * **没有** [ckpt-manipulator] 那一行（判别器被保留 ⇒ 不应出现 drop 提示）
  * env.yaml: feet_air_time params 里有 motion_threshold=0.15 / motion_ang_scale=0.5 / tau=0.08
  * env.yaml: action_rate_l2 weight = -0.06、track_default_pose_on_stand weight = -0.5
  * agent.yaml: amp_quad_k=0.125 / GP=1.0 / wd=1e-05 / logit_wd=0.0 / eps=0.05（与 V19b 相同）
  * 无 Traceback
EOF
