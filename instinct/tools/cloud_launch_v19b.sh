#!/usr/bin/env bash
# ============================================================================
# 一键：**停 V19a → 从零起 V19b**（跑在云端）
# ============================================================================
# ⚠️ 容器 CPU 配额只有 14 核 ⇒ 起新的必须先停旧的。
# ⚠️ V19a 的 checkpoint 保留在 `logs/.../amp_walk_v19a_warmV18_4096_*/model_*.pt`；
#    要续跑 V19a：
#      bash /root/run_amp_walk_v19a.sh 4096 1000000000 cuda:0 amp_walk_v19a_warmV18_4096 <run 目录> model_XXXX.pt
# ============================================================================
set -uo pipefail

echo "=== 1) 停掉所有训练进程 ==="
pkill -9 -f 'train[.]py' || true
sleep 2
pgrep -af 'train[.]py' | grep -v pgrep || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true

echo "=== 3) 从零起 V19b（步态带 exp τ=0.08 + quad k=0.125）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v19b.sh" \
  /root/train_amp_walk_v19b_4096.log

cat <<'EOF'
=== done ===
核验清单（启动日志 / params/*.yaml 里必须看到）：
  * [AMP-WALK PASS] ... rewards=21、cmd_ranges=((-0.5,1.0),(-0.5,0.5),(-0.8,0.8))
  * Learning iteration 0/                 ← 确实是**从零**
  * 没有 [ckpt-manipulator] 那一行（ckpt_manipulator=None）
  * Linear(in_features=670)               ← h=10 未动
  * env.yaml: feet_air_time 的 params 里有 good_lo=0.4 / good_hi=1.0 / soft=0.3 / tau=0.08
    （`tau` 一开，`soft` 只管**上侧** ⇒ 上侧在 1.3 s 归零）
  * agent.yaml: amp_quad_k=0.125 / discriminator_gradient_penalty_coef=1.0 /
    discriminator_weight_decay_coef=1e-05 / discriminator_logit_weight_decay_coef=0.0 /
    amp_reward_eps=0.05 / amp_reward_form=log_quad / discriminator_reward_type=wasserstein
EOF
