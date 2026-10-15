#!/usr/bin/env bash
# ============================================================================
# 一键：**停 V19d → 从零起 V20**（跑在云端）
# ============================================================================
# V20 = V19d 的奖励/指令/数据集/AMP 奖励口径**逐字不动**，只把判别器输入
#       从"关节空间 67 维 × 10 帧 = 670"换成"**刚体空间 195 维 × 2 帧 = 390**"
#       （照抄 AMP_mjlab：13 刚体 × (pos3 + ori6 + lin3 + ang3)，anchor = torso_link）。
#
# ⚠️ 容器 CPU 配额只有 14 核 ⇒ 起新的必须先停旧的。
# ⚠️ V19d 的 checkpoint 保留在
#    `logs/.../amp_walk_v19d_warmV19c_4096_*/model_*.pt`；要续跑 V19d：
#      bash /root/run_amp_walk_v19d.sh 4096 1000000000 cuda:0 amp_walk_v19d_warmV19c_4096 \
#           <run 目录> model_XXXX.pt
# ============================================================================
set -uo pipefail

echo "=== 1) 停掉所有训练进程 ==="
pkill -9 -f 'train[.]py' || true
sleep 2
pgrep -af 'train[.]py' | grep -v pgrep || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true

echo "=== 3) 启动前静态核验（不建 Isaac 场景，秒级）==="
cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
  source /root/.llp_env.sh && unset PYTHONPATH && \
  /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v20_cfg.py || {
    echo "  ❌ 静态核验失败 ⇒ 不启动训练" >&2
    exit 1
  }

echo "=== 4) 从零起 V20（刚体空间 195×2 帧，判别器输入 390）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v20.sh" \
  /root/train_amp_walk_v20_4096.log

cat <<'EOF'
=== done ===
核验清单（启动日志 / params/*.yaml 里必须看到）：
  * [AMP-WALK PASS] ... rewards=21、cmd_ranges=((-0.5,1.0),(-0.5,0.5),(-0.8,0.8))
  * Learning iteration 0/                 ← 确实是**从零**
  * 没有 [ckpt-manipulator] 那一行（ckpt_manipulator=None）
  * ⭐ Linear(in_features=390)            ← **本轮唯一变量的落地证据**（V19d 是 670）
  * agent.yaml: amp_quad_k=0.125 / discriminator_gradient_penalty_coef=1.0 /
    discriminator_weight_decay_coef=1e-05 / discriminator_logit_weight_decay_coef=0.0 /
    amp_reward_eps=0.05 / amp_reward_form=log_quad / discriminator_reward_type=wasserstein
  * env.yaml: amp_policy/amp_reference 的 4 项 = body_{pos,ori,lin_vel,ang_vel}_b，
    history 都是 2，asset_cfg.body_names 是那 13 个刚体，
    anchor_cfg.body_names = ('torso_link',)
EOF
