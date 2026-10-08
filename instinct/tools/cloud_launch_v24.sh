#!/usr/bin/env bash
# ============================================================================
# 一键：**停掉 V23 → 静态核验 → 从零起 V24**（跑在云端）
# ============================================================================
# V24 = V23（白化 + 进度判据 + 逐关节死区/权重，**逐字不动**）+ `dof_acc_l2` **全关节**。
# 本轮唯一变量 = 加速度罚的关节覆盖面（权重 −1.0e-6 与 func 都不动）。
#
# ⚠️ V23 在 2500 轮已确定进入"高频小跳"模态 ⇒ 从零重训（from 塌陷 ckpt 续训出不来自吸引子）。
# ⚠️ V23 的 checkpoint 保留在 `logs/.../*v23_scratch_4096*/model_*.pt`，需要对照时可续跑：
#      bash /root/run_amp_walk_v23.sh 4096 1000000000 cuda:0 amp_walk_v23_resume_4096 \
#           <V23 run 目录> model_2500.pt
# ============================================================================
set -uo pipefail

echo "=== 1) 停掉当前训练（V23）==="
pkill -9 -f 'train[.]py' || true
sleep 2
pgrep -af 'train[.]py' | grep -v pgrep || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  ❌ 没有可用 GPU ⇒ 不启动" >&2
  exit 1
}

echo "=== 3) 启动前静态核验（结果落 /root/v24_verify_result.txt）==="
cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
  source /root/.llp_env.sh && unset PYTHONPATH && \
  /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v24_cfg.py || {
    echo "  ❌ 静态核验失败 ⇒ 不启动训练" >&2
    tail -40 /root/v24_verify_result.txt >&2 || true
    exit 1
  }
if ! grep -q "\[V24 PASS\]" /root/v24_verify_result.txt; then
  echo "  ❌ 结果文件里没有 [V24 PASS] ⇒ 不启动训练" >&2
  tail -40 /root/v24_verify_result.txt >&2 || true
  exit 1
fi
echo "  ✅ 静态核验 PASS"

echo "=== 4) 从零起 V24（V23 三项 + acc 全关节）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v24.sh" \
  /root/train_amp_walk_v24_4096.log

cat <<'EOF'
=== done ===
启动日志必须看到：
  * [AMP-WALK PASS] ... rewards=21
  * Learning iteration 0/            ← 从零
  * Linear(in_features=1560)         ← 判别器输入 195×8
  * agent.yaml: normalizer_class_name=EmpiricalNormalization / amp_grad_pen_space=normalized
  * env.yaml: dof_acc_l2 的 asset_cfg.joint_names = '.*'（**不再**是 hip/knee 列表）
TB 新标签（本轮主读数）：
  * Step/hf_acc_all / hf_acc_legs / hf_acc_arms / hf_acc_waist
对比目标（V23 @2500 轮）：hop_frac 0.974 → 期望 <0.9；in_band_frac 0.092 → >0.3；
single_stance 0.016 s → >0.04 s；track_lin_vel 守住 ≈1.0
EOF
