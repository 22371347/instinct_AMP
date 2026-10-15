#!/usr/bin/env bash
# ============================================================================
# 一键：**停掉所有训练 → 静态核验 → 从零起 V23**（跑在云端）
# ============================================================================
# V23 = V21（刚体 195×8，判别器输入 1560）+ 三处改动：
#   ① feet_air_time 进度判据（方向投影 + EMA 低通），删掉关闭率 <1% 的瞬时幅值门
#   ② track_default_pose_on_stand 逐关节死区 + 内部权重（weight 0.8）
#   ③ 判别器输入白化（expert-only）+ GP 挪到归一化空间
#
# ⚠️ 容器 CPU 配额只有 14 核 ⇒ 起新的必须先停旧的。
# ============================================================================
set -uo pipefail

echo "=== 1) 停掉所有训练进程 ==="
pkill -9 -f 'train[.]py' || true
sleep 2
pgrep -af 'train[.]py' | grep -v pgrep || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  ❌ 没有可用 GPU（当前可能是无显卡模式）⇒ 不上传/不启动" >&2
  exit 1
}

echo "=== 3) 启动前静态核验（不建 Isaac 场景，秒级；结果落 /root/v23_verify_result.txt）==="
cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
  source /root/.llp_env.sh && unset PYTHONPATH && \
  /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v23_cfg.py || {
    echo "  ❌ 静态核验失败 ⇒ 不启动训练（详见 /root/v23_verify_result.txt）" >&2
    tail -40 /root/v23_verify_result.txt >&2 || true
    exit 1
  }
# 第二道闸：不信任退出码，直接看结果文件里的结论行（Isaac 会吞 stdout、且 app.close()
# 会直接终止进程 ⇒ 只看文件里已经写下的东西）
if ! grep -q "\[V23 PASS\]" /root/v23_verify_result.txt; then
  echo "  ❌ 结果文件里没有 [V23 PASS] ⇒ 不启动训练" >&2
  tail -40 /root/v23_verify_result.txt >&2 || true
  exit 1
fi
echo "  ✅ 静态核验 PASS（结果文件含 [V23 PASS]）"

echo "=== 4) 从零起 V23（白化 + 进度判据 + 逐关节死区）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v23.sh" \
  /root/train_amp_walk_v23_4096.log

cat <<'EOF'
=== done ===
核验清单（启动日志 / params/*.yaml 里必须看到）：
  * [AMP-WALK PASS] ... rewards=21、cmd_ranges=((-0.5,1.0),(-0.5,0.5),(-0.8,0.8))
  * Learning iteration 0/                 ← 确实**从零**
  * 没有 [ckpt-manipulator] 那一行（ckpt_manipulator=None）
  * ⭐ Linear(in_features=1560)            ← 判别器输入（195×8）
  * agent.yaml: discriminator_kwargs 含 normalizer_class_name=EmpiricalNormalization
    / normalizer_kwargs.eps=0.01；amp_normalizer_expert_only=True；amp_grad_pen_space=normalized；
    discriminator_gradient_penalty_coef=1.0；amp_quad_k=0.125
  * env.yaml: feet_air_time.params 里 motion_threshold=0.0、progress_kappa=0.35；
    track_default_pose_on_stand.weight=-0.8、pos_tol_default=0.08
  * TB 上应出现新标签：Step/air_time_progress_frac、Step/air_time_proj_ratio、
    Step/stand_pose_pos_cost、Step/stand_pose_pos_err_legs/arms/waist
EOF
