#!/usr/bin/env bash
# ============================================================================
# 一键：**静态核验 → 从 V26 最新 ckpt 热启动 V27**（跑在云端）
# ============================================================================
# V27 = V26（σ=0.20 + 300 轮退火，奖励/AMP 观测/`action_scale=1.0` 全不动）
#       + **去掉白化**（回 V19b/V19d/V20 口径）
#       + **`drop_amp_discriminator`**（判别器与其优化器从零重学，策略保留）
#
# 目的（用户指定）：V26 的判别器继承自 V24、已饱和 ⇒ 风格项从第 0 轮就是死的；
#                   必须重置判别器，才能干净地判定"白化是否杀死风格项"并拿到可用的 AMP 风格信号。
#
# ⚠️ 会先把当前训练停掉（V26 若还在跑）。
# ============================================================================
set -uo pipefail

RUN_ROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk

echo "=== 1) 停掉当前训练（若有）==="
pkill -9 -f 'train[.]py' || true
sleep 2
pgrep -af 'train[.]py' | grep -v pgrep || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  ❌ 没有可用 GPU ⇒ 不启动" >&2
  exit 1
}

echo "=== 3) 启动前静态核验（结果落 /root/v27_verify_result.txt）==="
cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
  source /root/.llp_env.sh && unset PYTHONPATH && \
  /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v27_cfg.py || {
    echo "  ❌ 静态核验失败 ⇒ 不启动训练" >&2
    tail -40 /root/v27_verify_result.txt >&2 || true
    exit 1
  }
if ! grep -q "\[V27 PASS\]" /root/v27_verify_result.txt; then
  echo "  ❌ 结果文件里没有 [V27 PASS] ⇒ 不启动训练" >&2
  tail -40 /root/v27_verify_result.txt >&2 || true
  exit 1
fi
echo "  ✅ 静态核验 PASS"

echo "=== 4) 定位热启动 ckpt（V26 最新）==="
V26_DIR=$(ls -td "$RUN_ROOT"/*v26* 2>/dev/null | head -1)
[[ -n "$V26_DIR" ]] || { echo "  ❌ 找不到 V26 run 目录" >&2; exit 1; }
V26_CK=$(ls -t "$V26_DIR"/model_*.pt 2>/dev/null | head -1)
[[ -n "$V26_CK" ]] || { echo "  ❌ $V26_DIR 下没有 model_*.pt" >&2; exit 1; }
echo "  ✅ 热启动：$(basename "$V26_DIR") / $(basename "$V26_CK")"

echo "=== 5) 起 V27（丢判别器 + 去白化）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v27.sh 4096 1000000000 cuda:0 amp_walk_v27_dropdisc_nodrop_white_4096 '$V26_DIR' $(basename "$V26_CK")" \
  /root/train_amp_walk_v27_4096.log

cat <<'EOF'
=== done ===
启动日志必须看到：
  * [AMP-WALK PASS] ... rewards=21
  * Linear(in_features=1560)                      ← 判别器输入 195×8（无 normalizer 也行）
  * Warning: using a hacky way to load the model. ← ckpt_manipulator 生效
  * Done: using a hacky way to load the model.
  * Learning iteration 9000/...                   ← 从 V26 的轮次续（热启动）
  * 判别器应从 **输出≈0**（未饱和）开始，而不是一上来就 −1.0
TB 新标签（本轮主读数）：
  * Step/amp_style_std（0.0008 → 期望 >0.03；V19b/V20 是 0.094~0.124）
  * Loss/discriminator_loss（1e-5 → >1e-3）、Step/amp_d（−0.998 → >−0.9）、Step/amp_d_ref（不再恒 +1.000）
  * 不退化：Step/hf_acc_all ≈45、Mean reward > +35、hop<0.648、in_band>0.446、single_stance>0.119
  * 任务项：Episode_Reward/track_lin_vel_xy_exp > 1.412（超过 V24）
EOF
