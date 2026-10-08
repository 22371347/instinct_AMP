#!/usr/bin/env bash
# ============================================================================
# 一键：**静态核验 → 从 V24@8000 热启动 V26**（跑在云端）
# ============================================================================
# V26 = V24 全部设置（奖励/AMP/白化/`action_scale=1.0`，逐字不动）
#       + `clip_max_std 0.6 → 0.20` + `clip_max_std_schedule = 300`（线性退火）
#       起点 = V24 `model_8000.pt`（热启动；AMP 同形 ⇒ 不丢判别器）。
#
# 实验 B 的目的：同一套权重下，只把 σ 从 0.57 压到 0.20，看
#   ① 噪声税是否按 D4 预测掉到 −0.58/步；② 步态（hop/in_band/single_stance）是否改善；
#   ③ `amp_style_std` 是否从 0.008 复活（判定"杀死风格项的是 σ 还是白化"）。
#
# ⚠️ 会先把当前训练停掉（若还有）。
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

echo "=== 3) 启动前静态核验（结果落 /root/v26_verify_result.txt）==="
cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
  source /root/.llp_env.sh && unset PYTHONPATH && \
  /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v26_cfg.py || {
    echo "  ❌ 静态核验失败 ⇒ 不启动训练" >&2
    tail -40 /root/v26_verify_result.txt >&2 || true
    exit 1
  }
if ! grep -q "\[V26 PASS\]" /root/v26_verify_result.txt; then
  echo "  ❌ 结果文件里没有 [V26 PASS] ⇒ 不启动训练" >&2
  tail -40 /root/v26_verify_result.txt >&2 || true
  exit 1
fi
echo "  ✅ 静态核验 PASS"

echo "=== 4) 定位热启动 ckpt（V24@8000）==="
V24_DIR=$(ls -td "$RUN_ROOT"/*v24* 2>/dev/null | head -1)
if [[ -z "$V24_DIR" ]]; then
  echo "  ❌ 找不到 V24 run 目录 ⇒ 不启动" >&2
  exit 1
fi
if [[ ! -f "$V24_DIR/model_8000.pt" ]]; then
  echo "  ❌ $V24_DIR/model_8000.pt 不存在 ⇒ 不启动" >&2
  ls -t "$V24_DIR"/model_*.pt | head -5 >&2
  exit 1
fi
echo "  ✅ 热启动：$(basename "$V24_DIR") / model_8000.pt"

echo "=== 5) 起 V26（热启动 + σ 退火 0.57→0.20）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v26.sh 4096 1000000000 cuda:0 amp_walk_v26_hot_v24_m8000_4096 '$V24_DIR' model_8000.pt" \
  /root/train_amp_walk_v26_4096.log

cat <<'EOF'
=== done ===
启动日志必须看到：
  * [AMP-WALK PASS] ... rewards=21
  * Linear(in_features=1560)                        ← 判别器输入 195×8（与 ckpt 匹配）
  * Learning iteration 8000/...                     ← 热启动（不是 0）
  * agent.yaml: clip_max_std=0.2 / clip_max_std_schedule=300
TB 新标签（本轮主读数）：
  * Train/sigma_cap（应 0.57 → 0.20，300 轮走完）、Train/sigma_actual
  * Policy/mean_noise_std（~300 轮后 ≈0.20）
  * Episode_Reward/action_rate_l2（−1.24 → ~−0.17）、dof_acc_l2（−1.46 → ~−0.16）
  * Mean reward（−0.48 → 期望 ≥ +3.5）
  * Step/air_time_hop_frac（<0.65）、in_band_frac（>0.44）、single_stance（>0.12）
  * Step/amp_style_std（0.008 → >0.03 ？= σ 是不是杀死风格项的元凶）
EOF
