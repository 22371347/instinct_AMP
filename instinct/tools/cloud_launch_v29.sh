#!/usr/bin/env bash
# ============================================================================
# 一键：**停旧训 → 静态核验 → 从 V27 最新 ckpt 热启动 V29 → 拉起崩溃守护**
# ============================================================================
# V29 = V27 逐字不动（去白化 / σ=0.20+退火 / 奖励表 / 判别器系数回退）
#       + 判别器向 AMP_mjlab 对齐：① 输入 2 帧 390  ② 专家批独立随机重采 (T,B)
#                                  ③ 有界奖励形状 `clamp(1−0.25(d−1)²,0,1)`
#       + `drop_amp_discriminator`（输入维度变了，判别器必须从零重学）
#
# 背景：V28（coef ×4 等）在 iter 10470 训崩（episode length 489→2.7），已停并弃用。
# ============================================================================
set -uo pipefail

RUN_ROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk

echo "=== 1) 停掉当前训练（若有）==="
pkill -9 -f 'train[.]py' || true
sleep 3
pgrep -af 'train[.]py' | grep -v pgrep || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  ❌ 没有可用 GPU ⇒ 不启动" >&2
  exit 1
}

echo "=== 3) 启动前静态核验（结果落 /root/v29_verify_result.txt）==="
cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
  source /root/.llp_env.sh && unset PYTHONPATH && \
  /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v29_cfg.py || {
    echo "  ❌ 静态核验脚本非 0 退出 ⇒ 不启动训练" >&2
    tail -60 /root/v29_verify_result.txt >&2 || true
    exit 1
  }
if ! grep -q "\[V29 PASS\]" /root/v29_verify_result.txt; then
  echo "  ❌ 结果文件里没有 [V29 PASS] ⇒ 不启动训练" >&2
  tail -60 /root/v29_verify_result.txt >&2 || true
  exit 1
fi
echo "  ✅ 静态核验 PASS"

echo "=== 4) 定位热启动 ckpt（V27 最新）==="
mapfile -t CAND < <(ls -td "$RUN_ROOT"/*v27* 2>/dev/null)
printf '  候选 run 目录：\n'
for d in ${CAND[@]+"${CAND[@]}"}; do printf '    %s\n' "$d"; done
V27_DIR=""
for d in ${CAND[@]+"${CAND[@]}"}; do
  if ls "$d"/model_*.pt >/dev/null 2>&1; then V27_DIR="$d"; break; fi
done
[[ -n "$V27_DIR" ]] || { echo "  ❌ 找不到含 model_*.pt 的 V27 run 目录" >&2; exit 1; }
V27_CK=$(ls "$V27_DIR"/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
[[ -n "$V27_CK" ]] || { echo "  ❌ 无法选出 V27 ckpt" >&2; exit 1; }
echo "  ✅ 热启动：$(basename "$V27_DIR") / $(basename "$V27_CK")"

echo "=== 5) 起 V29（mjlab 对齐 + 丢判别器）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v29.sh 4096 1000000000 cuda:0 amp_walk_v29_mjlabalign_4096 '$V27_DIR' $(basename "$V27_CK")" \
  /root/train_amp_walk_v29_4096.log

echo "=== 6) 拉起崩溃守护（V28 的教训）==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh" /root/v29_guard.log

cat <<'EOF'
=== done ===
启动日志必须看到：
  * [AMP-WALK PASS] ... rewards=21
  * Linear(in_features=390)                       ← ① 生效（V27 是 1560）
  * Warning: using a hacky way to load the model. ← ckpt_manipulator 生效
  * Learning iteration 10000/...                  ← 从 V27 的轮次续（热启动）

TB 读数（约 10 分钟后第一批点）：
  * ① 390 维判别器
  * ② Step/amp_d 不再 20 轮内贴 ±1；Step/amp_d_ref 同样
  * ③ Step/amp_style_reward ∈ [0,1]（V27 是 [0,3]，饱和时恒 0.693）
       Step/amp_style_zero_frac 长期 1.0 ⇒ 风格项死（需换杠杆，别再抬权重）
  * 不退化闸门：Mean reward ≥ 30、episode length ≥ 300、action_abs_mean ≤ 0.6、hf_acc ≈44
  * 行为侧：air_time_single_stance ↑ / air_time_hop_frac ↓

守护：/root/v29_guard.log（崩坏时自动停训并写原因）
EOF
