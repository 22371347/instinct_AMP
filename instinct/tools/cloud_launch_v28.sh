#!/usr/bin/env bash
# ============================================================================
# 一键：**停 V27 → 静态核验 → 从 V27 最新 ckpt 热启动 V28**（跑在云端）
# ============================================================================
# V28 = V27（去白化 + σ=0.20/300 轮退火 + drop_amp_discriminator，全部逐字不动）
#       + **`discriminator_reward_coef` 1.0 → 4.0**（AMP 风格项权重 0.25 → 1.0，4×；λ 不动）
#
# 目的（用户 2026-10-04 指定）：V27 已练到 10000 轮仍无类人行走风格。诊断显示判别器
#   是**活的**（`style_std` 0.058），但风格项的作用量只有 0.0145/步、比任务项小 10~70 倍
#   ⇒ **权重不足**，判别器再准也推不动步态。本轮把权重 ×4 作为单变量实验。
# ============================================================================
set -uo pipefail

RUN_ROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk

echo "=== 1) 停掉当前训练（V27）==="
pkill -9 -f 'train[.]py' || true
sleep 3
pgrep -af 'train[.]py' | grep -v pgrep || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  ❌ 没有可用 GPU ⇒ 不启动" >&2
  exit 1
}

echo "=== 3) 启动前静态核验（结果落 /root/v28_verify_result.txt）==="
cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
  source /root/.llp_env.sh && unset PYTHONPATH && \
  /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v28_cfg.py || {
    echo "  ❌ 静态核验脚本非 0 退出 ⇒ 不启动训练" >&2
    tail -50 /root/v28_verify_result.txt >&2 || true
    exit 1
  }
if ! grep -q "\[V28 PASS\]" /root/v28_verify_result.txt; then
  echo "  ❌ 结果文件里没有 [V28 PASS] ⇒ 不启动训练" >&2
  tail -50 /root/v28_verify_result.txt >&2 || true
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
V27_CK=$(ls -t "$V27_DIR"/model_*.pt | head -1)
# 取最新轮次的 ckpt（按文件名里的数字排，而不是 mtime —— 避免 mtime 被 touch 干扰）
V27_CK=$(ls "$V27_DIR"/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
[[ -n "$V27_CK" ]] || { echo "  ❌ 无法选出 V27 ckpt" >&2; exit 1; }
echo "  ✅ 热启动：$(basename "$V27_DIR") / $(basename "$V27_CK")"

echo "=== 5) 起 V28（coef ×4 + 丢判别器）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v28.sh 4096 1000000000 cuda:0 amp_walk_v28_ampcoef4_dropdisc_4096 '$V27_DIR' $(basename "$V27_CK")" \
  /root/train_amp_walk_v28_4096.log

cat <<'EOF'
=== done ===
启动日志必须看到：
  * [AMP-WALK PASS] ... rewards=21
  * Linear(in_features=1560)                      ← 判别器输入 195×8（未白化）
  * Warning: using a hacky way to load the model. ← ckpt_manipulator 生效
  * Learning iteration 10000/...                  ← 从 V27 的轮次续（热启动）

运行时核对（约 10 分钟后看第一批 TB 点）：
  * 判别器从 **输出≈0** 重新开始（不是一上来 −1.0）
  * Step/amp_style_std 仍 ≈0.05~0.08（信号本身没变，变的是它的权重）
  * 行为侧（本轮核心）：air_time_single_stance ↑ / air_time_lift_frac ↑ / hop_frac ↓
  * 风险闸门：Mean reward 不得 < +20；hf_acc_all 不得明显抬头
EOF
