#!/usr/bin/env bash
# ============================================================================
# 一键：**停旧训 → 静态核验 → 从 V27 最新 ckpt 热启动 V30 → 拉起崩溃守护**
# ============================================================================
# V30 = V29 逐字不动（输入 390 / bounded_quad / 去白化 / σ=0.20+退火 / 奖励表）
#       + 判别器 = **AMP_mjlab 原始配方**：
#         ① GP λ=1.0→10.0   ② GP 只在 expert 侧   ③ 优化器分组 wd（trunk 1e-3/head 1e-2）
#         ④ 专家批 = 离线运动数据集 i.i.d. 抽帧（/root/amp_expert_states.pt，2934×390）
#         ⑤ 风格强度刻意不动（coef 保持 1.0，有效权重 0.25）
#       + `drop_amp_discriminator`（判别器口径变了，必须从零重学）
#
# 背景：V29 的判别器 **1 个迭代（20 梯度步）内** 把 MSE 压到 0.035（V27 稳态 0.0014）
#       ⇒ 风格奖励变常数 ⇒ 风格项无梯度。V11 是唯一未饱和的版本（0.1214），也是风格最好的一版。
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

echo "=== 3) 启动前静态核验（结果落 /root/v30_verify_result.txt）==="
cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
  source /root/.llp_env.sh && unset PYTHONPATH && \
  /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v30_cfg.py || {
    echo "  ❌ 静态核验脚本非 0 退出 ⇒ 不启动训练" >&2
    tail -60 /root/v30_verify_result.txt >&2 || true
    exit 1
  }
if ! grep -q "\[V30 PASS\]" /root/v30_verify_result.txt; then
  echo "  ❌ 结果文件里没有 [V30 PASS] ⇒ 不启动训练" >&2
  tail -60 /root/v30_verify_result.txt >&2 || true
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

echo "=== 5) 起 V30（mjlab 判别器配方 + 丢判别器）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v30.sh 4096 1000000000 cuda:0 amp_walk_v30_mjlab_disc_recipe_4096 '$V27_DIR' $(basename "$V27_CK")" \
  /root/train_amp_walk_v30_4096.log

echo "=== 6) 拉起崩溃守护（V28 的教训）==="
bash /root/cloud_run_bg.sh "bash /root/v30_guard.sh" /root/v30_guard.log

cat <<'EOF'
=== done ===
启动日志必须看到：
  * [AMP-WALK PASS] ... rewards=21
  * Linear(in_features=390)                          ← 判别器输入（V29 起已是 390）
  * [wasabi] 专家批来自离线数据集：/root/amp_expert_states.pt shape=(2929, 390)
                                                     ← ⭐ ④ 生效（若不出现 = 专家 buffer 没接上）
  * Warning: using a hacky way to load the model.     ← ckpt_manipulator 生效
  * Learning iteration 10000/...                      ← 从 V27 的轮次续

TB 判据（**只看判别器是否活过来**，约 100~300 轮即可判）：
  * Loss/discriminator_loss：0.0014~0.035 → **期望 > 0.05**（V11=0.1214 是唯一未饱和的版本）
  * Step/amp_d / amp_d_ref：−0.92 / +0.87 → 期望 −0.2 ~ −0.7 / 不恒 +1
  * Step/amp_style_std：0.058 → **期望 > 0.10**
  * Step/amp_style_zero_frac：0.36→1 → 期望 < 0.3
  * Loss/discriminator_gradient_penalty：0.017 → 期望显著上升（λ=10）
  * 不退化闸门：Mean reward ≥ 30、episode length ≥ 300、action_abs_mean ≤ 0.6
  * 逐维探针（新版按迭代里程碑）：/root/amp_dim_probe_it0001.txt / it0100.txt / …

守护：/root/v30_guard.log（崩坏时自动停训并写原因）
EOF
