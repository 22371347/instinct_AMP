#!/usr/bin/env bash
# ============================================================================
# 一键：**停旧训 → 找回 V30 最新 ckpt → 热启动 V31（保留判别器）→ 拉起崩溃守护**
# ============================================================================
# V31 = V30 **逐字不动**（判别器 mjlab 配方 / 输入 390 / bounded_quad / 去白化 / σ=0.20+退火 /
#       奖励表 21 项 / coef=1.0 / λ=0.75）
#       + **唯一变量**：`rewards.flat_orientation_l2.weight` −2.0 → **−4.0**
#       + `ckpt_manipulator=None`（口径没变 ⇒ **不丢判别器**，无缝续训）
#
# 姿态诊断（2026-10-06）见 run_amp_walk_v31.sh 的注释与 amp_walk_v31_env_cfg.py。
# ============================================================================
set -uo pipefail

RUN_ROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk

echo "=== 1) 停掉当前训练（V30）==="
pkill -9 -f 'train[.]py' || true
sleep 3
pgrep -af 'train[.]py' | grep -v pgrep || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  ❌ 没有可用 GPU ⇒ 不启动" >&2
  exit 1
}

echo "=== 3) 定位热启动 ckpt（V30 最新）==="
mapfile -t CAND < <(ls -td "$RUN_ROOT"/*v30* 2>/dev/null)
printf '  候选 run 目录：\n'
for d in ${CAND[@]+"${CAND[@]}"}; do printf '    %s\n' "$d"; done
V30_DIR=""
for d in ${CAND[@]+"${CAND[@]}"}; do
  if ls "$d"/model_*.pt >/dev/null 2>&1; then V30_DIR="$d"; break; fi
done
[[ -n "$V30_DIR" ]] || { echo "  ❌ 找不到含 model_*.pt 的 V30 run 目录" >&2; exit 1; }
V30_CK=$(ls "$V30_DIR"/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
[[ -n "$V30_CK" ]] || { echo "  ❌ 无法选出 V30 ckpt" >&2; exit 1; }
echo "  ✅ 热启动：$(basename "$V30_DIR") / $(basename "$V30_CK")"

echo "=== 4) 起 V31（flat_orientation_l2 = −4，保留判别器）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v31.sh 4096 1000000000 cuda:0 amp_walk_v31_flat4_4096 '$V30_DIR' $(basename "$V30_CK")" \
  /root/train_amp_walk_v31_4096.log

echo "=== 5) 拉起崩溃守护（V28 的教训）==="
bash /root/cloud_run_bg.sh "bash /root/v31_guard.sh" /root/v31_guard.log

cat <<'EOF'
=== done ===
启动日志必须看到（**这是本轮唯一的静态核验**）：
  * [AMP-WALK PASS] ... rewards=21
  * [AMP-WALK WEIGHTS] ... flat_orientation_l2=**-4**   ← 唯一变量生效
  * Linear(in_features=390)                              ← 判别器输入（应与 V30 相同）
  * 载入 V30 的 model_*.pt（**保留判别器**，不应出现 "drop" 之类的提示）
EOF
