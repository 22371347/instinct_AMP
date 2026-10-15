#!/usr/bin/env bash
# ============================================================================
# V32 实验臂启动器（跑在云端）
# ============================================================================
# A / B / C 三臂共用同一 fork ckpt（默认 V31 `model_11500.pt`），各跑 800 iters：
#   A = V31 原样续训（纯对照）        task=Instinct-Amp-Walk-V31-G1-v0
#   B = A + 白化统计量改成"两侧累积"   task=Instinct-Amp-Walk-V32B-G1-v0
#   C = B + 拆脚手架（air_time 4→2、arms_dev 0）task=Instinct-Amp-Walk-V32C-G1-v0
# 三臂都**保留判别器**（AMP 口径没变 ⇒ `ckpt_manipulator` 只有 B/C 的"重置白化统计量"）。
#
# 用法（云端）：
#   bash /root/cloud_launch_v32.sh <A|B|C> [fork_iter]        # 默认 11500
#
# 手册：$1 num_envs $2 iters $3 device $4 tag $5 load_run $6 checkpoint $7 task
# ============================================================================
set -uo pipefail

ARM="${1:-B}"
FORK_ITER="${2:-11500}"
RUN_ROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk

case "${ARM,,}" in
  a) TASK="Instinct-Amp-Walk-V31-G1-v0"  ; TAG="amp_walk_v32A_control_from${FORK_ITER}" ;;
  b) TASK="Instinct-Amp-Walk-V32B-G1-v0" ; TAG="amp_walk_v32B_white_both_from${FORK_ITER}" ;;
  c) TASK="Instinct-Amp-Walk-V32C-G1-v0" ; TAG="amp_walk_v32C_noscaffold_from${FORK_ITER}" ;;
  *) echo "用法: bash /root/cloud_launch_v32.sh <A|B|C> [fork_iter]" >&2; exit 2 ;;
esac
LOG="/root/train_${TAG}.log"
GUARD_LOG="/root/guard_${TAG}.log"

echo "=== 1) 停掉当前训练（V31）==="
pkill -9 -f 'train[.]py' || true
sleep 3
pgrep -af 'train[.]py' | grep -v pgrep || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  ❌ 没有可用 GPU ⇒ 不启动" >&2; exit 1
}

echo "=== 3) 定位 fork ckpt（V31 的 model_${FORK_ITER}.pt）==="
V31_DIR=""
for d in $(ls -td "$RUN_ROOT"/*v31* 2>/dev/null); do
  if [[ -f "$d/model_${FORK_ITER}.pt" ]]; then V31_DIR="$d"; break; fi
done
[[ -n "$V31_DIR" ]] || {
  echo "  ❌ 找不到含 model_${FORK_ITER}.pt 的 V31 run 目录；现有 ckpt：" >&2
  ls "$RUN_ROOT"/*v31*/model_*.pt 2>/dev/null | sed 's#.*/##' | sort -u >&2
  exit 1
}
echo "  ✅ fork：$(basename "$V31_DIR") / model_${FORK_ITER}.pt"
echo "  该 ckpt 会被 A/B/C 三臂共用 ⇒ 起点完全一致"

echo "=== 4) 起 ${ARM^^} 臂（task=$TASK）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v32.sh 4096 1000000000 cuda:0 '$TAG' '$V31_DIR' model_${FORK_ITER}.pt '$TASK'" \
  "$LOG"

echo "=== 5) 拉起崩溃守护 ==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh '$LOG' 30 50 20 1.5" "$GUARD_LOG"

cat <<EOF
=== done（臂 ${ARM^^}）===
日志：$LOG     守护：$GUARD_LOG
启动后必须核对（**这是控制变量的有效性检查**）：
  * \`[AMP-WALK WEIGHTS] ...\`              ← A/B 臂应与 V31 一致；C 臂应看到 feet_air_time=2 / joint_deviation_arms=0
  * 运行期 \`params/agent.yaml\`：
      A 臂: disable_style_on_stand: true, amp_normalizer_expert_only: true
      B/C : disable_style_on_stand: true, amp_normalizer_expert_only: **false**
  * B/C 臂日志里出现 \`[ckpt-manipulator] reset_amp_normalizer: 重置 4 个白化 buffer …\`
    （**没出现就说明这一臂等于没改**，必须停下来查）
  * 三臂都不应出现 \`drop_amp_discriminator\`（判别器必须延续 V31）
EOF
