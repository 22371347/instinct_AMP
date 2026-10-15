#!/usr/bin/env bash
# ============================================================================
# V33 实验臂启动器（跑在云端）：S0 / S1 / A-cold
# ============================================================================
# 三臂共用同一 fork（V31 `model_11500.pt`），各再跑 **1000 iters**（11500 → **12500**，
# 500 的倍数 ⇒ 可导出 ONNX）；actor/critic 热启动，**判别器从零重学**（`drop_amp_discriminator`）。
#
#   S0    = 结构包：① 白化(eps=0.1) ② GP 归一化空间 ③ AMP 状态 396（anchor 投影重力）
#           task=Instinct-Amp-Walk-V33S-G1-v0     专家集=/root/amp_expert_states_396.pt
#   S1    = S0 + 拆脚手架（feet_air_time 4→2、joint_deviation_arms −0.1→0）
#           task=Instinct-Amp-Walk-V33S1-G1-v0    专家集同上
#   ACOLD = 旧口径（390 / 无白化 / raw GP）+ 丢判别器 ⇒ 参照系
#           task=Instinct-Amp-Walk-V33ACold-G1-v0 专家集=/root/amp_expert_states.pt
#
# 用法（云端）：
#   bash /root/cloud_launch_v33.sh <S0|S1|ACOLD> [fork_iter]     # 默认 fork=11500, 预算=1000
#
# 手册：$1 num_envs $2 iters $3 device $4 tag $5 load_run $6 checkpoint $7 task
# ============================================================================
set -uo pipefail

ARM="${1:-S0}"
FORK_ITER="${2:-11500}"
BUDGET="${3:-1000}"
#: ⭐ 可选：显式指定续训的 run 目录（相对 $RUN_ROOT 或绝对路径）。
#: 不填 = 按旧逻辑去搜 V31 的 run（fork 11500）；S4 要续训 S3 ⇒ 传 S3 的目录名。
LOAD_RUN_OVERRIDE="${4:-}"
RUN_ROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk

case "${ARM,,}" in
  s0)    TASK="Instinct-Amp-Walk-V33S-G1-v0"     ; TAG="amp_walk_v33S0_struct_from${FORK_ITER}" ;;
  s1)    TASK="Instinct-Amp-Walk-V33S1-G1-v0"    ; TAG="amp_walk_v33S1_noscaffold_from${FORK_ITER}" ;;
  s2)    TASK="Instinct-Amp-Walk-V33S2-G1-v0"    ; TAG="amp_walk_v33S2_white001_from${FORK_ITER}" ;;
  s3)    TASK="Instinct-Amp-Walk-V33S3-G1-v0"    ; TAG="amp_walk_v33S3_nowhite_weakened_from${FORK_ITER}" ;;
  s4)    TASK="Instinct-Amp-Walk-V33S4-G1-v0"    ; TAG="amp_walk_v33S4_noairtime_boostangvel_from${FORK_ITER}" ;;
  s5)    TASK="Instinct-Amp-Walk-V33S5-G1-v0"    ; TAG="amp_walk_v33S5_air1_boosttrack_from${FORK_ITER}" ;;
  s6)    TASK="Instinct-Amp-Walk-V33S6-G1-v0"    ; TAG="amp_walk_v33S6_ampcoef4_noairtime_from${FORK_ITER}" ;;
  s7)    TASK="Instinct-Amp-Walk-V33S7-G1-v0"    ; TAG="amp_walk_v33S7_mjlabdisc_from${FORK_ITER}" ;;
  s8)    TASK="Instinct-Amp-Walk-V33S8-G1-v0"    ; TAG="amp_walk_v33S8_coef4_adaptive_from${FORK_ITER}" ;;
  acold) TASK="Instinct-Amp-Walk-V33ACold-G1-v0" ; TAG="amp_walk_v33ACold_legacy_from${FORK_ITER}" ;;
  *) echo "用法: bash /root/cloud_launch_v33.sh <S0|S1|S2|S3|S4|S5|S6|S7|S8|ACOLD> [fork_iter] [budget] [load_run_dir]" >&2; exit 2 ;;
esac
LOG="/root/train_${TAG}.log"
GUARD_LOG="/root/guard_${TAG}.log"

echo "=== 1) 停掉当前训练 ==="
pkill -9 -f 'train[.]py' || true
sleep 3
pgrep -af 'train[.]py' | grep -v pgrep || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  ❌ 没有可用 GPU ⇒ 不启动" >&2; exit 1
}

echo "=== 3) 定位 fork ckpt（model_${FORK_ITER}.pt）==="
V31_DIR=""
if [[ -n "$LOAD_RUN_OVERRIDE" ]]; then
  # ⭐ 显式指定（S4 续训 S3 就是走这条）：先按 $RUN_ROOT 下的目录名找，再试绝对路径
  if [[ -d "$RUN_ROOT/$LOAD_RUN_OVERRIDE" ]]; then
    V31_DIR="$RUN_ROOT/$LOAD_RUN_OVERRIDE"
  elif [[ -d "$LOAD_RUN_OVERRIDE" ]]; then
    V31_DIR="$LOAD_RUN_OVERRIDE"
  fi
  [[ -n "$V31_DIR" ]] || { echo "  ❌ 指定的续训目录不存在：$LOAD_RUN_OVERRIDE" >&2; exit 1; }
  echo "  （显式指定续训目录）"
else
  for d in $(ls -td "$RUN_ROOT"/*v31* 2>/dev/null); do
    if [[ -f "$d/model_${FORK_ITER}.pt" ]]; then V31_DIR="$d"; break; fi
  done
fi
[[ -n "$V31_DIR" ]] || {
  echo "  ❌ 找不到含 model_${FORK_ITER}.pt 的 run 目录；现有 ckpt：" >&2
  ls "$RUN_ROOT"/*v31*/model_*.pt 2>/dev/null | sed 's#.*/##' | sort -u >&2
  exit 1
}
[[ -f "$V31_DIR/model_${FORK_ITER}.pt" ]] || {
  echo "  ❌ $V31_DIR 里没有 model_${FORK_ITER}.pt；现有：$(ls -1 "$V31_DIR"/model_*.pt 2>/dev/null | sed 's#.*/##' | tr '\n' ' ')" >&2
  exit 1
}
echo "  ✅ fork：$(basename "$V31_DIR") / model_${FORK_ITER}.pt"

echo "=== 4) 检查专家数据集（判别器的"专家批"来源）==="
if [[ "${ARM,,}" == "acold" ]]; then
  EXPERT=/root/amp_expert_states.pt
else
  # S0/S1/S2/S3 都带 anchor 重力通道（396）
  EXPERT=/root/amp_expert_states_396.pt
fi
if [[ -f "$EXPERT" ]]; then
  echo "  ✅ $EXPERT（$(du -h "$EXPERT" | cut -f1)）"
else
  echo "  ❌ 缺 $EXPERT" >&2
  if [[ "${ARM,,}" != "acold" ]]; then
    echo "     ⇒ S 臂需要 396 维专家集，请先跑 sweep 里的导出步骤：" >&2
    echo "        bash /root/cloud_v33_sweep.sh   （它会在起 S 臂前自动导出）" >&2
  fi
  exit 1
fi

echo "=== 5) 起 ${ARM^^} 臂（task=$TASK，再训 $BUDGET iters ⇒ 停在 $((FORK_ITER + BUDGET))）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v33.sh 4096 $BUDGET cuda:0 '$TAG' '$V31_DIR' model_${FORK_ITER}.pt '$TASK'" \
  "$LOG"

echo "=== 6) 拉起崩溃守护 ==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh '$LOG' 30 50 20 1.5" "$GUARD_LOG"

cat <<EOF
=== done（臂 ${ARM^^}）===
日志：$LOG     守护：$GUARD_LOG
启动后必须核对（**有效性检查，不是形式检查**）：
  * \`[AMP-WALK WEIGHTS]\` 横幅 ——
      S 臂：flat_orientation_l2=-4 / feet_air_time=4（S1 是 2）/ joint_deviation_arms=-0.1（S1 是 0）
  * 运行期 \`params/env.yaml\`：\`side_vel_min: 0.15\`（三臂都必须有 ⇒ 侧向档不再落进静止门）
  * 运行期 \`params/agent.yaml\`：
      S 臂   : \`amp_grad_pen_space: normalized\` + \`normalizer_class_name: EmpiricalNormalization\`
      ACOLD  : \`amp_grad_pen_space: raw\` + **没有** normalizer 键
  * 日志里 \`Discriminator Network Structure\` 的输入维：S 臂 **396**、ACOLD **390**
  * S 臂日志里必须出现 \`EmpiricalNormalization\`（= 白化真的接上了）
  * 日志里 \`[wasabi] 专家批来自离线数据集：$EXPERT\`
  * 判别器口径（**按臂别，别一刀切** —— 2026-10-06 这里误杀过一次 S4）：
      S0/S1/S2/S3/ACOLD : 日志**必须**出现 \`drop_amp_discriminator\`（口径变了 ⇒ 必须重学）
      S4（续训臂）       : 日志**不应**出现 \`drop_amp_discriminator\`（396 布局/专家集同 S3 ⇒ 故意保留权重）
EOF
