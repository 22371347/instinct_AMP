#!/usr/bin/env bash
# ============================================================================
# V32 **三臂串行 sweep**：A（对照）→ B（白化两侧）→ C（B + 拆脚手架）
# ============================================================================
# 三臂共同点：同一 fork ckpt（V31 `model_11500.pt`）、同预算 800 iters、同 seed、
#             保留判别器、`flat_orientation_l2=-4`、coef=1.0、λ=0.75、σ=0.20+退火、
#             站立档门控 true（早就开着）。
# 每臂起训后**先做有效性自检**（这是控制变量的前提），任何一项不过就**跳过该臂**并打 `!!!`：
#   ① 运行期 `params/agent.yaml` 的 `amp_normalizer_expert_only` 与期望一致
#   ③ C 必须 `feet_air_time=2` 且 `joint_deviation_arms=0`
#   ④ 三臂都**不得**出现 `drop_amp_discriminator`（判别器必须延续 V31）
# ⚠️ 2026-10-06 教训：原来还有一条「② 必须看到 `reset_amp_normalizer: 重置 4 个白化 buffer`」，
#   但复核发现**判别器压根没接 normalizer**（V22 是旁支）⇒ 白化不存在 ⇒ 那条自检必然失败，
#   它把 B/C 两臂全跳过了。⇒ **判据要认「有效果」，不要认「某个字符串」**。
# 臂列表默认已去掉 B（B 的设计前提是「白化两侧」，而白化不存在 ⇒ B ≡ A，留着是浪费 GPU）。
# 达到 11500+800=12300 轮后杀掉训练、杀掉守护、打印该臂摘要，再进下一臂。
# 每臂会留下 `model_12000.pt`（保存间隔 500）。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh "bash /root/cloud_v32_sweep.sh" /root/v32_sweep.log
# ============================================================================
set -uo pipefail

FORK="${1:-11500}"
BUDGET="${2:-800}"
TARGET=$((FORK + BUDGET))
RUN_ROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk
WATCH='Mean reward:|Mean episode length|discriminator_loss|Mean step amp_d:|Mean step amp_d_ref|Mean step amp_style_reward|Mean step amp_style_std|Mean step amp_style_zero_frac|Mean step tilt_deg_mean|Mean step tilt_p95_deg|Mean step tilt_gt30_frac|Mean step grav_bx_mean|Mean step air_time_single_stance|Mean step air_time_hop_frac|Mean step air_time_in_band_frac|Mean step action_abs_mean|Episode_Reward/feet_air_time|Episode_Reward/flat_orientation_l2|Episode_Reward/joint_deviation_arms|disc_grad/'

log() { echo "[$(date '+%F %T')] $*"; }

FAILED_ARMS=""

#: 要跑的臂（默认已去掉 B：白化不存在 ⇒ B ≡ A）；可用 $3 覆盖，例：`bash cloud_v32_sweep.sh 11500 800 "C"`
ARMS="${3:-A C}"
log "本次 sweep 的臂：$ARMS"

for ARM in $ARMS; do
  case "${ARM,,}" in
    a) TAG="amp_walk_v32A_control_from${FORK}"        ; EXPECT_NORM="true"  ;;
    b) TAG="amp_walk_v32B_white_both_from${FORK}"     ; EXPECT_NORM="false" ;;
    c) TAG="amp_walk_v32C_noscaffold_from${FORK}"     ; EXPECT_NORM="false" ;;
  esac
  LOG="/root/train_${TAG}.log"

  log "================ ${ARM} 臂启动（目标 ${TARGET} 轮，tag=$TAG）================"
  if ! bash /root/cloud_launch_v32.sh "$ARM" "$FORK"; then
    log "!!! ${ARM} 臂启动脚本失败 ⇒ 跳过"
    FAILED_ARMS="$FAILED_ARMS $ARM(启动失败)"
    continue
  fi

  # ---- 等日志与权重横幅（最多 12 分钟）----
  for _ in $(seq 1 72); do
    [[ -s "$LOG" ]] && grep -q 'AMP-WALK WEIGHTS' "$LOG" 2>/dev/null && break
    sleep 10
  done
  log "${ARM} 臂权重横幅已出现"

  # ---- 有效性自检 ----
  RUN_DIR="$(ls -td "$RUN_ROOT"/*v32"${ARM,,}"* 2>/dev/null | head -1)"
  # 注意：run 目录名里的臂标记是小写 v32a/v32b/v32c（tag 里是 v32A…）⇒ 用不区分大小写匹配
  RUN_DIR="$(ls -td "$RUN_ROOT"/* 2>/dev/null | grep -i "v32${ARM}" | head -1)"
  AGENT_YAML=""
  [[ -n "$RUN_DIR" ]] && AGENT_YAML=$(ls "$RUN_DIR"/params/agent.yaml 2>/dev/null | head -1)

  BAD=0
  if [[ -n "$AGENT_YAML" ]]; then
    ACTUAL=$(grep -E '^  amp_normalizer_expert_only:' "$AGENT_YAML" | awk '{print $2}')
    if [[ "$ACTUAL" == "$EXPECT_NORM" ]]; then
      log "  [自检 ①] amp_normalizer_expert_only = $ACTUAL ✅（期望 $EXPECT_NORM）"
    else
      log "  [自检 ①] amp_normalizer_expert_only = $ACTUAL ❌（期望 $EXPECT_NORM）"; BAD=1
    fi
  else
    log "  [自检 ①] 找不到 $RUN_DIR/params/agent.yaml（跳过）"; BAD=1
  fi

  # ⚠️ 原「自检 ②：必须看到 reset_amp_normalizer 日志」已删除 —— 判别器没接 normalizer（V22 是旁支），
  #    白化不存在 ⇒ 那条判据必然失败（2026-10-06 就是它把 B/C 全跳过了）。
  #    教训：**判据要认「有效果」，不要认「某个字符串」。**

  if [[ "${ARM,,}" == "c" ]]; then
    W=$(grep -m1 'AMP-WALK WEIGHTS' "$LOG")
    if [[ "$W" == *"feet_air_time=2"* && "$W" == *"joint_deviation_arms=0"* ]]; then
      log "  [自检 ③] feet_air_time=2 且 joint_deviation_arms=0 ✅"
    else
      log "  [自检 ③] 奖励权重横幅未生效 ❌"; echo "    $W"; BAD=1
    fi
  fi

  if grep -qi 'drop_amp_discriminator' "$LOG"; then
    log "  [自检 ④] ❌ 出现 drop_amp_discriminator（判别器被丢，非预期）"; BAD=1
  else
    log "  [自检 ④] 判别器未被丢弃 ✅"
  fi

  if [[ "$BAD" == "1" ]]; then
    log "!!! ${ARM} 臂有效性自检未通过 ⇒ **跳过该臂**（数据不可用于归因），继续下一臂"
    FAILED_ARMS="$FAILED_ARMS $ARM(自检失败)"
    pkill -9 -f 'train[.]py' || true
    pkill -9 -f 'v29_guard' || true
    sleep 5
    continue
  fi

  # ---- 等到达目标轮次 ----
  while true; do
    sleep 60
    CUR=$(grep -o 'Learning iteration [0-9]*' "$LOG" 2>/dev/null | tail -1 | awk '{print $3}')
    CUR=${CUR:-0}
    if ! pgrep -f 'train[.]py' >/dev/null 2>&1; then
      log "!!! ${ARM} 臂训练进程消失（崩溃 / 守护触发），当前轮次=$CUR ⇒ 停止 sweep"
      FAILED_ARMS="$FAILED_ARMS $ARM(进程异常退出@$CUR)"
      log "---------- ${ARM} 臂尾部日志 ----------"
      tail -25 "$LOG"
      exit 1
    fi
    if [[ "$CUR" -ge "$TARGET" ]]; then
      log "${ARM} 臂到达 $CUR 轮（目标 $TARGET）⇒ 收尾"
      break
    fi
  done

  pkill -9 -f 'train[.]py' || true
  pkill -9 -f 'v29_guard' || true
  sleep 5
  log "---------- ${ARM} 臂摘要（尾段 40 行关键指标）----------"
  grep -E "$WATCH" "$LOG" | sed 's/^ *//' | tail -40
  log "  该臂 ckpt：$(ls -1 "$RUN_DIR"/model_*.pt 2>/dev/null | sed 's#.*/##' | tr '\n' ' ')"
done

log "================ 三臂 sweep 结束 ================"
[[ -n "$FAILED_ARMS" ]] && log "⚠️ 有问题的臂：$FAILED_ARMS" || log "三臂全部完成且自检通过 ✅"
