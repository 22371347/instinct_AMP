#!/usr/bin/env bash
# ============================================================================
# V29 崩溃守护（V28 的教训：episode length 在 10 轮内 489→2.7，全程没有任何机制介入）
# ============================================================================
# 轮询训练日志，命中任一判据 ⇒ `pkill train.py`（ckpt 保留，TB 保留）⇒ 立刻止损。
#
# 判据（默认）：
#   ① 最近 20 个 log 块的 `Mean episode length` **全部** < 50  ⇒ 策略已经"一启动就倒"
#   ② 最近 3 个 log 块的 `Mean step action_abs_mean` **全部** > 1.5 ⇒ 动作幅度失控
#      （V27 基线 0.354；V28 崩坏时 2.2~2.7）
#
# 用法（云端，由 `cloud_launch_v29.sh` 自动拉起）：
#   bash /root/cloud_run_bg.sh 'bash /root/v29_guard.sh' /root/v29_guard.log
# 参数：$1 训练日志 $2 轮询秒 $3 episode_len 阈值 $4 连续块数 $5 action_abs_mean 阈值
# ============================================================================
set -uo pipefail

LOG="${1:-/root/train_amp_walk_v29_4096.log}"
POLL="${2:-30}"
BAD_EP="${3:-50}"
BAD_EP_N="${4:-20}"
BAD_A="${5:-1.5}"
GUARD_LOG=/root/v29_guard.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$GUARD_LOG"; }

log "===== V29 崩溃守护启动 ====="
log "监控日志: $LOG"
log "判据①: 最近 ${BAD_EP_N} 块的 Mean episode length 全部 < ${BAD_EP}"
log "判据②: 最近 3 块的 Mean step action_abs_mean 全部 > ${BAD_A}"
log "轮询周期: ${POLL}s"

# 等训练真正开始（最多 10 分钟）
for _ in $(seq 1 20); do
  [[ -f "$LOG" ]] && grep -q 'Learning iteration' "$LOG" 2>/dev/null && break
  sleep 30
done
log "训练日志已出现 Learning iteration，开始判据轮询"

while true; do
  if ! pgrep -f 'train[.]py' >/dev/null 2>&1; then
    log "train.py 已不在运行 ⇒ 守护退出（正常或已被停）"
    break
  fi

  # ---- 判据①：episode length ----
  eps=$(grep 'Mean episode length' "$LOG" 2>/dev/null | tail -n "$BAD_EP_N" | awk '{print $NF}')
  n=$(printf '%s\n' "$eps" | grep -c '[0-9]' || true)
  if [[ "$n" -ge "$BAD_EP_N" ]]; then
    bad=$(printf '%s\n' "$eps" | awk -v t="$BAD_EP" '$1+0 < t {c++} END {print c+0}')
    if [[ "$bad" -ge "$BAD_EP_N" ]]; then
      log "❌ 判据①命中：最近 ${BAD_EP_N} 块的 Mean episode length 全部 < ${BAD_EP}（最近值 $(printf '%s' "$eps" | tail -n 1)）"
      log "   ⇒ 判定策略崩坏，停训止损（ckpt/TB 保留）"
      pkill -9 -f 'train[.]py'
      sleep 3
      log "   已停。请检查 /root/train_amp_walk_v29_4096.log 与 TB。"
      break
    fi
  fi

  # ---- 判据②：动作幅度 ----
  a_cnt=$(grep 'Mean step action_abs_mean' "$LOG" 2>/dev/null | tail -n 3 | awk '{print $NF}' \
          | awk -v t="$BAD_A" '$1+0 > t {c++} END {print c+0}')
  if [[ "$a_cnt" -ge 3 ]]; then
    log "❌ 判据②命中：最近 3 块 action_abs_mean 全部 > ${BAD_A}（最近值 $(grep 'Mean step action_abs_mean' "$LOG" | tail -n 1 | awk '{print $NF}')）"
    log "   ⇒ 动作幅度失控，停训止损"
    pkill -9 -f 'train[.]py'
    sleep 3
    break
  fi

  sleep "$POLL"
done

log "===== V29 崩溃守护结束 ====="
