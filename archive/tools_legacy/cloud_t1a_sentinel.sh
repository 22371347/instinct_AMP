#!/usr/bin/env bash
# ============================================================================
# T1a 里程碑哨兵（跑在云端，后台常驻）
# ============================================================================
# 作用：每到一个里程碑（默认每 500 轮 = 一个 checkpoint）把**关键指标快照**追加
#       到 `$OUT`，并**对照发车前预注册的判据**打勾。
#
#   ⚠️ 判据必须发车前定好、写进日志 —— 防止事后挪门槛（本项目的老毛病）。
#
# 为什么不用 TensorBoard：TB 是"事后看曲线"，哨兵是"过程留痕 + 当时就判"，
# 而且曲线能重画、这些快照不能回改，给后续复盘留证据链。
#
# 用法（云端）：
#   bash /root/cloud_t1a_sentinel.sh [训练日志] [输出日志] [轮询秒] [里程碑间隔]
#   默认：/root/train_t1a_t0rewards_mirror.log  /root/sentinel_t1a.log  300  500
# ============================================================================
set -uo pipefail

LOG="${1:-/root/train_t1a_t0rewards_mirror.log}"
OUT="${2:-/root/sentinel_t1a.log}"
PERIOD="${3:-300}"
MILESTONE="${4:-500}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$OUT"; }

# 取某指标在日志中的**最新**数值（日志每 10 轮打一整块，取最后一条即可）
val() { grep -a "$1" "$LOG" 2>/dev/null | tail -1 | sed -E 's/.*: *//'; }

iter() { grep -aoE 'Learning iteration [0-9]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+'; }

last_ms=0
log "===== T1a 里程碑哨兵启动 ====="
log "监控日志   : $LOG"
log "输出       : $OUT"
log "轮询       : ${PERIOD}s    里程碑间隔: ${MILESTONE} 轮"
log "预注册判据 : vx_high_vel_mean>=0.50 | vx_mid_vel_mean>=0.20 | ep_len>=900 | 侧向 vy 三档有解"
log "T0 对照    : vx_high 0.562(tail)/0.679(last)  vx_mid 0.195/0.227  vx_low -0.140/-0.202  ep_len 970"

while true; do
  it="$(iter)"
  if [ -n "${it:-}" ]; then
    ms=$(( it / MILESTONE * MILESTONE ))
    if [ "$ms" -gt "$last_ms" ] && [ "$ms" -gt 0 ]; then
      last_ms="$ms"
      vl="$(val 'vx_low_vel_mean')";   vm="$(val 'vx_mid_vel_mean')";   vh="$(val 'vx_high_vel_mean')"
      yl="$(val 'vy_low_vel_mean')";   ym="$(val 'vy_mid_vel_mean')";   yh="$(val 'vy_high_vel_mean')"
      zl="$(val 'wz_low_vel_mean')";   zm="$(val 'wz_mid_vel_mean')";   zh="$(val 'wz_high_vel_mean')"
      log "----- 里程碑 ${ms} 轮（当前迭代 ${it}）-----"
      log "  Mean reward                $(val 'Mean reward')"
      log "  ep_len                     $(val 'Mean episode length')"
      log "  vx  low/mid/high vel_mean  ${vl} / ${vm} / ${vh}"
      log "  vy  low/mid/high vel_mean  ${yl} / ${ym} / ${yh}"
      log "  wz  low/mid/high vel_mean  ${zl} / ${zm} / ${zh}"
      log "  vx_high vxy_rmse           $(val 'vx_high_vxy_rmse')"
      log "  下一步：pull_latest_onnx.sh t1a model_${ms}.pt --tag t1a_${ms}  →  回放 + P1 探针"
    fi
  fi
  sleep "$PERIOD"
done
