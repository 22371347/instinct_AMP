#!/usr/bin/env bash
# 远端心跳：每 20s 记一次「进程 CPU / RSS、冒烟日志大小、ov 扩展缓存大小」。
# 目的：把「盲轮询 → 一次看趋势」变成可能，用来区分「慢」和「真卡住」。
#   CPU 一直 >0 且日志/缓存字节在涨 → 在干活
#   CPU ≈0 且三个数字都不动         → 真卡住，该动手了
set -uo pipefail
LOG="${1:-/root/smoke.log}"
OV=/root/.local/share/ov

for _ in $(seq 1 180); do
  pid=$(pgrep -f "train.py" | head -1 || true)
  if [[ -n "${pid:-}" ]]; then
    cpu=$(ps -o pcpu= -p "$pid" 2>/dev/null | tr -d ' ')
    rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
    et=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
  else
    cpu="-"; rss="-"; et="-"
  fi
  printf '%s pid=%s et=%ss cpu=%s%% rss=%sKB smoke=%s ov=%sMB kitproc=%s\n' \
    "$(date +%T)" "${pid:-none}" "$et" "$cpu" "$rss" \
    "$(wc -c < "$LOG" 2>/dev/null || echo 0)" \
    "$(du -sm "$OV" 2>/dev/null | cut -f1 || echo 0)" \
    "$(pgrep -c -f 'kit' 2>/dev/null || echo 0)"
  sleep 20
done
echo "watchdog 结束"
