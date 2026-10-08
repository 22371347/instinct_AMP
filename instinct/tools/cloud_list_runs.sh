#!/usr/bin/env bash
# ============================================================================
# 列出云端训练 run 的「进度 + checkpoint」总览（**机器可读**，由本地
# `pull_latest_onnx.sh` / `refresh_plots.sh` 解析）
#
# 为什么要单独一个脚本文件：
#   这段逻辑要塞进 `instinct_ssh.py "..."` 的话是三层嵌套引号 + `$(...)` + awk，
#   本项目已经在这上面栽过（见 cloud_run_bg.sh 的头注释）。
#   固化成文件 → scp 上去 → `bash /root/cloud_list_runs.sh`，一次都不会被拆坏。
#
# 用法：
#   bash cloud_list_runs.sh [N]        # N = 最多看几条日志，默认 5
#
# 输出（每行一条，`|` 分隔；**最后一行固定是 =ALIVE=**）：
#   <tag>|<迭代>|<日志mtime>|<run目录名>|<最新checkpoint>|<checkpoint个数>
#   =ALIVE=|<活着的 train.py 进程数>
#
# tag 取自日志名 /root/train_amp_walk_<tag>.log；run 目录按 "*<tag>" 后缀匹配。
# 日志按 mtime 倒序 ⇒ **第一行就是最近在跑的那次训练**。
# ============================================================================
set -uo pipefail

XU=/root/gpufree-data/instinct_xu
LOGROOT="$XU/logs/instinct_rl/g1_amp_walk"
N="${1:-5}"

for L in $(ls -t /root/train_amp_walk_*.log 2>/dev/null | head -"$N"); do
  T="$(basename "$L" .log)"
  T="${T#train_amp_walk_}"
  # 只留 "520/5000" 这种短形式 —— 全句 "Learning iteration 520/5000" 会把表头挤歪
  IT="$(grep -oE 'Learning iteration [0-9]+/[0-9]+' "$L" 2>/dev/null | tail -1 | sed 's/Learning iteration //')"
  # ⚠️ run 目录名 = `<启训时间戳>_<cfg.run_name>_<--run_name 的 tag>_from<fork 时间戳>`
  #    ⇒ tag 在**中间**，所以必须用 *包含* 匹配（`*"$T"*/`），
  #      旧写法 `*"$T"/`（后缀匹配）会自动失败 ⇒ 明明在训练却显示 `<无对应run目录> / ckpt×0>`
  #      （2026-10-06 实测：v33 各臂全中这一条）。
  D="$(ls -dt "$LOGROOT"/*"$T"*/ 2>/dev/null | head -1)"
  D="${D%/}"
  if [[ -n "$D" ]]; then
    C="$(ls "$D"/model_*.pt 2>/dev/null | sed 's#.*/##' | sort -t_ -k2 -n | tail -1)"
    NCK="$(ls "$D"/model_*.pt 2>/dev/null | wc -l | tr -d ' ')"
  else
    C=""
    NCK=0
  fi
  printf '%s|%s|%s|%s|%s|%s\n' \
    "$T" \
    "${IT:-<无迭代输出>}" \
    "$(date -r "$L" '+%m-%d %H:%M')" \
    "$(basename "${D:-<无对应run目录>}")" \
    "${C:-<无>}" \
    "$NCK"
done

printf '=ALIVE=|%s\n' "$(pgrep -cf '[t]rain\.py')"
