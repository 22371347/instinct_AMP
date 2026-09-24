#!/usr/bin/env bash
# ============================================================================
# 本地后台循环：定期让**云端**重新生成训练曲线图，并 scp 拉回本地看
#
# 与 AMP_mjlab 的 `tools/refresh_plots.sh` 同一套思路（云端画 → 本地拉），
# 差异是我们把图直接放到工作区内的 `instinct/plots/`，VS Code 里直接能看。
#
# 用法：
#   bash tools/refresh_plots.sh once      # 只刷一次（调试用）
#   bash tools/refresh_plots.sh           # 后台循环（默认每 600 s）
#   bash tools/refresh_plots.sh stop      # 停止循环
#   INTERVAL=300 bash tools/refresh_plots.sh
#
# 产物（本地）：
#   instinct/plots/01_overview.png          总览：回报 / 回合长度 / AMP 损失 / 噪声 std
#   instinct/plots/02_reward_terms.png      21 项 Episode_Reward/* 逐项收敛
#   instinct/plots/03_amp_diagnostics.png   判别器诊断（损失 / 惩罚 / 权重衰减 / aux 奖励）
#   instinct/plots/04_foot_diagnostics.png  ⭐ 步宽与交叉（本次迁移最关键的设计风险点）
#   instinct/plots/05_terminations_perf.png 终止方式 / FPS / 速度跟踪误差
#   instinct/plots/06_amp_task_split.png    最终奖励里 task 与 AMP 风格各占多少
#   instinct/plots/digest_latest.txt        ⭐ 文本摘要（不开 SSH 就能看健康度）
#
# ⚠️ 云端那份 plot_training.py 必须先在云端（见 tools/upload_plot_tool.sh 或
#    tools/instinct_ssh.py --put）。本脚本只负责"刷新 + 拉取"。
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SSH="$SCRIPT_DIR/instinct_ssh.py"
PY_LOCAL="$(command -v python3 || command -v python)"

DST="$PROJECT_DIR/plots"
REMOTE_PLOTS="/root/gpufree-data/instinct_xu/logs/instinct_rl/plots"
INTERVAL="${INTERVAL:-600}"
CLOUD_PY="/opt/conda/envs/isaaclab/bin/python"

FILES="01_overview.png 02_reward_terms.png 03_amp_diagnostics.png \
04_foot_diagnostics.png 05_terminations_perf.png 06_amp_task_split.png"

refresh_once() {
  mkdir -p "$DST"

  # 1) 云端重新生成（LLP_SKIP_FULL=1：监控循环里跨 run 合并太慢，没必要）
  echo "[$(date '+%F %T')] 云端生成曲线 ..."
  "$PY_LOCAL" "$SSH" "LLP_SKIP_FULL=1 $CLOUD_PY /root/plot_training.py" 2>&1 \
    | grep -E '\[INFO\]|\[WARN\]|@|摘要' | tail -25

  # 2) 拉回本地（逐文件；第一次跑可能还没有 06 图 —— 失败只记数，不中断）
  ok=0; fail=0
  for f in $FILES; do
    if "$PY_LOCAL" "$SSH" --get "$REMOTE_PLOTS/$f" "$DST/" >/dev/null 2>&1; then
      ok=$((ok + 1))
    else
      fail=$((fail + 1))
    fi
  done
  # 摘要单独改名，免得 full 模式那份覆盖 latest
  if "$PY_LOCAL" "$SSH" --get "$REMOTE_PLOTS/digest.txt" "$DST/digest_latest.txt" >/dev/null 2>&1; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
  fi

  echo "[$(date '+%F %T')] 拉回 $ok 个文件（失败 $fail）-> $DST"
  [[ -f "$DST/digest_latest.txt" ]] && sed 's/^/    /' "$DST/digest_latest.txt"
}

case "${1:-loop}" in
  once)
    refresh_once
    ;;
  stop)
    if pkill -f '[r]efresh_plots.sh'; then
      echo "已停止 refresh_plots.sh"
    else
      echo "没有在跑的 refresh_plots.sh"
    fi
    ;;
  loop)
    echo "[$(date '+%F %T')] 曲线刷新循环启动（每 ${INTERVAL}s）-> $DST"
    # 先立刻刷一次，别让用户干等一个周期
    refresh_once
    while true; do
      sleep "$INTERVAL"
      refresh_once
    done
    ;;
  *)
    echo "用法: $0 {loop|once|stop}" >&2
    exit 1
    ;;
esac
