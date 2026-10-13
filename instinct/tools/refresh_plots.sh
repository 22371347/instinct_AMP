#!/usr/bin/env bash
# ============================================================================
# 本地后台循环：定期让**云端**重新生成训练曲线图，并 scp 拉回本地看
#
# 与 AMP_mjlab 的 `tools/refresh_plots.sh` 同一套思路（云端画 → 本地拉），
# 差异是我们把图直接放到工作区内的 `instinct/plots/`，VS Code 里直接能看。
#
# 用法：
#   bash tools/refresh_plots.sh once            # 只刷一次（调试用）
#   bash tools/refresh_plots.sh                 # 后台循环（默认每 600 s）
#   bash tools/refresh_plots.sh stop            # 停止循环
#   bash tools/refresh_plots.sh list            # 列出云端可选 run（run tag / 最新 ckpt）
#   INTERVAL=300 bash tools/refresh_plots.sh loop
#
# ⭐ 第二个参数 = run 名字子串（`v9c` / `v9b` / `amp_walk_v9b_mvp_4096`）：
#   bash tools/refresh_plots.sh once v9c        # 只画 v9c 那份
#   bash tools/refresh_plots.sh loop v9b        # 后台只刷 v9b
#   留空 = 画云端**最新**那个 run（旧行为）。
#   ⚠️ 并行跑多份训练（v9b / v9c …）时**务必传子串**，否则看的是“最后启动的那份”。
#
# 产物（本地）：
#   不带 run  ⇒ instinct/plots/<固定文件名>.png（旧行为）
#   带 run    ⇒ instinct/plots/<run>/<固定文件名>.png   ← 不同版本互不覆盖
#                instinct/plots/digest_<run>.txt        ← 稳定路径的摘要
#   文件清单：
#     01_overview.png          总览：回报 / 回合长度 / AMP 损失 / 噪声 std
#     02_reward_terms.png      Episode_Reward/* 逐项收敛
#     03_amp_diagnostics.png   判别器诊断（损失 / 惩罚 / 权重衰减 / aux 奖励）
#                               + ⭐ amp_d / amp_d_ref 与两者的差距叠加图
#     04_foot_diagnostics.png  步宽与交叉
#     05_terminations_perf.png 终止方式 / FPS / 速度跟踪误差
#     06_amp_task_split.png    最终奖励里 task 与 AMP 风格各占多少
#     digest_latest.txt        文本摘要（不开 SSH 就能看健康度）
#
# ⚠️ 云端那份 plot_training.py 在**每次刷新前会自动上传**（幂等）⇒ 本地改完直接跑本脚本
#    就生效，不必再单独跑 upload_plot_tool.sh。
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SSH="$SCRIPT_DIR/instinct_ssh.py"
PY_LOCAL="$(command -v python3 || command -v python)"

DST_BASE="$PROJECT_DIR/plots"
REMOTE_PLOTS="/root/gpufree-data/instinct_xu/logs/instinct_rl/plots"
INTERVAL="${INTERVAL:-600}"
CLOUD_PY="/opt/conda/envs/isaaclab/bin/python"

FILES="01_overview.png 02_reward_terms.png 03_amp_diagnostics.png \
04_foot_diagnostics.png 05_terminations_perf.png 06_amp_task_split.png"

#: 与 plot_training.py 的 `slug()` 保持一致（云端按这个子目录名落盘）
slug() { printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'; }

#: 同步云端绘图脚本（幂等）：本地改完 plot_training.py 立刻生效。
#: `refresh_once` 与 `refresh_list` 都要先调它 —— 否则 `list` 会去调云端**旧版**的 CLI。
sync_tool() {
  if ! "$PY_LOCAL" "$SSH" --put "$SCRIPT_DIR/plot_training.py" /root/plot_training.py >/dev/null 2>&1; then
    echo "[WARN] 上传 plot_training.py 失败（继续用云端上的旧版本）"
  fi
}

#: ⭐ 云端绘图脚本读 `LLP_LOG_ROOT`（默认指向 AMP 的 g1_amp_walk）。
#:   基座任务（T0）落在 g1_unified_omni_loco ⇒ 显式指定：
#:     LLP_LOG_ROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_unified_omni_loco \
#:       bash instinct/tools/refresh_plots.sh once <run 子串>
#:   没指定但给了 run 子串时，自动去各日志根里找含该子串的那一个。
_plot_env() {
  if [[ -n "${LLP_LOG_ROOT:-}" ]]; then
    printf 'LLP_LOG_ROOT=%s ' "$LLP_LOG_ROOT"
  elif [[ -n "${RUN_TOKEN:-}" ]]; then
    local _b=/root/gpufree-data/instinct_xu/logs/instinct_rl _r
    _r="$("$PY_LOCAL" "$SSH" "for r in $_b/*/; do ls -1 \"\$r\" 2>/dev/null | grep -qF '$RUN_TOKEN' && { echo \"\${r%/}\"; exit 0; }; done" 2>/dev/null \
        | grep -vE 'credentials|password' | tr -d '\r' | tail -1)"
    [[ -n "$_r" ]] && printf 'LLP_LOG_ROOT=%s ' "$_r"
  fi
}

refresh_once() {
  mkdir -p "$DST"

  # 0) 同步云端绘图脚本（幂等）
  sync_tool

  # 1) 云端重新生成（LLP_SKIP_FULL=1：监控循环里跨 run 合并太慢，没必要）
  echo "[$(date '+%F %T')] 云端生成曲线（run≈${RUN_TOKEN:-最新}）..."
  "$PY_LOCAL" "$SSH" "LLP_SKIP_FULL=1 LLP_RUN='$RUN_TOKEN' $(_plot_env)$CLOUD_PY /root/plot_training.py" 2>&1 \
    | grep -E '\[INFO\]|\[WARN\]|\[ERROR\]|@|摘要' | tail -28

  # 2) 拉回本地（逐文件；第一次跑可能还没有 06 图 —— 失败只记数，不中断）
  ok=0; fail=0
  for f in $FILES; do
    if "$PY_LOCAL" "$SSH" --get "$REMOTE_PLOTS$REMOTE_SUB/$f" "$DST/" >/dev/null 2>&1; then
      ok=$((ok + 1))
    else
      fail=$((fail + 1))
    fi
  done
  # 摘要单独改名，免得 full 模式那份覆盖 latest
  if "$PY_LOCAL" "$SSH" --get "$REMOTE_PLOTS$REMOTE_SUB/digest.txt" "$DST/digest_latest.txt" >/dev/null 2>&1; then
    ok=$((ok + 1))
    # 带 run 时再存一份稳定路径的摘要，方便同时看多份训练
    [[ -n "$TAG" ]] && cp -f "$DST/digest_latest.txt" "$DST_BASE/digest_$TAG.txt" 2>/dev/null
  else
    fail=$((fail + 1))
  fi

  echo "[$(date '+%F %T')] 拉回 $ok 个文件（失败 $fail）-> $DST"
  [[ -f "$DST/digest_latest.txt" ]] && sed 's/^/    /' "$DST/digest_latest.txt"
}

CMD="${1:-loop}"
case "$CMD" in
  once|loop|stop|list) shift ;;
  *) CMD="loop" ;;          # 允许直接 `refresh_plots.sh v9c`（只给 run）
  esac

# run 子串：位置参数优先，其次 $RUN / $LLP_RUN；也接受 `--run X` 写法
RUN_TOKEN="${*:-${RUN:-${LLP_RUN:-}}}"
if [[ "$RUN_TOKEN" == --run* ]]; then RUN_TOKEN="${RUN_TOKEN#--run}"; RUN_TOKEN="${RUN_TOKEN# }}"; fi
[[ "$RUN_TOKEN" == "--run" ]] && RUN_TOKEN=""

TAG=""; DST="$DST_BASE"; REMOTE_SUB=""
if [[ -n "$RUN_TOKEN" ]]; then
  TAG="$(slug "$RUN_TOKEN")"
  DST="$DST_BASE/$TAG"
  REMOTE_SUB="/$TAG"       # plot_training.py 给了 --run 就会写进 <plots>/<run>/
fi

refresh_list() {
  sync_tool
  echo "云端可选 run（旧 → 新）："
  "$PY_LOCAL" "$SSH" "$(_plot_env)$CLOUD_PY /root/plot_training.py --list-runs" 2>&1 \
    | grep -vE 'credentials|password' | sed 's/^/  /'
  echo
  echo "用法： bash $0 once <run 子串>     # 例：bash $0 once v9c"
}

case "$CMD" in
  list)
    refresh_list
    ;;
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
    echo "[$(date '+%F %T')] 曲线刷新循环启动（每 ${INTERVAL}s；run=${RUN_TOKEN:-最新}）-> $DST"
    # 先立刻刷一次，别让用户干等一个周期
    refresh_once
    while true; do
      sleep "$INTERVAL"
      refresh_once
    done
    ;;
  *)
    echo "用法: $0 {loop|once|stop|list} [run 子串]" >&2
    exit 1
    ;;
esac
