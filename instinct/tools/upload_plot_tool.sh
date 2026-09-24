#!/usr/bin/env bash
# 把云端绘图脚本上传上去（新环境/改过 plot_training.py 后跑一次即可）。
# 之后用 refresh_plots.sh 定期刷新 + 拉回本地。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH="$SCRIPT_DIR/instinct_ssh.py"
PY="$(command -v python3 || command -v python)"

for f in plot_training.py; do
  "$PY" "$SSH" --put "$SCRIPT_DIR/$f" "/root/$f"
done

echo
echo "已上传。云端自检（列出最新 run 的 tag 数）："
"$PY" "$SSH" "/opt/conda/envs/isaaclab/bin/python /root/plot_training.py --list-tags | head -4"
echo
echo "接着跑：bash $SCRIPT_DIR/refresh_plots.sh once    # 立刻出图并拉回本地"
echo "        bash $SCRIPT_DIR/refresh_plots.sh         # 后台每 10 min 自动刷新"
