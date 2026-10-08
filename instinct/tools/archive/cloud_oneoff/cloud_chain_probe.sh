#!/usr/bin/env bash
# 链式规模试跑：等 2048 那轮退出后，自动接着用**同样时间参数**跑 4096。
# 目的：拿到显存曲线的第三个点，用来确认正式训练规模不会 OOM。
#
# ⚠️ 用 `[s]cale_probe.sh 2048` 的括号技巧匹配进程：这是 pgrep -f 的老坑 ——
# 直接写 `pgrep -f 'scale_probe.sh 2048'` 会把这**条命令自己的 cmdline** 也匹配上
# （因为它的命令行里就含这个字符串），于是永远等不到退出。
set -uo pipefail
TARGET_N="${1:-4096}"
ITERS="${2:-30}"
WAIT_FOR="${3:-2048}"

while pgrep -f "[s]cale_probe.sh ${WAIT_FOR}" >/dev/null; do
  sleep 5
done
echo "=== 前置 ${WAIT_FOR} 试跑已退出，开始 ${TARGET_N} ==="
exec bash /root/cloud_scale_probe.sh "$TARGET_N" "$ITERS" cuda:0
