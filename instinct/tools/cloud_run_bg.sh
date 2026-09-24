#!/usr/bin/env bash
# 在云上**后台**跑一条命令，输出写日志，便于轮询。
#
# 为什么要它
# ----------
# Kit 启动 + 建场景在 CPU 上可能要几分钟；而经 SSH 直接跑长命令时
# ① `instinct_ssh.py` 会阻塞到命令结束（看不到中间进度）
# ② 嵌套引号 + `&` 在 SSH 里极易被 shell/pexpect 拆坏（实测日志文件根本没生成）
# 所以：把「怎么跑」固化成脚本文件，scp 上去再执行（本项目已多次踩这个坑）。
#
# 用法：
#   bash cloud_run_bg.sh '<要执行的命令>' [/root/bg.log]
#   tail -f /root/bg.log        # 轮询
set -uo pipefail

CMD="${1:?用法: bash cloud_run_bg.sh '<命令>' [日志路径]}"
LOG="${2:-/root/bg.log}"

rm -f "$LOG"
setsid nohup bash -c "source /root/.llp_env.sh 2>/dev/null; cd /root/gpufree-data/instinct_xu; ${CMD}" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
sleep 3
echo "PID=$PID  LOG=$LOG"
echo "--- 启动 3s 后的日志 ---"
tail -5 "$LOG" 2>/dev/null || echo "(日志还没内容)"
