#!/usr/bin/env bash
# ============================================================================
# T2 专用的**放宽版**崩溃守护（跑在云端）
# ============================================================================
# 为什么需要放宽：T2 是 **RSI 冷启动**（机器人被扔到参考帧的任意相位）+ mjlab 的
# **1~3 s 强推**，早期 "一启动就倒"（~19 步）是**预期形态**，不是崩坏。
#
# 对照数据（T1o 冷启动，无 RSI）：第 4 块 8.5 → 第 10 块 51 → 第 20 块 331
# ⇒ 默认守护（连续 20 块 < 50 即杀）在 T2 上有**误杀风险**。
#
# 本脚本用更宽松的判据：
#   判据①：最近 **30** 块的 Mean episode length 全部 < **30**  ⇒ 才算"彻底卡住"
#   判据②：最近 3 块的 Mean step action_abs_mean 全部 > **2.5** ⇒ 动作失控
# （原始默认：20 块 < 50 / 3 块 > 1.5）
# ============================================================================
set -uo pipefail

LOG=/root/train_t2_mjlabcopy_obs1500_rsi_cold.log
GUARD_LOG=/root/guard_t2_relaxed.log

echo "=== 1) 停掉旧的守护（默认判据那一个）==="
for p in $(pgrep -f 'v29_guard[.]sh'); do echo "  kill -9 $p"; kill -9 "$p"; done
sleep 2
pgrep -af 'v29_guard[.]sh' || echo "  OK 已无守护进程"

echo "=== 2) 用放宽判据重新拉起 ==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh $LOG 30 30 30 2.5" "$GUARD_LOG"
sleep 6
echo "--- 新守护日志 ---"
tail -8 "$GUARD_LOG"
echo "--- 守护进程 ---"
pgrep -af 'v29_guard[.]sh' | head -2
echo "--- 训练进程 ---"
pgrep -af 'train[.]py' | head -1
