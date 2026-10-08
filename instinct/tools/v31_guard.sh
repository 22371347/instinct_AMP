#!/usr/bin/env bash
# ============================================================================
# V31 崩溃守护：完全复用 V28/V29 时代验证过的 `v29_guard.sh` 实现，只换日志路径。
#   判据：连续 20 个统计块 `episode length < 50`，或 3 个块 `action_abs_mean > 1.5`
#         ⇒ `pkill -9 -f 'train[.]py'`（V28 的教训：崩了就停，别浪费时间）
# 位置参数（与 v29_guard.sh 一致）：$1 log $2 poll $3 ep_thr $4 n $5 act_thr
# ============================================================================
set -uo pipefail

exec bash /root/v29_guard.sh /root/train_amp_walk_v31_4096.log 30 50 20 1.5
