#!/usr/bin/env bash
# V30 崩溃守护 = 复用 V29 的守护脚本，只换成 V30 的训练日志。
# 判据（沿用）：① 最近 20 个 log 块 `Mean episode length` 全 <50 ⇒ 停训
#              ② 最近 3 个 log 块 `Mean step action_abs_mean` 全 >1.5 ⇒ 停训
exec bash /root/v29_guard.sh /root/train_amp_walk_v30_4096.log 30 50 20 1.5
