#!/usr/bin/env bash
# ============================================================================
# 干净复核 AMP history 的帧语义（V15=2 帧 / V16=10 帧 各跑一次）
# ============================================================================
# 目的（上一轮误报的纠正）：用"逐 env 滑窗断言 + reset 检测"确认
#   ① 每个 term 每步恰好 append 一次（append 计数）；
#   ② frame0@t == frameLast@t-1（非 reset env）⇒ 帧是**连续控制步**；
#   ③ 末帧 ≠ 首帧（h=2 时）⇒ 2 帧不是复制；h=10 填满后 10 帧互不相同。
# 判据：非 reset env 的滑窗 max < 1e-6；h=2 的 |末帧−首帧| > 0。
#
# ⚠️ 必须在 V16 训练同时在跑（占 14 核 CPU）时也能量出结果；跑两次约 5~8 分钟。
# 用法：bash /root/cloud_run_bg.sh 'bash /root/check_amp_history.sh' /root/_append2_wrap.log
# ============================================================================
set -uo pipefail

PY=/opt/conda/envs/isaaclab/bin/python
OUT=/root/_append2.txt
: > "$OUT"

echo "########## V15（2 帧 / 134 维）##########" >> "$OUT"
"$PY" /root/cloud_probe_amp_append.py --task Instinct-Amp-Walk-V15-G1-v0 \
    --num_envs 4 --steps 12 --device cuda:0 --headless >> "$OUT" 2>&1

echo "########## V16（10 帧 / 670 维）##########" >> "$OUT"
"$PY" /root/cloud_probe_amp_append.py --task Instinct-Amp-Walk-V16-G1-v0 \
    --num_envs 4 --steps 14 --device cuda:0 --headless >> "$OUT" 2>&1

echo "########## DONE ##########" >> "$OUT"
