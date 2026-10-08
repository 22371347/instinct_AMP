#!/usr/bin/env bash
# V9 第二轮诊断：最后一轮的**奖励分解**（任务项 vs AMP）与终止原因 / σ 打印精度
# 用法： bash /root/cloud_v9_diag2.sh [/root/train_amp_walk_v9_4096.log]
set -u
L="${1:-/root/train_amp_walk_v9_4096.log}"
[ -f "$L" ] || { echo "找不到 $L"; exit 1; }
n=$(grep -n 'Learning iteration' "$L" | tail -1 | cut -d: -f1)

echo "== 最后一轮：迭代 / 回报 / 长度 / σ =="
sed -n "${n},\$p" "$L" | grep -E 'Learning iteration|Mean reward|Mean episode length|Mean action noise std|Value function' \
  | sed 's/^ *//' | cut -c1-90

echo
echo "== 最后一轮：Episode_Reward/* 逐项（每回合累计；÷ 长度≈每步）=="
sed -n "${n},\$p" "$L" | grep -E 'Episode_Reward/' | sed 's/^ *//' | cut -c1-90 | sort -t: -k2 -g

echo
echo "== 最后一轮：终止原因 =="
sed -n "${n},\$p" "$L" | grep -E 'Episode_Termination/' | sed 's/^ *//' | cut -c1-90 | head -10

echo
echo "== 最后一轮：Mean step amp_* / discriminator_reward =="
sed -n "${n},\$p" "$L" | grep -E 'Mean step (amp_|discriminator_reward)' | sed 's/^ *//' | cut -c1-90

echo
echo "== σ 原始行（看打印精度，最近 3 条）=="
grep -E 'Mean action noise std' "$L" | tail -3 | sed 's/^ *//'

echo
echo "== 镜像损失最近 3 条 =="
grep -E '\bmirror\b' "$L" | tail -3 | sed 's/^ *//' | cut -c1-90
