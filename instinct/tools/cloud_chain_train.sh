#!/usr/bin/env bash
# ============================================================================
# 链式续训：等当前 play/eval 退出后，从指定 checkpoint 继续训练
#
# 为什么要有它
#   play.py 的录像/导出会占用 GPU；直接同时启动续训会互相抢显存与算力。
#   这里先等 `[p]lay.py` 退出（括号技巧防自匹配：不加括号会匹配到本脚本
#   自己的命令行，导致永远等不到退出）。
#
# 为什么显式给 --load_run / --checkpoint
#   只写 `--resume` 时 runner 会去挑"最新 run 目录"，而 play 等操作也可能
#   在实验目录下留下新目录 → 有挑错 run 的风险。显式指定最稳。
#
# 不动任何奖励函数与权重（用户的硬要求）：只延长训练轮数。
#
# 用法：bash cloud_chain_train.sh [续训轮数] [num_envs]
#   bash cloud_chain_train.sh 5000 4096
# ============================================================================
set -uo pipefail

ITERS="${1:-5000}"
INST="${2:-4096}"
RUN="20260925_003514_G1AmpWalkPlane_obs626_directional_rew5_amp_walk_4096"
CKPT="model_5000.pt"

echo "=== 链式续训准备 ==="
echo "  等待 play/eval 退出 ..."
while pgrep -f '[p]lay.py' >/dev/null; do
  sleep 10
done
echo "  play/eval 已退出，开始续训：run=$RUN ckpt=$CKPT +$ITERS 轮 × $INST 环境"

exec bash /root/cloud_train_amp_walk.sh "$INST" "$ITERS" cuda:0 \
  "--resume --load_run $RUN --checkpoint $CKPT"
