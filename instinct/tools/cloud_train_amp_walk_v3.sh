#!/usr/bin/env bash
# ============================================================================
# 正式训练 v3：G1 AMP 行走特权策略（Instinct-Amp-Walk-V3-G1-v0，奖励集 14 项）
#
# 与 v1 的 cloud_train_amp_walk.sh 的区别：
#   1. task 是 **V3**（命令分布 = AMP_mjlab v4，参考数据集 = 精选 7 clip）；
#   2. 奖励集 14 项（v2 的 12 项 + 恢复的 flat_orientation_l2 / joint_deviation_torso）；
#   3. **默认从零开始**（不带 --resume）—— 奖励语义变了，续训没有意义；
#   4. 直接调 scripts/instinct_rl/train.py，不用 engine 里的 train.sh。
#
# 用法（⚠️ 必须通过 cloud_run_bg.sh 挂后台，SSH 长连接断了会丢输出）：
#   bash /root/cloud_run_bg.sh \
#     'bash /root/cloud_train_amp_walk_v3.sh 4096 5000 cuda:0 amp_walk_v3c_4096' \
#     /root/train_amp_walk_v3c_4096.log
#
# 参数：
#   $1 num_envs   (默认 4096)
#   $2 max_iters  (默认 5000)
#   $3 device     (默认 cuda:0)
#   $4 run_name   (默认 amp_walk_v3c_<num_envs>) —— 会拼在 AMP_WALK_V3_RUN_NAME 后面
#
# 启动后核验（必须全绿才算真的起来了）：
#   [AMP-WALK PASS] ... rewards=14        ← 奖励集项数
#   Learning iteration 0/5000             ← 确实从零
#   init_noise_std 0.3 / clip_min_std 0.015
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-5000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v3c_${INST}}"

XU=/root/gpufree-data/instinct_xu
PY=/opt/conda/envs/isaaclab/bin/python

cd "$XU" || exit 1
[[ -f /root/.llp_env.sh ]] && source /root/.llp_env.sh
unset PYTHONPATH
# ⚠️ 必须关掉 Python 块缓冲：stdout 重定向到文件时默认块缓冲，日志会滞后数分钟才落盘
# （会让人误判"训练没进展"）。
export PYTHONUNBUFFERED=1

{
  echo "############ AMP 行走 v3 正式训练（奖励集 14 项）############"
  echo "时间      : $(date '+%F %T')"
  echo "task      : Instinct-Amp-Walk-V3-G1-v0"
  echo "num_envs  : $INST"
  echo "迭代数    : $ITERS"
  echo "device    : $DEV"
  echo "run_name  : $TAG"
  echo "python    : $PY"
  echo "resume    : <从头开始>"
  echo "--------------------------------------------"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  echo "--------------------------------------------"
}

CUDA_VISIBLE_DEVICES="${DEV##*:}" "$PY" -u scripts/instinct_rl/train.py \
  --task Instinct-Amp-Walk-V3-G1-v0 \
  --num_envs "$INST" --max_iterations "$ITERS" \
  --device "$DEV" --headless \
  --run_name "$TAG"
RC=$?

echo "======== 训练结束 退出码 $RC  ($(date '+%F %T')) ========"
echo "RC=$RC"
