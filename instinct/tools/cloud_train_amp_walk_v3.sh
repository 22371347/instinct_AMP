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
#   # 续训（第 5 个参数 = load_run 目录名；非空即触发 --resume）：
#   # ⭐ 默认**无上限**，训练一直跑到人工叫停（用户自己盯曲线决定何时停）
#   bash /root/cloud_run_bg.sh \
#     'bash /root/cloud_train_amp_walk_v3.sh 4096 1000000000 cuda:0 amp_walk_v3c_cont <run> model_5000.pt' \
#     /root/train_amp_walk_v3c_cont.log
#
# 参数：
#   $1 num_envs   (默认 4096)
#   $2 max_iters  (默认 **1000000000 = 实质无上限**)  —— ✅ 语义是**相对**的：
#                                `OnPolicyRunner.learn(num_learning_iterations)` 里是
#                                `tot_iter = current_learning_iteration + num_learning_iterations`
#                                `while current_learning_iteration < tot_iter:`
#                                ⇒ 传入值 = **从当前点还要再训多少轮**（不是绝对终点）。
#                                `runner.load()` 已恢复 `current_learning_iteration`，
#                                所以续训想“再训 5000 轮”就传 5000（**不是** 10000）。
#                                ⚠️ 实测：v2 那次 `--resume`(from 5000) + 5000 显示的就是
#                                `7410/10000`；v3c_cont 传 10⁹ 显示 `5000/1000005000`。
#                                LR 是固定 schedule（cfg 里无 `schedule` 字段），不受影响。
#   $3 device     (默认 cuda:0)
#   $4 run_name   (默认 amp_walk_v3c_<num_envs>) —— 拼在 AMP_WALK_V3_RUN_NAME 后面
#   $5 load_run   (可选) 要续训的 run 目录名；**非空即自动加 --resume**
#   $6 checkpoint (默认 model_5000.pt)
#
# ⚠️ 续训会重置 AMP 参考动作的播放游标（它不在 checkpoint 里）：所有环境从
#    参考 clip 的第 0 帧重新开始。这是小不连续，可接受，但现象上首轮会有扰动。
#
# 启动后核验（必须全绿才算真的起来了）：
#   [AMP-WALK PASS] ... rewards=14        ← 奖励集项数
#   Learning iteration 0/5000             ← 确实从零
#   init_noise_std 0.3 / clip_min_std 0.015
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
#: 默认**无上限**：一直跑到人工叫停（用户自己盯曲线，按存点决定何时停）。
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v3c_${INST}}"
LOAD_RUN="${5:-}"
CKPT="${6:-model_5000.pt}"
#: $7 任务 id（默认 v3）。v4 用 `Instinct-Amp-Walk-V4-G1-v0`。
TASK_ARG="${7:-Instinct-Amp-Walk-V3-G1-v0}"

# 续训还是从零：resume 相关参数只在脚本**内部**拼，外层命令行保持“全是单 token”，
# 这样经 cloud_run_bg.sh → bash -c → train.py 三层传递时不会被引号拆坏。
RESUME_FLAGS=()
if [[ -n "$LOAD_RUN" ]]; then
  RESUME_FLAGS=(--resume --load_run "$LOAD_RUN" --checkpoint "$CKPT")
  RESUME_DESC="续训  from $LOAD_RUN / $CKPT"
else
  RESUME_DESC="<从头开始>"
fi

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
  echo "task      : $TASK_ARG"
  echo "run_name  : $TAG"
  echo "python    : $PY"
  echo "resume    : $RESUME_DESC"
  echo "--------------------------------------------"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  echo "--------------------------------------------"
}

CUDA_VISIBLE_DEVICES="${DEV##*:}" "$PY" -u scripts/instinct_rl/train.py \
  --task "$TASK_ARG" \
  --num_envs "$INST" --max_iterations "$ITERS" \
  --device "$DEV" --headless \
  --run_name "$TAG" ${RESUME_FLAGS[@]+"${RESUME_FLAGS[@]}"}
RC=$?

echo "======== 训练结束 退出码 $RC  ($(date '+%F %T')) ========"
echo "RC=$RC"
