#!/usr/bin/env bash
# 正式训练：G1 AMP 行走特权策略（Instinct-Amp-Walk-G1-v0）
#
# 用法：
#   bash cloud_train_amp_walk.sh [num_envs] [max_iters] [device] [resume]
#   bash cloud_train_amp_walk.sh 4096 3000 cuda:0
#   bash cloud_train_amp_walk.sh 4096 3000 cuda:0 --resume        # 从 checkpoints 续训
#
# ⚠️ 一定要通过 cloud_run_bg.sh 挂后台跑（SSH 长连接会断，断掉就丢输出）：
#   bash /root/cloud_run_bg.sh 'bash /root/cloud_train_amp_walk.sh 4096 3000 cuda:0' /root/train_driver.log
#
# 训练完/中途想看健康度：
#   grep -E 'Mean step foot_sep_mean|Mean step discriminator_reward|mean_reward_0' <log>
#   tensorboard --logdir /root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-3000}"
DEV="${3:-cuda:0}"
RESUME="${4:-}"
TS="$(date +%Y%m%d_%H%M%S)"
XU=/root/gpufree-data/instinct_xu
LOG="/root/train_amp_walk_${INST}_${TS}.log"

cd "$XU" || exit 1
[[ -f /root/.llp_env.sh ]] && source /root/.llp_env.sh
unset PYTHONPATH

PY=/opt/conda/envs/isaaclab/bin/python

{
  echo "############ AMP 行走正式训练 ############"
  echo "时间      : $(date '+%F %T')"
  echo "num_envs  : $INST"
  echo "迭代数    : $ITERS"
  echo "device    : $DEV"
  echo "resume    : ${RESUME:-<从头开始>}"
  echo "日志      : $LOG"
  echo "--------------------------------------------"
  "$PY" - <<'PYEOF'
import platform, torch
print("python :", platform.python_version())
print("torch  :", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu    :", torch.cuda.get_device_name(0))
try:
    import isaacsim
    print("isaacsim:", getattr(isaacsim, "__version__", "?"))
except Exception as e:
    print("isaacsim: <未导入>", e)
try:
    import isaaclab, instinctlab, instinct_rl
    print("isaaclab:", getattr(isaaclab, "__version__", "?"))
    print("instinctlab:", getattr(instinctlab, "__version__", "?"))
    print("instinct_rl:", getattr(instinct_rl, "__version__", "?"))
except Exception as e:
    print("包版本查询失败(不影响训练):", e)
PYEOF
  # 训练前的显存基线，方便和规模试跑对比
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  echo "--------------------------------------------"
} > "$LOG"

python_bin="$PY"
# ⚠️ 必须关掉 Python 块缓冲：stdout 重定向到文件时默认块缓冲，日志会滞后数分钟
# 才落盘（会让人误判“训练没进展”）。
export PYTHONUNBUFFERED=1
# ── 训练 ──────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="${DEV##*:}" ./train.sh \
  --task Instinct-Amp-Walk-G1-v0 \
  --num_envs "$INST" --max_iterations "$ITERS" \
  --device "$DEV" --headless \
  --run_name "amp_walk_${INST}" $RESUME >> "$LOG" 2>&1
RC=$?

{
  echo "======== 训练结束 退出码 $RC  ($(date '+%F %T')) ========"
  echo "--- 最后一轮的关键量 ---"
  grep -E 'Mean step foot_sep_mean|Mean step foot_cross_frac|Mean step discriminator_reward|Mean step amp_style_reward' "$LOG" | tail -4
  grep -E 'mean_reward_0|mean_episode_length' "$LOG" | tail -2
  echo "--- 异常检索 ---"
  grep -nE 'Traceback|KeyError|out of memory|CUDA error' "$LOG" | head -5 || echo "  (无)"
  echo "--- 轮数 / 最终耗时 ---"
  # ⚠️ 口径：runner 每 10 轮才打一次表，所以这里数的是**表格数不是轮数**；
  # 且 total_timesteps 只在打表时累加，比真实步数少 10 倍。看稳态吞吐请直接用
  # `Iteration time`（它确实是“一轮”的墙钟时间）。
  echo "打印的表格数（非轮数）: $(grep -c 'Iteration time' "$LOG")"
  echo "每轮耗时（最后一次最接近稳态）:"
  grep -E 'Iteration time' "$LOG" | tail -1
  grep -E 'Total time:' "$LOG" | tail -1
  echo "checkpoints: $(ls -d "$XU"/logs/instinct_rl/g1_amp_walk/*/ 2>/dev/null | tail -1)"
} 2>&1 | tee -a "$LOG"

echo "RC=$RC LOG=$LOG"
