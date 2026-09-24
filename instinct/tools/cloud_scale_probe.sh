#!/usr/bin/env bash
# 规模试跑（scale probe）：测 **显存占用 / 训练吞吐**，用来决定正式训练的 num_envs。
# 依据 CLOUD_ENV_SETUP.md §7 第 4 条：镜像给的 4096 环境是按 24 GB 显存估的，
# 必须先用小规模探一遍再放大，否则 OOM 会白烧一整次启动时间。
#
# 用法：bash cloud_scale_probe.sh [num_envs] [iters] [device]
#   bash cloud_scale_probe.sh 512 30 cuda:0
#   bash cloud_scale_probe.sh 2048 30 cuda:0
#
# 输出：
#   /root/scale_probe.log       训练日志（含每轮 Iteration time）
#   /root/scale_probe_gpu.log   每 5 s 采一次的显存/利用率（峰值看这里）
# 输出（**按 num_envs 分文件**，多次试跑不会互相覆盖）：
#   /root/scale_probe_<N>.log           训练日志（含每轮 Iteration time）
#   /root/scale_probe_<N>_gpu.log       每 5 s 采一次的显存/利用率（峰值看这里）
#   /root/scale_probe_<N>_summary.log   一次性看完的摘要
set -uo pipefail

INST="${1:-512}"
ITERS="${2:-30}"
DEV="${3:-cuda:0}"
LOG=/root/scale_probe_${INST}.log
GPULOG=/root/scale_probe_${INST}_gpu.log
SUM=/root/scale_probe_${INST}_summary.log
XU=/root/gpufree-data/instinct_xu

# ⚠️ 必须关掉 Python 的块缓冲：stdout 被重定向到**文件**时默认是块缓冲，
# 训练日志会滞后数分钟才落盘（实测因此白等了好几轮轮询，还以为训练没进展）。
export PYTHONUNBUFFERED=1

cd "$XU" || exit 1
[[ -f /root/.llp_env.sh ]] && source /root/.llp_env.sh
unset PYTHONPATH

{
  echo "=== 规模试跑 num_envs=$INST iters=$ITERS device=$DEV  ($(date '+%F %T')) ==="
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
} > "$LOG"

# ── 显存/利用率采样器（后台；用 nohup 保证 train 退出后不阻塞）────────────
# 720 × 5 s = 60 min，足够覆盖一次试跑
: > "$GPULOG"
nohup bash -c '
  for _ in $(seq 1 720); do
    printf "%s " "$(date +%T)"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    sleep 5
  done' >> "$GPULOG" 2>&1 &

# ── 训练 ─────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="${DEV##*:}" ./train.sh \
  --task Instinct-Amp-Walk-G1-v0 \
  --num_envs "$INST" --max_iterations "$ITERS" \
  --device "$DEV" --headless \
  --run_name "scale_probe_${INST}" >> "$LOG" 2>&1
RC=$?
echo "=== 训练退出码 $RC  ($(date '+%F %T')) ===" >> "$LOG"

# ── 摘要（方便一次轮询就看懂）─────────────────────────────────────────────
# ⚠️ 写进**独立文件** $SUM，不能追加进 $LOG ——
# 否则摘要里的 `grep -c 'Iteration time' "$LOG"` 会把自己前面刚 echo 的那几行也数进去，
# 轮数直接虚高 3 倍（实测踩过：3 张表被数成 9）。
{
  echo "======== 摘要 num_envs=$INST iters=$ITERS ========"
  echo "退出码: $RC   （0 才算跑完；非 0 看上方 traceback）"
  echo "显存峰值: $(awk -F', ' '{gsub(/ MiB/,"",$1); if ($1+0>m) m=$1+0} END{print m" MiB"}' "$GPULOG")"
  echo
  echo "--- 每轮耗时（最后一次最接近稳态）---"
  grep -E 'Iteration time' "$LOG" | sed 's/^\s*/     /'
  echo "--- Total time（累计）---"
  grep -E 'Total time:' "$LOG" | sed 's/^\s*/     /'
  echo
  echo "--- ⚠️ 口径提醒（别看错）---"
  echo "     runner 每 10 轮才打一次表，所以「Iteration time / Total timesteps」的"
  echo "     行数 = 表格数，**不是轮数**；且 total_timesteps 只在打表时累加，"
  echo "     比真实步数少 10 倍。看稳态吞吐请直接用最后一次的 Iteration time。"
  echo "     → 每轮 steps = num_steps_per_env(24) × num_envs($INST) = $((24*INST))"
  echo
  echo "--- 关键检查 ---"
  grep -E '\[AMP-WALK PASS\]|Mean episode|Mean step foot_sep_mean|Mean step amp_style_reward|Mean step discriminator_reward' "$LOG" | tail -8 | sed 's/^\s*/     /'
  echo
  echo "--- 异常检索 ---"
  grep -nE 'Traceback|KeyError|RuntimeError|out of memory|NaN' "$LOG" | head -5 | sed 's/^/     /' || echo "     无"
} > "$SUM"

# 摘要也附到主日志末尾，但并不影响 $SUM 里的计数（它们已算完）
cat "$SUM" >> "$LOG"

echo "RC=$RC LOG=$LOG GPULOG=$GPULOG SUM=$SUM"
