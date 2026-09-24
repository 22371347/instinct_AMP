#!/usr/bin/env bash
# A/B 对照实验：跑**学长原版 OmniLoco 任务**（纯 PPO + 原奖励 + 原观测 623），
# 参数与我们失败的那轮完全一致（num_envs / iters / device）。
#
# 目的：把「环境/配置被我们改坏了」和「单纯还在训练早期、没学会站」
# 这两件事区分开。判据只看两个量（都在迭代表里）：
#   Mean episode length              —— 原版应该很快涨到几百~上千步
#   Episode_Termination/base_contact —— 原版应该快速下降
# 若原版同样一直摔 → 说明 40 轮就是不够，不是我们的配置问题；
# 若原版能站住 → 说明问题出在我们的改动（新奖励 / 指令分档 / AMP 混合）里。
#
# 用法：bash cloud_ab_baseline.sh [num_envs] [iters] [device]
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-50}"
DEV="${3:-cuda:0}"
TS="$(date +%Y%m%d_%H%M%S)"
XU=/root/gpufree-data/instinct_xu
LOG="/root/ab_baseline_${INST}_${TS}.log"
export PYTHONUNBUFFERED=1

cd "$XU" || exit 1
[[ -f /root/.llp_env.sh ]] && source /root/.llp_env.sh
unset PYTHONPATH

{
  echo "############ A/B 对照：学长原版 OmniLoco（纯 PPO）############"
  echo "时间     : $(date '+%F %T')"
  echo "num_envs : $INST    iters: $ITERS    device: $DEV"
  echo "对照对象 : Instinct-Amp-Walk-G1-v0（我们的 AMP 任务，4096×300 那轮）"
  echo "日志     : $LOG"
  echo "--------------------------------------------"
} > "$LOG"

CUDA_VISIBLE_DEVICES="${DEV##*:}" ./train.sh \
  --task Instinct-Unified-OmniLoco-G1-v0 \
  --num_envs "$INST" --max_iterations "$ITERS" \
  --device "$DEV" --headless \
  --run_name "ab_baseline_omniloco_${INST}" >> "$LOG" 2>&1
RC=$?
echo "=== 训练退出码 $RC  ($(date '+%F %T')) ===" >> "$LOG"

SUM="/root/ab_baseline_${INST}_${TS}_summary.log"
{
  echo "======== A/B 对照摘要（原版 OmniLoco）========"
  echo "退出码: $RC"
  echo "--- 逐表：Mean reward / episode length / base_contact / 噪声std ---"
  grep -E 'Learning iteration|Mean reward:|Mean episode length:|Episode_Termination/base_contact|Mean action noise std' "$LOG" \
    | sed 's/^ *//' | sed 's/^/  /'
  echo "--- 异常 ---"
  grep -nE 'Traceback|KeyError|RuntimeError|out of memory' "$LOG" | head -5 || echo "  无"
  echo "--- 对照结论怎么读 ---"
  echo "  episode length 快速涨到几百+ 且 base_contact 下降"
  echo "    → 原版能站住：问题在我们的改动（新奖励/指令分档/AMP 混合）"
  echo "  episode length 同样停在个位数、base_contact 恒 1.0"
  echo "    → 原版也摔：40~50 轮就是不够，属于正常训练早期"
} > "$SUM"
cat "$SUM" >> "$LOG"
echo "RC=$RC LOG=$LOG SUM=$SUM"
