#!/bin/bash
# D2 驱动的串行评测：先 V24（小跳候选），再 V19d（已知会走，作对照）。
# 两者各自用自己的任务 ⇒ AMP 判别器输入维度与 ckpt 匹配，都不需要 --drop_amp。
# 结果分别落 /root/d2_eval_v24.txt、/root/d2_eval_v19d.txt。
set -u

RUN_ROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk
EVAL=/root/cloud_eval_noise_free.py
PY=/opt/conda/envs/isaaclab/bin/python

cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab || exit 1
# shellcheck disable=SC1091
source /root/.llp_env.sh
unset PYTHONPATH

_run() {  # $1=结果文件 $2=task $3=run 目录 glob $4=ckpt
  local out="$1" task="$2" glob="$3" ckpt="$4"
  local d
  d=$(ls -td "$RUN_ROOT"/$glob 2>/dev/null | head -1)
  if [[ -z "$d" ]]; then
    echo "[driver] 找不到 run 目录：$glob" | tee -a "$out"
    return 1
  fi
  echo "[driver] ===== $task  from $(basename "$d") / $ckpt  -> $out ====="
  D2_OUT="$out" "$PY" -u "$EVAL" \
    --task "$task" --load_run "$d" --checkpoint "$ckpt" --num_envs 128 --steps 600
  local rc=$?
  echo "[driver] $task rc=$rc"
  return $rc
}

_run /root/d2_eval_v24.txt  Instinct-Amp-Walk-V24-G1-v0  '*v24*'  model_3500.pt
_run /root/d2_eval_v19d.txt Instinct-Amp-Walk-V19d-G1-v0 '*v19d*' model_10000.pt
echo "[driver] ALL DONE"
