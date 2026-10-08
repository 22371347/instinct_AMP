#!/bin/bash
# D3：对 V24@最新 ckpt 跑"探针去噪"诊断（5 个噪声工况）。
#   无噪声真值 = mean 模式；纯 σ 缩放 = noisy(1.0σ) / scale:0.5 / scale:0.25；
#   执行侧低通 = ema:0.5。结果落 /root/d3_v24.txt
set -u

RUN_ROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk
cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab || exit 1
# shellcheck disable=SC1091
source /root/.llp_env.sh
unset PYTHONPATH

D=$(ls -td "$RUN_ROOT"/*v24* 2>/dev/null | head -1)
C=$(ls -t "$D"/model_*.pt 2>/dev/null | head -1 | xargs basename)
echo "[driver] run=$(basename "$D")  ckpt=$C"

D3_OUT=/root/d3_v24.txt /opt/conda/envs/isaaclab/bin/python -u /root/cloud_d3_filter_probe.py \
  --task Instinct-Amp-Walk-V24-G1-v0 \
  --load_run "$D" --checkpoint "$C" \
  --num_envs 128 --steps 600 \
  --modes noisy,mean,scale:0.5,scale:0.25,ema:0.5
echo "[driver] D3 rc=$?"
echo "[driver] D3 DONE"
