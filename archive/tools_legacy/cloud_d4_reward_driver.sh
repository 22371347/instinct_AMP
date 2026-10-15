#!/bin/bash
# D4：奖励账目 —— 同 ckpt 跑 noisy / mean 两模式，比较**逐项加权奖励**与逐 env 奖励标准差。
# 回答"σ 造成的噪声罚占了多少、把训练信号的 SNR 压到什么程度"。
# 结果落 /root/d4_v24.txt
set -u

RUN_ROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk
cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab || exit 1
# shellcheck disable=SC1091
source /root/.llp_env.sh
unset PYTHONPATH

D=$(ls -td "$RUN_ROOT"/*v24* 2>/dev/null | head -1)
C=$(ls -t "$D"/model_*.pt 2>/dev/null | head -1 | xargs basename)
echo "[driver] run=$(basename "$D")  ckpt=$C"

D3_OUT=/root/d4_v24.txt /opt/conda/envs/isaaclab/bin/python -u /root/cloud_d3_filter_probe.py \
  --task Instinct-Amp-Walk-V24-G1-v0 \
  --load_run "$D" --checkpoint "$C" \
  --num_envs 128 --steps 600 \
  --modes noisy,mean \
  --dump /root/d4_series.json
echo "[driver] D4 rc=$?"
echo "[driver] D4 DONE"
