#!/bin/bash
# Step 0：σ 标定扫描 —— 用 V24 的**同一份权重**，只缩放采样噪声到指定绝对 σ，
# 输出 σ → 噪声税 / 探针 / 步态 的响应曲线，用来给训练实验选 σ*。
# 结果落 /root/sigma_scan_v24.txt
set -u

RUN_ROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk
cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab || exit 1
# shellcheck disable=SC1091
source /root/.llp_env.sh
unset PYTHONPATH

D=$(ls -td "$RUN_ROOT"/*v24* 2>/dev/null | head -1)
C=model_8000.pt
echo "[driver] run=$(basename "$D")  ckpt=$C"

D3_OUT=/root/sigma_scan_v24.txt /opt/conda/envs/isaaclab/bin/python -u /root/cloud_d3_filter_probe.py \
  --task Instinct-Amp-Walk-V24-G1-v0 \
  --load_run "$D" --checkpoint "$C" \
  --num_envs 128 --steps 600 \
  --modes noisy,abs:0.30,abs:0.20,abs:0.15,abs:0.10,mean \
  --dump /root/sigma_scan_series.json
echo "[driver] sigma scan rc=$?"
echo "[driver] SIGMA SCAN DONE"
