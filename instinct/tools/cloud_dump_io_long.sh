#!/usr/bin/env bash
# 云端：dump P1 play 环境的长时间轨迹（10 s = 500 步）供本地长程对照
set -uo pipefail
XU=/root/gpufree-data/instinct_xu
cd "$XU" || exit 1
[[ -f /root/.llp_env.sh ]] && source /root/.llp_env.sh
unset PYTHONPATH
export PYTHONPATH="$XU/source/instinctlab:$XU/source/instinct_rl"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
python scripts/instinct_rl/dump_io.py \
  --task Instinct-Amp-Walk-Pelvis-G1-Play-v0 \
  --load_run 20261013_162510_p1_pelvis_official_rev10_cold_p1_pelvis_official_rev10 \
  --checkpoint model_2000.pt \
  --steps 500 --out /root/dump_io_p1_long.npz \
  --headless --device cuda:0
