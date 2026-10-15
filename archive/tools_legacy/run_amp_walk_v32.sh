#!/usr/bin/env bash
# ============================================================================
# V32 实验臂训练入口（跑在云端）
# 手册：$1 num_envs $2 iters $3 device $4 tag $5 load_run $6 checkpoint $7 task
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v32_arm}"
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="${7:-Instinct-Amp-Walk-V32B-G1-v0}"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
