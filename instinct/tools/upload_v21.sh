#!/usr/bin/env bash
# 把 V21 的全部改动上传到云端（云端开机后跑一次即可）。
#
# 上传内容：
#   1) V21 的 env cfg（刚体 195 维 × 8 帧）+ agent cfg（从零）
#   2) `config/g1/__init__.py`（任务注册，含 V20/V21）
#   3) 云端脚本：`run_amp_walk_v21.sh` / `cloud_launch_v21.sh` / `cloud_verify_v21_cfg.py`
#
# 注：V21 **复用** V20 已上传的 `envs/mdp/observations/amp_body_state.py`（刚体观测函数）
#     —— 若那批文件在云端还在（同一实例），无需重复上传；换机器/重装时才需要，
#     那时先跑 `bash instinct/tools/upload_v20.sh`。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH="$SCRIPT_DIR/instinct_ssh.py"
PY="$(command -v python3 || command -v python)"

SRC="/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab"

"$PY" "$SSH" --put "$REPO/instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/amp_walk_v21_env_cfg.py" "$SRC/tasks/locomotion/config/g1/amp_walk_v21_env_cfg.py"
"$PY" "$SSH" --put "$REPO/instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/amp_walk_v21_ppo_cfg.py" "$SRC/tasks/locomotion/config/g1/agents/amp_walk_v21_ppo_cfg.py"
"$PY" "$SSH" --put "$REPO/instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py" "$SRC/tasks/locomotion/config/g1/__init__.py"

for f in run_amp_walk_v21.sh cloud_launch_v21.sh cloud_verify_v21_cfg.py; do
  "$PY" "$SSH" --put "$SCRIPT_DIR/$f" "/root/$f"
done

echo
echo "已上传。云端自检（确认 V21 文件与刚体观测函数都在）："
"$PY" "$SSH" "ls -l $SRC/tasks/locomotion/config/g1/amp_walk_v21_env_cfg.py \
  $SRC/envs/mdp/observations/amp_body_state.py /root/cloud_verify_v21_cfg.py"
echo
echo "接着：bash /root/cloud_launch_v21.sh   （停旧训练 → 静态核验 → 从零起 V21）"
