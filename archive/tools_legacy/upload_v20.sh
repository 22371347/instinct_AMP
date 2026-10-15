#!/usr/bin/env bash
# 把 V20 的全部改动上传到云端（改过这些文件后跑一次即可）。
#
# 上传内容：
#   1) 新观测模块 `amp_body_state.py`（刚体空间 4 项 × 策略/参考两侧）+ 其 `__init__.py`
#   2) V20 的 env cfg / agent cfg + `config/g1/__init__.py`（任务注册）
#   3) 云端启动脚本：`run_amp_walk_v20.sh` / `cloud_launch_v20.sh` / `cloud_verify_v20_cfg.py`
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH="$SCRIPT_DIR/instinct_ssh.py"
PY="$(command -v python3 || command -v python)"

SRC="/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab"

# 1) instinctlab 源码
"$PY" "$SSH" --put "$REPO/instinct_xu/source/instinctlab/instinctlab/envs/mdp/observations/amp_body_state.py" "$SRC/envs/mdp/observations/amp_body_state.py"
"$PY" "$SSH" --put "$REPO/instinct_xu/source/instinctlab/instinctlab/envs/mdp/observations/__init__.py" "$SRC/envs/mdp/observations/__init__.py"
"$PY" "$SSH" --put "$REPO/instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/amp_walk_v20_env_cfg.py" "$SRC/tasks/locomotion/config/g1/amp_walk_v20_env_cfg.py"
"$PY" "$SSH" --put "$REPO/instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/amp_walk_v20_ppo_cfg.py" "$SRC/tasks/locomotion/config/g1/agents/amp_walk_v20_ppo_cfg.py"
"$PY" "$SSH" --put "$REPO/instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py" "$SRC/tasks/locomotion/config/g1/__init__.py"

# 2) 云端脚本
for f in run_amp_walk_v20.sh cloud_launch_v20.sh cloud_verify_v20_cfg.py; do
  "$PY" "$SSH" --put "$SCRIPT_DIR/$f" "/root/$f"
done

echo
echo "已上传。云端自检："
"$PY" "$SSH" "ls -l $SRC/envs/mdp/observations/amp_body_state.py $SRC/tasks/locomotion/config/g1/amp_walk_v20_env_cfg.py /root/cloud_verify_v20_cfg.py"
echo
echo "接着跑：bash $SCRIPT_DIR/cloud_launch_v20.sh 的 #3 段（先核验再启动）"
