#!/usr/bin/env bash
# 上传 V31：**只改 config/g1**（env cfg + agent cfg + 注册）—— 库文件（wasabi / amp_storage /
# wrappers）与 V30 **逐字相同**，无需重传；专家数据集 `/root/amp_expert_states.pt` 已在云端。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH="$SCRIPT_DIR/instinct_ssh.py"
PY="$(command -v python3 || command -v python)"

SRC_ROOT="/root/gpufree-data/instinct_xu/source"
PATCH="/tmp/llp_v31_patch.tgz"

FILES=(
  "instinctlab/instinctlab/tasks/locomotion/config/g1"
)

cd "$REPO/instinct_xu/source"
for f in "${FILES[@]}"; do
  [[ -e "$f" ]] || { echo "❌ 缺文件：$f" >&2; exit 1; }
done
tar czf "$PATCH" "${FILES[@]}"
echo "打包：$PATCH（$(du -h "$PATCH" | cut -f1)）"

"$PY" "$SSH" --put "$PATCH" "/root/llp_v31_patch.tgz"

for f in run_amp_walk_v31.sh cloud_launch_v31.sh v31_guard.sh; do
  "$PY" "$SSH" --put "$SCRIPT_DIR/$f" "/root/$f"
done

echo
echo "=== 云端解开 + 落地检查 ==="
"$PY" "$SSH" "set -e; cd $SRC_ROOT && tar xzf /root/llp_v31_patch.tgz && \
  echo -n '① V31 权重常量: ' && grep -n 'AMP_WALK_V31_FLAT_ORIENTATION_WEIGHT = ' instinctlab/instinctlab/tasks/locomotion/config/g1/amp_walk_v31_env_cfg.py && \
  echo -n '② V31 取消 drop-disc: ' && grep -c 'ckpt_manipulator = None' instinctlab/instinctlab/tasks/locomotion/config/g1/agents/amp_walk_v31_ppo_cfg.py && \
  echo -n '③ V31 注册行: ' && grep -c 'Instinct-Amp-Walk-V31-G1-v0' instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py && \
  echo -n '④ 库文件仍在（沿用 V30）: ' && grep -c 'discriminator_gp_expert_only' instinct_rl/instinct_rl/algorithms/wasabi.py && \
  ls -l /root/cloud_launch_v31.sh /root/run_amp_walk_v31.sh /root/v31_guard.sh"

echo
echo "接着：bash /root/cloud_launch_v31.sh"
