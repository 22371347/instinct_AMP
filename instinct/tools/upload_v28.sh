#!/usr/bin/env bash
# 上传 V28（**只放大 AMP 风格项权重**：discriminator_reward_coef 1.0 → 4.0）。
# 只改 config/g1（V28 env/agent cfg + 注册）——**不需要**重传 instinct_rl / wrapper
# （σ 退火、去白化、drop_amp_discriminator 机制都已在云端就位）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH="$SCRIPT_DIR/instinct_ssh.py"
PY="$(command -v python3 || command -v python)"

SRC_ROOT="/root/gpufree-data/instinct_xu/source"
PATCH="/tmp/llp_v28_patch.tgz"

FILES=(
  "instinctlab/instinctlab/tasks/locomotion/config/g1"
)

cd "$REPO/instinct_xu/source"
for f in "${FILES[@]}"; do
  [[ -e "$f" ]] || { echo "❌ 缺文件：$f" >&2; exit 1; }
done
tar czf "$PATCH" "${FILES[@]}"
echo "打包：$PATCH（$(du -h "$PATCH" | cut -f1)）"

"$PY" "$SSH" --put "$PATCH" "/root/llp_v28_patch.tgz"

for f in run_amp_walk_v28.sh cloud_launch_v28.sh cloud_verify_v28_cfg.py; do
  "$PY" "$SSH" --put "$SCRIPT_DIR/$f" "/root/$f"
done

echo
echo "=== 云端解开 + 落地检查 ==="
"$PY" "$SSH" "set -e; cd $SRC_ROOT && tar xzf /root/llp_v28_patch.tgz && \
  ls -l instinctlab/instinctlab/tasks/locomotion/config/g1/amp_walk_v28_env_cfg.py \
        instinctlab/instinctlab/tasks/locomotion/config/g1/agents/amp_walk_v28_ppo_cfg.py \
        /root/cloud_verify_v28_cfg.py && \
  echo -n 'V28 注册行: ' && \
  grep -c 'Instinct-Amp-Walk-V28-G1-v0' instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py && \
  echo -n 'V28 coef 行: ' && \
  grep -n 'discriminator_reward_coef' instinctlab/instinctlab/tasks/locomotion/config/g1/agents/amp_walk_v28_ppo_cfg.py | head -3"

echo
echo "接着：bash /root/cloud_launch_v28.sh   （停 V27 → 静态核验 → 从 V27 最新 ckpt 热启动 V28）"
