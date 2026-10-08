#!/usr/bin/env bash
# 上传 V27（判别器"重启"：去白化 + drop_amp_discriminator）。
# 只改 config/g1（V27 env/agent cfg + 注册）——**不需要**重传 instinct_rl / wrapper
# （白化三件套与 σ 退火机制都已在云端就位）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH="$SCRIPT_DIR/instinct_ssh.py"
PY="$(command -v python3 || command -v python)"

SRC_ROOT="/root/gpufree-data/instinct_xu/source"
PATCH="/tmp/llp_v27_patch.tgz"

FILES=(
  "instinctlab/instinctlab/tasks/locomotion/config/g1"
)

cd "$REPO/instinct_xu/source"
for f in "${FILES[@]}"; do
  [[ -e "$f" ]] || { echo "❌ 缺文件：$f" >&2; exit 1; }
done
tar czf "$PATCH" "${FILES[@]}"
echo "打包：$PATCH（$(du -h "$PATCH" | cut -f1)）"

"$PY" "$SSH" --put "$PATCH" "/root/llp_v27_patch.tgz"

for f in run_amp_walk_v27.sh cloud_launch_v27.sh cloud_verify_v27_cfg.py; do
  "$PY" "$SSH" --put "$SCRIPT_DIR/$f" "/root/$f"
done

echo
echo "=== 云端解开 + 落地检查 ==="
"$PY" "$SSH" "set -e; cd $SRC_ROOT && tar xzf /root/llp_v27_patch.tgz && \
  ls -l instinctlab/instinctlab/tasks/locomotion/config/g1/amp_walk_v27_env_cfg.py \
        instinctlab/instinctlab/tasks/locomotion/config/g1/agents/amp_walk_v27_ppo_cfg.py \
        /root/cloud_verify_v27_cfg.py && \
  echo -n 'V27 注册行: ' && \
  grep -c 'Instinct-Amp-Walk-V27-G1-v0' instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py && \
  echo -n 'drop_amp_discriminator 引用: ' && \
  grep -c 'drop_amp_discriminator' instinctlab/instinctlab/tasks/locomotion/config/g1/agents/amp_walk_v27_ppo_cfg.py"

echo
echo "接着（明天）：bash /root/cloud_launch_v27.sh   （静态核验 → 从 V26 最新 ckpt 热启动 V27）"
