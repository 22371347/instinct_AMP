#!/usr/bin/env bash
# 把 **V26 的全部改动**打进 tar 上传（一次 scp）。
#
# V26 只改两处（+ 6 个新文件都在 config/g1 里）：
#   1. `instinctlab/.../utils/wrappers/instinct_rl/algorithms.py`
#      —— 新增 `clip_max_std_schedule`（σ 上限退火）+ 把 `Train/sigma_cap` 打进日志
#   2. `instinctlab/.../config/g1/` 整目录（V26 env/agent cfg + `__init__.py` 注册）
#
# ⚠️ 本轮**不需要**动 `instinct_rl` 那棵树（白化三件套已在云端就位）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH="$SCRIPT_DIR/instinct_ssh.py"
PY="$(command -v python3 || command -v python)"

SRC_ROOT="/root/gpufree-data/instinct_xu/source"
PATCH="/tmp/llp_v26_patch.tgz"

FILES=(
  # ---- ⚠️ **整目录**上传 config/g1（V26 的 cfg 父链很长，本实例上部分历史 cfg 缺失）
  "instinctlab/instinctlab/tasks/locomotion/config/g1"
  # ---- σ 上限退火（wrapper）
  "instinctlab/instinctlab/utils/wrappers/instinct_rl/algorithms.py"
)

cd "$REPO/instinct_xu/source"
for f in "${FILES[@]}"; do
  [[ -e "$f" ]] || { echo "❌ 缺文件：$f" >&2; exit 1; }
done
tar czf "$PATCH" "${FILES[@]}"
echo "打包：$PATCH（$(du -h "$PATCH" | cut -f1)）"
echo "包含 $(tar tzf "$PATCH" | wc -l) 个条目（含 config/g1 全目录）"

"$PY" "$SSH" --put "$PATCH" "/root/llp_v26_patch.tgz"

# ---- 云端脚本 ----
for f in run_amp_walk_v26.sh cloud_launch_v26.sh cloud_verify_v26_cfg.py; do
  "$PY" "$SSH" --put "$SCRIPT_DIR/$f" "/root/$f"
done

echo
echo "=== 云端解开 + 落地检查 ==="
"$PY" "$SSH" "set -e; cd $SRC_ROOT && tar xzf /root/llp_v26_patch.tgz && \
  echo '--- 解开后的关键文件（应与本地一致）---' && \
  ls -l instinctlab/instinctlab/tasks/locomotion/config/g1/amp_walk_v26_env_cfg.py \
        instinctlab/instinctlab/tasks/locomotion/config/g1/agents/amp_walk_v26_ppo_cfg.py \
        instinctlab/instinctlab/utils/wrappers/instinct_rl/algorithms.py \
        /root/cloud_verify_v26_cfg.py && \
  echo '--- 关键改动是否在位 ---' && \
  echo -n 'clip_max_std_schedule 出现次数: ' && \
  grep -c 'clip_max_std_schedule' instinctlab/instinctlab/utils/wrappers/instinct_rl/algorithms.py && \
  echo -n 'sigma_cap 日志键: ' && \
  grep -c 'sigma_cap' instinctlab/instinctlab/utils/wrappers/instinct_rl/algorithms.py && \
  echo -n 'V26 注册行: ' && \
  grep -c 'Instinct-Amp-Walk-V26-G1-v0' instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py"

echo
echo "接着：bash /root/cloud_launch_v26.sh   （静态核验 → 从 V24@8000 热启动 V26）"
