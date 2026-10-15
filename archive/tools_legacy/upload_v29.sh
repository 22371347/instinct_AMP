#!/usr/bin/env bash
# 上传 V29（**判别器向 AMP_mjlab 对齐**：输入 2 帧 / 专家批解耦 / 有界奖励形状 / 系数回退）。
# ⚠️ 本轮**必须**一起上传两个库文件（不像 V27/V28 只改 config/g1）：
#   * `instinct_rl/instinct_rl/storage/amp_storage.py`  ← 专家批独立重采样（V29 ②）
#   * `instinctlab/.../wrappers/instinct_rl/algorithms.py` ← `bounded_quad` 形状（V29 ③）
#                                                          + 把开关接到 amp storage（V29 ②）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH="$SCRIPT_DIR/instinct_ssh.py"
PY="$(command -v python3 || command -v python)"

SRC_ROOT="/root/gpufree-data/instinct_xu/source"
PATCH="/tmp/llp_v29_patch.tgz"

FILES=(
  "instinct_rl/instinct_rl/storage/amp_storage.py"
  "instinctlab/instinctlab/utils/wrappers/instinct_rl/algorithms.py"
  "instinctlab/instinctlab/tasks/locomotion/config/g1"
)

cd "$REPO/instinct_xu/source"
for f in "${FILES[@]}"; do
  [[ -e "$f" ]] || { echo "❌ 缺文件：$f" >&2; exit 1; }
done
tar czf "$PATCH" "${FILES[@]}"
echo "打包：$PATCH（$(du -h "$PATCH" | cut -f1)）"

"$PY" "$SSH" --put "$PATCH" "/root/llp_v29_patch.tgz"

for f in run_amp_walk_v29.sh cloud_launch_v29.sh cloud_verify_v29_cfg.py v29_guard.sh; do
  "$PY" "$SSH" --put "$SCRIPT_DIR/$f" "/root/$f"
done

echo
echo "=== 云端解开 + 落地检查 ==="
"$PY" "$SSH" "set -e; cd $SRC_ROOT && tar xzf /root/llp_v29_patch.tgz && \
  echo -n '② amp_storage 独立重采样: ' && \
  grep -c 'independent_reference_sampling' instinct_rl/instinct_rl/storage/amp_storage.py && \
  echo -n '③ algorithms bounded_quad: ' && \
  grep -c 'bounded_quad' instinctlab/instinctlab/utils/wrappers/instinct_rl/algorithms.py && \
  echo -n '② 开关接线: ' && \
  grep -c 'amp_expert_independent_sampling' instinctlab/instinctlab/utils/wrappers/instinct_rl/algorithms.py && \
  echo -n '① V29 注册行: ' && \
  grep -c 'Instinct-Amp-Walk-V29-G1-v0' instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py && \
  echo -n '① 2 帧常量: ' && \
  grep -n 'AMP_V29_STATE_HISTORY = ' instinctlab/instinctlab/tasks/locomotion/config/g1/amp_walk_v29_env_cfg.py && \
  ls -l /root/cloud_verify_v29_cfg.py /root/v29_guard.sh"

echo
echo "接着：bash /root/cloud_launch_v29.sh  （停旧训 → 静态核验 → 从 V27 model_10000 热启动 V29 → 拉起崩溃守护）"
