#!/usr/bin/env bash
# 上传 V32 实验臂所需的代码：**2 个库文件 + config/g1 + 3 个脚本**。
#   * `instinct_rl/algorithms/wasabi.py`                       ← 新增"判别器梯度归因"只读探针（Train/disc_grad/*）
#   * `instinctlab/.../utils/ckpt_manipulators.py`             ← 新增 `reset_amp_normalizer`（白化口径切换必须清零）
#   * `instinctlab/.../config/g1`                              ← V32B/V32C env cfg + v32b agent cfg + 注册
# ⚠️ 专家数据集 `/root/amp_expert_states.pt` 口径未变 ⇒ 不重传。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH="$SCRIPT_DIR/instinct_ssh.py"
PY="$(command -v python3 || command -v python)"

SRC_ROOT="/root/gpufree-data/instinct_xu/source"
PATCH="/tmp/llp_v32_patch.tgz"

FILES=(
  "instinct_rl/instinct_rl/algorithms/wasabi.py"
  "instinctlab/instinctlab/utils/ckpt_manipulators.py"
  "instinctlab/instinctlab/tasks/locomotion/config/g1"
)

cd "$REPO/instinct_xu/source"
for f in "${FILES[@]}"; do
  [[ -e "$f" ]] || { echo "❌ 缺文件：$f" >&2; exit 1; }
done
tar czf "$PATCH" "${FILES[@]}"
echo "打包：$PATCH（$(du -h "$PATCH" | cut -f1)）"

"$PY" "$SSH" --put "$PATCH" "/root/llp_v32_patch.tgz"

for f in run_amp_walk_v32.sh cloud_launch_v32.sh; do
  "$PY" "$SSH" --put "$SCRIPT_DIR/$f" "/root/$f"
done

echo
echo "=== 云端解开 + 落地检查 ==="
"$PY" "$SSH" "set -e; cd $SRC_ROOT && tar xzf /root/llp_v32_patch.tgz && \
  echo -n '① 白化重置器: ' && grep -c 'def reset_amp_normalizer' instinctlab/instinctlab/utils/ckpt_manipulators.py && \
  echo -n '② 梯度归因探针: ' && grep -c '_disc_grad_attribution' instinct_rl/instinct_rl/algorithms/wasabi.py && \
  echo -n '③ 归因接进日志: ' && grep -c 'disc_grad/' instinct_rl/instinct_rl/algorithms/wasabi.py && \
  echo -n '④ B 臂白化开关: ' && grep -n 'amp_normalizer_expert_only = NORMALIZER' instinctlab/instinctlab/tasks/locomotion/config/g1/agents/amp_walk_v32b_ppo_cfg.py && \
  echo -n '⑤ C 臂两个权重: ' && grep -nE 'AMP_WALK_V32C_(AIR_TIME|ARMS_DEVIATION)_WEIGHT = ' instinctlab/instinctlab/tasks/locomotion/config/g1/amp_walk_v32_env_cfg.py && \
  echo -n '⑥ 注册行（应为 4）: ' && grep -c 'Instinct-Amp-Walk-V32' instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py && \
  echo '--- V31 现有 ckpt（找 fork 点用）---' && ls -1 /root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk/*v31*/model_*.pt 2>/dev/null | sed 's#.*/##' | sort -u | tr '\n' ' '; echo"

echo
echo "接着（等 V31 到 11500 后）：bash /root/cloud_launch_v32.sh A   # 然后 B、C，串行"
