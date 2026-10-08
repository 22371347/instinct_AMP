#!/usr/bin/env bash
# 上传 V30（**判别器 = mjlab 原始配方**）。
# ⚠️ 本轮要一起上传**三个库文件**（不像 V27/V28 只改 config/g1）：
#   * `instinct_rl/algorithms/wasabi.py`         ← ①GP λ=10 ②GP 只在专家侧 ③优化器分组 wd ④专家 buffer 载入
#   * `instinct_rl/storage/amp_storage.py`       ← ④参考批从离线数据集 i.i.d. 抽样
#   * `instinctlab/.../wrappers/instinct_rl/algorithms.py` ← 探针改为按**迭代里程碑**采样
# 另外 config/g1 里的 V30 env/agent cfg + 注册。
# ⚠️ 专家数据集 `/root/amp_expert_states.pt` **已在云端**（2934×390，先前导出）⇒ 不重传。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH="$SCRIPT_DIR/instinct_ssh.py"
PY="$(command -v python3 || command -v python)"

SRC_ROOT="/root/gpufree-data/instinct_xu/source"
PATCH="/tmp/llp_v30_patch.tgz"

FILES=(
  "instinct_rl/instinct_rl/algorithms/wasabi.py"
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

"$PY" "$SSH" --put "$PATCH" "/root/llp_v30_patch.tgz"

for f in run_amp_walk_v30.sh cloud_launch_v30.sh cloud_verify_v30_cfg.py v30_guard.sh; do
  "$PY" "$SSH" --put "$SCRIPT_DIR/$f" "/root/$f"
done

echo
echo "=== 云端解开 + 落地检查 ==="
"$PY" "$SSH" "set -e; cd $SRC_ROOT && tar xzf /root/llp_v30_patch.tgz && \
  echo -n '① GP 只在专家侧: ' && grep -c 'discriminator_gp_expert_only' instinct_rl/instinct_rl/algorithms/wasabi.py && \
  echo -n '③ 优化器分组 wd: ' && grep -c 'optimizer_head_weight_decay' instinct_rl/instinct_rl/algorithms/wasabi.py && \
  echo -n '④ 专家 buffer 分支: ' && grep -c 'expert_buffer' instinct_rl/instinct_rl/storage/amp_storage.py && \
  echo -n '④ buffer 载入: ' && grep -c 'amp_expert_buffer_path' instinct_rl/instinct_rl/algorithms/wasabi.py && \
  echo -n '探针按迭代里程碑: ' && grep -c 'amp_probe_steps_per_iter' instinctlab/instinctlab/utils/wrappers/instinct_rl/algorithms.py && \
  echo -n 'V30 注册行: ' && grep -c 'Instinct-Amp-Walk-V30-G1-v0' instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py && \
  echo '--- 专家数据集 ---' && ls -l /root/amp_expert_states.pt && \
  ls -l /root/cloud_verify_v30_cfg.py /root/v30_guard.sh"

echo
echo "接着：bash /root/cloud_launch_v30.sh   （停旧训 → 静态核验 → 从 V27 model_10000 热启动 V30 → 拉起守护）"
