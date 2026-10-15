#!/usr/bin/env bash
# 把 V24 的**全部**改动打成 tar 上传（一次 scp 搞定，含之前改过的 instinct_rl 树）。
#
# 为什么要 tar：本轮改动跨两棵树、共 11 个文件（instinctlab 8 + instinct_rl 3），
# 逐个 `--put` 要 11 次 SSH。tar 保留相对路径 ⇒ 云端一次解开即可。
#
# ⚠️ 云端 `instinct_rl` 是 **editable 安装**（`pip install -e source/instinct_rl`），
#    白化的 `normalizer.py` / `discriminator.py` / `wasabi.py` **必须**一并上传，
#    否则 `Discriminator.forward` 会因为多一个 kwarg 直接 TypeError。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH="$SCRIPT_DIR/instinct_ssh.py"
PY="$(command -v python3 || command -v python)"

SRC_ROOT="/root/gpufree-data/instinct_xu/source"
PATCH="/tmp/llp_v24_patch.tgz"

FILES=(
  # ---- instinctlab：奖励函数 + 新数学模块 ----
  "instinctlab/instinctlab/tasks/locomotion/mdp/rewards_amp.py"
  "instinctlab/instinctlab/tasks/locomotion/mdp/amp_reward_utils.py"
  # ---- ⚠️ **整目录**上传 config/g1：V24 的 cfg 父链很长
  #      （v24 → v21 → v20 → v19d → …），而本实例上 V21 从没跑过 ⇒ 其 cfg 文件缺失，
  #      只传 V24 自己的文件会 ModuleNotFoundError: ... amp_walk_v21_env_cfg。tar 整目录最稳。
  "instinctlab/instinctlab/tasks/locomotion/config/g1"
  "instinctlab/instinctlab/utils/wrappers/instinct_rl/algorithms.py"
  # ---- instinct_rl（editable 安装的那棵树）：白化实现 ----
  "instinct_rl/instinct_rl/modules/normalizer.py"
  "instinct_rl/instinct_rl/modules/discriminator.py"
  "instinct_rl/instinct_rl/algorithms/wasabi.py"
)

cd "$REPO/instinct_xu/source"
for f in "${FILES[@]}"; do
  [[ -e "$f" ]] || { echo "❌ 缺文件：$f" >&2; exit 1; }
done
tar czf "$PATCH" "${FILES[@]}"
echo "打包：$PATCH（$(du -h "$PATCH" | cut -f1)）"
echo "包含 $(tar tzf "$PATCH" | wc -l) 个条目（含 config/g1 全目录）"

"$PY" "$SSH" --put "$PATCH" "/root/llp_v24_patch.tgz"

# ---- 云端脚本 ----
for f in run_amp_walk_v24.sh cloud_launch_v24.sh cloud_verify_v24_cfg.py \
         cloud_dump_stand_probes.py cloud_dump_stand_accounting.py cloud_dump_tags.py; do
  "$PY" "$SSH" --put "$SCRIPT_DIR/$f" "/root/$f"
done

echo
echo "=== 云端解开 + 落地检查 ==="
"$PY" "$SSH" "set -e; cd $SRC_ROOT && tar xzf /root/llp_v24_patch.tgz && \
  echo '--- 解开后的关键文件（应与本地一致）---' && \
  ls -l instinctlab/instinctlab/tasks/locomotion/mdp/amp_reward_utils.py \
        instinctlab/instinctlab/tasks/locomotion/config/g1/amp_walk_v24_env_cfg.py \
        instinct_rl/instinct_rl/modules/normalizer.py \
        instinct_rl/instinct_rl/algorithms/wasabi.py /root/cloud_verify_v24_cfg.py && \
  echo '--- 关键改动是否在位（应有 1 行）---' && \
  grep -c 'progress_kappa' instinctlab/instinctlab/tasks/locomotion/mdp/rewards_amp.py && \
  grep -c 'update_normalizer' instinct_rl/instinct_rl/algorithms/wasabi.py && \
  echo '--- instinct_rl 是否 editable 安装 ---' && \
  /opt/conda/envs/isaaclab/bin/python -c \"import instinct_rl, sys; print(instinct_rl.__file__)\""

echo
echo "接着：bash /root/cloud_launch_v24.sh   （停旧训练 → 静态核验 → 从零起 V24）"
