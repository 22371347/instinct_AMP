#!/usr/bin/env bash
# ============================================================================
# 上传 V33 所需的全部代码（**4 处库改动 + config/g1 + 4 个云端脚本**）
# ============================================================================
#   * `instinct_rl/algorithms/wasabi.py`
#       - 梯度归因索引改为**从运行时 `obs_format` 段读**（V32 那版硬编码"帧优先"布局是错的）
#       - `self._amp_obs_segments` 记录判别器输入的分项布局
#   * `instinctlab/envs/mdp/observations/amp_body_state.py`
#       - 新增 `anchor_gravity_b` / `anchor_gravity_b_reference_as_state`（V33-S 的第 5 个 AMP 项）
#   * `instinctlab/tasks/locomotion/mdp/commands.py`
#       - 纯侧向档新增 `side_vel_min`（`|vy| ∈ [side_vel_min, 0.5]`，避开静止门控）
#   * `instinctlab/tasks/locomotion/mdp/rewards_amp.py`
#       - 新增 `track_lin_vel_xy_axis_diag`（无损 x/y 拆分 + 分桶诊断，**顶替**原跟踪项、奖励不变）
#   * `instinctlab/tasks/locomotion/config/g1`（V33 env/agent cfg + 注册）
# ⚠️ `amp_expert_states.pt`（390）口径未变 ⇒ 不重传；396 那份由云端 sweep 自动导出。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH="$SCRIPT_DIR/instinct_ssh.py"
PY="$(command -v python3 || command -v python)"

SRC_ROOT="/root/gpufree-data/instinct_xu/source"
PATCH="/tmp/llp_v33_patch.tgz"

FILES=(
  "instinct_rl/instinct_rl/algorithms/wasabi.py"
  "instinctlab/instinctlab/envs/mdp/observations/amp_body_state.py"
  "instinctlab/instinctlab/tasks/locomotion/mdp/commands.py"
  "instinctlab/instinctlab/tasks/locomotion/mdp/rewards_amp.py"
  "instinctlab/instinctlab/tasks/locomotion/config/g1"
)

cd "$REPO/instinct_xu/source"
for f in "${FILES[@]}"; do
  [[ -e "$f" ]] || { echo "❌ 缺文件：$f" >&2; exit 1; }
done
tar czf "$PATCH" "${FILES[@]}"
echo "打包：$PATCH（$(du -h "$PATCH" | cut -f1)）"

"$PY" "$SSH" --put "$PATCH" "/root/llp_v33_patch.tgz"

for f in run_amp_walk_v33.sh cloud_launch_v33.sh cloud_v33_sweep.sh cloud_verify_v33_cfg.py cloud_export_amp_expert.py; do
  "$PY" "$SSH" --put "$SCRIPT_DIR/$f" "/root/$f"
done

echo
echo "=== 云端解开 + 落地检查 ==="
"$PY" "$SSH" "set -e; cd $SRC_ROOT && tar xzf /root/llp_v33_patch.tgz && \
  echo -n '① 侧向档下限 : ' && grep -c 'side_vel_min' instinctlab/instinctlab/tasks/locomotion/mdp/commands.py && \
  echo -n '② 无损诊断项 : ' && grep -c 'def track_lin_vel_xy_axis_diag' instinctlab/instinctlab/tasks/locomotion/mdp/rewards_amp.py && \
  echo -n '③ 重力通道   : ' && grep -c 'def anchor_gravity_b' instinctlab/instinctlab/envs/mdp/observations/amp_body_state.py && \
  echo -n '④ 归因读布局 : ' && grep -c '_amp_obs_segments' instinct_rl/instinct_rl/algorithms/wasabi.py && \
  echo -n '⑤ V33 注册行 : ' && grep -c 'Instinct-Amp-Walk-V33' instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py && \
  echo -n '⑥ S 臂白化eps: ' && grep -nE 'AMP_V33S_NORMALIZER_EPS = ' instinctlab/instinctlab/tasks/locomotion/config/g1/agents/amp_walk_v33s_ppo_cfg.py && \
  echo -n '⑦ 396 专家集  : ' && grep -n 'AMP_V33S_EXPERT_BUFFER = ' instinctlab/instinctlab/tasks/locomotion/config/g1/agents/amp_walk_v33s_ppo_cfg.py && \
  echo -n '⑧ 导出脚本开关: ' && grep -c 'with_anchor_gravity' /root/cloud_export_amp_expert.py && \
  echo '--- 云端现有 V31 ckpt（fork 点）---' && ls -1 /root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk/*v31*/model_*.pt 2>/dev/null | sed 's#.*/##' | sort -u | tr '\n' ' '; echo; \
  echo '--- 专家集现状 ---' && ls -lh /root/amp_expert_states*.pt 2>/dev/null; echo"

echo
echo "接着：bash /root/cloud_run_bg.sh \"bash /root/cloud_v33_sweep.sh\" /root/v33_sweep.log"
echo "      （顺序 S0 → S1 → A-cold；起 S 臂前会自动导出 396 专家集）"
