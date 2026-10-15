#!/usr/bin/env bash
# ============================================================================
# T1n 一键发车（**本地**脚本）
# ============================================================================
# 作用：同步 5 个文件 → 用通用发车器起车（热启动 T1a2 model_7000.pt）→ 等自检。
#
#   TASK = Instinct-Amp-Walk-T1n-G1-v0
#   环境侧：**与 T1l 逐字相同**（`track_base_height` std 已是 0.3；RSI 撤回）
#   算法侧（本臂唯一变量：**判别器的输入度量空间**）：
#       ① `discriminator_kwargs` += `normalizer_class_name="EmpiricalNormalization"`
#       ② `amp_grad_pen_space` "raw" → "normalized"
#       ③ `fresh_discriminator_on_load` False → True（丢弃预训练判别器）
#
# 用法：bash instinct/tools/launch_t1n.sh [TAG]
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

TAG="${1:-t1n_whiten_freshdisc}"
L=instinct_xu/source/instinctlab/instinctlab
R=/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab
SSH=(python3 instinct/tools/instinct_ssh.py)

echo "=== 0) 连通性检查 ==="
"${SSH[@]}" "echo 实例在线" || { echo "❌ 实例未开机 —— 请先开机再跑本脚本"; exit 1; }

echo "=== 1) 同步 5 个文件 ==="
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/amp_walk_t1n_env_cfg.py" \
                "$R/tasks/locomotion/config/g1/amp_walk_t1n_env_cfg.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/__init__.py" \
                "$R/tasks/locomotion/config/g1/__init__.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/agents/amp_walk_t1n_ppo_cfg.py" \
                "$R/tasks/locomotion/config/g1/agents/amp_walk_t1n_ppo_cfg.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/agents/amp_walk_ppo_cfg.py" \
                "$R/tasks/locomotion/config/g1/agents/amp_walk_ppo_cfg.py" || exit 1
# ⭐ 本臂唯一"基础设施"改动：桥接层的 load_state_dict 覆盖（丢弃预训练判别器）
"${SSH[@]}" --put "$L/utils/wrappers/instinct_rl/algorithms.py" \
                "$R/utils/wrappers/instinct_rl/algorithms.py" || exit 1
echo "  OK 5 个文件已上传"
"${SSH[@]}" "grep -c 'T1n' $R/tasks/locomotion/config/g1/__init__.py; \
             grep -n 'fresh_discriminator_on_load' $R/tasks/locomotion/config/g1/agents/amp_walk_ppo_cfg.py; \
             grep -n 'def load_state_dict' $R/utils/wrappers/instinct_rl/algorithms.py"

echo "=== 2) 发车（通用发车器：闸门 + 停旧训练 + 起车 + 守护）==="
"${SSH[@]}" "bash /root/cloud_launch_arm.sh Instinct-Amp-Walk-T1n-G1-v0 $TAG"

echo "=== 3) 等自检（条件等待）==="
"${SSH[@]}" "timeout 280 sh -c 'while ! grep -qa \"AMP-WALK PASS\" /root/train_${TAG}.log 2>/dev/null; do sleep 5; done'; echo '--- READY ---'; grep -aE 'T1n\]|WasabiPPOAmpMix\]|AMP-WALK PASS|resume|wasabi|ValueError|Error' /root/train_${TAG}.log | head -30"

cat <<EOF

=== 必须核对 ===
  * \`resume    : 续训  from <T1a2 run> / model_7000.pt\`
  * ⭐ \`[WasabiPPOAmpMix] fresh_discriminator_on_load=True ⇒ 丢弃 checkpoint 里的 \`discriminator\`\`
    （**这一行是本臂的核心证据**：判别器确实没被加载）
  * \`[T1n][DISCRIMINATOR WHITENING]\` 回读：normalizer_class_name / amp_grad_pen_space='normalized'
    / fresh_discriminator_on_load=True / amp_normalizer_expert_only=True
  * \`[T1n][ENV = T1L]\` track_base_height std=0.3 weight=1.5（证明环境侧零改动）
  * \`[T1l][REGRESS TO V33/MJLAB] 守卫通过；非零奖励项 = 17\`（继承自 T1l 的守卫）
  * \`[AMP-WALK PASS] policy=378 ... rewards=21\`（契约未变）
  * \`[wasabi] ... /root/amp_expert_states_396.pt\`

=== 本臂判据（判别器从零学起，**不要**与 T1l/T1m 比数值，看绝对水平）===
  * ⭐ \`Loss/discriminator_loss\`：应明显**大于** T1l 的 ≈0.017（判别器在学，而不是已收敛）
  * ⭐ \`Step/amp_style_zero_frac\`：目标是从 0.35~0.38 **降下来**（死区缩小）
  * ⭐ \`Step/amp_d\` vs \`amp_d_ref\`：**gap 应该变小**（从 1.86 降下来 = 不再一击分开）
  * \`Loss/discriminator_gradient_penalty\`：白化后归一化空间下应与旧量级相当（文档说 1.0~3.2×）
  * \`Mean episode length\` / \`vx_high\`：**必须不塌**（判别器若反过来主导会把策略压坏）
  * 若前 200 轮内 \`amp_style_zero_frac → 0\` **且** \`discriminator_loss → 0\`：判别器又饱和了
    ⇒ 下一刀把 \`amp_grad_pen_space\` 之外再加 mjlab 的 \`clip_obs=10\`（我们缺这个夹紧）
EOF
