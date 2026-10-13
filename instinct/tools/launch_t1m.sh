#!/usr/bin/env bash
# ============================================================================
# T1m 一键发车（**本地**脚本）
# ============================================================================
# 作用：实例开机后，一条命令完成「同步 → 发车 → 等自检」。
#
#   TASK = Instinct-Amp-Walk-T1m-G1-v0
#   唯一变量：`scene.motion_reference.motion_buffers[*].motion_start_from_middle_range`
#             (0.0, 0.0) → **(0.0, 1.0)**   —— 打开 RSI 相位随机化（对齐 mjlab）
#   起点：热启动自 **T1a2 的 model_7000.pt**（与 T1j/T1i/T1k/T1l 同起点）
#   算法：复用 `amp_walk_t1i_ppo_cfg`（λ=0.975，未动 `coef=4.0`）
#
# 用法：bash instinct/tools/launch_t1m.sh [TAG]
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

TAG="${1:-t1m_rsiphase}"
L=instinct_xu/source/instinctlab/instinctlab/tasks/locomotion
R=/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab/tasks/locomotion
SSH=(python3 instinct/tools/instinct_ssh.py)

echo "=== 0) 连通性检查 ==="
"${SSH[@]}" "echo 实例在线" || { echo "❌ 实例未开机（Connection refused）——请先开机再跑本脚本"; exit 1; }

echo "=== 1) 同步 3 个文件 ==="
# ① T1m 新增配置  ② 注册文件（含 T1m 注册 + 镜像损失/V9i 勘误）  ③ rewards_amp.py（仅注释勘误）
"${SSH[@]}" --put "$L/config/g1/amp_walk_t1m_env_cfg.py" "$R/config/g1/amp_walk_t1m_env_cfg.py" || exit 1
"${SSH[@]}" --put "$L/config/g1/__init__.py"            "$R/config/g1/__init__.py"            || exit 1
"${SSH[@]}" --put "$L/mdp/rewards_amp.py"               "$R/mdp/rewards_amp.py"               || exit 1
echo "  OK 三个文件已上传"
"${SSH[@]}" "grep -c 'T1m' $R/config/g1/__init__.py && ls -l $R/config/g1/amp_walk_t1m_env_cfg.py"

echo "=== 2) 发车（通用发车器：闸门 + 停旧训练 + 起车 + 守护）==="
"${SSH[@]}" "bash /root/cloud_launch_arm.sh Instinct-Amp-Walk-T1m-G1-v0 $TAG"

echo "=== 3) 等自检（条件等待，不是轮询）==="
"${SSH[@]}" "timeout 280 sh -c 'while ! grep -qa \"AMP-WALK PASS\" /root/train_${TAG}.log 2>/dev/null; do sleep 5; done'; echo '--- READY ---'; grep -aE 'T1m\]|AMP-WALK PASS|resume|wasabi|ValueError|Error' /root/train_${TAG}.log | head -25"

cat <<EOF

=== 必须核对 ===
  * \`resume    : 续训  from <T1a2 run> / model_7000.pt\`
  * \`[T1m][RSI PHASE] motion_buffers['...'].motion_start_from_middle_range  (0.0, 0.0) → (0.0, 1.0)\`
  * \`[T1m][RSI PHASE] 奖励守卫通过；非零奖励项 = 17\`
  * \`[AMP-WALK PASS] policy=378 ... rewards=21\`（契约未变）
  * \`[wasabi] ... /root/amp_expert_states_396.pt\`

=== 本臂判据（见 cfg 文件头）===
  * ⭐ \`Loss/discriminator_loss\` / \`Step/amp_d\`：判别器是否"重新有话说"（V27 稳态 ≈0.0014 = 死透）
  * ⭐ \`Style/zero_frac\`：死区零梯度占比应从 ~0.36 下降
  * \`vx_high\` / \`track_lin_vel_xy_exp\`：不能为相位牺牲速度跟踪
  * \`Mean episode length\`：若 <100 则"热启动 + 相位随机化"不可行 ⇒ 下一刀改冷启动或降到 (0.0, 0.5)
EOF
