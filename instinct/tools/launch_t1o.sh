#!/usr/bin/env bash
# ============================================================================
# T1o 一键发车（**本地**脚本）：方案 C —— 663 观测契约 + 从 0 轮冷启动
# ============================================================================
# 同步 6 个文件 → 跑云端冷启动脚本 → 等自检。
#
#   TASK = Instinct-Amp-Walk-T1o-G1-v0
#   actor 观测 378（单帧）→ **663** = 279（height_scan 单帧）+ 4×96（本体感受 4 帧）
#   ❌ 删 `base_lin_vel`（对齐 mjlab：它只在 critic；真机也没有）
#   起点 = **从零**（不 resume）
#
# 用法：bash instinct/tools/launch_t1o.sh
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

TAG="t1o_obs663_h4_cold"
L=instinct_xu/source/instinctlab/instinctlab
R=/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab
SSH=(python3 instinct/tools/instinct_ssh.py)

echo "=== 0) 连通性检查 ==="
"${SSH[@]}" "echo 实例在线" || { echo "❌ 实例未开机 —— 请先开机再跑本脚本"; exit 1; }

echo "=== 1) 同步 6 个文件 ==="
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/amp_walk_t1o_env_cfg.py" \
                "$R/tasks/locomotion/config/g1/amp_walk_t1o_env_cfg.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/agents/amp_walk_t1o_ppo_cfg.py" \
                "$R/tasks/locomotion/config/g1/agents/amp_walk_t1o_ppo_cfg.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/__init__.py" \
                "$R/tasks/locomotion/config/g1/__init__.py" || exit 1
# ⭐ 镜像算子改成"帧感知"（原实现遇到 4 帧堆叠会抛形状错误）
"${SSH[@]}" --put "$L/utils/mirror.py" \
                "$R/utils/mirror.py" || exit 1
# [AMP-WALK PASS] 里过期的 "base_lin_vel" 标签修正（纯诊断文案）
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/amp_walk_env_cfg.py" \
                "$R/tasks/locomotion/config/g1/amp_walk_env_cfg.py" || exit 1
# ⭐ 桥接层新增 `policy_obs_frames`（帧数显式传给镜像：isaaclab 只报展平标量）
"${SSH[@]}" --put "$L/utils/wrappers/instinct_rl/algorithms.py" \
                "$R/utils/wrappers/instinct_rl/algorithms.py" || exit 1
# ⭐ 算法 @configclass 新增 `policy_obs_frames` 字段声明
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/agents/amp_walk_ppo_cfg.py" \
                "$R/tasks/locomotion/config/g1/agents/amp_walk_ppo_cfg.py" || exit 1
# 冷启动发车脚本（⚠️ 空字符串必须写在**文件里**，不能走命令行引号）
"${SSH[@]}" --put "instinct/tools/cloud_launch_t1o_remote.sh" \
                "/root/cloud_launch_t1o_remote.sh" || exit 1
echo "  OK 8 个文件已上传"
"${SSH[@]}" "grep -c 'T1o' $R/tasks/locomotion/config/g1/__init__.py; \
             grep -n 'policy_obs_frames' $R/tasks/locomotion/config/g1/agents/amp_walk_t1o_ppo_cfg.py | head -2; \
             grep -n 'frames_by_term=' $R/utils/wrappers/instinct_rl/algorithms.py; \
             grep -n 'out\[..., sl\] = _sign' $R/utils/mirror.py"

echo "=== 2) ⭐ 冷启动发车（不 resume）==="
"${SSH[@]}" "bash /root/cloud_launch_t1o_remote.sh"

echo "=== 3) 等自检（条件等待）==="
"${SSH[@]}" "timeout 300 sh -c 'while ! grep -qa \"AMP-WALK PASS\" /root/train_${TAG}.log 2>/dev/null; do sleep 5; done'; sleep 25; echo '--- READY ---'; grep -aE 'T1o\]|AMP-WALK PASS|resume|wasabi' /root/train_${TAG}.log | head -30; echo; echo '--- ⭐ 存活检查（上一次自检报 READY 时训练其实已经崩了，所以这一项必须有）---'; pgrep -af 'train[.]py' | head -1 || echo '  ❌ train.py 已不在 ⇒ 看下面的 Traceback'; echo; echo '--- 报错（若有）---'; grep -aE 'Traceback|KeyError|RuntimeError|ValueError' -A 2 /root/train_${TAG}.log | tail -24"

cat <<EOF

=== 必须核对 ===
  * ⚠️ **不应**出现 \`resume    : 续训\`（本臂从零）
  * \`[T1o][HISTORY = 4]\` 逐项维度 = (279, 12, 12, 12, 116, 116, 116)（sum=663）
  * \`[T1o][COLD START]\`
  * \`[AMP-WALK PASS] policy=663\`
  * \`[wasabi] ... amp_expert_states_396.pt\`（AMP 状态未变）
  * 若报 \`'joint_pos_rel' 的 shape=... 无法解析为 (frames, dim)\` ⇒ isaaclab 的
    history 形状顺序与我们预期都不同，把那一行发我，镜像里加一种顺序即可。

=== 判据 ===
  * 爬升斜率（对比 \`V33ACold\` 的 378 冷启动）
  * \`vx_high\` / \`track_lin_vel_xy\`：删 base_lin_vel 的代价
  * ⭐ **步态**：抬脚高度 / 交替性 —— 本臂唯一的目的
EOF
