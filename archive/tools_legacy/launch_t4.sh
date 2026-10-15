#!/usr/bin/env bash
# ============================================================================
# T4 一键发车（**本地**脚本）：**消融臂** —— T3 之上整项移除 `track_base_height`
# ============================================================================
#   TASK = Instinct-Amp-Walk-T4-G1-v0
#   环境侧唯一改动：`track_base_height` → **None**（字段保留，一行可改回）⇒ 非零奖励项 12→11
#   算法侧：与 T3 逐字相同（`fresh_discriminator_on_load=False`，保住判别器）
#   起点：`t2_mjlabcopy_obs1500_rsi_cold` / `model_11500.pt`（**与 T3 同起点**）
#
# ⚠️ 发车前预判（analyze_gait_trace.py 实测）：该项在真实轨迹上的姿态力仅
#    0.0035 reward/s（总奖励幅度的 0.08%），且 T3 已把它的方向**整体翻转**
#    （+0.14/m 推高 → −0.53/m 压低）而 12600 轮姿态纹丝不动（躯干 +0.81° vs 参考 +10.10°）
#    ⇒ **预计本臂也不改变躯干姿态**。本臂价值 = 把这条腿钉死 + 给 pelvis-root 臂留对照基座。
# ⚠️ 移除后不再有防塌正则（终止只在 0.5 m 兜底）⇒ 出现蹲走只能读成"撤了保险"。
#
# 用法：bash instinct/tools/launch_t4.sh
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

TASK_ID="Instinct-Amp-Walk-T4-G1-v0"
TAG="t4_noh_from11500"
BASE_PAT="t2_mjlabcopy_obs1500_rsi_cold"
CKPT="model_11500.pt"
L=instinct_xu/source/instinctlab/instinctlab
R=/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab
RUNROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk
SSH=(python3 instinct/tools/instinct_ssh.py)

echo "=== 0) 连通性检查 ==="
"${SSH[@]}" "echo 实例在线" || { echo "❌ 实例未开机"; exit 1; }

echo "=== 1) 同步文件（T4 两个新文件 + 改动的 __init__/mujoco 取证工具）==="
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/amp_walk_t4_env_cfg.py" \
                "$R/tasks/locomotion/config/g1/amp_walk_t4_env_cfg.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/agents/amp_walk_t4_ppo_cfg.py" \
                "$R/tasks/locomotion/config/g1/agents/amp_walk_t4_ppo_cfg.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/__init__.py" \
                "$R/tasks/locomotion/config/g1/__init__.py" || exit 1
"${SSH[@]}" --put "instinct/tools/cloud_launch_arm.sh" "/root/cloud_launch_arm.sh" || exit 1
echo "  OK 4 个文件已上传"

echo "=== 2) 远端依赖自检 ==="
"${SSH[@]}" "set -e; \
  for f in $L/tasks/locomotion/config/g1/amp_walk_t3_env_cfg.py \
            $L/tasks/locomotion/config/g1/amp_walk_t2_env_cfg.py \
            /root/run_amp_walk_v33.sh; do
      [ -f \"\$f\" ] && echo \"  OK \$f\" || { echo \"  ❌ 缺 \$f\"; exit 1; }; done; \
  echo '--- 热启动源 ---'; \
  ls -d $RUNROOT/*${BASE_PAT}* 2>/dev/null | tail -1; \
  ls $RUNROOT/*${BASE_PAT}*/${CKPT} 2>/dev/null; \
  echo '--- T4 注册与移除点 ---'; \
  grep -c 'T4' $R/tasks/locomotion/config/g1/__init__.py; \
  grep -n 'T4_REMOVE_TERMS\|T4_EXPECT_NONZERO' $R/tasks/locomotion/config/g1/amp_walk_t4_env_cfg.py | head -4"

echo "=== 3) ⭐ 发车（cloud_launch_arm.sh：解析 LOAD_RUN → 校验 ckpt → kill 旧 → 起车 → 起守护）==="
"${SSH[@]}" "bash /root/cloud_launch_arm.sh $TASK_ID $TAG $BASE_PAT $CKPT 4096" || exit 1

echo "=== 4) 等自检（条件等待 + ⭐ 存活检查）==="
"${SSH[@]}" "timeout 420 sh -c 'while ! grep -qa \"AMP-WALK PASS\" /root/train_${TAG}.log 2>/dev/null; do sleep 5; done'; sleep 30; echo '--- READY ---'; grep -aE 'resume|\[T3\]|\[T4\]|AMP-WALK PASS|AMP-WALK WEIGHTS|wasabi' /root/train_${TAG}.log | head -40; echo; echo '--- ⭐ 存活检查 ---'; pgrep -af 'instinct_rl/train' | head -1 || echo '  ❌ train.py 已不在 ⇒ 看下面的 Traceback'; echo; echo '--- 报错（若有）---'; grep -aE 'Traceback|KeyError|RuntimeError|ValueError|AttributeError' -A 3 /root/train_${TAG}.log | tail -30"

cat <<EOF

=== 必须核对（逐条对照远端输出）===
  * ⭐ \`resume    : 续训  from <T2 的 run 目录> / model_11500.pt\`（**与 T3 同起点**）
  * \`[T3][减一项偏置] ...\` 仍会出现（T4 继承 T3，先应用 T3 的改动）
  * ⭐ \`[T4][消融] 整项移除 track_base_height：\` + \`非零奖励项 12 → **11**\` + 保留项清单
  * \`[T4][ALGO] 算法侧逐字沿用 T3\`（fresh_discriminator_on_load=False）
  * \`[AMP-WALK PASS] policy=1500\`，且 \`[AMP-WALK WEIGHTS]\` 里**没有** \`track_base_height\`
  * 没有 Traceback，且训练进程仍在

=== 判读（续训后看 2~3k 轮，≈2~3 h）===
  * ⭐ **唯一判据**：MuJoCo 回放 + \`analyze_gait_trace.py\` 的
    "躯干**世界系**前倾" 是否从 +0.81° 往参考的 +10.10° 移动
      python instinct/tools/g1_mujoco_play.py --policy <onnx> --cmd 0.8 0 0 \\
        --duration 14 --no-view --no-realtime --trace-csv /tmp/t4.csv
      python3 instinct/tools/analyze_gait_trace.py --csv /tmp/t4.csv --cmd 0.8
  * ⚠️ 辅助项（**不得**当判据）：\`Mean episode length\`（撤了防塌保险，掉一点是意料之中）
  * 预判：**不变**。若不变 ⇒ 这条腿关闭，直接进"root→pelvis 冷启动"或"给躯干倾角加靶"。
EOF
