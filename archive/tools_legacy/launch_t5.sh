#!/usr/bin/env bash
# ============================================================================
# T5 一键发车（本地）：高度项**换体** —— 量躯干（root=torso_link） → 量骨盆（pelvis）
# ============================================================================
#   TASK = Instinct-Amp-Walk-T5-G1-v0        TAG = t5_pelvis_height_from11500
#   环境侧唯一改动：`track_base_height`(root, 靶 0.82) → **None**；
#                   新增 `track_pelvis_height`(pelvis, 靶 **0.7585**, weight 1.5 / std 0.3)
#   算法侧：与 T3 相同（fresh_discriminator_on_load=False，保住判别器）
#   起点：`t2_mjlabcopy_obs1500_rsi_cold` / model_11500.pt（**与 T3/T4 同起点**）
#
# 为什么这样定靶（用户提醒："过高会导致腿部伸直进入奇异点动作受限"）：
#   * mjlab 字面值 = 0.78（他们 root 就是 pelvis；`KNEES_BENT_KEYFRAME.pos_z`）
#   * 我们资产"生成高度等价" = 0.776（= 0.82 − 0.044）
#   * ⭐ 采用 = **0.7585** = FK 实测的"default 站姿 + 脚掌着地"骨盆离地高度
#     ⇒ 0.78/0.776 都比站姿高 2 cm 多，会逼腿伸展（= 当年躯干靶 0.82 同款坑）；
#       取 0.7585 后该项是"别蹲也别过伸"的**对称**约束，伸展方向零压力。
#
# ✅ obs/动作/资产一字未改 ⇒ **可热启动**（不需冷启动）、**与越障仍同源、蒸馏零影响**
#
# 用法：bash instinct/tools/launch_t5.sh
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

TASK_ID="Instinct-Amp-Walk-T5-G1-v0"
TAG="t5_pelvis_height_from11500"
BASE_PAT="t2_mjlabcopy_obs1500_rsi_cold"
CKPT="model_11500.pt"
L=instinct_xu/source/instinctlab/instinctlab
R=/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab
RUNROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk
SSH=(python3 instinct/tools/instinct_ssh.py)

echo "=== 0) 连通性检查 ==="
"${SSH[@]}" "echo 实例在线" || { echo "❌ 实例未开机 / 端口变了（改 ~/.llp_cloud_credentials 的 LLP_SSH_PORT）"; exit 1; }

echo "=== 1) 同步文件（含改动的 rewards_amp.py —— 新奖励函数在里面）==="
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/amp_walk_t5_env_cfg.py" \
                "$R/tasks/locomotion/config/g1/amp_walk_t5_env_cfg.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/agents/amp_walk_t5_ppo_cfg.py" \
                "$R/tasks/locomotion/config/g1/agents/amp_walk_t5_ppo_cfg.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/__init__.py" \
                "$R/tasks/locomotion/config/g1/__init__.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/mdp/rewards_amp.py" \
                "$R/tasks/locomotion/mdp/rewards_amp.py" || exit 1
"${SSH[@]}" --put "instinct/tools/cloud_launch_arm.sh" "/root/cloud_launch_arm.sh" || exit 1
echo "  OK 5 个文件已上传"

echo "=== 2) 远端依赖自检 ==="
"${SSH[@]}" "set -e; \
  grep -c 'def track_pelvis_height_exp' $R/tasks/locomotion/mdp/rewards_amp.py; \
  grep -c 'T5' $R/tasks/locomotion/config/g1/__init__.py; \
  echo '--- 热启动源 ---'; ls -d $RUNROOT/*${BASE_PAT}* 2>/dev/null | tail -1; ls $RUNROOT/*${BASE_PAT}*/${CKPT} 2>/dev/null; \
  echo '--- 目标高度常量 ---'; grep -n 'T5_FK_STANDING\|T5_PELVIS_TARGET\|T5_MJLAB_LITERAL' $R/tasks/locomotion/config/g1/amp_walk_t5_env_cfg.py | head -5"

echo "=== 3) ⭐ 发车 ==="
"${SSH[@]}" "bash /root/cloud_launch_arm.sh $TASK_ID $TAG $BASE_PAT $CKPT 4096" || exit 1

echo "=== 4) 等自检 + 存活检查 ==="
"${SSH[@]}" "timeout 420 sh -c 'while ! grep -qa \"AMP-WALK PASS\" /root/train_${TAG}.log 2>/dev/null; do sleep 5; done'; sleep 30; echo '--- READY ---'; grep -aE 'resume|\[T5\]|AMP-WALK PASS|AMP-WALK WEIGHTS|wasabi' /root/train_${TAG}.log | head -40; echo; echo '--- ⭐ 存活 ---'; pgrep -af 'instinct_rl/train' | head -1 || echo '  ❌ train.py 已不在'; echo; echo '--- 报错（若有）---'; grep -aE 'Traceback|KeyError|RuntimeError|ValueError|AttributeError' -A 4 /root/train_${TAG}.log | tail -30"

cat <<EOF

=== 必须核对 ===
  * \`resume : 续训 from <T2 目录> / model_11500.pt\`（与 T3/T4 同起点）
  * ⭐ \`[T5][换体] 高度项：root(torso_link) → **pelvis**\` + \`非零奖励项仍为 12\` + 保留项清单
  * \`[AMP-WALK WEIGHTS]\` 里**没有** \`track_base_height\`、**有** \`track_pelvis_height=1.5\`
  * \`[AMP-WALK PASS] policy=1500 ... rewards=12\`、\`[T5][ALGO]\`、无 Traceback、进程在

=== 判读（2~3k 轮，≈2~3 h）===
  * ⭐ 探针 \`Step/pelvis_height\`：实测骨盆高度是否稳定在靶 0.7585 附近（±2~3 cm 内）
      —— 若长期低 5 cm 以上 ⇒ 又出现"靶不自洽"，立刻回看
  * ⭐ **躯干世界系前倾**是否从 +0.81° 往参考的 +10.10° 移动（回放取证，唯一判据）：
      python instinct/tools/g1_mujoco_play.py --policy <onnx> --cmd 0.8 0 0 \\
        --duration 14 --no-view --no-realtime --trace-csv /tmp/t5.csv
      python3 instinct/tools/analyze_gait_trace.py --csv /tmp/t5.csv --cmd 0.8
  * 腿的伸展量：膝关节是否变得更直（p50 从 0.383 下降）—— 若明显变直且出现"动作受限"
    现象（\`dof_pos_limits\` 变负更多 / \`action_abs_mean\` 变大），说明靶仍偏高 ⇒ 降靶
  * 速度跟踪（0.8 m/s 的效率是否从 34~46% 回升）、\`Mean episode length\` 不掉
EOF
