#!/usr/bin/env bash
# ============================================================================
# T5 **冷启动** 一键发车（本地）：停掉 warm 的 T5 → 从零起 T5
# ============================================================================
#   TASK = Instinct-Amp-Walk-T5-G1-v0        TAG = t5_pelvis_height_cold
#   与 warm 版（TAG=t5_pelvis_height_from11500，已停）**环境和算法完全相同**，
#   唯一差别 = **不加 --resume**（从零）。
#
#   为什么冷启动（2026-10-10 用户决策）：
#     * `track_pelvis_height` 靶 0.7585 在语义上 = "平均保持默认屈膝量(膝≈0.669)"，
#       而当前策略是**伸腿走**（膝 p50 0.38）⇒ 这一项要求的是**结构性重组**
#       （把腿弯回来 + 重心后移后**重新平衡**），不是微调；
#     * warm 路线在"姿态/高度"这个维度上已被 T3（改靶）、T4（移除）、T5-warm **三次**
#       证明**打不破习惯**（各 0.5~2.1k 轮纹丝不动）；
#     * 对照 T2 本身就是**冷启动臂** ⇒ T5-cold vs T2-cold 是同源最干净的对照。
#   ⚠️ 冷启动初期判别器可能是死的（`amp_d_ref` 贴死 ≈0.9）⇒ **那种状态下没有前倾不算反例**。
#
# 用法：bash instinct/tools/launch_t5_cold.sh
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

TAG="t5_pelvis_height_cold"
L=instinct_xu/source/instinctlab/instinctlab
R=/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab
SSH=(python3 instinct/tools/instinct_ssh.py)

echo "=== 0) 连通性检查 ==="
"${SSH[@]}" "echo 实例在线" || { echo "❌ 实例未开机 / 端口变了"; exit 1; }

echo "=== 1) 同步文件 ==="
"${SSH[@]}" --put "$L/tasks/locomotion/mdp/rewards_amp.py" \
                "$R/tasks/locomotion/mdp/rewards_amp.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/amp_walk_t5_env_cfg.py" \
                "$R/tasks/locomotion/config/g1/amp_walk_t5_env_cfg.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/agents/amp_walk_t5_ppo_cfg.py" \
                "$R/tasks/locomotion/config/g1/agents/amp_walk_t5_ppo_cfg.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/__init__.py" \
                "$R/tasks/locomotion/config/g1/__init__.py" || exit 1
"${SSH[@]}" --put "instinct/tools/cloud_launch_t5_cold_remote.sh" \
                "/root/cloud_launch_t5_cold_remote.sh" || exit 1
echo "  OK 5 个文件已上传"

echo "=== 2) ⭐ 云端冷启动（内部：查代码/专家集 → 停旧训练+守护 → 从零起车 → 放宽守护）==="
"${SSH[@]}" "bash /root/cloud_launch_t5_cold_remote.sh 1000000000 4096" || exit 1

echo "=== 3) 等自检 + 存活检查 ==="
"${SSH[@]}" "timeout 420 sh -c 'while ! grep -qa \"AMP-WALK PASS\" /root/train_${TAG}.log 2>/dev/null; do sleep 5; done'; sleep 30; echo '--- READY ---'; grep -aE 'resume|\[T5\]|AMP-WALK PASS|AMP-WALK WEIGHTS|wasabi' /root/train_${TAG}.log | head -40; echo; echo '--- ⭐ 存活 ---'; pgrep -af 'instinct_rl/train' | head -1 || echo '  ❌ train.py 已不在'; echo; echo '--- 报错（若有）---'; grep -aE 'Traceback|KeyError|RuntimeError|ValueError|AttributeError' -A 4 /root/train_${TAG}.log | tail -30"

cat <<EOF

=== 必须核对 ===
  * ⚠️ **不应**出现 \`resume    : 续训\`（冷启动！）
  * \`[T5][换体] ...\` + \`非零奖励项仍为 12\` + 目标取舍（0.78 / 0.776 / **0.7585**）
  * \`[AMP-WALK PASS] policy=1500 ... rewards=12\`
  * \`[AMP-WALK WEIGHTS]\`：无 \`track_base_height\`、有 **\`track_pelvis_height=1.5\`**
  * \`[T2][COPY MJLAB] RSI 起点 (0.0, 0.0) → (0.0, 1.0)\`（真 RSI 仍在）
  * 进程存活、无 Traceback

=== 判读（**先看判别器活没活，再看姿态**）===
  * ① \`Step/amp_d_ref\` 是否从 ≈+0.9 掉到 **+0.5 量级**、\`Step/amp_style_zero_frac\` < 0.30
        —— 没活之前，**没有前倾不算反例**
  * ② 探针 \`Step/pelvis_height\` 是否收敛到 **0.7585 ± 2~3 cm**
  * ③ 回放取证：躯干**世界系**前倾是否往 **+10.10°** 爬（基线 T2/T3/T4 = +0.8~3.3°）
      python instinct/tools/pull_latest_onnx.sh t5_pelvis_height_cold
      python instinct/tools/g1_mujoco_play.py --policy <onnx> --cmd 0.8 0 0 \\
        --duration 14 --no-view --no-realtime --trace-csv /tmp/t5c.csv
      python3 instinct/tools/analyze_gait_trace.py --csv /tmp/t5c.csv --cmd 0.8
  * ⚠️ 对照基线用 **T2-cold**（不是 T2@11500 的续训臂）
EOF
