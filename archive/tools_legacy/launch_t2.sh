#!/usr/bin/env bash
# ============================================================================
# T2 一键发车（**本地**脚本）：全权照抄 AMP_mjlab walk + 3 处例外 + 真 RSI（冷启动）
# ============================================================================
#   TASK = Instinct-Amp-Walk-T2-G1-v0
#   actor 观测 378 → **1500** = 279×4（height_scan 也发 4 帧）+ 96×4（本体感受 4 帧）
#   奖励 = mjlab 的 12 项；命令/终止/事件 = mjlab；⭐ 复位 = 真 RSI
#   算法 = λ0.75/c0.1（风格 0.025）+ 判别器 mjlab 口径 + 例外 B（噪声 0.25×）
#   起点 = **从零**（不 resume）
#
# ⚠️ 本脚本会**先 kill 掉云端正在跑的训练**（T1o）⇒ 只在决定发车时执行。
#
# 用法：bash instinct/tools/launch_t2.sh
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

TAG="t2_mjlabcopy_obs1500_rsi_cold"
L=instinct_xu/source/instinctlab/instinctlab
R=/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab
SSH=(python3 instinct/tools/instinct_ssh.py)

echo "=== 0) 连通性检查 ==="
"${SSH[@]}" "echo 实例在线" || { echo "❌ 实例未开机 —— 请先开机再跑本脚本"; exit 1; }

echo "=== 1) 同步文件 ==="
# ① T2 环境侧（新）
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/amp_walk_t2_env_cfg.py" \
                "$R/tasks/locomotion/config/g1/amp_walk_t2_env_cfg.py" || exit 1
# ② T2 算法侧（新）
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/agents/amp_walk_t2_ppo_cfg.py" \
                "$R/tasks/locomotion/config/g1/agents/amp_walk_t2_ppo_cfg.py" || exit 1
# ③ 注册（新增 T2 独立注册块，因为 obs 契约 1500 ≠ 378）
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/__init__.py" \
                "$R/tasks/locomotion/config/g1/__init__.py" || exit 1
# ④ ⭐ 奖励：新增 `track_anchor_lin_vel_exp` / `track_anchor_ang_vel_exp`（照抄 mjlab 的形式：
#    前者把 **v_z** 也算进误差 —— 这正是 mjlab 不需要 lin_vel_z 项的原因；后者含 ω_xy）
"${SSH[@]}" --put "$L/tasks/locomotion/mdp/rewards_amp.py" \
                "$R/tasks/locomotion/mdp/rewards_amp.py" || exit 1
# ⑤ ⭐ 镜像算子：支持**堆叠的 height_scan**（T2 的 1500；顺带修掉 `scan_dim` 属性不存在）
"${SSH[@]}" --put "$L/utils/mirror.py" \
                "$R/utils/mirror.py" || exit 1
# ⑥ 冷启动发车脚本（⚠️ 空字符串必须写在**文件里**，不能走命令行引号）
"${SSH[@]}" --put "instinct/tools/cloud_launch_t2_remote.sh" \
                "/root/cloud_launch_t2_remote.sh" || exit 1
echo "  OK 6 个文件已上传"

echo "=== 1b) 远端依赖自检（T2 继承 T1l/T1o 的 cfg，必须都在）==="
"${SSH[@]}" "set -e; \
  for f in $L/tasks/locomotion/config/g1/amp_walk_t1l_env_cfg.py \
            $L/tasks/locomotion/config/g1/agents/amp_walk_t1o_ppo_cfg.py \
            $L/tasks/locomotion/config/g1/agents/amp_walk_t1n_ppo_cfg.py \
            $L/utils/wrappers/instinct_rl/algorithms.py; do
      [ -f \"\$f\" ] && echo \"  OK \$f\" || { echo \"  ❌ 缺 \$f\"; exit 1; }; done; \
  echo '--- 上传结果 ---'; \
  grep -c 'T2' $R/tasks/locomotion/config/g1/__init__.py; \
  grep -n 'track_anchor_lin_vel_exp' $R/tasks/locomotion/mdp/rewards_amp.py | head -2; \
  grep -n '_per_scan' $R/utils/mirror.py | head -3; \
  grep -n 'scan_frames' $R/tasks/locomotion/config/g1/amp_walk_t2_env_cfg.py | head -2"

echo "=== 2) ⭐ 冷启动发车（会先 kill 掉当前训练）==="
"${SSH[@]}" "bash /root/cloud_launch_t2_remote.sh"

echo "=== 3) 等自检（条件等待 + ⭐ 存活检查）==="
"${SSH[@]}" "timeout 420 sh -c 'while ! grep -qa \"AMP-WALK PASS\" /root/train_${TAG}.log 2>/dev/null; do sleep 5; done'; sleep 30; echo '--- READY ---'; grep -aE '\[T2\]|AMP-WALK PASS|AMP-WALK WEIGHTS|resume|COLD' /root/train_${TAG}.log | head -40; echo; echo '--- ⭐ 存活检查 ---'; pgrep -af 'train[.]py' | head -1 || echo '  ❌ train.py 已不在 ⇒ 看下面的 Traceback'; echo; echo '--- 报错（若有）---'; grep -aE 'Traceback|KeyError|RuntimeError|ValueError|AttributeError' -A 3 /root/train_${TAG}.log | tail -30"

cat <<'EOF'

=== 必须核对（逐条对照远端输出）===
  * ⚠️ **不应**出现 `resume    : 续训`（本臂从零）
  * `[T2][COPY MJLAB] actor 观测 378 → **1500**`，逐项 = (1116, 12, 12, 12, 116, 116, 116)
  * `[T2][COPY MJLAB] 奖励 12 项` + 逐项权重表
  * `[AMP-WALK PASS] policy=1500`、`cmd_ranges=((-0.5, 1.0), (-1.0, 1.0), (-1.57, 1.57))`
  * `[T2][COPY MJLAB] RSI 起点 (0.0, 0.0) → (0.0, 1.0)`
  * 终止项 = time_out / bad_orientation / bad_base_height（**没有 base_contact**）
  * `[T2][COPY MJLAB · ALGO] 风格强度 (1−λ)·c = 0.0250`
  * 没有 Traceback，且 `train.py` **仍在运行**

=== 下一步 ===
  * 回放：`bash instinct/tools/pull_latest_onnx.sh t2_mjlabcopy_obs1500_rsi_cold`（按 run 名匹配注册表）
      ⚠️ 回放要用 `--obs-mode amp1500`（或 `auto`：按 ONNX 输入 1500 维自动选中）
  * 回放脚本：`python instinct/tools/g1_mujoco_play.py --policy <onnx> --obs-mode auto --duration 0`
  * 曲线：`bash instinct/tools/refresh_plots.sh loop`
EOF
