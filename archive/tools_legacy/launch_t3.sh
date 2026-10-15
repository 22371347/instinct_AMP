#!/usr/bin/env bash
# ============================================================================
# T3 一键发车（**本地**脚本）：T2 + 只改一项奖励的靶/权重（做减法）+ **续训保留判别器**
# ============================================================================
#   TASK = Instinct-Amp-Walk-T3-G1-v0
#   环境侧唯一改动：`track_base_height`
#       weight  1.5 → **1.0**
#       target_height  None(=0.82 生成高度) → **0.8025**（= default 站姿实测离地高度）
#       ⇒ 改完该项值 ≈0.998~1.0 = **基本变成常数**（只留防塌，不再对"立腰/伸膝"出梯度）
#   算法侧唯一改动：`fresh_discriminator_on_load` True → **False**（保住刚活过来的判别器）
#   起点：**热启动** `t2_mjlabcopy_obs1500_rsi_cold` / `model_11500.pt`（obs 契约未变）
#
# ⚠️ 复用通用发车器 `cloud_launch_arm.sh`（它会自己解析热启动源、校验 ckpt、起守护）。
# ⚠️ 本脚本会 **kill 掉云端当前任何训练**（现在 T2 已停，故无影响）。
#
# 用法：bash instinct/tools/launch_t3.sh
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

TASK_ID="Instinct-Amp-Walk-T3-G1-v0"
TAG="t3_h08025_w1p0_from11500"
BASE_PAT="t2_mjlabcopy_obs1500_rsi_cold"
CKPT="model_11500.pt"
L=instinct_xu/source/instinctlab/instinctlab
R=/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab
RUNROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk
SSH=(python3 instinct/tools/instinct_ssh.py)

echo "=== 0) 连通性检查 ==="
"${SSH[@]}" "echo 实例在线" || { echo "❌ 实例未开机 —— 请先开机再跑本脚本"; exit 1; }

echo "=== 1) 同步文件 ==="
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/amp_walk_t3_env_cfg.py" \
                "$R/tasks/locomotion/config/g1/amp_walk_t3_env_cfg.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/agents/amp_walk_t3_ppo_cfg.py" \
                "$R/tasks/locomotion/config/g1/agents/amp_walk_t3_ppo_cfg.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/__init__.py" \
                "$R/tasks/locomotion/config/g1/__init__.py" || exit 1
# 通用发车器（幂等上传，避免云端副本过期）
"${SSH[@]}" --put "instinct/tools/cloud_launch_arm.sh" "/root/cloud_launch_arm.sh" || exit 1
echo "  OK 4 个文件已上传"

echo "=== 2) 远端依赖自检（T3 继承 T2 的 cfg，必须都在 + 热启动源必须在）==="
"${SSH[@]}" "set -e; \
  for f in $L/tasks/locomotion/config/g1/amp_walk_t2_env_cfg.py \
            $L/tasks/locomotion/config/g1/agents/amp_walk_t2_ppo_cfg.py \
            /root/run_amp_walk_v33.sh; do
      [ -f \"\$f\" ] && echo \"  OK \$f\" || { echo \"  ❌ 缺 \$f\"; exit 1; }; done; \
  echo '--- 热启动源 ---'; \
  ls -d $RUNROOT/*${BASE_PAT}* 2>/dev/null | tail -1; \
  ls -t $RUNROOT/*${BASE_PAT}*/model_*.pt 2>/dev/null | head -3; \
  echo '--- T3 注册与开关 ---'; \
  grep -c 'T3' $R/tasks/locomotion/config/g1/__init__.py; \
  grep -n 'T3_HEIGHT_TARGET\|T3_HEIGHT_WEIGHT' $R/tasks/locomotion/config/g1/amp_walk_t3_env_cfg.py | head -3; \
  grep -n 'fresh_discriminator_on_load = False' $R/tasks/locomotion/config/g1/agents/amp_walk_t3_ppo_cfg.py"

echo "=== 3) ⭐ 发车（cloud_launch_arm.sh 内部：解析 LOAD_RUN → 校验 ckpt → kill 旧训练 → 起车 → 起守护）==="
"${SSH[@]}" "bash /root/cloud_launch_arm.sh $TASK_ID $TAG $BASE_PAT $CKPT 4096" || exit 1

echo "=== 4) 等自检（条件等待 + ⭐ 存活检查）==="
"${SSH[@]}" "timeout 420 sh -c 'while ! grep -qa \"AMP-WALK PASS\" /root/train_${TAG}.log 2>/dev/null; do sleep 5; done'; sleep 30; echo '--- READY ---'; grep -aE 'resume|\[T3\]|AMP-WALK PASS|AMP-WALK WEIGHTS|wasabi' /root/train_${TAG}.log | head -40; echo; echo '--- ⭐ 存活检查 ---'; pgrep -af 'instinct_rl/train' | head -1 || echo '  ❌ train.py 已不在 ⇒ 看下面的 Traceback'; echo; echo '--- 报错（若有）---'; grep -aE 'Traceback|KeyError|RuntimeError|ValueError|AttributeError' -A 3 /root/train_${TAG}.log | tail -30"

cat <<EOF

=== 必须核对（逐条对照远端输出）===
  * ⭐ \`resume    : 续训  from <T2 的 run 目录> / model_11500.pt\`
        （**不应**出现 \`resume    : <从头开始>\` —— 那是冷启动）
  * \`[T2][COPY MJLAB] ...\` 各条（环境侧仍按 T2 打印，T3 的改动叠加在其后）
  * \`[T3][减一项偏置] track_base_height：weight 1.5 → 1.0；target ... → 0.8025\`
  * \`[T3][KEEP DISCRIMINATOR] fresh_discriminator_on_load True → False\`
  * \`[AMP-WALK PASS] policy=1500\`，且 \`[AMP-WALK WEIGHTS]\` 里
    \`track_base_height=1\`（其余 11 项与 T2 逐字相同）
  * 没有 Traceback，且训练进程仍在

=== 判读（续训后看 2~3k 轮）===
  * \`Episode_Reward/track_base_height\` 应从 1.49 升到 **≈1.00（即 1.0×~1.0）**
    —— 值升上去 = 该项不再参与决策（减法生效的机械证据）
  * \`Step/amp_d_ref\` / \`amp_style_zero_frac\`：**不应**回到贴死（这是"保住判别器"的验收）
  * ⭐ **步态**：MuJoCo 回放看腰是否不再端着（\`--obs-mode amp1500\`）
  * \`vxy_rmse\` / \`vx_high\`：速度跟踪是否改善（T2 末值 vxy_rmse ≈0.37~0.42）
  * \`Mean episode length\` 应维持 1000（不许掉——那就是"防塌"没兜住）
EOF
