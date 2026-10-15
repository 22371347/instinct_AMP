#!/usr/bin/env bash
# ============================================================================
# T1c 启动器（跑在云端）：**只改一个数** —— `style_lerp` 1.0 → **0.9**
# ============================================================================
# TASK = Instinct-Amp-Walk-T1c-G1-v0
#   环境：复用 `amp_walk_t1a2_env_cfg.py`（**同一个类**，环境零改动）
#   算法：`amp_walk_t1c_ppo_cfg.py` = T1a2 全部超参 + `style_lerp=0.9`
#   起点：**热启动** —— 从 **T1a2** 的 `model_7000.pt`（**不是** T1b 的 ckpt！）
#   预算：不设上限
#
# 为什么是 0.9（T1b 的读数给的结论）：
#   `r_total = r_task·(1 − (1−λ)/c) + (1−λ)·style` ⇒ **风格权重 = (1−λ)**
#     λ=1.0（T1a2）→ 风格权重 0    ，`vx_high` 0.62 ✓
#     λ=0.75（T1b）→ 风格权重 0.25，`vx_high` 崩到 0.014 ✗
#     λ=0.9 （T1c）→ 风格权重 0.10 = T1b 的 40%   ← 本臂
#   ⚠️ 必须从 T1a2 的 ckpt 重新开始，**不要**从已崩的 T1b 续训（否则带坏先验）。
#
# 用法（云端）：bash /root/cloud_launch_t1c.sh [budget_iters] [num_envs] [ckpt]
# ============================================================================
set -uo pipefail

BUDGET="${1:-1000000000}"
INST="${2:-4096}"
CKPT="${3:-model_7000.pt}"
TASK="Instinct-Amp-Walk-T1c-G1-v0"
TAG="t1c_style0p9_from_t1a2"
LOAD_RUN="20261007_032827_G1AmpWalkT1a2MVPPlane_obs378_rew16_t0impl_sym_t1a2_t0impl_t0explore"
LOG="/root/train_${TAG}.log"
GUARD_LOG="/root/guard_${TAG}.log"
EXPERT=/root/amp_expert_states_396.pt
RUNROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk

echo "=== 1) 停掉当前训练 ==="
pkill -9 -f 'train[.]py' || true
sleep 3
pgrep -af 'train[.]py' | grep -v pgrep || echo "  OK 已无 train.py 进程"

echo "=== 2) 检查热启动源（T1a2 的 run 目录 + ckpt）==="
if [[ -f "$RUNROOT/$LOAD_RUN/$CKPT" ]]; then
  echo "  OK $RUNROOT/$LOAD_RUN/$CKPT"
else
  echo "  FAIL 找不到热启动源：$RUNROOT/$LOAD_RUN/$CKPT" >&2
  exit 1
fi

echo "=== 3) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  FAIL 没有可用 GPU ⇒ 不启动" >&2; exit 1; }

echo "=== 4) 检查判别器专家集 ==="
if [[ -f "$EXPERT" ]]; then
  echo "  OK $EXPERT（$(du -h "$EXPERT" | cut -f1)）"
else
  echo "  FAIL 缺 $EXPERT" >&2; exit 1
fi

echo "=== 5) 起 T1c（热启动 from $CKPT，预算=$BUDGET，num_envs=$INST）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v33.sh $INST $BUDGET cuda:0 '$TAG' '$LOAD_RUN' '$CKPT' '$TASK'" \
  "$LOG"

echo "=== 6) 拉起崩溃守护 ==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh '$LOG' 30 50 20 1.5" "$GUARD_LOG"

cat <<EOF
=== done（T1c）===
日志：$LOG     守护：$GUARD_LOG
必须核对：
  * \`resume    : 续训  from $LOAD_RUN / $CKPT\`（**不是**"从头开始"）
  * AMP 行走教师契约横幅（obs 378 / action 29 / scale 1.0 / 50 Hz）
  * 运行期 \`params/agent.yaml\`：**\`style_lerp: 0.9\`**（相对 T1a2 的唯一数值改动）
  * ⭐ 风格项真的打开了：\`discriminator_reward\` / \`amp_style_std\` 非零
    （T1a2 恒为 0；T1b 实测 0.0222 / 0.1763，量级可作对照）
预注册判据（**核心矛盾：风格改善 vs 前进能力**）：
  * **必须守住**：\`vx_high_vel_mean\` ≥ 0.5（T1b 崩到 0.014 的教训；这是本臂成败线）
  * **必须改善**：\`track_ang_vel_z_exp\` > T1a2 的 0.148（T1b 到过 0.329）
  * **应当改善**：\`action_rate_l2\` 惩罚绝对值下降（T1a2 −0.456）、\`ep_len\` 上升（621）
  * 观察 \`vx_mid_vel_mean\`（T1a2 只有 0.02~0.04，中速档是长期弱项）
EOF
