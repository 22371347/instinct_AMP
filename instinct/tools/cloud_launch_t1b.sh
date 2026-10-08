#!/usr/bin/env bash
# ============================================================================
# T1b 启动器（跑在云端）：**只改一个数** —— `style_lerp` 1.0 → 0.75
# ============================================================================
# TASK = Instinct-Amp-Walk-T1b-G1-v0
#   环境：复用 `amp_walk_t1a2_env_cfg.py`（**同一个类**，不做任何复制 ⇒ 环境零改动）
#   算法：`amp_walk_t1b_ppo_cfg.py` = T1a2 的全部超参 + `style_lerp=0.75`
#   起点：**热启动** —— 从 T1a2 的 `model_7000.pt` 续训（判别器已训练充分：
#         T1a2 里判别器照常训练了 7000+ 轮，只是奖励贡献为 0）
#   预算：**不设上限**
#
# ⚠️ 已知行为：`cloud_train_amp_walk_v3.sh` 续训会**重置 AMP 参考动作播放游标**
#    （它不在 checkpoint 里）；本实验无碍，记录在案。
#
# 用法（云端）：bash /root/cloud_launch_t1b.sh [budget_iters] [num_envs] [ckpt]
# ============================================================================
set -uo pipefail

BUDGET="${1:-1000000000}"
INST="${2:-4096}"
CKPT="${3:-model_7000.pt}"
TASK="Instinct-Amp-Walk-T1b-G1-v0"
TAG="t1b_style0p75_from_t1a2"
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
  echo "  该目录下现有 pt：$(ls "$RUNROOT/$LOAD_RUN" 2>/dev/null | grep -c 'model_.*\.pt') 个" >&2
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

echo "=== 5) 起 T1b（热启动 from $CKPT，预算=$BUDGET，num_envs=$INST）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v33.sh $INST $BUDGET cuda:0 '$TAG' '$LOAD_RUN' '$CKPT' '$TASK'" \
  "$LOG"

echo "=== 6) 拉起崩溃守护 ==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh '$LOG' 30 50 20 1.5" "$GUARD_LOG"

cat <<EOF
=== done（T1b）===
日志：$LOG     守护：$GUARD_LOG
必须核对（**有效性检查，不是形式检查**）：
  * 日志出现 **\`resume    : 续训  from $LOAD_RUN / $CKPT\`**（**不是**"从头开始"）
  * \`Learning iteration 0/\` 的起始 reward 应明显高于随机（继承了 T1a2 的能力）
  * AMP 行走教师契约横幅（obs 378 / action 29 / scale 1.0 / 50 Hz）
  * 运行期 \`params/agent.yaml\`：**\`style_lerp: 0.75\`**（相对 T1a2 的**唯一**数值改动），
    其余与 T1a2 完全一致（init_noise_std 1.0 / clip_max_std null / [256,128,128] / entropy 0.008）
  * ⭐ **风格项真的打开了**：日志/指标里 \`discriminator_reward\` 或 \`amp_style_std\`
    必须显著非零 —— T1a2 因 λ=1.0 该项恒为 0，T1b 若仍为 0 说明"没真的打开"
  * 指标 \`command_bins/*_vel_mean\` + \`Loss/mirror\` 仍在打点
预注册判据：
  * **转向必须改善**：\`track_ang_vel_z_exp\` 明显高于 T1a2 的 +0.148（T0 为 +0.538）
  * **动作平滑/寿命改善**：\`action_rate_l2\` 惩罚绝对值下降、\`ep_len\` 上升（T1a2 621 / T0 970）
  * **不得牺牲速度**：\`vx_high_vel_mean\` 保持 ≥0.5（这是"风格项是否又把高速弄没"的判据）
EOF
