#!/usr/bin/env bash
# ============================================================================
# T1a2 启动器（跑在云端）：T1a + 奖励**实现/作用域**逐字对齐 T0 + 探索/网络回 T0
# ============================================================================
# TASK = Instinct-Amp-Walk-T1a2-G1-v0
#   环境：`amp_walk_t1a2_env_cfg.py`（V33-S 骨架 ⇒ 378 观测 / 396 AMP 状态 /
#         全向分档指令；奖励 = T0 的 16 项**权重 + 实现/作用域**）
#   算法：`amp_walk_t1a2_ppo_cfg.py`（style_lerp=1.0 风格归零；镜像损失 ON；
#         init_noise_std=1.0、clip_max_std 关闭、[256,128,128]、entropy_coef 0.008）
#   起点：**冷启动**（不传 load_run ⇒ 训练脚本不加 `--resume`）
#   预算：**不设上限**（跑到人工 `pkill -f 'train[.]py'` 叫停；本项目惯例）
#
# ⚠️ T1a2 相对 T1a 只有两处改动（2026-10-07 控制变量审计的结论）：
#   A. **奖励实现**：feet_air_time→`positive_biped`(threshold=0.5)、
#      dof_pos_limits→`isaaclab...joint_pos_limits`(仅踝)、dof_acc_l2→仅 hip/knee
#   B. **探索/网络**：init_noise_std 1.0、clip_min_std 1e-12、clip_max_std=None、
#      hidden [256,128,128]、entropy_coef 0.008
#   **保留不动**：track_default_pose_on_stand(−0.8)、mirror_loss(0.1)、AMP 判别器
#
# ⚠️ 本文件必须**推到云端再执行**：命令要经 ssh → cloud_run_bg.sh → bash -c → train.py
#    四层，空字符串参数在命令行里会被引号吃掉（本项目踩过多次）。
#
# 用法（云端）：bash /root/cloud_launch_t1a2.sh [budget_iters] [num_envs]
# ============================================================================
set -uo pipefail

BUDGET="${1:-1000000000}"
INST="${2:-4096}"
TASK="Instinct-Amp-Walk-T1a2-G1-v0"
TAG="t1a2_t0impl_t0explore"
LOG="/root/train_${TAG}.log"
GUARD_LOG="/root/guard_${TAG}.log"
EXPERT=/root/amp_expert_states_396.pt

echo "=== 1) 停掉当前训练 ==="
pkill -9 -f 'train[.]py' || true
sleep 3
pgrep -af 'train[.]py' | grep -v pgrep || echo "  OK 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  FAIL 没有可用 GPU ⇒ 不启动" >&2; exit 1; }

echo "=== 3) 检查判别器专家集（AMP 判别器会被构造）==="
if [[ -f "$EXPERT" ]]; then
  echo "  OK $EXPERT（$(du -h "$EXPERT" | cut -f1)）"
else
  echo "  FAIL 缺 $EXPERT" >&2; exit 1
fi

echo "=== 4) 起 T1a2（task=$TASK，冷启动，预算=$BUDGET，num_envs=$INST）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v33.sh $INST $BUDGET cuda:0 '$TAG' '' '' '$TASK'" \
  "$LOG"

echo "=== 5) 拉起崩溃守护 ==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh '$LOG' 30 50 20 1.5" "$GUARD_LOG"

cat <<EOF
=== done（T1a2）===
日志：$LOG     守护：$GUARD_LOG
启动后必须核对（**有效性检查，不是形式检查**）：
  * 日志出现 \`resume    : <从头开始>\` + \`Learning iteration 0/\`
  * 日志出现 **\`[T1a][REWARD WHITELIST]\`** 与 **\`[T1a2][REWARD IMPL]\`** 两个块
      —— 前者 = 权重实证；后者 = "实现已对齐 T0"实证（**T1a 缺的就是后者**）
  * AMP 行走教师契约横幅（obs 378 / action 29 / scale 1.0 / 50 Hz）
  * 运行期 \`params/agent.yaml\`：
      - \`init_noise_std: 1.0\` / \`clip_max_std: null\` / \`[256,128,128]\` / \`entropy_coef: 0.008\`
      - \`style_lerp: 1.0\`          （风格项归零）
      - \`use_mirror_loss: true\` + \`mirror_loss_coeff: 0.1\`
      - \`schedule: adaptive\` / \`discriminator_reward_coef: 4.0\`（沿用 S8）
  * 日志里 \`[wasabi] 专家批来自离线数据集：$EXPERT\`
  * 指标里出现 \`command_bins/*_vel_mean\` + \`Loss/mirror\`（镜像损失真的在算）
EOF
