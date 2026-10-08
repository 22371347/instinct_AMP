#!/usr/bin/env bash
# ============================================================================
# T0：从零复现 OmniLoco 基座（纯速度跟踪，**无 AMP**）
# ============================================================================
# 目的：在加回任何复杂度之前，先把"这条管线能不能从零学会速度跟踪"问清楚。
#   TASK  = Instinct-Unified-OmniLoco-G1-v0
#           actor 观测 623 = 527(scan) + 3(cmd) + 3(grav) + 3(ang) + 29 + 29 + 29
#           **不含 base_lin_vel**（基座无此特权量）⇒ 回放用 `--obs-mode loco623`
#   从零  = 不传 load_run ⇒ cloud_train_amp_walk_v3.sh 不加 --resume
#   指令  = OMNI_LOCO_COMMAND_RANGES ((-0.5,1.0),(-0.5,0.5),(-1.0,1.0))，三轴独立均匀
#   动作  = JointPositionAction scale=1.0 + use_default_offset=True（与 AMP-Walk 同）
#
# ⚠️ 本文件必须**推到云端再执行**：命令要经 ssh → cloud_run_bg.sh → bash -c →
#    train.py 四层，空字符串参数在命令行里会被引号吃掉（本项目踩过多次）
#    ⇒ 写进文件只传一个 token。
#
# 用法（云端）：bash /root/cloud_launch_t0.sh [budget_iters] [num_envs]
# ============================================================================
set -uo pipefail

BUDGET="${1:-2000}"
INST="${2:-4096}"
TASK="Instinct-Unified-OmniLoco-G1-v0"
TAG="loco_base_obs623_scratch"
LOG="/root/train_${TAG}.log"
GUARD_LOG="/root/guard_${TAG}.log"

echo "=== 1) 停掉当前训练 ==="
pkill -9 -f 'train[.]py' || true
sleep 3
pgrep -af 'train[.]py' | grep -v pgrep || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  ❌ 没有可用 GPU ⇒ 不启动" >&2; exit 1; }

echo "=== 3) 起 T0（task=$TASK，从零，预算 $BUDGET iters，num_envs=$INST）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v33.sh $INST $BUDGET cuda:0 '$TAG' '' '' '$TASK'" \
  "$LOG"

echo "=== 4) 拉起崩溃守护 ==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh '$LOG' 30 50 20 1.5" "$GUARD_LOG"

cat <<EOF
=== done（T0 基座冷启动）===
日志：$LOG     守护：$GUARD_LOG
启动后必须核对（**有效性检查，不是形式检查**）：
  * 日志出现 \`resume    : <从头开始>\`（不是续训）
  * \`Learning iteration 0/\`（确实从零）
  * 运行期 params/env.yaml：
      - lin_vel_x (-0.5, 1.0) / lin_vel_y (-0.5, 0.5) / ang_vel_z (-1.0, 1.0)
      - actions.joint_pos.scale: 1.0 + use_default_offset: true
      - **不应**出现任何 AMP 键（amp_* / discriminator_* / wasabi）
  * 指标里出现新的 \`command_bins/*_vel_mean\`（前置件 2 已推上云）
EOF
