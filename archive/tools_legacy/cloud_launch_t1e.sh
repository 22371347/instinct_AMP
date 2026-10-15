#!/usr/bin/env bash
# ============================================================================
# T1e 启动器（跑在云端）：T1d 配置 + **全关节限位惩罚**（恢复 V33 的 margin 版）
# ============================================================================
# TASK = Instinct-Amp-Walk-T1e-G1-v0
#   环境：`amp_walk_t1e_env_cfg.py`（= T1a2 的奖励表，**只**把 `dof_pos_limits`
#         换成全关节软边距版 `joint_pos_limits_margin`，margin 0.15 / power 2）
#   算法：复用 `amp_walk_t1b_ppo_cfg`（λ=0.75，与 T1d 逐字相同）
#   起点：**热启动自 T1d 的最新 ckpt**（默认 model_7500.pt）⇒ 保留风格训练成果，
#         只叠加"全身限位约束"这一个变量
#   预算：不设上限
#
# ⚠️ 单 GPU：本臂会 **pkill 掉仍在跑的 T1d**（训练脚本内部行为）。T1d 的 ckpt 与
#    曲线已落盘保留（model_500 … model_7500）。
#
# 用法（云端）：bash /root/cloud_launch_t1e.sh [budget_iters] [num_envs] [ckpt]
# ============================================================================
set -uo pipefail

BUDGET="${1:-1000000000}"
INST="${2:-4096}"
CKPT="${3:-model_7500.pt}"
TASK="Instinct-Amp-Walk-T1e-G1-v0"
TAG="t1e_fullbodylimit_from_t1d"
LOG="/root/train_${TAG}.log"
GUARD_LOG="/root/guard_${TAG}.log"
EXPERT=/root/amp_expert_states_396.pt
BROKEN_MARK=/root/amp_expert_states_396_broken_linvelb.pt
RUNROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk

echo "=== 1) 定位 T1d 的 run（热启动源）==="
LOAD_RUN="$(basename "$(ls -dt "$RUNROOT"/*t1d*/ 2>/dev/null | head -1)")"
if [[ -z "$LOAD_RUN" ]]; then
  echo "  FAIL 找不到 T1d 的 run 目录" >&2; exit 1
fi
echo "  LOAD_RUN = $LOAD_RUN"
if [[ ! -f "$RUNROOT/$LOAD_RUN/$CKPT" ]]; then
  echo "  FAIL 缺 $RUNROOT/$LOAD_RUN/$CKPT；该目录现有：" >&2
  ls "$RUNROOT/$LOAD_RUN" 2>/dev/null | grep 'model_.*\.pt' | sort -V | tail -3 >&2
  exit 1
fi
echo "  OK $RUNROOT/$LOAD_RUN/$CKPT"

echo "=== 2) 专家集必须是**修好的**那份 ==="
if [[ "$EXPERT" -nt "$BROKEN_MARK" ]]; then
  echo "  OK $EXPERT 比旧备份新"
else
  echo "  FAIL $EXPERT 不比旧备份新 ⇒ 可能没重导成功" >&2; exit 1
fi
/opt/conda/envs/isaaclab/bin/python - <<'PY'
import torch
raw = torch.load('/root/amp_expert_states_396.pt', map_location='cpu')
x = (next(v for v in raw.values() if isinstance(v, torch.Tensor)) if isinstance(raw, dict) else raw).float()
n = x[:, 234:237].norm(dim=1)
print(f"    anchor |v| mean = {float(n.mean()):.4f} m/s（应 ~0.5-0.7）")
raise SystemExit(0 if float(n.mean()) > 0.1 else 1)
PY
if [[ $? -ne 0 ]]; then
  echo "  FAIL 专家集根速度仍近零 ⇒ 口径没修好" >&2; exit 1
fi

echo "=== 3) 停掉当前训练（⚠️ 会停掉仍在跑的 T1d）==="
pkill -9 -f 'train[.]py' || true
sleep 3
pgrep -af 'train[.]py' | grep -v pgrep || echo "  OK 已无 train.py 进程"

echo "=== 4) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  FAIL 没有可用 GPU ⇒ 不启动" >&2; exit 1; }

echo "=== 5) 起 T1e（task=$TASK，热启动 from $CKPT）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v33.sh $INST $BUDGET cuda:0 '$TAG' '$LOAD_RUN' '$CKPT' '$TASK'" \
  "$LOG"

echo "=== 6) 拉起崩溃守护 ==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh '$LOG' 30 50 20 1.5" "$GUARD_LOG"

cat <<EOF
=== done（T1e）===
日志：$LOG   守护：$GUARD_LOG
热启动源：$LOAD_RUN / $CKPT
必须核对（**有效性检查**）：
  * 日志出现 \`resume    : 续训  from $LOAD_RUN / $CKPT\`
  * 日志出现 **\`[T1e][LIMIT REWARD] dof_pos_limits 已换回**全关节**软边距版\`**
      （启动期打印，含 func 全名 / joint_names=['.*'] / weight=-1）
  * \`params/env.yaml\`：\`rewards.dof_pos_limits\` 的 func 含 \`joint_pos_limits_margin\`
  * AMP 契约横幅（obs 378 / action 29 / scale 1.0）+ \`[wasabi] ... $EXPERT\`
预注册判据：
  * **限位（本臂的目的）**：日志/指标里应出现 margin 版诊断
      \`limit_margin_u_max\`（应显著 < 1.0 ⇒ 无关节顶到硬限位）、
      \`limit_margin_u_frac\`（进入边距的关节比例，应较低）、
      \`waist_pitch_abs_mean\` / \`waist_pitch_enter_frac\`（腰部是否长期饱和）
      以及 TB 里的 \`Episode_Reward/dof_pos_limits\` 会明显变负（现在全关节都收费）
  * **不得牺牲行走**：\`vx_high_vel_mean\` 保持 >= 0.4（T1d 的平台），力争 >= 0.5
  * **风格保持**：\`track_ang_vel_z_exp\` 不应比 T1d 的 0.247 明显退步
EOF
