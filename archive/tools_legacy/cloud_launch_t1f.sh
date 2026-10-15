#!/usr/bin/env bash
# ============================================================================
# T1f 启动器（云端）：**恢复 AMP_mjlab 的奖励机理** + PPO 探索口径对齐，从零训练
# ============================================================================
# TASK = Instinct-Amp-Walk-T1f-G1-v0
#   契约（与 T1a2~T1e 一致，已多次验证）：obs 378 / critic 99 / action 29 /
#     JointPositionAction scale=1.0 use_default_offset=True / 50 Hz / AMP 396
#   环境 `amp_walk_t1f_env_cfg.py`：T1a2 契约 + 恢复 mjlab 机理
#     feet_lateral_width −10.0（sep_floor 0.145）｜feet_crossing −20.0
#     dof_pos_limits 全关节 −10.0｜flat_orientation_l2 −2.0
#     track_lin_vel 1.5｜track_ang_vel 1.2
#   算法 `agents/amp_walk_t1f_ppo_cfg.py`：style_lerp 0.90｜clip_min_std 0.05
#     hidden (512,256,128)｜entropy_coef 0.005｜mirror loss 保留
#   **从零训练**（不 resume）⇒ 检验"早期就带正确机理"能否避免踝 roll 冻结
#   预算：不设上限
#
# ⚠️ 单 GPU：会 **pkill 掉仍在跑的 T1e**（其诊断价值已榨干，ckpt/曲线已保留）。
#
# 用法（云端）：bash /root/cloud_launch_t1f.sh [budget_iters] [num_envs]
# ============================================================================
set -uo pipefail

BUDGET="${1:-1000000000}"
INST="${2:-4096}"
TASK="Instinct-Amp-Walk-T1f-G1-v0"
TAG="t1f_mjlabalign"
LOG="/root/train_${TAG}.log"
GUARD_LOG="/root/guard_${TAG}.log"
EXPERT=/root/amp_expert_states_396.pt
BROKEN_MARK=/root/amp_expert_states_396_broken_linvelb.pt

echo "=== 1) 专家集必须是**修好的**那份（锚点根速度不得近零）==="
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
if [[ $? -ne 0 ]]; then echo "  FAIL 专家集根速度仍近零" >&2; exit 1; fi

echo "=== 2) ⭐ 发车前硬校验（mjlab 机理是否真恢复）==="
cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab || exit 1
source /root/.llp_env.sh
unset PYTHONPATH
/opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_t1f.py
VERIFY_RC=$?
echo "  [gate] 校验退出码 = $VERIFY_RC"
if [[ "$VERIFY_RC" -ne 0 ]]; then
  echo "  FAIL T1f 校验失败 ⇒ **不发车**" >&2
  exit 1
fi

echo "=== 3) 停掉当前训练（⚠️ 会停掉仍在跑的 T1e）==="
pkill -9 -f 'train[.]py' || true
sleep 3
pgrep -af 'train[.]py' | grep -v pgrep || echo "  OK 已无 train.py 进程"

echo "=== 4) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  FAIL 没有可用 GPU" >&2; exit 1; }

echo "=== 5) 从零起 T1f ==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v33.sh $INST $BUDGET cuda:0 '$TAG' '' '' '$TASK'" \
  "$LOG"

echo "=== 6) 拉起崩溃守护 ==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh '$LOG' 30 50 20 1.5" "$GUARD_LOG"

cat <<'EOF'
=== done（T1f）===
日志：$LOG   守护：$GUARD_LOG
必须核对：
  * 日志出现 **[T1f][MJLAB ALIGN]** 与 **[T1f][ALGO ALIGN]** 两块回读打印
  * `[AMP-WALK PASS] policy=378`（契约未变）
  * 无 `resume`（本臂从零训练）
  * `[wasabi] ... /root/amp_expert_states_396.pt`
预注册判据（对照 T1a2 / T1e）：
  * **足部（本臂的核心）**：踝 roll 的 \`tracking_joint_error_rmse\` 应 **明显低于 T1a2 的
    0.242→0.257 平台**（目标 <0.15）；回放目视**脚掌着地**、无侧边缘支撑
  * **限位**：\`Episode_Reward/dof_pos_limits\` 应显著负（−10 权重生效），
    T1e 的 \`limit_margin_u_max\` 参照应 <1.0
  * **跟踪**：\`vx_high_vel_mean >= 0.6\`（T1e 0.60），且 \`vx_low/vx_mid\` 转正
  * **风格**：\`track_default_pose_on_stand\`、\`foot_slip\`、\`body_ang_vel_xy\` 明显好转
  * **姿态**：躯干 pitch 均值 |·| 下降（后倾改善）
EOF
