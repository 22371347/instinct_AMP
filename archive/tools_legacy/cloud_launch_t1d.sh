#!/usr/bin/env bash
# ============================================================================
# T1d 启动器（跑在云端）：**修好专家集后重跑 T1b**
# ============================================================================
# 背景（2026-10-07 断点定位）：
#   `cloud_export_amp_expert.py` 曾用缓冲里的 `link_lin_vel_b`（= d(base 相对位姿)/dt，
#   **锚点恒为 0**）当速度通道，而训练观测用的是
#   `_vel_in_body_frame(link_quat_w, link_lin_vel_w)`（世界速度转自身系）。
#   实测：专家集 anchor |v| = 0.013 m/s vs 源数据 0.750 m/s ⇒ **差 57×**。
#   ⇒ 判别器的「真」样本里根速度被清零 ⇒ 学到「根不动 = 真」⇒ 风格项惩罚根位移
#   ⇒ T1b（λ=0.75）/ T1c（λ=0.9）都在 ~700 轮内把 `vx_high` 从 0.6 压到 0.02，
#     且**与风格权重无关**（方向性偏置）。
#
# 本臂的设计：**配置一字未动，只换数据** —— 直接复用 T1b 的任务
#   TASK = Instinct-Amp-Walk-T1b-G1-v0（= T1a2 的 env cfg + λ=0.75 的 agent cfg）
#   ⇒ T1b（坏专家集）vs T1d（修好的专家集）是"唯一变量 = 数据"的最干净对照。
#
# 起点：热启动自 **T1a2** 的 `model_7000.pt`（与 T1b 同源，可逐轮对照）
# 预算：不设上限
#
# 用法（云端）：bash /root/cloud_launch_t1d.sh [budget_iters] [num_envs] [ckpt]
# ============================================================================
set -uo pipefail

BUDGET="${1:-1000000000}"
INST="${2:-4096}"
CKPT="${3:-model_7000.pt}"
TASK="Instinct-Amp-Walk-T1b-G1-v0"
TAG="t1d_style0p75_fixedexp"
LOAD_RUN="20261007_032827_G1AmpWalkT1a2MVPPlane_obs378_rew16_t0impl_sym_t1a2_t0impl_t0explore"
LOG="/root/train_${TAG}.log"
GUARD_LOG="/root/guard_${TAG}.log"
EXPERT=/root/amp_expert_states_396.pt
FIXED_MARK=/root/amp_expert_states_396_broken_linvelb.pt
RUNROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk

echo "=== 1) 停掉当前训练 ==="
pkill -9 -f 'train[.]py' || true
sleep 3
pgrep -af 'train[.]py' | grep -v pgrep || echo "  OK 已无 train.py 进程"

echo "=== 2) ⭐ 检查专家集确实是**修好的**那份（不是回滚成了旧备份）==="
if [[ ! -f "$EXPERT" ]]; then
  echo "  FAIL 缺 $EXPERT" >&2; exit 1
fi
ls -l "$EXPERT" "$FIXED_MARK" 2>/dev/null
if [[ "$EXPERT" -nt "$FIXED_MARK" ]]; then
  echo "  OK 当前 $EXPERT 比旧备份更新（应为重导产物）"
else
  echo "  FAIL $EXPERT 不比旧备份新 ⇒ 可能没重导成功，先别跑" >&2; exit 1
fi
echo "  ── 锚点根速度自检（必须 ~0.5 m/s 量级；若 ~0.01 说明仍是坏口径）──"
/opt/conda/envs/isaaclab/bin/python - <<'PY'
import torch
raw = torch.load('/root/amp_expert_states_396.pt', map_location='cpu')
x = (next(v for v in raw.values() if isinstance(v, torch.Tensor)) if isinstance(raw, dict) else raw).float()
v = x[:, 234:237]                       # body0 帧 t 的线速度
n = v.norm(dim=1)
print(f"    anchor |v| mean = {float(n.mean()):.4f} m/s   max = {float(n.max()):.4f}")
print("    ✅ 正常" if float(n.mean()) > 0.1 else "    ⛔ 仍近零 ⇒ 口径没修好，停止发车")
raise SystemExit(0 if float(n.mean()) > 0.1 else 1)
PY
[[ $? -ne 0 ]] && { echo "  FAIL 专家集根速度仍近零" >&2; exit 1; }

echo "=== 3) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  FAIL 没有可用 GPU ⇒ 不启动" >&2; exit 1; }

echo "=== 4) 检查热启动源 ==="
if [[ -f "$RUNROOT/$LOAD_RUN/$CKPT" ]]; then
  echo "  OK $RUNROOT/$LOAD_RUN/$CKPT"
else
  echo "  FAIL 找不到热启动源：$RUNROOT/$LOAD_RUN/$CKPT" >&2; exit 1
fi

echo "=== 5) 起 T1d（task=$TASK，λ=0.75，热启动 from $CKPT）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v33.sh $INST $BUDGET cuda:0 '$TAG' '$LOAD_RUN' '$CKPT' '$TASK'" \
  "$LOG"

echo "=== 6) 拉起崩溃守护 ==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh '$LOG' 30 50 20 1.5" "$GUARD_LOG"

cat <<EOF
=== done（T1d）===
日志：$LOG     守护：$GUARD_LOG
必须核对：
  * \`resume    : 续训  from $LOAD_RUN / $CKPT\`
  * AMP 教师契约横幅（obs 378 / action 29 / scale 1.0 / 50 Hz）
  * \`params/agent.yaml\`：style_lerp=0.75（与 T1b **逐字相同**）
  * \`[wasabi] 专家批来自离线数据集：$EXPERT\`
预注册判据（**与 T1b 逐轮对照，唯一变量 = 专家集**）：
  * ⭐ **成败线**：\`vx_high_vel_mean\` 必须保持 >= 0.5（T1b 同期崩到 0.014/0.030）
      对照点（全局轮次）：7200 / 7400 / 7600 / 7800 / 8000
  * 风格项应仍在发力：\`track_ang_vel_z_exp\` 高于 T1a2 的 0.156
  * 平滑/寿命：\`action_rate_l2\` 惩罚绝对值 < T1a2 的 0.456、\`ep_len\` > 621
  * 若这次守住速度且风格改善 ⇒ **"判别器破坏高速行走"的真因就是这个数据链 bug**
EOF
