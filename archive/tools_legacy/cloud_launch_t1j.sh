#!/usr/bin/env bash
# ============================================================================
# T1j 启动器（云端）：**解除"相对夹角的常数拉力"**（相对 T1i 的两项改动）
# ============================================================================
# TASK = Instinct-Amp-Walk-T1j-G1-v0
#   环境：`amp_walk_t1j_env_cfg.py` = T1a2 的环境 + 只置零
#         `joint_deviation_torso`（作用于 `waist_.*`，**含 waist_pitch**）
#         `joint_deviation_knee`（把膝角拉向常数）
#   算法：**复用 T1i 的 agent cfg**（λ=0.975 ⇒ 风格强度 = mjlab）
#   起点：**热启动自 T1a2 的 model_7000.pt**（与 T1i/T1b/T1c/T1d 同起点）
#   预算：不设上限
#
# ⚠️ 自查：环境侧的 fail-fast 断言烘在 `__post_init__`（守卫基线权重 + 非零项数=14），
#   配置错 ⇒ 训练 ~9 秒内自行退出。
#
# 用法（云端）：bash /root/cloud_launch_t1j.sh [budget_iters] [num_envs]
# ============================================================================
set -uo pipefail

BUDGET="${1:-1000000000}"
INST="${2:-4096}"
TASK="Instinct-Amp-Walk-T1j-G1-v0"
TAG="t1j_noconstantpull_from_t1a2"
LOG="/root/train_${TAG}.log"
GUARD_LOG="/root/guard_${TAG}.log"
EXPERT=/root/amp_expert_states_396.pt
BROKEN_MARK=/root/amp_expert_states_396_broken_linvelb.pt
RUNROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk
BASE_PAT="t1a2_t0impl_t0explore"      # 唯一子串，避免 T1b/T1c/T1d 目录撞车
CKPT="model_7000.pt"

echo "=== 1) 定位热启动源（T1a2 的 run）==="
LOAD_RUN="$(basename "$(ls -dt "$RUNROOT"/*"$BASE_PAT"* 2>/dev/null | head -1)")"
if [[ -z "$LOAD_RUN" ]]; then echo "  FAIL 找不到 T1a2 run（pat=$BASE_PAT）" >&2; exit 1; fi
echo "  LOAD_RUN = $LOAD_RUN"
if [[ ! -f "$RUNROOT/$LOAD_RUN/$CKPT" ]]; then
  echo "  FAIL 缺 $RUNROOT/$LOAD_RUN/$CKPT；该目录现有：" >&2
  ls "$RUNROOT/$LOAD_RUN" 2>/dev/null | grep 'model_.*\.pt' | sort -V | tr '\n' ' ' >&2; echo >&2
  exit 1
fi
echo "  OK $RUNROOT/$LOAD_RUN/$CKPT"

echo "=== 2) 专家集必须是**修好的**那份 ==="
if [[ "$EXPERT" -nt "$BROKEN_MARK" ]]; then echo "  OK $EXPERT 比旧备份新"; else
  echo "  FAIL $EXPERT 不比旧备份新" >&2; exit 1; fi
/opt/conda/envs/isaaclab/bin/python - <<'PY'
import torch
raw = torch.load('/root/amp_expert_states_396.pt', map_location='cpu')
x = (next(v for v in raw.values() if isinstance(v, torch.Tensor)) if isinstance(raw, dict) else raw).float()
n = x[:, 234:237].norm(dim=1)
print(f"    anchor |v| mean = {float(n.mean()):.4f} m/s（应 ~0.5-0.7）")
raise SystemExit(0 if float(n.mean()) > 0.1 else 1)
PY
if [[ $? -ne 0 ]]; then echo "  FAIL 专家集根速度仍近零" >&2; exit 1; fi

echo "=== 3) 停掉当前训练（⚠️ 会停掉仍在跑的 T1i）==="
for p in $(pgrep -f 'train[.]py'); do echo "  kill -9 $p"; kill -9 "$p"; done
for p in $(pgrep -f 'v29_guard[.]sh'); do echo "  kill -9 $p"; kill -9 "$p"; done
sleep 3
pgrep -af 'train[.]py' || echo "  OK 已无 train.py 进程"

echo "=== 4) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  FAIL 没有可用 GPU" >&2; exit 1; }

echo "=== 5) 热启动 T1j（from $CKPT）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v33.sh $INST $BUDGET cuda:0 '$TAG' '$LOAD_RUN' '$CKPT' '$TASK'" \
  "$LOG"

echo "=== 6) 拉起崩溃守护 ==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh '$LOG' 30 50 20 1.5" "$GUARD_LOG"

cat <<'EOF'
=== done（T1j）===
必须核对：
  * `resume    : 续训  from <T1a2 run> / model_7000.pt`
  * `[T1j][NO CONSTANT PULL]` 回读打印：两项 −0.100/−0.050 → +0.000、非零奖励项 = 14
  * `[AMP-WALK PASS] policy=378 ... rewards=21`（契约未变）
  * `[wasabi] ... /root/amp_expert_states_396.pt`
预注册判据（对照 **T1i**，同起点、同 λ=0.975）：
  * ⭐ **倾角代理**：`−Episode_Reward/flat_orientation_l2` 明显小于 T1i 的 **0.0353**
    （目标 < 0.020，≈<8°；T1f 为 0.078）
  * ⭐ `vx_high_vel_mean` **不退化**（T1i 全程 0.55–0.67）
  * ⭐ `waist_pitch` 的跟踪误差 `tracking_joint_error_rmse/waist_pitch_joint` 应下降
  * ✅ 副作用检查：`joint_deviation_torso/knee` 归零后 `action_rate_l2` / `dof_torques_l2`
    不应显著恶化（若恶化 ⇒ 说明那两项确实在抑制抖动）
EOF
