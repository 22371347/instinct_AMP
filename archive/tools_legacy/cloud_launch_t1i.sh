#!/usr/bin/env bash
# ============================================================================
# T1i 启动器（云端）：**热启动自 T1a2 的 model_7000.pt**，只改风格奖励口径
# ============================================================================
# TASK = Instinct-Amp-Walk-T1i-G1-v0
#   环境：**复用 `G1AmpWalkT1a2EnvCfg`**（奖励表/契约逐字不动 ⇒ 环境零改动）
#   算法：T1a2 的 PPO + 只改 3 个字段
#         amp_reward_form           "raw"      → "bounded_quad"（= mjlab 的有界二次）
#         discriminator_reward_type "quad"     → "wasserstein"（bounded_quad 需要原始 d）
#         disable_style_on_stand    False      → True（mjlab 变体 B）
#   起点：**热启动自 T1a2 的 model_7000.pt** ⇒ 与 T1b/T1c/T1d/T1e 同起点可比
#   预算：不设上限（跑到 ≥9000 与 T1b~T1e 的 7100~8000 区间对照）
#
# ⚠️ 单 GPU：会 pkill 掉当前任何训练进程（现在应无训练在跑）。
#
# 用法（云端）：bash /root/cloud_launch_t1i.sh [budget_iters] [num_envs]
# ============================================================================
set -uo pipefail

BUDGET="${1:-1000000000}"
INST="${2:-4096}"
TASK="Instinct-Amp-Walk-T1i-G1-v0"
TAG="t1i_style0p975_from_t1a2"
LOG="/root/train_${TAG}.log"
GUARD_LOG="/root/guard_${TAG}.log"
EXPERT=/root/amp_expert_states_396.pt
BROKEN_MARK=/root/amp_expert_states_396_broken_linvelb.pt
RUNROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk
#: ⭐ 起点：T1a2 的 run（用**唯一子串**避免 T1b/T1c/T1d 目录撞车）
BASE_PAT="t1a2_t0impl_t0explore"
CKPT="model_7000.pt"

echo "=== 1) 定位 T1a2 的 run（热启动源）==="
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

echo "=== 3) 发车前自检（默认跳过独立校验：自检已烘入 cfg 的 __post_init__）==="
#: 用户 2026-10-08 要求"明天不校验直接启动" ⇒ 默认 SKIP_VERIFY=1。
#: 安全性由 `amp_walk_t1i_ppo_cfg.py` 的 fail-fast 断言保证（配置错 ⇒ 训练 9 秒内自行退出）。
if [[ "${SKIP_VERIFY:-1}" == "0" ]]; then
  cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab || exit 1
  source /root/.llp_env.sh
  unset PYTHONPATH
  /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_t1i.py 2>&1 | tee /root/verify_t1i.out
  VERIFY_RC="${PIPESTATUS[0]}"
  grep -q '✅ T1i 校验全部通过' /root/verify_t1i.out || VERIFY_RC=1
  echo "  [gate] python RC=$VERIFY_RC"
  if [[ "$VERIFY_RC" -ne 0 ]]; then
    echo "  FAIL 校验失败 ⇒ **不发车**" >&2; exit 1
  fi
else
  echo "  [gate] SKIP_VERIFY=1 ⇒ 跳过（配置内 fail-fast 仍生效）"
fi

echo "=== 4) 停掉当前训练（若有）==="
for p in $(pgrep -f 'train[.]py'); do echo "  kill -9 $p"; kill -9 "$p"; done
for p in $(pgrep -f 'v29_guard[.]sh'); do echo "  kill -9 $p"; kill -9 "$p"; done
sleep 3
pgrep -af 'train[.]py' || echo "  OK 已无 train.py 进程"

echo "=== 5) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  FAIL 没有可用 GPU" >&2; exit 1; }

echo "=== 6) 热启动 T1i（from $CKPT）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v33.sh $INST $BUDGET cuda:0 '$TAG' '$LOAD_RUN' '$CKPT' '$TASK'" \
  "$LOG"

echo "=== 7) 拉起崩溃守护 ==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh '$LOG' 30 50 20 1.5" "$GUARD_LOG"

cat <<'EOF'
=== done（T1i）===
必须核对（**有效性检查**）：
  * `resume    : 续训  from <T1a2 run> / model_7000.pt`
  * `[T1i][STYLE FORM ALIGN]` 回读打印：bounded_quad / 0.25 / wasserstein / True
  * `[AMP-WALK PASS] policy=378 ... rewards=21`（契约未变）
  * `[wasabi] ... /root/amp_expert_states_396.pt`
预注册判据（与 T1b/T1c/T1d/T1e **同起点**对照）：
  * ⭐ `vx_low_vel_mean / vx_mid_vel_mean` **转正**（T1d/T1e 为 −0.19…−0.25）
  * ⭐ `vx_high_vel_mean` 保持 ≥ T1a2 的 0.62 区间（不退化；T1d 0.42 / T1e 0.60）
  * ⭐ `wz_low/mid/high_vel_mean` 脱离 ≈0（T1a2~T1e 都是 ~0.01）
  * `amp_stand_frac` 应≈0（站立时风格已被关闭）且 `amp_style_std` 有界
  * `track_default_pose_on_stand` 不再恶化（站立姿态不再被风格拉扯）
  * 踝 roll 的 `tracking_joint_error_rmse` 参照（热启动臂预计仍 ~0.26，本臂不负责修它）
EOF
