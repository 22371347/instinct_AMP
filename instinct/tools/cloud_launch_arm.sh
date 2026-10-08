#!/usr/bin/env bash
# ============================================================================
# **通用发车器**（云端）：一条命令拉起任意"热启动 + 硬闸门"的训练臂
# ============================================================================
# 用法：
#   bash /root/cloud_launch_arm.sh <TASK> <TAG> [BASE_PAT] [CKPT] [NUM_ENVS] [BUDGET]
#
#   <TASK>      注册的任务 id，如 Instinct-Amp-Walk-T1k-G1-v0
#   <TAG>       日志/run 名后缀，如 t1k_height_footsmall
#   [BASE_PAT]  热启动源的 run 目录**唯一子串**（默认 t1a2_t0impl_t0explore）
#   [CKPT]      检查点名（默认 model_7000.pt）
#   [NUM_ENVS]  默认 4096
#   [BUDGET]    默认不限
#
# ⚠️ 单 GPU：会 **pkill 掉当前任何训练进程**。
# ⚠️ 闸门：① 热启动源存在；② 专家集比旧备份新且 anchor |v| > 0.1；③ GPU 可用。
#    配置层自检由各 env cfg 的 `__post_init__` fail-fast 承担（配置错 ⇒ 训练 ~9 秒自行退出）。
#
# 例：
#   bash /root/cloud_launch_arm.sh Instinct-Amp-Walk-T1k-G1-v0 t1k_height_footsmall
# ============================================================================
set -uo pipefail

TASK="${1:?需要 TASK}"
TAG="${2:?需要 TAG}"
BASE_PAT="${3:-t1a2_t0impl_t0explore}"
CKPT="${4:-model_7000.pt}"
INST="${5:-4096}"
BUDGET="${6:-1000000000}"

LOG="/root/train_${TAG}.log"
GUARD_LOG="/root/guard_${TAG}.log"
EXPERT=/root/amp_expert_states_396.pt
BROKEN_MARK=/root/amp_expert_states_396_broken_linvelb.pt
RUNROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk

echo "=== 0) 参数 ==="
echo "  TASK=$TASK"
echo "  TAG=$TAG"
echo "  BASE_PAT=$BASE_PAT   CKPT=$CKPT   NUM_ENVS=$INST"

echo "=== 1) 定位热启动源（唯一子串 ⇒ 避免 T1b/T1c/T1d 目录撞车）==="
LOAD_RUN="$(basename "$(ls -dt "$RUNROOT"/*"$BASE_PAT"* 2>/dev/null | head -1)")"
if [[ -z "$LOAD_RUN" ]]; then echo "  FAIL 找不到 run（pat=$BASE_PAT）" >&2; exit 1; fi
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

echo "=== 3) 停掉当前训练 / 守护 ==="
for p in $(pgrep -f 'train[.]py'); do echo "  kill -9 $p"; kill -9 "$p"; done
for p in $(pgrep -f 'v29_guard[.]sh'); do echo "  kill -9 $p"; kill -9 "$p"; done
sleep 3
pgrep -af 'train[.]py' || echo "  OK 已无 train.py 进程"

echo "=== 4) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  FAIL 没有可用 GPU" >&2; exit 1; }

echo "=== 5) 热启动起车（from $CKPT）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v33.sh $INST $BUDGET cuda:0 '$TAG' '$LOAD_RUN' '$CKPT' '$TASK'" \
  "$LOG"

echo "=== 6) 拉起崩溃守护 ==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh '$LOG' 30 50 20 1.5" "$GUARD_LOG"

cat <<EOF
=== done（$TAG）===
TASK=$TASK
日志：$LOG
热启动源：$LOAD_RUN / $CKPT
必须核对：
  * \`resume    : 续训  from $LOAD_RUN / $CKPT\`
  * 该臂 env cfg 的 \`[T1?][...]\` 回读打印（含非零奖励项计数）+ 算法侧的 \`[T1?][...]\`
  * \`[AMP-WALK PASS] policy=378\`（契约未变）
  * \`[wasabi] ... $EXPERT\`
EOF
