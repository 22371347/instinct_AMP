#!/usr/bin/env bash
# ============================================================================
# T1o 冷启动（**跑在云端**）：663 观测契约（本体感受 4 帧）+ 从 0 轮开始
# ============================================================================
# TASK = Instinct-Amp-Walk-T1o-G1-v0
#   环境侧：actor 观测 **378（单帧）→ 663** = height_scan 279（单帧）
#           + 本体感受 4 帧 × 96（= mjlab 的 actor 输入 384）；❌ 删 base_lin_vel
#   算法侧：沿用 T1n（判别器白化 + GP 在归一化空间）
#   起点：**从零**（不 resume）
#
# ⚠️ **为什么空字符串要写在这个文件里**：命令要经
#      `本地 → ssh → cloud_run_bg.sh → bash -c → train.py` 五层，
#    命令行里传 `''` 很容易被某一层引号吃掉（本项目踩过多次；见
#    `cloud_launch_v14_scratch.sh` 的同款注释）。写进文件后传的只有 1 个 token。
#
# 用法（云端）：bash /root/cloud_launch_t1o_remote.sh [budget_iters] [num_envs]
# ============================================================================
set -uo pipefail

BUDGET="${1:-1000000000}"
INST="${2:-4096}"
TASK="Instinct-Amp-Walk-T1o-G1-v0"
TAG="t1o_obs663_h4_cold"
LOG="/root/train_${TAG}.log"
GUARD_LOG="/root/guard_${TAG}.log"
EXPERT=/root/amp_expert_states_396.pt
BROKEN_MARK=/root/amp_expert_states_396_broken_linvelb.pt

echo "=== 1) 专家集必须是**修好的**那份（AMP 状态 396 **未变** ⇒ 专家集继续有效）==="
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

echo "=== 2) 停掉当前训练 / 守护 ==="
for p in $(pgrep -f 'train[.]py'); do echo "  kill -9 $p"; kill -9 "$p"; done
for p in $(pgrep -f 'v29_guard[.]sh'); do echo "  kill -9 $p"; kill -9 "$p"; done
sleep 3
pgrep -af 'train[.]py' || echo "  OK 已无 train.py 进程"

echo "=== 3) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || {
  echo "  FAIL 没有可用 GPU" >&2; exit 1; }

echo "=== 4) ⭐ 从零起车（\$5/\$6 传**空** ⇒ 不加 --resume）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v33.sh $INST $BUDGET cuda:0 '$TAG' '' '' '$TASK'" \
  "$LOG"

echo "=== 5) 拉起崩溃守护 ==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh '$LOG' 30 50 20 1.5" "$GUARD_LOG"

cat <<EOF
=== done（T1o，冷启动）===
必须核对：
  * ⚠️ 日志中**不应**出现 \`resume    : 续训\`（冷启动）
  * \`[T1o][HISTORY = 4]\` 逐项维度 = (279, 12, 12, 12, 116, 116, 116)（sum=663）
  * \`[T1o][COLD START]\` 算法沿用 T1n（白化 + GP normalized）
  * \`[AMP-WALK PASS] policy=663\`（契约已改；旧臂是 378）
  * \`[wasabi] ... /root/amp_expert_states_396.pt\`（AMP 状态 396 未变）
  * \`Mean episode length\` 从**个位数**开始爬 —— 冷启动的正常现象

本臂判据：
  * ⭐ \`Mean episode length\` 与 \`Mean reward\` 的**爬升斜率**（对比 \`V33ACold\` 的 378 冷启动）
  * ⭐ \`vx_high\` / \`track_lin_vel_xy\`：删了 base_lin_vel 后速度跟踪会不会变差
  * ⭐ **步态本身**：抬脚高度 / 交替性（这是本臂唯一的目的）
  * \`Style/amp_style_zero_frac\`（判别器饱和程度）—— 冷启动下从 0 开始演化，可看全轨迹
EOF
