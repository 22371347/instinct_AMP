#!/usr/bin/env bash
# ============================================================================
# T2 冷启动（**跑在云端**）：全权照抄 AMP_mjlab walk + 3 处例外 + 真 RSI
# ============================================================================
# TASK = Instinct-Amp-Walk-T2-G1-v0
#   * actor 观测 **1500** = height_scan 279×4 + 本体感受 96×4（例外 C：height_scan 也发 4 帧）
#   * 奖励换成 mjlab 的 **12 项**（13 项里缺 self_collisions，我们无自碰撞传感器）
#   * 命令 = mjlab 的分档结构（⚠️ 速度包络按用户决策收到 (-0.5, 1.0)）
#   * 终止 = time_out / bad_orientation 70° / bad_base_height 0.5（**去掉 base_contact**）
#   * 事件 = 强推（1~3 s）+ base_com；**去掉** reset_base / reset_robot_joints / add_base_mass
#   * 复位 = ⭐ **真 RSI**（`reset_robot_state_by_reference`，起点在 clip 内均匀采样）
#   * 算法 = λ 0.75 / c 0.1（风格 0.025）、判别器 [1024,512,256] + 白化 + GP10(expert)
#            + trunk/head wd 1e-3/1e-2 + 离线专家集；例外 B：噪声 0.25×
#   起点：**从零**（不 resume；obs 维度变了）
#
# ⚠️ **为什么空字符串要写在这个文件里**：命令要经
#      `本地 → ssh → cloud_run_bg.sh → bash -c → train.py` 五层，
#    命令行里传 `''` 很容易被某一层引号吃掉（本项目踩过多次；见
#    `cloud_launch_v14_scratch.sh` 的同款注释）。写进文件后传的只有 1 个 token。
#
# 用法（云端）：bash /root/cloud_launch_t2_remote.sh [budget_iters] [num_envs]
# ============================================================================
set -uo pipefail

BUDGET="${1:-1000000000}"
INST="${2:-4096}"
TASK="Instinct-Amp-Walk-T2-G1-v0"
TAG="t2_mjlabcopy_obs1500_rsi_cold"
LOG="/root/train_${TAG}.log"
GUARD_LOG="/root/guard_${TAG}.log"
EXPERT=/root/amp_expert_states_396.pt
BROKEN_MARK=/root/amp_expert_states_396_broken_linvelb.pt

echo "=== 1) 专家集必须是**修好的**那份（T2 的 AMP 状态仍是 396 ⇒ 继续有效）==="
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

echo "=== 5) 拉起崩溃守护（⭐ **放宽判据**，见下）==="
# ⚠️ 为什么放宽：T2 是 **RSI 冷启动**（机器人被扔到参考帧的任意相位）+ mjlab 的 1~3 s 强推，
#    早期 "一启动就倒"（实测 ~19 步、`Episode_Termination/bad_base_height ≈ 0.93`）是**预期形态**。
#    对照：T1o 冷启动早期 episode length = 4.1 / 20.0 / 9.8 … 第 10 块 51、第 20 块 331。
#    默认判据（连续 20 块 < 50）在 T2 上有**误杀风险** ⇒ 改为：
#      判据①：最近 **30** 块全部 < **30**（才算彻底卡住）
#      判据②：最近 3 块 action_abs_mean 全部 > **2.5**（配合 scale=1.0 的动作量级）
#    训练中途想换判据：`bash /root/cloud_guard_t2_relaxed.sh`（上传版）。
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh '$LOG' 30 30 30 2.5" "$GUARD_LOG"

cat <<EOF
=== done（T2，冷启动）===
必须核对：
  * ⚠️ 日志中**不应**出现 \`resume    : 续训\`（冷启动）
  * \`[T2][COPY MJLAB] actor 观测 378 → **1500**\`，逐项 = (1116, 12, 12, 12, 116, 116, 116)
  * \`[T2][COPY MJLAB] 奖励 12 项\`（含 track_anchor_linear_velocity / feet_crossing 等）
  * \`[AMP-WALK PASS] policy=1500\` + \`cmd_ranges=((-0.5, 1.0), (-1.0, 1.0), (-1.57, 1.57))\`
  * \`[T2][COPY MJLAB] RSI 起点 (0.0, 0.0) → (0.0, 1.0)\`（⭐ 真 RSI）
  * \`[T2][COPY MJLAB · ALGO] 风格强度 (1−λ)·c = 0.0250\`
  * \`[AMP-WALK WEIGHTS] ...\`（跨版本 diff 用）

本臂判据（**与 mjlab 复现性直接相关**）：
  * ⭐ **步态**：抬脚 / 交替 / 不跳 —— 本臂唯一的目的
  * \`Style/amp_style_zero_frac\`：能否长期 < 0.3（T1l/T1n 是 0.35~0.38 的死区）
  * \`Loss/discriminator_loss\`：应显著 > 0.05（T1l/T1n 是 0.0014~0.035 = 饱和）
  * \`Step/amp_d\` / \`amp_d_ref\`：不应贴边（T1l/T1n 是 −0.94 / +0.93）
  * \`Mean episode length\` 与 \`Mean reward\` 的爬升斜率
  * ⚠️ 风格强度只有 T1l 的 1/4 ⇒ \`Episode_Reward\` 的构成会明显不同（不可与 T1* 逐项比大小）
EOF
