#!/usr/bin/env bash
# ============================================================================
# T5 **冷启动**（跑在云端）：高度项换体（量躯干 → 量骨盆）+ 从零训练
# ============================================================================
# TASK = Instinct-Amp-Walk-T5-G1-v0        TAG = t5_pelvis_height_cold
#   环境侧唯一改动（相对 T2 的 mjlab 照抄版）：
#     `track_base_height`(root=torso_link, 靶 0.82) → **None**
#     新增 `track_pelvis_height`(pelvis, 靶 **0.7585**, weight 1.5 / std 0.3)
#       * 0.7585 = MJCF FK 实测的 "default 站姿 + 脚掌着地" 骨盆离地高度
#       * 语义上等价于"平均保持默认屈膝量（膝≈0.669）"；当前策略是伸腿走（膝 p50 0.38）
#         ⇒ 这一项要把它按回屈膝 ⇒ 重心后移 ⇒ 必须靠前倾躯干重新平衡（这才是"前倾"的通路）
#     ⇒ 非零奖励项仍 12；obs/动作/资产/命令/终止/事件/RSI **一字未改**
#   算法 = 与 T3 相同（`fresh_discriminator_on_load=False`；冷启动下是 no-op，但保留正确语义）
#   起点：**从零**（不 resume）—— 用户 2026-10-10 决策：
#     "warm 已被 T3/T4/T5warm 三次证明解不开姿态习惯，姿态/结构类改动一律走冷启动"
#
# ⚠️ **为什么空字符串必须写在文件里**：命令经
#      `本地 → ssh → cloud_run_bg.sh → bash -c → train.py` 五层，
#    命令行传 `''` 会被某一层引号吃掉（本项目踩过多次）⇒ 写进文件只传 1 个 token。
#
# ⚠️ **冷启动判读纪律**（重要）：前倾本质是**风格特征**，而冷启动初期判别器可能是"死的"
#    （`Step/amp_d_ref` 贴死 ≈+0.9 / `Step/amp_style_zero_frac` > 0.35）——
#    那种状态下**不出现前倾是正常现象，不得提前下结论**。
#    必须等 ① 判别器活了（amp_d_ref 掉到 +0.5 量级）之后，再看 ② 躯干世界系前倾是否开始爬。
#
# 用法（云端）：bash /root/cloud_launch_t5_cold_remote.sh [budget_iters] [num_envs]
# ============================================================================
set -uo pipefail

BUDGET="${1:-1000000000}"
INST="${2:-4096}"
TASK="Instinct-Amp-Walk-T5-G1-v0"
TAG="t5_pelvis_height_cold"
LOG="/root/train_${TAG}.log"
GUARD_LOG="/root/guard_${TAG}.log"
EXPERT=/root/amp_expert_states_396.pt
BROKEN_MARK=/root/amp_expert_states_396_broken_linvelb.pt

echo "=== 0) T5 代码必须在位（新奖励函数 + 新字段）==="
R=/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab
for f in "$R/tasks/locomotion/mdp/rewards_amp.py" \
         "$R/tasks/locomotion/config/g1/amp_walk_t5_env_cfg.py" \
         "$R/tasks/locomotion/config/g1/agents/amp_walk_t5_ppo_cfg.py"; do
  [[ -f "$f" ]] && echo "  OK $f" || { echo "  FAIL 缺 $f" >&2; exit 1; }
done
grep -q 'def track_pelvis_height_exp' "$R/tasks/locomotion/mdp/rewards_amp.py" \
  && echo "  OK 新奖励函数在位" || { echo "  FAIL 缺 track_pelvis_height_exp" >&2; exit 1; }

echo "=== 1) 专家集必须是**修好的**那份（AMP 状态仍 396）==="
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

echo "=== 2) 停掉当前训练 / 守护（含 warm 的 T5）==="
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

echo "=== 5) 拉起崩溃守护（**放宽判据**：RSI 冷启动早期「一启动就倒」是预期形态）==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh '$LOG' 30 30 30 2.5" "$GUARD_LOG"

cat <<EOF
=== done（T5，冷启动）===
必须核对：
  * ⚠️ 日志中**不应**出现 \`resume    : 续训\`（冷启动）
  * \`[T5][换体] 高度项：root(torso_link) → **pelvis**\` + \`非零奖励项仍为 12\` + 目标取舍三行
  * \`[AMP-WALK PASS] policy=1500 ... rewards=12\`（契约一字未改）
  * \`[AMP-WALK WEIGHTS]\` 里**无** \`track_base_height\`、**有** \`track_pelvis_height=1.5\`
  * \`[T2][COPY MJLAB] RSI 起点 (0.0, 0.0) → (0.0, 1.0)\`（⭐ 真 RSI 仍在）
  * \`[wasabi] ... /root/amp_expert_states_396.pt\`

判读（**先看判别器是否活，再看姿态**）：
  * ① \`Step/amp_d_ref\` 是否从 ≈0.9 掉到 **+0.5 量级**、\`amp_style_zero_frac\` 是否 < 0.30
        —— 没活之前**不得**用"没有前倾"下结论
  * ② 探针 \`Step/pelvis_height\` 是否收敛到 **0.7585** 附近（±2~3 cm）
  * ③ 回放取证的**躯干世界系前倾**是否往参考的 **+10.10°** 爬（基线：T2/T3/T4 都是 +0.8~3.3°）
EOF
