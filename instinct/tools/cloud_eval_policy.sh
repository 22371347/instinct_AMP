#!/usr/bin/env bash
# ============================================================================
# 策略验证：用 play.py 跑训练好的 checkpoint
#   ① 录像（多环境）→ <run>/videos/play/*.mp4
#   ② 导出 ONNX（单环境）→ <run>/exported/*.onnx   —— 给本地 MuJoCo 部署用
#
# ⚠️ 三条实测踩到的硬约束（写死在这个脚本里，别再踩）：
#   1. `--exportonnx` **只支持单环境**：`assert env.unwrapped.num_envs == 1`
#      → 和 `--num_envs 16` 一起用会直接 AssertionError 崩掉整次运行（连录像都没了）。
#   2. `play.py` 主循环是 `while simulation_app.is_running()`，**只有带 `--video`
#      时才会在 `video_length` 步后 break**。不加 `--video` 的后台任务永远不会自己结束。
#   3. `--exportonnx` 在 `if agent_cfg.load_run is not None:` 里 → 必须给 `--load_run`。
#
# 用法：
#   bash cloud_eval_policy.sh [run_dir] [checkpoint] [video_steps]
#   bash cloud_eval_policy.sh 20260925_003514_G1AmpWalkPlane_obs626_directional_rew5_amp_walk_4096 model_5000.pt 500
#   （run_dir 也可从 /root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk 里自动取最新）
set -uo pipefail

XU=/root/gpufree-data/instinct_xu
LOGROOT="$XU/logs/instinct_rl/g1_amp_walk"
RUN_ARG="${1:-}"
CKPT="${2:-model_5000.pt}"
VID_STEPS="${3:-500}"
VID_ENVS="${4:-16}"
#: 可覆盖任务 id（v2 用 Instinct-Amp-Walk-V2-G1-Play-v0）。
#: 奖励在 play 阶段不生效，且 v1/v2 的观测/动作契约完全相同，
#: 所以两者可以互用；但显式传对名称能避免日志误导。
TASK="${TASK:-Instinct-Amp-Walk-G1-Play-v0}"

cd "$XU" || exit 1
[[ -f /root/.llp_env.sh ]] && source /root/.llp_env.sh
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

RUN="$RUN_ARG"
if [[ -z "$RUN" ]]; then
  RUN="$(ls -dt "$LOGROOT"/*/ 2>/dev/null | head -1)"
  RUN="$(basename "${RUN%/}")"
fi
RUNDIR="$LOGROOT/$RUN"
echo "=== 策略验证 run=$RUN ckpt=$CKPT video=${VID_STEPS}步 ==="
ls -l "$RUNDIR/$CKPT" 2>/dev/null || { echo "❌ checkpoint 不存在: $RUNDIR/$CKPT"; exit 2; }

# ── ① 录像（多环境；带 --video 才会自然结束）────────────────────────────────
echo
echo "---------- ① 录像（${VID_ENVS} 环境 × ${VID_STEPS} 步）----------"
./play.sh \
  --task "$TASK" \
  --load_run "$RUN" --checkpoint "$CKPT" \
  --num_envs "$VID_ENVS" --device cuda:0 --headless \
  --video --video_length "$VID_STEPS" 2>&1 | tail -20
echo "录像退出码: ${PIPESTATUS[0]}"

# ── ② 导出 ONNX（单环境）────────────────────────────────────────────────────
echo
echo "---------- ② 导出 ONNX（单环境）----------"
./play.sh \
  --task "$TASK" \
  --load_run "$RUN" --checkpoint "$CKPT" \
  --num_envs 1 --device cuda:0 --headless \
  --exportonnx 2>&1 | tail -15
echo "导出退出码: ${PIPESTATUS[0]}"

# ── 产物 ───────────────────────────────────────────────────────────────────
echo
echo "======== 产物 ========"
find "$RUNDIR/videos" -name '*.mp4' 2>/dev/null | while read -r f; do
  printf '  MP4  %s  (%s)\n' "$f" "$(du -h "$f" | cut -f1)"
done
find "$RUNDIR/exported" -name '*.onnx' 2>/dev/null | while read -r f; do
  printf '  ONNX %s  (%s)\n' "$f" "$(du -h "$f" | cut -f1)"
done
echo "======== 验证结束 ========"
