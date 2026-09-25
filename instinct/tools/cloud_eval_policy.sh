#!/usr/bin/env bash
# ============================================================================
# 策略验证：从训练好的 checkpoint 导出 ONNX（→ <run>/exported/），给本地 MuJoCo 用
#
# ⚠️ 三条实测踩到的硬约束（写死在这个脚本里，别再踩）：
#   1. `--exportonnx` **只支持单环境**：`assert env.unwrapped.num_envs == 1`
#      → 和 `--num_envs 16` 一起用会直接 AssertionError 崩掉整次运行。
#   2. `--exportonnx` 在 `if agent_cfg.load_run is not None:` 里 → 必须给 `--load_run`。
#   3. ⭐ **`--exportonnx` 分支导出完会自己结束**（play.py 里依次
#      `env.close(); simulation_app.close(); return`） ⇒ **不需要靠 `--video` 让它退出**。
#      旧版本在这里先跑了一遍「16 环境 × N 步录像」，纯粹白占 GPU 时间和磁盘
#      （`--video` 的 break 只对下面的常规播放循环成立，对导出分支根本不成立）。
#      ⇒ **录像已移除**（用户不需要）。若确实要录像，见文件末尾的注释。
#
# 用法：
#   bash cloud_eval_policy.sh [run_dir] [checkpoint]
#   bash cloud_eval_policy.sh 20260925_203531_G1AmpWalkV3Plane_obs626_rew14_ar0.02_ns0.3_amp_walk_v3c_4096 model_1000.pt
#   （run_dir 留空则自动取 $LOGROOT 下**最新修改**的目录）
#
# 环境变量：
#   TASK=<play 任务 id>   默认 Instinct-Amp-Walk-G1-Play-v0
#
# 产物：
#   <run>/exported/*.onnx
#   <run>/exported/policy_normalizer.npz
# ============================================================================
set -uo pipefail

XU=/root/gpufree-data/instinct_xu
LOGROOT="$XU/logs/instinct_rl/g1_amp_walk"
RUN_ARG="${1:-}"
CKPT="${2:-model_5000.pt}"
#: 可覆盖任务 id（v2/v3 用 Instinct-Amp-Walk-V2/V3-G1-Play-v0）。
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
echo "=== 策略验证 run=$RUN ckpt=$CKPT task=$TASK ==="
ls -l "$RUNDIR/$CKPT" 2>/dev/null || { echo "❌ checkpoint 不存在: $RUNDIR/$CKPT"; exit 2; }

# 视频录像已移除：--exportonnx 导出后 play.py 自己 return，无需 --video 结束────────────────────────────────
# 导出 ONNX（单环境）────────────────────────────────────────────────────
echo
echo "---------- 导出 ONNX（单环境，不录像）----------"
./play.sh \
  --task "$TASK" \
  --load_run "$RUN" --checkpoint "$CKPT" \
  --num_envs 1 --device cuda:0 --headless \
  --exportonnx 2>&1 | tail -15
echo "导出退出码: ${PIPESTATUS[0]}"

# ── 产物 ───────────────────────────────────────────────────────────────────
echo
echo "======== 产物 ========"
find "$RUNDIR/exported" -maxdepth 1 -type f \( -name '*.onnx' -o -name '*.npz' \) 2>/dev/null \
  | while read -r f; do
      printf '  %s  (%s)\n' "$f" "$(du -h "$f" | cut -f1)"
    done
echo "======== 验证结束 ========"

# 如需录像（默认不做）：
#   cd $XU && ./play.sh --task "$TASK" --load_run "$RUN" --checkpoint "$CKPT" \
#     --num_envs 16 --device cuda:0 --headless --video --video_length 500
