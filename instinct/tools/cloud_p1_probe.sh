#!/usr/bin/env bash
# ============================================================================
# P1 初速探针（一键验收）：判定策略的固定点里**有没有"前进"这个解**
# ============================================================================
# 背景：`cmd=+0.8` 时策略实际只跑 ~0.01 m/s（原地踏步）。要区分两件事：
#   (a) 固定点里**没有**前进解（一给速度就刹停） ⇒ 奖励/风格梯度冲突
#   (b) 固定点**存在**，但策略学不会从静止加速上去 ⇒ 自举/探索问题
# 办法：复位后直接注入初速 v0，看它是"维持"还是"刹车"。
#
# 判读（预注册）：
#   v0=0.8 稳态 ≥0.5·cmd        ⇒ ②维持：有前进解（缺自举机制）
#   v0=0.8 稳态 ≤0.2·cmd        ⇒ ①刹停：固定点里没有前进解（梯度冲突）
#   中间                        ⇒ ③部分维持：高速步态不稳定
#
# 用法（本机）：
#   bash instinct/tools/cloud_p1_probe.sh instinct/eval/v33S8_14000        # 默认 cmd=0.8
#   bash instinct/tools/cloud_p1_probe.sh instinct/eval/v33S8_14000 0.5 10
#     $1 policy 目录（内含 actor.onnx + policy_normalizer.npz）
#     $2 指令 vx（默认 0.8）   $3 时长秒（默认 10）   $4 跳过瞬态秒（默认 3）
# ⚠️ 依赖 conda 环境 `instinct`（mujoco）。这里只做 headless 数值探针；
#    带窗口的观感回放由用户自己跑。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

POLICY_DIR="${1:?用法: cloud_p1_probe.sh <policy_dir> [cmd_vx] [dur] [skip]}"
CMD="${2:-0.8}"
DUR="${3:-10}"
SKIP="${4:-3}"
ONNX="$POLICY_DIR/actor.onnx"
[[ -f "$ONNX" ]] || { echo "❌ 找不到 $ONNX" >&2; exit 1; }

source /home/leo/miniconda3/etc/profile.d/conda.sh && conda activate instinct

echo "=================================================================="
echo "P1 初速探针  policy=$POLICY_DIR  cmd=+$CMD  dur=${DUR}s  skip=${SKIP}s"
echo "=================================================================="
printf '%-6s %-11s %-12s %-9s %-8s %s\n' "v0" "稳态vx" "净位移Δx" "跟踪效率" "末高度" "判定"

for V in 0.0 0.3 0.6 0.8 1.0; do
  OUT=$(python instinct/tools/g1_mujoco_play.py --policy "$ONNX" \
        --no-view --no-teleop --no-realtime \
        --duration "$DUR" --track-skip "$SKIP" --init-vx "$V" --cmd "$CMD" 0 0 2>&1)

  VX=$(echo "$OUT"  | grep -oE '实际均值 vx=[-+0-9.]+' | tail -1 | sed 's/.*vx=//')
  DX=$(echo "$OUT"  | grep -oE '净位移   Δx=[-+0-9.]+'  | tail -1 | sed 's/.*Δx=//')
  EFF=$(echo "$OUT" | grep -oE '跟踪效率 = [-0-9]+%'    | tail -1 | sed 's/.*= //')
  H=$(echo "$OUT"   | grep -oE 'height=[0-9.]+'          | tail -1 | sed 's/height=//')
  : "${VX:=}" "${DX:=}" "${EFF:=—}" "${H:=}"

  # ⚠️  回放没出 [track] 汇总 = 它崩了。**绝不能把这种情况当成数据**（会把 nan
  #    误判成“维持”）。首次出现就直接把错误尾部打出来并退出。
  if [[ -z "$VX" ]]; then
    VX="nan"; DX="nan"; EFF="—"; H="nan"
    printf '%-6s %-11s %-12s %-9s %-8s %s\n' "$V" "nan" "nan" "—" "nan" "?? 回放失败"
    if [[ ! -f /tmp/.p1_failed_mark ]]; then
      touch /tmp/.p1_failed_mark
      echo
      echo "======== 回放崩溃尾部（请先修这个，不要看表）========"
      printf '%s\n' "$OUT" | tail -20
    fi
    continue
  fi

  # 判定：只在注入了初速的行上判；v0=0 是自然起步的对照
  VERDICT="—(对照)"
  if awk "BEGIN{exit !($V > 0.01)}"; then
    VERDICT=$(awk -v vx="$VX" -v c="$CMD" 'BEGIN{
      if (vx+0 >= 0.5*c)      print "②维持";
      else if (vx+0 <= 0.2*c) print "①刹停";
      else                    print "③部分维持";
    }')
  fi
  printf '%-6s %-11s %-12s %-9s %-8s %s\n' "$V" "$VX" "$DX" "$EFF" "$H" "$VERDICT"
done
rm -f /tmp/.p1_failed_mark

cat <<EOF

判读规则（预注册）：
  · v0=0.8 出现"②维持" ⇒ 固定点里有前进解 ⇒ 缺的是**自举机制**
  · v0=0.8 出现"①刹停" ⇒ 固定点里没有前进解 ⇒ 是**梯度冲突**（奖励/风格）
  · "③部分维持"        ⇒ 高速步态不稳定，需先做稳定化
附：末高度 <0.3 说明摔了，该行结论不作数。
EOF
