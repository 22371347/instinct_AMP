#!/usr/bin/env bash
# ============================================================================
# 拉取云端「最新训练好的策略」的 ONNX 到本地，并做本地自检
#
# ⚠️ 为什么不能只 scp 两个文件了事 —— `actor.onnx` 与 `policy_normalizer.npz`
#    **必须严格成对**：ONNX 的输入是「归一化之后」的观测，归一化器配错 =
#    OOD 观测 → 标称站姿就输出疯狂动作、1.6 s 摔倒（见 CLOUD_ENV_SETUP.md §13.1 坑 1）。
#    而历史上 `instinct/eval/policy_normalizer.npz` 是**不带步骤号**的裸名字，
#    拉第二个模型就会**静默覆盖**第一个的归一化器 → 旧的 onnx 配上新的 normalizer，
#    现象是「同一份 onnx 突然跑不动了」，极难排查。
#    ⇒ 本脚本**每个模型放独立子目录**，靠目录天然配对：
#          instinct/eval/<tag>/actor.onnx
#          instinct/eval/<tag>/policy_normalizer.npz
#      `g1_mujoco_play.py` 的 `find_normalizer()` 就在 onnx 同目录找同名文件 → 自动配上 ✓
#
# 用法：
#   bash instinct/tools/pull_latest_onnx.sh                 # 自动取云端最新 checkpoint
#   bash instinct/tools/pull_latest_onnx.sh --list          # 只看训练进度 + 可用 checkpoint（不占 GPU）
#   bash instinct/tools/pull_latest_onnx.sh <run_dir> <ckpt>  # 指定 run 与 checkpoint
#   bash instinct/tools/pull_latest_onnx.sh --tag v3_1000    # 覆盖本地目录名
#
# 产物：
#   instinct/eval/<tag>/actor.onnx
#   instinct/eval/<tag>/policy_normalizer.npz
#
# ⚠️ 本脚本**不录云端视频**。`--exportonnx` 分支导出完会自己 `return`，play.py 随之退出 ——
#    录像从来不是必需的（旧版先跑 16 环境 × N 步录像，纯浪费 GPU 与磁盘）。
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1
ROOT="$PWD"
SSH=(python instinct/tools/instinct_ssh.py)

XU=/root/gpufree-data/instinct_xu
LOGROOT="$XU/logs/instinct_rl/g1_amp_walk"
LOCAL_EVAL="$ROOT/instinct/eval"

LIST_ONLY=0
TAG_OVERRIDE=""
RUN_ARG=""
CKPT_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) LIST_ONLY=1; shift ;;
    --tag)  TAG_OVERRIDE="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) if [[ -z "$RUN_ARG" ]]; then RUN_ARG="$1"; else CKPT_ARG="$1"; fi; shift ;;
  esac
done

# ── 0. 把导出脚本同步到云端（幂等；脚本改了就生效）───────────────────────────
"${SSH[@]}" --put instinct/tools/cloud_eval_policy.sh /root/cloud_eval_policy.sh >/dev/null 2>&1
"${SSH[@]}" --put instinct/tools/cloud_list_runs.sh  /root/cloud_list_runs.sh  >/dev/null 2>&1

# ⭐ 云端训练进度总览
#    回答「最新那次训练跑到第几轮、存了 checkpoint 没有」—— 光看 model_*.pt 会答错：
#    最新一版训练在过**第一个存盘点**之前，一个模型文件都没有，于是会被误认为
#    「没有在训练」，或者默默拉回上一个 run 的 ckpt（用户实际遇到的困惑）。
RAWPROG="$("${SSH[@]}" "bash /root/cloud_list_runs.sh 5" 2>/dev/null \
  | grep -vE 'credentials|password' | tr -d '\r')"
echo "════════ 云端训练进度（按日志修改时间倒序）════════"
printf '  %-24s %-12s %-13s %-10s %s\n' "日志 tag" "迭代" "日志更新" "ckpt×个" "run 目录（只显示尾部）"
if [[ -z "$RAWPROG" ]]; then
  echo "  （/root 下没有 train_amp_walk_*.log）"
else
  echo "$RAWPROG" | while IFS='|' read -r t it mt rd ck nck; do
    [[ -z "$t" ]] && continue
    if [[ "$t" == "=ALIVE=" ]]; then
      if [[ "${it:-0}" -gt 0 ]]; then
        echo "  ✅ 当前有训练在跑（train.py 进程数 = $it）"
      else
        echo "  ⛔ 当前没有 train.py 进程（云端空闲）"
      fi
      continue
    fi
    ckd="${ck:-<无>}"; ckd="${ckd#model_}"; ckd="${ckd%.pt}"
    rdd="$rd"; [[ ${#rdd} -gt 38 ]] && rdd="…${rdd: -38}"
    printf '  %-24s %-12s %-13s %-10s %s\n' "$t" "$it" "$mt" "${ckd}×${nck}" "$rdd"
  done
fi
echo

# ── 1. 列出云端「真正含 checkpoint」的 run（跳过空的续训目录）────────────────
#    在**本地**做格式化，避免嵌套引号把 awk 的 $1/$2 提前展开。
echo "════════ 云端可用 checkpoint（最新 10 个）════════"
RAW="$("${SSH[@]}" "find $LOGROOT -maxdepth 2 -name 'model_*.pt' -printf '%TY-%Tm-%Td %TH:%TM %p\\n' 2>/dev/null | sort" 2>/dev/null \
  | grep -vE 'credentials|password' | tr -d '\r')"
if [[ -z "$RAW" ]]; then
  echo "  （云端一个 model_*.pt 都没有）"
else
  echo "$RAW" | tail -10 | while read -r d t p; do
    [[ -z "$p" ]] && continue
    printf '  %s %s  %s/%s\n' "$d" "$t" "$(basename "$(dirname "$p")")" "$(basename "$p")"
  done
fi
echo

if [[ "$LIST_ONLY" == "1" ]]; then
  echo "（--list 模式：未占用 GPU。真正拉取请去掉 --list）"
  exit 0
fi

# ── 2. 选定 run + checkpoint ────────────────────────────────────────────────
if [[ -z "$RUN_ARG" ]]; then
  # 取**最晚修改的** model_*.pt（= 真正最新的模型），而不是"最新目录"
  LATEST="$("${SSH[@]}" "find $LOGROOT -maxdepth 2 -name 'model_*.pt' -printf '%T@ %h/%f\n' 2>/dev/null | sort -n | tail -1" 2>/dev/null \
    | grep -vE "credentials|password" | tr -d '\r' | tail -1)"
  [[ -z "$LATEST" ]] && { echo "❌ 云端找不到任何 model_*.pt"; exit 2; }
  RUN_ARG="$(basename "$(echo "$LATEST" | awk '{print $2}' | xargs dirname)")"
  CKPT_ARG="$(basename "$(echo "$LATEST" | awk '{print $2}')")"
fi
CKPT_ARG="${CKPT_ARG:-model_5000.pt}"
RUNDIR="$LOGROOT/$RUN_ARG"
STEP="${CKPT_ARG#model_}"; STEP="${STEP%.pt}"
echo "选定: run=$RUN_ARG  ckpt=$CKPT_ARG"

# ── 3. 本地标签 / 任务 ID ───────────────────────────────────────────────────
if [[ -n "$TAG_OVERRIDE" ]]; then
  TAG="$TAG_OVERRIDE"
elif [[ "$RUN_ARG" == *V3* ]]; then
  TAG="v3_${STEP}"
elif [[ "$RUN_ARG" == *V2* ]]; then
  TAG="v2_${STEP}"
else
  TAG="v1_${STEP}"
fi
# ⚠️ 任务 id 必须跟 run 的版本对上：V3 的指令分布/数据集与 v1/v2 不同。
#    用错任务不会崩（obs/动作契约相同），但 play 阶段的环境配置是错的。
if [[ "$RUN_ARG" == *V3* ]]; then
  PLAY_TASK=Instinct-Amp-Walk-V3-G1-Play-v0
elif [[ "$RUN_ARG" == *V2* ]]; then
  PLAY_TASK=Instinct-Amp-Walk-V2-G1-Play-v0
else
  PLAY_TASK=Instinct-Amp-Walk-G1-Play-v0
fi
echo "本地标签: $TAG   任务: $PLAY_TASK"

# ⭐ 若最新日志的 run 不是本次选的 run ⇒ 说明「最新那次训练还没存 checkpoint」
NEWEST_TAG="$(echo "$RAWPROG" | grep -v '^=ALIVE=' | head -1 | cut -d'|' -f1)"
if [[ -n "$NEWEST_TAG" && "$RUN_ARG" != *"$NEWEST_TAG"* ]]; then
  echo "⚠️  最新日志是「$NEWEST_TAG」，但它还没有 model_*.pt；本次拉的是更早的 run「$RUN_ARG」。"
fi

# ── 4. 云端导出（后台跑 + 轮询产物）─────────────────────────────────────────
# ⚠️ `--exportonnx` 只支持单环境；`play.py` 只有带 `--video` 才会自然退出。
#    这些约束都已写死在 cloud_eval_policy.sh 里（见它的头注释）。
echo
echo "──── 云端导出中（单环境，仅 ONNX，不录像；约 2~4 分钟）────"
# ⚠️ 用一个时间戳标记**排除历史 ONNX**：否则轮询会立刻看到上一次导出的旧文件而「成功」，
#    拉到本地就是「旧 onnx + 新 normalizer」错配 —— 正是本脚本头部警告的那类坑。
T0_MARK="/root/.eval_${TAG}.t0"
"${SSH[@]}" "touch $T0_MARK; bash /root/cloud_run_bg.sh 'TASK=$PLAY_TASK bash /root/cloud_eval_policy.sh $RUN_ARG $CKPT_ARG' /root/eval_${TAG}.log; echo launched" >/dev/null 2>&1

ONNX_REMOTE=""
for _ in $(seq 1 40); do
  sleep 15
  ONNX_REMOTE="$("${SSH[@]}" "find $RUNDIR/exported -maxdepth 1 -name '*.onnx' -newer $T0_MARK 2>/dev/null | head -1" 2>/dev/null \
    | grep -vE "credentials|password" | tr -d '\r' | tail -1)"
  [[ -n "$ONNX_REMOTE" ]] && break
  printf '.'
done
echo
if [[ -z "$ONNX_REMOTE" ]]; then
  echo "❌ 没等到 ONNX。云端日志尾部："
  "${SSH[@]}" "tail -12 /root/eval_${TAG}.log" 2>/dev/null | grep -vE "credentials|password" | tail -12
  exit 3
fi
echo "云端产物: $ONNX_REMOTE"

# ── 5. 拉回本地（每个模型独立子目录 → 天然配对）─────────────────────────────
DEST="$LOCAL_EVAL/$TAG"
mkdir -p "$DEST"
"${SSH[@]}" --get "$ONNX_REMOTE" "$DEST/actor.onnx" >/dev/null 2>&1
"${SSH[@]}" --get "$RUNDIR/exported/policy_normalizer.npz" "$DEST/policy_normalizer.npz" >/dev/null 2>&1

echo
echo "════════ 本地产物 $DEST ════════"
ls -l "$DEST" || exit 4
if [[ ! -f "$DEST/policy_normalizer.npz" ]]; then
  echo "❌ 缺 policy_normalizer.npz！**不要**直接用（会 OOD 摔倒）"
  exit 5
fi

# ── 6. 本地自检：IO 维度 + 跑 2 秒确认能加载能闭环 ──────────────────────────
source /home/leo/miniconda3/etc/profile.d/conda.sh && conda activate instinct
echo
echo "──── 本地自检 ────"
python instinct/tools/g1_mujoco_play.py --onnx-info --policy "$DEST/actor.onnx" 2>&1 | grep -E "输入|输出|input|output|\[onnx"
python instinct/tools/g1_mujoco_play.py --policy "$DEST/actor.onnx" \
  --no-view --no-teleop --duration 2 --cmd 0.5 0 0 --no-realtime 2>&1 \
  | grep -E "\[scene\]|\[obs\] 第 1 帧|\[pace\]|❌" | head -6

cat <<EOF

════════ 直接用 ════════
# 开窗 + WASD/QE 键盘遥控（会**自动配上**同目录的 policy_normalizer.npz）
python instinct/tools/g1_mujoco_play.py --policy $DEST/actor.onnx --duration 0

# 与别的版本对比时，各自指向自己的子目录即可（不会互相覆盖归一化器）：
#   --policy instinct/eval/v1_5000/actor.onnx
EOF
