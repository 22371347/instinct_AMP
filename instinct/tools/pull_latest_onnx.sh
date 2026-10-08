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
#   bash instinct/tools/pull_latest_onnx.sh v9c             # ⭐ 用「子串」定位 run（并行训练时必用）
#   bash instinct/tools/pull_latest_onnx.sh v9c model_4000.pt   # 再指定 checkpoint（默认取最新）
#   bash instinct/tools/pull_latest_onnx.sh <完整 run 目录名> <ckpt>   # 也支持直接写目录名
#   bash instinct/tools/pull_latest_onnx.sh --tag v3_1000    # 覆盖本地目录名
#
# ⭐ 并行训练（v9b / v9c …）时请**一定**带上 run 子串：不带参数时“最新”= 最后写盘的那份，
#    很容易把 v9c 的 checkpoint 当成 v9b 的（或反之）。命中多个时取最新写盘的那份，并打印候选。
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
#: ⭐ 日志根由 agent 的 `experiment_name` 决定，**不止一个**：
#:   AMP-Walk 系列 ⇒ logs/instinct_rl/g1_amp_walk
#:   OmniLoco 基座（T0）⇒ logs/instinct_rl/g1_unified_omni_loco
#: 所以 LOGROOT 不能写死：给了 run 子串就自动切到「含该 run 的那个根」；
#: 显式 `LLP_LOG_ROOT=...` 优先（与 plot_training.py 同一个环境变量口径）。
LLP_ROOT_BASE="$XU/logs/instinct_rl"
LOGROOT="$LLP_ROOT_BASE/g1_amp_walk"
LOCAL_EVAL="$ROOT/instinct/eval"

LIST_ONLY=0
TAG_OVERRIDE=""
RUN_ARG=""
CKPT_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) LIST_ONLY=1; shift ;;
    --tag)  TAG_OVERRIDE="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) if [[ -z "$RUN_ARG" ]]; then RUN_ARG="$1"; else CKPT_ARG="$1"; fi; shift ;;
  esac
done

# ── 0a. 解析日志根（必须在 RUN_ARG 之后才能按 run 子串找）────────────────────
_switch_logroot() {
  if [[ -n "${LLP_LOG_ROOT:-}" ]]; then LOGROOT="$LLP_LOG_ROOT"; return 0; fi
  [[ -z "$RUN_ARG" ]] && return 0
  local _r
  _r="$("${SSH[@]}" "for r in $LLP_ROOT_BASE/*/; do ls -1 \"\$r\" 2>/dev/null | grep -qF '$RUN_ARG' && { echo \"\${r%/}\"; exit 0; }; done" 2>/dev/null \
      | grep -vE 'credentials|password' | tr -d '\r' | tail -1)"
  [[ -n "$_r" ]] && LOGROOT="$_r"
  return 0
}
_switch_logroot

# ── 2b 的实现（提前定义：`--list` 也要用它显示“会选中哪一份”）──────────────
# `RUN_ARG` 允许只写「run 名字的子串」（v9c / amp_walk_v9b_mvp_4096）。
# 背景：V9b/V9c 是**并行**跑的，不带参数时的"最新 model_*.pt"= 最后写盘的那份，
# 非常容易拿错版本；给一个子串就能确定性地定位到那一份（命中多个取最新写盘的）。
resolve_pattern() {
  [[ -z "$RUN_ARG" || -d "$LOGROOT/$RUN_ARG" ]] && return 0
  local PAT="$RUN_ARG" RAW2 PICK NEW_RUN NEW_CKPT
  RAW2="$("${SSH[@]}" "for d in \$(ls -1 $LOGROOT 2>/dev/null); do case \"\$d\" in *$PAT*) p=\$(ls -1t $LOGROOT/\$d/model_*.pt 2>/dev/null | head -1); if [ -n \"\$p\" ]; then echo \"\$d|\$(basename \$p)|\$(stat -c %Y \"\$p\")\"; else echo \"\$d||0\"; fi;; esac; done" 2>/dev/null \
    | grep -vE "credentials|password" | tr -d '\r' | sed '/^$/d')"
  if [[ -z "$RAW2" ]]; then
    echo "❌ '$PAT' 没匹配到任何 run。云端现有（新 → 旧）："
    "${SSH[@]}" "ls -1t $LOGROOT | head -12" 2>/dev/null \
      | grep -vE "credentials|password" | sed 's/^/    /'
    return 2
  fi
  PICK="$(echo "$RAW2" | awk -F'|' '$3+0>m {m=$3+0; l=$0} END{print l}')"
  if [[ -z "$PICK" ]]; then
    # ⚠️ 命中项一个都没存 checkpoint（`$3` 全为 0 ⇒ awk 的 `$3>m` 永不成立）
    #    ⇒ 退回"目录名最新的那个"，并明确提示还没东西可导。
    PICK="$(echo "$RAW2" | tail -1)"
    echo "⚠️ 命中的 run 都还没有 model_*.pt（训练刚开始 / 未到存盘点）"
  fi
  NEW_RUN="$(echo "$PICK" | cut -d'|' -f1)"
  NEW_CKPT="$(echo "$PICK" | cut -d'|' -f2)"
  echo "按 '$PAT' 命中 $(echo "$RAW2" | grep -c .) 个 run，选中（最新写盘）：$NEW_RUN"
  echo "$RAW2" | awk -F'|' '{printf "    %-64s %s\n", $1, ($2=="" ? "(无 ckpt)" : $2)}' | head -6
  RUN_ARG="$NEW_RUN"
  [[ -z "$CKPT_ARG" && -n "$NEW_CKPT" ]] && CKPT_ARG="$NEW_CKPT"
  return 0
}

# ── 3a 的实现（提前定义：`--list` 也要能预告“会选哪个任务”）───────────────────
# 以**云端注册表**为准定版本号/后缀：取「run 名里真出现 `G1AmpWalk<X>`」的全部候选中
# **X 最长**的那个（`V33S` > `V33`）⇒ 不依赖任何“数字+单字母”的正则假设。
# 输入：全局 `RUN_ARG`；输出：全局 `VER` / `PLAY_TASK`（/ `RUN_MAIN`）。
resolve_play_task() {
  RUN_MAIN="${RUN_ARG%%_from*}"
  VER=""
  PLAY_TASK=""
  local _IDS _id _x
  _IDS="$("${SSH[@]}" "grep -oE '\"Instinct-Amp-Walk-[^\"]*-Play-v0\"' $XU/source/instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py | tr -d '\"'" 2>/dev/null \
    | grep -vE 'credentials|password' | tr -d '\r' | sed '/^$/d' | sort -u)"
  while read -r _id; do
    [[ -z "$_id" ]] && continue
    _x="${_id#Instinct-Amp-Walk-}"; _x="${_x%-G1-Play-v0}"
    [[ -z "$_x" ]] && continue
    if printf '%s' "$RUN_MAIN" | grep -qiF "G1AmpWalk${_x}"; then
      if [[ ${#_x} -gt ${#VER} ]]; then VER="$_x"; PLAY_TASK="$_id"; fi
    fi
  done <<< "$_IDS"
  # ⭐ 基座任务（T0 / OmniLoco 教师）：run 名里**不含** `G1AmpWalk<X>` 锚点，
  #   走另一套注册 id。放在 AMP 分支之后 ⇒ 不会误判 AMP 臂。
  if [[ -z "$PLAY_TASK" ]] && printf '%s' "$RUN_MAIN" | grep -qiE 'G1UnifiedOmniLoco|omni_loco|loco_base'; then
    VER="t0"
    PLAY_TASK="Instinct-Unified-OmniLoco-G1-Play-v0"
  fi
  if [[ -z "$VER" ]]; then
    # 兼容极老的 run（run 名里没有 G1AmpWalk 版本锚点）⇒ 退回正则
    VER="$(printf '%s' "$RUN_MAIN" | grep -oE 'V[0-9]+[a-z]?' | tail -1)"
    [[ -n "$VER" ]] && PLAY_TASK="Instinct-Amp-Walk-${VER}-G1-Play-v0"
  fi
  return 0
}

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
  if [[ -n "$RUN_ARG" ]]; then
    echo "── 按子串解析（假若真拉取，会选这份）──"
    resolve_pattern || exit $?
    # ⚠️ 顺便预告任务 id 与本地目录名 —— 否则用户要先等 2 分钟 Kit 启动才发现 id 不对
    #    （2026-10-06：`v33S` 被旧正则归成 V33 ⇒ 没注册 ⇒ exit 6）。
    resolve_play_task
    _ck="${CKPT_ARG:-（该 run 还没有 model_*.pt）}"
    _st="${_ck#model_}"; _st="${_st%.pt}"
    echo "    → 任务: ${PLAY_TASK:-（无法判定）}"
    echo "    → 本地目录: instinct/eval/${TAG_OVERRIDE:-v$(printf '%s' "${VER#V}" | tr 'A-Z' 'a-z')_${_st%.pt}}"
    echo "    → checkpoint: $_ck"
    echo
  fi
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

# ── 2b. `RUN_ARG` 允许只写「run 名字的子串」（v9c / amp_walk_v9b_mvp_4096）──────
resolve_pattern || exit $?
# ⚠️ **不要**在这里给 `model_5000.pt` 之类的默认值（2026-10-06 实测踩坑）：
#    解析不到 ckpt 时静默用默认名 ⇒ 云端根本沒这个文件（或拿到上一代的旧 ckpt）
#    ⇒ 现象就是「拉不下来 / 拉下来的是旧的」。宁可**明确报错**。
if [[ -z "$CKPT_ARG" ]]; then
  _EXIST="$("${SSH[@]}" "ls -1 $LOGROOT/$RUN_ARG/model_*.pt 2>/dev/null | sed 's#.*/##' | tr '\n' ' '" 2>/dev/null \
    | grep -vE 'credentials|password' | tr -d '\r' | sed '/^$/d')"
  echo "❌ 该 run 没有可用的 checkpoint（也没指定）。run=$RUN_ARG"
  echo "   目录里现有：${_EXIST:-（一个 model_*.pt 都没有）}"
  echo "   ⇒ 训练还没到存盘点（保存间隔 500 ⇒ 11500 续训的下一份是 model_12000.pt）。"
  echo "   可以在第二个参数里明确指定要导哪一份："
  echo "       bash instinct/tools/pull_latest_onnx.sh $RUN_ARG model_12000.pt"
  exit 4
fi
RUNDIR="$LOGROOT/$RUN_ARG"
STEP="${CKPT_ARG#model_}"; STEP="${STEP%.pt}"
echo "选定: run=$RUN_ARG  ckpt=$CKPT_ARG"

# ── 3. 本地标签 / 任务 ID ───────────────────────────────────────────────────
# ⚠️ 任务 id 必须跟 run 的版本对上，**而且这件事在 v9 起变成了硬约束**：
#    v9 的 actor 观测是 **378**（279 高程图 + 前移 1.5 m），v1~v8 是 **626**（527 高程图），
#    观测维度不同 ⇒ 用错任务在云上会**直接契约校验失败退出**（v1~v3 之间只是配置不同，
#    不会崩，所以旧脚本只特判到 V3，这里是它漏掉 V4~V9 的根因）。
#    规则：run 名里出现 `V<N>[小写后缀]?` ⇒ 任务 = `Instinct-Amp-Walk-V<N>[后缀]-G1-Play-v0`、
#          标签 = `v<N>[后缀]_<step>`；没有版本号（v1 时代的 run 名）⇒ 基础任务 + `v1_<step>`。
#    ⚠️ 后缀要带上：`V9b`（消融：v9 − 镜像损失）与 `V9` 是**两个任务**，只匹 `V[0-9]+` 会归错。
#
# ⚠️⚠️ 还必须**先砍掉 `_from<旧版本>` 尾巴**（2026-09-30 实测踩坑）：
#     V12 的 run 名形如
#       `20260930_114005_G1AmpWalkV12MVPPlane_..._fromV11_amp_walk_v12_4096_from20260930_093037`
#     直接 `grep -oE 'V[0-9]+[a-z]?' | tail -1` 会**匹配到 `fromV11` 里的 V11**
#     ⇒ 用 V11 的 Play 任务（AMP 观测 670 维）去载 V12 的 ckpt（134 维）⇒ 云上
#       `RuntimeError: size mismatch for model.model.0.weight: [1024,134] vs [1024,670]`
#     而本地脚本还在那儿轮询 10 分钟 ⇒ 用户看到的现象就是「导出无响应」。
#     规则：先 `%%_from*` 截断，再**以云端注册表为准**定版本号/后缀（见下）。
#
# ⚠️⚠️ **不要再用正则猜版本后缀**（2026-10-06 实测踩坑）：
#     V33 的实验臂 run 名形如 `G1AmpWalkV33SMVPPlane_...` / `G1AmpWalkV33S1MVPPlane_...` /
#     `G1AmpWalkV33AColdMVPPlane_...`，而旧写法
#         grep -oE 'G1AmpWalkV[0-9]+[0-9]*'      # ⇒ 只能取到 V33（S/S1/ACold 被吃掉）
#     ⇒ 去找没注册的 `Instinct-Amp-Walk-V33-G1-Play-v0` ⇒ 直接 `exit 6`
#     （用户现象：`pull_latest_onnx.sh v33S` / `v33` 都拉不下来）。
#     新写法：把云端注册表里的 Play 任务全取回来，挑出「run 名里**真的出现** `G1AmpWalk<X>`」
#     的那些，取 **X 最长**的一个：
#        · `V33S` 比 `V33` 长 ⇒ 选中 V33S ✅（前缀命中的歧义由「更长者胜」消解）
#        · `..._fromV11_...` 里的 V11 **没有** `G1AmpWalk` 前缀 ⇒ 天然排除 ✅（旧坑一并解决）
#        · 不限于「数字+单个字母」⇒ 以后 `S0/S2/ACold/B/C` 之类后缀全都能自动对上 ✅
RUN_MAIN="${RUN_ARG%%_from*}"
resolve_play_task   # 见顶部 `resolve_play_task()`（以云端注册表为准，取最长匹配）
if [[ -n "$VER" ]]; then
  TAG="${TAG_OVERRIDE:-v$(printf '%s' "${VER#V}" | tr 'A-Z' 'a-z')_${STEP}}"
else
  TAG="${TAG_OVERRIDE:-v1_${STEP}}"
  PLAY_TASK="Instinct-Amp-Walk-G1-Play-v0"
fi
echo "本地标签: $TAG   任务: $PLAY_TASK"

# ⭐ 预检：这个 Play 任务**真的注册过**吗？版本号判错就别白等 Kit 启动 2 分钟。
REG_SRC="$XU/source/instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py"
task_registered() {
  "${SSH[@]}" "grep -q '\"$1\"' $REG_SRC && echo __FOUND__" 2>/dev/null | grep -q '__FOUND__'
}
if ! task_registered "$PLAY_TASK"; then
  # ⭐ 大小写兜底（2026-10-02 实测踩坑）：字母后缀的历史约定是**小写**（`V9b` / `V10b`），
  #    但注册时很容易手写成大写（如 `V19A`），而上面的正则只从 run 名里取小写后缀
  #    ⇒ id 对不上。这里按**大小写无关**在云端注册表里找**唯一**候选，
  #    找到就直接改用云端真实 id（而不是白等 2 分钟 Kit 启动再报 size mismatch）。
  ALT="$("${SSH[@]}" "grep -oiE '\"Instinct-Amp-Walk-[^\"]*-Play-v0\"' $REG_SRC | tr -d '\"'" 2>/dev/null \
    | grep -vE 'credentials|password' | tr -d '\r' \
    | grep -ixE "Instinct-Amp-Walk-${VER}-G1-Play-v0" | head -1)"
  if [[ -n "$ALT" && "$ALT" != "$PLAY_TASK" ]]; then
    echo "⚠️ 期望 id '$PLAY_TASK' 未注册，但云端有大小写不同的 '$ALT' ⇒ 自动改用云端真实 id。"
    echo "   （根治办法：把注册表里那个 id 也改成小写后缀，与 V9b/V10b 的惯例保持一致。）"
    PLAY_TASK="$ALT"
  fi
fi
if ! task_registered "$PLAY_TASK"; then
  echo "❌ 云端没有注册任务 $PLAY_TASK（版本号判错？）—— 已注册的 Play 任务（尾部 8 个）："
  "${SSH[@]}" "grep -oE '\"Instinct-Amp-Walk-[^\"]*-Play-v0\"' $REG_SRC | tail -8" 2>/dev/null \
    | grep -vE 'credentials|password' | sed 's/^/    /'
  exit 6
fi

# ⭐ 若最新日志的 run 不是本次选的 run ⇒ 说明「最新那次训练还没存 checkpoint」
# ⚠️ 只比**版本号**（`v19a` 这种），不比整串 tag：用户给 `--run_name` 的 tag 常与
#    日志文件名不同（如 run 目录里是 `..._v19a_warmV18_4096_...`，日志却是
#    `train_amp_walk_v19a_4096.log`）⇒ 直接比整串会把**同一个 run** 误报成"更早的 run"。
ver_of() { printf '%s' "$1" | grep -oiE 'v[0-9]+[a-z]?' | head -1 | tr 'A-Z' 'a-z'; }
NEWEST_TAG="$(echo "$RAWPROG" | grep -v '^=ALIVE=' | head -1 | cut -d'|' -f1)"
if [[ -n "$NEWEST_TAG" && "$(ver_of "$NEWEST_TAG")" != "$(ver_of "$RUN_ARG")" ]]; then
  echo "⚠️  最新日志是「$NEWEST_TAG」，与本次的 run 不是同一版本；若它还没有 model_*.pt，"
  echo "    说明「最新那次训练还没到存盘点」，本次拉的是「$RUN_ARG」。"
fi

# ── 4. 云端导出（后台跑 + 轮询产物）─────────────────────────────────────────
# ⚠️ `--exportonnx` 只支持单环境；`play.py` 只有带 `--video` 才会自然退出。
#    这些约束都已写死在 cloud_eval_policy.sh 里（见它的头注释）。
echo
echo "──── 云端导出中（单环境，仅 ONNX，不录像；约 2~4 分钟）────"
# ⚠️ 用一个时间戳标记**排除历史 ONNX**：否则轮询会立刻看到上一次导出的旧文件而「成功」，
#    拉到本地就是「旧 onnx + 新 normalizer」错配 —— 正是本脚本头部警告的那类坑。
T0_MARK="/root/.eval_${TAG}.t0"
"${SSH[@]}" "touch $T0_MARK; bash /root/cloud_run_bg.sh 'TASK=$PLAY_TASK bash /root/cloud_eval_policy.sh $RUNDIR $CKPT_ARG' /root/eval_${TAG}.log; echo launched" >/dev/null 2>&1

# ⭐ 轮询：**同一次 SSH 里既看产物、也看云上是否已经报错**。
#    早失败很重要：旧版只空等 10 分钟什么都不说，现象就是「导出无响应」。
ONNX_REMOTE=""
ERR_KIND=""
for i in $(seq 1 40); do
  sleep 15
  POLL="$("${SSH[@]}" "f=\$(find $RUNDIR/exported -maxdepth 1 -name '*.onnx' -newer $T0_MARK 2>/dev/null | head -1); e=\$(grep -m1 -aoE 'RuntimeError|Traceback|AssertionError|Error executing job' /root/eval_${TAG}.log 2>/dev/null); echo \"ONNX=\$f\"; echo \"ERR=\$e\"" 2>/dev/null \
    | grep -vE "credentials|password" | tr -d '\r')"
  ONNX_REMOTE="$(printf '%s\n' "$POLL" | sed -n 's/^ONNX=//p' | tail -1)"
  ERR_KIND="$(printf '%s\n' "$POLL" | sed -n 's/^ERR=//p' | tail -1)"
  [[ -n "$ONNX_REMOTE" ]] && break
  if [[ -n "$ERR_KIND" ]]; then
    echo
    echo "❌ 云端导出已报错（第 $i 次轮询发现：$ERR_KIND）—— 提前退出，不再空等。日志尾部："
    "${SSH[@]}" "tail -22 /root/eval_${TAG}.log" 2>/dev/null | grep -vE "credentials|password" | tail -22
    exit 3
  fi
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
  | grep -E "\[obs\] obs_mode|\[obs\] 第 1 帧|\[pace\]|❌" | head -8

cat <<EOF

════════ 直接用 ════════
# 开窗 + WASD/QE 键盘遥控（会**自动配上**同目录的 policy_normalizer.npz）
python instinct/tools/g1_mujoco_play.py --policy $DEST/actor.onnx --duration 0

# 与别的版本对比时，各自指向自己的子目录即可（不会互相覆盖归一化器）：
#   --policy instinct/eval/v1_5000/actor.onnx
#
# ⭐ 观测布局按 ONNX 输入维度**自动选**：626 → amp626（527 射线、居中）；
#    378 → amp378（279 射线、中心前移 1.5 m）。叠加的橙色扫描框会跟着切。
#    想强制指定：--obs-mode amp626 | amp378 | legacy
EOF
