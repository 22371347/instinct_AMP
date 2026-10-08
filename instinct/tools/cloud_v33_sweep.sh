#!/usr/bin/env bash
# ============================================================================
# V33 **三臂串行 sweep**：S0（结构包）→ S1（结构包+拆脚手架）→ A-cold（旧口径参照系）
# ============================================================================
# 三臂共同点：同一 fork（V31 `model_11500.pt`）、同预算 1000 iters、同 seed、
#             **丢判别器从零重学**（口径变了：白化 / 输入维度）、`flat_orientation_l2=-4`、
#             `style_lerp=0.75`、`discriminator_reward_coef=1.0`、GP=10(expert-only)、
#             `bounded_quad`、σ=0.20+退火、`side_vel_min=0.15`。
# 到达 **12500** 轮（= 11500 + 1000，500 的倍数 ⇒ 可导出 ONNX）后杀掉训练与守护、
# 打印该臂摘要，再进下一臂。
#
# ⚠️ 起 S 臂前会**自动导出 396 维专家数据集**（`--with_anchor_gravity`，约 2–4 分钟，
#    需要短暂独占 GPU ⇒ 放在击杀之后、起训之前）。已存在则跳过。
#
# 每臂起训后**先做有效性自检**（控制变量的前提），不过就**跳过该臂**并打 `!!!`：
#   ① `params/env.yaml` 里 `side_vel_min: 0.15`（三臂共有；这是本次"侧向档不再与静止门冲突"的落地证据）
#   ② 判别器输入维（从日志的 `Discriminator Network Structure` 读）：S 臂 396 / ACOLD 390
#   ③ 白化的**效果级**证据：S 臂日志出现 `EmpiricalNormalization`，ACOLD 不出现
#   ④ `params/agent.yaml` 的 `amp_grad_pen_space`：S 臂 `normalized` / ACOLD `raw`
#   ⑤ 奖励权重横幅：S1 必须 `feet_air_time=2` 且 `joint_deviation_arms=0`；S0 必须是 4 / −0.1
#   ⑥ 三臂**都必须**出现 `drop_amp_discriminator`（判别器口径变了 ⇒ 必须重学；与 V32 相反）
#   ⑦ 专家批路径：S 臂 `/root/amp_expert_states_396.pt` / ACOLD `/root/amp_expert_states.pt`
# ⚠️ 教训（2026-10-06，V32）：**判据要认「有效果」，不要认「某个字符串」**。
#    那次的「必须看到 reset 白化 buffer」自检把两臂全跳过了，因为白化压根没接上。
#    所以这里每条都尽量落到"运行期生效的产物"（yaml 值 / 网络输入维 / 出现的模块名）上。
#
# 用法（云端）：
#   bash /root/cloud_run_bg.sh "bash /root/cloud_v33_sweep.sh" /root/v33_sweep.log
#   # 只跑部分臂：bash /root/cloud_v33_sweep.sh "S0 S1"     （$1 = 臂列表，默认 "S0 S1 ACOLD"）
#   # 跳过导出：  SKIP_EXPORT=1 bash /root/cloud_v33_sweep.sh
# ============================================================================
set -uo pipefail

ARMS="${1:-S0 S1 ACOLD S2}"
FORK="${2:-11500}"
BUDGET="${3:-1000}"
TARGET=$((FORK + BUDGET))
SKIP_EXPORT="${SKIP_EXPORT:-0}"

RUN_ROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk
EXPERT_396=/root/amp_expert_states_396.pt
WATCH='Mean reward:|Mean episode length|discriminator_loss|Mean step amp_d:|Mean step amp_d_ref|Mean step amp_style_reward|Mean step amp_style_std|Mean step amp_style_zero_frac|Mean step tilt_deg_mean|Mean step tilt_p95_deg|Mean step tilt_gt30_frac|Mean step grav_bx_mean|Mean step air_time_single_stance|Mean step air_time_hop_frac|Mean step air_time_in_band_frac|Mean step action_abs_mean|Mean step lin_track_fx:|Mean step lin_track_fy:|Mean step lin_track_side_joint:|Mean step lin_track_side_fy:|Mean step lin_track_side_ey_abs:|Mean step lin_track_lin_joint:|disc_grad/|Episode_Reward/feet_air_time|Episode_Reward/flat_orientation_l2|Episode_Reward/joint_deviation_arms'

log() { echo "[$(date '+%F %T')] $*"; }
FAILED_ARMS=""

log "V33 sweep：臂=[$ARMS]  fork=$FORK  预算=$BUDGET  目标轮次=$TARGET"

# ---------------------------------------------------------------------------
# 0) 配置预检（秒级，不建 sim）：三臂的 env/agent cfg 全部断言一遍
#    ⚠️ 祖先链的 `_patch_amp_history` 有"覆盖 AMP 观测组"的前科 ⇒ 必须在 cfg 层就炸。
# ---------------------------------------------------------------------------
log "==== [0] cfg 预检（cloud_verify_v33_cfg.py）===="
pkill -9 -f 'train[.]py' || true
pkill -9 -f 'v29_guard' || true
sleep 3
V33_VERIFY_OUT=/root/v33_verify_result.txt \
  bash -c "cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && source /root/.llp_env.sh && unset PYTHONPATH && \
           /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v33_cfg.py" || true
if grep -q '\[V33-VERIFY PASS\]' /root/v33_verify_result.txt 2>/dev/null; then
  log "  ✅ cfg 预检通过"
  grep -E '^\[A\]|^\[B\]|^\[C\]|^\[D\]|结果：' /root/v33_verify_result.txt | tail -12
else
  log "  ❌ cfg 预检失败 ⇒ **不启动任何臂**（避免白烧 GPU）"
  tail -40 /root/v33_verify_result.txt 2>/dev/null
  exit 1
fi

# ---------------------------------------------------------------------------
# 0.5) 396 维专家数据集（S 臂的前置条件）
# ---------------------------------------------------------------------------
needs_export=0
for ARM in $ARMS; do
  [[ "${ARM,,}" == "acold" ]] || needs_export=1
done

if [[ "$needs_export" == "1" && "$SKIP_EXPORT" != "1" && ! -f "$EXPERT_396" ]]; then
  log "==== 导出 396 维专家数据集（含 anchor 投影重力）===="
  pkill -9 -f 'train[.]py' || true
  pkill -9 -f 'v29_guard' || true
  sleep 5
  # ⚠️ 导出脚本会把 stdout **dup2 到 `AMP_EXPORT_OUT`**（不是 cloud_run_bg 的日志）
  #    ⇒ 必须轮询这个结果文件，否则会永远等不到 `[AMP-EXPORT PASS]`（2026-10-06 踩过：
  #    sweep 白白空转 25 分钟，差点把 S0/S1 全跳过）。
  EXPORT_RESULT=/root/amp_export_result_396.txt
  rm -f "$EXPORT_RESULT"
  bash /root/cloud_run_bg.sh \
    "cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && source /root/.llp_env.sh && unset PYTHONPATH && \
     AMP_EXPORT_CKPT=$EXPERT_396 AMP_EXPORT_STATS=/root/amp_expert_stats_396.txt AMP_EXPORT_OUT=$EXPORT_RESULT \
     /opt/conda/envs/isaaclab/bin/python -u /root/cloud_export_amp_expert.py \
       --task Instinct-Amp-Walk-V33S-G1-v0 --num_envs 256 --live_steps 30 --with_anchor_gravity" \
    /root/v33_export_396.log
  log "  导出已挂后台（结果文件 $EXPORT_RESULT），等待完成（最多 25 分钟）…"
  ok=0
  for _ in $(seq 1 150); do
    if grep -q '\[AMP-EXPORT PASS\]' "$EXPORT_RESULT" 2>/dev/null; then ok=1; break; fi
    if grep -q '\[AMP-EXPORT FAIL\]' "$EXPORT_RESULT" 2>/dev/null; then break; fi
    # 进程没了但还没写 RC ⇒ 也算结束（避免死等）
    if ! pgrep -f 'cloud_export_amp_expert' >/dev/null 2>&1 && grep -q 'RC=' "$EXPORT_RESULT" 2>/dev/null; then break; fi
    sleep 10
  done
  if [[ "$ok" == "1" && -f "$EXPERT_396" ]]; then
    log "  ✅ 396 专家集已就绪（$(du -h "$EXPERT_396" | cut -f1)）"
    grep -E '新通道口径自检|对状态 \(N,|anchor 投影重力通道|σ 放大倍数中位' "$EXPORT_RESULT" | tail -8
  else
    log "  ❌ 396 导出失败 ⇒ **跳过 S0/S1**（只跑 A-cold 没意义，直接退出）"
    log "    尾部结果："; tail -25 "$EXPORT_RESULT" 2>/dev/null
    exit 1
  fi
elif [[ "$needs_export" == "1" ]]; then
  log "  396 专家集已存在 ⇒ 跳过导出（$(du -h "$EXPERT_396" 2>/dev/null | cut -f1)）"
fi

# ---------------------------------------------------------------------------
# 1) 逐臂串行
# ---------------------------------------------------------------------------
for ARM in $ARMS; do
  #: 默认：判别器口径变了 ⇒ 必须丢弃重学；**S4 例外**（显式写 no，故意保留权重）
  EXPECT_DROP_DISC=yes
  case "${ARM,,}" in
    s0)    EXPECT_DIM=396 ; EXPECT_TAG="amp_walk_v33S0_struct_from${FORK}"        ; EXPECT_NORM=yes ; EXPECT_GP=normalized ; EXPECT_EXPERT="$EXPERT_396" ; EXPECT_EPS=0.1  ; EXPECT_AIR=4   ; EXPECT_ARMS=-0.1 ; EXPECT_TRACK=3 ; EXPECT_DROP_DISC=yes ;;
    s1)    EXPECT_DIM=396 ; EXPECT_TAG="amp_walk_v33S1_noscaffold_from${FORK}"   ; EXPECT_NORM=yes ; EXPECT_GP=normalized ; EXPECT_EXPERT="$EXPERT_396" ; EXPECT_EPS=0.1  ; EXPECT_AIR=2   ; EXPECT_ARMS=0    ; EXPECT_TRACK=3 ; EXPECT_DROP_DISC=yes ;;
    s2)    EXPECT_DIM=396 ; EXPECT_TAG="amp_walk_v33S2_white001_from${FORK}"     ; EXPECT_NORM=yes ; EXPECT_GP=normalized ; EXPECT_EXPERT="$EXPERT_396" ; EXPECT_EPS=0.01 ; EXPECT_AIR=4   ; EXPECT_ARMS=-0.1 ; EXPECT_TRACK=3 ; EXPECT_DROP_DISC=yes ;;
    s3)    EXPECT_DIM=396 ; EXPECT_TAG="amp_walk_v33S3_nowhite_weakened_from${FORK}" ; EXPECT_NORM=no ; EXPECT_GP=raw ; EXPECT_EXPERT="$EXPERT_396" ; EXPECT_EPS="" ; EXPECT_AIR=2 ; EXPECT_ARMS=0 ; EXPECT_TRACK=2 ; EXPECT_DROP_DISC=yes ;;
    s4)    # ⭐ 续训臂：从 S3 的 12500 继续（显式指定续训目录）；奖励={} 见 G1AmpWalkV33S4EnvCfg 的 docstring
           # ⭐ 判别器**故意保留**（ckpt_manipulator=None；396 布局/专家集与 S3 相同）⇒ EXPECT_DROP_DISC=no
           EXPECT_DIM=396 ; EXPECT_TAG="amp_walk_v33S4_noairtime_boostangvel_from${FORK}" ; EXPECT_NORM=no ; EXPECT_GP=raw ; EXPECT_EXPERT="$EXPERT_396" ; EXPECT_EPS="" ; EXPECT_AIR=0 ; EXPECT_ARMS=0 ; EXPECT_TRACK=3 ; EXPECT_DROP_DISC=no ; EXPECT_LOADRUN="$(ls -td "$RUN_ROOT"/*v33S3* 2>/dev/null | head -1)" ;;
    s5)    # ⭐ 续训臂：从 S3 的 12500 继续；air 2→1（折半）+ 保住跟踪权重增强；**保留判别器**
           EXPECT_DIM=396 ; EXPECT_TAG="amp_walk_v33S5_air1_boosttrack_from${FORK}" ; EXPECT_NORM=no ; EXPECT_GP=raw ; EXPECT_EXPERT="$EXPERT_396" ; EXPECT_EPS="" ; EXPECT_AIR=1 ; EXPECT_ARMS=0 ; EXPECT_TRACK=3 ; EXPECT_DROP_DISC=no ; EXPECT_LOADRUN="$(ls -td "$RUN_ROOT"/*v33S3* 2>/dev/null | head -1)" ;;
    s6)    # ⭐ 续训臂：从 S3 的 12500 继续；**保持 S4 其余不变，只把 coef 1.0→4.0**；保留判别器
           #      （与 S4 完全同起点 ⇒ 唯一变量 = coef；见 amp_walk_v33_env_cfg.py::G1AmpWalkV33S6EnvCfg）
           EXPECT_DIM=396 ; EXPECT_TAG="amp_walk_v33S6_ampcoef4_noairtime_from${FORK}" ; EXPECT_NORM=no ; EXPECT_GP=raw ; EXPECT_EXPERT="$EXPERT_396" ; EXPECT_EPS="" ; EXPECT_AIR=0 ; EXPECT_ARMS=0 ; EXPECT_TRACK=3 ; EXPECT_DROP_DISC=no ; EXPECT_LOADRUN="$(ls -td "$RUN_ROOT"/*v33S3* 2>/dev/null | head -1)" ;;
    s7)    # ⭐ 判别器口径修复臂：只恢复 mjlab 的判别器正则（GP 10→5、wd 3e-4、logit_wd 0.04、trunk=head=3e-4）；
           #      权重全不动（coef=1.0 / λ=0.75 / 奖励表同 S5，air=1）；判别器**必顶重学** ⇒ EXPECT_DROP_DISC=yes
           EXPECT_DIM=396 ; EXPECT_TAG="amp_walk_v33S7_mjlabdisc_from${FORK}" ; EXPECT_NORM=no ; EXPECT_GP=raw ; EXPECT_EXPERT="$EXPERT_396" ; EXPECT_EPS="" ; EXPECT_AIR=1 ; EXPECT_ARMS=0 ; EXPECT_TRACK=3 ; EXPECT_DROP_DISC=yes ; EXPECT_LOADRUN="$(ls -td "$RUN_ROOT"/*v33S5* 2>/dev/null | head -1)" ;;
    s8)    # ⭐ S8-a：**抬风格分量（coef=4）+ 自适应 LR（adaptive）配对**；判别器保留（AMP 口径未变）
           #      fork = **S5@13500**（用户判定 S7 风格恶化 ⇒ 不从 S7 续）；奖励表同 S5（air=1）
           EXPECT_DIM=396 ; EXPECT_TAG="amp_walk_v33S8_coef4_adaptive_from${FORK}" ; EXPECT_NORM=no ; EXPECT_GP=raw ; EXPECT_EXPERT="$EXPERT_396" ; EXPECT_EPS="" ; EXPECT_AIR=1 ; EXPECT_ARMS=0 ; EXPECT_TRACK=3 ; EXPECT_DROP_DISC=no ; EXPECT_LOADRUN="$(ls -td "$RUN_ROOT"/*v33S5* 2>/dev/null | head -1)" ;;
    acold) EXPECT_DIM=390 ; EXPECT_TAG="amp_walk_v33ACold_legacy_from${FORK}"    ; EXPECT_NORM=no  ; EXPECT_GP=raw        ; EXPECT_EXPERT=/root/amp_expert_states.pt ; EXPECT_EPS="" ; EXPECT_AIR=4 ; EXPECT_ARMS=-0.1 ; EXPECT_TRACK=3 ; EXPECT_DROP_DISC=yes ;;
    *) log "!!! 未知臂名 $ARM ⇒ 跳过"; continue ;;
  esac
  LOG="/root/train_${EXPECT_TAG}.log"

  log "================ ${ARM} 臂启动（目标 ${TARGET} 轮，tag=${EXPECT_TAG}）================"
  if ! bash /root/cloud_launch_v33.sh "$ARM" "$FORK" "$BUDGET" "${EXPECT_LOADRUN:-}"; then
    log "!!! ${ARM} 臂启动脚本失败 ⇒ 跳过"
    FAILED_ARMS="$FAILED_ARMS $ARM(启动失败)"
    continue
  fi

  # ---- 等日志与权重横幅（最多 12 分钟）----
  for _ in $(seq 1 72); do
    [[ -s "$LOG" ]] && grep -q 'AMP-WALK WEIGHTS' "$LOG" 2>/dev/null && break
    sleep 10
  done

  RUN_DIR=""; ENV_YAML=""; AGENT_YAML=""
  # ---- ⭐ 就绪等待（⚠️ 2026-10-06 **第二次**踩坑）：
  #   横幅 `AMP-WALK WEIGHTS` 出现时，run 目录 / `params/agent.yaml` **尚未落盘**，
  #   而 `drop_amp_discriminator` 更晚（要等 ckpt 加载才打印）⇒ 此时判 ①④⑥ 会把
  #   **健康的臂**当成无效臂 `pkill` 掉（S7 就这样被误杀：21:34:12 判 BAD=1 ⇒ 21:34:14 RC=137）。
  #   ⇒ 这里再等最多 12 分钟，等到「run 目录 + agent.yaml 就绪」且（若该臂要求丢判别器）
  #     「日志里出现 drop_amp_discriminator」再开始判据；进程若已死则立即跳出。
  for _ in $(seq 1 72); do
    RUN_DIR="$(ls -td "$RUN_ROOT"/* 2>/dev/null | grep -i "v33${ARM}" | head -1)"
    if [[ -n "$RUN_DIR" && -d "$RUN_DIR/params" ]]; then
      ENV_YAML=$(ls "$RUN_DIR"/params/env.yaml 2>/dev/null | head -1)
      AGENT_YAML=$(ls "$RUN_DIR"/params/agent.yaml 2>/dev/null | head -1)
      if [[ -n "$ENV_YAML" && -n "$AGENT_YAML" ]]; then
        if [[ "$EXPECT_DROP_DISC" == "yes" ]]; then
          grep -qi 'drop_amp_discriminator' "$LOG" 2>/dev/null && break
        else
          break
        fi
      fi
    fi
    pgrep -f 'train[.]py' >/dev/null 2>&1 || break
    sleep 10
  done
  log "  run 目录：${RUN_DIR:-（未找到）}"
  log "  env.yaml：${ENV_YAML:-（未找到）}   agent.yaml：${AGENT_YAML:-（未找到）}"

  BAD=0

  # ① side_vel_min（三臂共有的指令分布修正）
  if [[ -n "$ENV_YAML" ]] && grep -qE '^\s*side_vel_min: 0\.15' "$ENV_YAML"; then
    log "  [自检 ①] env.yaml side_vel_min = 0.15 ✅"
  elif [[ -z "$ENV_YAML" ]]; then
    # ⚠️ “文件还没落盘”是**数据不可得**，不是“证据矛盾” ⇒ 只告警、不杀臂
    log "  ⚠️ [自检 ①] 还读不到 env.yaml（就绪等待超时）—— 仅告警，不杀臂"
  else
    log "  [自检 ①] env.yaml 里没找到 side_vel_min: 0.15 ❌（侧向档会再落进静止门！）"; BAD=1
  fi

  # ② AMP 状态维度（**效果级、权威口径**）—— 从 wasabi 那行反推：
  #    `[wasabi] 专家批来自离线数据集：<path> shape=(N, D)` 里的 `D` 就是 AMP 状态维度，
  #    且 `wasabi.py::init_storage` 里有
  #        assert 专家集维度 == amp_storage.reference_states.shape[-1]
  #    ⇒ **能打印出这一行**本身就证明 `D == AMP 状态维度`（否则训练早就 assert 崩了）。
  # ⚠️⚠️ 2026-10-06 踩过的坑（白烧半小时）：原来用 `grep -oE "Linear\(in_features=" | head -1`
  #     取第一个 Linear —— 但日志里 **actor/critic 的网络先打印**，第一个 Linear 是
  #     **actor obs = 378** ⇒ S0/S1/A-cold 三臂被逐个误判成 ❌ 并**全部跳过**。
  #     教训：**判据要取"没有歧义的那一行"，并且推断性的判据不要用来杀臂**。
  EXPERT_SHAPE=$(grep -m1 '专家批来自离线数据集' "$LOG" 2>/dev/null | sed -nE 's/.*shape=\(([0-9]+), *([0-9]+)\).*/\1x\2/p')
  DIM=$(echo "$EXPERT_SHAPE" | sed -nE 's/.*x([0-9]+)$/\1/p')
  DIM=${DIM:-0}
  # 锚点：日志里第一个 Linear 必须是 actor obs = 378（确认我们没有读错行）
  A_DIM=$(grep -oE 'Linear\(in_features=[0-9]+' "$LOG" 2>/dev/null | head -1 | grep -oE '[0-9]+')
  if [[ "$A_DIM" == "378" ]]; then
    log "  [自检 ②-锚点] 日志首个 Linear = actor obs 378 ✅"
  else
    log "  ⚠️ [自检 ②-锚点] 日志首个 Linear = ${A_DIM:-?}（期望 378；仅提示，不跳过）"
  fi
  if [[ "$DIM" == "$EXPECT_DIM" ]]; then
    log "  [自检 ②] AMP 状态维 = $DIM ✅（期望 $EXPECT_DIM；专家批 shape=${EXPERT_SHAPE:-?}）"
  else
    log "  ⚠️ [自检 ②] AMP 状态维 = ${DIM:-?}（期望 $EXPECT_DIM）——**仅警告、不跳过**："\
"该值由 wasabi 的 assert 钉住，真不符的话训练会自己崩（不会静默）"
  fi

  # ③ 白化的效果级证据
  if grep -q 'EmpiricalNormalization' "$LOG" 2>/dev/null; then HAVE_NORM=yes; else HAVE_NORM=no; fi
  if [[ "$HAVE_NORM" == "$EXPECT_NORM" ]]; then
    log "  [自检 ③] 白化模块出现 = $HAVE_NORM ✅（期望 $EXPECT_NORM）"
  else
    log "  [自检 ③] 白化模块出现 = $HAVE_NORM ❌（期望 $EXPECT_NORM）"; BAD=1
  fi

  # ④ GP 空间
  if [[ -n "$AGENT_YAML" ]]; then
    GP=$(grep -E '^\s*amp_grad_pen_space:' "$AGENT_YAML" | awk '{print $2}')
    if [[ "$GP" == "$EXPECT_GP" ]]; then
      log "  [自检 ④] amp_grad_pen_space = $GP ✅（期望 $EXPECT_GP）"
    else
      log "  [自检 ④] amp_grad_pen_space = $GP ❌（期望 $EXPECT_GP）"; BAD=1
    fi
    # ④b 白化强度 eps（S0/S1 = 0.1、S2 = 0.01；A-cold 无此项）
    if [[ -n "$EXPECT_EPS" ]]; then
      EPS=$(grep -A3 'normalizer_kwargs' "$AGENT_YAML" | grep -E '^\s*eps:' | head -1 | awk '{print $2}')
      if [[ "$EPS" == "$EXPECT_EPS" ]]; then
        log "  [自检 ④b] normalizer eps = $EPS ✅（期望 $EXPECT_EPS）"
      else
        log "  [自检 ④b] normalizer eps = ${EPS:-（没找到）} ❌（期望 $EXPECT_EPS）"; BAD=1
      fi
    fi
  else
    # ⚠️ 同 ①：“还没落盘”只是数据不可得 ⇒ 只告警
    log "  ⚠️ [自检 ④] 还读不到 agent.yaml（就绪等待超时）—— 仅告警，不杀臂"
  fi

  # ⑤ 奖励权重横幅：**按臂别的期望值逐项核**（比"是不是某个字符串"牢靠；
  #    新增臂只需在 case 里填 EXPECT_AIR/EXPECT_ARMS/EXPECT_TRACK）
  # ⚠️ 2026-10-06：这里曾被挤成一行 `if ...; then    if ...; then` ⇒ `bash -n` 直接挂。
  W=$(grep -m1 'AMP-WALK WEIGHTS' "$LOG")
  _okw=1
  _kvs="joint_deviation_arms=$EXPECT_ARMS track_lin_vel_xy_exp=$EXPECT_TRACK"
  # ⭐ `feet_air_time=0` 特殊处理：RewardManager 若跳过 weight=0 的 term，横幅里可能**没有**这一项
  #    ⇒ 两种都算过（否则会把臂误判为失败 —— 2026-10-06 就吃过“判据太紧”的亏）
  if [[ "$EXPECT_AIR" == "0" ]]; then
    if [[ "$W" == *"feet_air_time=0"* || "$W" != *"feet_air_time="* ]]; then
      log "  [自检 ⑤-a] feet_air_time 已取消（横幅：$( [[ "$W" == *"feet_air_time=0"* ]] && echo '列出为 0' || echo '未列出' )）✅"
    else
      log "  [自检 ⑤-a] feet_air_time 仍然非 0 ❌"
      _okw=0
    fi
  else
    _kvs="feet_air_time=$EXPECT_AIR $_kvs"
  fi
  for _kv in $_kvs; do
    if [[ "$W" != *"$_kv"* ]]; then
      log "  [自检 ⑤] 横幅里缺 '$_kv' ❌"
      _okw=0
    fi
  done
  if [[ "$_okw" == "1" ]]; then
    log "  [自检 ⑤] 奖励横幅 = air$EXPECT_AIR / arms$EXPECT_ARMS / track$EXPECT_TRACK ✅"
  else
    log "  [自检 ⑤] 奖励横幅与期望不符 ❌"; echo "    $W" | cut -c1-260; BAD=1
  fi

  # ⑥ 判别器口径：**按臂别期望**（`EXPECT_DROP_DISC`）
  #    S0/S1/S2/S3/ACOLD：判别器口径变了（396/白化/GP 空间）⇒ **必须**丢弃重学
  #    ⭐ S4（续训臂）：396 布局与专家集与 S3 完全相同 ⇒ **故意保留**权重
  #       ⇒ 此时"日志里没有 drop_amp_discriminator"才是对的。
  # ⚠️⚠️ 2026-10-06 踩过的坑（白烧一次启动）：这里原来写死"必须有 drop_amp_discriminator"
  #    ⇒ S4 在横幅出现 **6 秒后**被判 BAD=1 ⇒ `pkill -9 train.py` + 杀守护，
  #    TB events 一直是 **0 字节**，绘图脚本报"找不到数据"。
  #    教训（同自检 ②）：**臂别相关的判据必须参数化**，否则会静默杀臂。
  if grep -qi 'drop_amp_discriminator' "$LOG"; then HAVE_DROP=yes; else HAVE_DROP=no; fi
  if [[ "$HAVE_DROP" == "$EXPECT_DROP_DISC" ]]; then
    log "  [自检 ⑥] drop_amp_discriminator 出现 = $HAVE_DROP ✅（期望 $EXPECT_DROP_DISC）"
  else
    log "  [自检 ⑥] ❌ drop_amp_discriminator 出现 = $HAVE_DROP（期望 $EXPECT_DROP_DISC）"
    if [[ "$EXPECT_DROP_DISC" == "no" ]]; then
      log "     ⇒ 本臂应**保留**判别器却丢了它（风格从零学起，与本臂设计不符）"
    else
      log "     ⇒ 本臂应**丢弃**判别器却延续了旧权重（口径变了 ⇒ 会崩或退化）"
    fi
    BAD=1
  fi

  # ⑦ 专家批路径
  if grep -qF "$EXPECT_EXPERT" "$LOG" 2>/dev/null; then
    log "  [自检 ⑦] 专家批 = $EXPECT_EXPERT ✅"
  else
    log "  [自检 ⑦] 专家批路径不符（期望 $EXPECT_EXPERT）❌"; BAD=1
  fi

  if [[ "$BAD" == "1" ]]; then
    log "!!! ${ARM} 臂有效性自检未通过 ⇒ **跳过该臂**（数据不可用于归因），继续下一臂"
    FAILED_ARMS="$FAILED_ARMS $ARM(自检失败)"
    pkill -9 -f 'train[.]py' || true
    pkill -9 -f 'v29_guard' || true
    sleep 5
    # 把刚被杀的 run 目录挪走：它的 TB events 很可能是 0 字节，
    # 会污染按 tag 匹配的绘图脚本（2026-10-06 就是这个现象让"曲线找不到数据"）。
    if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
      mkdir -p /root/dead_runs
      mv "$RUN_DIR" /root/dead_runs/ 2>/dev/null \
        && log "  （死掉的 run 目录已移到 /root/dead_runs/，避免绘图脚本匹配到空 events）"
    fi
    continue
  fi

  # ---- 等到达目标轮次 ----
  while true; do
    sleep 60
    CUR=$(grep -o 'Learning iteration [0-9]*' "$LOG" 2>/dev/null | tail -1 | awk '{print $3}')
    CUR=${CUR:-0}
    if ! pgrep -f 'train[.]py' >/dev/null 2>&1; then
      log "!!! ${ARM} 臂训练进程消失（崩溃 / 守护触发），当前轮次=$CUR ⇒ 停止 sweep"
      FAILED_ARMS="$FAILED_ARMS $ARM(进程异常退出@$CUR)"
      log "---------- ${ARM} 臂尾部日志 ----------"
      tail -25 "$LOG"
      exit 1
    fi
    # 白化失效**预警**（不自动退出，供人判断）：判别器死掉的三联征
    if (( CUR > FORK + 150 && CUR % 100 < 2 )); then
      SSD=$(grep -oE 'Mean step amp_style_std: *[0-9.eE+-]+' "$LOG" | tail -1 | awk '{print $NF}')
      SZF=$(grep -oE 'Mean step amp_style_zero_frac: *[0-9.eE+-]+' "$LOG" | tail -1 | awk '{print $NF}')
      log "  ⚠️ 白化三联征监测 @$CUR：style_std=${SSD:-?}（<0.03 危险）  zero_frac=${SZF:-?}（>0.5 危险）"
    fi
    if [[ "$CUR" -ge "$TARGET" ]]; then
      log "${ARM} 臂到达 $CUR 轮（目标 $TARGET）⇒ 收尾"
      break
    fi
  done

  pkill -9 -f 'train[.]py' || true
  pkill -9 -f 'v29_guard' || true
  sleep 5
  log "---------- ${ARM} 臂摘要（尾段 40 行关键指标）----------"
  grep -E "$WATCH" "$LOG" | sed 's/^ *//' | tail -40
  log "  该臂 ckpt：$(ls -1 "$RUN_DIR"/model_*.pt 2>/dev/null | sed 's#.*/##' | tr '\n' ' ')"
done

log "================ V33 sweep 结束 ================"
[[ -n "$FAILED_ARMS" ]] && log "⚠️ 有问题的臂：$FAILED_ARMS" || log "三臂全部完成且自检通过 ✅"
