#!/usr/bin/env bash
# =============================================================================
# 云端冒烟验证：把 ②③④⑤⑥ 积累的「本机无法验证的断言」一次性清掉
# =============================================================================
#
# 为什么需要它
# ------------
# 本机显存 4 GB 跑不了 Isaac Sim（最低 8 GB），所以 ①~⑥ 的代码**从未真实运行过**。
# 其中若干条假设只能在这里验证（详见 CLOUD_ENV_SETUP.md §5）：
#   1. isaaclab.envs.mdp.UniformVelocityCommand（类）能否导入
#   2. 它的内部缓冲是否真的叫 vel_command_b
#   3. robot_reference + motion_reference 叠加到平面场景后能否构建
#   4. motion_reference 关节序与机器人是否对齐（AMP 两侧同构依赖它）
#   5. isaaclab.envs.mdp.joint_vel_rel 是否存在
#   6. 步骤⑤ 的奖励集能否被 RewardManager 实例化并跑通前向
#      （任一 func 缺失 / 参数名写错 / asset.data.<attr> 不存在都会在这里暴露）
#   7. LL_FOOT / LR_FOOT 是否真的在 robot.body_names 里（步宽量的参考点）
#   8. 步骤⑥ 的 WasabiPPOAmpMix 能否被 runner 按 "module:Class" 解析到、
#      判别器能否构建、amp_policy/amp_reference 能否送进判别器
#
# 用法（在云机器上、已激活 conda 环境后）：
#   bash /path/to/llp/instinct/tools/cloud_smoke_amp_walk.sh
#   INSTINCT_XU_ROOT=/root/instinct_xu bash ...   # 若包在别处
#
# 默认不跑原任务的安装冒烟（README §5），要跑加 --with-install-smoke
# =============================================================================
set -uo pipefail

INSTINCT_XU_ROOT="${INSTINCT_XU_ROOT:-$HOME/instinct_xu}"
NUM_ENVS="${NUM_ENVS:-16}"
MAX_ITERS="${MAX_ITERS:-2}"
DEVICE="${DEVICE:-cuda:0}"
WITH_INSTALL_SMOKE=0
[[ "${1:-}" == "--with-install-smoke" ]] && WITH_INSTALL_SMOKE=1

FAIL=0
hr() { printf '%s\n' "------------------------------------------------------------------------"; }
ok()   { printf '  \033[32m✅ %s\033[0m\n' "$1"; }
bad()  { printf '  \033[31m❌ %s\033[0m\n' "$1"; FAIL=1; }
warn() { printf '  \033[33m⚠️  %s\033[0m\n' "$1"; }

echo "================================================================"
echo " AMP-Walk 云端冒烟验证"
echo " instinct_xu : $INSTINCT_XU_ROOT"
echo " envs/iters  : $NUM_ENVS / $MAX_ITERS   device=$DEVICE"
echo "================================================================"

if [[ ! -x "$INSTINCT_XU_ROOT/train.sh" ]]; then
  bad "找不到可执行的 $INSTINCT_XU_ROOT/train.sh（用 INSTINCT_XU_ROOT=... 指定）"
  exit 1
fi
cd "$INSTINCT_XU_ROOT" || exit 1

# ---------------------------------------------------------------------------
hr; echo "第 0 层：环境与 GPU"
# ---------------------------------------------------------------------------
python -c "import sys; print('  Python', sys.version.split()[0], '->', sys.executable)" || bad "python 不可用"
python - <<'PY' || warn "torch/CUDA 自检未通过（若后面训练能起来可忽略）"
try:
    import torch
    print(f"  PyTorch {torch.__version__}  CUDA runtime {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"显存 {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    else:
        print("  ⚠️ torch.cuda.is_available() = False")
except Exception as e:
    print(f"  ⚠️ {e}")
PY

# ---------------------------------------------------------------------------
hr; echo "第 1 层：安装指向（交付包自带脚本）"
# ---------------------------------------------------------------------------
if [[ -f scripts/check_install.py ]]; then
  # ⚠️ 降级为**提示**（warn），不是失败项：这是交付包自带的「pip 元数据」自检，
  # 已知在本镜像里会因为 isaaclab_assets 没有 dist-info 而报
  # `PackageNotFoundError: No package metadata was found for isaaclab_assets` ——
  # 那是**缺元数据**，不是缺包，也**不影响本任务**：
  # 第 3 层的端到端训练退出码 0 就是最直接的证据（CLOUD_ENV_SETUP.md §10.6）。
  # 真正的裁决永远看第 3 层。
  if python scripts/check_install.py; then
    ok "check_install.py 通过"
  else
    warn "check_install.py 报告问题（提示项，见 CLOUD_ENV_SETUP.md §10.6；不影响本任务）"
    python scripts/check_install.py 2>&1 | grep -E 'PackageNotFoundError|RuntimeError' | head -3 | sed 's/^/     /'
  fi
  echo "  -- editable 安装 --"
  python -m pip list --editable 2>/dev/null | head -10
else
  warn "未找到 scripts/check_install.py，跳过"
fi

# ---------------------------------------------------------------------------
hr; echo "第 2 层：★ 关键 API 假设探测（在 Kit 内跑）"
# ---------------------------------------------------------------------------
# ⚠️ 必须在 **Kit 内** 探测：`isaaclab.envs` / `instinctlab` 的导入链都要
#    `pxr`(USD) / `omni` / `carb`，而这些只在 Kit 启动后才可用。
#    第一版这一层用**普通 python** 跑 → 6 项全部假失败（`ModuleNotFoundError: pxr`），
#    等于没验。现在改成：先生成「带 AppLauncher 前导」的脚本，再跑。
cat > /tmp/llp_layer2_probe.py <<'PY'
import argparse

from isaaclab.app import AppLauncher

_parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_parser)
_args, _unknown = _parser.parse_known_args([])
_args.headless = True
_app = AppLauncher(_args).app

import importlib  # noqa: E402

problems = []

def probe(desc, fn):
    try:
        r = fn()
        print(f"  ✅ {desc}" + (f"  -> {r}" if r is not None else ""))
    except Exception as e:
        problems.append(f"{desc}: {type(e).__name__}: {e}")
        print(f"  ⚠️  {desc}  -> {type(e).__name__}: {e}")

def get_mdp():
    return importlib.import_module("isaaclab.envs.mdp")

probe("能导入 isaaclab.envs.mdp", get_mdp)

# ③ 依赖：UniformVelocityCommand 的**类**（仓库内只有 ...Cfg 的先例）
def _cls():
    m = get_mdp()
    c = getattr(m, "UniformVelocityCommand", None)
    if c is None:
        raise AttributeError("isaaclab.envs.mdp 未导出 UniformVelocityCommand")
    return c.__name__
probe("isaaclab.envs.mdp.UniformVelocityCommand 存在", _cls)

def _cfg():
    m = get_mdp()
    return getattr(m, "UniformVelocityCommandCfg").__name__
probe("isaaclab.envs.mdp.UniformVelocityCommandCfg 存在", _cfg)

# ④ 依赖：joint_vel_rel（instinctlab 未转发，我们取自 isaaclab）
probe("isaaclab.envs.mdp.joint_vel_rel 存在",
      lambda: getattr(get_mdp(), "joint_vel_rel").__name__)

# ② 依赖：instinctlab 侧必须**没有** base_lin_vel / joint_vel_rel（我们的写法前提）
def _inst():
    m = importlib.import_module("instinctlab.envs.mdp")
    bad = [n for n in ("base_lin_vel", "joint_vel_rel") if hasattr(m, n)]
    if bad:
        raise AssertionError(f"instinctlab.envs.mdp 竟然导出了 {bad}（我们假设它没有）")
    return "确认未导出 base_lin_vel / joint_vel_rel"
probe("instinctlab.envs.mdp 未导出 base_lin_vel / joint_vel_rel", _inst)

# ④ 依赖：*_reference_as_state 全家都在 instinctlab 侧
def _ref():
    m = importlib.import_module("instinctlab.envs.mdp")
    need = ["projected_gravity_reference_as_state", "joint_pos_rel_reference_as_state",
            "joint_vel_rel_reference_as_state", "base_lin_vel_reference_as_state",
            "base_ang_vel_reference_as_state"]
    miss = [n for n in need if not hasattr(m, n)]
    if miss:
        raise AttributeError(f"缺少 {miss}")
    return f"{len(need)} 项齐全"
probe("instinctlab.envs.mdp 的 5 个 *_reference_as_state 齐全", _ref)

print()
if problems:
    print("  ⚠️  有 %d 项探测失败。**以第 3 层的训练结果为准**。" % len(problems))
    for p in problems:
        print(f"     - {p}")
else:
    print("  ✅ 关键 API 假设全部成立")
_app.close()
PY
# 落到文件再过滤，避免 pipe 缓冲导致全程看不到进度
python /tmp/llp_layer2_probe.py > /tmp/llp_layer2.out 2>&1
_layer2_rc=$?
grep -vE "^\[[0-9.]+s\]|\[carb\]|\[omni|Warning: " /tmp/llp_layer2.out | tail -45
[[ $_layer2_rc -eq 0 ]] && ok "第 2 层探测脚本退出码 0" || warn "第 2 层探测脚本退出码 $_layer2_rc"

# ---------------------------------------------------------------------------
hr; echo "第 3 层：★ 我们新任务的端到端冒烟（真正的裁决）"
# ---------------------------------------------------------------------------
echo "  命令: ./train.sh --task Instinct-Amp-Walk-G1-v0 --num_envs $NUM_ENVS \\"
echo "                  --max_iterations $MAX_ITERS --device $DEVICE --headless"
echo
RUN_NAME="amp_walk_smoke_$(date +%Y%m%d_%H%M%S)"
LOG="/tmp/${RUN_NAME}.log"
CUDA_VISIBLE_DEVICES="${DEVICE##*:}" ./train.sh \
  --task Instinct-Amp-Walk-G1-v0 \
  --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERS" \
  --device "$DEVICE" --headless \
  --run_name "$RUN_NAME" 2>&1 | tee "$LOG"
TRAIN_RC="${PIPESTATUS[0]}"
echo

if [[ "$TRAIN_RC" -eq 0 ]]; then
  ok "训练进程退出码 0"
else
  bad "训练进程退出码 $TRAIN_RC（日志：$LOG）"
fi

# 契约校验器（步骤② 复用学长那套 fail-fast）是否打印了通过标记
if grep -q "\[AMP-WALK PASS\]" "$LOG"; then
  ok "契约校验通过（[AMP-WALK PASS]）"
  grep -m1 "\[AMP-WALK PASS\]" "$LOG" | sed 's/^/     /'
else
  bad "未看到 [AMP-WALK PASS] —— 契约校验没跑到或被跳过"
fi
# 父类校验器（OmniLoco 那套 fail-fast）也会打印 `[A1 PASS]`，而且**本来就该出现**：
# `G1AmpWalkTeacherEnv._validate_teacher_contract()` 是先 `super()` 再补 AMP 专项检查。
# 所以正确的判据不是「有没有 A1 PASS」，而是「A1 报的维度是不是本任务的 626」——
# 若报 623 才说明注册到了旧 env class。
_A1="$(grep -m1 "\[A1 PASS\]" "$LOG" || true)"
if [[ -z "$_A1" ]]; then
  bad "未看到 [A1 PASS] —— 父类契约校验没跑到（正常应由 super() 先执行）"
elif grep -qE "\[A1 PASS\].*policy=626[ )]" <<<"$_A1"; then
  ok "父类契约校验通过（[A1 PASS]，policy=626 与本任务一致）"
  echo "     $_A1" | sed 's/^     //' | sed 's/^/     /'
else
  bad "父类校验器报的不是 626（可能用错了 env class）：$_A1"
fi

# 步骤⑤：脚距诊断量（`extras["step"]` 通道）——**在线自检脚部参考 body 是否选对**。
# 数据集直线走实测中位 0.2140、p1 0.1455，所以训练时的均值应落在 0.05~0.6 区间。
# 注意：`extras["step"]` 跨 step 始终是同一个 dict 对象，runner 里 `step_infos` 存的
# 是一组同一对象的引用，因此这条曲线是「本轮最后一次 step 的瞬时值」，不是轮内均值。
_SEP="$(grep -oE 'Mean step foot_sep_mean:[^0-9-]*[-0-9.]+' "$LOG" | tail -1 || true)"
if [[ -z "$_SEP" ]]; then
  warn "未找到 foot_sep_mean —— 步骤⑤ 的诊断量没走到 extras['step']（检查 _log_metric）"
else
  _V="$(grep -oE '[-0-9]+\.?[0-9]*$' <<<"$_SEP")"
  if awk "BEGIN{exit !($_V > 0.05 && $_V < 0.6)}"; then
    ok "脚距诊断量合理：$_SEP（数据集直线走中位 0.2140）"
  else
    warn "脚距诊断量超出合理区间 0.05~0.6：$_SEP —— 检查脚部参考 body（ankle_roll_link 退路）是否选错"
  fi
fi

# 步骤⑥：判别器是否真的被构建（`WasabiAlgoMixin.init_storage` 里会 print）
if grep -q "Discriminator Network Structure" "$LOG"; then
  ok "判别器已构建（步骤⑥ WASABI 接线生效）"
  grep -m1 "Discriminator Network Structure" "$LOG" | sed 's/^/     /'
else
  bad "未看到 'Discriminator Network Structure' —— 算法可能不是 WasabiPPOAmpMix，或 init_storage 没跑到"
fi
# 步骤⑥：奖励混合是靠 `infos['observations']['amp_policy'/'amp_reference']` 喂判别器的，
# 任一 key 缺失会在 `WasabiAlgoMixin.process_env_step` 里直接 ValueError。
if grep -q "is not found in the observation dictionary" "$LOG"; then
  bad "amp_policy / amp_reference 没出现在 infos['observations'] 里（检查观测组注册与展平）"
else
  ok "amp_policy / amp_reference 已被正确送入判别器"
fi

echo
echo "  -- 关键报错检索 --"
if grep -qiE "Traceback|RuntimeError|AttributeError|KeyError|AssertionError|CUDA out of memory" "$LOG"; then
  bad "日志中存在异常，摘录如下："
  grep -niE "Traceback|RuntimeError|AttributeError|KeyError|AssertionError|CUDA out of memory" "$LOG" | head -15 | sed 's/^/     /'
else
  ok "未发现异常关键字"
fi

# ---------------------------------------------------------------------------
if [[ "$WITH_INSTALL_SMOKE" -eq 1 ]]; then
  hr; echo "第 4 层：原任务安装冒烟（README §5）"
  CUDA_VISIBLE_DEVICES="${DEVICE##*:}" ./train.sh \
    --task Instinct-Perceptive-Vault-G1-v0 \
    --num_envs 64 --max_iterations 3 --device "$DEVICE" --headless \
    --run_name instinct_xu_install_smoke_$(date +%H%M%S) 2>&1 | tail -25
  [[ "${PIPESTATUS[0]}" -eq 0 ]] && ok "原任务冒烟通过" || bad "原任务冒烟失败"
fi

# ---------------------------------------------------------------------------
hr
if [[ "$FAIL" -eq 0 ]]; then
  echo "✅ 冒烟验证全部通过。①~④ 的未验证断言已清除，可以放心继续 ⑤⑥。"
  echo "   日志：$LOG"
else
  echo "❌ 存在失败项（见上方 ❌）。请把本脚本输出与 $LOG 一起反馈。"
  echo
  echo "   对照 CLOUD_ENV_SETUP.md §5 的「已知假设」清单逐条排查："
  echo "     - UniformVelocityCommand / vel_command_b  → 步骤③ 需改用其它内部访问方式"
  echo "     - motion_reference / robot_reference      → 步骤④ 场景接线需调整"
  echo "     - AMP 两侧尺寸不等                        → 步骤④ 观测项需重新对齐"
fi
exit "$FAIL"
