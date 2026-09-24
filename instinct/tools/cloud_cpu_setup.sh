#!/usr/bin/env bash
# =============================================================================
# CPU-only 云平台：装机（第 1 层验收）+ 免 Kit 的导入探测
# =============================================================================
# 前提：/root/gpufree-data/instinct_xu.tgz 已上传
#
# 做的事
#   1. 解开 instinct_xu.tar
#   2. 两份源码 editable 安装（**必须 --no-deps**：不能动已装好的 Isaac Sim/Torch）
#   3. 跑交付包自带的 scripts/check_install.py（第 1 层验收）
#   4. 探测「不启动 Isaac Sim 能导入到哪一层」——
#      ★ 这正是 CPU-only 机器上的关键问题：只要能 import isaaclab.envs，
#        就能在本机把 G1AmpWalkEnvCfg / G1AmpWalkPPORunnerCfg 实例化并 to_dict()，
#        从而**实证**验证步骤⑤⑥ 的配置接线（不需要 GPU）
# =============================================================================
set -uo pipefail

PY=/opt/conda/envs/isaaclab/bin/python
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1

hr() { printf '%s\n' "----------------------------------------------------------------"; }
sec() { echo; hr; echo "$1"; hr; }

sec "0. 环境引导文件 /root/.llp_env.sh"
if [[ -f /root/.llp_env.sh ]]; then
  # shellcheck disable=SC1091
  source /root/.llp_env.sh
  echo "[OK] 已 source /root/.llp_env.sh"
  echo "PYTHONPATH=$PYTHONPATH"
  echo "LLP_DATA_ROOT=${LLP_DATA_ROOT:-<未设>}"
else
  echo "[WARN] 没有 /root/.llp_env.sh，退化为手工设置"
fi
echo "python: $($PY -V 2>&1)"

sec "1. 解包 instinct_xu"
cd /root/gpufree-data || exit 1
if [[ -f instinct_xu.tgz ]]; then
  tar xzf instinct_xu.tgz
  echo "[OK] 已解包"
else
  echo "[WARN] 没找到 instinct_xu.tgz（可能已解包）"
fi
ROOT=/root/gpufree-data/instinct_xu
[[ -d "$ROOT" ]] || { echo "[FAIL] 找不到 $ROOT"; exit 1; }
ls -1 "$ROOT"
du -sh "$ROOT"

sec "2. editable 安装两份源码（--no-deps）"
cd "$ROOT" || exit 1
for pkg in instinctlab instinct_rl; do
  if [[ -d "source/$pkg" ]]; then
    echo "--- pip install -e source/$pkg --no-deps"
    $PY -m pip install -e "source/$pkg" --no-deps -q 2>&1 | tail -5
    echo "    exit=$?"
  fi
done
echo "--- pip list --editable ---"
$PY -m pip list --editable 2>&1 | tail -6

sec "3. 交付包自带自检 scripts/check_install.py"
if [[ -f "$ROOT/scripts/check_install.py" ]]; then
  $PY "$ROOT/scripts/check_install.py" 2>&1 | tail -30
  echo "(exit=$?)"
else
  echo "[WARN] 没有 scripts/check_install.py"
  ls -1 "$ROOT/scripts" 2>/dev/null
fi

sec "4. 免 Kit 导入探测（决定 CPU-only 机器上还能做什么）"
cd "$ROOT" || exit 1
$PY - <<'PYEOF'
import importlib, sys, traceback

probes = [
    "torch", "numpy", "gymnasium",
    "isaaclab",
    "isaaclab.utils", "isaaclab.utils.math", "isaaclab.utils.configclass",
    "isaaclab.managers",
    "isaaclab.envs",
    "isaaclab.envs.mdp",
    "isaaclab.sensors",
    "instinct_rl",
    "instinct_rl.algorithms",
    "instinctlab",
    "instinctlab.utils.wrappers.instinct_rl",
    "instinctlab.tasks.locomotion.config.g1",
]
ok = {}
for m in probes:
    try:
        importlib.import_module(m)
        ok[m] = True
        print(f"  OK   {m}")
    except BaseException as e:            # noqa: BLE001  (Kit 抛的可能是 SystemExit)
        ok[m] = False
        print(f"  FAIL {m}: {type(e).__name__}: {str(e)[:110]}")

print()
if ok.get("instinctlab.tasks.locomotion.config.g1"):
    print("== 能导入任务配置包 → 尝试实例化配置并 to_dict() ==")
    try:
        from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_ppo_cfg import (
            G1AmpWalkPPORunnerCfg,
        )
        cfg = G1AmpWalkPPORunnerCfg()
        d = cfg.to_dict()
        alg = d.get("algorithm", {})
        print("  构造 G1AmpWalkPPORunnerCfg 成功")
        print("  algorithm.class_name      =", alg.get("class_name"))
        for k in ("reward_mix_mode", "style_lerp", "disable_style_on_stand",
                  "stand_speed_threshold", "discriminator_reward_coef",
                  "discriminator_reward_type", "discriminator_loss_func",
                  "discriminator_kwargs", "auxiliary_reward_per_env_reward_coefs",
                  "clip_min_std"):
            present = k in alg
            print(f"  {'OK  ' if present else 'LOST'} algorithm.{k} = {alg.get(k)!r}")
        print("  policy.actor_hidden_dims  =", d.get("policy", {}).get("actor_hidden_dims"))
        print("  experiment_name           =", d.get("experiment_name"))
        print("  run_name                  =", d.get("run_name"))
        print("  algorithm 全部键:", sorted(alg.keys()))
    except BaseException as e:  # noqa: BLE001
        print("  FAIL 实例化/ to_dict()：")
        traceback.print_exc()
        sys.exit(0)

    try:
        from instinctlab.tasks.locomotion.config.g1 import G1AmpWalkEnvCfg
        ecfg = G1AmpWalkEnvCfg()
        print()
        print("  构造 G1AmpWalkEnvCfg 成功；rewards 类型 =",
              type(ecfg.rewards).__name__)
        names = [k for k in vars(ecfg.rewards) if not k.startswith("_")]
        print(f"  奖励项 {len(names)} 个:", sorted(names))
        print("  ---- 是否有 track_base_height / body_ang_vel_xy / feet_* ----")
        for k in ("track_base_height", "body_ang_vel_xy", "track_default_pose_on_stand",
                  "feet_lateral_width", "feet_crossing"):
            print(f"    {'OK  ' if k in names else 'MISS'} {k}")
    except BaseException:  # noqa: BLE001
        print("  FAIL 实例化 G1AmpWalkEnvCfg：")
        traceback.print_exc()
else:
    print("== 无法导入任务配置包（多半卡在 isaaclab.envs / Kit）→ 只能做源码审计 ==")
PYEOF

sec "完成"
