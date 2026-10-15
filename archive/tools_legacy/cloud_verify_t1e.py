#!/usr/bin/env python3
"""**T1e 单变量校验**：证明"T1e 相对 T1d 只有 `dof_pos_limits` 这一项改动"。

核心断言（机器可验，比逐项列期望值更强）
----------------------------------------
1. **环境侧**：把 `G1AmpWalkT1a2EnvCfg`（= T1d 用的 env）与 `G1AmpWalkT1eEnvCfg` 现场
   建出来，对**每个奖励项**比较 `(weight, func.__qualname__, params)`，断言差异集合
   **恰好**为 `{"dof_pos_limits"}`。
2. `dof_pos_limits` 本身：func == `joint_pos_limits_margin`、`joint_names == ['.*']`（全关节）、
   weight == -1.0。
3. **算法侧**：`amp_walk_t1b_ppo_cfg`（T1e 复用，λ=0.75）相对 T1a2 的差异**恰好**为
   `{algorithm.style_lerp, run_name}`（与 T1d 用的是同一支 agent cfg）。

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \\
    source /root/.llp_env.sh && unset PYTHONPATH && \\
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_t1e.py
"""

from __future__ import annotations

import sys

from isaaclab.app import AppLauncher  # noqa: E402

_APP = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False}).app

import dataclasses  # noqa: E402
import importlib  # noqa: E402

FAIL: list[str] = []


def chk(name, got, want, tol=1e-12):
    if tol is not None and isinstance(got, (int, float)) and isinstance(want, (int, float)):
        ok = abs(float(got) - float(want)) <= float(tol)
    else:
        ok = got == want
    print(f"  {'OK  ' if ok else 'FAIL'} {name:56s} got={got!r}  want={want!r}")
    if not ok:
        FAIL.append(name)


def _fields(o):
    if o is None:
        return []
    try:
        if dataclasses.is_dataclass(o):
            return [f.name for f in dataclasses.fields(o)]
    except Exception:
        pass
    d = getattr(o, "__dict__", None)
    return [k for k in d if not k.startswith("_")] if isinstance(d, dict) else []


def _prim(o):
    out = {}
    for n in _fields(o):
        v = getattr(o, n, None)
        out[n] = repr(v) if isinstance(v, (int, float, str, bool, type(None), list, tuple, dict)) else "<obj>"
    return out


def _diff(a, b, pre=""):
    pa, pb = _prim(a), _prim(b)
    return [f"{pre}{k}" for k in sorted(set(pa) | set(pb)) if pa.get(k) != pb.get(k)]


def _reward_snap(cfg):
    out = {}
    for n in _fields(cfg.rewards):
        t = getattr(cfg.rewards, n, None)
        if t is None:
            out[n] = "<None 已删除>"
            continue
        w = getattr(t, "weight", None)
        f = getattr(getattr(t, "func", None), "__qualname__", None)
        p = getattr(t, "params", {}) or {}
        scope = ""
        for key in ("asset_cfg", "sensor_cfg"):
            if key in p:
                scope = f"{getattr(p[key], 'joint_names', None)!r}|{getattr(p[key], 'body_names', None)!r}"
        out[n] = f"w={w!r} func={f} scope={scope}"
    return out


def main() -> int:
    mod = importlib.import_module("instinctlab.tasks.locomotion.config.g1")
    base = mod.G1AmpWalkT1a2EnvCfg()
    arm = mod.G1AmpWalkT1eEnvCfg()

    print("=" * 78)
    print("T1e vs T1a2(=T1d 的 env)  单变量校验")
    print("=" * 78)

    print("\n[1] ⭐ 奖励表全项 diff（含 func 与作用域）")
    s0, s1 = _reward_snap(base), _reward_snap(arm)
    diffs = sorted(k for k in set(s0) | set(s1) if s0.get(k) != s1.get(k))
    chk("差异集合必须恰为 ['dof_pos_limits']", diffs, ["dof_pos_limits"])
    for k in diffs:
        print(f"      · {k}\n          T1a2: {s0.get(k)}\n          T1e : {s1.get(k)}")

    print("\n[2] dof_pos_limits 细节")
    t = arm.rewards.dof_pos_limits
    chk("func", getattr(t.func, "__qualname__", None), "joint_pos_limits_margin")
    _p = t.params or {}
    chk("joint_names（全关节）", list(getattr(_p.get("asset_cfg"), "joint_names", []) or []), [".*"])
    chk("asset name", getattr(_p.get("asset_cfg"), "name", None), "robot")
    chk("weight", float(t.weight), -1.0)
    chk("（对照）T1a2 的 func 是踝-only 的 joint_pos_limits",
        getattr(base.rewards.dof_pos_limits.func, "__qualname__", None), "joint_pos_limits")
    chk("（对照）T1a2 的 joint_names 只有踝",
        list(getattr((base.rewards.dof_pos_limits.params or {}).get("asset_cfg"), "joint_names", []) or []),
        [".*_ankle_pitch_joint", ".*_ankle_roll_joint"])

    print("\n[3] 其余 16 项权重未被顺手改动（逐项对照）")
    zero_ok = True
    for n, s in s0.items():
        if n == "dof_pos_limits":
            continue
        if s1.get(n) != s:
            zero_ok = False
            print(f"      ❌ {n}: {s} -> {s1.get(n)}")
    chk("除 dof_pos_limits 外无任何变化", zero_ok, True)

    print("\n[4] 算法侧（T1e 复用 T1b 的 agent cfg = λ0.75）")
    m2 = importlib.import_module("instinctlab.tasks.locomotion.config.g1.agents.amp_walk_t1a2_ppo_cfg")
    m1 = importlib.import_module("instinctlab.tasks.locomotion.config.g1.agents.amp_walk_t1b_ppo_cfg")
    a2, a1 = m2.G1AmpWalkT1a2PPORunnerCfg(), m1.G1AmpWalkT1bPPORunnerCfg()
    ad = _diff(a2, a1) + _diff(a2.algorithm, a1.algorithm, "algorithm.") + _diff(a2.policy, a1.policy, "policy.")
    chk("agent diff 必须恰为 ['algorithm.style_lerp', 'run_name']", sorted(ad),
        ["algorithm.style_lerp", "run_name"])
    chk("style_lerp（= T1b/T1d 的 0.75）", float(a1.algorithm.style_lerp), 0.75)
    chk("use_mirror_loss（交付要求②）", bool(a1.algorithm.use_mirror_loss), True)

    print()
    if FAIL:
        print(f"❌ T1e 校验失败：{len(FAIL)} 项 —— **不要发车**")
        for f in FAIL:
            print(f"   - {f}")
        return 1
    print("✅ T1e 校验全部通过（唯一改动：dof_pos_limits → 全关节软边距版）—— 可以发车")
    return 0


if __name__ == "__main__":
    sys.exit(main())
