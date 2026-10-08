#!/usr/bin/env python3
"""**T1i 发车前硬校验**（fail-fast）：证明"只改了风格奖励口径"。

断言
----
1. 算法侧相对 T1a2 的差异**恰好**为
   `{algorithm.amp_reward_form, algorithm.discriminator_reward_type,
     algorithm.disable_style_on_stand, run_name}`
2. 取值：`bounded_quad` / `wasserstein` / `True`
3. 守卫 T1a2 口径未被顺手改动：`style_lerp=0.75`、`clip_min_std=1e-12`、
   hidden `[256,128,128]`、`entropy_coef=0.008`、`init_noise_std=1.0`
4. **环境侧零改动**：`G1AmpWalkT1a2EnvCfg` 与 T1i 用的环境是同一个类
   ⇒ 奖励表全项逐字符快照必须完全一致（本臂复用 T1a2 的 env cfg）

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \\
    source /root/.llp_env.sh && unset PYTHONPATH && \\
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_t1i.py
"""

from __future__ import annotations

import sys

from isaaclab.app import AppLauncher  # noqa: E402

_APP = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False}).app

import dataclasses  # noqa: E402
import importlib  # noqa: E402

FAIL: list[str] = []


def chk(name, got, want, tol=1e-12):
    if isinstance(got, (int, float)) and isinstance(want, (int, float)):
        ok = abs(float(got) - float(want)) <= tol
    else:
        ok = got == want
    print(f"  {'OK  ' if ok else 'FAIL'} {name:50s} got={got!r}  want={want!r}")
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
    return {
        n: (repr(v) if isinstance(v, (int, float, str, bool, type(None), list, tuple, dict)) else "<obj>")
        for n in _fields(o)
        for v in [getattr(o, n, None)]
    }


def _diff(a, b, pre=""):
    pa, pb = _prim(a), _prim(b)
    return [f"{pre}{k}" for k in sorted(set(pa) | set(pb)) if pa.get(k) != pb.get(k)]


def _reward_snap(cfg):
    out = {}
    for n in _fields(cfg.rewards):
        t = getattr(cfg.rewards, n, None)
        if t is None:
            out[n] = "<None>"
            continue
        p = getattr(t, "params", {}) or {}
        scope = ""
        for key in ("asset_cfg", "sensor_cfg"):
            if key in p:
                scope = f"{getattr(p[key], 'joint_names', None)!r}|{getattr(p[key], 'body_names', None)!r}"
        out[n] = f"w={getattr(t, 'weight', None)!r} func={getattr(getattr(t, 'func', None), '__qualname__', None)} scope={scope}"
    return out


def main() -> int:
    b_mod = importlib.import_module("instinctlab.tasks.locomotion.config.g1.agents.amp_walk_t1a2_ppo_cfg")
    a_mod = importlib.import_module("instinctlab.tasks.locomotion.config.g1.agents.amp_walk_t1i_ppo_cfg")
    base = b_mod.G1AmpWalkT1a2PPORunnerCfg()
    arm = a_mod.G1AmpWalkT1iPPORunnerCfg()

    print("=" * 78)
    print("T1i 发车前校验（只改风格奖励口径）")
    print("=" * 78)

    print("\n[1] ⭐ 算法侧全字段 diff")
    d = _diff(base, arm) + _diff(base.algorithm, arm.algorithm, "algorithm.") + _diff(base.policy, arm.policy, "policy.")
    want = sorted(["run_name", "algorithm.style_lerp"])
    chk("差异集合必须恰为 ['algorithm.style_lerp','run_name']（唯一变量=风格强度）", sorted(d), want)
    #: ⚠️ 2026-10-08 修正：`amp_reward_form` / `discriminator_reward_type` /
    #:    `disable_style_on_stand` **本来就已经是 mjlab 形式**（T1a 链继承自 V33 系 cfg，
    #:    不是我先前误读的 `amp_walk_ppo_cfg` 默认值）⇒ 它们不该出现在 diff 里。
    #:    唯一变量 = `style_lerp` 1.0 → 0.975（风格强度 0.025 = mjlab 的 0.25×0.1）。
    for k in sorted(d):
        print(f"      · {k}: {_prim(base)[k.split('.')[-1]] if '.' not in k else '<algorithm>'} -> ...")

    print("\n[2] 目标取值")
    chk("amp_reward_form", arm.algorithm.amp_reward_form, "bounded_quad")
    chk("amp_quad_k（mjlab 的 ¼）", float(arm.algorithm.amp_quad_k), 0.25)
    chk("discriminator_reward_type", arm.algorithm.discriminator_reward_type, "wasserstein")
    chk("disable_style_on_stand", bool(arm.algorithm.disable_style_on_stand), True)

    print("\n[3] 守卫 T1a2 口径未被顺手改动")
    chk("style_lerp", float(arm.algorithm.style_lerp), 0.75)
    chk("clip_min_std", float(arm.algorithm.clip_min_std), 1e-12)
    chk("clip_max_std", getattr(arm.algorithm, "clip_max_std", None), None)
    chk("actor_hidden_dims", list(arm.policy.actor_hidden_dims), [256, 128, 128])
    chk("entropy_coef", float(arm.algorithm.entropy_coef), 0.008)
    chk("init_noise_std", float(arm.policy.init_noise_std), 1.0)
    chk("use_mirror_loss", bool(getattr(arm.algorithm, "use_mirror_loss", False)), True)

    print("\n[4] 环境侧零改动（本臂复用 T1a2 的 env cfg 类）")
    m = importlib.import_module("instinctlab.tasks.locomotion.config.g1")
    import gymnasium as gym  # noqa: E402

    spec = gym.spec("Instinct-Amp-Walk-T1i-G1-v0")
    entry = spec.kwargs.get("env_cfg_entry_point")
    chk("T1i 的 env_cfg_entry_point 就是 T1a2 的同一个类",
        entry is m.G1AmpWalkT1a2EnvCfg, True)
    print(f"      entry_point = {entry}")
    nonzeros = sorted(
        n for n in dir(m.G1AmpWalkT1a2EnvCfg().rewards)
        if getattr(getattr(m.G1AmpWalkT1a2EnvCfg().rewards, n, None), "weight", None) not in (None, 0)
    )
    chk("T1a2 非零奖励项 = 16（本臂不变）", len(nonzeros), 16)

    print()
    if FAIL:
        print(f"❌ T1i 校验失败：{len(FAIL)} 项 —— **不要发车**")
        for f in FAIL:
            print(f"   - {f}")
        return 1
    print("✅ T1i 校验全部通过（仅风格奖励口径改动）—— 可以发车")
    return 0


if __name__ == "__main__":
    sys.exit(main())
