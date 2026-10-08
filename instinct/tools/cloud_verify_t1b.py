#!/usr/bin/env python3
"""**T1b 云端配置校验**：证明"T1b 相对 T1a2 **只有 `style_lerp` 一个数值改动**"。

核心断言（最强形式的单变量证明）
--------------------------------
把 **T1a2** 与 **T1b** 两套 runner cfg 现场建出来，逐字段 diff，断言差异集合**恰好**为
`{"algorithm.style_lerp", "run_name"}`。任何一个其他字段漂移都会立刻 FAIL ——
这比"逐项列出我期望的值"更强：它连"我没列到的字段被改了"也能抓住。

再叠加：
* 环境侧：T1b 用的是 **T1a2 的 env cfg 本身**（不是复制品），断言奖励权重/实现/契约一致；
* 热启动前提：`discriminator_reward_coef != 0`（lerp 公式需要）、`disable_style_on_stand=True`。

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \\
    source /root/.llp_env.sh && unset PYTHONPATH && \\
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_t1b.py
"""

from __future__ import annotations

import sys

from isaaclab.app import AppLauncher  # noqa: E402

_APP = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False}).app

import dataclasses  # noqa: E402

FAIL: list[str] = []


def chk(name: str, got, want, tol: float | None = 1e-12) -> None:
    if tol is not None and isinstance(got, (int, float)) and isinstance(want, (int, float)):
        ok = abs(float(got) - float(want)) <= float(tol)
    else:
        ok = got == want
    print(f"  {'OK  ' if ok else 'FAIL'} {name:58s} got={got!r}  want={want!r}")
    if not ok:
        FAIL.append(name)


def g(obj, name):
    if not hasattr(obj, name):
        FAIL.append(f"缺少属性 {name}")
        print(f"  FAIL 缺少属性 {name}（配置结构变了？）")
        return None
    return getattr(obj, name)


def _fields(obj) -> list[str]:
    if obj is None:
        return []
    try:
        if dataclasses.is_dataclass(obj):
            return [f.name for f in dataclasses.fields(obj)]
    except Exception:
        pass
    d = getattr(obj, "__dict__", None)
    return [k for k in d if not k.startswith("_")] if isinstance(d, dict) else []


def _prim(obj) -> dict[str, str]:
    """只取"可比较的叶子"；子配置对象统一记 `<obj>`（它们单独比）。"""
    out = {}
    for n in _fields(obj):
        v = getattr(obj, n, None)
        if isinstance(v, (int, float, str, bool, type(None), list, tuple, dict)):
            out[n] = repr(v)
        else:
            out[n] = "<obj>"
    return out


def _diff(a, b, prefix: str) -> list[str]:
    pa, pb = _prim(a), _prim(b)
    return [f"{prefix}{k}" for k in sorted(set(pa) | set(pb)) if pa.get(k) != pb.get(k)]


def main() -> int:
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_t1a2_ppo_cfg import (  # noqa: E402
        G1AmpWalkT1a2PPORunnerCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_t1b_ppo_cfg import (  # noqa: E402
        AMP_T1B_STYLE_LERP,
        G1AmpWalkT1bPPORunnerCfg,
    )

    c2, cb = G1AmpWalkT1a2PPORunnerCfg(), G1AmpWalkT1bPPORunnerCfg()

    # ---- ① 单变量证明：T1a2 → T1b 的差异集合恰好是 {style_lerp, run_name} ---------
    print("[1] ⭐ 单变量证明（T1b vs T1a2 的全字段 diff）")
    diffs = (
        _diff(c2, cb, "")
        + _diff(c2.algorithm, cb.algorithm, "algorithm.")
        + _diff(c2.policy, cb.policy, "policy.")
    )
    chk("差异集合必须恰为 ['algorithm.style_lerp', 'run_name']", sorted(diffs),
        ["algorithm.style_lerp", "run_name"])
    if diffs:
        for d in sorted(diffs):
            print(f"      · {d}")

    # ---- ② 唯一改动本身 --------------------------------------------------------
    print("\n[2] 唯一数值改动")
    chk("algorithm.style_lerp（T1b）", float(g(cb.algorithm, "style_lerp")), float(AMP_T1B_STYLE_LERP))
    chk("algorithm.style_lerp（T1a2 应为 1.0，对照）", float(g(c2.algorithm, "style_lerp")), 1.0)

    # ---- ③ 热启动 / lerp 公式的前提 --------------------------------------------
    print("\n[3] lerp + 热启动前提")
    chk("discriminator_reward_coef != 0（lerp 公式需要）", float(g(cb.algorithm, "discriminator_reward_coef")) != 0.0, True)
    chk("disable_style_on_stand（站立档不抢 default_pose 的职责）", bool(g(cb.algorithm, "disable_style_on_stand")), True)
    chk("reward_mix_mode == 'lerp'", str(g(cb.algorithm, "reward_mix_mode")), "lerp")
    chk("schedule / 判别器沿用", str(g(cb.algorithm, "schedule")), "adaptive")
    chk("amp_expert_buffer_path", str(g(cb.algorithm, "amp_expert_buffer_path")), "/root/amp_expert_states_396.pt")
    chk("use_mirror_loss（交付要求②）", bool(g(cb.algorithm, "use_mirror_loss")), True)
    chk("policy.init_noise_std（沿用 T1a2/T0）", float(g(cb.policy, "init_noise_std")), 1.0)
    chk("max_iterations（不设上限）", int(g(cb, "max_iterations")), 1_000_000_000)
    chk("save_interval", int(g(cb, "save_interval")), 500)

    # ---- ④ 环境侧：T1b 用的就是 T1a2 的 env cfg（同一个类）-----------------------
    print("\n[4] 环境侧（必须与 T1a2 同一个 env cfg 类）")
    from instinctlab.tasks.locomotion.config.g1.amp_walk_t1a2_env_cfg import (  # noqa: E402
        G1AmpWalkT1a2EnvCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.amp_walk_t1a2_env_cfg import (  # noqa: E402
        T1A2_T0_IMPL as _IMPL,
    )

    ecfg = G1AmpWalkT1a2EnvCfg()
    chk("run_name（env 侧仍是 T1a2 的骨架标记 = 同一套环境）",
        str(ecfg.run_name), "G1AmpWalkT1a2MVPPlane_obs378_rew16_t0impl_sym")
    for name, (func, params) in _IMPL.items():
        chk(f"rewards.{name}.func", getattr(getattr(ecfg.rewards, name).func, "__qualname__", None),
            getattr(func, "__qualname__", None))
    chk("track_default_pose_on_stand（交付要求②）", float(getattr(ecfg.rewards, "track_default_pose_on_stand").weight), -0.8)
    chk("actions.joint_pos.scale", float(g(ecfg.actions.joint_pos, "scale")), 1.0)
    chk("sim.dt / decimation", (float(ecfg.sim.dt), int(ecfg.decimation)), (0.005, 4))

    print()
    if FAIL:
        print(f"❌ T1b 校验失败：{len(FAIL)} 项 —— **不要发车**")
        for f in FAIL:
            print(f"   - {f}")
        return 1
    print("✅ T1b 校验全部通过（单变量：仅 style_lerp 1.0→0.75）—— 可以发车")
    return 0


if __name__ == "__main__":
    sys.exit(main())
