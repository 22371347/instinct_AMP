#!/usr/bin/env python3
"""**风格臂单变量校验（通用）**：证明"臂 cfg 相对基准 cfg 只有 `style_lerp` 一个数值改动"。

用法（云端）
------------
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \\
    source /root/.llp_env.sh && unset PYTHONPATH && \\
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_style_arm.py \\
        instinctlab.tasks.locomotion.config.g1.agents.amp_walk_t1a2_ppo_cfg:G1AmpWalkT1a2PPORunnerCfg \\
        instinctlab.tasks.locomotion.config.g1.agents.amp_walk_t1c_ppo_cfg:G1AmpWalkT1cPPORunnerCfg \\
        0.9

核心断言（比"逐项列期望值"更强）
--------------------------------
把**基准**与**实验臂**两套 runner cfg 现场建出来，**逐字段 diff**，断言差异集合
**恰好**为 `{"algorithm.style_lerp", "run_name"}` —— 连"我没列到的字段被改了"也能抓住。
这是在 T1a 那次"权重对齐但实现没对齐"事故之后立的规矩：**单变量必须有机器可验的证明**。

退出码：0=通过；1=有 FAIL；2=用法/导入错误。
"""

from __future__ import annotations

import sys

from isaaclab.app import AppLauncher  # noqa: E402

_APP = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False}).app

import dataclasses  # noqa: E402
import importlib  # noqa: E402

FAIL: list[str] = []


def chk(name: str, got, want, tol: float | None = 1e-12) -> None:
    if tol is not None and isinstance(got, (int, float)) and isinstance(want, (int, float)):
        ok = abs(float(got) - float(want)) <= float(tol)
    else:
        ok = got == want
    print(f"  {'OK  ' if ok else 'FAIL'} {name:52s} got={got!r}  want={want!r}")
    if not ok:
        FAIL.append(name)


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
    """只比"可比较的叶子"；子配置对象统一记 `<obj>`（它们由调用方单独比）。"""
    out = {}
    for n in _fields(obj):
        v = getattr(obj, n, None)
        if isinstance(v, (int, float, str, bool, type(None), list, tuple, dict)):
            out[n] = repr(v)
        else:
            out[n] = "<obj>"
    return out


def _diff(a, b, prefix: str = "") -> list[str]:
    pa, pb = _prim(a), _prim(b)
    return [f"{prefix}{k}" for k in sorted(set(pa) | set(pb)) if pa.get(k) != pb.get(k)]


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    base_spec, arm_spec, expect = sys.argv[1], sys.argv[2], float(sys.argv[3])

    try:
        bmod, _, bcls = base_spec.partition(":")
        amod, _, acls = arm_spec.partition(":")
        base = getattr(importlib.import_module(bmod), bcls)()
        arm = getattr(importlib.import_module(amod), acls)()
    except Exception as exc:
        print(f"[FATAL] 构建失败: {type(exc).__name__}: {exc}")
        return 2

    print("=" * 78)
    print(f"基准: {base_spec}")
    print(f"臂  : {arm_spec}   期望 style_lerp = {expect}")
    print("=" * 78)

    print("\n[1] ⭐ 单变量证明（全字段 diff）")
    diffs = _diff(base, arm) + _diff(base.algorithm, arm.algorithm, "algorithm.") + _diff(
        base.policy, arm.policy, "policy."
    )
    chk("差异集合必须恰为 ['algorithm.style_lerp', 'run_name']", sorted(diffs),
        ["algorithm.style_lerp", "run_name"])
    for d in sorted(diffs):
        print(f"      · {d}")

    print("\n[2] 唯一数值改动")
    chk("arm.style_lerp", float(getattr(arm.algorithm, "style_lerp")), expect)
    chk("base.style_lerp（基准应为 1.0 = 风格恰为零）", float(getattr(base.algorithm, "style_lerp")), 1.0)

    print("\n[3] lerp 公式 / 热启动前提")
    chk("reward_mix_mode == 'lerp'", str(getattr(arm.algorithm, "reward_mix_mode")), "lerp")
    chk("discriminator_reward_coef != 0", float(getattr(arm.algorithm, "discriminator_reward_coef")) != 0.0, True)
    chk("disable_style_on_stand", bool(getattr(arm.algorithm, "disable_style_on_stand")), True)
    chk("use_mirror_loss（交付要求②）", bool(getattr(arm.algorithm, "use_mirror_loss")), True)
    chk("mirror_loss_coeff", float(getattr(arm.algorithm, "mirror_loss_coeff")), 0.1)
    chk("init_noise_std（沿用 T1a2/T0）", float(getattr(arm.policy, "init_noise_std")), 1.0)
    chk("clip_max_std（关闭）", getattr(arm.algorithm, "clip_max_std"), None)
    chk("save_interval", int(getattr(arm, "save_interval")), 500)

    print()
    if FAIL:
        print(f"❌ 风格臂校验失败：{len(FAIL)} 项 —— **不要发车**")
        for f in FAIL:
            print(f"   - {f}")
        return 1
    print(f"✅ 校验全部通过（单变量：仅 style_lerp 1.0 → {expect}）—— 可以发车")
    return 0


if __name__ == "__main__":
    sys.exit(main())
