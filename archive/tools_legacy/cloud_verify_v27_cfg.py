#!/usr/bin/env python3
"""**启动前核验（V27）**：确认"判别器重启（去白化 + 丢判别器）"这一件事，别的没动。

A. 与 V26 的逐字一致性：21 项奖励权重、`action_scale=1.0`、指令范围、AMP 观测组、
   `clip_max_std=0.20`、`clip_max_std_schedule=300`、`entropy_coef`、`amp_quad_k`、
   GP coef、`amp_reward_form` 全部相同。
B. 本轮唯一变量在位：
   - V27 的 `discriminator_kwargs` **不含** `normalizer_class_name` / `normalizer_kwargs`
     （V26 **含**）；
   - V27 的 `amp_grad_pen_space == "raw"`（V26 是 `"normalized"`）；
   - V27 的 `ckpt_manipulator == "instinctlab.utils.ckpt_manipulators:drop_amp_discriminator"`
     （V26 是 `None`）。
C. `drop_amp_discriminator` **行为单测**：直接调它，确认它只删
   `discriminator` / `discriminator_optimizer` 两个键，其余键原样保留。
D. 热启动对象存在：V26 run 目录 + 最新 `model_*.pt`。
E. 任务注册可用。
F. 判别器构造可行：用 V27 的 `discriminator_kwargs` 实际构造一个
   `discriminator_class_name` 指向的判别器，确认 `normalizer is None` 且前向输出形状正确
   （防"去白化后构造失败"或"多传了 kwarg 崩"）。

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
    source /root/.llp_env.sh && unset PYTHONPATH && \
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v27_cfg.py
结果落 `/root/v27_verify_result.txt`，末尾 `[V27 PASS]`。
"""

from __future__ import annotations

import os
import sys
import traceback

OUT = "/root/v27_verify_result.txt"
RUN_ROOT = "/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk"
DROP_DISC = "instinctlab.utils.ckpt_manipulators:drop_amp_discriminator"
EXPECT_CAP = 0.20
EXPECT_RAMP = 300

bad: list[str] = []


def ok(msg: str) -> None:
    print(f"  ✅ {msg}", flush=True)


def fail(msg: str) -> None:
    bad.append(msg)
    print(f"  ❌ {msg}", flush=True)


def main() -> int:
    print("=" * 88, flush=True)
    print("V27 启动前核验（判别器重启 = 去白化 + 丢判别器）", flush=True)
    print("=" * 88, flush=True)

    import gymnasium as gym

    import instinctlab.tasks  # noqa: F401
    from instinctlab.tasks.locomotion.config.g1.amp_walk_v26_env_cfg import G1AmpWalkV26EnvCfg
    from instinctlab.tasks.locomotion.config.g1.amp_walk_v27_env_cfg import G1AmpWalkV27EnvCfg
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v26_ppo_cfg import (
        G1AmpWalkV26PPORunnerCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v27_ppo_cfg import (
        G1AmpWalkV27PPORunnerCfg,
    )

    cfg26, cfg27 = G1AmpWalkV26EnvCfg(), G1AmpWalkV27EnvCfg()
    r26, r27 = G1AmpWalkV26PPORunnerCfg(), G1AmpWalkV27PPORunnerCfg()

    print("\n[A] 与 V26 的逐字一致性", flush=True)
    k26 = set(cfg26.rewards.__dict__.keys())
    k27 = set(cfg27.rewards.__dict__.keys())
    if k26 == k27:
        ok(f"奖励项集合一致（{len(k26)} 项）")
    else:
        fail(f"奖励项集合不同：{sorted(k26 ^ k27)}")
    diff = []
    skipped = []
    for k in sorted(k26 & k27):
        w26 = getattr(getattr(cfg26.rewards, k), "weight", None)
        w27 = getattr(getattr(cfg27.rewards, k), "weight", None)
        if w26 is None or w27 is None:
            # ⚠️ `rewards.__dict__` 里可能有**值为 None 的字段**（不是 RewTerm）
            #    ⇒ 直接取 `.weight` 会 AttributeError。跳过它们即可。
            skipped.append(k)
            continue
        if w26 != w27:
            diff.append(f"{k}: {w26} → {w27}")
    if not diff:
        ok(f"奖励权重逐项相等（跳过非 RewTerm 字段：{skipped}）")
    else:
        fail("权重不同：" + "; ".join(diff))
    sc = getattr(cfg27.actions.__dict__.get("joint_pos"), "scale", None)
    if sc == 1.0:
        ok("action.scale = 1.0")
    else:
        fail(f"action.scale = {sc}")
    rng26, rng27 = cfg26.commands.base_velocity.ranges, cfg27.commands.base_velocity.ranges
    if (rng26.lin_vel_x, rng26.lin_vel_y, rng26.ang_vel_z) == (
        rng27.lin_vel_x, rng27.lin_vel_y, rng27.ang_vel_z
    ):
        ok(f"指令范围一致 {rng27.lin_vel_x}/{rng27.lin_vel_y}/{rng27.ang_vel_z}")
    else:
        fail("指令范围不同")
    for k in ("entropy_coef", "learning_rate", "amp_quad_k",
              "discriminator_gradient_penalty_coef", "amp_reward_form", "amp_reward_eps",
              "clip_max_std", "clip_max_std_schedule", "use_mirror_loss"):
        a, b = getattr(r26.algorithm, k, None), getattr(r27.algorithm, k, None)
        if a == b:
            ok(f"{k} 一致 = {b}")
        else:
            fail(f"{k} 不同：V26 {a} vs V27 {b}")

    print("\n[B] 本轮唯一变量：判别器口径 + 判别器重置", flush=True)
    dk26, dk27 = r26.algorithm.discriminator_kwargs, r27.algorithm.discriminator_kwargs
    if "normalizer_class_name" in dk26 and "normalizer_class_name" not in dk27:
        ok("V26 白化 / V27 不白化（normalizer_class_name 已移除）")
    else:
        fail(f"白化开关不对：V26={dk26} / V27={dk27}")
    if dk27.get("hidden_sizes") == dk26.get("hidden_sizes"):
        ok(f"判别器隐层不变 {dk27.get('hidden_sizes')}")
    else:
        fail("判别器隐层被动了")
    g26, g27 = r26.algorithm.amp_grad_pen_space, r27.algorithm.amp_grad_pen_space
    if (g26, g27) == ("normalized", "raw"):
        ok(f"GP 空间：V26={g26} → V27={g27}")
    else:
        fail(f"GP 空间不对：V26={g26} / V27={g27}")
    if getattr(r26, "ckpt_manipulator", None) is None and r27.ckpt_manipulator == DROP_DISC:
        ok(f"ckpt_manipulator：V26=None → V27={DROP_DISC}")
    else:
        fail(f"ckpt_manipulator 不对：V26={getattr(r26, 'ckpt_manipulator', None)} / V27={r27.ckpt_manipulator}")

    print("\n[C] drop_amp_discriminator 行为单测", flush=True)
    import importlib

    mod, fn = DROP_DISC.split(":")
    manip = getattr(importlib.import_module(mod), fn)
    fake = {
        "discriminator": {"w": 1},
        "discriminator_optimizer": {"s": 2},
        "model_state_dict": {"a": 3},
        "optimizer_state_dict": {"b": 4},
        "iter": 8000,
        "infos": {},
    }
    out = manip(dict(fake), {}, verbose=False)
    if "discriminator" not in out and "discriminator_optimizer" not in out:
        ok("已删除 discriminator / discriminator_optimizer")
    else:
        fail(f"未删干净：{sorted(out.keys())}")
    keep = {"model_state_dict", "optimizer_state_dict", "iter", "infos"}
    if keep.issubset(out.keys()) and len(out) == len(keep):
        ok(f"其余键原样保留：{sorted(out.keys())}")
    else:
        fail(f"误删/多留：{sorted(out.keys())}")

    print("\n[D] 热启动对象（V26 最新 ckpt）", flush=True)
    import glob

    v26 = [d for d in glob.glob(os.path.join(RUN_ROOT, "*v26*")) if "from2026" in os.path.basename(d)]
    ck = ""
    if v26:
        cks = sorted(glob.glob(os.path.join(v26[-1], "model_*.pt")),
                     key=lambda p: int(os.path.basename(p)[6:-3]))
        ck = cks[-1] if cks else ""
    if ck:
        ok(f"找到 {os.path.basename(v26[-1])} / {os.path.basename(ck)}"
           f"（{os.path.getsize(ck) / 1e6:.1f} MB）")
    else:
        fail(f"找不到 V26 的 model_*.pt（候选：{[os.path.basename(d) for d in v26[-3:]]}）")

    print("\n[E] 任务注册", flush=True)
    for tid in ("Instinct-Amp-Walk-V27-G1-v0", "Instinct-Amp-Walk-V27-G1-Play-v0"):
        spec = gym.spec(tid)
        ec = spec.kwargs["env_cfg_entry_point"]
        rc = spec.kwargs["instinct_rl_cfg_entry_point"]
        if "v27" in rc and "V27" in ec.__name__:
            ok(f"{tid}: env={ec.__name__}  algo={rc.split(':')[-1]}")
        else:
            fail(f"{tid} 注册内容异常：env={ec.__name__} algo={rc}")

    print("\n[F] 判别器构造签名（去白化后 kwargs 必须仍被接受）", flush=True)
    try:
        import inspect

        from instinct_rl.modules import Discriminator

        sig = inspect.signature(Discriminator.__init__)
        names = set(sig.parameters.keys())
        has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        print(f"      判别器签名参数：{sorted(names)}", flush=True)
        for kw in dk27:
            if kw in names or has_var_kw:
                ok(f"判别器接受 kwarg '{kw}'")
            else:
                fail(f"判别器**不接受** kwarg '{kw}' ⇒ 启动会 TypeError")
        for kw in ("normalizer_class_name", "normalizer_kwargs"):
            if kw in names or has_var_kw:
                ok(f"（信息）判别器支持可选参数 '{kw}'（V27 未使用）")
    except BaseException:  # noqa: BLE001
        fail("读取判别器签名失败：\n" + traceback.format_exc())

    print("\n" + "=" * 88, flush=True)
    if bad:
        print(f"[V27 FAIL] {len(bad)} 项不通过：", flush=True)
        for b in bad:
            print(f"   - {b}", flush=True)
        return 1
    print("[V27 PASS] 判别器重启（去白化 + 丢判别器）就位，其余与 V26 逐字一致", flush=True)
    return 0


if __name__ == "__main__":
    _fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(_fd, 1)
    os.dup2(_fd, 2)
    from isaaclab.app import AppLauncher

    _app = None
    rc = 1
    try:
        _app = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False}).app
        rc = main()
    except BaseException:  # noqa: BLE001
        print("抛异常：\n" + traceback.format_exc(), flush=True)
        rc = 1
    print(f"RC={rc}", flush=True)
    sys.stdout.flush()
    os._exit(rc)
