#!/usr/bin/env python3
"""**云端配置预检（V33）**：起训前把三臂的 env / agent cfg 全部断言一遍（**秒级，不建 sim**）。

为什么值得单独写一个
--------------------
V33 的变量跨了两个层面（env 侧的观测组 / 奖励权重、agent 侧的判别器配方），而祖先链的
`__post_init__` 有**覆盖观测组**的前科（`_patch_amp_history` 会把 `amp_policy` 整个冲掉）。
cfg 层面的错误如果留到训起来才发现，就是几十分钟 GPU 白烧 —— 所以这里在**建 cfg 时**就炸。

检查项（三臂各自）：
  * 观测组类型 + 是否带 `anchor_gravity_b`（S 臂要带、A-cold 不能带）
  * `commands.base_velocity.side_vel_min == 0.15` 且 **> low_vel**
  * `rewards.track_lin_vel_xy_exp.func` 是**无损诊断版**、权重仍是 3.0
  * S1 的 `feet_air_time.weight == 2.0` / `joint_deviation_arms.weight == 0.0`；
    S0 / A-cold 必须是 4.0 / −0.1
  * agent：`amp_grad_pen_space`、normalizer 键、`amp_normalizer_expert_only`、
    `amp_expert_buffer_path`、`ckpt_manipulator`（三臂都必须是 `drop_amp_discriminator`）
  * AMP 状态维度常量：390（A-cold）/ 396（S）
  * 阈值常量：`style_lerp=0.75`、`discriminator_reward_coef=1.0`、GP=10.0、`amp_quad_k=0.25`

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
    source /root/.llp_env.sh && unset PYTHONPATH && \
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v33_cfg.py

⚠️ **必须起 `AppLauncher`**（~1 分钟）：`import instinctlab.tasks` 会走到 `isaaclab.envs`，
    而 `isaaclab` 的子包路径是在 app 启动时才挂上的 —— 不起 app 会直接
    `ModuleNotFoundError: No module named 'isaaclab.envs'`（2026-10-06 踩过）。
    所以本脚本虽然不建 sim，仍然包在 AppLauncher 里跑。
"""

from __future__ import annotations

import os
import sys
import traceback

OUT = os.environ.get("V33_VERIFY_OUT", "/root/v33_verify_result.txt")

_fails: list[str] = []
_oks: list[str] = []


def ok(msg: str) -> None:
    _oks.append(msg)
    print(f"  ✅ {msg}", flush=True)


def bad(msg: str) -> None:
    _fails.append(msg)
    print(f"  ❌ {msg}", flush=True)


def eq(label: str, actual, expect) -> None:
    if actual == expect:
        ok(f"{label} = {actual!r}")
    else:
        bad(f"{label} = {actual!r}（期望 {expect!r}）")


def main() -> int:
    print("=" * 88, flush=True)
    print("V33 配置预检（三臂）", flush=True)
    print("=" * 88, flush=True)

    import instinctlab.tasks  # noqa: F401
    from instinctlab.tasks.locomotion.mdp import track_lin_vel_xy_axis_diag
    from instinctlab.tasks.locomotion.config.g1.amp_walk_v33_env_cfg import (
        AMP_WALK_V33_FRAME_SIZE,
        AMP_WALK_V33_SIDE_VEL_MIN,
        AMP_WALK_V33_STATE_SIZE,
        AmpV33PolicyStateObsCfg,
        AmpV33ReferenceStateObsCfg,
        G1AmpWalkV33AColdEnvCfg,
        G1AmpWalkV33S1EnvCfg,
        G1AmpWalkV33S2EnvCfg,
        G1AmpWalkV33S3EnvCfg,
        G1AmpWalkV33S4EnvCfg,
        G1AmpWalkV33S5EnvCfg,
        G1AmpWalkV33S6EnvCfg,
        G1AmpWalkV33S7EnvCfg,
        G1AmpWalkV33S8EnvCfg,
        G1AmpWalkV33SEnvCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.amp_walk_v20_env_cfg import (
        AmpV20PolicyStateObsCfg,
        AmpV20ReferenceStateObsCfg,
    )

    print("\n[A] 常量", flush=True)
    eq("AMP_WALK_V33_FRAME_SIZE", AMP_WALK_V33_FRAME_SIZE, 198)
    eq("AMP_WALK_V33_STATE_SIZE", AMP_WALK_V33_STATE_SIZE, 396)
    eq("AMP_WALK_V33_SIDE_VEL_MIN", AMP_WALK_V33_SIDE_VEL_MIN, 0.15)

    # ------------------------------------------------------------------ env 侧
    arms = (
        ("S0", G1AmpWalkV33SEnvCfg(), True, 4.0, -0.1, 3.0, 1.0),
        ("S1", G1AmpWalkV33S1EnvCfg(), True, 2.0, 0.0, 3.0, 1.0),
        ("S2", G1AmpWalkV33S2EnvCfg(), True, 4.0, -0.1, 3.0, 1.0),
        ("S3", G1AmpWalkV33S3EnvCfg(), True, 2.0, 0.0, 2.0, 1.0),
        ("S4", G1AmpWalkV33S4EnvCfg(), True, 0.0, 0.0, 3.0, 1.5),
        ("S5", G1AmpWalkV33S5EnvCfg(), True, 1.0, 0.0, 3.0, 1.5),
        ("S6", G1AmpWalkV33S6EnvCfg(), True, 0.0, 0.0, 3.0, 1.5),
        ("S7", G1AmpWalkV33S7EnvCfg(), True, 1.0, 0.0, 3.0, 1.5),
        ("S8", G1AmpWalkV33S8EnvCfg(), True, 1.0, 0.0, 3.0, 1.5),
        ("A-cold", G1AmpWalkV33AColdEnvCfg(), False, 4.0, -0.1, 3.0, 1.0),
    )
    for name, cfg, with_grav, air_w, arms_w, track_w, ang_w in arms:
        print(f"\n[B] env cfg：{name}", flush=True)
        # 观测组
        ap, ar = cfg.observations.amp_policy, cfg.observations.amp_reference
        eq(f"{name} amp_policy 类型", type(ap).__name__, "AmpV33PolicyStateObsCfg" if with_grav else "AmpV20PolicyStateObsCfg")
        eq(f"{name} amp_reference 类型", type(ar).__name__, "AmpV33ReferenceStateObsCfg" if with_grav else "AmpV20ReferenceStateObsCfg")
        eq(f"{name} 策略侧带 anchor_gravity_b", isinstance(ap, AmpV33PolicyStateObsCfg), with_grav)
        eq(f"{name} 参考侧带 anchor_gravity_b", isinstance(ar, AmpV33ReferenceStateObsCfg), with_grav)
        # 两侧的项名必须一一对应（判别器把两侧当同一套状态比较）
        # ⚠️ 用**实例级** `hasattr`：`@configclass` 的字段不一定留在类的 `__dict__` 里
        #    （`hasattr(type(x), 字段名)` 会误报 False，2026-10-06 踩过）。
        if with_grav:
            eq(
                f"{name} 策略侧实例有 anchor_gravity_b",
                hasattr(ap, "anchor_gravity_b"),
                True,
            )
            eq(
                f"{name} 参考侧实例有 anchor_gravity_b",
                hasattr(ar, "anchor_gravity_b"),
                True,
            )
        # 指令
        cmd = cfg.commands.base_velocity
        eq(f"{name} side_vel_min", cmd.side_vel_min, 0.15)
        if not (cmd.side_vel_min > float(cmd.low_vel)):
            bad(f"{name} side_vel_min({cmd.side_vel_min}) 未严格大于 low_vel({cmd.low_vel})")
        # 跟踪项已换成无损诊断版
        t = cfg.rewards.track_lin_vel_xy_exp
        eq(f"{name} track_lin_vel_xy_exp.func 是诊断版", t.func is track_lin_vel_xy_axis_diag, True)
        eq(f"{name} track_lin_vel_xy_exp.weight", t.weight, track_w)
        eq(f"{name} track_lin_vel_xy_exp.params", dict(t.params), {"command_name": "base_velocity", "std": 0.5})
        eq(f"{name} track_ang_vel_z_exp.weight", cfg.rewards.track_ang_vel_z_exp.weight, ang_w)
        # 脚手架
        eq(f"{name} feet_air_time.weight", cfg.rewards.feet_air_time.weight, air_w)
        eq(f"{name} joint_deviation_arms.weight", cfg.rewards.joint_deviation_arms.weight, arms_w)
        # V31 的单变量必须保持
        eq(f"{name} flat_orientation_l2.weight", cfg.rewards.flat_orientation_l2.weight, -4.0)

    # ------------------------------------------------------------------ agent 侧
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v33acold_ppo_cfg import (
        G1AmpWalkV33AColdPPORunnerCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v33s2_ppo_cfg import (
        AMP_V33S2_NORMALIZER_EPS,
        G1AmpWalkV33S2PPORunnerCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v33s3_ppo_cfg import (
        G1AmpWalkV33S3PPORunnerCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v33s4_ppo_cfg import (
        AMP_V33S4_STYLE_LERP,
        G1AmpWalkV33S4PPORunnerCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v33s5_ppo_cfg import (
        AMP_V33S5_STYLE_LERP,
        G1AmpWalkV33S5PPORunnerCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v33s6_ppo_cfg import (
        AMP_V33S6_DISC_REWARD_COEF,
        G1AmpWalkV33S6PPORunnerCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v33s7_ppo_cfg import (
        G1AmpWalkV33S7PPORunnerCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v33s8_ppo_cfg import (
        G1AmpWalkV33S8PPORunnerCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v33s_ppo_cfg import (
        AMP_V33S_EXPERT_BUFFER,
        AMP_V33S_NORMALIZER_EPS,
        G1AmpWalkV33SPPORunnerCfg,
    )

    _DROP = "instinctlab.utils.ckpt_manipulators:drop_amp_discriminator"
    # 末 5 位 = 期望的判别器口径：GP 系数 / loss 级 wd / logit wd / 优化器 trunk wd / head wd
    #   （所有旧臂 = 10.0 / 0 / 0 / 1e-3 / 1e-2；**S7/S8 = 5.0 / 3e-4 / 0.04 / 3e-4 / 3e-4**）
    # 末位 = 期望的 `algorithm.schedule`（**只有 S8 = adaptive**）
    _OLD = (10.0, 0.0, 0.0, 1e-3, 1e-2)
    _S7 = (5.0, 3.0e-4, 0.04, 3.0e-4, 3.0e-4)
    for name, acfg, white, gp_space, expert, exp_eps, exp_drop, exp_lerp, exp_coef, exp_disc, exp_sched in (
        ("S0/S1", G1AmpWalkV33SPPORunnerCfg(), True, "normalized", AMP_V33S_EXPERT_BUFFER, AMP_V33S_NORMALIZER_EPS, _DROP, 0.75, 1.0, _OLD, "fixed"),
        ("S2", G1AmpWalkV33S2PPORunnerCfg(), True, "normalized", AMP_V33S_EXPERT_BUFFER, AMP_V33S2_NORMALIZER_EPS, _DROP, 0.75, 1.0, _OLD, "fixed"),
        ("S3", G1AmpWalkV33S3PPORunnerCfg(), False, "raw", AMP_V33S_EXPERT_BUFFER, None, _DROP, 0.75, 1.0, _OLD, "fixed"),
        ("S4", G1AmpWalkV33S4PPORunnerCfg(), False, "raw", AMP_V33S_EXPERT_BUFFER, None, None, AMP_V33S4_STYLE_LERP, 1.0, _OLD, "fixed"),
        ("S5", G1AmpWalkV33S5PPORunnerCfg(), False, "raw", AMP_V33S_EXPERT_BUFFER, None, None, AMP_V33S5_STYLE_LERP, 1.0, _OLD, "fixed"),
        ("S6", G1AmpWalkV33S6PPORunnerCfg(), False, "raw", AMP_V33S_EXPERT_BUFFER, None, None, AMP_V33S4_STYLE_LERP, AMP_V33S6_DISC_REWARD_COEF, _OLD, "fixed"),
        ("S7", G1AmpWalkV33S7PPORunnerCfg(), False, "raw", AMP_V33S_EXPERT_BUFFER, None, _DROP, AMP_V33S5_STYLE_LERP, 1.0, _S7, "fixed"),
        ("S8", G1AmpWalkV33S8PPORunnerCfg(), False, "raw", AMP_V33S_EXPERT_BUFFER, None, None, AMP_V33S5_STYLE_LERP, 4.0, _S7, "adaptive"),
        ("A-cold", G1AmpWalkV33AColdPPORunnerCfg(), False, "raw", "/root/amp_expert_states.pt", None, _DROP, 0.75, 1.0, _OLD, "fixed"),
    ):
        print(f"\n[C] agent cfg：{name}", flush=True)
        a = acfg.algorithm
        has_norm = "normalizer_class_name" in a.discriminator_kwargs
        eq(f"{name} 白化键存在", has_norm, white)
        if white:
            eq(f"{name} normalizer_class_name", a.discriminator_kwargs["normalizer_class_name"], "EmpiricalNormalization")
            eq(f"{name} normalizer eps", a.discriminator_kwargs["normalizer_kwargs"]["eps"], exp_eps)
            eq(f"{name} normalizer until", a.discriminator_kwargs["normalizer_kwargs"]["until"], None)
        eq(f"{name} hidden_sizes", list(a.discriminator_kwargs.get("hidden_sizes", [])), [1024, 512])
        eq(f"{name} nonlinearity", a.discriminator_kwargs.get("nonlinearity"), "ReLU")
        eq(f"{name} amp_grad_pen_space", a.amp_grad_pen_space, gp_space)
        eq(f"{name} amp_normalizer_expert_only", a.amp_normalizer_expert_only, True)
        eq(f"{name} amp_expert_buffer_path", a.amp_expert_buffer_path, expert)
        eq(f"{name} ckpt_manipulator", acfg.ckpt_manipulator, exp_drop)
        eq(f"{name} discriminator_gradient_penalty_coef", a.discriminator_gradient_penalty_coef, exp_disc[0])
        eq(f"{name} discriminator_weight_decay_coef", a.discriminator_weight_decay_coef, exp_disc[1])
        eq(f"{name} discriminator_logit_weight_decay_coef", a.discriminator_logit_weight_decay_coef, exp_disc[2])
        eq(f"{name} discriminator_gp_expert_only", a.discriminator_gp_expert_only, True)
        eq(f"{name} discriminator_reward_coef", a.discriminator_reward_coef, exp_coef)
        eq(f"{name} style_lerp", a.style_lerp, exp_lerp)
        eq(f"{name} schedule", a.schedule, exp_sched)
        eq(f"{name} amp_reward_form", a.amp_reward_form, "bounded_quad")
        eq(f"{name} amp_quad_k", a.amp_quad_k, 0.25)
        eq(f"{name} discriminator_optimizer_trunk_weight_decay", a.discriminator_optimizer_trunk_weight_decay, exp_disc[3])
        eq(f"{name} discriminator_optimizer_head_weight_decay", a.discriminator_optimizer_head_weight_decay, exp_disc[4])

    # ------------------------------------------------------------------ task 注册
    print("\n[D] gym 注册", flush=True)
    import importlib

    import gymnasium as gym

    for tid in (
        "Instinct-Amp-Walk-V33S-G1-v0",
        "Instinct-Amp-Walk-V33S1-G1-v0",
        "Instinct-Amp-Walk-V33S2-G1-v0",
        "Instinct-Amp-Walk-V33S3-G1-v0",
        "Instinct-Amp-Walk-V33S4-G1-v0",
        "Instinct-Amp-Walk-V33ACold-G1-v0",
    ):
        try:
            spec = gym.spec(tid)
            env_cfg_cls = spec.kwargs["env_cfg_entry_point"]
            entry = spec.kwargs["instinct_rl_cfg_entry_point"]
            ok(f"注册 {tid}：env_cfg={env_cfg_cls.__name__}")
            mod_name, cls_name = entry.split(":")
            runner_cfg = getattr(importlib.import_module(mod_name), cls_name)
            runner_cfg()  # 真建一次（与 runner 的路径一致）⇒ 能建出来才算过
            ok(f"  agent cfg 可构建：{mod_name.split('.')[-1]}:{cls_name}")
        except Exception as exc:  # noqa: BLE001
            bad(f"注册 {tid} 失败：{type(exc).__name__}: {exc}")

    print("\n" + "=" * 88, flush=True)
    print(f"结果：{len(_oks)} 通过 / {len(_fails)} 失败", flush=True)
    for m in _fails:
        print(f"  ❌ {m}", flush=True)
    print("=" * 88, flush=True)
    return 1 if _fails else 0


if __name__ == "__main__":
    _fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(_fd, 1)
    os.dup2(_fd, 2)
    print("===== V33 配置预检（需要 AppLauncher）=====", flush=True)

    from isaaclab.app import AppLauncher

    _rc = 1
    try:
        # ⚠️ 必须起 app：`isaaclab.envs` 只有在 app 启动后才在 sys.path 里
        _app = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False}).app
        _rc = main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        _rc = 1
    print(f"[V33-VERIFY {'PASS' if _rc == 0 else 'FAIL'}]", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # ⚠️ 不调 `_app.close()`（在这个环境里会挂住不返回）⇒ 直接退出，OS 回收
    os._exit(_rc)
