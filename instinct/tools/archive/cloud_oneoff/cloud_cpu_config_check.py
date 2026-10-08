#!/usr/bin/env python3
"""CPU-only 云平台：**启动 Kit 后**验证配置接线（不需要 GPU 训练）。

背景
----
发现：本平台的 Kit 在**没有 GPU**时也能启动（`AppLauncher(headless=True)` 返回成功，
只有 PhysX/Warp 的 GPU 部分报错）。于是「配置实例化 + to_dict()」这类验证可以做，
它们本来是我标为「必须上云才能做、且必须有 GPU」的那一批。

> ⚠️ 这是**源码审计已确认**过的事情的实测复核：`isaaclab.utils.configclass.add_annotations`
> 会给**没有类型注解**的类属性从默认值推断类型并补注解，所以无注解的新字段
> （我们步骤⑥ 的 WASABI 参数全是这种）**确实会成为 dataclass 字段**、
> `to_dict()` 也**确实会**带上它们。本脚本把这条从「读源码得出结论」升级为「实测通过」。

跑法（在云上、instinct_xu 目录内）：
    /opt/conda/envs/isaaclab/bin/python /root/cloud_cpu_config_check.py --headless --device cpu
"""
from __future__ import annotations

import argparse
import sys
import traceback

# ⚠️ AppLauncher 必须最先（它会先起 Kit，之后才能 import 需要 pxr/carb 的模块）
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("\n" + "=" * 78)
print("Kit 已启动，开始配置接线验证")
print("=" * 78, flush=True)

_fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'✅' if cond else '❌'} {name}" + (f"  {detail}" if detail else ""), flush=True)
    if not cond:
        _fails.append(name)


try:
    import importlib

    print("\n[A] 外层依赖导入")
    for m in ("isaaclab.envs", "isaaclab.envs.mdp", "isaaclab.managers",
              "isaaclab.sensors", "instinct_rl", "instinctlab",
              "instinctlab.tasks.locomotion.config.g1"):
        try:
            importlib.import_module(m)
            print(f"  ✅ {m}")
        except BaseException as e:  # noqa: BLE001
            print(f"  ❌ {m}: {type(e).__name__}: {str(e)[:120]}")
            _fails.append(m)

    print("\n[B] 步骤⑥：算法/策略/runner 配置（AmpWalkAlgorithmCfg）")
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_ppo_cfg import (
        G1AmpWalkPPORunnerCfg,
    )
    runner_cfg = G1AmpWalkPPORunnerCfg()
    d = runner_cfg.to_dict()
    alg = d.get("algorithm", {})
    check("G1AmpWalkPPORunnerCfg() 可构造 + to_dict()",
          isinstance(d, dict) and bool(alg), f"{len(d)} 个顶层键")
    # ★ 这一组就是「@configclass 无注解字段」的实测复核
    expected = {
        "class_name": "instinctlab.utils.wrappers.instinct_rl:WasabiPPOAmpMix",
        "reward_mix_mode": "lerp",
        "style_lerp": 0.75,
        "disable_style_on_stand": False,
        "stand_speed_threshold": 0.1,
        "discriminator_reward_coef": 0.1,
        "discriminator_reward_type": "quad",
        "discriminator_loss_func": "MSELoss",
        "discriminator_class_name": "Discriminator",
        "auxiliary_reward_per_env_reward_coefs": [1.0],
        "clip_min_std": 0.05,
        "discriminator_gradient_penalty_coef": 5.0,
        "discriminator_optimizer_class_name": "AdamW",
        "discriminator_weight_decay_coef": 3.0e-4,
        "discriminator_logit_weight_decay_coef": 0.04,
    }
    for k, want in expected.items():
        got = alg.get(k, "<**缺失**>")
        check(f"algorithm.{k} 透传到 dict", got == want, f"= {got!r}")
    hk = alg.get("discriminator_kwargs")
    check("algorithm.discriminator_kwargs 含 hidden_sizes（MlpModel 必填）",
          isinstance(hk, dict) and hk.get("hidden_sizes") == [1024, 512], f"= {hk!r}")
    # 继承来的 PPO 超参也必须还在（证明继承没被破坏）
    for k, want in (("value_loss_coef", 1.0), ("clip_param", 0.2),
                    ("entropy_coef", 0.008), ("learning_rate", 1e-3),
                    ("gamma", 0.99), ("lam", 0.95), ("max_grad_norm", 1.0)):
        check(f"algorithm.{k}（继承自 AlgorithmCfg）", alg.get(k) == want, f"= {alg.get(k)!r}")
    pol = d.get("policy", {})
    check("policy.actor_hidden_dims = [512,256,128]",
          pol.get("actor_hidden_dims") == [512, 256, 128], f"= {pol.get('actor_hidden_dims')!r}")
    check("policy.init_noise_std = 1.0（clip_min_std 等价性的前提）",
          pol.get("init_noise_std") == 1.0, f"= {pol.get('init_noise_std')!r}")
    check("policy.class_name 仍是 ActorCritic",
          pol.get("class_name") == "ActorCritic", f"= {pol.get('class_name')!r}")
    check("normalizers 有 policy/critic 两组",
          set(d.get("normalizers", {})) == {"policy", "critic"}, f"= {list(d.get('normalizers', {}))}")
    check("run_name / experiment_name", d.get("run_name") == "step6_wasabi_amp_quad_c0.1_lerp0.75"
          and d.get("experiment_name") == "g1_amp_walk",
          f"run_name={d.get('run_name')!r}")

    print("\n[C] 步骤⑤：奖励集配置（G1AmpWalkRewardsCfg）")
    from instinctlab.tasks.locomotion.config.g1 import G1AmpWalkEnvCfg
    ecfg = G1AmpWalkEnvCfg()
    check("G1AmpWalkEnvCfg() 可构造", True)
    check("env.rewards 是 G1AmpWalkRewardsCfg（不是基类）",
          type(ecfg.rewards).__name__ == "G1AmpWalkRewardsCfg",
          f"= {type(ecfg.rewards).__name__}")
    names = {k for k in vars(ecfg.rewards) if not k.startswith("_")}
    print(f"     奖励项共 {len(names)} 个")
    want_w = {
        "track_lin_vel_xy_exp": 1.5, "track_ang_vel_z_exp": 1.2, "feet_slide": -0.25,
        "track_base_height": 1.5, "body_ang_vel_xy": 0.7,
        "track_default_pose_on_stand": -1.0, "feet_lateral_width": -10.0,
        "feet_crossing": -20.0,
    }
    for k, w in want_w.items():
        term = getattr(ecfg.rewards, k, None)
        check(f"rewards.{k} 存在且 weight={w}",
              term is not None and getattr(term, "weight", None) == w,
              f"= {getattr(term, 'weight', None)!r}")
    for k in ("termination_penalty", "stand_still", "flat_orientation_l2",
              "feet_air_time", "action_rate_l2", "lin_vel_z_l2", "dof_pos_limits"):
        term = getattr(ecfg.rewards, k, None)
        check(f"rewards.{k} 仍被继承（未丢）", term is not None,
              f"weight={getattr(term, 'weight', None)!r}")
    check("feet_lateral_width 的 sep_floor = 0.145（实测标定值）",
          getattr(ecfg.rewards.feet_lateral_width, "params", {}).get("sep_floor") == 0.145)
    check("feet_crossing 的 gate_lateral = True（有意偏离 AMP_mjlab）",
          getattr(ecfg.rewards.feet_crossing, "params", {}).get("gate_lateral") is True)

    print("\n[D] 步骤③/④：命令与观测组接线")
    check("commands.base_velocity 用 DirectionalVelocityCommand",
          type(ecfg.commands.base_velocity).__name__ == "DirectionalVelocityCommandCfg",
          f"= {type(ecfg.commands.base_velocity).__name__}")
    check("heading_command=False（否则 wz 采样会被航向环重算）",
          ecfg.commands.base_velocity.heading_command is False)
    check("观测组含 amp_policy / amp_reference",
          all(hasattr(ecfg.observations, g) for g in ("policy", "critic", "amp_policy", "amp_reference")),
          f"= {[k for k in vars(ecfg.observations) if not k.startswith('_')]}")
    check("场景含 robot_reference / motion_reference",
          all(hasattr(ecfg.scene, g) for g in ("robot_reference", "motion_reference")),
          f"= {[k for k in vars(ecfg.scene) if not k.startswith('_')]}")

    print("\n[E] ★ 建环境：会跑 G1AmpWalkTeacherEnv._validate_teacher_contract()")
    import gymnasium as gym
    import instinctlab.tasks  # noqa: F401  （触发注册）
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    task_id = "Instinct-Amp-Walk-G1-v0"
    entry = gym.registry.get(task_id)
    check(f"任务 {task_id} 已注册", entry is not None)
    if entry is not None:
        env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
        env_cfg.scene.num_envs = 8
        env_cfg.sim.device = "cpu"
        print("     正在构造 env（16 envs → 用 8；device=cpu）...", flush=True)
        env = gym.make(task_id, cfg=env_cfg).unwrapped
        print(f"     env 构造成功: {type(env).__name__}")
        check("env 类型是 G1AmpWalkTeacherEnv", type(env).__name__ == "G1AmpWalkTeacherEnv",
              f"= {type(env).__name__}")
        # 契约校验器在 load_managers() 里跑，失败会直接 RuntimeError；
        # 能走到这里说明动作空间/时序/执行器/命令范围全部对上了。
        check("契约校验通过（走到这里即说明 _validate_teacher_contract 没抛错）", True)
        obs, _ = env.reset()
        print("     已 reset，观测分组:", {k: tuple(v.shape) for k, v in obs.items()} if isinstance(obs, dict) else type(obs))
        for _ in range(2):
            obs, rew, term, trunc, extras = env.step(env.action_space.sample())
        print(f"     走完 2 步；reward 形状 = {tuple(rew.shape) if hasattr(rew, 'shape') else rew}")
        step_info = extras.get("step", {})
        check("step info 里有 amp_style_reward（步骤⑥ 的日志钩子）",
              "amp_style_reward" in step_info, f"keys={sorted(step_info)[:8]}")
        check("step info 里有 discriminator_reward",
              "discriminator_reward" in step_info)
        if "amp_style_reward" in step_info:
            v = step_info["amp_style_reward"]
            print(f"     amp_style_reward = {float(v.flatten()[0]) if hasattr(v, 'flatten') else v:.4f}"
                  f"（quad 形式，值域 [0,1]）")
        log = extras.get("log", {})
        check("extras['log'] 里有 foot_sep_mean（步骤⑤ 的标定诊断量）",
              "Metrics/foot_sep_mean" in log, f"keys={sorted(log)[:6]}")
        env.close()

except BaseException:  # noqa: BLE001
    traceback.print_exc()
    _fails.append("异常中断")

print("\n" + "=" * 78)
if _fails:
    print(f"❌ 未通过 {len(_fails)} 项：")
    for f in _fails:
        print(f"   - {f}")
else:
    print("✅ 全部通过")
print("=" * 78, flush=True)

simulation_app.close()
sys.exit(1 if _fails else 0)
