#!/usr/bin/env python3
"""打印 InstinctLab G1 任务的 **Action Contract** 全项（供「契约固定」用）。

一次性钉死下面这些（都从**运行中的 env 对象**里读，不是从配置文字推断）：

  1. 动作输出维度 / 每一维含义（关节名顺序）
  2. action → joint target 的**确切公式**（scale / offset / clip / squash）
  3. default joint position offset 的实际值（兼容性）
  4. control decimation / policy Hz / sim dt
  5. actuator / PD target 的解释（stiffness / damping / effort_limit / 驱动模型）
  6. 观测组各段维度（确认与策略输入一致）
  7. 是否对动作做 tanh/squash（读 actor 网络结构）

用法：
  python cloud_probe_action_contract.py --task Instinct-Amp-Walk-G1-v0
"""
import argparse
import json

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--task", default="Instinct-Amp-Walk-G1-v0")
ap.add_argument("--term", default="joint_pos", help="动作项名")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import instinctlab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env = gym.make(args.task, cfg=cfg).unwrapped

def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

# ---------------------------------------------------------------- 1. 时序
sec("1. 时序 / control decimation / policy Hz")
print("sim.dt                    =", env.cfg.sim.dt)
print("decimation                =", env.cfg.decimation)
print("step_dt (policy 周期)      =", env.step_dt, "→ policy Hz =", 1.0 / env.step_dt)
print("episode_length_s          =", env.cfg.episode_length_s)
print("max_episode_length (steps)=", env.max_episode_length)

# ---------------------------------------------------------------- 2. 动作项
sec(f"2. 动作项 '{args.term}'（action → joint target 的确切定义）")
am = env.action_manager
term = am.get_term(args.term)
tcfg = term.cfg
print("动作总维度 (action_manager.total_action_dim) =", am.total_action_dim)
print("本项维度 (action_dim)                        =", term.action_dim)
print("cfg.scale        =", tcfg.scale)
print("cfg.offset       =", tcfg.offset)
print("cfg.clip         =", tcfg.clip)
print("cfg.use_default_offset =", tcfg.use_default_offset)
print("cfg.preserve_order     =", getattr(tcfg, "preserve_order", "<无此字段>"))
print("cfg.joint_names  =", getattr(tcfg, "joint_names", None))
print("_joint_names (解析后) =", list(term._joint_names))
print("_scale   (运行值) =", term._scale if not torch.is_tensor(term._scale) else term._scale.tolist())
off = term._offset
print("_offset  (运行值) 前 6 =", off[:6].tolist() if torch.is_tensor(off) else off)
print("_offset == asset.data.default_joint_pos ?",
      bool(torch.allclose(off if torch.is_tensor(off) else torch.tensor(off),
                          env.scene["robot"].data.default_joint_pos[0])))

# 实测一遍：喂一个已知 action，看 target 是不是 default + scale*action
with torch.no_grad():
    probe = torch.arange(1, term.action_dim + 1, dtype=torch.float32, device=env.device).unsqueeze(0) * 0.1
    term.process_actions(probe)
    pa = term._processed_actions[0]
    dj = env.scene["robot"].data.default_joint_pos[0]
    expect = dj + (tcfg.scale * probe[0] if not torch.is_tensor(tcfg.scale) else probe[0] * tcfg.scale)
    print("\n实测 action=[0.1,0.2,...] → processed_actions 前 4 =", pa[:4].tolist())
    print("对照 default + scale*action 前 4              =", expect[:4].tolist())
    print("两者一致 ?", bool(torch.allclose(pa, expect, atol=1e-5)))
    print("processed_actions 是否被裁剪到 joint limits 之外 ?",
          "见上面数值（clip=None 则原样传出，限位由执行器/PhysX 处理）")

# ---------------------------------------------------------------- 3. 执行器
sec("3. actuator / PD target 解释")
act_cfgs = env.cfg.scene.robot.actuators
print("actuator 组数 =", len(act_cfgs))
for name, ac in act_cfgs.items():
    print(f"\n--- {name}  (class={type(ac).__name__}) ---")
    for k in ("joint_names_expr", "effort_limit", "velocity_limit", "stiffness",
              "damping", "friction", "armature", "effort_limit_sim",
              "velocity_limit_sim", "dynamic_friction", "viscous_friction"):
        if hasattr(ac, k):
            v = getattr(ac, k)
            if isinstance(v, dict):
                v = {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()}
            print(f"  {k:20s} = {v}")

# ---------------------------------------------------------------- 4. 观测组
sec("4. 观测组（确认与策略输入一致）")
om = env.observation_manager
for gname in om.active_terms:
    dims = om.group_obs_term_dim.get(gname) if hasattr(om, "group_obs_term_dim") else None
    print(f"\n[{gname}] terms={list(om.active_terms[gname])}")
    print(f"         term dims={dims}")
    print(f"         group dim ={om.group_obs_dim.get(gname) if hasattr(om,'group_obs_dim') else '?'}")
print("\nobs 实际形状:", {k: tuple(v.shape) for k, v in env.reset()[0].items()} if isinstance(env.reset()[0], dict) else "?")

# ---------------------------------------------------------------- 5. actor 有无 squash
sec("5. 策略网络输出是否有 tanh / squash")
try:
    from instinct_rl.modules import ActorCritic
    import inspect
    src = inspect.getsource(ActorCritic.act)
    print(src)
except Exception as e:  # noqa: BLE001
    print("无法读取 ActorCritic.act 源码:", e)

env.close()
simulation_app.close()
