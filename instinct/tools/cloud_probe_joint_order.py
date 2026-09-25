#!/usr/bin/env python3
"""打印 InstinctLab G1 的**articulation 关节顺序**（= 观测里 joint_pos_rel 的顺序）。

为什么必须实测而不是推断
------------------------
同一台 G1 在这个项目里至少存在**三套互不相同的关节顺序**：
  1. MuJoCo MJCF / 执行器序：腰 → 左腿 → 右腿 → …（本地 `joint_order.MUJOCO_ORDER`）
  2. OMNI_LOCO 序（**动作**与 `last_action` 用）：肩优先、左右交错
     （本地 `joint_order.ISAACLAB_ORDER`，由契约校验器断言）
  3. 训练数据 npz 的 `joint_names` 序：左腿 → 右腿 → …（AMP 参考用）
而观测里的 `joint_pos_rel` / `joint_vel` 用的是 `SceneEntityCfg("robot")` 的默认
（`joint_names=None` → 整段跳过解析 → `joint_ids=slice(None)`）⇒ **articulation 序**，
它到底等于上面哪一个，只能问 IsaacLab。

输出：`JOINT_NAMES=[...]` / `DEFAULT_JOINT_POS=[...]`，供本地比对与修正观测拼接。
"""
import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--task", default="Instinct-Amp-Walk-G1-v0")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import instinctlab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env = gym.make(args.task, cfg=cfg).unwrapped
robot = env.scene["robot"]

print("TASK =", args.task)
print("JOINT_NAMES =", list(robot.joint_names))
print("DEFAULT_JOINT_POS =", [round(v, 6) for v in robot.data.default_joint_pos[0].tolist()])
print("ACTION_TERM_JOINTS =", list(env.action_manager.get_term("joint_pos")._joint_names))
print("ACTION_SCALE =", env.action_manager.get_term("joint_pos")._scale)
print("ACTION_OFFSET =", env.action_manager.get_term("joint_pos")._offset[:3].tolist())
# 观测组里的关节项到底用了哪些 joint（验证是不是「全部/articulation 序」）
om = env.observation_manager
print("POLICY_GROUP_TERMS =", list(om.active_terms.get("policy", [])))

env.close()
simulation_app.close()
