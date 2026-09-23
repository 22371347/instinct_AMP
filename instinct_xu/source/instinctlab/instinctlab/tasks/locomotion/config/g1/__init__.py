"""Register the G1 OmniLoco experiment, the AMP-walk teacher, and their play configs."""

import gymnasium as gym

from . import agents
from .amp_walk_env_cfg import G1AmpWalkEnvCfg, G1AmpWalkEnvCfg_PLAY, G1AmpWalkTeacherEnv
from .omni_loco_env_cfg import G1OmniLocoEnvCfg, G1OmniLocoEnvCfg_PLAY, G1OmniLocoTeacherEnv

for task_id, env_cfg in (
    ("Instinct-Unified-OmniLoco-G1-v0", G1OmniLocoEnvCfg),
    ("Instinct-Unified-OmniLoco-G1-Play-v0", G1OmniLocoEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1OmniLocoTeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.omni_loco_ppo_cfg:G1OmniLocoPPORunnerCfg",
        },
    )

# --- AMP 行走特权教师（PLAN_amp_walk.md）-------------------------------------
# 与 OmniLoco 的唯一接口差异：actor 观测 623 -> 626（新增 3 维 base_lin_vel）。
# 沿用同一套 fail-fast 契约校验器（G1AmpWalkTeacherEnv 只覆盖期望值）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-G1-v0", G1AmpWalkEnvCfg),
    ("Instinct-Amp-Walk-G1-Play-v0", G1AmpWalkEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkTeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_ppo_cfg:G1AmpWalkPPORunnerCfg",
        },
    )
