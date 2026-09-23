"""Register only the G1 OmniLoco experiment and its play configuration."""

import gymnasium as gym

from . import agents
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
