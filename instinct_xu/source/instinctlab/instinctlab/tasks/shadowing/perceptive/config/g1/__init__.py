"""Register only the perceptive Vault teacher and Vault student tasks."""

import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.shadowing.perceptive.config.g1"

for task_id, config_file, env_class, runner_file, runner_class in (
    (
        "Instinct-Perceptive-Vault-G1-v0",
        "perceptive_vault_cfg",
        "G1PerceptiveVaultEnvCfg",
        "instinct_rl_vault_cfg",
        "G1PerceptiveVaultPPORunnerCfg",
    ),
    (
        "Instinct-Perceptive-Vault-G1-Play-v0",
        "perceptive_vault_cfg",
        "G1PerceptiveVaultEnvCfg_PLAY",
        "instinct_rl_vault_cfg",
        "G1PerceptiveVaultPPORunnerCfg",
    ),
    (
        "Instinct-Perceptive-Vault-VaeDistill-G1-v0",
        "perceptive_vault_vae_distill_cfg",
        "G1PerceptiveVaultVaeDistillEnvCfg",
        "instinct_rl_vault_vae_distill_cfg",
        "G1PerceptiveVaultVaeDistillPPORunnerCfg",
    ),
    (
        "Instinct-Perceptive-Vault-VaeDistill-G1-Play-v0",
        "perceptive_vault_vae_distill_cfg",
        "G1PerceptiveVaultVaeDistillEnvCfg_PLAY",
        "instinct_rl_vault_vae_distill_cfg",
        "G1PerceptiveVaultVaeDistillPPORunnerCfg",
    ),
):
    gym.register(
        id=task_id,
        entry_point="instinctlab.envs:InstinctRlEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{task_entry}.{config_file}:{env_class}",
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.{runner_file}:{runner_class}",
        },
    )
