"""Register the G1 OmniLoco experiment, the AMP-walk teacher, and their play configs."""

import gymnasium as gym

from . import agents
from .amp_walk_env_cfg import G1AmpWalkEnvCfg, G1AmpWalkEnvCfg_PLAY, G1AmpWalkTeacherEnv
from .amp_walk_v2_env_cfg import G1AmpWalkV2EnvCfg, G1AmpWalkV2EnvCfg_PLAY
from .amp_walk_v3_env_cfg import G1AmpWalkV3EnvCfg, G1AmpWalkV3EnvCfg_PLAY, G1AmpWalkV3TeacherEnv
from .amp_walk_v4_env_cfg import G1AmpWalkV4EnvCfg, G1AmpWalkV4EnvCfg_PLAY
from .amp_walk_v5_env_cfg import G1AmpWalkV5EnvCfg, G1AmpWalkV5EnvCfg_PLAY
from .amp_walk_v6_env_cfg import G1AmpWalkV6EnvCfg, G1AmpWalkV6EnvCfg_PLAY
from .amp_walk_v7_env_cfg import G1AmpWalkV7EnvCfg, G1AmpWalkV7EnvCfg_PLAY
from .amp_walk_v8_env_cfg import G1AmpWalkV8EnvCfg, G1AmpWalkV8EnvCfg_PLAY
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

# --- AMP 行走特权教师 v2：奖励集换成 AMP_mjlab 的 12 项配方 -------------------
# 动机与逐项对照见 `amp_walk_v2_env_cfg.py` 的模块 docstring。
# ⚠️ 环境类 **仍是 `G1AmpWalkTeacherEnv`**：动作/观测/时序/执行器契约与 v1 逐字相同，
#    因此 checkpoint、ONNX 导出、MuJoCo 回放、**蒸馏 teacher 侧接口**全部保持兼容；
#    agent cfg 也复用同一份（v2 只改奖励，不改策略/算法结构）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V2-G1-v0", G1AmpWalkV2EnvCfg),
    ("Instinct-Amp-Walk-V2-G1-Play-v0", G1AmpWalkV2EnvCfg_PLAY),
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

# --- AMP 行走特权教师 v3：v2 奖励配方 + action_rate_l2 −0.02 + init_noise_std 0.3 ---
# 动机、实测证据（σ 0.766→1.580）与「从零训练」的理由见
# `amp_walk_v3_env_cfg.py` 与 `agents/amp_walk_v3_ppo_cfg.py` 的模块 docstring。
# ⚠️ 环境类仍是 `G1AmpWalkTeacherEnv`（契约与 v1/v2 逐字相同）；
#    唯一区别是 **agent cfg 换成 v3 那一份**（它只改 init_noise_std/clip_min_std）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V3-G1-v0", G1AmpWalkV3EnvCfg),
    ("Instinct-Amp-Walk-V3-G1-Play-v0", G1AmpWalkV3EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        # ⚠️ 用 v3 自己的校验器：指令范围与保持时长与 v1 不同，其余期望值逐字相同
        entry_point=G1AmpWalkV3TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v3_ppo_cfg:G1AmpWalkV3PPORunnerCfg",
        },
    )

# --- AMP 行走特权教师 v4：AMP 侧两项修正（镜像损失 + 站立关风格）---------------
# 环境 / 观测 / 动作 / 奖励集（14 项）/ σ 起点 **全部继承 v3**；
# 只换 agent cfg。动机与两边差异见 `agents/amp_walk_v4_ppo_cfg.py` 的模块 docstring。
# ⚠️ teacher 侧仍用 `G1AmpWalkV3TeacherEnv`（环境完全相同，契验期望值一致）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V4-G1-v0", G1AmpWalkV4EnvCfg),
    ("Instinct-Amp-Walk-V4-G1-Play-v0", G1AmpWalkV4EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV3TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v4_ppo_cfg:G1AmpWalkV4PPORunnerCfg",
        },
    )

# --- AMP 行走特权教师 v5：风格奖励形式 quad -> log ---------------------------
# ⚠️ 判别器训练逻辑一个字没动，只改「判别器输出 → 奖励」那一步的函数形式，
#    所以这属于**奖励构成**改动，可以 `--resume` 续训。动机与实测证据见
#    `agents/amp_walk_v5_ppo_cfg.py` 的模块 docstring。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V5-G1-v0", G1AmpWalkV5EnvCfg),
    ("Instinct-Amp-Walk-V5-G1-Play-v0", G1AmpWalkV5EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV3TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v5_ppo_cfg:G1AmpWalkV5PPORunnerCfg",
        },
    )

# --- AMP 行走特权教师 v6：v5 + 恢复基类 `joint_deviation_arms`（-0.1）----------
# ⭐ 用户实测：无论静止还是移动，**肩/肘高举且接近限位**
#   （`right_shoulder_roll` 偏差 2.03 rad ≈ 116°、双侧 elbow ≈ 85°）。
#   原因是 v2 迁移 AMP_mjlab 12 项配方时把这一项整项删掉了 ⇒ 全任务奖励里
#   **没有任何一项约束手臂**。环境侧改动（奖励集 14 → 15 项），算法侧未动。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V6-G1-v0", G1AmpWalkV6EnvCfg),
    ("Instinct-Amp-Walk-V6-G1-Play-v0", G1AmpWalkV6EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV3TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v6_ppo_cfg:G1AmpWalkV6PPORunnerCfg",
        },
    )

# --- AMP 行走特权教师 v7：收敛性修复（σ 0.3→0.2 + AMP 系数 0.1→0.05）------------
# ⭐ **从零训练**（不带 `--resume`）。实测依据（逐关节监控 `joint_error_rmse`，@V6-13840）：
#   `shoulder_roll` 偏差 2.038 rad ⇒ 实际角 2.238（硬限位 2.252）⚠️ 卡死
#   `elbow`         偏差 1.490 rad ⇒ 实际角 2.090（硬限位 2.094）⚠️ 卡死
#   且 v1 起就存在 ⇒ 已在吸引子里，续训拉不回来。
# AMP 减半：实测 AMP 占真实回报 ~80%，减半后约 66%。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V7-G1-v0", G1AmpWalkV7EnvCfg),
    ("Instinct-Amp-Walk-V7-G1-Play-v0", G1AmpWalkV7EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV3TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v7_ppo_cfg:G1AmpWalkV7PPORunnerCfg",
        },
    )

# --- AMP 行走特权教师 v8：风格奖励换成 LLP 论文形式 r = -log(1 - quad(d)) ------
# ⚠️ 父类用 `discriminator_reward_type="wasserstein"`（唯一原样返回 d 的分支），
#    塑形在我们自己的桥接层做 ⇒ **不改上游 `instinct_rl`**；判别器自身损失未动
#    ⇒ 属于**奖励构成**改动 ⇒ 可续训。
# ⚠️ `-log(1-D)` 在 d=1 处会 +inf（D 恰好=1，而 MSELoss 参考侧目标就是 +1）
#    ⇒ 实现里把 D 夹到 1-1e-6，奖励上界 ≈ 13.8。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V8-G1-v0", G1AmpWalkV8EnvCfg),
    ("Instinct-Amp-Walk-V8-G1-Play-v0", G1AmpWalkV8EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV3TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v8_ppo_cfg:G1AmpWalkV8PPORunnerCfg",
        },
    )
