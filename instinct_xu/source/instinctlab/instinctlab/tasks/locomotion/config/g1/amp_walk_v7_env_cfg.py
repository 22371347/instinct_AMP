"""AMP 行走 **v7** 的环境配置 —— 与 v6 **完全相同**，只换 `run_name`。

只加一个文件的原因同 v4/v5/v6：`G1AmpWalkV3EnvCfg.__post_init__` 会把 `run_name`
钉成 v3 的名字，复用 v6 会让日志目录看不出"这轮 σ 与 AMP 系数都改了"。

**只改 `run_name`，其余一律不动**：奖励集 **15 项**（v3 的 14 项 + v6 恢复的
`joint_deviation_arms`）、指令分布、精选 7-clip 数据集、观测 626/99/670、
动作契约（`scale=1.0`）全部继承 v3→v6。

⚠️ 本任务按「**从零训练**」启动（不带 `--resume`），理由见
`agents/amp_walk_v7_ppo_cfg.py` 的模块 docstring。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v6_env_cfg import (
    G1AmpWalkV6EnvCfg,
    G1AmpWalkV6EnvCfg_PLAY,
)

#: v7 的 run_name（= 日志目录前缀）。
AMP_WALK_V7_RUN_NAME = "G1AmpWalkV7Plane_obs626_rew15_ar0.02_ns0.2_ampc0.05_arms"


@configclass
class G1AmpWalkV7EnvCfg(G1AmpWalkV6EnvCfg):
    """训练用：**一切继承 v6**（15 项奖励），只把 `run_name` 改掉。

    行为改动全在 agent cfg（σ 0.2 / `discriminator_reward_coef` 0.05），
    所以这个类刻意保持"零逻辑"。
    """

    def __post_init__(self):
        super().__post_init__()
        # 父类会写回 AMP_WALK_V6_RUN_NAME，所以必须在它之后再写一次。
        self.run_name = AMP_WALK_V7_RUN_NAME


@configclass
class G1AmpWalkV7EnvCfg_PLAY(G1AmpWalkV6EnvCfg_PLAY):
    """回放用：同样只换 run_name。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V7_RUN_NAME


__all__ = [
    "AMP_WALK_V7_RUN_NAME",
    "G1AmpWalkV7EnvCfg",
    "G1AmpWalkV7EnvCfg_PLAY",
]
