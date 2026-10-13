"""AMP 行走 **v7**：与 v6 **完全相同**，只换 `run_name`（v3 的 `__post_init__` 把 `run_name` 钉死）。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— `run_name` 已被后层覆盖**（今天 = `G1AmpWalkT1l...`）⇒ 本文件是"15 项奖励 / 7-clip 数据集 /
  观测 626/99/670 / 动作契约 `scale=1.0`"那一代配置的**重命名副本**，其余逐字继承 v3→v6。
* 判读（已归档）：本任务按**从零训练**（不带 `--resume`）启动的理由 → `agents/amp_walk_v7_ppo_cfg.py` docstring

⚠️ `RewTerm(...)` 字面值 **≠ 生效值**（后层整体覆盖 `params`）→ `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V7-G1-v0`（算法侧见同节）
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
