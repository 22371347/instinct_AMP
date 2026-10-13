"""AMP 行走 **v5**：环境与 v4 **逐字相同**，只换 `run_name`（v3 的 `__post_init__` 会把 `run_name` 钉死）。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— `run_name` 已被后层覆盖**（今天 = `G1AmpWalkT1l...`）⇒ 本文件只是"14 项奖励 / 7-clip 数据集 / 观测 626/99/670 / 动作契约"那一代配置的**重命名副本**；teacher 侧始终复用 `G1AmpWalkV3TeacherEnv`。
* 唯一实质改动在 agent cfg（`agents/amp_walk_v5_ppo_cfg.py`：风格奖励换 `logamp` 形式）→ 见 `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V5-G1-v0` 的「算法侧」节。
⚠️ `RewTerm(...)` 字面值 **≠ 生效值**（后层整体覆盖 `params`）→ `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V5-G1-v0`（算法侧见同节）"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v4_env_cfg import (
    G1AmpWalkV4EnvCfg,
    G1AmpWalkV4EnvCfg_PLAY,
)

#: v5 的 run_name（= 日志目录前缀）。`logamp` 表示风格奖励换成 `log` 形式。
AMP_WALK_V5_RUN_NAME = "G1AmpWalkV5Plane_obs626_rew14_ar0.02_ns0.3_mirror_logamp"


@configclass
class G1AmpWalkV5EnvCfg(G1AmpWalkV4EnvCfg):
    """训练用：**一切继承 v4**，只把 `run_name` 改掉。

    唯一的行为改动（`discriminator_reward_type`）在 agent cfg
    （`G1AmpWalkV5PPORunnerCfg`）里，所以这个类刻意保持"零逻辑"。
    """

    def __post_init__(self):
        super().__post_init__()
        # 父类会写回 AMP_WALK_V4_RUN_NAME，所以必须在它之后再写一次。
        self.run_name = AMP_WALK_V5_RUN_NAME


@configclass
class G1AmpWalkV5EnvCfg_PLAY(G1AmpWalkV4EnvCfg_PLAY):
    """回放用：同样只换 run_name。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V5_RUN_NAME


__all__ = [
    "AMP_WALK_V5_RUN_NAME",
    "G1AmpWalkV5EnvCfg",
    "G1AmpWalkV5EnvCfg_PLAY",
]
