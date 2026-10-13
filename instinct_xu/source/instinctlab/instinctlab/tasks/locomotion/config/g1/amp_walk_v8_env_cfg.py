"""AMP 行走 **v8**：环境与 v7 **逐字相同**，只换 `run_name`（v3 的 `__post_init__` 会把 `run_name` 钉死）。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— `run_name` 已被后层覆盖**（今天 = `G1AmpWalkT1l...`）⇒ 本文件只是"15 项奖励（含
  `joint_deviation_arms`）/ 7-clip 数据集 / 观测 626/99/670 / 动作契约 `scale=1.0`"那一代配置的**重命名副本**。
* 本版唯一实质改动在 agent cfg（`agents/amp_walk_v8_ppo_cfg.py`：风格奖励换 `log` 形式）⇒
  是否仍生效见 `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V8-G1-v0` 的「算法侧」节。

⚠️ `RewTerm(...)` 字面值 **≠ 生效值**（后层整体覆盖 `params`）→ `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V8-G1-v0`（算法侧见同节）"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v7_env_cfg import (
    G1AmpWalkV7EnvCfg,
    G1AmpWalkV7EnvCfg_PLAY,
)

#: v8 的 run_name（= 日志目录前缀）。`logquad` = r=-log(1-quad(d))。
AMP_WALK_V8_RUN_NAME = "G1AmpWalkV8Plane_obs626_rew15_ar0.02_ns0.2_ampc0.05_logquad_arms"


@configclass
class G1AmpWalkV8EnvCfg(G1AmpWalkV7EnvCfg):
    """训练用：**一切继承 v7**（σ 0.2 + 15 项奖励），只把 `run_name` 改掉。"""

    def __post_init__(self):
        super().__post_init__()
        # 父类会写回 AMP_WALK_V7_RUN_NAME，所以必须在它之后再写一次。
        self.run_name = AMP_WALK_V8_RUN_NAME


@configclass
class G1AmpWalkV8EnvCfg_PLAY(G1AmpWalkV7EnvCfg_PLAY):
    """回放用：同样只换 run_name。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V8_RUN_NAME


__all__ = [
    "AMP_WALK_V8_RUN_NAME",
    "G1AmpWalkV8EnvCfg",
    "G1AmpWalkV8EnvCfg_PLAY",
]
