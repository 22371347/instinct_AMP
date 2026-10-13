"""AMP 行走 **v4**：环境与 v3 **逐字相同**，只换 `run_name`（v3 的 `__post_init__` 会把 `run_name` 钉死）。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— `run_name` 已被后层覆盖**（今天 = `G1AmpWalkT1l...`）⇒ 本文件只是"14 项奖励 / 7-clip 数据集 /
  观测 626/99/670 / 动作契约"那一代配置的**重命名副本**；teacher 侧直接复用 `G1AmpWalkV3TeacherEnv`（环境相同、契验期望值也一样）。
* 本版唯一实质改动在 agent cfg（`agents/amp_walk_v4_ppo_cfg.py`：镜像损失 + AMP 侧两处修正）⇒
  是否仍生效见 `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V4-G1-v0` 的「算法侧」节。
⚠️ `RewTerm(...)` 字面值 **≠ 生效值**（后层整体覆盖 `params`）→ `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V4-G1-v0`（算法侧见同节）"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v3_env_cfg import (
    G1AmpWalkV3EnvCfg,
    G1AmpWalkV3EnvCfg_PLAY,
)

#: v4 的 run_name（= 日志目录前缀）。加了 `mirror` 后缀便于与 v3 区分。
AMP_WALK_V4_RUN_NAME = "G1AmpWalkV4Plane_obs626_rew14_ar0.02_ns0.3_mirror"


@configclass
class G1AmpWalkV4EnvCfg(G1AmpWalkV3EnvCfg):
    """训练用：**一切继承 v3**，只把 `run_name` 改掉。

    AMP 侧的两处修正都在 agent cfg（`G1AmpWalkV4PPORunnerCfg`）里，
    所以这个类刻意保持"零逻辑"。
    """

    def __post_init__(self):
        super().__post_init__()
        # 父类会写回 AMP_WALK_V3_RUN_NAME，所以必须在它之后再写一次。
        self.run_name = AMP_WALK_V4_RUN_NAME


@configclass
class G1AmpWalkV4EnvCfg_PLAY(G1AmpWalkV3EnvCfg_PLAY):
    """回放用：同样只换 run_name。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V4_RUN_NAME


__all__ = [
    "AMP_WALK_V4_RUN_NAME",
    "G1AmpWalkV4EnvCfg",
    "G1AmpWalkV4EnvCfg_PLAY",
]
