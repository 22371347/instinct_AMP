"""AMP 行走 **v8** 的环境配置 —— 与 v7 **完全相同**，只换 `run_name`。

只加一个文件的原因同 v4~v7：`G1AmpWalkV3EnvCfg.__post_init__` 会把 `run_name`
钉成 v3 的名字，复用 v7 会让日志目录看不出"这轮换了风格奖励形式"。

**只改 `run_name`，其余一律不动**：奖励集 **15 项**（含 `joint_deviation_arms`）、
指令分布、精选 7-clip 数据集、观测 626/99/670、动作契约（`scale=1.0`）全部继承 v3→v7。

本版的实质改动（风格奖励 `r = -log(1 - quad(d))`）在 agent cfg
（`G1AmpWalkV8PPORunnerCfg`）里，所以这个类刻意保持"零逻辑"。
"""

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
