"""AMP 行走 **v5** 的环境配置 —— 与 v4 **完全相同**，只换 `run_name`。

只加一个文件的原因和 v4 一样：`G1AmpWalkV3EnvCfg.__post_init__` 会把 `run_name`
钉成 v3 的名字，复用 v4 会让日志目录看不出"这轮换了风格奖励形式"，事后极易混淆。
**只改 `run_name`，其余一律不动**（奖励集 14 项、指令分布、精选 7-clip 数据集、
观测 626/99/670、动作契约全部继承 v3/v4）。

⚠️ teacher 侧（蒸馏接口）继续复用 `G1AmpWalkV3TeacherEnv`：环境完全相同。
"""

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
