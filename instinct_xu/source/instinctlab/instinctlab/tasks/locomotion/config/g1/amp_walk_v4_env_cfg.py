"""AMP 行走 **v4** 的环境配置 —— 与 v3 **完全相同**，只换 `run_name`。

为什么要单独一个文件：`G1AmpWalkV3EnvCfg.__post_init__` 会把 `run_name` 钉成 v3 的
`AMP_WALK_V3_RUN_NAME`，直接复用 v3 会让 v4 的日志目录名带 `rew14_...` 但看不出
"这轮加了镜像损失"，事后极易混淆。**只改 run_name，其余一律不动**（奖励集 14 项、
指令分布、精选 7-clip 数据集、观测 626/99/670、动作契约全部继承 v3）。

⚠️ teacher 侧（蒸馏接口）直接复用 `G1AmpWalkV3TeacherEnv`：环境完全相同，
   契验期望值也一样，没必要再复制一份。
"""

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
