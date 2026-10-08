"""V19a 的环境侧：**逐字沿用 V18**，只把步态目标带的「下侧衰竭尺度」`soft` 从 0.3 → 0.4。

唯一改动
--------
`feet_air_time_target_biped(soft=0.3 → 0.4)`。

为什么（V18 实测，`in_band_frac` 恒为 0 的直接原因）
--------------------------------------------------
V18 打出的两个探针：

    Step/air_time_single_stance   0.0071 → 0.0113 s     （V17 是 0.007 s）
    Step/air_time_in_band_frac    0.0000 → 0.0000       （**恒为 0**）
    Episode_Reward/feet_air_time  0.0000 → 0.0011

核函数是

    score = 1 − clamp((good_lo − d)/soft, 0, 1) − clamp((d − good_hi)/soft, 0, 1)
    reward = score · 1[单支撑] · 1[‖cmd_xy‖ > 0.1]

`soft = 0.3`、`good_lo = 0.4` ⇒ 下侧在 `d < good_lo − soft = 0.1 s` 处**饱和**
（`clamp` 到 1）⇒ `score ≡ 0`，**奖励与梯度同时恒为 0**。
V18 的单支撑时长实测只有 `0.009~0.011 s < 0.1 s` ⇒ 该奖励**从头到尾没工作过**，
"小跳"没有任何一项在定价它。

取 `soft = good_lo = 0.4` 后，下侧退化成 `score = d / 0.4`
⇒ 在 `d ∈ [0, 0.4]` 上**处处有梯度**（`∂score/∂d = 2.5 /s`，`d = 0.01 s` 时每步 ≈ 0.025
—— 很小，但**不再是 0**，足以 bootstrap）。上侧 `good_hi + soft = 1.4 s` 才归零，
比 V18 的 1.3 s 略宽一点，可接受。

其余（5 clip 数据集、四档全向指令、`action_rate_l2 −0.10`、`flat_orientation_l2 −2`、
`feet_swing_time_penalty 0`、三个跟踪项 3.0/0.8/1.0）**一律不动**。
"""

from __future__ import annotations

import instinctlab.tasks.locomotion.mdp as locomotion_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .amp_walk_v18_env_cfg import (
    AMP_WALK_V18_AIR_TIME_GOOD_HI,
    AMP_WALK_V18_AIR_TIME_GOOD_LO,
    AMP_WALK_V18_AIR_TIME_WEIGHT,
    G1AmpWalkV18EnvCfg,
    G1AmpWalkV18EnvCfg_PLAY,
    G1AmpWalkV18RewardsCfg,
    G1AmpWalkV18TeacherEnv,
)

#: ⭐ 本轮唯一改动：下侧衰竭尺度 0.3 → **0.4**（= `good_lo`）
#:   ⇒ `d < 0.1 s` 的"恒 0 且无梯度"死区消失（见模块 docstring 的推导）。
AMP_WALK_V19A_AIR_TIME_SOFT = 0.4

AMP_WALK_V19A_RUN_NAME = (
    "G1AmpWalkV19aMVPPlane_obs378_rew21_amp10frame_flat2.0_airband_soft0.4_ar0.10_5clip_omnidir"
)


@configclass
class G1AmpWalkV19ARewardsCfg(G1AmpWalkV18RewardsCfg):
    """V18 的 21 项逐字保留，只替换 `feet_air_time` 的 `soft`。"""

    def __post_init__(self):
        super().__post_init__()

        # ⭐ 只改 `soft`：`func` / `weight` / 其余 params 与 V18 完全一致
        self.feet_air_time = RewTerm(
            func=locomotion_mdp.feet_air_time_target_biped,
            weight=AMP_WALK_V18_AIR_TIME_WEIGHT,          # 4.0，不动
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "good_lo": AMP_WALK_V18_AIR_TIME_GOOD_LO,   # 0.4，不动
                "good_hi": AMP_WALK_V18_AIR_TIME_GOOD_HI,   # 1.0，不动
                "soft": AMP_WALK_V19A_AIR_TIME_SOFT,        # 0.3 → 0.4 ⭐
            },
        )


@configclass
class G1AmpWalkV19AEnvCfg(G1AmpWalkV18EnvCfg):
    """训练用：V18 全部 + 步态带下侧消死区。"""

    rewards: G1AmpWalkV19ARewardsCfg = G1AmpWalkV19ARewardsCfg()

    def __post_init__(self):
        super().__post_init__()          # 含 V18 的 `_apply_v18_dataset_commands`（5 clip / 全向指令）
        self.run_name = AMP_WALK_V19A_RUN_NAME


@configclass
class G1AmpWalkV19AEnvCfg_PLAY(G1AmpWalkV18EnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广。"""

    rewards: G1AmpWalkV19ARewardsCfg = G1AmpWalkV19ARewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V19A_RUN_NAME


class G1AmpWalkV19ATeacherEnv(G1AmpWalkV18TeacherEnv):
    """指令范围与 V18 相同（全向四档）⇒ 期望值直接继承 V18。"""


__all__ = [
    "AMP_WALK_V19A_AIR_TIME_SOFT",
    "AMP_WALK_V19A_RUN_NAME",
    "G1AmpWalkV19AEnvCfg",
    "G1AmpWalkV19AEnvCfg_PLAY",
    "G1AmpWalkV19ARewardsCfg",
    "G1AmpWalkV19ATeacherEnv",
]
