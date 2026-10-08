"""V19b 的环境侧：**V19a 全部不动，只把步态目标带的「下侧」换成近零端加密的指数形状**。

唯一改动
--------
`feet_air_time_target_biped(tau=0.08)`（V19a 是线性下侧 `(good_lo−d)/soft`）。

    below(d) = clamp( (exp(−d/τ) − exp(−good_lo/τ)) / (1 − exp(−good_lo/τ)), 0, 1 )     # τ = 0.08 s

为什么（V18/V19a 的实测证据）
----------------------------
两种形状都留了"小跳拿不到钱、但也几乎不亏"的平台：

| | V18 | V19a | **V19b** |
|---|---|---|---|
| 下侧形状 | 线性，`soft=0.3 < good_lo=0.4` | 线性，`soft=good_lo=0.4` | **指数，τ=0.08** |
| `d = 0.01 s` 处 score | **0（且梯度恒 0，死区）** | 0.025（梯度 2.5/s） | **0.118（梯度 ≈12.5/s）** |
| 实测 `air_time_single_stance` | 0.0096 s | 0.0096 → 0.0111 s（**1150 轮平台化**） | ? |
| 实测 `air_time_in_band_frac` | **0.0000** | 0.0005~0.0015 | ? |

步态项的上限是 `4.0 × 0.6 ≈ 2.4/步`（真步行的单支撑时间占比 ~0.6），V19a 只拿到 **0.038/步
= 1.6%**，而 `track_lin_vel_xy_exp` 已经 1.28/步（接近饱和）⇒ **上涨空间全在步态项，
但"路径坡度"不够**：线性下侧在 `d=0.01 s` 处梯度只有 2.5/s（要提高 0.04 s 才多拿 0.1 奖励）。
指数下侧在近零端下凸 ⇒ 同样位置梯度 ≈12.5/s，且**0.35 s 后已≈1**，不与 `[0.4, 1.0]`
好区间打架（见 `rewards_amp.py` 里的完整数值表）。

⚠️ 本轮**从零训练**（起点也是变量之一）：V18（从零）与 V19a（热启动 V18）**都**落在
小跳模态 ⇒ 说明"热启动"不是唯一原因，而"小跳在早期无价"是共同点。从零 + 近零端加密
两条一起改，才能同时回答"起点"与"定价"两个问题。
"""

from __future__ import annotations

import instinctlab.tasks.locomotion.mdp as locomotion_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .amp_walk_v18_env_cfg import (
    AMP_WALK_V18_AIR_TIME_GOOD_HI,
    AMP_WALK_V18_AIR_TIME_GOOD_LO,
    AMP_WALK_V18_AIR_TIME_SOFT,
    AMP_WALK_V18_AIR_TIME_WEIGHT,
)
from .amp_walk_v19a_env_cfg import (
    G1AmpWalkV19AEnvCfg,
    G1AmpWalkV19AEnvCfg_PLAY,
    G1AmpWalkV19ARewardsCfg,
    G1AmpWalkV19ATeacherEnv,
)

#: ⭐ 本轮唯一改动：下侧改为指数型（近零端加密），时间尺度 0.08 s。
#:   含义：`d = 0.08 s` 处短缺已衰减到 1/e ≈ 0.37；`d = 0.4 s`（= good_lo）处恰好 0。
AMP_WALK_V19B_AIR_TIME_TAU = 0.08

AMP_WALK_V19B_RUN_NAME = (
    "G1AmpWalkV19bMVPPlane_obs378_rew21_amp10frame_flat2.0_airband_exptau0.08_ar0.10_5clip_omnidir"
)


@configclass
class G1AmpWalkV19BRewardsCfg(G1AmpWalkV19ARewardsCfg):
    """V19a 的 21 项逐字保留，只替换 `feet_air_time` 的**下侧形状**（加 `tau`）。"""

    def __post_init__(self):
        super().__post_init__()

        self.feet_air_time = RewTerm(
            func=locomotion_mdp.feet_air_time_target_biped,
            weight=AMP_WALK_V18_AIR_TIME_WEIGHT,        # 4.0，不动（形状改了，权重不必抬）
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "good_lo": AMP_WALK_V18_AIR_TIME_GOOD_LO,   # 0.4，不动
                "good_hi": AMP_WALK_V18_AIR_TIME_GOOD_HI,   # 1.0，不动
                # ⚠️ `tau` 一开，`soft` **就只管上侧**（`d > good_hi` 的线性衰竭）。
                #    这里取 **V18 原值 0.3** ⇒ 上侧在 `1.0+0.3 = 1.3 s` 归零
                #    （V19a 是 0.4 ⇒ 1.4 s；差 0.1 s，可忽略）。
                "soft": AMP_WALK_V18_AIR_TIME_SOFT,         # 0.3（V18 原值）：仅上侧
                "tau": AMP_WALK_V19B_AIR_TIME_TAU,          # ⭐ 新增：下侧指数化 0.08 s
            },
        )


@configclass
class G1AmpWalkV19BEnvCfg(G1AmpWalkV19AEnvCfg):
    """训练用：V19a 全部 + 步态带下侧指数化（近零端加密）。"""

    rewards: G1AmpWalkV19BRewardsCfg = G1AmpWalkV19BRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V19B_RUN_NAME


@configclass
class G1AmpWalkV19BEnvCfg_PLAY(G1AmpWalkV19AEnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广。"""

    rewards: G1AmpWalkV19BRewardsCfg = G1AmpWalkV19BRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V19B_RUN_NAME


class G1AmpWalkV19BTeacherEnv(G1AmpWalkV19ATeacherEnv):
    """指令范围与 V19a 相同（全向四档）⇒ 期望值直接继承。"""


__all__ = [
    "AMP_WALK_V19B_AIR_TIME_TAU",
    "AMP_WALK_V19B_RUN_NAME",
    "G1AmpWalkV19BEnvCfg",
    "G1AmpWalkV19BEnvCfg_PLAY",
    "G1AmpWalkV19BRewardsCfg",
    "G1AmpWalkV19BTeacherEnv",
]
