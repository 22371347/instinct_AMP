"""V19d 的环境侧：**V19c 全部保留**，只把步态带在**站立/低速档**的输出改成**负值**。

| 项 | V19c | **V19d** |
|---|---|---|
| `feet_air_time` 在 `cmd_speed ≤ 0.1` 时 | **0**（`gate_cmd=0`，既不奖也不罚） | **负值**：`−stand_penalty · min(d/good_lo, 1)`，`stand_penalty = 0.5` |

## 为什么（V19c@9700 的实测定位）

"站立时原地左右脚交替抬放"在 V19c 里**几乎无价**，逐项排查：

| 奖励项 | 站立 env（`cmd_speed ≤ 0.1`）下的取值 | 对"原地抬脚"的态度 |
|---|---|---|
| `feet_air_time` | **0**（`gate_cmd = 1[cmd>0.1] = 0`） | **零价** ← 就是这一项 |
| AMP 风格项 | **0**（`disable_style_on_stand=True`、`stand_speed_threshold=0.1` 已实测确认） | 零价 |
| `track_lin_vel_xy_exp` (3.0) | ≈ 3.0/步（指令=0，**不动**就是满分） | 只罚"漂移"，原地踏步不漂移 ⇒ 几乎不罚 |
| `track_ang_vel_z_exp` (0.8) | 同上 | 同上 |
| **`track_default_pose_on_stand` (−0.5)** | 原始 cost ≈ 0.72 ⇒ **−0.36/步** | ✅ 唯一的成本（实测：权重 −0.3→−0.5 后原始 cost 0.846→0.722 ⇒ **该项确实在工作**，但不足以改掉踏步） |

⇒ 把"站立档"从 **0 价** 改成 **负价**，给策略一个**稠密、单一、方向明确**的信号。
`min(d/good_lo, 1)` 让 `d → 0` 时罚也 → 0（"抬一下立刻放"几乎不罚），`d ≥ good_lo` 罚满。

## 罚额量级（weight = 4.0）

| 单支撑时长 d | 0.02 s | 0.05 s | 0.08 s | 0.15 s | 0.30 s | ≥0.40 s |
|---|---|---|---|---|---|---|
| 每步罚额 | −0.10 | −0.25 | −0.40 | −0.75 | −1.50 | **−2.00** |

站立 env 的正奖励约 **3.5/步**（`track_lin 3.0` + `base_height` + `body_ang_vel`）⇒ 该罚约占
其 12%~57%，**偏激进但不至于压垮站立**。若 2k 轮后仍在踏步 ⇒ 提到 1.0（满罚 −4.0/步）。

## 新增探针（回答"到底有没有在踏步"）

* `Step/air_time_stand_step_frac`：**站立 + 真的抬了脚（d > 0.03 s）的步数占比** ← 本轮的**主指标**，目标 → 0；
* `Step/air_time_stand_pen_mean`：实际付出的站立罚均值（确认罚到东西了没）；
* `Step/stand_pose_pos_err` / `stand_pose_vel_err` / `stand_pose_active`（V19d 新增）：
  把 `track_default_pose_on_stand` 拆成"回默认姿态"与"关节速度阻尼"两部分 ⇒
  **直接回答用户的问题"速度阻尼项在不在工作"**（pos/vel 两个原始量应随训练下降）。

## 不动

`tau = 0.08`、`motion_threshold = 0.15`、`motion_ang_scale = 0.5`、`good_lo/hi = 0.4/1.0`、
`soft = 0.3`（仅上侧）、weight `4.0`；`action_rate_l2 = −0.06`、
`track_default_pose_on_stand = −0.5`；AMP 侧与 V19b/V19c **逐字相同**（`quad k=0.125`、
GP 1.0、wd 1e-5、logit_wd 0、eps 0.05、λ0.75、c1.0）。
起点：**热启动 V19c 最新 checkpoint**（保留判别器）。
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
from .amp_walk_v19b_env_cfg import AMP_WALK_V19B_AIR_TIME_TAU
from .amp_walk_v19c_env_cfg import (
    AMP_WALK_V19C_MOTION_ANG_SCALE,
    AMP_WALK_V19C_MOTION_THRESHOLD,
    G1AmpWalkV19CEnvCfg,
    G1AmpWalkV19CEnvCfg_PLAY,
    G1AmpWalkV19CRewardsCfg,
    G1AmpWalkV19CTeacherEnv,
)

#: ⭐ 本轮唯一改动：站立档的负罚强度（在 score 空间；×weight 4.0 才是每步罚额）
AMP_WALK_V19D_STAND_PENALTY = 0.5
#: 判"真的抬了脚"的时长阈值（探针用）
AMP_WALK_V19D_STAND_MIN_D = 0.03

AMP_WALK_V19D_RUN_NAME = (
    "G1AmpWalkV19dMVPPlane_obs378_rew21_amp10frame_flat2.0_airband_exptau0.08_motiongate0.15"
    "_standpen0.5_ar0.06_standpose0.5_5clip_omnidir"
)


@configclass
class G1AmpWalkV19DRewardsCfg(G1AmpWalkV19CRewardsCfg):
    """V19c 的 21 项 + 站立档负罚。"""

    def __post_init__(self):
        super().__post_init__()

        self.feet_air_time = RewTerm(
            func=locomotion_mdp.feet_air_time_target_biped,
            weight=AMP_WALK_V18_AIR_TIME_WEIGHT,                       # 4.0，不动
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "good_lo": AMP_WALK_V18_AIR_TIME_GOOD_LO,              # 0.4
                "good_hi": AMP_WALK_V18_AIR_TIME_GOOD_HI,              # 1.0
                "soft": AMP_WALK_V18_AIR_TIME_SOFT,                    # 0.3（仅上侧）
                "tau": AMP_WALK_V19B_AIR_TIME_TAU,                     # 0.08（下侧指数）
                "motion_threshold": AMP_WALK_V19C_MOTION_THRESHOLD,    # 0.15
                "motion_ang_scale": AMP_WALK_V19C_MOTION_ANG_SCALE,    # 0.5
                "stand_penalty": AMP_WALK_V19D_STAND_PENALTY,          # ⭐ 0.5
                "stand_min_d": AMP_WALK_V19D_STAND_MIN_D,              # 0.03（探针阈值）
            },
        )


@configclass
class G1AmpWalkV19DEnvCfg(G1AmpWalkV19CEnvCfg):
    """训练用：V19c + 站立档负罚。"""

    rewards: G1AmpWalkV19DRewardsCfg = G1AmpWalkV19DRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V19D_RUN_NAME


@configclass
class G1AmpWalkV19DEnvCfg_PLAY(G1AmpWalkV19CEnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广。"""

    rewards: G1AmpWalkV19DRewardsCfg = G1AmpWalkV19DRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V19D_RUN_NAME


class G1AmpWalkV19DTeacherEnv(G1AmpWalkV19CTeacherEnv):
    """指令范围与 V19c 相同（全向四档）⇒ 期望值直接继承。"""


__all__ = [
    "AMP_WALK_V19D_RUN_NAME",
    "AMP_WALK_V19D_STAND_MIN_D",
    "AMP_WALK_V19D_STAND_PENALTY",
    "G1AmpWalkV19DEnvCfg",
    "G1AmpWalkV19DEnvCfg_PLAY",
    "G1AmpWalkV19DRewardsCfg",
    "G1AmpWalkV19DTeacherEnv",
]
