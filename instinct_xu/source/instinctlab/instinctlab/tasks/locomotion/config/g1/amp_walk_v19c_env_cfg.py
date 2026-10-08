"""V19c 的环境侧：**V19b 全部保留**，只改三处奖励（都是"把漏洞堵上 / 把过贵的罚降下来"）。

起点：**热启动 V19b 的最新 checkpoint**（保留判别器 —— AMP 口径与 V19b 逐字相同）
⇒ 目的是"保留已经学会的抬脚步态，只改激励结构"，而不是从头再学。

| # | 项 | V19b | **V19c** | 为什么 |
|---|---|---|---|---|
| ① | `feet_air_time` 的**实测运动门控** | 无（只有命令门控） | **`motion_threshold=0.15`、`motion_ang_scale=0.5`** | 命令门控只挡住"指令≈0"；挡不住"指令 0.6 m/s 但不前进、原地左右脚交替抬放刷单支撑分红"（用户实测观察到静止/慢速时也在原地交替站） |
| ② | `action_rate_l2` | −0.10 | **−0.06**（回到 V17 值） | 用户看 MuJoCo 判断"动作幅度/频率在正常范围内" ⇒ 该项在过度收费：raw 从 V18 的 5.6 升到 V19b 的 ~15 ⇒ 罚额 −1.5/步，占正奖励 Σ≈2.4 的 **60%** |
| ③ | `track_default_pose_on_stand` | −0.3 | **−0.5** | 该术语**只在指令接近 0 时生效**（`active = exp(−cmd_speed/0.25)`），内容 = `4·mean|q−q_def| + 0.5·mean|dq|` ⇒ 加大它能直接抑制"站立时抬脚/晃腿" |

## ① 门控细节（为什么要用"实测"速度）

```
gate_cmd    = 1[ ‖v_cmd,xy‖ > 0.1 ]                      # 原有
gate_motion = 1[ ‖v_actual,yaw‖_xy + 0.5·|ω_z,actual| > 0.15 ]   # ⭐ 新增
reward = score · 1[恰好单支撑] · gate_cmd · gate_motion
```
`motion_ang_scale=0.5` 把**原地转身**也算"在动"（转身本来就要迈步，不该被掐掉）；
新探针 `Step/air_time_motion_gate_off_frac` 直接告诉你有多少步被这道门关掉。
`motion_threshold = 0` 可完全关闭（= V19b 行为）。

## ② 代价：`action_rate_l2` 降权会削弱"防抖"

V18 当年是**因为 V17 抖动变大**才把该项从 −0.06 提到 −0.10 的。本轮按用户实测判断回调，
但**必须盯住 raw 值**：若 raw `action_rate`（= `Episode_Reward/action_rate_l2 / 0.06`）
继续往上超过 ~18–20 而步态指标不涨，说明抖动回来了 ⇒ 应恢复 −0.10 或改用
**只罚高频分量**的版本（`(a_t − 2a_{t−1} + a_{t−2})²`，把"有意的慢速大幅摆动"豁免）。

⚠️ 三处改动可分别从不同探针读出（① `air_time_motion_gate_off_frac` / `in_band_frac`；
② `Episode_Reward/action_rate_l2`；③ `track_default_pose_on_stand`），因此事后仍可归因。
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
from .amp_walk_v19b_env_cfg import (
    G1AmpWalkV19BEnvCfg,
    G1AmpWalkV19BEnvCfg_PLAY,
    G1AmpWalkV19BRewardsCfg,
    G1AmpWalkV19BTeacherEnv,
)

#: ① "实际在动"门控：实测速度（前向/侧向 + 0.5·|ωz|）低于此值则不给步态分红
AMP_WALK_V19C_MOTION_THRESHOLD = 0.15
AMP_WALK_V19C_MOTION_ANG_SCALE = 0.5
#: ② 关节抖动权重：−0.10 → −0.06（回 V17 值）
AMP_WALK_V19C_ACTION_RATE_WEIGHT = -0.06
#: ③ 站立回位姿态：−0.3 → −0.5（只在指令≈0 时生效）
AMP_WALK_V19C_STAND_POSE_WEIGHT = -0.5

AMP_WALK_V19C_RUN_NAME = (
    "G1AmpWalkV19cMVPPlane_obs378_rew21_amp10frame_flat2.0_airband_exptau0.08_motiongate0.15"
    "_ar0.06_standpose0.5_5clip_omnidir"
)


@configclass
class G1AmpWalkV19CRewardsCfg(G1AmpWalkV19BRewardsCfg):
    """V19b 的 21 项 + ①②③。"""

    def __post_init__(self):
        super().__post_init__()

        # ① 步态分红：加"实测在动"门控（其余 params 与 V19b 逐字相同）
        self.feet_air_time = RewTerm(
            func=locomotion_mdp.feet_air_time_target_biped,
            weight=AMP_WALK_V18_AIR_TIME_WEIGHT,                      # 4.0，不动
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "good_lo": AMP_WALK_V18_AIR_TIME_GOOD_LO,             # 0.4
                "good_hi": AMP_WALK_V18_AIR_TIME_GOOD_HI,             # 1.0
                "soft": AMP_WALK_V18_AIR_TIME_SOFT,                   # 0.3（仅上侧）
                "tau": AMP_WALK_V19B_AIR_TIME_TAU,                    # 0.08（下侧指数）
                "motion_threshold": AMP_WALK_V19C_MOTION_THRESHOLD,   # ⭐ 0.15 m/s
                "motion_ang_scale": AMP_WALK_V19C_MOTION_ANG_SCALE,   # ⭐ 0.5
            },
        )

        # ② 关节抖动：−0.10 → −0.06
        base_ar = self.action_rate_l2
        self.action_rate_l2 = RewTerm(
            func=base_ar.func,
            weight=AMP_WALK_V19C_ACTION_RATE_WEIGHT,
            params=base_ar.params,
        )

        # ③ 站立回位姿态：−0.3 → −0.5
        base_sp = self.track_default_pose_on_stand
        self.track_default_pose_on_stand = RewTerm(
            func=base_sp.func,
            weight=AMP_WALK_V19C_STAND_POSE_WEIGHT,
            params=base_sp.params,
        )


@configclass
class G1AmpWalkV19CEnvCfg(G1AmpWalkV19BEnvCfg):
    """训练用：V19b + 运动门控 + 抖动降权 + 站立姿态加强。"""

    rewards: G1AmpWalkV19CRewardsCfg = G1AmpWalkV19CRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V19C_RUN_NAME


@configclass
class G1AmpWalkV19CEnvCfg_PLAY(G1AmpWalkV19BEnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广。"""

    rewards: G1AmpWalkV19CRewardsCfg = G1AmpWalkV19CRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V19C_RUN_NAME


class G1AmpWalkV19CTeacherEnv(G1AmpWalkV19BTeacherEnv):
    """指令范围与 V19b 相同（全向四档）⇒ 期望值直接继承。"""


__all__ = [
    "AMP_WALK_V19C_ACTION_RATE_WEIGHT",
    "AMP_WALK_V19C_MOTION_ANG_SCALE",
    "AMP_WALK_V19C_MOTION_THRESHOLD",
    "AMP_WALK_V19C_RUN_NAME",
    "AMP_WALK_V19C_STAND_POSE_WEIGHT",
    "G1AmpWalkV19CEnvCfg",
    "G1AmpWalkV19CEnvCfg_PLAY",
    "G1AmpWalkV19CRewardsCfg",
    "G1AmpWalkV19CTeacherEnv",
]
