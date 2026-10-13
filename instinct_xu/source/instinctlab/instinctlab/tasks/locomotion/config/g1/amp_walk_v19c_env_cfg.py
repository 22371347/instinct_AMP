"""V19c：**V19b 全部保留，只改三处奖励**（加"实测在动"门控 / 降 `action_rate_l2` / 提高站立姿态罚；2026-10-03）。
起点 = **热启动 V19b 的最新 checkpoint**（保留判别器 ⇒ AMP 口径与 V19b 逐字相同），只改激励结构、不重学。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— 表中三项今天均已失效**（① 被 T1a2、② 被 T1a 白名单、③ 被 V23 覆盖）。
* 仍"活着"的是**结论**：③ 所在的那类"**只在低速段生效的项，其价格会被'该省的动作'过度征收**"
  → `docs/amp_walk/EXPERIMENTS.md` §B1。

## 本层改动一览

| # | 项 | V19b → **V19c** | 现状（今天实际生效）|
|---|---|---|---|
| ① | `feet_air_time` 的**实测运动门控** | 无（只有命令门控）→ **`motion_threshold=0.15` / `motion_ang_scale=0.5`**（转原地转身也算"在动"）| ❌ **已被 T1a2 覆盖** → 换回 `feet_air_time_positive_biped`（`+1` / `threshold=0.5`），**无任何门控** |
| ② | `action_rate_l2` | −0.10 → **−0.06**（回 V17 值）| ❌ **已被 T1a 白名单（抄 T0 值）覆盖** → **−0.05** |
| ③ | `track_default_pose_on_stand` | −0.3 → **−0.5** | ❌ **已被 V23 覆盖** → **−0.8** + 逐关节权重（knee/ankle 2.0、`vel_scale 0.8`…）|
| ④ | 起点 | 热启动 V19b 最新 ckpt | 已不适用（今天由 T1l 的起点决定）|

## 推理与判读（已归档，勿在本文件重复）

* ① 的动机：命令门控只挡"指令≈0"，挡不住"指令 0.6 m/s 却不前进、**原地左右脚交替抬放刷单支撑分红**"
  （用户实测观察到静止/慢速时也在原地交替站）→ `docs/amp_walk/EXPERIMENTS.md` §B1
* ⭐ ① 的**后续反证（重要）**：这道"实测根速度瞬时幅值"门在 V19c/V19d/V20 三代的**关闭率只有
  0.46% / 0.46% / 0.63~0.95% ⇒ 形同虚设**（踏步时根速度前后摇摆的瞬时幅值本来就在 0.3~1 m/s）
  ⇒ 后来改成"方向投影 + EMA 低通"的**进度判据** → `RESULTS_v19cd_v20_stand_probes.md` §2.3 + §B1
* ② 的两难：降权会削弱"防抖"（V18 当年正是因 V17 抖动变大才把该项提到 −0.10）；须盯 raw `action_rate`
  （= `Episode_Reward/action_rate_l2 / 0.06`）超过 ~18–20 而步态指标不涨 ⇒ 回退，或改"只罚高频分量"版
  → §B5 + **勘误记录（三）**（2026-10-08 反证：放松它会让**直腿/碎步模态更容易出现**，方向与目标相反）
* ③ 的语义：`active = exp(−cmd_speed/0.25)`，内容 = `4·mean|q−q_def| + 0.5·mean|dq|` ⇒ **只在指令≈0 时生效**
  → §B1
* ⚠️ 三处改动可从不同探针分离读出（① `air_time_motion_gate_off_frac` / `in_band_frac`；
  ② `Episode_Reward/action_rate_l2`；③ `track_default_pose_on_stand`）⇒ 事后仍可归因 → §E3

⚠️ 其余逐字不动：V19b 的步态目标带实现（`tau 0.08` / `soft 0.3` / `good_lo,hi 0.4,1.0` / `weight 4.0`）、
   21 项结构、5 clip 数据集、四档全向指令（今天 = 5 clip + 全向分档指令，**grep 旁证、未运行期核对**）。
   ⚠️ 算法侧本层不动 → 见 `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V19c-G1-v0` 的「算法侧」节。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V19c-G1-v0`
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
