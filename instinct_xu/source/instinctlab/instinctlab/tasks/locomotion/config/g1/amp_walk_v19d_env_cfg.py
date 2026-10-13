"""V19d：把「站立/低速档」下 `feet_air_time` 的输出由 **0 价改为负价**（V19c 其余全部保留）。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— 已被后层完全覆盖**：整项 `feet_air_time`（含本层新增的 `stand_penalty` / `stand_min_d`）
  已在 **T1a2** 换成基类旧版 `feet_air_time_positive_biped`（`threshold=0.5`、`weight=+1`）
  ⇒ 本层的"站立负罚"**今天不存在**。
* 仍"活着"的是**探针设计**：`Step/air_time_stand_step_frac` / `stand_pose_pos_err` /
  `stand_pose_vel_err` / `stand_pose_active` —— 后续版本（V23/V24/V26）仍靠 `stand_pose_*` 定位站姿问题。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | ⭐ 唯一变量：站立档（`cmd_speed ≤ 0.1`）`feet_air_time` 由 `gate_cmd=0`（零价）改为**负价** `-stand_penalty·min(d/good_lo, 1)`，`stand_penalty=0.5`、`stand_min_d=0.03` | ❌ **已被 T1a2 覆盖** → 今天 = `feet_air_time_positive_biped`（`+1` / `threshold=0.5`），**无 `stand_penalty` 参数** |
| ② | 新增探针（站立踏步占比 / 站立罚均值 / 站姿 pos·vel 拆分）| 代码侧；`stand_pose_*` 读数沿用至今 |
| ③ | 起点 = **热启动 V19c 最新 ckpt**（保留判别器）| 已不适用（今天由 T1l 起点决定）|

## 推理与判读（已归档，勿在本文件重复）

* ⭐ 「该类项**只在低速段生效**，其价格会被'该省的动作'过度征收」→ `docs/amp_walk/EXPERIMENTS.md` §B1
* 定位过程：V19c@9700"站立时原地左右脚交替抬放几乎无价"的逐项排查表（`feet_air_time`=0 / AMP 风格项=0 /
  `track_lin`·`track_ang`≈满分 / 唯一成本 `track_default_pose_on_stand` —— 实测权重 −0.3→−0.5 后原始
  cost 0.846→0.722 ⇒ **该项确实在工作，但不足以改掉踏步**）→ 该版 run 日志
* 罚额量级表（weight=4.0 下 d=0.02→0.30 s 对应每步 −0.10→−1.50）+"偏激进但不至于压垮站立"的判读、
  以及备用方案（2k 轮后仍踏步 ⇒ `stand_penalty` 提到 1.0）→ 该版 run 日志

⚠️ 其余逐字不动（**均已随①一并失效**）：`tau 0.08` / `motion_threshold 0.15` / `motion_ang_scale 0.5` /
   `good_lo,hi 0.4/1.0` / `soft 0.3` / `weight 4.0`；`action_rate_l2 −0.06`（今天 **−0.05**）、
   `track_default_pose_on_stand −0.5`（**已被 V23 覆盖 → −0.8**）；AMP 侧口径（quad k=0.125 / GP 1.0 /
   wd 1e-5 / logit_wd 0 / eps 0.05 / λ0.75 / c1.0）**属算法侧 ⇒ 未核实**。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V19d-G1-v0`
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
