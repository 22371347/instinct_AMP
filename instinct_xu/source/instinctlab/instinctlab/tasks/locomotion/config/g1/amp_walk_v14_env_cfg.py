"""V14：**V12 的退化修正版**（保留 2 帧判别器、回退站姿权重）+ **新增软边距限位惩罚**
（2026-09-30，依据 V11 vs V12 同口径对照）。

## ⚠️ 本层对**今天**仍有影响的部分

* ⚠️ **两个改动今天都已被覆盖**（见下表）。仍"活着"的是本层确立的**判读口径**：
  "**只在低速段生效的惩罚项，其价格会被'该省的动作'过度征收**"（→ §B1）。
* ⭐ 本层把限位项扩到**全关节**的思路**今天没生效** —— `dof_pos_limits` **至今仍是"仅踝"**
  ⇒ 腰 / 髋 / 膝的限位**今天依旧没被罚**，问题挂在 §附待办 **#2**。

## 本层改动一览（⚠️ 本轮 **2 个变量**，判读须分开看 —— 见下）

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | `track_default_pose_on_stand` **−1.0 → −0.3**（修 V12 回归）| ❌ **已被 V23 覆盖** → **−0.8** + 全关节 `pos_joint_weights`（knee 2.0 等），**至今生效** |
| ② | 限位项换新写的 `joint_pos_limits_margin`（`[".*"]` 全关节 / `margin=0.15` rad / `power=2` / `weight=−1.5`）| ❌ **已被 T1a2 覆盖** → 回到 `joint_pos_limits`（**仅踝**、`weight=−1`）；软边距思路见 §附待办 #2 |
| ③ | 保留 **2 帧**判别器（当时 134 维 = 关节系）| ⚪ 帧数 **2 仍沿用**（V33-S 起 = 2×198 body 系、396 维）；"134 维关节系"表示已被 body 系取代 |

## 推理与判读（已归档，勿在本文件重复）

* **V11 vs V12 同口径对照 → 归因**：V12 掉的 ~18.5/回合任务回报里**一半以上（−10.2）可直接归到
  站姿权重的提高**（`−0.3 → −1.0`，而它只在低速段生效）；其余差额是"**以任务换风格**"
  （碎步 / 抖动 / 跟踪变差都朝同一方向）→ `docs/amp_walk/EXPERIMENTS.md` §B1
  （⚠️ 同处提醒：**不要**把这条读成"该项要调小"—— V23 又加码到 −0.8 + knee 2.0 且至今生效）
* **软边距限位的两条事实**：1. `waist_pitch/roll` URDF 限位 ±0.52 rad；2. 奖励集里的
  `dof_pos_limits`（−10）`joint_names` **只写了踝**（`G1FlatRewardsCfg` 的既有偏离）⇒ 腰 / 髋 / 膝
  完全没被罚；且 `joint_pos_limits` 是"越界才线性收费" ⇒ 可**免费停在限位内侧一点**
  → §附待办 #2；实现见 `mdp/rewards_amp.py`
* ⏱️ 判读建议（① 看 `Mean reward` 是否回 27 附近、`feet_air_time` 是否回 0.15+；
  ② 看 `Step/waist_pitch_abs_mean` 探针与下一轮的实际姿态）→ 见该版 run 日志

## 本轮的"不动"清单

⚠️ 其余逐字不动：21 项奖励结构（含 `feet_air_time 8.0`、`action_rate_l2 −0.06`、`dof_acc_l2 −1e−6`、
   `feet_swing_time_penalty −0.5`）、三档指令、2-clip 数据集、镜像增广 / 损失、σ 上限 0.6、
   λ0.5 / c0.5、`ckpt_manipulator`。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V14-G1-v0`
"""

from __future__ import annotations

import instinctlab.tasks.locomotion.mdp as locomotion_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .amp_walk_v10b_env_cfg import AMP_WALK_V10B_COMMAND_RANGES
from .amp_walk_v12_env_cfg import (
    G1AmpWalkV12EnvCfg,
    G1AmpWalkV12EnvCfg_PLAY,
    G1AmpWalkV12RewardsCfg,
    G1AmpWalkV12TeacherEnv,
)

AMP_WALK_V14_RUN_NAME = (
    "G1AmpWalkV14MVPPlane_obs378_rew21_amp2frame_standpose0.3_limitmargin_fromV11"
)

#: ① 回退：站姿权重（V12 用 −1.0；V9h/V10b/V11 都是 −0.3）
AMP_WALK_V14_STAND_POSE_WEIGHT = -0.3
#: ② 软边距限位惩罚：距限位 0.15 rad 以内开始按二次曲线收费，权重 −1.5
AMP_WALK_V14_LIMIT_MARGIN = 0.15
AMP_WALK_V14_LIMIT_POWER = 2.0
AMP_WALK_V14_LIMIT_WEIGHT = -1.5


@configclass
class G1AmpWalkV14RewardsCfg(G1AmpWalkV12RewardsCfg):
    """V12 的 21 项（逐字继承）+ ① 站姿权重回退 + ② 限位项换软边距版本（全关节）。"""

    def __post_init__(self):
        super().__post_init__()

        # ① 站姿回退（只改 weight，func/params —— 含 gate_tau 与 pos_joint_weights —— 原样保留）
        base_pose = self.track_default_pose_on_stand
        self.track_default_pose_on_stand = RewTerm(
            func=base_pose.func,
            weight=AMP_WALK_V14_STAND_POSE_WEIGHT,
            params=base_pose.params,
        )

        # ② 限位：踝 → **全关节**，且"越界才罚" → "距限位 margin 内就开始二次罚"
        self.dof_pos_limits = RewTerm(
            func=locomotion_mdp.joint_pos_limits_margin,
            weight=AMP_WALK_V14_LIMIT_WEIGHT,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "margin": AMP_WALK_V14_LIMIT_MARGIN,
                "power": AMP_WALK_V14_LIMIT_POWER,
            },
        )


@configclass
class G1AmpWalkV14EnvCfg(G1AmpWalkV12EnvCfg):
    """训练用：V12（2 帧判别器 / 三档指令 / 2-clip）+ 站姿回退 + 限位软边距。"""

    rewards: G1AmpWalkV14RewardsCfg = G1AmpWalkV14RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V14_RUN_NAME


@configclass
class G1AmpWalkV14EnvCfg_PLAY(G1AmpWalkV12EnvCfg_PLAY):
    """回放用：与训练同奖励/同分布/同增广。"""

    rewards: G1AmpWalkV14RewardsCfg = G1AmpWalkV14RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V14_RUN_NAME


class G1AmpWalkV14TeacherEnv(G1AmpWalkV12TeacherEnv):
    """契约期望值只能写在 **env 类**上（指令范围与 V12/V11/V10b 相同）。"""

    expected_command_ranges = AMP_WALK_V10B_COMMAND_RANGES


__all__ = [
    "AMP_WALK_V14_LIMIT_MARGIN",
    "AMP_WALK_V14_LIMIT_POWER",
    "AMP_WALK_V14_LIMIT_WEIGHT",
    "AMP_WALK_V14_RUN_NAME",
    "AMP_WALK_V14_STAND_POSE_WEIGHT",
    "G1AmpWalkV14EnvCfg",
    "G1AmpWalkV14EnvCfg_PLAY",
    "G1AmpWalkV14RewardsCfg",
    "G1AmpWalkV14TeacherEnv",
]
