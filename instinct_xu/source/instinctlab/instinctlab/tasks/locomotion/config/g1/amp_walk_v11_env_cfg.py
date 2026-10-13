"""V11：**从 V10b 的走路 ckpt 热启动**，主打"保走姿、去碎步、让 AMP 说得上话"。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— 三项奖励改动已被后层全部覆盖**（见下表）。
* 仍"活着"的只有**结论**：`feet_air_time` 的支付量级曾是"反碎步的唯一杠杆"、
  `action_rate_l2` 是**步频代理量**（同时还是奖励集里**主要的密集梯度来源**，§B5），
  以及**续训的两条坑**（见文末）。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | `feet_air_time` **2.0 / threshold 0.5 → 8.0 / threshold 0.6** | ❌ **已被覆盖** → 今天 T1l = `feet_air_time_positive_biped`（`+1` / `threshold=0.5`；中途 V18 换过"目标带"版、T1a2 又换回本条）|
| ② | `action_rate_l2` **−0.02 → −0.06** | ❌ **已被覆盖** → **−0.05**（flat 基类值；T1a 之后）|
| ③ | `dof_acc_l2` **−2.5e-7 → −1.0e-6**（仍只作用髋/膝）| ❌ **已被覆盖** → **−2e-07** + 只作用 `.*_hip_.*` / `.*_knee_joint`（T1l）|
| ④ | 风格压力 `(1−λ)c` **0.0625 → 0.25**（λ 0.75→0.5、c 0.25→0.5）〔算法侧〕| **未核实（超出 EFFECTIVE_REWARDS.md 覆盖范围）** |

## 推理与判读（已归档，勿在本文件重复）

* ⭐ V10b@7930 的"走路成功、但步态是**高频小碎步**"逐条数据（`action_rate_l2` ×4、`dof_acc_l2` ×3、
  `feet_slide` ×2.5、`feet_air_time` ↓半、`amp_d` 更负）+"步频这个维度没有任何一项直接惩罚"
  → 该版 run 日志；相关机制 → §B5（`action_rate_l2` 不只是平滑项，还是主要密集梯度来源）
* ⭐ "风格权重 ×4 无效"（本层重测 λ/c 的动机）→ §A3
* 反小碎步的正确修法是**换掉 `feet_air_time` 的结构**（旧版两个结构性漏洞：短到几毫秒也给分 /
  越长越给 ⇒ "单脚吊着能长期拿满"）→ §B2
* 「从已塌陷 ckpt 续训基本出不来自吸引子」→ §C4；"单脚侧身跳着走"→ §C3

⚠️ 其余逐字不动：21 项奖励结构（含 `feet_swing_time_penalty −0.5`）、三档指令、2-clip 数据集、
   数据层镜像增广 / 镜像损失 ON、fixed LR。
   ⚠️ **续训两条坑（今天仍适用）**：(a) `init_noise_std` 在续训时**不生效** —— σ 从 ckpt 里学到的值继续，
   当时由**上限 0.6** 兜住（`entropy_coef` 仍会把 σ 往上推）；(b) critic 因奖励语义变化需重新适应
   ⇒ 前几百轮**回报先掉再涨是预期**；AMP 参考游标被重置 ⇒ 首轮指标不可信（按"看 300 轮后"读）。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V11-G1-v0`
"""

from __future__ import annotations

import instinctlab.tasks.locomotion.mdp as locomotion_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .amp_walk_v10b_env_cfg import (
    AMP_WALK_V10B_COMMAND_RANGES,
    G1AmpWalkV10bEnvCfg,
    G1AmpWalkV10bEnvCfg_PLAY,
    G1AmpWalkV10bTeacherEnv,
)
from .amp_walk_v9j_env_cfg import G1AmpWalkV9jRewardsCfg

AMP_WALK_V11_RUN_NAME = "G1AmpWalkV11MVPPlane_obs378_rew21_airtime8_ar0.06_bigstep_fromV10b7500"

#: ① 单脚支撑时长奖励：权重与时长上限（小碎步≈0，大步伐≈满分）
AMP_WALK_V11_AIR_TIME_WEIGHT = 8.0
AMP_WALK_V11_AIR_TIME_THRESHOLD = 0.6
#: ②③ 抖动定价（高频碎步的物理特征）
AMP_WALK_V11_ACTION_RATE_WEIGHT = -0.06
AMP_WALK_V11_DOF_ACC_WEIGHT = -1.0e-6


@configclass
class G1AmpWalkV11RewardsCfg(G1AmpWalkV9jRewardsCfg):
    """V9j 的 **21 项**（含 `feet_swing_time_penalty −0.5`）+ 本版三处权重/阈值调整。

    ⚠️ 必须继承 V9j 而不是 V9h：V9h 只有 20 项（没有交替支撑约束）。
    """

    def __post_init__(self):
        super().__post_init__()
        # ① `feet_air_time` 必须整体重建（threshold 在 params 里）
        self.feet_air_time = RewTerm(
            func=locomotion_mdp.feet_air_time_positive_biped,
            weight=AMP_WALK_V11_AIR_TIME_WEIGHT,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "threshold": AMP_WALK_V11_AIR_TIME_THRESHOLD,
            },
        )
        # ② 动作变化率
        self.action_rate_l2 = RewTerm(
            func=self.action_rate_l2.func, weight=AMP_WALK_V11_ACTION_RATE_WEIGHT
        )
        # ③ 关节加速度（只作用髋/膝，逐字沿用基类的 asset_cfg）
        self.dof_acc_l2 = RewTerm(
            func=self.dof_acc_l2.func,
            weight=AMP_WALK_V11_DOF_ACC_WEIGHT,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
        )


@configclass
class G1AmpWalkV11EnvCfg(G1AmpWalkV10bEnvCfg):
    """训练用：V10b 的环境（2-clip / 三档指令 / 镜像增广）+ 步态形状三改。"""

    rewards: G1AmpWalkV11RewardsCfg = G1AmpWalkV11RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V11_RUN_NAME


@configclass
class G1AmpWalkV11EnvCfg_PLAY(G1AmpWalkV10bEnvCfg_PLAY):
    """回放用：与训练同奖励/同分布/同增广。"""

    rewards: G1AmpWalkV11RewardsCfg = G1AmpWalkV11RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V11_RUN_NAME


class G1AmpWalkV11TeacherEnv(G1AmpWalkV10bTeacherEnv):
    """契约期望值只能写在 env 类上（指令范围与 V10b 相同）。"""

    expected_command_ranges = AMP_WALK_V10B_COMMAND_RANGES


__all__ = [
    "AMP_WALK_V11_ACTION_RATE_WEIGHT",
    "AMP_WALK_V11_AIR_TIME_THRESHOLD",
    "AMP_WALK_V11_AIR_TIME_WEIGHT",
    "AMP_WALK_V11_DOF_ACC_WEIGHT",
    "AMP_WALK_V11_RUN_NAME",
    "G1AmpWalkV11EnvCfg",
    "G1AmpWalkV11EnvCfg_PLAY",
    "G1AmpWalkV11RewardsCfg",
    "G1AmpWalkV11TeacherEnv",
]
