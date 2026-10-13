"""AMP 行走 **v6**：v5 全部继承，**只多一项奖励** —— 恢复基类 `joint_deviation_arms`（weight -0.1）。

## ⚠️ 本层对**今天**仍有影响的部分

* ⭐ **这一项今天仍生效**：T1l 的 `joint_deviation_arms = -0.1`（`joint_deviation_l1`）正是本层从基类复原的条目
  ⇒ 至今是"手臂漂移"的持续刹车。
* 奖励项数：本层 **14 → 15**；今天 T1l 为 **非零 17**（后续多轮改动所致，与本层无关）。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | ⭐ 恢复 `joint_deviation_arms`（weight **-0.1**）⇒ 奖励集 14 → 15 项 | ✅ **-0.1**（`joint_deviation_l1`，逐字延续）|

## 推理与判读（已归档，勿在本文件重复）

* 动机「肩/肘异常高举贴限位」（v3c@5390：`right_shoulder_roll` 偏差 **2.03 rad ≈116°**、臂上 Σ(R-L)=+1.739）→ 见该版 run 日志
* 根因「v2 采用 AMP_mjlab 12 项配方时整项删掉 ⇒ 全任务无任何手臂约束」→ 见该版 run 日志
* ⚠️ 不复用 v3 的 `__post_init__`（`AMP_WALK_V3_CHANGES` 是"唯一事实来源"，v4/v5 日志已按 14 项发布）⇒ 新建 v6 保持单变量 → 见该版 run 日志

⚠️ `rewards={n}` 在契约校验里**只打印、不断言**（`PLAN` §11.5）⇒ 14→15 不会 fail-fast。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V6-G1-v0`（算法侧见同节）
"""

from __future__ import annotations

from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .amp_walk_v3_env_cfg import G1AmpWalkV3RewardsCfg
from .amp_walk_v5_env_cfg import (
    G1AmpWalkV5EnvCfg,
    G1AmpWalkV5EnvCfg_PLAY,
)

#: v6 的 run_name（= 日志目录前缀）。`arms` 表示恢复了手臂偏差约束。
AMP_WALK_V6_RUN_NAME = "G1AmpWalkV6Plane_obs626_rew15_ar0.02_ns0.3_mirror_logamp_arms"

#: 恢复的手臂偏差权重 —— **与基类 `G1FlatRewardsCfg.joint_deviation_arms` 逐字一致**。
AMP_WALK_V6_ARMS_WEIGHT = -0.1

#: 基类覆盖的 7 个手臂关节（左右各一套，共 14 个）—— 逐字沿用基类。
AMP_WALK_V6_ARM_JOINTS = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
]


@configclass
class G1AmpWalkV6RewardsCfg(G1AmpWalkV3RewardsCfg):
    """v3 的 14 项 → **15 项**：恢复基类的 `joint_deviation_arms`。

    基类的定义是**无门控**的 `joint_deviation_l1`（对 14 个手臂关节求 |q − q_default|
    之和），所以**走路和站立都生效** —— 这正是需要的：手臂高举在两个状态下都出现。

    ⚠️ 与 `track_default_pose_on_stand` 的区别（为什么不靠那一项）：
       那一项带 `exp(-cmd_speed/0.25)` 门控（实测等效门控质量只有 0.037），
       **走路时基本被关掉**，而且它是 29 关节的 mean（单关节偏 2.03 rad 只贡献 0.070，
       且对左右对称完全盲目）。所以手臂必须靠本项这种**无门控、按关节**的 L1。
    """

    def __post_init__(self):
        super().__post_init__()
        # 基类定义逐字沿用：func / weight / params 完全一致。
        self.joint_deviation_arms = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=AMP_WALK_V6_ARMS_WEIGHT,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(AMP_WALK_V6_ARM_JOINTS),
                )
            },
        )


@configclass
class G1AmpWalkV6EnvCfg(G1AmpWalkV5EnvCfg):
    """训练用：v5（镜像损失 + 站立关风格 + `log` 风格奖励）**全部继承**，只换奖励集。

    15 项 = v3 的 14 项（`action_rate_l2` −0.02、恢复的 `flat_orientation_l2` −1.0
    与 `joint_deviation_torso` −0.1）+ 恢复的 `joint_deviation_arms` −0.1。
    """

    rewards: G1AmpWalkV6RewardsCfg = G1AmpWalkV6RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 父类会写回 AMP_WALK_V5_RUN_NAME，所以必须在它之后再写一次。
        self.run_name = AMP_WALK_V6_RUN_NAME


@configclass
class G1AmpWalkV6EnvCfg_PLAY(G1AmpWalkV5EnvCfg_PLAY):
    """回放用：同样只换奖励集与 run_name（回放时奖励不参与，但保持一致便于对照）。"""

    rewards: G1AmpWalkV6RewardsCfg = G1AmpWalkV6RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V6_RUN_NAME


__all__ = [
    "AMP_WALK_V6_ARM_JOINTS",
    "AMP_WALK_V6_ARMS_WEIGHT",
    "AMP_WALK_V6_RUN_NAME",
    "G1AmpWalkV6EnvCfg",
    "G1AmpWalkV6EnvCfg_PLAY",
    "G1AmpWalkV6RewardsCfg",
]
