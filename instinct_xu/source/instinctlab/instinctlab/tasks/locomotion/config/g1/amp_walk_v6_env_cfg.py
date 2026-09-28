"""AMP 行走 **v6** 的环境配置 —— v5 全部继承，**只多一项奖励**。

唯一改动：恢复基类 `G1FlatRewardsCfg` 的 **`joint_deviation_arms`**（weight −0.1）
⇒ 奖励集 **14 → 15 项**。

为什么（用户实测）
------------------
无论静止还是移动，**肩、肘关节都异常：高举且接近限位**。实测（v3c @5390
的 `joint_error_rmse/<joint>`）：

| 关节 | 偏差 | 备注 |
|---|---|---|
| `right_shoulder_roll` | **2.0338 rad ≈ 116.5°** | 全场最差 |
| `left_shoulder_roll` | 1.5495 rad ≈ 88.8° | |
| 双侧 `elbow` | ≈ 1.49 rad ≈ 85° | |

而 default 姿态是 `elbow +0.600` / `shoulder_pitch +0.200` / `shoulder_roll ±0.200`
⇒ 手臂被抬到离 default 很远的位置。臂上 Σ(R−L) = **+1.739**（腿只有 +0.159）。

原因：v2 采用 AMP_mjlab 的 12 项配方时把 `joint_deviation_arms` 整项**删掉**了
（连同 `flat_orientation_l2` / `joint_deviation_torso`，后两项已在 v3c 恢复）。
⇒ **全任务奖励里没有任何一项约束手臂**，而 AMP 判别器只给整条 transition 一个
全局标量（无法定位到关节），所以手臂可以自由漂到限位。

⚠️ 为什么不复用 v3 的 `__post_init__`：
   v3 的 `AMP_WALK_V3_CHANGES` 是「唯一事实来源」，且 v4/v5 的日志（`rewards=14`）
   已按 14 项发布。所以这里**新建 v6**：v3/v4/v5 语义一字不动，V6 = V5 + 这一项，
   保持"单变量"与历史 run 的可比性。

⚠️ `rewards={n}` 在契约校验里**只打印、不断言**（见 PLAN §11.5），
   所以 14 → 15 不会被 fail-fast 拦下；选项探针见 `cloud_probe_v6_rewards.py`。
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
