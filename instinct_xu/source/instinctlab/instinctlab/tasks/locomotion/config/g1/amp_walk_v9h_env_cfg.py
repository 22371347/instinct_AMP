"""V9h：抗跳 + 恢复平衡刹车 + 补回被删的基类项（唯一新设计 = 躯干竖直速度惩罚）。

## ⚠️ 本层对**今天**仍有关键影响的一点（2026-10-08 核查确认）

**`track_base_height` 的核宽 `std` 0.3 → 0.2 是本层引入的**
（`AMP_WALK_V9H_BASE_HEIGHT_STD = 0.2`），并**一路继承到 T1a/T1a2/T1j/T1k/T1l**。
后层只改 `weight` 不改 `params` ⇒ T1k 把 weight 恢复成 1.5 时，**std 仍留在 0.2**。
⇒ 这就是 2026-10-08 由 T1l 的失败断言当场抓出的“隐藏偏离”（V5 原始值是 0.3 = AMP_mjlab 值）。

## 本层改动一览

| 类别 | 改动 |
|---|---|
| **会传给后代的参数** | `track_base_height` **0.5/std0.3 → 1.0/std0.2**；`track_ang_vel_z_exp` 0.2→**0.8**；`body_ang_vel_xy` 0.2→**0.5**；`track_default_pose_on_stand` 0.0→**−0.3**；`track_lin_vel_xy_exp`/`feet_air_time` 保持 3.0/2.0 |
| **奖励集 16→20 项** | 4 项均从基类**复原**（非新设计）：`lin_vel_z_l2 −0.2`（基类值加倍）/ `dof_torques_l2` / `joint_deviation_hip` / `joint_deviation_knee`。刻意**不**恢复 `stand_still(−0.8)`（与 `track_default_pose_on_stand` 同族，重复惩罚）|
| **推理与判读** | 已归档，**勿在本文件重复**：→ `docs/amp_walk/EXPERIMENTS.md` §C1（“95% 摔倒”只是断崖快照）/ §C2（三个刹车砍过头）/ §A3（风格权重 ×4 无效）/ §附待办第 6–7 条（`termination_penalty` 账目反推、`amp_stand_frac` 不一致）|

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V9h-G1-v0`
"""

from __future__ import annotations

import instinctlab.envs.mdp as instinct_mdp
from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .amp_walk_v9e_env_cfg import G1AmpWalkV9eEnvCfg, G1AmpWalkV9eEnvCfg_PLAY, G1AmpWalkV9eRewardsCfg

#: run_name = 日志目录前缀（把关键改动写进目录名，便于事后对账）
AMP_WALK_V9H_RUN_NAME = (
    "G1AmpWalkV9hMVPPlane_obs378_rew20_linvel3_airtime2_antihop_vz_nohipdev"
    "_ns0.3_lerp0.75_ampc0.25_mirror"
)

#: 相对 V9g 的权重覆盖（只改 weight，不动 func/params）
AMP_WALK_V9H_WEIGHTS = {
    "track_lin_vel_xy_exp": 3.0,
    "feet_air_time": 2.0,
    "feet_slide": -0.1,
    "track_ang_vel_z_exp": 0.8,          # 0.2 → 0.8（恢复旋转刹车）
    "body_ang_vel_xy": 0.5,              # 0.2 → 0.5（恢复躯干刹车）
    "track_base_height": 1.0,            # 0.5 → 1.0
    "track_default_pose_on_stand": -0.3,  # 0.0 → −0.3（小权重保留静止姿态稳定）
}
#: `track_base_height` 的核宽：0.3 → 0.2（跳 = 高度大幅振荡 ⇒ 收紧即定价）
AMP_WALK_V9H_BASE_HEIGHT_STD = 0.2
#: ⭐ 本版唯一"新设计"：躯干竖直速度惩罚（基类 −0.1，这里加倍）
AMP_WALK_V9H_LIN_VEL_Z_WEIGHT = -0.2
#: 以下三项权重**逐字沿用基类 `G1FlatRewardsCfg`**
AMP_WALK_V9H_TORQUE_WEIGHT = -4.0e-6
AMP_WALK_V9H_HIP_DEV_WEIGHT = -0.1
AMP_WALK_V9H_KNEE_DEV_WEIGHT = -0.05


@configclass
class G1AmpWalkV9hRewardsCfg(G1AmpWalkV9eRewardsCfg):
    """v9e 的 16 项（含 `feet_air_time`）+ 4 项基类恢复项 ⇒ **20 项**。"""

    # --- 基类恢复项（func/params 逐字沿用 `G1FlatRewardsCfg`）----------------
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=AMP_WALK_V9H_LIN_VEL_Z_WEIGHT)
    dof_torques_l2 = RewTerm(
        func=instinct_mdp.joint_torques_l2,
        weight=AMP_WALK_V9H_TORQUE_WEIGHT,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=AMP_WALK_V9H_HIP_DEV_WEIGHT,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=AMP_WALK_V9H_KNEE_DEV_WEIGHT,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])},
    )

    def __post_init__(self):
        super().__post_init__()
        # 权重覆盖（见 AMP_WALK_V9H_WEIGHTS）
        for name, w in AMP_WALK_V9H_WEIGHTS.items():
            term = getattr(self, name, None)
            if term is None:
                raise AttributeError(f"V9h 想改权重，但奖励集里没有 {name!r}")
            term.weight = w
        # `track_base_height` 换核宽（0.3 → 0.2）⇒ 必须整体重建（params 与 weight 一起写）。
        # `func` 直接取继承来的那个（= `locomotion_mdp.track_base_height_exp`），逐字不改。
        self.track_base_height = RewTerm(
            func=self.track_base_height.func,
            weight=AMP_WALK_V9H_WEIGHTS["track_base_height"],
            params={"std": AMP_WALK_V9H_BASE_HEIGHT_STD},
        )
        # 4 个恢复项再写一次（与 v9e 同样的防坑：父类可能把它们置 None）
        self.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=AMP_WALK_V9H_LIN_VEL_Z_WEIGHT)
        self.dof_torques_l2 = RewTerm(
            func=instinct_mdp.joint_torques_l2,
            weight=AMP_WALK_V9H_TORQUE_WEIGHT,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
        )
        self.joint_deviation_hip = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=AMP_WALK_V9H_HIP_DEV_WEIGHT,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
        )
        self.joint_deviation_knee = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=AMP_WALK_V9H_KNEE_DEV_WEIGHT,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])},
        )


@configclass
class G1AmpWalkV9hEnvCfg(G1AmpWalkV9eEnvCfg):
    """训练用：v9e 的环境（指令分布 / 观测 / 数据集全不动），只换奖励集与 run_name。"""

    rewards: G1AmpWalkV9hRewardsCfg = G1AmpWalkV9hRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V9H_RUN_NAME


@configclass
class G1AmpWalkV9hEnvCfg_PLAY(G1AmpWalkV9eEnvCfg_PLAY):
    """回放用：与训练同奖励/同分布（回放时奖励不生效，保持一致便于对照）。"""

    rewards: G1AmpWalkV9hRewardsCfg = G1AmpWalkV9hRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V9H_RUN_NAME


__all__ = [
    "AMP_WALK_V9H_BASE_HEIGHT_STD",
    "AMP_WALK_V9H_HIP_DEV_WEIGHT",
    "AMP_WALK_V9H_KNEE_DEV_WEIGHT",
    "AMP_WALK_V9H_LIN_VEL_Z_WEIGHT",
    "AMP_WALK_V9H_RUN_NAME",
    "AMP_WALK_V9H_TORQUE_WEIGHT",
    "AMP_WALK_V9H_WEIGHTS",
    "G1AmpWalkV9hEnvCfg",
    "G1AmpWalkV9hEnvCfg_PLAY",
    "G1AmpWalkV9hRewardsCfg",
]
