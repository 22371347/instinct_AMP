"""V9h：**抗跳 + 恢复平衡刹车 + 补回被删的基类项**（唯一一项新设计是"竖直速度"）。

## 为什么这样改（V9g 的复盘）

1. **V9g 的"95% 摔倒"是断崖窗口的快照，不是稳态**。日志显示 V9g 反复出现
   `rew 37 → 0.31（len=79）` 这种**崩一下再恢复**的振荡，崩溃时刻伴随：
   `amp_style_std 0.10 → 0.19~0.24`（风格奖励跨 env 离散度翻倍）、`d −0.39 → −0.51/−0.56`、
   `|a| 0.48 → 0.58~0.61`。好轮次其实能到 `rew 45~46 / len 750~795`（≈ v9f 水平）。
   ⇒ 曲线要按"包络"看，别拿单个快照下结论。
2. 但 **V9g 把三个平衡刹车砍过头了**（`track_ang_vel_z 1.2→0.2`、`body_ang_vel_xy 0.7→0.2`、
   `track_base_height 1.5→0.5`），同时把 `track_lin_vel` 抬到 3.0 ⇒ 跳跃性价比是我自己调出来的。
3. **风格权重 ×4（`(1−λ)c` 0.0625→0.25）实测无效**：`amp_d` 从 −0.433 只变到 −0.407，
   且伴随上条崩溃振荡 ⇒ 本版**回退到 0.75/0.25**（= v9e/v9f 的稳定值）。

## 本版奖励集：16 → **20 项**（4 项全部是"基类里本来就有、v2 被删掉"的，不是新设计）

| 项 | 权重 | 作用 |
|---|---|---|
| `lin_vel_z_l2` | **−0.2** | ⭐ **躯干竖直速度**。跳：`vz≈±1 m/s` → `−0.2/步`；走：`vz≈±0.1` → 几乎免费。**直接给跳跃定价**（基类原值 −0.1，这里加倍） |
| `dof_torques_l2` | −4.0e-6 | 髋/膝力矩（基类原值）⇒ "猛蹬"有代价 |
| `joint_deviation_hip` | −0.1 | 髋 yaw/roll 偏差（基类原值）⇒ 腿不歪、不劈叉 |
| `joint_deviation_knee` | −0.05 | 膝偏差（基类原值）⇒ 守住屈膝默认姿态 |

**刻意不恢复 `stand_still`（−0.8）**：它和 `track_default_pose_on_stand` 是同一族（"命令小时别乱动关节"），
本版后者已按用户意见以 −0.3 小权重保留 ⇒ 加它是重复惩罚，且可能压制"从站姿起步"。

## 权重调整（相对 V9g）

| 项 | V9g | **V9h** | 理由 |
|---|---|---|---|
| `track_ang_vel_z_exp` | 0.2 | **0.8** | 恢复旋转刹车 |
| `body_ang_vel_xy` | 0.2 | **0.5** | 恢复躯干刹车 |
| `track_base_height` | 0.5 / std 0.3 | **1.0 / std 0.2** | 跳 = 高度大幅振荡，收紧核直接定价 |
| `track_default_pose_on_stand` | 0.0 | **−0.3** | 用户：小权重保留静止姿态稳定 |
| `track_lin_vel_xy_exp` | 3.0 | 3.0 | 保住"动起来"的动力 |
| `feet_air_time` | 2.0 | 2.0 | 唯一的步态时序项（"恰好单脚着地"） |

## 判读（≥3000 轮）

- ⭐ `Episode_Reward/feet_air_time` **> 0.05 /s**（离开 0.003 的"从未单脚支撑"区）⇒ 真的开始交替支撑；
- `Episode_Termination/base_contact` **< 0.5**（V9f 0.38 / V9g 0.95）；
- `Mean reward` 包络（不看单点）回到 ≥45；
- 新增诊断 `Mean step amp_d_ref`（参考侧判别器输出）：**`gap = d_ref − d_actor` 是否明显** ⇒
  直接回答"AMP 还有没有分辨力"，这是本版的核心观测目标。

## 已知未解（下一轮要单独查）

- `termination_penalty` 名义 −200，但账目反推每回合只有 ≈ −2.3（95% 摔倒时也一样）⇒ **摔倒可能几乎不痛**。
- `amp_stand_frac` 实测 0.05，但 `prob_low=0.30` 的设计值应 ≈0.30 ⇒ 指令分档或门控读的量可能不一致。
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
