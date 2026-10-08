"""V11：**从 V10b 的走路 checkpoint 热启动**，主打"保走姿、去碎步、让 AMP 说得上话"。

## 为什么这么做（V10b@7930 的复盘）

✅ **走路本身成功了**：中位回合长度跑满 1000 步（20 s）、`base_contact` 0.375（历史最低）、
`vx_high` RMSE 0.212（历史最低）、`track_lin_vel` 1.89/s（历史最高）、σ 稳在 0.58。

❌ **但步态是"高频小碎步"**，数据逐条指认：

| 指标 | V9h | V10b@7930 | 读法 |
|---|---|---|---|
| `action_rate_l2` | −0.077/s | **−0.320/s（4×）** | ⭐ 动作变化率 = **步频代理量** ⇒ 高频抖 |
| `dof_acc_l2` | −0.0133 | **−0.0426（3×）** | 关节加速度抖 |
| `feet_slide` | −0.020 | −0.051（2.5×） | 滑步增多 |
| `dof_pos_limits` | −0.021 | **−0.082（4×）** | σ=0.58 的代价 |
| **`feet_air_time`** | 0.0410 | **0.0191（↓半）** | ⭐ 单支撑时长更短 ⇒ **步长短/步频高** |
| `amp_d` | −0.561 | **−0.781** | 策略**更不像**参考 |
| `r_amp` | 0.50 | 0.238 | 风格奖励更低 |

⇒ 反小碎步的只有 `feet_air_time` 一项，而它的支付只有 **0.38/回合（总回报 34 的 1%）**，
而"步频"这个维度**没有任何一项直接惩罚** ⇒ 策略选了它最容易维持的模态。

## 三个改动（都在"步态形状"这一组概念变量内）

| # | 项 | V10b | **V11** | 理由 |
|---|---|---|---|---|
| ① | `feet_air_time` | 2.0 / threshold 0.5 | **8.0 / threshold 0.6** | 小碎步该项≈0，大步伐≈满分 ⇒ 支付从 0.38/回合 提到 **~9/回合**（相对 34 的量级） |
| ② | `action_rate_l2` | −0.02 | **−0.06** | 高频小步的物理特征就是动作高频 ⇒ 直接定价 |
| ③ | `dof_acc_l2` | −2.5e-7 | **−1.0e-6** | 同上（加速度抖） |
| ④ | 风格压力 `(1−λ)c` | 0.0625（λ0.75,c0.25） | **0.25（λ0.5,c0.5）** | 上次 ×4 无效是在"17 clip + 坏门控"下；现在 2-clip + 门控正确 ⇒ 重测 |

**其余逐字不动**：21 项奖励结构（含 `feet_swing_time_penalty −0.5`）、三档指令
（静止 0.30 / 纯前进 0.35 / 前向斜向 0.35）、2-clip 数据集、数据层镜像增广 ON、
镜像损失 ON、fixed LR、σ 上限见 agent cfg。

## ⚠️ 续训的注意事项

- **`init_noise_std` 在续训时不生效**：σ 从 checkpoint 里**学到的值**继续（V10b 末值 0.58）
  ⇒ 本轮的 σ 由 **上限 0.6** 兜住，而不是初值。`entropy_coef 0.005` 仍会把 σ 往上推。
- 价值函数/critic 会因奖励语义变化重新适应 ⇒ 前几百轮**回报先掉再涨是预期**。
- AMP 参考动作播放游标会被重置 ⇒ 首轮指标不可信（按 §4 的"看 300 轮后"）。
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
