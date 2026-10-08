"""V14：**V12 的退化修正版**（保留 2 帧判别器、回退站姿权重）+ **软边距限位惩罚**。

## 一、为什么（V11 vs V12 同口径对照，2026-09-30 实测）

| 指标 | V11@9390 | V12@13530 | 读法 |
|---|---|---|---|
| `Mean reward`（任务回报/回合） | **27.73** | **9.28** | ❌ 任务回报掉 2/3 |
| `amp_d` / `r_amp` | −0.675 / 0.362 | **−0.407 / 0.716** | ✅ 2 帧判别器**确实生效**（风格信号翻倍） |
| `vx_mid` / `vx_high` RMSE | 0.243 / 0.288 | 0.323 / 0.381 | ❌ 速度跟踪差 ~33% |
| `feet_air_time` | 0.160 | 0.099 | ❌ 更碎步 |
| `action_rate_l2` / `dof_acc_l2` / `feet_slide` | −0.45 / −0.10 / −0.033 | −0.75 / −0.16 / −0.051 | ❌ 抖/滑全面变差 |
| `flat_orientation_l2` | −0.073 | −0.115 | ❌ 倾角更大（更后仰） |
| `track_default_pose_on_stand` | −0.178 | **−0.689** | ⚠️ 多扣 0.51/s ⇒ **×20 = −10.2/回合** |
| `σ` 末值 | 0.44 | 0.54 | 噪声涨回去（碎步/抖动的上游） |

**归因**：V12 掉掉的 ~18.5/回合任务回报里，**一半以上（−10.2）可以直接归到站姿权重的提高**
（`track_default_pose_on_stand −0.3 → −1.0`，而它只在低速段生效，等于对"该省的动作"过度收费）；
剩下的差额是"风格压力变强后**以任务换风格**"（碎步、抖动、跟踪变差都朝同一方向）。

⇒ 本版**回退站姿权重到 −0.3**、**保留 2 帧判别器**（它有效）⇒ 回到我们原本要做的
"只换判别器输入"单变量实验。

## 二、新增：软边距限位惩罚（治"腰部长期顶在 ±0.52"）

两条事实（都已核验）：

1. `waist_pitch_joint` / `waist_roll_joint` 的 URDF 限位是 **±0.52 rad**（`waist_yaw` ±2.618）；
2. 我们奖励集里的限位项 `dof_pos_limits`（权重 −10）**`joint_names` 只写了踝关节**
   （`G1FlatRewardsCfg` 的既有偏离，v2 时代就标注过"仍只作用于踝"）
   ⇒ **腰/髋/膝的限位完全没有被罚**，TB 里的 `Episode_Reward/dof_pos_limits` 只反映踝。

而且 `isaaclab.envs.mdp.joint_pos_limits` 的形式是"限位内严格为 0、越界才线性收费"
⇒ 即使把腰纳入，策略照样能**免费停在限位内侧一点**（这正是"顶住"成为最优解的原因）。

⇒ 本版把它换成新写的 `joint_pos_limits_margin`（见 `mdp/rewards_amp.py`）：

| 参数 | 值 | 含义 |
|---|---|---|
| `asset_cfg.joint_names` | `[".*"]` | **全部关节**（含腰/髋/膝/臂） |
| `margin` | 0.15 rad（≈8.6°） | 距限位 0.15 rad 以内开始收费（此前是"必须越界才收费"） |
| `power` | 2.0 | 进入边距时梯度为 0、越靠近越陡（二次软墙，不"抽"） |
| `weight` | **−1.5** | 单关节顶到硬限位 = −1.5/步（与满额速度跟踪 1.5/步同量级）；贴近限位处斜率 ≈ **−10/rad**，与此前踝关节 `−10×线性` 同量级 |

⇒ 是"把此前**完全没人管**的关节纳入**同级别**约束"，而不是无脑加倍。

## 三、其余逐字不动

21 项奖励结构（含 `feet_air_time 8.0`、`action_rate_l2 −0.06`、`dof_acc_l2 −1e−6`、
`feet_swing_time_penalty −0.5`）、2 帧判别器（134 维）、三档指令、2-clip 数据集、
镜像增广/损失、σ 上限 0.6、λ0.5 / c0.5、`ckpt_manipulator`。

⚠️ 本轮有**两个变量**（① 站姿回退 = 修回归；② 限位软边距 = 新增缺失约束），
判读时分开看：① 看 `Mean reward` 是否回到 27 附近、`feet_air_time` 是否回 0.15+；
② 看 `Step/waist_pitch_abs_mean`（新增探针）与下一轮的实际姿态。
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
