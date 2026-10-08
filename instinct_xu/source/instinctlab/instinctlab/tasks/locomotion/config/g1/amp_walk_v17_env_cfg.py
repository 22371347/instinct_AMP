"""V17：**从零开始**训练人类步行风格（用户 2026-10-01 指定）——基准 = V16（10 帧 AMP）。

## 本轮改的两项（用户 2026-10-01 决定：一并改，早于“只改一项”的原计划）

| # | 项 | V16 | **V17** | 理由 |
|---|---|---|---|---|
| ① | `flat_orientation_l2` 权重 | −1 | **−2** | 倾角是唯一仍在恶化的量（V16 22.7°→28.0°）；该项 $=\sin^2\theta$，28° 时只收 0.127/步（`track_lin_vel_xy` 是 +1.45/步）⇒ 定价过低 |
| ② | `feet_air_time` 权重 | 8.0 | **4.0（减半）** | 降低“单脚跳刷分”的收益（V14 前车之鉴：该项的核是“单支撑 $\min$(接触,腾空) clamp ≤0.6”，一脚踩久+另一脚抬着每步能白拿 ≈4.8） |

⚠️ 这两项都只是 **`weight`** 改变，`func`/`params`（含 air-time 的 `threshold 0.6`、
`sensor_cfg`、`command_name`）逐字不变。

## 为什么要动 `flat_orientation_l2`（定价核算）

V15(2 帧) ↔ V16(10 帧) 消融结论（见 `RESULTS_amp_walk_v16_frame_ablation.md`）：
10 帧让判别器分辨力 ×2.9（`Loss/discriminator_loss` 0.386→0.134），行为全面更像人
（倾角 30.3°→28.0°、步幅 0.104→0.241、抖动 −0.568→−0.356、跟踪 1.302→1.447）。
**但唯一仍在恶化的量是躯干倾角**：V16 从 22.7°（起点）一路爬到 28.0°，`tilt_gt30_frac` 仍有 34%。

| 倾角 θ | `flat_orientation_l2` 每步收费（−1） | **−2 后** | 对比 |
|---|---|---|---|
| 15° | 0.067 | 0.134 | `track_lin_vel_xy` +1.45/步 |
| 28° | 0.127 | **0.254** | `feet_air_time` +0.12/步（权重减半后） |
| 35° | 0.170 | 0.340 | |

## ⚠️ 本轮与 V16 的差异有三处，判读时必须知道

1. `flat_orientation_l2` −1 → **−2**；
2. `feet_air_time` 8.0 → **4.0**；
3. **从零开始**（V16 是热启动 V11 `model_9000.pt`）⇒ 没有“会走路的起点”可利用。

⇒ 因此**不能**把 V17 与 V16 直接比“风格好坏”（起点不同）；V17 要看的是：
**它能不能从零学会走路、学出来的姿态是不是更直、步态是不是自然（非跳）**。

## ⚠️ 早期监控：单脚跳的止损信号（权重减半后阈值要按新权重看）

`Episode_Reward/feet_air_time` 在 TB 里是 **权重×核值** ⇒ 权重减半后数值也减半：

| 信号 | 健康（权重 4.0） | 疑似跳/单脚 ⇒ 止损 |
|---|---|---|
| `Episode_Reward/feet_air_time`（每步） | **0.05 ~ 0.15**（= V16 的 0.10~0.24 减半） | **≥ 0.3**（单脚姿态的核值 ~0.5~0.6 ⇒ 8.0 时是 1.31，4.0 时 ~0.65） |
| `Episode_Reward/feet_swing_time_penalty` | ≈ −0.0005 | 明显更负（−0.05 以上） |
| `Step/tilt_deg_mean` | 随训练下降 | 高且不降 |

⇒ 若 **~1.5k 轮内** `feet_air_time` ≥ 0.3 且倾角不降 ⇒ 权重减半**没有**治好结构陷阱
（陷阱来自核函数形状，不只是权重），下一版（V18）需把 air-time 改成
“**落地触发、每次迈步只付一次**”，再叠加直立惩罚。

## ⚠️ 另一个“从零”必须面对的点：没有相位约束

`motion_start_from_middle_range = (0.0, 0.0)`（关闭 RSI）⇒ 所有 env 的参考 clip 从第 0 帧
开始、**不与策略相位对齐**；从零阶段策略还没形成步态时，判别器给出的只是“姿态是否像人”，
不会拉出步频。若 V17 从零能站直但步态碎/不前进，下一刀应加 **步频/相位**类奖励或
打开 RSI —— 属于 V18 的变量。

## 其余逐字不动

环境（含 `joint_pos_limits_margin`、`track_default_pose_on_stand −0.3`）、其余 19 项奖励、
10 帧判别器（670 维）、三档指令、2-clip 数据集、镜像增广/损失、σ 上限 0.6、固定 LR、
λ 0.75 / c 1.0（有效风格权重 0.25）。
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .amp_walk_v14_env_cfg import G1AmpWalkV14RewardsCfg
from .amp_walk_v16_env_cfg import (
    G1AmpWalkV16EnvCfg,
    G1AmpWalkV16EnvCfg_PLAY,
    G1AmpWalkV16TeacherEnv,
)

#: ⭐ 改动①：直立惩罚 −1 → **−2**
AMP_WALK_V17_FLAT_ORIENTATION_WEIGHT = -2.0
#: ⭐ 改动②：反小碎步/反单脚的 `feet_air_time` 权重 8.0 → **4.0（减半）**
AMP_WALK_V17_AIR_TIME_WEIGHT = 4.0

AMP_WALK_V17_RUN_NAME = (
    "G1AmpWalkV17MVPPlane_obs378_rew21_amp10frame_flat2.0_airtime4.0_scratch"
)


@configclass
class G1AmpWalkV17RewardsCfg(G1AmpWalkV14RewardsCfg):
    """V14/V16 的 21 项（逐字继承）+ ① 直立惩罚 ×2 + ② air-time 权重减半。

    两项都**只改 `weight`**，`func` / `params`（含 air-time 的 `threshold 0.6`、
    `sensor_cfg=ankle_roll_link`、`command_name`）原样保留。
    """

    def __post_init__(self):
        super().__post_init__()

        base = self.flat_orientation_l2
        self.flat_orientation_l2 = RewTerm(
            func=base.func,
            weight=AMP_WALK_V17_FLAT_ORIENTATION_WEIGHT,
            params=base.params,
        )

        air = self.feet_air_time
        self.feet_air_time = RewTerm(
            func=air.func,
            weight=AMP_WALK_V17_AIR_TIME_WEIGHT,
            params=air.params,
        )


@configclass
class G1AmpWalkV17EnvCfg(G1AmpWalkV16EnvCfg):
    """训练用：V16（10 帧判别器 / 三档指令 / 2-clip / 限位软边距）+ 直立惩罚 ×2。"""

    rewards: G1AmpWalkV17RewardsCfg = G1AmpWalkV17RewardsCfg()

    def __post_init__(self):
        super().__post_init__()          # 其中会打 10 帧补丁（V16 的 __post_init__）
        self.run_name = AMP_WALK_V17_RUN_NAME


@configclass
class G1AmpWalkV17EnvCfg_PLAY(G1AmpWalkV16EnvCfg_PLAY):
    """回放用：与训练同观测（10 帧判别器）/同奖励/同分布/同增广。"""

    rewards: G1AmpWalkV17RewardsCfg = G1AmpWalkV17RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V17_RUN_NAME


class G1AmpWalkV17TeacherEnv(G1AmpWalkV16TeacherEnv):
    """契约期望值只写在 env 类上（指令范围与 V16/V15/V14/V12/V11/V10b 相同）。"""


__all__ = [
    "AMP_WALK_V17_AIR_TIME_WEIGHT",
    "AMP_WALK_V17_FLAT_ORIENTATION_WEIGHT",
    "AMP_WALK_V17_RUN_NAME",
    "G1AmpWalkV17EnvCfg",
    "G1AmpWalkV17EnvCfg_PLAY",
    "G1AmpWalkV17RewardsCfg",
    "G1AmpWalkV17TeacherEnv",
]
