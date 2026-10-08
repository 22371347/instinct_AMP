"""V9j：**堵住"单脚吊腿"这条漏洞**（V9h 的"单脚侧身跳着走"的针对性补丁）。

## 病灶（V9h@4770 实测）

| 证据 | 数值 | 含义 |
|---|---|---|
| 步态观察 | "单脚侧身跳着走" | 一侧脚一直踩地、另一侧吊着，靠上身侧倾推进 |
| `Episode_Reward/lin_vel_z_l2` | **−0.0034/s** ⇒ `v_z ≈ 0.018 m/s` | **几乎没有竖直弹跳** ⇒ 抗跳项"生效了"，但只在竖直方向 |
| 镜像损失 | **0.0368**（很小） | **镜像损失拦不住单侧步态**（见 `rewards_amp.py` 的机理说明） |
| `Episode_Reward/feet_air_time` | 0.0410/s（还在缓升） | "恰好一只脚着地"被**永久满足** ⇒ 该项在**奖励**单脚吊腿 |
| `amp_d` / `amp_d_ref` | −0.561 / +0.577 ⇒ **gap 1.14** | 判别器健康、AMP 没失效，策略只是待在别的局部最优里 |

⇒ 奖励集里**没有任何一项**要求"支撑脚必须换"：`feet_air_time` 只要 1 只脚着地就给满
（不区分踩了 0.3 s 还是 10 s），`lin_vel_z_l2` 只管竖直方向 ⇒ 单脚吊腿成了合法最优解。

## 唯一改动：新增 1 项 `feet_swing_time_penalty`（20 → **21 项**）

```
cost = Σ_f clamp(air_time_f − 0.6 s, 0) · 1[‖cmd_xy‖ > 0.1]     权重 −0.5
```

- **走路**：摆动相 0.2~0.5 s ⇒ **恒为 0（完全免费）**；
- **单脚吊腿**：腾空 1 s ⇒ −0.5/步（−25/s），**压过** `feet_air_time` 最多 +1.0/步的支付
  ⇒ "把脚放下来"成为唯一划算的走法，而放下来就必然进入交替支撑。

其余**全部不动**（20 项里的其他权重、σ 0.3 / 上限 0.4 / entropy 0.0025 / fixed LR /
`style_lerp 0.75` / `discriminator_reward_coef 0.25` / 镜像 ON）⇒ 与 V9h 只差这一项。

## 判读（≥2000 轮，看包络）

| 判据 | 期望 |
|---|---|
| 新诊断 `Mean step feet_swing_excess` | 走路 → **≈ 0**；单脚吊腿 → 明显 > 0 ⇒ 直接量化"脚放没放下来" |
| `Episode_Reward/feet_air_time` | **冲上去**（>0.1/s，真交替时每步最多 1.0） |
| `vx_high` RMSE | 保持 < 0.25（V9h 已到 0.2185，历史最好） |
| `Episode_Termination/base_contact` | < 0.5（V9h 0.449） |
| `Mean reward` | 若真开始交替，会明显高于 V9h 的 46.6（`feet_air_time` 满分可达 +800/回合） |

若 2000 轮后 `feet_swing_excess` 仍大、`feet_air_time` 仍 < 0.1 ⇒ 说明"交替"在**探索**上
到不了，此时才轮到 V9i 的探索量级（σ 0.5 / entropy 0.005），两者是互补的。
"""

from __future__ import annotations

import instinctlab.tasks.locomotion.mdp as locomotion_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .amp_walk_v9h_env_cfg import (
    AMP_WALK_V9H_RUN_NAME,
    G1AmpWalkV9hEnvCfg,
    G1AmpWalkV9hEnvCfg_PLAY,
    G1AmpWalkV9hRewardsCfg,
)

AMP_WALK_V9J_RUN_NAME = AMP_WALK_V9H_RUN_NAME.replace("V9h", "V9j").replace(
    "_antihop_vz_nohipdev", "_antihop_vz_nohipdev_swingpen"
)

#: ⭐ 唯一新增项：单脚超长腾空的惩罚权重（见 `rewards_amp.py::feet_swing_time_penalty`）
AMP_WALK_V9J_SWING_PENALTY_WEIGHT = -0.5
#: 允许的最长腾空（s）：走路摆动相 0.2~0.5 s ⇒ 0.6 留余量，正常步态完全免费
AMP_WALK_V9J_MAX_AIR_TIME = 0.6


@configclass
class G1AmpWalkV9jRewardsCfg(G1AmpWalkV9hRewardsCfg):
    """V9h 的 20 项 + `feet_swing_time_penalty` ⇒ **21 项**（强制交替支撑）。"""

    feet_swing_time_penalty = RewTerm(
        func=locomotion_mdp.feet_swing_time_penalty,
        weight=AMP_WALK_V9J_SWING_PENALTY_WEIGHT,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "max_air_time": AMP_WALK_V9J_MAX_AIR_TIME,
            "command_threshold": 0.1,
        },
    )


@configclass
class G1AmpWalkV9jEnvCfg(G1AmpWalkV9hEnvCfg):
    """训练用：V9h 的环境 + 新增的交替支撑约束。"""

    rewards: G1AmpWalkV9jRewardsCfg = G1AmpWalkV9jRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V9J_RUN_NAME


@configclass
class G1AmpWalkV9jEnvCfg_PLAY(G1AmpWalkV9hEnvCfg_PLAY):
    """回放用：与训练同奖励/同分布。"""

    rewards: G1AmpWalkV9jRewardsCfg = G1AmpWalkV9jRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V9J_RUN_NAME


__all__ = [
    "AMP_WALK_V9J_MAX_AIR_TIME",
    "AMP_WALK_V9J_RUN_NAME",
    "AMP_WALK_V9J_SWING_PENALTY_WEIGHT",
    "G1AmpWalkV9jEnvCfg",
    "G1AmpWalkV9jEnvCfg_PLAY",
    "G1AmpWalkV9jRewardsCfg",
]
