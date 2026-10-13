"""V9j：**堵住"单脚吊腿"这条漏洞**（V9h "单脚侧身跳着走"的针对性补丁；2026-09-29）。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— 本层唯一的代码改动（`feet_swing_time_penalty`）已被后层覆盖**（V18 置 0 ⇒ 今天权重 0），见表。
* ⭐ 仍"活着"的是**一条被写进结论的机理教训**：**镜像损失拦不住单侧步态**（本层核心观察）
  —— 其归因已于 **2026-10-08 勘误**（根因是当时**命令分布只给前向速度** ⇒ 镜像损失无计算空间，
  而非"镜像态从未被访问"）→ `docs/amp_walk/EXPERIMENTS.md` §A1。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | 新增 1 项 `feet_swing_time_penalty`（**20 → 21 项**）：`cost = Σ_f clamp(air_time_f − 0.6 s, 0)·1[‖cmd_xy‖>0.1]`，权重 **−0.5**（走路恒 0；单脚吊腿 −0.5/步 压过 `feet_air_time` 的 +1.0/步）| ❌ **已被 V18 覆盖** → 权重 **0**（V18「方案 B」避与目标带对撞）|
| ② | 其余 20 项权重、σ 0.3 / 上限 0.4 / entropy 0.0025 / fixed LR / `style_lerp 0.75` / `discriminator_reward_coef 0.25` / 镜像 ON 一律不动〔算法侧〕| 见 `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V9j-G1-v0` 的「算法侧」节（算法侧值不在此抄录；今天是否仍生效见同表后层各臂）|

## 推理与判读（已归档，勿在本文件重复）

* ⭐ 病灶（V9h@4770 实测）：奖励集里**没有任何一项要求"支撑脚必须换"**
  —— `feet_air_time` 只要 1 只脚着地就给满（不分踩 0.3 s 还是 10 s）、`lin_vel_z_l2` 只管竖直方向
  （`v_z ≈ 0.018 m/s`，几乎无弹跳）⇒ **单脚吊腿是合法最优解**；
  判别器其实健康（`amp_d −0.561` / `amp_d_ref +0.577`，gap 1.14），策略只是待在别的局部最优里
  → §C3（退化模态"单脚侧身跳着走"）
* ⚠️ 原写"镜像损失只有 0.0368，拦不住单侧步态"—— **该归因已勘误**
  （2026-10-08：根因是当时**命令分布只有前向速度** ⇒ 镜像损失无计算空间；且状态左右对称时该值天然偏小，
  **不能拿 0.0368 反推"策略已对称"**）→ §A1（+ `PLAN_amp_walk.md` 镜像等变分析）
* ① 的另一动机：旧 `feet_air_time` 的**结构性漏洞**（短到几毫秒也给分 / 越长越给 ⇒ 单脚吊着能拿满）→ §B2
* 判读口径（≥2000 轮看包络）：新诊断 `Mean step feet_swing_excess` 走路 ≈0 / 吊腿 >0；
  `Episode_Reward/feet_air_time` 应 >0.1/s；`vx_high` RMSE <0.25；`base_contact` <0.5；
  若 2000 轮后仍不见效 ⇒ 才轮到 V9i 的探索量级（σ 0.5 / entropy 0.005，与本事**互补**）→ 该版 run 日志

⚠️ 其余逐字不动：20 项（除新增项外）权重与实现一律沿用 V9h。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V9j-G1-v0`
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
