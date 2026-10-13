"""V19b：**V19a 全部不动，只把步态目标带的「下侧」换成近零端加密的指数形状**（`tau=0.08 s`；2026-10-02）。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— 已被后层完全覆盖**：整条"目标带"实现已在 **T1a2** 换回 `feet_air_time_positive_biped`
  （`threshold=0.5`、`weight=+1`）⇒ 本层新增的指数下侧 `tau` 今天不存在。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | ⭐ 唯一变量：下侧由线性（V19a `soft`）改为**指数** `below(d) = clamp((exp(−d/τ) − exp(−good_lo/τ)) / (1 − exp(−good_lo/τ)), 0, 1)`，`τ = 0.08 s`；`soft` 退回 V18 原值 0.3（此时**只管上侧**，上侧在 1.3 s 归零）| ❌ **已被 T1a2 覆盖** → `feet_air_time_positive_biped`（`+1` / `threshold=0.5`）|
| ② | 起点 = **从零训练**（V19a 是热启动 V18）| ❌ 已不适用（今天由 T1l 的起点决定）|

## 推理与判读（已归档，勿在本文件重复）

* ⭐ **动机（"路径坡度不够"）**：步态项上限 `4.0×0.6 ≈ 2.4/步`，V19a 只拿到 **0.038/步 = 1.6%**，
  而 `track_lin_vel_xy_exp` 已 1.28/步（接近饱和）⇒ 上涨空间全在步态项；但线性下侧在 `d=0.01 s`
  处梯度只有 2.5/s；指数下侧在近零端下凸 ⇒ 同位置梯度 ≈12.5/s，且 0.35 s 后已≈1，不与 `[0.4,1.0]` 打架
  （完整数值表在 `rewards_amp.py`）→ 该版 run 日志
* **三种下侧形状对照**（`d = 0.01 s` 处的 score）：V18 = **0（梯度恒 0，死区）** /
  V19a = 0.025（梯度 2.5/s）/ V19b = **0.118（梯度 ≈12.5/s）**；
  实测 `Step/air_time_single_stance`：V18 0.0096 s / V19a 0.0096 → 0.0111 s（**1150 轮平台化**）→ 该版 run 日志
* ⚠️ **起点也是变量之一**：V18（从零）与 V19a（热启动 V18）**都**落在小跳模态 ⇒ "热启动"不是唯一原因，
  而"小跳在早期无价"是共同点 ⇒ 本版把"从零 + 近零端加密"两条一起改，才能同时回答"起点"与"定价"→ 该版 run 日志
* ⚠️ 同臂的**算法 cfg**（`agents/amp_walk_v19b_ppo_cfg.py`）另把 `amp_quad_k` 0.25 → 0.125
  （`d` 展宽后 5% 样本越过 `d ≤ −1` 的纯死区；k=0.125 把下端推到 −1.83）〔算法侧〕
  → 见 `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V19b-G1-v0` 的「算法侧」节

⚠️ 其余逐字不动：V19a 的 21 项奖励结构 / 5 clip 数据集 / 四档全向指令 / `action_rate_l2 −0.10`。
   ⚠️ 参考集与指令分布在此后已被 V18 系列沿用、并被 V33 收口 —— **grep 旁证、未运行期核对**。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V19b-G1-v0`
"""

from __future__ import annotations

import instinctlab.tasks.locomotion.mdp as locomotion_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .amp_walk_v18_env_cfg import (
    AMP_WALK_V18_AIR_TIME_GOOD_HI,
    AMP_WALK_V18_AIR_TIME_GOOD_LO,
    AMP_WALK_V18_AIR_TIME_SOFT,
    AMP_WALK_V18_AIR_TIME_WEIGHT,
)
from .amp_walk_v19a_env_cfg import (
    G1AmpWalkV19AEnvCfg,
    G1AmpWalkV19AEnvCfg_PLAY,
    G1AmpWalkV19ARewardsCfg,
    G1AmpWalkV19ATeacherEnv,
)

#: ⭐ 本轮唯一改动：下侧改为指数型（近零端加密），时间尺度 0.08 s。
#:   含义：`d = 0.08 s` 处短缺已衰减到 1/e ≈ 0.37；`d = 0.4 s`（= good_lo）处恰好 0。
AMP_WALK_V19B_AIR_TIME_TAU = 0.08

AMP_WALK_V19B_RUN_NAME = (
    "G1AmpWalkV19bMVPPlane_obs378_rew21_amp10frame_flat2.0_airband_exptau0.08_ar0.10_5clip_omnidir"
)


@configclass
class G1AmpWalkV19BRewardsCfg(G1AmpWalkV19ARewardsCfg):
    """V19a 的 21 项逐字保留，只替换 `feet_air_time` 的**下侧形状**（加 `tau`）。"""

    def __post_init__(self):
        super().__post_init__()

        self.feet_air_time = RewTerm(
            func=locomotion_mdp.feet_air_time_target_biped,
            weight=AMP_WALK_V18_AIR_TIME_WEIGHT,        # 4.0，不动（形状改了，权重不必抬）
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "good_lo": AMP_WALK_V18_AIR_TIME_GOOD_LO,   # 0.4，不动
                "good_hi": AMP_WALK_V18_AIR_TIME_GOOD_HI,   # 1.0，不动
                # ⚠️ `tau` 一开，`soft` **就只管上侧**（`d > good_hi` 的线性衰竭）。
                #    这里取 **V18 原值 0.3** ⇒ 上侧在 `1.0+0.3 = 1.3 s` 归零
                #    （V19a 是 0.4 ⇒ 1.4 s；差 0.1 s，可忽略）。
                "soft": AMP_WALK_V18_AIR_TIME_SOFT,         # 0.3（V18 原值）：仅上侧
                "tau": AMP_WALK_V19B_AIR_TIME_TAU,          # ⭐ 新增：下侧指数化 0.08 s
            },
        )


@configclass
class G1AmpWalkV19BEnvCfg(G1AmpWalkV19AEnvCfg):
    """训练用：V19a 全部 + 步态带下侧指数化（近零端加密）。"""

    rewards: G1AmpWalkV19BRewardsCfg = G1AmpWalkV19BRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V19B_RUN_NAME


@configclass
class G1AmpWalkV19BEnvCfg_PLAY(G1AmpWalkV19AEnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广。"""

    rewards: G1AmpWalkV19BRewardsCfg = G1AmpWalkV19BRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V19B_RUN_NAME


class G1AmpWalkV19BTeacherEnv(G1AmpWalkV19ATeacherEnv):
    """指令范围与 V19a 相同（全向四档）⇒ 期望值直接继承。"""


__all__ = [
    "AMP_WALK_V19B_AIR_TIME_TAU",
    "AMP_WALK_V19B_RUN_NAME",
    "G1AmpWalkV19BEnvCfg",
    "G1AmpWalkV19BEnvCfg_PLAY",
    "G1AmpWalkV19BRewardsCfg",
    "G1AmpWalkV19BTeacherEnv",
]
