"""V19a：**逐字沿用 V18，只把步态目标带的「下侧衰竭尺度」`soft` 0.3 → 0.4**（2026-10-02）。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— 已被后层完全覆盖**：整条"目标带"实现（`feet_air_time_target_biped`，含本层调的 `soft`）
  已在 **T1a2** 换回基类旧版 `feet_air_time_positive_biped`（`threshold=0.5`、`weight=+1`）
  ⇒ 本层调的 `soft` 今天不存在。
* 仍"活着"的是**结论**（见下表与推理小节）：核宽必须 ≥ 下边界，否则下侧进死区 ⇒ 该项恒 0 且无梯度。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | ⭐ 唯一变量：`feet_air_time_target_biped(soft 0.3 → 0.4)`（等于 `good_lo`；上侧 `good_hi+soft` 1.3 → 1.4 s）| ❌ **已被 T1a2 覆盖** → `feet_air_time_positive_biped`（`+1` / `threshold=0.5`）|

## 推理与判读（已归档，勿在本文件重复）

* ⭐ **为什么必须改**：核函数 `score = 1 − clamp((good_lo−d)/soft,0,1) − clamp((d−good_hi)/soft,0,1)`；
  V18 取 `soft=0.3 < good_lo=0.4` ⇒ 下侧在 `d < good_lo−soft = 0.1 s` 处**饱和**（`clamp` 到 1）
  ⇒ `score ≡ 0`，**奖励与梯度同时恒为 0**；而 V18 实测单支撑只有 `0.009~0.011 s < 0.1 s`
  ⇒ 该奖励**从头到尾没工作过**，"小跳"没有任何一项在定价它（`Step/air_time_in_band_frac` 恒 0）→ 该版 run 日志
* 取 `soft = good_lo = 0.4` 后下侧退化成 `score = d/0.4` ⇒ 在 `d ∈ [0,0.4]` 上**处处有梯度**
  （`∂score/∂d = 2.5/s`；`d=0.01 s` 时每步 ≈ 0.025 —— 很小，但**不再是 0**，足以 bootstrap）→ 该版 run 日志
* ⚠️ 旧 `feet_air_time_positive_biped` 的两个结构性漏洞（短到几毫秒也给分 / 越长越给）
  → `docs/amp_walk/EXPERIMENTS.md` §B2

⚠️ 其余逐字不动：V18 的 21 项奖励结构 / 5 clip 数据集 / 四档全向指令 / `action_rate_l2 −0.10` /
   `flat_orientation_l2 −2` / `feet_swing_time_penalty 0` / 三个跟踪项 3.0·0.8·1.0。
   ⚠️ 本层的"目标带 + `soft 0.4`"口径此后已被 V19b/V19c/V19d 继续改写、最终被 T1a2 换掉
   （今天 = 5 clip + 全向分档指令，grep 旁证、**未运行期核对**）。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V19a-G1-v0`
"""

from __future__ import annotations

import instinctlab.tasks.locomotion.mdp as locomotion_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .amp_walk_v18_env_cfg import (
    AMP_WALK_V18_AIR_TIME_GOOD_HI,
    AMP_WALK_V18_AIR_TIME_GOOD_LO,
    AMP_WALK_V18_AIR_TIME_WEIGHT,
    G1AmpWalkV18EnvCfg,
    G1AmpWalkV18EnvCfg_PLAY,
    G1AmpWalkV18RewardsCfg,
    G1AmpWalkV18TeacherEnv,
)

#: ⭐ 本轮唯一改动：下侧衰竭尺度 0.3 → **0.4**（= `good_lo`）
#:   ⇒ `d < 0.1 s` 的"恒 0 且无梯度"死区消失（见模块 docstring 的推导）。
AMP_WALK_V19A_AIR_TIME_SOFT = 0.4

AMP_WALK_V19A_RUN_NAME = (
    "G1AmpWalkV19aMVPPlane_obs378_rew21_amp10frame_flat2.0_airband_soft0.4_ar0.10_5clip_omnidir"
)


@configclass
class G1AmpWalkV19ARewardsCfg(G1AmpWalkV18RewardsCfg):
    """V18 的 21 项逐字保留，只替换 `feet_air_time` 的 `soft`。"""

    def __post_init__(self):
        super().__post_init__()

        # ⭐ 只改 `soft`：`func` / `weight` / 其余 params 与 V18 完全一致
        self.feet_air_time = RewTerm(
            func=locomotion_mdp.feet_air_time_target_biped,
            weight=AMP_WALK_V18_AIR_TIME_WEIGHT,          # 4.0，不动
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "good_lo": AMP_WALK_V18_AIR_TIME_GOOD_LO,   # 0.4，不动
                "good_hi": AMP_WALK_V18_AIR_TIME_GOOD_HI,   # 1.0，不动
                "soft": AMP_WALK_V19A_AIR_TIME_SOFT,        # 0.3 → 0.4 ⭐
            },
        )


@configclass
class G1AmpWalkV19AEnvCfg(G1AmpWalkV18EnvCfg):
    """训练用：V18 全部 + 步态带下侧消死区。"""

    rewards: G1AmpWalkV19ARewardsCfg = G1AmpWalkV19ARewardsCfg()

    def __post_init__(self):
        super().__post_init__()          # 含 V18 的 `_apply_v18_dataset_commands`（5 clip / 全向指令）
        self.run_name = AMP_WALK_V19A_RUN_NAME


@configclass
class G1AmpWalkV19AEnvCfg_PLAY(G1AmpWalkV18EnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广。"""

    rewards: G1AmpWalkV19ARewardsCfg = G1AmpWalkV19ARewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V19A_RUN_NAME


class G1AmpWalkV19ATeacherEnv(G1AmpWalkV18TeacherEnv):
    """指令范围与 V18 相同（全向四档）⇒ 期望值直接继承 V18。"""


__all__ = [
    "AMP_WALK_V19A_AIR_TIME_SOFT",
    "AMP_WALK_V19A_RUN_NAME",
    "G1AmpWalkV19AEnvCfg",
    "G1AmpWalkV19AEnvCfg_PLAY",
    "G1AmpWalkV19ARewardsCfg",
    "G1AmpWalkV19ATeacherEnv",
]
