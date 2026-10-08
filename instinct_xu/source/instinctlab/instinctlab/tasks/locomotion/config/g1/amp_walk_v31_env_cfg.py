"""V31 的环境配置：与 V30 **逐字相同**（AMP 观测 390 / 奖励表 21 项 / `action_scale=1.0`），
**只改一个数**：`rewards.flat_orientation_l2.weight` 从 **−2.0 → −4.0**。

## 为什么（2026-10-06 的姿态诊断，数据在案）

* 关节树根 = `torso_link`（URDF 实测）⇒ `flat_orientation_l2` 读的就是**躯干绝对倾角**；
* V27 → V30：`Step/tilt_deg_mean` **12.56° → 23.86°**、`tilt_p95` 23.3° → 41.3°、
  `tilt_gt30_frac` 1.2% → **36%**、`Step/grav_bx_mean` −0.037 → **−0.366**（负 = **后仰** ≈21.5°）；
* 参考动作基线（17 clip / 7417 帧实测）：`tilt_mean = 9.0°`、`>30° = 0%`、`g_x = +0.147`（**前倾**）
  ⇒ 策略与参考的姿态相差约 **30°**；
* 而 **AMP 判别器对绝对倾角是盲的**（V20 口径 = 13 体相对 anchor 的位姿/速度；anchor 自身
  倾角不在状态里）⇒ 姿态**只能**靠任务奖励定价，而 `flat_orientation_l2` 是**唯一**管它的项；
* 边际核算：该二次项在 24° 处的边际价格 = 2|w|·sin2θ = 2.97/rad（w=−2），而 V27→V30 的
  "张到 24°"平均换来 ≈4.3/rad 的收益 ⇒ **净梯度仍为正 ⇒ 策略理性地继续后仰**。
  把 |w| 提到 4 ⇒ 边际价格 5.94/rad > 4.3/rad ⇒ 净梯度转负。

## 预登记判据（300~500 轮内）

| 指标 | V30 现状 | 期望 |
|---|---|---|
| `Step/tilt_deg_mean` | 23.86° | **≤ 15°** |
| `Step/grav_bx_mean` | −0.366 | **≥ −0.15** |
| `tilt_gt30_frac` | 0.36 | **≤ 0.10** |
| 不退化闸门 | — | `air_time_single_stance ≥ 0.18`、`feet_air_time ≥ 0.95`、`mean_episode_length ≥ 500` |

⚠️ 若闸门破了（为"站直"把步态压保守）⇒ 退到 **−3** 或改走"门控线性俯仰项"（见后续讨论）。

其余一切都**逐字继承 V30**（含判别器 = mjlab 原始配方、coef=1.0、λ=0.75、σ=0.20+退火）。
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .amp_walk_v29_env_cfg import AMP_V29_STATE_HISTORY
from .amp_walk_v30_env_cfg import (
    G1AmpWalkV30EnvCfg,
    G1AmpWalkV30EnvCfg_PLAY,
    G1AmpWalkV30TeacherEnv,
    _group_with_history,
)
from .amp_walk_v20_env_cfg import AmpV20PolicyStateObsCfg, AmpV20ReferenceStateObsCfg

AMP_WALK_V31_RUN_NAME = (
    "G1AmpWalkV31MVPPlane_obs378_rew21_amp13body2frame_nodrop_white_sigma0.20_flat4"
)

#: 躯干绝对倾角惩罚权重：−2.0（V17 定的、V30 实测已不够）→ **−4.0**
AMP_WALK_V31_FLAT_ORIENTATION_WEIGHT = -4.0


def _apply_v31_rewards(cfg) -> None:
    """只覆盖 `flat_orientation_l2` 的 weight，`func`/`params` 与父链逐字一致。"""
    base = cfg.rewards.flat_orientation_l2
    cfg.rewards.flat_orientation_l2 = RewTerm(
        func=base.func,
        weight=AMP_WALK_V31_FLAT_ORIENTATION_WEIGHT,
        params=base.params,
    )


@configclass
class G1AmpWalkV31EnvCfg(G1AmpWalkV30EnvCfg):
    """训练用：V30 逐字 + `flat_orientation_l2 −2 → −4`。"""

    def __post_init__(self):
        super().__post_init__()
        # ⚠️ 父链的 `_patch_amp_history()` 会覆盖这两组 ⇒ 必须在 super() 之后再确认一次
        self.observations.amp_policy = _group_with_history(
            AmpV20PolicyStateObsCfg, AMP_V29_STATE_HISTORY
        )
        self.observations.amp_reference = _group_with_history(
            AmpV20ReferenceStateObsCfg, AMP_V29_STATE_HISTORY
        )
        _apply_v31_rewards(self)
        self.run_name = AMP_WALK_V31_RUN_NAME


@configclass
class G1AmpWalkV31EnvCfg_PLAY(G1AmpWalkV30EnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广/同 AMP 观测口径。"""

    def __post_init__(self):
        super().__post_init__()
        self.observations.amp_policy = _group_with_history(
            AmpV20PolicyStateObsCfg, AMP_V29_STATE_HISTORY
        )
        self.observations.amp_reference = _group_with_history(
            AmpV20ReferenceStateObsCfg, AMP_V29_STATE_HISTORY
        )
        _apply_v31_rewards(self)
        self.run_name = AMP_WALK_V31_RUN_NAME


class G1AmpWalkV31TeacherEnv(G1AmpWalkV30TeacherEnv):
    """指令范围与前几代相同（全向四档）⇒ 期望值直接继承。"""


__all__ = [
    "AMP_WALK_V31_FLAT_ORIENTATION_WEIGHT",
    "AMP_WALK_V31_RUN_NAME",
    "G1AmpWalkV31EnvCfg",
    "G1AmpWalkV31EnvCfg_PLAY",
    "G1AmpWalkV31TeacherEnv",
]
