"""V31：与 V30 **逐字相同**，只改一个数 —— 躯干绝对倾角惩罚 `flat_orientation_l2` 的 weight **−2.0 → −4.0**（2026-10-06）。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— 该权重已被后层覆盖**：今天 `flat_orientation_l2 = −1.0`（**T1a 白名单抄 T0 值**），不是本层的 −4.0。
  ⚠️ 本层 `run_name` 里带 `white` 字样，但 2026-10-06 复核发现判别器**根本没接 `normalizer`**
  （V22 是旁支）⇒ **该 tag 是幽灵开关**（详见 `amp_walk_v32_env_cfg.py` 的 B 臂复核记录）。
* 仍"活着"的**结论**：`flat_orientation_l2` 是奖励集里**唯一**给"躯干绝对姿态"定价的项，
  而 **AMP 判别器对绝对倾角是盲的**（V20 口径 = 13 体相对 anchor 的位姿/速度，**anchor 自身倾角不在状态里**）
  ⇒ 姿态**只能**靠这一项管。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | ⭐ 唯一变量：`flat_orientation_l2` **−2.0 → −4.0**（`func`/`params` 与父链逐字一致）| ❌ **已被 T1a 白名单覆盖** → **−1.0** |

## 推理与判读（已归档，勿在本文件重复）

* **依据（2026-10-06 姿态诊断，数据在案）**：关节树根 = `torso_link`（URDF 实测）⇒ 该项读的**就是躯干绝对倾角**；
  V27→V30 实测 `Step/tilt_deg_mean` **12.56° → 23.86°**、`tilt_p95` 23.3° → 41.3°、`tilt_gt30_frac` 1.2% → **36%**、
  `Step/grav_bx_mean` −0.037 → **−0.366**（负 = **后仰** ≈21.5°）；而**参考动作基线**（17 clip / 7417 帧实测）
  `tilt_mean = 9.0°` / `>30° = 0%` / `g_x = +0.147`（**前倾**）⇒ 策略与参考的姿态相差约 **30°** → 该版 run 日志
* **边际核算**：该二次项在 24° 处的边际价格 = `2|w|·sin2θ` = **2.97/rad**（w=−2），而 V27→V30 的"张到 24°"
  平均换来 ≈**4.3/rad** 的收益 ⇒ **净梯度仍为正 ⇒ 策略理性地继续后仰**；把 `|w|` 提到 4
  ⇒ 边际价格 5.94/rad > 4.3/rad ⇒ **净梯度转负** → 该版 run 日志
* **预登记判据（300~500 轮内）**：`tilt_deg_mean ≤ 15°`、`grav_bx_mean ≥ −0.15`、`tilt_gt30_frac ≤ 0.10`；
  不退化闸门 `air_time_single_stance ≥ 0.18`、`feet_air_time ≥ 0.95`、`mean_episode_length ≥ 500`；
  闸门破 ⇒ 退到 **−3** 或改走"门控线性俯仰项" → 该版 run 日志
* 相关既有条目：倾角退化"**没根治**"（10 帧判别器只是减缓）→ `docs/amp_walk/EXPERIMENTS.md` §A2；
  判别器 `raw` GP 陷阱 + 白化确实改善离线区分性 → §A4

⚠️ 其余一切都**逐字继承 V30**：奖励表（21 项）/ 判别器 = mjlab 原始配方 / `coef=1.0` / `λ=0.75` / σ=0.20+退火。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V31-G1-v0`
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
