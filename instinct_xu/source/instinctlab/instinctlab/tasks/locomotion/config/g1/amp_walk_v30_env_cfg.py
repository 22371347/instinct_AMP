"""V30：环境与 V29 **逐字相同**（AMP 观测 13body×2frame = 390、σ=0.20+退火、奖励表 21 项、`action_scale=1.0`），**只有 run_name 不同**
—— 显式继承 V29 的 env cfg（`__post_init__` 要**再确认一次** AMP 观测口径，防父链 `_patch_amp_history()` 覆盖 → `amp_walk_v20_env_cfg.py:44-48`）。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— env 侧已被后层覆盖**（今天 = `G1AmpWalkT1l...`）；V30 的全部实质改动都在判别器侧
  （`agents/amp_walk_v30_ppo_cfg.py`）⇒ 是否仍生效见 `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V30-G1-v0` 的「算法侧」节。
* ⚠️ 原文"**去白化**"是**幽灵开关**：V23 起"启用白化"很可能从未真正生效（V27"去白化"= no-op）→ `EXPERIMENTS.md` 勘误记录（四）；
  同理"GP λ=10 / GP 仅专家侧 / 优化器分组 wd（trunk 1e-3 / head 1e-2）/ 专家批=离线运动数据集 / 风格强度刻意不动（coef=1.0）"；`RewTerm(...)` 字面值 ≠ 生效值。"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v20_env_cfg import AmpV20PolicyStateObsCfg, AmpV20ReferenceStateObsCfg
from .amp_walk_v23_env_cfg import _group_with_history
from .amp_walk_v29_env_cfg import (
    AMP_V29_STATE_HISTORY,
    G1AmpWalkV29EnvCfg,
    G1AmpWalkV29EnvCfg_PLAY,
    G1AmpWalkV29TeacherEnv,
)

AMP_WALK_V30_RUN_NAME = (
    "G1AmpWalkV30MVPPlane_obs378_rew21_amp13body2frame_nodrop_white_sigma0.20_mjlabdisc"
)


@configclass
class G1AmpWalkV30EnvCfg(G1AmpWalkV29EnvCfg):
    """训练用：与 V29 逐字相同（含 AMP 观测 2 帧），只换 run_name。"""

    def __post_init__(self):
        super().__post_init__()
        # ⚠️ 父链的 `_patch_amp_history()` 会覆盖这两组 ⇒ 必须在 super() 之后再确认一次
        self.observations.amp_policy = _group_with_history(
            AmpV20PolicyStateObsCfg, AMP_V29_STATE_HISTORY
        )
        self.observations.amp_reference = _group_with_history(
            AmpV20ReferenceStateObsCfg, AMP_V29_STATE_HISTORY
        )
        self.run_name = AMP_WALK_V30_RUN_NAME


@configclass
class G1AmpWalkV30EnvCfg_PLAY(G1AmpWalkV29EnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广/同 AMP 观测口径。"""

    def __post_init__(self):
        super().__post_init__()
        self.observations.amp_policy = _group_with_history(
            AmpV20PolicyStateObsCfg, AMP_V29_STATE_HISTORY
        )
        self.observations.amp_reference = _group_with_history(
            AmpV20ReferenceStateObsCfg, AMP_V29_STATE_HISTORY
        )
        self.run_name = AMP_WALK_V30_RUN_NAME


class G1AmpWalkV30TeacherEnv(G1AmpWalkV29TeacherEnv):
    """指令范围与前几代相同（全向四档）⇒ 期望值直接继承。"""


__all__ = [
    "AMP_WALK_V30_RUN_NAME",
    "G1AmpWalkV30EnvCfg",
    "G1AmpWalkV30EnvCfg_PLAY",
    "G1AmpWalkV30TeacherEnv",
]
