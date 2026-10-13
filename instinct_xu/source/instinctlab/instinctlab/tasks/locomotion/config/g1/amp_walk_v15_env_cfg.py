"""V15：环境**逐字沿用 V14**；唯一改动在算法侧（`discriminator_reward_coef` 0.25 → 1.0，风格分量 ×4）。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— 已被后层整体覆盖**；今日生效值 `discriminator_reward_coef = 4.0`（非本层的 1.0）
  ⇒ 本层改动一览仅①这一项（`agents/amp_walk_v15_ppo_cfg.py`）。

## 推理与判读（已归档，勿在本文件重复）

* ⭐ 「风格权重 ×4 能改善风格」**已实测证伪**（`amp_d` -0.433→-0.407 且伴随崩溃振荡）→ `EXPERIMENTS.md` §A3
* 「均值倾角小 ≠ 任何时刻都竖直」+ 本层起的倾角分布探针（`Step/tilt_p95_deg`、`[LIMIT-TOP]` 等）→ 见该版 run 日志

⚠️ `RewTerm(...)` 字面值 **≠ 生效值**（后层整体覆盖 `params`）→ 见 `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V15-G1-v0`（算法侧见同节）
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v14_env_cfg import (
    G1AmpWalkV14EnvCfg,
    G1AmpWalkV14EnvCfg_PLAY,
    G1AmpWalkV14TeacherEnv,
)

AMP_WALK_V15_RUN_NAME = "G1AmpWalkV15MVPPlane_obs378_rew21_amp2frame_style_x4_c1.0_fromV11"


@configclass
class G1AmpWalkV15EnvCfg(G1AmpWalkV14EnvCfg):
    """训练用：与 V14 同环境/同奖励（差别只在 agent cfg 的 c）。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V15_RUN_NAME


@configclass
class G1AmpWalkV15EnvCfg_PLAY(G1AmpWalkV14EnvCfg_PLAY):
    """回放用：与训练同奖励/同分布/同增广。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V15_RUN_NAME


class G1AmpWalkV15TeacherEnv(G1AmpWalkV14TeacherEnv):
    """契约期望值只写在 env 类上（指令范围与 V14/V12/V11 相同）。"""


__all__ = [
    "AMP_WALK_V15_RUN_NAME",
    "G1AmpWalkV15EnvCfg",
    "G1AmpWalkV15EnvCfg_PLAY",
    "G1AmpWalkV15TeacherEnv",
]
