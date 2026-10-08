"""V32 实验的环境配置（控制变量实验用）：**A / B / C 三臂共用同一 fork ckpt**。

实验设计（2026-10-06，用户确认）：fork 点 = V31 `model_11500.pt`，三臂各 800 iters。

| 臂 | 环境 cfg | 算法 cfg | 相对 V31 的差异 |
|---|---|---|---|
| **A 对照** | `G1AmpWalkV31EnvCfg`（不改）| `G1AmpWalkV31PPORunnerCfg`（不改）| **无**（纯续训对照）|
| **B 机制对齐** | `G1AmpWalkV32BEnvCfg`（只换 run_name）| `G1AmpWalkV32BPPORunnerCfg` | 白化统计量改成两侧累积 + 重置白化 buffer |
| **C 全结构包** | `G1AmpWalkV32CEnvCfg`（= B + 拆脚手架）| 同 B | B + `feet_air_time 4→2` + `joint_deviation_arms −0.1→0` |

⇒ **C 对 B 只差"拆脚手架"两行**；**B 对 A 只差"白化口径"一处**。三臂同起点、同预算、同种子。

## C 臂两项的依据（都不是拍脑袋）

* `feet_air_time` **4.0 → 2.0**：这是我们**自己加的**手写步态节奏器（AMP_mjlab 的行走奖励集里
  **没有这一项**）。V30 的逐项会计显示它是"买后仰"的钱袋子（+0.33/步）；它的存在让策略有
  "不类人也能拿分"的解。它是 V18 为治"小跳"加的症状级补丁，而当时小跳的根因是**判别器已死**
  ——V30 判别器已活 ⇒ 可以开始撤脚手架。**保留 2.0（而非直接 0）是为了守住 hop 红线**。
* `joint_deviation_arms` **−0.1 → 0.0**：实测**手臂是 default↔reference 差异最大的部位**
  （`right_elbow 0.508`、`left_elbow 0.409`、`left_shoulder_yaw 0.403` rad，且 default 落在
  参考范围**之外**），而这一项正是把手往 default 拉 ⇒ **与 AMP 先验正面对撞**，还把判别器最擅长
  判别的自由度按住。V9c 有先例（当年归零的理由：吃 −2.76/回合 + 抑制手臂探索）。
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .amp_walk_v20_env_cfg import AmpV20PolicyStateObsCfg, AmpV20ReferenceStateObsCfg
from .amp_walk_v29_env_cfg import AMP_V29_STATE_HISTORY
from .amp_walk_v30_env_cfg import _group_with_history
from .amp_walk_v31_env_cfg import G1AmpWalkV31EnvCfg, G1AmpWalkV31EnvCfg_PLAY

AMP_WALK_V32B_RUN_NAME = (
    "G1AmpWalkV32BMVPPlane_obs378_rew21_amp13body2frame_white_bothside_flat4"
)
AMP_WALK_V32C_RUN_NAME = (
    "G1AmpWalkV32CMVPPlane_obs378_rew19_amp13body2frame_white_bothside_flat4_noscaffold"
)

#: ③a 手写步态节奏器：4.0 → 2.0（mjlab 无此项）
AMP_WALK_V32C_AIR_TIME_WEIGHT = 2.0
#: ③b 手臂偏离 default 的惩罚：−0.1 → 0（与 AMP 先验在手臂上正面对撞）
AMP_WALK_V32C_ARMS_DEVIATION_WEIGHT = 0.0


def _reassert_amp_obs(cfg) -> None:
    """父链 `_patch_amp_history()` 会覆盖 AMP 观测组 ⇒ 在 `super().__post_init__()` 之后再确认一次。"""
    cfg.observations.amp_policy = _group_with_history(AmpV20PolicyStateObsCfg, AMP_V29_STATE_HISTORY)
    cfg.observations.amp_reference = _group_with_history(AmpV20ReferenceStateObsCfg, AMP_V29_STATE_HISTORY)


def _reweight(cfg, term: str, weight: float) -> None:
    """只覆盖某个奖励项的 `weight`，`func`/`params` 与父链逐字一致。"""
    base = getattr(cfg.rewards, term)
    setattr(cfg.rewards, term, RewTerm(func=base.func, weight=weight, params=base.params))


@configclass
class G1AmpWalkV32BEnvCfg(G1AmpWalkV31EnvCfg):
    """B 臂：环境侧**与 V31 逐字相同**，只换 run_name（差异全在算法 cfg）。"""

    def __post_init__(self):
        super().__post_init__()
        _reassert_amp_obs(self)
        self.run_name = AMP_WALK_V32B_RUN_NAME


@configclass
class G1AmpWalkV32BEnvCfg_PLAY(G1AmpWalkV31EnvCfg_PLAY):
    """回放用：与训练同口径。"""

    def __post_init__(self):
        super().__post_init__()
        _reassert_amp_obs(self)
        self.run_name = AMP_WALK_V32B_RUN_NAME


@configclass
class G1AmpWalkV32CEnvCfg(G1AmpWalkV32BEnvCfg):
    """C 臂：B + 拆脚手架（`feet_air_time` 4→2、`joint_deviation_arms` −0.1→0）。"""

    def __post_init__(self):
        super().__post_init__()
        _reweight(self, "feet_air_time", AMP_WALK_V32C_AIR_TIME_WEIGHT)
        _reweight(self, "joint_deviation_arms", AMP_WALK_V32C_ARMS_DEVIATION_WEIGHT)
        self.run_name = AMP_WALK_V32C_RUN_NAME


@configclass
class G1AmpWalkV32CEnvCfg_PLAY(G1AmpWalkV32BEnvCfg_PLAY):
    """回放用：与 C 臂训练同奖励。"""

    def __post_init__(self):
        super().__post_init__()
        _reweight(self, "feet_air_time", AMP_WALK_V32C_AIR_TIME_WEIGHT)
        _reweight(self, "joint_deviation_arms", AMP_WALK_V32C_ARMS_DEVIATION_WEIGHT)
        self.run_name = AMP_WALK_V32C_RUN_NAME


__all__ = [
    "AMP_WALK_V32B_RUN_NAME",
    "AMP_WALK_V32C_ARMS_DEVIATION_WEIGHT",
    "AMP_WALK_V32C_AIR_TIME_WEIGHT",
    "AMP_WALK_V32C_RUN_NAME",
    "G1AmpWalkV32BEnvCfg",
    "G1AmpWalkV32BEnvCfg_PLAY",
    "G1AmpWalkV32CEnvCfg",
    "G1AmpWalkV32CEnvCfg_PLAY",
]
