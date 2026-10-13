"""V32：**控制变量实验**（A / B / C 三臂共用同一 fork ckpt = V31 `model_11500.pt`，各 800 iters；2026-10-06 用户确认）。

| 臂 | 环境 cfg | 算法 cfg | 相对 V31 的差异 |
|---|---|---|---|
| **A 对照** | `G1AmpWalkV31EnvCfg`（不改）| V31（不改）| **无**（纯续训对照）|
| **B 机制对齐** | `G1AmpWalkV32BEnvCfg`（只换 run_name）| `G1AmpWalkV32BPPORunnerCfg` | 原设计 = 白化统计量两侧累积 + 重置白化 buffer |
| **C 全结构包** | `G1AmpWalkV32CEnvCfg`（= B + 拆脚手架）| 同 B | B + `feet_air_time 4→2` + `joint_deviation_arms −0.1→0` |

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— 实验臂，不在当前训练链**（T1l 的奖励/算法口径与这里无关；三个臂的值见下表）。
* ⭐⭐ 仍"活着"的是**一条核查结论（今天最重要）**：**2026-10-06 复核确认 B 臂两项改动都是 no-op**
  —— 判别器**根本没接** `normalizer`（运行期 `discriminator_kwargs` 里没有 `normalizer_class_name`；
  ckpt 的 `discriminator` 只有 6 个键、**无** `normalizer.*`；且 **V22 是旁支**，不在 V27~V31 的继承链上）
  ⇒ **V27 以来"白化 / 去白化"整套叙事疑为幽灵开关** ⇒ B 臂已回退成 no-op
  ⇒ **C 对 A 只差环境侧那两个奖励权重**。另**改正一处旧错**：曾写"27% 的步里风格项在与
  `default_pose` 对扣"是错的 —— 运行期 `params/agent.yaml` 实测 `disable_style_on_stand: true`（我们早就开着）
  → 见 `agents/amp_walk_v32b_ppo_cfg.py`（以及 `agents/amp_walk_v27_ppo_cfg.py` / `amp_walk_v29_env_cfg.py` 的同条存疑）

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | B 臂：白化统计量"只用专家侧" → **两侧都用** + **重置白化 buffer**〔算法侧〕| ❌ **复核后回退成 no-op**（判别器没接 normalizer）⇒ B 与 A 算法侧零差异 |
| ② | C 臂：`feet_air_time` **4.0 → 2.0** | ❌ **已被 T1a2 覆盖** → **+1.0**（`feet_air_time_positive_biped` / `threshold=0.5`）|
| ③ | C 臂：`joint_deviation_arms` **−0.1 → 0.0** | ❌ **已被 T1a 白名单覆盖** → **−0.1** |

## 推理与判读（已归档，勿在本文件重复）

* ② 的依据：`feet_air_time` 是**我们自己加的手写步态节奏器**（AMP_mjlab 的行走奖励集里**没有这一项**）；
  V30 的逐项会计显示它是"**买后仰**"的钱袋子（+0.33/步）—— 它让策略有"不类人也能拿分"的解；
  它是 V18 为治"小跳"加的症状级补丁，而当时小跳的根因是**判别器已死**，V30 判别器已活 ⇒ 可以开始撤脚手架；
  **保留 2.0（而非直接 0）是为了守住 hop 红线** → `docs/amp_walk/EXPERIMENTS.md` §B2
* ③ 的依据：实测**手臂是 default↔reference 差异最大的部位**（`right_elbow 0.508` / `left_elbow 0.409` /
  `left_shoulder_yaw 0.403` rad，且 default 落在参考范围**之外**）⇒ 该项与 AMP 先验**正面对撞**，
  还把判别器最擅长判别的自由度按住；有先例：V9c 归零（理由：吃 −2.76/回合 + 抑制手臂探索）
  → 见该版 run 日志（**EXPERIMENTS.md 未单列此条**）
* ⚠️ "白化可能本就不存在"这条**尚未进 `EXPERIMENTS.md`**（v27/v29 稿里都标注为"待归档"）；
  相关但**不等价**的既有条目 = §A4（`raw` GP 陷阱；白化确实改善离线区分性 AUC 0.850 → 0.927、`style_std` ×2.7）

⚠️ 其余逐字不动：三臂同起点 / 同预算 / 同种子；B 对 A 只差"白化口径"一处，C 对 B 只差"拆脚手架"两行。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V32B-G1-v0`（B 臂）/ `-V32C-G1-v0`（C 臂）
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
