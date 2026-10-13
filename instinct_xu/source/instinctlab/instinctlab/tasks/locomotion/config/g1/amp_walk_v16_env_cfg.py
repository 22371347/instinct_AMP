"""V16：**单变量消融 —— 只把 AMP 判别器输入从 2 帧回退到 10 帧**（用户 2026-09-30 指定）。基准 = V15。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— 已被后层完全覆盖**：判别器输入帧数今天 = **2 帧**（`AMP_V29_STATE_HISTORY = 2`），
  且状态构成自 V20 起换成 **13body×15** 口径 → V33 再补 anchor 重力 = **396 维**
  ⇒ 本层的 **670 维 / 10 帧**（`67×10`，旧口径）**已不存在**。
* 唯一仍"活着"的是**方法论**：V11↔V15 的判别器 A/B 设计（对齐有效风格权重 (1−λ)c=0.25、
  同一起点 ckpt、两边判别器**同样从零**）→ 见下 §A2。

## 本层改动一览

| # | 改动 | 现状（今日 T1l 实际生效）|
|---|---|---|
| ① | ⭐ 唯一变量：判别器输入 **2 → 10 帧**（`_patch_amp_history(self, 10)`；67×10 = **670 维**）| **已被覆盖** → **2 帧**：V20/V29 用 **13body×15×2 = 390**，V33 补 anchor 重力 → **396 维** |
| ② | 热启动自 V11 `model_9000.pt`、判别器**丢弃**（`ckpt_manipulator`，与 V15 保持可比特）| 已不适用（V17 起**从零**，见 §C4）|
| ③ | λ=0.75 / c=1.0（有效风格权重仍 0.25，= V11）| 已变（今天 T1l λ=0.975，见 agent cfg）|

## 推理与判读（已归档，勿在本文件重复）

* 「2 帧是 V15 退化的原因」—— **部分成立，但没根治**：10 帧把判别器 loss 0.39→0.13、
  步幅/抖动/跟踪/倾角全面占优，**但倾角仍 22.7°→28.0°、`tilt_gt30_frac` 仍 34%** → §A2
* 离线机制：h=1–2 对**时间结构类**退化「完全瞎」（AUC≈0.5），对纯幅度类任何 h 都能分
  → §A2 / `RESULTS_amp_frames_offline_study.md`
* 「判别器长期被**权重衰减**支配」（`discriminator_weight_decay` ≫ `discriminator_loss`）→ §A2
* 判读三口径（10 帧更好 / 一样差 / 更差）与本版监控清单（`Step/amp_d`、`amp_style_reward`
  斜率、`tilt_deg_mean/p95`、`action_rate_l2`、`feet_air_time`）→ 该版 run 日志 +
  `RESULTS_amp_walk_v16_frame_ablation.md`

⚠️ 「看着多余但别改」：本层 670 维判别器与 V11 ckpt 形状**完全一致**、技术上**可载入**，
   但本轮**仍丢弃** —— 否则 V16 =「10 帧 + 预训练判别器」而 V15 =「2 帧 + 新判别器」
   ⇒ 又多一个变量。要试「带 V11 预训练判别器的 10 帧」，把
   `agents/amp_walk_v16_ppo_cfg.py` 的 `ckpt_manipulator` 置 `None` 即可（一行）。
⚠️ 本轮**不要**同时改 `discriminator_weight_decay_coef`（`3e-4`）或姿态权重
   —— 那会把帧数的效应混掉（可留作 V17）。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V16-G1-v0`
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_env_cfg import AMP_STATE_HISTORY
from .amp_walk_v12_env_cfg import _patch_amp_history
from .amp_walk_v15_env_cfg import (
    G1AmpWalkV15EnvCfg,
    G1AmpWalkV15EnvCfg_PLAY,
    G1AmpWalkV15TeacherEnv,
)

#: ⭐ 唯一变量：判别器输入帧数（V15 的 2 → **10**）
AMP_STATE_HISTORY_V16 = AMP_STATE_HISTORY          # == 10
#: 10 帧时的判别器输入维度：10 × (3+29+29+3+3) = **670**
AMP_STATE_SIZE_V16 = AMP_STATE_HISTORY_V16 * 67

AMP_WALK_V16_RUN_NAME = (
    "G1AmpWalkV16MVPPlane_obs378_rew21_amp10frame_style_c1.0_fromV11"
)


@configclass
class G1AmpWalkV16EnvCfg(G1AmpWalkV15EnvCfg):
    """训练用：V15 逐字沿用，**只把判别器输入回退到 10 帧**（670 维）。"""

    def __post_init__(self):
        super().__post_init__()          # 其中会打 2 帧补丁
        _patch_amp_history(self, AMP_STATE_HISTORY_V16)   # ⭐ 覆盖回 10 帧
        self.run_name = AMP_WALK_V16_RUN_NAME


@configclass
class G1AmpWalkV16EnvCfg_PLAY(G1AmpWalkV15EnvCfg_PLAY):
    """回放用：与训练同观测（含 10 帧判别器输入）、同奖励、同分布、同增广。"""

    def __post_init__(self):
        super().__post_init__()
        _patch_amp_history(self, AMP_STATE_HISTORY_V16)
        self.run_name = AMP_WALK_V16_RUN_NAME


class G1AmpWalkV16TeacherEnv(G1AmpWalkV15TeacherEnv):
    """契约期望值只写在 env 类上（指令范围与 V15/V14/V12/V11/V10b 相同）。

    ⚠️ 判别器输入长度**不在**契约里（`AMP_STATE_SIZE_*` 只是文档常量，
    运行时维度由 obs 群自动推断）⇒ 换帧数不需要动契约校验器。
    """


__all__ = [
    "AMP_STATE_HISTORY_V16",
    "AMP_STATE_SIZE_V16",
    "AMP_WALK_V16_RUN_NAME",
    "G1AmpWalkV16EnvCfg",
    "G1AmpWalkV16EnvCfg_PLAY",
    "G1AmpWalkV16TeacherEnv",
]
