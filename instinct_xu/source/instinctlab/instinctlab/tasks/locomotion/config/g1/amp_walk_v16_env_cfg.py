"""V16：**单变量消融 —— 只把 AMP 判别器输入从 2 帧回退到 10 帧**（用户 2026-09-30 指定）。

## 要回答的问题

V11（10 帧 AMP，会走路）与 V15（2 帧 AMP，10k 轮后姿态/风格退化）之间，**AMP 侧**到底差什么？

| 项 | V11 | V15 | 是否相同 |
|---|---|---|---|
| λ（`style_lerp`） | 0.5 | 0.75 | ❌ 不同 |
| c（`discriminator_reward_coef`） | 0.5 | 1.0 | ❌ 不同 |
| **有效风格权重 (1−λ)c** | **0.25** | **0.25** | ✅ **相同** |
| 判别器输入帧数 | 10（670 维） | 2（134 维） | ❌ **唯一的结构性差异** |
| 判别器 MLP / 损失 / 学习率 | [1024,512] / BCE / 同 LR | 同 | ✅ |
| 判别器初始化 | V11 起从零训了 9000 轮 | 从零重学 | 需对齐 |

⇒ V11↔V15 的 AMP 侧差异其实**只有两项**：**帧数** 与 **判别器是否预训练**。

## 本轮怎么做（严格单变量）

**V15 的配置逐字不动**，只把判别器输入回退到 10 帧：

```
G1AmpWalkV16EnvCfg  ≔  G1AmpWalkV15EnvCfg
                        + _patch_amp_history(self, AMP_STATE_HISTORY)   # 2 → 10 帧
```

- 环境/奖励/指令/数据集：V15（= V14）逐字沿用（含 `joint_pos_limits_margin`、
  `track_default_pose_on_stand −0.3`、2 帧→10 帧以外的所有项）。
- 算法侧：V15 逐字沿用（λ=0.75、c=1.0 ⇒ 有效风格权重仍是 **0.25**，与 V11 相同）。
- **起点也相同**：热启动自 V11 的 `model_9000.pt`（与 V15 完全同一个 ckpt）。

⇒ 于是 **V15 与 V16 的唯一差异 = 判别器输入帧数（2 vs 10）**，
判别器都从零重学（都带 `ckpt_manipulator`），起点 ckpt 也相同 ⇒ 真正的 A/B。

| | 起点 | 有效风格权重 | 判别器输入 | 判别器初始化 |
|---|---|---|---|---|
| V15 | V11 `model_9000.pt` | 0.25 | 2 帧 134 维 | 从零（丢弃 V11 的 670 维判别器） |
| **V16** | V11 `model_9000.pt` | 0.25 | **10 帧 670 维** | **从零**（同样丢弃，保证可比特） |

## ⚠️ 一个"看起来多余"的地方，别改

V16 的判别器输入是 670 维，**与 V11 的 ckpt 里的判别器形状完全一致**
（`Wasabi.state_dict()` 会把判别器写进 ckpt）⇒ 技术上"不丢"也能载入。
但本轮**仍然丢弃**（保留 `ckpt_manipulator`）：否则 V16 = "10 帧 + 预训练判别器"，
而 V15 = "2 帧 + 新判别器" ⇒ 又多一个变量。要严格单变量，两边判别器必须同样从零开始。

若之后想试"带 V11 预训练判别器的 10 帧"（= 更贴近"纯 V11 续训"），
把 `agents/amp_walk_v16_ppo_cfg.py` 里的 `ckpt_manipulator` 置 `None` 即可（一行）。

## 判读口径（与 V15 同窗口对比）

用 `instinct/tools/cloud_amp_compare.py` 按 250 轮窗口并排看：

| 关注点 | 量 | 2 帧（V15 实测） | 10 帧（V16 期望？） |
|---|---|---|---|
| 判别器是否还"说话" | `Step/amp_d` | 钉在 −0.384（饱和） | 若更有区分度 ⇒ 更负/波动更大 |
| 风格梯度是否还在 | `Step/amp_style_reward` 斜率 | 0.833→0.747 后走平 | 是否持续上升 |
| **姿态（真正的痛点）** | `Step/tilt_deg_mean/p95` | 23.5°→30~35°（后仰） | 是否更小 |
| 抖动 | `Episode_Reward/action_rate_l2` | −0.31→−0.54 | 是否更小 |
| 步态 | `Episode_Reward/feet_air_time` | 0.076→0.10 | 是否回到 0.15+ |
| 判别器健康 | `Loss/discriminator_weight_decay` vs `discriminator_loss` | 74 vs 0.39（**被权重衰减支配**） | 同（与帧数无关） |

**结论怎么写**（三种可能）：
1. 10 帧反而更好（tilt ↓、amp_d 更负但姿态更好）⇒ "2 帧是 V15 退化的原因"成立；
2. 10 帧与 2 帧**一样差** ⇒ 帧数不是主因，主因在**判别器口径/奖励定价**（见 V14/V15 的
   `discriminator_weight_decay ≈ 76` 与 `flat_orientation_l2` 定价过低）；
3. 10 帧更差 ⇒ 2 帧是"更对"的方向（但需要从别处解释 V15 的退化）。

⚠️ 本轮**不要**同时改 `discriminator_weight_decay_coef`（`3e-4`）或姿态权重 ——
那会把帧数的效应混掉（可留作 V17）。
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
