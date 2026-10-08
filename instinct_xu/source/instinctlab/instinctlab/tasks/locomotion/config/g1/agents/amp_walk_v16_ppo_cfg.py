"""V16 的算法配置：**V15 逐字沿用**（λ=0.75 / c=1.0 / σ / 熵 / 镜像损失 / 固定 LR）。

唯一变化在**环境侧**（`amp_walk_v16_env_cfg.py` 把判别器输入从 2 帧回退到 10 帧），
算法超参一个都没动 —— 这样 V15 ↔ V16 的差异才严格只有"帧数"。

## λ / c 的账（为什么保持 V15 的值是对的）

总回报 `R = λ·r_task + (1−λ)·c·r_amp`：

| 版本 | λ | c | 有效风格权重 (1−λ)c | 判别器输入 |
|---|---|---|---|---|
| V11 | 0.5 | 0.5 | **0.25** | 10 帧 670 |
| V15 | 0.75 | 1.0 | **0.25** | 2 帧 134 |
| **V16** | 0.75 | 1.0 | **0.25** | **10 帧 670** |

⇒ V11 与 V16 的**风格权重完全相同**（0.25），差别只在"起点/环境/判别器初始化"；
V15 与 V16 的**风格权重也完全相同**（0.25），差别只在帧数 ⇒ 这正是我们要的 A/B。

## ⚠️ `ckpt_manipulator` 故意保留（丢弃 V11 的判别器）

V16 的判别器输入是 670 维，**与 V11 ckpt 里的判别器形状一致** ⇒ 技术上可以不用丢弃。
但**必须丢**，否则 V16 = "10 帧 + 已训练 9000 轮的判别器"，而 V15 = "2 帧 + 全新判别器"
⇒ 帧数与判别器初始化两个变量混在一起，A/B 失效。

改成"带预训练判别器"的版本（想复现"纯 V11 续训 10 帧"时用）：把下面这行置 `None`。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v15_ppo_cfg import DISC_REWARD_COEF_V15, G1AmpWalkV15PPORunnerCfg
from .amp_walk_v14_ppo_cfg import STYLE_LERP_V14
from .amp_walk_v11_ppo_cfg import CLIP_MAX_STD_V11, CLIP_MIN_STD_V11, NOISE_STD_V11

#: ⭐ 与 V15 **完全相同**（唯一变量在 env 侧：判别器输入帧数 2 → 10）
DISC_REWARD_COEF_V16 = DISC_REWARD_COEF_V15      # == 1.0
STYLE_LERP_V16 = STYLE_LERP_V14                  # == 0.75


@configclass
class G1AmpWalkV16PPORunnerCfg(G1AmpWalkV15PPORunnerCfg):
    """v16：算法侧与 V15 逐字相同（含丢弃判别器的 manipulator）。"""

    def __post_init__(self):
        super().__post_init__()

        self.algorithm.style_lerp = STYLE_LERP_V16
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V16
        self.algorithm.clip_max_std = CLIP_MAX_STD_V11
        self.algorithm.clip_min_std = CLIP_MIN_STD_V11
        self.policy.init_noise_std = NOISE_STD_V11
        # ⭐ 故意保留：让 V16 的判别器也**从零重学**，才能与 V15 严格单变量可比
        #    （V16 的输入 670 维其实与 V11 ckpt 里的判别器同形，不丢也能载入 —— 但那会
        #     引入第二个变量"判别器是否预训练"）
        self.ckpt_manipulator = "instinctlab.utils.ckpt_manipulators:drop_amp_discriminator"

        self.run_name = (
            f"v16_amp10frame_style_c{DISC_REWARD_COEF_V16}_lerp{STYLE_LERP_V16}"
            f"_standpose0.3_limitmargin_fromV11"
            f"_ns{NOISE_STD_V11}_maxstd{CLIP_MAX_STD_V11}_ent0.005_fixedlr"
        )


__all__ = [
    "DISC_REWARD_COEF_V16",
    "STYLE_LERP_V16",
    "G1AmpWalkV16PPORunnerCfg",
]
