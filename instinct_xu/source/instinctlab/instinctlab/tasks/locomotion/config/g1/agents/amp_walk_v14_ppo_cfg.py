"""V14 的算法配置：与 V12 相同，**但 λ/c 改回"从零重训的已验证配方"**。

## ⭐ 为什么这一版 λ 从 0.5 改回 0.75（用户决定 V14 从零重训，2026-09-30）

总回报是 `R = λ·r_task + (1−λ)·c·r_amp`：

| 版本 | λ | c | 任务入账 | 风格入账 | 起点 |
|---|---|---|---|---|---|
| V10b（**唯一一份从零学会走路**的） | **0.75** | 0.25 | **×0.75** | 0.0625·r_amp | 从零 |
| V11 / V12（都是热启动） | 0.5 | 0.5 | ×0.5 | 0.25·r_amp | 从 V10b/ V11 的 ckpt |

⇒ `λ=0.5` 会在**从零**阶段把任务信号砍掉一半（热启动时靠"已经会走"能扛住，从零不行）；
而"从零学会走路"这件事历史上是在 `λ=0.75` 下发生的。
⇒ 本轮 V14 从零 ⇒ **λ=0.75 / c=0.25**（风格项仍然存在、足以把步态往参考拉，但不与任务项抢权重）。
🔁 若你更想严格延续 V12 的算法，把下面两个常量改成 `0.5 / 0.5` 即可（一行）。

| 字段 | V12 | **V14** |
|---|---|---|
| `style_lerp`（λ） | 0.5 | **0.75** |
| `discriminator_reward_coef`（c） | 0.5 | **0.25** |
| `clip_max_std` / `clip_min_std` | 0.6 / 0.02 | 同 |
| `init_noise_std` | 0.35 | 0.35（**从零时生效**） |
| `entropy_coef` / `schedule` / `use_mirror_loss` | 0.005 / fixed / True | 同 |

判别器仍是 **2 帧（134 维）**；本轮**从零开始**（不 resume），所以 `ckpt_manipulator` 不会被用到
（保留着无害：将来若要从 V11 的 670 维 ckpt 热启动仍然需要它）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v11_ppo_cfg import (
    CLIP_MAX_STD_V11,
    CLIP_MIN_STD_V11,
    NOISE_STD_V11,
)
from .amp_walk_v12_ppo_cfg import G1AmpWalkV12PPORunnerCfg

#: λ：任务/风格凸组合（**从零重训用 V10b 验证过的 0.75**）
STYLE_LERP_V14 = 0.75
#: c：风格项缩放 ⇒ 有效风格权重 (1−λ)c = 0.0625
DISC_REWARD_COEF_V14 = 0.25


@configclass
class G1AmpWalkV14PPORunnerCfg(G1AmpWalkV12PPORunnerCfg):
    """v14：环境侧改站姿/限位；算法侧只改 λ/c（适配"从零重训"）。"""

    def __post_init__(self):
        super().__post_init__()

        self.algorithm.style_lerp = STYLE_LERP_V14
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V14
        self.algorithm.clip_max_std = CLIP_MAX_STD_V11
        self.algorithm.clip_min_std = CLIP_MIN_STD_V11
        self.policy.init_noise_std = NOISE_STD_V11
        # 保留（本轮从零用不到；将来从 V11 的 670 维 ckpt 热启动仍需要它）
        self.ckpt_manipulator = "instinctlab.utils.ckpt_manipulators:drop_amp_discriminator"

        self.run_name = (
            f"v14_scratch_amp2frame_standpose0.3_limitmargin0.15w1.5"
            f"_ns{NOISE_STD_V11}_maxstd{CLIP_MAX_STD_V11}_ent0.005_fixedlr"
            f"_lerp{STYLE_LERP_V14}_ampc{DISC_REWARD_COEF_V14}"
        )


__all__ = ["DISC_REWARD_COEF_V14", "STYLE_LERP_V14", "G1AmpWalkV14PPORunnerCfg"]
