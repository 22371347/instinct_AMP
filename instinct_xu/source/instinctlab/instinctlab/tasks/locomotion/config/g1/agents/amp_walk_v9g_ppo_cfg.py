"""V9g 的算法配置：**让风格项成为主导**（奖励重平衡见 env cfg 的完整说明）。

`style_lerp 0.75 → 0.5`、`discriminator_reward_coef 0.25 → 0.5`
⇒ 有效风格权重 `(1−λ)·c`：**0.0625 → 0.25（4×）**。

理由：v9e 里风格项的"教学力"只有 `(1−λ)·c·std(r_amp) ≈ 0.009/步`，与任务奖励的 env 间涨落
同量级 ⇒ AMP 的"模仿参考"信号被淹没。放宽 λ 后任务项仍占 50%（`λ·r_task`），
但风格项的反差翻了 4 倍 —— 这才是 AMP 该有的权重结构。

其余全部沿用 v9f：`schedule="fixed"`（LR 稳定 1e-3）、σ 初值 0.25 / 上限 0.4、
`entropy_coef 0.0025`、镜像 ON。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v9f_ppo_cfg import G1AmpWalkV9fPPORunnerCfg

#: 任务:风格 的凸组合系数（AMP_mjlab 的 `amp_task_reward_lerp`）
STYLE_LERP_V9G = 0.5
#: 风格项缩放（`WasabiPPO` 的 `discriminator_reward_coef`）
DISC_REWARD_COEF_V9G = 0.5


@configclass
class G1AmpWalkV9gPPORunnerCfg(G1AmpWalkV9fPPORunnerCfg):
    """v9g：v9f + `λ 0.5` / `c 0.5`（风格项权重 ×4）。"""

    def __post_init__(self):
        super().__post_init__()

        self.algorithm.style_lerp = STYLE_LERP_V9G
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V9G

        self.run_name = (
            f"v9g_ns0.25_maxstd0.4_fixedlr_ent0.0025_lerp{STYLE_LERP_V9G}"
            f"_ampc{DISC_REWARD_COEF_V9G}_rebalance_mirror"
        )


__all__ = ["DISC_REWARD_COEF_V9G", "STYLE_LERP_V9G", "G1AmpWalkV9gPPORunnerCfg"]
