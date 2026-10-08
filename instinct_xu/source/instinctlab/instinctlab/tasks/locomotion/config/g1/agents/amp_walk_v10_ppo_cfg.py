"""V10 的算法配置：**探索量级**（= V9i 的旋钮）+ 其余沿用 V9h/V9j。

| 字段 | V9h / V9j | **V10** | 参照 |
|---|---|---|---|
| `policy.init_noise_std` | 0.3 | **0.6**（±34°） | 仓库里不含 AMP 的行走配方 `G1OmniLocoPPORunnerCfg` = **1.0**（±57°） |
| `algorithm.clip_max_std` | 0.4 | **0.7**（硬天花板） | 参考配方无上限；但 v8 无上限时 σ 失控到 1.04 rad |
| `algorithm.clip_min_std` | 0.0125 | **0.03** | 5% × init 的比例 |
| `algorithm.entropy_coef` | 0.0025 | **0.005** | 参考配方 = 0.008 |
| `schedule` | fixed | fixed | 不动（参考配方是 adaptive，留作后续单变量） |
| `style_lerp` / `discriminator_reward_coef` | 0.75 / 0.25 | 0.75 / 0.25 | 不动（×4 已证无效） |
| `use_mirror_loss` | True | True | 不动（策略层约束；数据层增广是**补充**不是替代） |
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v9i_ppo_cfg import G1AmpWalkV9iPPORunnerCfg

#: 探索噪声初值（±34°）
NOISE_STD_V10 = 0.6
#: σ 上限（有界：避免 v8 那种无上限失控）
CLIP_MAX_STD_V10 = 0.7
#: σ 下限（5% × init）
CLIP_MIN_STD_V10 = 0.03
#: 熵系数
ENTROPY_COEF_V10 = 0.005


@configclass
class G1AmpWalkV10PPORunnerCfg(G1AmpWalkV9iPPORunnerCfg):
    """V10：探索量级（σ 0.6 / 上限 0.7 / ent 0.005）+ 其余沿用 V9h/V9j。"""

    def __post_init__(self):
        super().__post_init__()

        self.policy.init_noise_std = NOISE_STD_V10
        self.algorithm.clip_max_std = CLIP_MAX_STD_V10
        self.algorithm.clip_min_std = CLIP_MIN_STD_V10
        self.algorithm.entropy_coef = ENTROPY_COEF_V10

        self.run_name = (
            f"v10_ns{NOISE_STD_V10}_maxstd{CLIP_MAX_STD_V10}_ent{ENTROPY_COEF_V10}"
            f"_fixedlr_lerp0.75_ampc0.25_rew21_aug_cmdLT"
        )


__all__ = [
    "CLIP_MAX_STD_V10",
    "CLIP_MIN_STD_V10",
    "ENTROPY_COEF_V10",
    "NOISE_STD_V10",
    "G1AmpWalkV10PPORunnerCfg",
]
