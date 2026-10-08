"""V9c 的 PPO/算法配置：见 `amp_walk_v9c_env_cfg.py` 顶部的对照表。

一句话：**降 σ（entropy 0.008→0.0025、上限 0.4→0.3）+ 抬 AMP 权重（c 0.05→0.25）+
松手臂约束（−0.1→0）**，在 V9b（已关镜像损失）之上。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v9b_ppo_cfg import G1AmpWalkV9bPPORunnerCfg

#: 探索噪声初值（比 v9 的 0.25 再低一档，配合上限 0.3）
NOISE_STD_V9C = 0.2
#: σ 上限：0.4 → 0.3（±17°）。仍允许自适应上升，但把天花板下移。
CLIP_MAX_STD_V9C = 0.3
#: ⭐ 熵系数：0.008 → 0.0025。这是把 σ 顶到天花板的真正推手。
ENTROPY_COEF_V9C = 0.0025
#: ⭐ 风格奖励缩放：0.05 → 0.25 ⇒ AMP 占训练回报 ~11% → ~55%。
DISC_REWARD_COEF_V9C = 0.25


@configclass
class G1AmpWalkV9cPPORunnerCfg(G1AmpWalkV9bPPORunnerCfg):
    """v9c：v9b（无镜像损失）+ 降 σ + 抬 AMP 权重。"""

    def __post_init__(self):
        super().__post_init__()

        self.policy.init_noise_std = NOISE_STD_V9C
        self.algorithm.clip_max_std = CLIP_MAX_STD_V9C
        # ⚠️ `entropy_coef` / `discriminator_reward_coef` 都是算法 cfg 的正式字段，
        #    直接改实例即可（与 v7 改 `discriminator_reward_coef` 的做法一致）。
        self.algorithm.entropy_coef = ENTROPY_COEF_V9C
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V9C

        self.run_name = (
            f"v9c_ns{NOISE_STD_V9C}_maxstd{CLIP_MAX_STD_V9C}"
            f"_ent{ENTROPY_COEF_V9C}_ampc{DISC_REWARD_COEF_V9C}_noarms_nomirror"
        )


__all__ = [
    "CLIP_MAX_STD_V9C",
    "DISC_REWARD_COEF_V9C",
    "ENTROPY_COEF_V9C",
    "NOISE_STD_V9C",
    "G1AmpWalkV9cPPORunnerCfg",
]
