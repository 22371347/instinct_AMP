"""V21 的算法配置：**AMP 口径逐字沿用 V19b/V19c/V19d**。

唯一变化在**判别器输入**（由 env 侧 obs 群决定）：刚体空间 **195 维 × 8 帧 = 1560**（V20 是 195×2=390）。

    amp_quad_k = 0.125  ·  gradient_penalty_coef = 1.0  ·  weight_decay = 1e-5
    logit_weight_decay = 0.0  ·  amp_reward_eps = 0.05
    log_quad + wasserstein  ·  λ = 0.75  ·  c = 1.0

起点：**从零**（`ckpt_manipulator = None`）—— 判别器首层形状变为 `Linear(1560, 1024)`，
与 V20（390）及 V19x（670）都不兼容；且从零才能干净地比较"表征/帧数"。

为什么 h=8（离线研究 v2 的数据，见 `amp_walk_v21_env_cfg.py` 与
`RESULTS_amp_frames_body_vs_joint.md`）：

| h | body195 AUC | `style_std` | 输入维 |
|---|---|---|---|
| 2（V20） | 0.625 | 0.0070 | 390 |
| **8（V21）** | **0.890** | **0.0791** | **1560** |
| 10 | 0.910 | 0.0988 | 1950 |

h=8 的 0.890 已高于 legacy67@h=10 的 0.838（= V19b 走出正常步态时的判别水平），
且比 h=10 省 20% 算力。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v18_ppo_cfg import DISC_REWARD_COEF_V18, STYLE_LERP_V18
from .amp_walk_v19a_ppo_cfg import (
    AMP_REWARD_EPS_V19A,
    GP_COEF_V19A,
    LOGIT_WD_COEF_V19A,
    WD_COEF_V19A,
)
from .amp_walk_v19b_ppo_cfg import QUAD_K_V19B
from .amp_walk_v19d_ppo_cfg import G1AmpWalkV19DPPORunnerCfg


@configclass
class G1AmpWalkV21PPORunnerCfg(G1AmpWalkV19DPPORunnerCfg):
    """v21：判别器输入换成刚体空间 195×8 帧；AMP 参数与 V19d 相同；**从零**训练。"""

    def __post_init__(self):
        super().__post_init__()

        # 显式重申（便于日后核对；值与 V19b/V19c/V19d/V20 相同）
        self.algorithm.amp_quad_k = QUAD_K_V19B                              # 0.125
        self.algorithm.discriminator_gradient_penalty_coef = GP_COEF_V19A     # 1.0
        self.algorithm.discriminator_weight_decay_coef = WD_COEF_V19A         # 1e-5
        self.algorithm.discriminator_logit_weight_decay_coef = LOGIT_WD_COEF_V19A  # 0.0
        self.algorithm.amp_reward_eps = AMP_REWARD_EPS_V19A                   # 0.05
        self.algorithm.style_lerp = STYLE_LERP_V18                            # 0.75
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V18        # 1.0

        # 起点：**从零**（判别器输入维度 1560，旧权重不可用）
        self.ckpt_manipulator = None

        self.run_name = f"v21_scratch_amp13body8frame_quadk{QUAD_K_V19B:g}"


__all__ = ["G1AmpWalkV21PPORunnerCfg"]
