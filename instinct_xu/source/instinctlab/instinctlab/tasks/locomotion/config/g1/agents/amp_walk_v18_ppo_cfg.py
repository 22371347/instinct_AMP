"""V18 的算法配置：**逐字沿用 V17/V16**（λ 0.75 / c 1.0 / σ / 熵 / 镜像损失 / 固定 LR）。

唯一变化在环境侧（`amp_walk_v18_env_cfg.py`：步态目标带 + 抖动权重 + 方案 A 数据集/指令），
算法超参一个没动 —— 保持"起点的可比性"。

⚠️ `ckpt_manipulator = None`：V18 默认**从零**训练（与 V17 相同）。将来若要热启动：
- 自 V11 / V16 / V17（都是 10 帧 670 维）⇒ **可直载**（还白捡一个训好的判别器）；
- 自 V12/V14/V15（2 帧 134 维）⇒ 必须改成 `drop_amp_discriminator`。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v11_ppo_cfg import CLIP_MAX_STD_V11, CLIP_MIN_STD_V11, NOISE_STD_V11
from .amp_walk_v14_ppo_cfg import STYLE_LERP_V14
from .amp_walk_v15_ppo_cfg import DISC_REWARD_COEF_V15
from .amp_walk_v17_ppo_cfg import G1AmpWalkV17PPORunnerCfg

#: 与 V17/V16/V15 相同（本轮变量全在 env 侧）
DISC_REWARD_COEF_V18 = DISC_REWARD_COEF_V15      # == 1.0
STYLE_LERP_V18 = STYLE_LERP_V14                  # == 0.75


@configclass
class G1AmpWalkV18PPORunnerCfg(G1AmpWalkV17PPORunnerCfg):
    """v18：算法侧与 V17 逐字相同。"""

    def __post_init__(self):
        super().__post_init__()

        self.algorithm.style_lerp = STYLE_LERP_V18
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V18
        self.algorithm.clip_max_std = CLIP_MAX_STD_V11
        self.algorithm.clip_min_std = CLIP_MIN_STD_V11
        self.policy.init_noise_std = NOISE_STD_V11      # 从零时生效（0.35）
        self.ckpt_manipulator = None                    # 从零；热启动自 V11/V16/V17 也直载

        self.run_name = (
            f"v18_scratch_amp10frame_flat2.0_airband0.4-1.0_ar0.10_5clip_omnidir"
            f"_style_c{DISC_REWARD_COEF_V18}_lerp{STYLE_LERP_V18}"
            f"_ns{NOISE_STD_V11}_maxstd{CLIP_MAX_STD_V11}_ent0.005_fixedlr"
        )


__all__ = [
    "DISC_REWARD_COEF_V18",
    "STYLE_LERP_V18",
    "G1AmpWalkV18PPORunnerCfg",
]
