"""V17 的算法配置：**逐字沿用 V16**（λ 0.75 / c 1.0 / σ / 熵 / 镜像损失 / 固定 LR）。

唯一变化在环境侧（`amp_walk_v17_env_cfg.py`：`flat_orientation_l2` −1 → −2），
算法超参一个没动。

## ⚠️ `ckpt_manipulator = None`（与 V16 不同，这是有意的）

V17 默认**从零**训练 ⇒ 用不到 manipulator。而**将来若要热启动**，需要注意：

| 来源 ckpt | 判别器输入维度 | 能否直载 |
|---|---|---|
| V11 / V16（10 帧） | 670 | ✅ 与本版一致，**直接载入即可**（还白捡一个训好的判别器） |
| V15 / V14 / V12（2 帧） | 134 | ❌ 必须丢判别器（把下面那行改成 `drop_amp_discriminator`） |

⇒ 设 `None` 是为了**不误丢** V16 那颗已经训好的 10 帧判别器（V16 的 discriminator 是
本轮消融里"分辨力 ×2.9"的那个，很值钱）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v11_ppo_cfg import CLIP_MAX_STD_V11, CLIP_MIN_STD_V11, NOISE_STD_V11
from .amp_walk_v14_ppo_cfg import STYLE_LERP_V14
from .amp_walk_v15_ppo_cfg import DISC_REWARD_COEF_V15
from .amp_walk_v16_ppo_cfg import G1AmpWalkV16PPORunnerCfg

#: 与 V16/V15 相同（唯一变量在 env 侧的 `flat_orientation_l2`）
DISC_REWARD_COEF_V17 = DISC_REWARD_COEF_V15      # == 1.0
STYLE_LERP_V17 = STYLE_LERP_V14                  # == 0.75


@configclass
class G1AmpWalkV17PPORunnerCfg(G1AmpWalkV16PPORunnerCfg):
    """v17：算法侧与 V16 逐字相同；判别器**不丢弃**（从零训练用不到，热启动时也不该丢）。"""

    def __post_init__(self):
        super().__post_init__()

        self.algorithm.style_lerp = STYLE_LERP_V17
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V17
        self.algorithm.clip_max_std = CLIP_MAX_STD_V11
        self.algorithm.clip_min_std = CLIP_MIN_STD_V11
        self.policy.init_noise_std = NOISE_STD_V11      # 从零时生效（0.35）
        # 从零训练 ⇒ 不丢判别器；热启动自 V11/V16（670 维）也照样直载 ✅
        self.ckpt_manipulator = None

        self.run_name = (
            f"v17_scratch_amp10frame_flat2.0_airtime4.0_style_c{DISC_REWARD_COEF_V17}_lerp{STYLE_LERP_V17}"
            f"_standpose0.3_limitmargin"
            f"_ns{NOISE_STD_V11}_maxstd{CLIP_MAX_STD_V11}_ent0.005_fixedlr"
        )


__all__ = [
    "DISC_REWARD_COEF_V17",
    "STYLE_LERP_V17",
    "G1AmpWalkV17PPORunnerCfg",
]
