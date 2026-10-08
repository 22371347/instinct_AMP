"""V13 的算法配置：**与 V12 逐字相同**（V13 的变量全在环境侧：指令四档 + 5-clip 数据集）。

| 字段 | V12 | **V13** |
|---|---|---|
| `style_lerp`（λ） | 0.5 | 0.5 |
| `discriminator_reward_coef`（c） | 0.5 | 0.5 |
| `clip_max_std` | 0.6 | 0.6 |
| `init_noise_std` | 0.35 | 0.35（⚠️ 热启动时无效） |
| `clip_min_std` | 0.02 | 0.02 |
| `entropy_coef` | 0.005 | 0.005 |
| `schedule` / `use_mirror_loss` | fixed / True | fixed / True |

判别器输入仍是 **2 帧（134 维）**（继承 V12 的 obs 覆盖）。
⚠️ V13 继承 **V12** 的 runner cfg ⇒ 也带着 `ckpt_manipulator`（丢掉 ckpt 里的旧判别器）：
判别器**是存在 checkpoint 里的**（`wasabi.py` override 了 `state_dict`/`load_state_dict`），
而 `Wasabi` 对键缺失是容忍的（只打 Warning）⇒ "actor/critic 热启动 + 判别器从零重学"。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v11_ppo_cfg import (
    CLIP_MAX_STD_V11,
    CLIP_MIN_STD_V11,
    DISC_REWARD_COEF_V11,
    NOISE_STD_V11,
    STYLE_LERP_V11,
)
from .amp_walk_v12_ppo_cfg import G1AmpWalkV12PPORunnerCfg


@configclass
class G1AmpWalkV13PPORunnerCfg(G1AmpWalkV12PPORunnerCfg):
    """v13：算法侧与 v11/v12 完全一致（改动全在环境侧）。"""

    def __post_init__(self):
        super().__post_init__()

        self.algorithm.style_lerp = STYLE_LERP_V11
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V11
        self.algorithm.clip_max_std = CLIP_MAX_STD_V11
        self.algorithm.clip_min_std = CLIP_MIN_STD_V11
        self.policy.init_noise_std = NOISE_STD_V11

        self.run_name = (
            f"v13_cmd4_clip5_omnidir_ns{NOISE_STD_V11}_maxstd{CLIP_MAX_STD_V11}"
            f"_ent0.005_fixedlr_lerp{STYLE_LERP_V11}_ampc{DISC_REWARD_COEF_V11}_amp2frame"
        )


__all__ = ["G1AmpWalkV13PPORunnerCfg"]
