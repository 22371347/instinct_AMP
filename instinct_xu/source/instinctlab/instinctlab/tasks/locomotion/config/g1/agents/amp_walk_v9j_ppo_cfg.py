"""V9j 的算法配置：**与 V9h 逐字相同**（本版唯一变量是奖励集里的新项）。

| 字段 | V9h | **V9j** |
|---|---|---|
| `init_noise_std` | 0.3 | 0.3 |
| `clip_max_std` | 0.4 | 0.4 |
| `entropy_coef` | 0.0025 | 0.0025 |
| `schedule` | fixed | fixed |
| `style_lerp` / `discriminator_reward_coef` | 0.75 / 0.25 | 0.75 / 0.25 |
| `use_mirror_loss` | True | True |

⇒ **V9h vs V9j 只差 `feet_swing_time_penalty`（−0.5）这一项**，
这样"单脚侧身跳"是否被堵住、交替步是否出现，归因就是干净的。

（探索量级那条线留在 V9i：σ 0.5 / 上限 0.8 / entropy 0.005。两条线互补，先做约束。）
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v9h_ppo_cfg import G1AmpWalkV9hPPORunnerCfg


@configclass
class G1AmpWalkV9jPPORunnerCfg(G1AmpWalkV9hPPORunnerCfg):
    """v9j：算法设置与 v9h 完全相同，只换 run_name。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "v9j_ns0.3_maxstd0.4_fixedlr_ent0.0025_lerp0.75_ampc0.25_rew21_swingpen"


__all__ = ["G1AmpWalkV9jPPORunnerCfg"]
