"""V9h 的算法配置：**风格侧回退到 v9e/v9f 的稳定值**（v9g 的 ×4 实测无效且伴随崩溃振荡）。

| 字段 | V9g | **V9h** | 依据 |
|---|---|---|---|
| `style_lerp` | 0.5 | **0.75** | ×4 后 `amp_d` 从 −0.433 只变到 −0.407（几乎不动）；同期出现 `rew 37→0.31` 的崩溃振荡，且 `amp_style_std` 在崩溃时翻倍（0.10→0.19/0.24） ⇒ 风格权重不是杠杆，先回退 |
| `discriminator_reward_coef` | 0.5 | **0.25** | 同上（`(1−λ)c` 回到 0.0625 = v9e/v9f 的稳定值） |
| `schedule` | fixed | fixed | 继承 v9f |
| `entropy_coef` | 0.0025 | 0.0025 | 继承 v9f（0.008 会把 σ 顶到天花板） |
| `init_noise_std` / `clip_max_std` | 0.25 / 0.4 | **0.3 / 0.4** | 继承 v9f。⚠️ 口径见下 |
| `use_mirror_loss` | True | True | 继承 v9d（关掉会退化成单脚跳） |

⚠️ **σ 的口径**（学长 `init_noise_std≈0.5` 的参考值要按这个换算再比）：
`instinct_rl` 的 `std` 是**归一化动作空间**的标准差，`a_joint = default + action_scale · a`
⇒ **关节空间噪声 σ_joint = init_noise_std × action_scale**。
我们 `action_scale = 1.0` ⇒ `0.3` 就是 **±0.3 rad ≈ ±17°**；
若学长是 `init_noise_std=0.5 / action_scale=0.25`（AMP_mjlab 风格），则 σ_joint ≈ **0.125 rad ≈ ±7°**，
**比我们现在的探索噪声还小一半**。⇒ 在拿到学长的 `action_scale`（以及 `stiffness/damping/decimation`）之前，
不要直接照搬 0.5 这个数。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v9f_ppo_cfg import G1AmpWalkV9fPPORunnerCfg

#: 风格/任务 凸组合：0.5 → 0.75（回到 v9e/v9f 的稳定值）
STYLE_LERP_V9H = 0.75
#: 风格奖励缩放：0.5 → 0.25 ⇒ 有效风格权重 (1−λ)·c = 0.0625
DISC_REWARD_COEF_V9H = 0.25
#: 探索噪声：沿用 v9f（关节空间 ±0.3 rad，见模块 docstring 的口径说明）
NOISE_STD_V9H = 0.3


@configclass
class G1AmpWalkV9hPPORunnerCfg(G1AmpWalkV9fPPORunnerCfg):
    """v9h：v9f 的算法设置 + 风格侧回退（本版唯一算法改动）。"""

    def __post_init__(self):
        super().__post_init__()

        self.algorithm.style_lerp = STYLE_LERP_V9H
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V9H
        self.policy.init_noise_std = NOISE_STD_V9H

        self.run_name = (
            f"v9h_ns{NOISE_STD_V9H}_maxstd0.4_fixedlr_ent0.0025"
            f"_lerp{STYLE_LERP_V9H}_ampc{DISC_REWARD_COEF_V9H}_rew20_antihop"
        )


__all__ = [
    "DISC_REWARD_COEF_V9H",
    "NOISE_STD_V9H",
    "STYLE_LERP_V9H",
    "G1AmpWalkV9hPPORunnerCfg",
]
