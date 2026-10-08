"""V15：**风格分量放大实验**（用户 2026-09-30 指定）。

目的
----
V14 里 `Step/amp_style_reward` 的**均值在 5400 轮内只动 1.7%**（`Step/amp_d` 从 450 轮起
钉在 −0.47 一直到 5400 轮）⇒ 风格项"饱和/学不动"。但它的**绝对量并不小**：
(1−λ)c·r_amp = 0.0625×0.60 ≈ 0.0375/步（比任务净额 0.0314/步还大）。
⇒ 真正的问题不是"占比小"，而是**平坦**（梯度≈0）。

本实验把**风格分量单独放大 4 倍**（任务分量不动），看 `amp_style_reward`/`amp_d` 的
**变化率是否恢复**（斜率显著非零 ⇒ 风格梯度真的回来了）。

| 字段 | V14 | **V15** | 说明 |
|---|---|---|---|
| `style_lerp`（λ） | 0.75 | **0.75（不动）** | ⇒ 任务入账保持 ×0.75 |
| `discriminator_reward_coef`（c） | 0.25 | **1.0** | ⭐ 唯一变量：风格入账 (1−λ)c：0.0625 → **0.25（×4）** |
| `clip_max_std` / `clip_min_std` | 0.6 / 0.02 | 同 | 不动 |
| `init_noise_std` | 0.35 | 同 | 不动 |
| `entropy_coef` / `schedule` / `use_mirror_loss` | 0.005 / fixed / True | 同 | 不动 |
| `ckpt_manipulator` | drop_amp_discriminator | 同 | 判别器仍是 2 帧 134 维 |

起点：**从 V11 的 `model_9000.pt` 热启动**（V11 是"会走路"的策略）。
⚠️ 不从 V14 热启动：V14 从零已经收敛到**单脚跳**，从一个 hop 策略起步会继承 hop，
就答不了"放大风格能不能把步态/姿态拉向参考"。

⚠️ 与 V11 相比本轮同时多了"2 帧判别器 + 限位软边距"两项（它们是 V14 起就带着的修复），
所以严格说这不是与 V11 的单变量对比；它的目的是**回答风格分量×4 后 AMP 贡献的动向**。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v11_ppo_cfg import CLIP_MAX_STD_V11, CLIP_MIN_STD_V11, NOISE_STD_V11
from .amp_walk_v14_ppo_cfg import STYLE_LERP_V14, G1AmpWalkV14PPORunnerCfg

#: ⭐ 唯一变量：风格缩放 c（0.25 → 1.0）⇒ 风格入账 (1−λ)c：0.0625 → 0.25（×4）
DISC_REWARD_COEF_V15 = 1.0


@configclass
class G1AmpWalkV15PPORunnerCfg(G1AmpWalkV14PPORunnerCfg):
    """v15：λ 保持 V14 的 0.75，只把风格分量 c 放大 4 倍。"""

    def __post_init__(self):
        super().__post_init__()

        self.algorithm.style_lerp = STYLE_LERP_V14          # 0.75，不动
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V15   # ⭐ 0.25 → 1.0
        self.algorithm.clip_max_std = CLIP_MAX_STD_V11
        self.algorithm.clip_min_std = CLIP_MIN_STD_V11
        self.policy.init_noise_std = NOISE_STD_V11
        self.ckpt_manipulator = "instinctlab.utils.ckpt_manipulators:drop_amp_discriminator"

        self.run_name = (
            f"v15_ampstyle_x4_c{DISC_REWARD_COEF_V15}_lerp{STYLE_LERP_V14}"
            f"_standpose0.3_limitmargin_fromV11"
            f"_ns{NOISE_STD_V11}_maxstd{CLIP_MAX_STD_V11}_ent0.005_fixedlr"
        )


__all__ = ["DISC_REWARD_COEF_V15", "G1AmpWalkV15PPORunnerCfg"]
