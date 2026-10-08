"""V20 的算法配置：**AMP 口径逐字沿用 V19b/V19c/V19d**，唯一变化在**判别器输入的表征**
（由 env 侧 obs 群决定：刚体空间 195 维 × 2 帧 = 390），因此：

    amp_quad_k = 0.125  ·  gradient_penalty_coef = 1.0  ·  weight_decay = 1e-5
    logit_weight_decay = 0.0  ·  amp_reward_eps = 0.05
    log_quad + wasserstein  ·  λ = 0.75  ·  c = 1.0

起点：**从零**（`ckpt_manipulator = None`，不加载任何 checkpoint）。

## 为什么必须从零

输入维度 670 → **390**，判别器首层 `Linear(390, 1024)` 与旧权重 `Linear(670, 1024)`
形状不兼容；actor/critic 虽然维度没变（626 / flat critic），但"判别器从零 + 策略热启动"
会让风格项在前期给出与 policy 分布不匹配的信号（V19b 的实测：从零才能把小跳模态
整体换掉）。用户 2026-10-02 明确要求："**先照抄他们的输入，用 2 帧 195 维从零开始训练**"。

## 与"他们"（AMP_mjlab）的对照（本轮**只**照抄输入，不抄奖励/优化器）

| 项 | AMP_mjlab | V20 |
|---|---|---|
| 判别器输入 | 刚体空间 13×15 × 2 帧 = **390** | **同** |
| 判别器损失 | MSE→±1（归一化空间）+ GP `λ=10`（**只对 expert**、原始空间） | MSE→±1（原始空间） + GP `coef=1.0`（对 `cat[policy, expert]`） |
| 判别器优化器 | Adam lr=1e-3（与策略**共用**）、trunk wd=1e-3 / head wd=0.1 | 独立 AdamW lr=1e-4、wd=1e-5 |
| 归一化器 | 启用（`amp_normalizer`） | **无** |
| 风格奖励 | `c·clamp(1−(d−1)²/4, 0)`（= `quad k=0.25`） | `quad k=0.125`（V19b 起） |
| 帧数 | 2 | **同** |

⇒ 本轮是**单变量**（表征），不是"整套搬 mjlab"。若 V20 相对 V19d 有明显收益，
下一步才是逐项对齐上表里剩下的差异（归一化器 / GP 口径 / `k`）。
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
class G1AmpWalkV20PPORunnerCfg(G1AmpWalkV19DPPORunnerCfg):
    """v20：判别器输入换成刚体空间 195×2 帧；AMP 参数与 V19d 相同；**从零**训练。"""

    def __post_init__(self):
        super().__post_init__()

        # 显式重申（便于日后核对；值与 V19b/V19c/V19d 相同）
        self.algorithm.amp_quad_k = QUAD_K_V19B                              # 0.125
        self.algorithm.discriminator_gradient_penalty_coef = GP_COEF_V19A     # 1.0
        self.algorithm.discriminator_weight_decay_coef = WD_COEF_V19A         # 1e-5
        self.algorithm.discriminator_logit_weight_decay_coef = LOGIT_WD_COEF_V19A  # 0.0
        self.algorithm.amp_reward_eps = AMP_REWARD_EPS_V19A                   # 0.05
        self.algorithm.style_lerp = STYLE_LERP_V18                            # 0.75
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V18        # 1.0

        # 起点：**从零**（新的判别器输入维度 390，旧权重不可用）
        self.ckpt_manipulator = None

        self.run_name = f"v20_scratch_amp13body2frame_quadk{QUAD_K_V19B:g}"


__all__ = ["G1AmpWalkV20PPORunnerCfg"]
