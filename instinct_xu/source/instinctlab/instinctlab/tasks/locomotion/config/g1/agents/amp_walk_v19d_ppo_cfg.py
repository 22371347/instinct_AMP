"""V19d 的算法配置：**与 V19b/V19c 逐字相同**（AMP 侧一个参数都不动）。

理由同 V19c：本轮全部变量都在 env 侧（站立档负罚 + 两个分解探针都在 reward 函数里），
AMP 口径必须保持原样才能归因：

    amp_quad_k = 0.125  ·  gradient_penalty_coef = 1.0  ·  weight_decay = 1e-5
    logit_weight_decay = 0.0  ·  amp_reward_eps = 0.05
    log_quad + wasserstein  ·  λ = 0.75  ·  c = 1.0

起点：**热启动 V19c 的最新 checkpoint，并保留判别器**（输入仍是 670 维、`k` 仍是 0.125）。

## 顺带回答"`amp_total_loss ≈ 0.01` 正常吗"

V19c@9700 的实测分解（`amp_total_loss = Σ coef·loss`）：

| 分项 | 系数 | 实测值 | 贡献 |
|---|---|---|---|
| `Loss/discriminator_loss`（MSE → ±1） | 1.0 | 0.00457 | 0.0046 |
| `Loss/discriminator_gradient_penalty` | 1.0 | 0.00704 | 0.0070 |
| `Loss/discriminator_weight_decay`（`Σw²`） | 1e-5 | ~? | ≈0 |
| `Loss/logit_weight_decay` | 1.0 | 0（已关闭） | 0 |
| **合计** | | | **≈ 0.012** ≈ TB 的 `amp_total_loss` ✅ |

⇒ **0.01 量级完全正常，而且正是"判别器已学会"的标志**：MSE 0.0046 ⇒ 两类输出与目标
只差 ≈0.07（实测 `d = −0.989`、`d_ref = +0.987`，目标 ±1）。对照 V18 的 0.154（那时
`d` 只能到 ±0.62）。
⚠️ 但要同时看两个配套量：`Step/amp_style_std`（排序信号，V19c = 0.131，未塌）与
`Step/amp_style_zero_frac`（= 0，`k=0.125` 把下端 clamp 推到 d ≤ −1.83 的结果）。
⚠️ 另一个隐忧：`gradient_penalty`（= `mean‖∂d/∂x‖²`）只有 **0.0070** ⇒ `‖∇d‖ ≈ 0.084`
⇒ 判别器又变得**很平**。若政策侧还有改善空间却推不动风格项，下一步就该动
`discriminator_gradient_torlerance`（0 → 1.0）或把 GP 关掉。
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
from .amp_walk_v19c_ppo_cfg import G1AmpWalkV19CPPORunnerCfg


@configclass
class G1AmpWalkV19DPPORunnerCfg(G1AmpWalkV19CPPORunnerCfg):
    """v19d：AMP 口径与 V19c 相同；只换 env 侧奖励（站立档负罚），热启动 V19c。"""

    def __post_init__(self):
        super().__post_init__()

        # 显式重申（便于日后核对；值与 V19b/V19c 相同）
        self.algorithm.amp_quad_k = QUAD_K_V19B                              # 0.125
        self.algorithm.discriminator_gradient_penalty_coef = GP_COEF_V19A     # 1.0
        self.algorithm.discriminator_weight_decay_coef = WD_COEF_V19A         # 1e-5
        self.algorithm.discriminator_logit_weight_decay_coef = LOGIT_WD_COEF_V19A  # 0.0
        self.algorithm.amp_reward_eps = AMP_REWARD_EPS_V19A                   # 0.05
        self.algorithm.style_lerp = STYLE_LERP_V18                            # 0.75
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V18        # 1.0

        # 起点：热启动 V19c（**保留判别器**）
        self.ckpt_manipulator = None

        self.run_name = f"v19d_warmV19c_standpen0.5_ar0.06_standpose0.5_quadk{QUAD_K_V19B:g}"


__all__ = ["G1AmpWalkV19DPPORunnerCfg"]
