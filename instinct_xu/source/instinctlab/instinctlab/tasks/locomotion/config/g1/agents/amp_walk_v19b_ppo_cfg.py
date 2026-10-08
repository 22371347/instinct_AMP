"""V19b 的算法配置：**V19a 的判别器口径逐字保留**，只放宽 `quad` 的下端 clamp。

| 项 | V19a | **V19b** | 依据（V19a@3140 实测） |
|---|---|---|---|
| `amp_quad_k`（`D = clamp(1 − k(d−1)², 0, 1−eps)` 的曲率） | 0.25 | **0.125** | `d` 展宽到 −0.81 后 **5.0%** 的样本越过 `d ≤ −1.0` ⇒ `style ≡ 0` **且 `dr/dd = 0`**（纯死区）；k=0.125 把下端推到 `d ≤ −1.83` |
| `discriminator_gradient_penalty_coef` | 1.0 | 1.0（不动） | V19a 已把它从 5.0 降下来：实测 `mean‖∂d/∂x‖²` **0.042 → 0.140**（‖∇d‖ ×1.8）✓ |
| `discriminator_weight_decay_coef` | 1e-5 | 1e-5 | — |
| `discriminator_logit_weight_decay_coef` | 0.0 | 0.0 | — |
| `amp_reward_eps` | 0.05 | 0.05 | — |
| λ / c / 形式 | 0.75 / 1.0 / `log_quad`+`wasserstein` | 同左 | — |

**为什么只动 k（而不是加权重 `c`）**
---------------------------------
把 `style = −log(1−D) = −log k − 2·ln|d−1|` 拆开看：

* `d` 的**局部斜率** `dr/dd = 2/(1−d)` **与 k 无关**（在 `d=−0.81` 处 = **1.10**）；
* `k` 只改**奖励水平**（闭式里的 `−log k` 项）：`d = −0.81` 时
  `style = −log k − 2·ln|d−1|` ⇒ k=0.25 → `1.386 − 1.187 = 0.199`、
  k=0.125 → `2.079 − 1.187 = 0.892`（**×4.5**）⇒ 风格项/步 `0.25·style` 由
  **0.049 → 0.223**（占 r_task 由 3.4% → ~15%，回到 V18 的 11% 量级），**且**消掉那 5% 死区。
* ⚠️ 诚实说明：**k 并不增加策略侧的梯度**（斜率不变），它解决的是"奖励水平太低 +
  死区"两件事；梯度只能靠判别器的分辨力（已由 GP 放松拿到 ×1.8）。

起点
----
**从零训练**（`ckpt_manipulator = None`、`--resume` 不带）。理由见
`amp_walk_v19b_env_cfg.py`：V18 从零也落在小跳模态，所以"起点"必须与"定价形状"一起换。
从零 ⇒ `init_noise_std = 0.35`（V11 的值，继承）、σ 上限/下限不变。

监测（与 V19a 同样的口径 + 新增两项）
------------------------------------
* ⭐ `Step/air_time_single_stance`：应从 0.011 s 明显上爬（这是本轮的**主指标**）；
* ⭐ `Step/air_time_hop_frac`（**V19b 新增探针**）：`d < 0.1 s` 的占比，应显著下降；
* `Step/air_time_in_band_frac`：应 > 0.01（V19a 是 0.0005~0.0015）；
* `Episode_Reward/feet_air_time`：上限 2.4/步，V19a 只拿 0.038 ⇒ 看能爬到多少；
* `Step/amp_style_zero_frac`：应从 0.050 **掉回 ≈0**（k 放宽的直接验证）；
* `Step/amp_style_reward` / `Episode_Reward` 里风格项的占比：应回到 ~10%+；
* 守住 `track_lin_vel_xy_exp ≥ 1.0`、`tilt ≤ 12°`、`Mean episode length`。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v18_ppo_cfg import (
    DISC_REWARD_COEF_V18,
    STYLE_LERP_V18,
)
from .amp_walk_v19a_ppo_cfg import (
    AMP_REWARD_EPS_V19A,
    GP_COEF_V19A,
    LOGIT_WD_COEF_V19A,
    WD_COEF_V19A,
    G1AmpWalkV19APPORunnerCfg,
)

#: ⭐ 本轮唯一算法侧改动：quad 曲率 0.25 → 0.125（下端 clamp 由 d≤−1.0 推到 d≤−1.83）
QUAD_K_V19B = 0.125


@configclass
class G1AmpWalkV19BPPORunnerCfg(G1AmpWalkV19APPORunnerCfg):
    """v19b：V19a 的判别器口径 + quad 下端放宽；**从零训练**。"""

    def __post_init__(self):
        super().__post_init__()

        # ⭐ 只改这一项（其余在 V19a 里已设好：GP 1.0 / wd 1e-5 / logit_wd 0.0 / eps 0.05）
        self.algorithm.amp_quad_k = QUAD_K_V19B

        # 显式重申（便于以后核对，值与 V19a 相同）
        self.algorithm.discriminator_gradient_penalty_coef = GP_COEF_V19A
        self.algorithm.discriminator_weight_decay_coef = WD_COEF_V19A
        self.algorithm.discriminator_logit_weight_decay_coef = LOGIT_WD_COEF_V19A
        self.algorithm.amp_reward_eps = AMP_REWARD_EPS_V19A
        self.algorithm.style_lerp = STYLE_LERP_V18
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V18

        # ⭐ **从零**：不装载任何 checkpoint，判别器自然也是新建的
        self.ckpt_manipulator = None

        self.run_name = (
            f"v19b_scratch_quadk{QUAD_K_V19B:g}_gp{GP_COEF_V19A}_wd{WD_COEF_V19A:g}"
            f"_airtau0.08_style_c{DISC_REWARD_COEF_V18}_lerp{STYLE_LERP_V18}"
        )


__all__ = [
    "QUAD_K_V19B",
    "G1AmpWalkV19BPPORunnerCfg",
]
