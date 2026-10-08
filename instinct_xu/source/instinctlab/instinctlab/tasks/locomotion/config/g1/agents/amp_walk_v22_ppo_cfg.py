"""V22 的算法配置：**V21 + 判别器输入白化**（"新版 AMP 风格方案"的真正落地，2026-10-03）。

V21 = 刚体空间 **195 维 × 8 帧 = 1560** + AMP 口径与 V19d **逐字相同**。
V22 = V21 + **白化**，相对 V21 仍是**单变量**（env 侧一字不改，复用 V21 的 env cfg 类）。

三个必须**成对**的改动
----------------------
1. `discriminator_kwargs` 追加 normalizer：

       "normalizer_class_name": "EmpiricalNormalization",
       "normalizer_kwargs": {"eps": 0.01, "until": None}

   对应 AMP_mjlab 判别器里的 `amp_normalizer`。

2. `amp_grad_pen_space = "normalized"` —— **不改这条就会崩**：

   | 空间 | ‖∇d‖ | GP 项 | 判别器饱和后 GP / MSE |
   |---|---|---|---|
   | `"raw"`（白化 + 旧 GP） | 1.90 | 3.62 | **1.2e6×** ⇒ 判别器损失被 GP 支配 |
   | `"normalized"` | 0.108 | 0.0116 | ≈ 0.3×（同旧配置量级） |

   原因：`∂d/∂x = (∂d/∂x̃)/(σ+eps)`，单维放大上限 = `1/eps = 100`（`eps` 是 σ 下限）。
   改到归一化空间后，GP 项与"无白化旧配置"（0.0079）**同量级**（1.04× 未训 / 3.2× 饱和）
   ⇒ `discriminator_gradient_penalty_coef = 1.0` **原样沿用**，不做第二次改动。

3. `amp_normalizer_expert_only = True`（基类默认值，这里显式重申）——

   `Discriminator.forward` 默认会顺手更新统计量，而 rollout 期**每个 env step** 都会用
   **策略数据**前向一次来算风格奖励（`wasabi.py: compute_auxiliary_reward`）；
   不关掉的话统计量会被策略分布带偏（实测 mean 漂移 **1.51**）⇒ 白化等于没做。

依据与验证
----------
* 离线：白化让"时间结构类退化" AUC 从 0.850 提到 **0.927**（h=8），`style_std` ×2.7；
  且**拐点前移**（AUC≈0.85：raw 需 h=8、白化只需 h≈5）⇒ 本代先保持 h=8，把"白化→降 h"
  留给下一代单变量。见 `RESULTS_amp_vs_mjlab_params.md` §F。
* 本地单测（无需 Isaac/GPU）：`python instinct/tools/test_amp_normalizer.py` ⇒ **27/27 通过**
  （白化零均值/单位方差、近常数维被 σ 下限压住、统计量专家独占、两个空间的 GP 量级、
  真实 `WasabiAlgoMixin.compute_discriminator_gradient` 两条分支、所有判别器调用点都显式指定
  `update_normalizer`）。

⚠️ 一个易踩的坑：`eps` **同时**是"白化力度"和"raw 空间 GP 爆炸倍数的上界"。
   若日后想把白化做彻底（`eps` 更小，让近常数维也接近单位方差），**必须**保持
   `amp_grad_pen_space="normalized"`，否则 GP 会再爆一到两个数量级。

起点：**从零**（`ckpt_manipulator = None`）。判别器新增了 normalizer 的 buffer
（`_mean/_std/count`），且输入分布的**定义**变了（白化）⇒ 不复用 V21 权重。
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
from .amp_walk_v21_ppo_cfg import G1AmpWalkV21PPORunnerCfg

#: 白化判别器的 σ 下限（`EmpiricalNormalization.eps`）：
#: * 刚体空间里有若干**近常数维**（静止刚体的姿态 / 零速分量，σ ≈ 1e-5），
#:   没有下限时会被放大 ~1e5 倍 ⇒ 等于把量化噪声/常量当特征喂给判别器；
#: * `0.01` 把它们**压住**（白化后 std ≈ 1e-3），同时常规维几乎不受影响
#:   （白化后 std 0.92~0.98）。
AMP_V22_NORMALIZER_EPS = 0.01

#: 统计量累积样本数上限（`None` = 永不冻结，与 AMP_mjlab 的 running normalizer 一致）。
#: 若日后想要"统计量收敛后冻住"（更确定性），可改成具体样本数（例如 1e7），
#: 但那时 `amp_grad_pen_space` 仍必须是 `"normalized"`。
AMP_V22_NORMALIZER_UNTIL = None


@configclass
class G1AmpWalkV22PPORunnerCfg(G1AmpWalkV21PPORunnerCfg):
    """v22：判别器输入白化（expert-only 统计 + GP 移到归一化空间）；AMP 其余参数与 V19d 相同。"""

    def __post_init__(self):
        super().__post_init__()

        # ① 判别器输入白化：在 V21 的 `discriminator_kwargs` 上**追加** normalizer
        #    （`hidden_sizes` / `nonlinearity` 严格保持 V21 的值 ⇒ 仍是单变量）
        self.algorithm.discriminator_kwargs = {
            **self.algorithm.discriminator_kwargs,
            "normalizer_class_name": "EmpiricalNormalization",
            "normalizer_kwargs": {
                "eps": AMP_V22_NORMALIZER_EPS,
                "until": AMP_V22_NORMALIZER_UNTIL,
            },
        }

        # ② 统计量只由 expert/reference 侧累积（否则被 rollout 期的策略数据带偏）
        self.algorithm.amp_normalizer_expert_only = True

        # ③ GP 必须在**归一化空间**求（白化 + raw 空间 GP ⇒ 判别器损失被支配）
        self.algorithm.amp_grad_pen_space = "normalized"

        # 显式重申 AMP 口径与 V19b/V19d/V20/V21 相同（便于日后核对）
        self.algorithm.amp_quad_k = QUAD_K_V19B                              # 0.125
        self.algorithm.discriminator_gradient_penalty_coef = GP_COEF_V19A     # 1.0
        self.algorithm.discriminator_weight_decay_coef = WD_COEF_V19A         # 1e-5
        self.algorithm.discriminator_logit_weight_decay_coef = LOGIT_WD_COEF_V19A  # 0.0
        self.algorithm.amp_reward_eps = AMP_REWARD_EPS_V19A                   # 0.05
        self.algorithm.style_lerp = STYLE_LERP_V18                            # 0.75
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V18        # 1.0

        # 起点：**从零**（判别器多了 normalizer buffer，且白化改变了输入分布的定义）
        self.ckpt_manipulator = None

        self.run_name = f"v22_scratch_whiten_amp13body8frame_eps{AMP_V22_NORMALIZER_EPS:g}"


__all__ = [
    "AMP_V22_NORMALIZER_EPS",
    "AMP_V22_NORMALIZER_UNTIL",
    "G1AmpWalkV22PPORunnerCfg",
]
