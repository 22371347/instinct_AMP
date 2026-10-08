"""V23 的算法配置：**V22 的白化（AMP 风格）+ V23 的奖励结构**，AMP 其余口径逐字沿用 V19d。

| 项 | V19d/V21 | **V22（白化）** | **V23（本版）** |
|---|---|---|---|
| 判别器输入 | 原始 195×8 | + `EmpiricalNormalization(eps=0.01)` | 同 V22 |
| 白化统计量 | — | **只由 expert 侧累积**（`amp_normalizer_expert_only=True`） | 同 V22 |
| GP 作用空间 | `"raw"` | **`"normalized"`**（原始空间会被 `1/eps` 放大 311×、饱和后 1.2e6×） | 同 V22 |
| `GP coef` | 1.0 | 1.0（沿用：归一化空间的 GP 与旧量级 1.04×/1.48×） | 1.0 |
| 奖励结构 | 21 项 | 同 V21 | **改 2 项**（进度判据 / 逐关节死区），见 `amp_walk_v23_env_cfg.py` |

为什么白化是必需项：三代 run 的判别器**全线饱和**（`amp_d ≈ −0.99`、`gap = d_ref − d ≈ 1.97`
上界 2、`Loss/discriminator_loss` 0.0027~0.0050、`Step/amp_style_std` 只 0.09~0.14）
⇒ 风格项几乎不提供排序信号（V20 从零 + 2 帧在第 2000 轮就饱和到 0.003）。
白化的依据：离线 AUC 0.850 → **0.927**（h=8），`style_std` ×2.7。

起点 **从零**（判别器多了 normalizer buffer，且白化改变了输入分布的定义 ⇒ 不复用 V21/V20 权重）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v22_ppo_cfg import (
    AMP_V22_NORMALIZER_EPS,
    AMP_V22_NORMALIZER_UNTIL,
    G1AmpWalkV22PPORunnerCfg,
)
from .amp_walk_v19b_ppo_cfg import QUAD_K_V19B
from .amp_walk_v19a_ppo_cfg import GP_COEF_V19A


@configclass
class G1AmpWalkV23PPORunnerCfg(G1AmpWalkV22PPORunnerCfg):
    """v23：白化（V22）+ 奖励结构（进度判据 / 逐关节死区）；从零。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- ③ 判别器输入白化（与 V22 逐字相同，这里显式重申便于核对）----
        self.algorithm.discriminator_kwargs = {
            **self.algorithm.discriminator_kwargs,
            "normalizer_class_name": "EmpiricalNormalization",
            "normalizer_kwargs": {"eps": AMP_V22_NORMALIZER_EPS, "until": AMP_V22_NORMALIZER_UNTIL},
        }
        self.algorithm.amp_normalizer_expert_only = True
        self.algorithm.amp_grad_pen_space = "normalized"
        self.algorithm.discriminator_gradient_penalty_coef = GP_COEF_V19A  # 1.0

        # ---- 风格奖励的其余口径（与 V19b/V19d/V21 一致）----
        self.algorithm.amp_quad_k = QUAD_K_V19B  # 0.125

        # 起点：从零
        self.ckpt_manipulator = None
        self.run_name = "v23_scratch_amp13body8frame_white_progress_posdead"


__all__ = ["G1AmpWalkV23PPORunnerCfg"]
