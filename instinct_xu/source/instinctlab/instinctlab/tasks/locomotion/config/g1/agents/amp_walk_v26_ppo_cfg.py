"""V26 的算法配置：**V24 逐字不动 + σ 上限退火**（唯一变量）。

| 项 | V24 | **V26** |
|---|---|---|
| 白化（`normalizer_class_name` / `amp_normalizer_expert_only` / `amp_grad_pen_space`） | V23 起 | **同** |
| 奖励结构、`action_scale=1.0`、LR、`entropy_coef`、`amp_quad_k`、GP coef | — | **全不动** |
| `clip_max_std` | 0.6 | **0.20** |
| `clip_max_std_schedule` | — | **300** |
| 起点 | 从零 | **V24 `model_8000.pt` 热启动** |

热启动为什么不需要 `ckpt_manipulator`：V26 与 V24 的 AMP 输入完全同形
（`discriminator` 第一层 1560 = 195×8，白化 buffer 也在）⇒ 判别器/优化器**都可以
直接复用**，这恰好也是我们想测的："在同一套已经退化（`style_std≈0.008`）的判别器上，
单纯把 σ 压下来，风格项会不会自己复活"。

⚠️ 不要同轮改 `entropy_coef`：它是**唯一**把 σ 往上推的力，而 σ 的自然平衡点
（≈0.57~0.60）就是它与噪声罚的平衡。cap 是"硬夹"、`entropy_coef` 是"软引导"，
两者同时动就没法归因。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v24_ppo_cfg import G1AmpWalkV24PPORunnerCfg

#: σ 上限的目标值（rad；`action_scale=1.0` ⇒ 就是关节位置目标的标准差）
SIGMA_CAP_V26 = 0.20
#: σ 上限的退火周期（本 run 的 update 次数）：0.57 → 0.20 用 300 轮走完
SIGMA_CAP_RAMP_V26 = 300


@configclass
class G1AmpWalkV26PPORunnerCfg(G1AmpWalkV24PPORunnerCfg):
    """v26：V24 全部设置 + `clip_max_std=0.20`（300 轮退火）；从 V24@8000 热启动。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- 唯一变量：σ 上限（目标 + 退火周期）----
        self.algorithm.clip_max_std = SIGMA_CAP_V26
        self.algorithm.clip_max_std_schedule = SIGMA_CAP_RAMP_V26

        # 起点：热启动（AMP 同形 ⇒ 不丢判别器）
        self.ckpt_manipulator = None
        self.run_name = "v26_hot_v24_m8000_sigma0.20_ramp300"


__all__ = [
    "SIGMA_CAP_V26",
    "SIGMA_CAP_RAMP_V26",
    "G1AmpWalkV26PPORunnerCfg",
]
