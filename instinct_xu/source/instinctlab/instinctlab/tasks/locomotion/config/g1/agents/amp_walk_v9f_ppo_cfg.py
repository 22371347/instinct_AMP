"""V9f 的 PPO/算法配置：**去掉自适应学习率 + 放大探索幅度**（见 env cfg 的完整说明）。

两处改动：
1. `schedule: "adaptive" → "fixed"` —— 真实 LR 稳定在 1e-3（原先每步 ×/÷1.5、夹在 [1e-5, 1e-2]）。
2. `init_noise_std 0.2 → 0.3`、`clip_max_std 0.3 → 0.4` —— 探索**幅度**（走路要大幅度、跳只要小角度）。

其余（镜像 ON / entropy 0.0025 / AMP c=0.25 / arms 归零）全部沿用 v9e。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v9e_ppo_cfg import G1AmpWalkV9ePPORunnerCfg

#: σ 初值：0.2 → 0.3（±17°）。走路需要 0.4~0.6 rad 的髋/膝往复，0.2 的 σ 下这是 2~3σ。
NOISE_STD_V9F = 0.3
#: σ 上限：0.3 → 0.4，给上面的初值留上升空间（仍封顶 ±23°）。
CLIP_MAX_STD_V9F = 0.4


@configclass
class G1AmpWalkV9fPPORunnerCfg(G1AmpWalkV9ePPORunnerCfg):
    """v9f：v9e + `schedule="fixed"` + σ 0.3/上限 0.4。"""

    def __post_init__(self):
        super().__post_init__()

        # ① 学习率：去掉自适应（`ppo.py:236-263` 的 ×/÷1.5 与 [1e-5, 1e-2] 夹取）。
        #    想动学习率数值就改 `self.algorithm.learning_rate`（现在是基类的 1e-3）。
        self.algorithm.schedule = "fixed"

        # ② 探索幅度：走路需要大幅度动作，σ=0.2 探索不到 ⇒ 回到 0.3，并把上限放到 0.4。
        self.policy.init_noise_std = NOISE_STD_V9F
        self.algorithm.clip_max_std = CLIP_MAX_STD_V9F

        self.run_name = (
            f"v9f_ns{NOISE_STD_V9F}_maxstd{CLIP_MAX_STD_V9F}_fixedlr"
            "_ent0.0025_ampc0.25_noarms_mirror_airtime_halfstand"
        )


__all__ = ["CLIP_MAX_STD_V9F", "NOISE_STD_V9F", "G1AmpWalkV9fPPORunnerCfg"]
