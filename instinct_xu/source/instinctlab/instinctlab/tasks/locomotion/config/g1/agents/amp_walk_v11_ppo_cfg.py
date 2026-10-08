"""V11 的算法配置：**加强风格压力**（唯一算法侧改动）+ σ 上限放宽到 0.6。

| 字段 | V10b | **V11** | 说明 |
|---|---|---|---|
| `style_lerp`（λ） | 0.75 | **0.5** | 与下面一项共同决定有效风格权重 |
| `discriminator_reward_coef`（c） | 0.25 | **0.5** | ⇒ `(1−λ)·c`：**0.0625 → 0.25（×4）** |
| `clip_max_std` | 0.7 | **0.6** | 用户指定"宽容一些"；V10b 峰值 0.63 ⇒ 本轮相当于**轻收紧** |
| `init_noise_std` | 0.6 | 0.35 | ⚠️ **续训时该项不生效**（σ 用 checkpoint 里学到的 0.58）；写小只是给"从零"时的默认 |
| `clip_min_std` | 0.03 | 0.02 | 沿用 5% × init 的比例 |
| `entropy_coef` | 0.005 | 0.005 | 不动 |
| `schedule` | fixed | fixed | 不动 |
| `use_mirror_loss` | True | True | 不动 |

## 为什么现在敢把风格权重 ×4

上次（V9g）把 `(1−λ)c` 提到 0.25 时：参考集是 **17 个 clip**（"走"的模态被摊薄）、
`disable_style_on_stand` 门控**读错了数组**（30% 站立 env 一直在被风格拉），
所以"×4 没效果"这个结论是在两个 bug 同时存在下得到的，**不可信**。

现在：参考集收敛到 **2 个 walk clip**（判别器第一次真正专注在"走"上）、
门控用真值指令（`amp_stand_frac ≈ 0.30` 已验证）、策略**已经会走** ⇒
风格项可以只负责"把步态拉向参考"，而不是同时还要"发现走路"。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v10b_ppo_cfg import G1AmpWalkV10bPPORunnerCfg

#: 风格/任务 凸组合（λ）：0.75 → 0.5
STYLE_LERP_V11 = 0.5
#: 风格奖励缩放（c）：0.25 → 0.5 ⇒ 有效权重 (1−λ)c = 0.25
DISC_REWARD_COEF_V11 = 0.5
#: σ 上限（用户指定 0.6；V10b 峰值 0.63）
CLIP_MAX_STD_V11 = 0.6
#: σ 初值（**续训时无效**，见模块 docstring）
NOISE_STD_V11 = 0.35
#: σ 下限
CLIP_MIN_STD_V11 = 0.02


@configclass
class G1AmpWalkV11PPORunnerCfg(G1AmpWalkV10bPPORunnerCfg):
    """v11：加强风格压力 + σ 上限 0.6。"""

    def __post_init__(self):
        super().__post_init__()

        self.algorithm.style_lerp = STYLE_LERP_V11
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V11
        self.algorithm.clip_max_std = CLIP_MAX_STD_V11
        self.algorithm.clip_min_std = CLIP_MIN_STD_V11
        self.policy.init_noise_std = NOISE_STD_V11

        self.run_name = (
            f"v11_ns{NOISE_STD_V11}_maxstd{CLIP_MAX_STD_V11}_ent0.005_fixedlr"
            f"_lerp{STYLE_LERP_V11}_ampc{DISC_REWARD_COEF_V11}_airtime8_ar0.06_fromV10b"
        )


__all__ = [
    "CLIP_MAX_STD_V11",
    "CLIP_MIN_STD_V11",
    "DISC_REWARD_COEF_V11",
    "NOISE_STD_V11",
    "STYLE_LERP_V11",
    "G1AmpWalkV11PPORunnerCfg",
]
