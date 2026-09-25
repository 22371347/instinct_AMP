"""AMP 行走教师 **v3** 的 PPO runner 配置：只把探索噪声起点 `1.0 → 0.3`。

为什么这一步是 v3 的关键（比换奖励更关键）
----------------------------------------
`instinct_rl` 的 `ActorCritic.std` 是**绝对**标准差（`init_noise_std * ones(29)`），
而本仓库的动作尺度是 `action_scale = 1.0`（`q_target = q_default + 1.0 · a`）。
两者相乘意味着：

| `init_noise_std` | 物理上的关节目标噪声 | 后果 |
|---|---|---|
| **1.0**（v1/v2） | **±1.0 rad ≈ ±57°** | 只有「最抗噪」的解能活下来 |
| **0.3**（v3） | **±0.3 rad ≈ ±17°** | 需要精确时序的交替步态才可能学出来 |

实测证据：v1 从 σ=1.0 起步，到 5000 轮仍有 **0.756**；v2 更是涨到 **1.580**
（≈±90°）。对比 AMP_mjlab：它用**逐关节** scale（0.0745~0.5476），
同样 σ=1.0 只等价于 **4°~31°** —— 所以「同一个 1.0」在两套约定下物理含义差 2~13 倍。

⭐ **为什么这也解释了「局部收敛到单脚跳跃」**：跳跃是**大探索噪声下才成立的吸引子**
（支撑腿处于自稳构型、对 ±57° 目标噪声鲁棒）；而平滑交替步态依赖精确的摆动腿时序，
±57° 噪声会直接摧毁它。所以从 σ≈0.3 起步的策略**不会进入**那个吸引子；
反过来，从一个已经跳跃的 checkpoint 续训，大概率就困在里面出不来。

`clip_min_std` 为什么也要跟着改
------------------------------
`amp_walk_ppo_cfg.py` 里原来的注释写着：`init_noise_std=1.0` 时，
AMP_mjlab 的 `min_normalized_std = 0.05`（**归一化**下限 5%）**等价于**绝对下限 0.05。
`init_noise_std` 降到 0.3 后这个等价关系就断了 —— 若仍写 0.05，σ 还能一路塌到
±2.9°，相当于把「归一化下限」从 5% 放宽到 17%，会让判别器过早单方面主导。

⇒ 按同一比例缩放：**0.05 × (0.3 / 1.0) = 0.015**，保持「归一化下限 5%」的语义不变。

本次**不动**的：`actor_hidden_dims` / `critic_hidden_dims` / `entropy_coef` /
`learning_rate` / 判别器全部超参 / `style_lerp` / `discriminator_reward_coef`。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_ppo_cfg import G1AmpWalkPPORunnerCfg

#: v3 的探索噪声起点（绝对标准差，rad）
NOISE_STD_V3 = 0.3
#: = 0.05 × (0.3 / 1.0)：保持 AMP_mjlab「归一化下限 5%」的语义
CLIP_MIN_STD_V3 = 0.015


@configclass
class G1AmpWalkV3PPORunnerCfg(G1AmpWalkPPORunnerCfg):
    """v3：把探索噪声起点从 1.0 降到 0.3，其余全部继承（含 AMP 判别器全套超参）。

    ⚠️ 只改两个字段，**不改任何结构** —— 所以策略网络/算法与 v1/v2 完全同构，
    checkpoint、ONNX 导出、蒸馏 teacher 侧接口都不受影响。
    """

    def __post_init__(self):
        super().__post_init__()
        # 绝对标准差：±0.3 rad ≈ ±17° 关节目标噪声（v1/v2 是 ±57°）
        self.policy.init_noise_std = NOISE_STD_V3
        # 同比例缩放下限，见模块 docstring
        self.algorithm.clip_min_std = CLIP_MIN_STD_V3


__all__ = ["CLIP_MIN_STD_V3", "NOISE_STD_V3", "G1AmpWalkV3PPORunnerCfg"]
