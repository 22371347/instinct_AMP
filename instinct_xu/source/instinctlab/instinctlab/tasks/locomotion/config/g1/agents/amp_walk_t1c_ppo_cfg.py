"""**T1c 的 PPO runner**：T1a2 之上**只改一个数** —— `style_lerp` 1.0 → **0.9**。

============================================================================
为什么是 0.9（T1b 的读数直接给出结论）
============================================================================
lerp 模式下总奖励（`algorithms.py:572` 的实际代码）：

    r_total = r_task + aux,   aux = (1−λ)·(style − r_task/c)
    ⇒ r_total = r_task·(1 − (1−λ)/c) + (1−λ)·style

⇒ **风格权重 = (1−λ)**，与 `discriminator_reward_coef` 无关（c=4 只把任务项乘 0.9375）。

T1b（λ=0.75 ⇒ 风格权重 **0.25**）实测：**1100 轮内 `vx_high` 从 0.59 单调崩到 0.014**
（九档 `vel_mean` 全≈0，回到"站着不动"），而三项"丑模态"指标同时改善
（`track_ang_vel_z_exp` 0.156→0.329、`action_rate_l2` −0.456→−0.283、
`track_default_pose_on_stand` −0.394→−0.205）—— **风格项确实在发力，只是强到把前进压掉了**。

T1c 把风格权重降到 T1b 的 **40%**（0.25 → 0.10），其余逐字不动，从 **T1a2 的
`model_7000.pt` 重新开始**（⚠️ 不从已崩的 T1b ckpt 续训，否则带着坏先验）。

## 路径设计（为什么取 0.9）
    1.0  → 纯任务项（T1a2，`vx_high` 0.62，无风格）
    0.9  → 风格权重 0.10   ← 本臂
    0.75 → 风格权重 0.25（T1b，`vx_high` 崩到 0.014）

⇒ 若 T1c 能"风格有改善 且 守住 `vx_high` ≥0.5"，则 0.10 是可行点，再往下做细调；
   若 T1c 仍然崩，说明问题不在**强度**而在**方向**（判别器的 r_amp 本身在奖励"低能量"
   模态）⇒ 下一个杠杆应改成 `amp_reward_form: bounded_quad → log_quad`
   （消除实测 35% 的零梯度死区）或检查参考数据。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_t1a2_ppo_cfg import G1AmpWalkT1a2PPORunnerCfg

#: T1c 的风格项权重（T1c 相对 T1a2 的**唯一**数值改动）
AMP_T1C_STYLE_LERP = 0.9


@configclass
class G1AmpWalkT1cPPORunnerCfg(G1AmpWalkT1a2PPORunnerCfg):
    """T1c：T1a2 + `style_lerp` 1.0 → 0.9（风格权重 0.10 = T1b 的 40%）。"""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.style_lerp = AMP_T1C_STYLE_LERP
        self.run_name = "t1c_style0p9_from_t1a2"
