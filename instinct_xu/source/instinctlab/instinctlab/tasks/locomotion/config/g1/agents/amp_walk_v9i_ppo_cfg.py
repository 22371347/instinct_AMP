"""V9i 的算法配置：**只调"探索量级"**（奖励集与 V9h 逐字相同）。

| 字段 | V9h | **V9i** | 参照 |
|---|---|---|---|
| `policy.init_noise_std` | 0.3 | **0.6** | 仓库里不含 AMP 的行走配方 `G1OmniLocoPPORunnerCfg` = **1.0** |
| `algorithm.clip_max_std` | 0.4 | **0.7** | 参考配方没有上限；给 σ 留上升空间，但**必须有天花板**（v8 无上限时 σ 涨到 1.04 rad） |
| `algorithm.entropy_coef` | 0.0025 | **0.005** | 参考配方 = 0.008；取中间值 |
| `schedule` | fixed | fixed | 不动（参考配方是 adaptive，留作下一个单变量） |
| 风格侧（`style_lerp` / `discriminator_reward_coef`） | 0.75 / 0.25 | 0.75 / 0.25 | **不动**（V9h vs V9i 要只差探索） |
| 镜像损失 | ON | ON | 不动 |

### 为什么是 0.6（而不是参考的 1.0，也不是一开始写的 0.5）

用户实测印象：**σ=1 的那一版（v1/v2，基类 `PolicyCfg.init_noise_std=1.0`）反而略有
"学会走路模态"的倾向**，但 σ=1 带来限位与不稳定。复盘本仓库的 σ=1 时代：
真正的问题不是"初值高"，而是 **σ 会失控**（`entropy_coef=0.008` + 无上限 ⇒ v8 实测
σ 涨到 **1.04 rad**）。⇒ 本版取 **初值 0.6 + 硬上限 0.7 + entropy 0.005**，
把 σ 关在 **0.3→0.7** 这个有界区间里：足够高以触发参考配方那种"大幅度探索"，
又不会像无上限时那样一路跑飞。

⚠️ 高 σ 的两个已知副作用与应对：
1. **限位**：V9h 实测 `Episode_Reward/dof_pos_limits` 只有 −0.0207/s（很干净），
   说明现在的奖励集（`dof_pos_limits −10` + `dof_torques_l2` + `action_rate_l2`）
   本身就能压住；σ 升高后**盯住这个数**，若明显恶化再上备用手段（见下）。
2. **不稳定**：盯 `Episode_Termination/base_contact`（V9h 0.449）。

备用手段（本版**不做**，留给后续单变量）：把动作项 `clip` 从 `None` 改成 ±3
（只砍尾部：V9h 的 `|a|` 均值 0.77、`max` 6.1 ⇒ 只影响病理样本）。
⚠️ 这会改**动作契约** ⇒ 必须同步改 `INTERFACE_CONTRACT.md` 与 MuJoCo 回放工具
（`g1_mujoco_play.py` 也得施加同一个 clip），不能只改训练端。

⚠️ 三个旋钮是**同一个概念变量**（"探索量级"）的三个侧面：`init` 是起点、`entropy` 是维持力、
`cap` 是天花板 ⇒ 一起改才是一次干净的"探索量级"实验（而不是三个混杂变量）。

⚠️ 口径：`σ_joint = init_noise_std × action_scale`，我们 `action_scale = 1.0`
⇒ `0.5` 就是 **±0.5 rad ≈ ±29°**（V9h 的 0.3 是 ±17°，参考配方 1.0 是 ±57°）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v9f_ppo_cfg import G1AmpWalkV9fPPORunnerCfg

#: 探索噪声初值（±34°）—— 参考配方 1.0，取中间偏保守（用户：σ=1 时已略有走路倾向）
NOISE_STD_V9I = 0.6
#: σ 上限：0.4 → 0.7（给 σ 上升空间，但必须有天花板：v8 无上限时涨到 1.04）
CLIP_MAX_STD_V9I = 0.7
#: 熵系数：0.0025 → 0.005（参考 0.008）
ENTROPY_COEF_V9I = 0.005
#: 探索下限：不允许 σ 塌到 0（沿用 5% × init 的比例）
CLIP_MIN_STD_V9I = 0.03


@configclass
class G1AmpWalkV9iPPORunnerCfg(G1AmpWalkV9fPPORunnerCfg):
    """v9i：v9h 的奖励/AMP 设置 + **探索量级**（σ 0.6 / 上限 0.7 / entropy 0.005）。"""

    def __post_init__(self):
        super().__post_init__()

        self.policy.init_noise_std = NOISE_STD_V9I
        self.algorithm.clip_max_std = CLIP_MAX_STD_V9I
        self.algorithm.clip_min_std = CLIP_MIN_STD_V9I
        self.algorithm.entropy_coef = ENTROPY_COEF_V9I
        # 风格侧与 v9h 一致（V9h/V9i 只差探索）
        self.algorithm.style_lerp = 0.75
        self.algorithm.discriminator_reward_coef = 0.25

        self.run_name = (
            f"v9i_ns{NOISE_STD_V9I}_maxstd{CLIP_MAX_STD_V9I}_ent{ENTROPY_COEF_V9I}"
            f"_fixedlr_lerp0.75_ampc0.25_rew20_explore"
        )


__all__ = [
    "CLIP_MAX_STD_V9I",
    "CLIP_MIN_STD_V9I",
    "ENTROPY_COEF_V9I",
    "NOISE_STD_V9I",
    "G1AmpWalkV9iPPORunnerCfg",
]
