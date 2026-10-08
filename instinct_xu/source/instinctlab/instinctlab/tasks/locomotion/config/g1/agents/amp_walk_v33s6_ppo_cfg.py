"""V33-S6 的算法配置：**AMP 口径与 S3/S4/S5 逐字相同**（判别器保留），唯一变量 = `discriminator_reward_coef`。

## 为什么是这个实验（本轮最重要的结构发现）

混合公式（`instinctlab/.../utils/wrappers/instinct_rl/algorithms.py:572` + `instinct_rl/.../ppo.py:162`）：

$$r = \\lambda\\cdot r_\\text{task} + (1-\\lambda)\\cdot \\text{coef}\\cdot \\text{style}$$

用 **S5 末点实测**：

| 量 | 值（每步） |
|---|---|
| 任务项加权和 $r_\\text{task}$（`Episode_Reward/*`） | ≈ **2.8** |
| 风格项 $\\text{coef}(1-\\lambda)\\text{style}$ = 1.0×0.25×0.2517 | ≈ **0.063** |
| **风格占 $r$ 的比例** | **2.9%** |

⇒ **风格项根本没有能力改写步态**：λ 在 0.70↔0.75 之间调只是把 2.6% 挪到 2.9%（≈空转）；
步态实际全部由手写项（`feet_air_time` / `flat_orientation_l2` / `feet_lateral_width` /
`feet_crossing` / `joint_deviation_*`）塑造 ⇒ **S4 取消 `feet_air_time` 后风格"接不住"，步态崩**（倾角 19.2°、hop 0.71）。

## 与 V28（coef 4.0 崩）的区别 —— 条件已变

| V28 崩因 | V28 时 | **今天** |
|---|---|---|
| 风格奖励形状**无上界**（`−log(1−D)` 单步可达 +12）⇒ 奖励离群 | 存在 | ✅ V29 起已换 **`bounded_quad` ∈ [0,1]** |
| 风格占比冲到 **47%**（当时 `style≈0.70`） | 存在 | coef=4 下只有 **10.6%** |
| 判别器**饱和**、信号无信息（`style_std` 0.058） | 存在 | ✅ S5 `style_std` **0.173（×3）**、`zero_frac` 0.104 |

## 本臂唯一变量（相对 S4，fork 点也相同 = S3@12500）

| 项 | S4 | **S6** |
|---|---|---|
| `discriminator_reward_coef` | 1.0 | **4.0** ⇒ 风格项权重 0.25 → **1.0（4×）** |
| `style_lerp` (λ) | 0.70 | **0.70（不动）** |
| 环境侧奖励表 | air0 / track 3.0+1.5 / flat−4 / arms0 | **完全继承 S4**（`G1AmpWalkV33S6EnvCfg` 直接继承 `G1AmpWalkV33S4EnvCfg`） |
| 判别器口径 / AMP 状态 396 / 无白化 / raw GP / 同专家集 | — | **全不动** ⇒ `ckpt_manipulator=None`（**保留判别器**） |
| 起点 | S3@12500 | **S3@12500（同一 ckpt）** ⇒ 与 S4 单变量对比 |

## 预登记判据（步态风格为主 —— 用户已明确"速度可以放低，但需要较好的步态风格"）

**必须变好（风格真的在起作用）**
1. `air_time_single_stance` ↑（S4 无此探针；对标 S5 0.084 / S3 0.103 / V31 0.231）
2. `air_time_in_band_frac` ↑（S5 0.361）、`air_time_hop_frac` ↓（S5 0.710）
3. `tilt_deg_mean` ↓（S4 19.2° / S5 15.0°）、`foot_sep_mean` ↓（S4 0.320）
4. `amp_style_reward` 明显 > 0.25，且 **`amp_style_std` > 0.05**、`amp_style_zero_frac` **不长期 = 1.0**

**红线（**500 轮处必查** —— V28 是"渐变退化 470 轮后摔死"）**
- `Mean reward` **不跌破 20**（V28 是 42→15）
- `Train/mean_episode_length ≥ 500`、`Step/action_abs_mean ≤ 0.62`
- `tracking/vxy_rmse ≤ 0.35`（S4 是 0.404；允许变差但不许崩）

**风险（`bounded_quad` 的正反馈）**：$d\\to1$ 时 style→1.0 ⇒ coef=4 下风格占比可达 ~40% ⇒
若 500~800 轮看到 `tracking` 崩 + `action_abs_mean` 上冲 ⇒ 回退 coef 到 2.0。

## 早停点

`model_13000.pt` / `model_13500.pt` 各存一份（`save_interval=500`）⇒ 与 S4 的 13000/13500 可直接对比回放。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v31_ppo_cfg import G1AmpWalkV31PPORunnerCfg
from .amp_walk_v33s_ppo_cfg import AMP_V33S_EXPERT_BUFFER

#: 风格占比（λ）：与 S4 逐字相同（0.70）。
#: ⚠️ 注意：λ 只决定"名义占比"，真正决定风格影响力的是下面的 `coef`。
AMP_V33S6_STYLE_LERP = 0.70

#: ⭐ 本臂唯一变量：风格项的权重系数 1.0 → **4.0**
#: 有效风格权重 `(1−λ)·coef`：0.30×1.0 = 0.30 → **0.30×4.0 = 1.20（4×）**。
#: 实测风格占比：2.9% → **10.6%**（$r_\\text{task}\\approx2.8$/步、style≈0.25）。
AMP_V33S6_DISC_REWARD_COEF = 4.0


@configclass
class G1AmpWalkV33S6PPORunnerCfg(G1AmpWalkV31PPORunnerCfg):
    """v33-S6：S4 的判别器口径与环境侧奖励 + **coef 4.0** + 保留判别器续训（fork = S3@12500）。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- 判别器口径：与 S3/S4/S5 逐字相同（防御性 pop，确保没有 normalizer 残留）----
        dk = dict(self.algorithm.discriminator_kwargs)
        dk.pop("normalizer_class_name", None)
        dk.pop("normalizer_kwargs", None)
        self.algorithm.discriminator_kwargs = dk
        self.algorithm.amp_grad_pen_space = "raw"
        self.algorithm.amp_normalizer_expert_only = True
        # ---- 396 维专家集（维度必须与环境侧的 anchor 重力通道一致）--------------
        self.algorithm.amp_expert_buffer_path = AMP_V33S_EXPERT_BUFFER
        # ---- 风格占比：不动（= S4 的 0.70）---------------------------------------
        self.algorithm.style_lerp = AMP_V33S6_STYLE_LERP
        # ---- ⭐ 唯一变量：风格项权重 ×4 ------------------------------------------
        self.algorithm.discriminator_reward_coef = AMP_V33S6_DISC_REWARD_COEF
        # ---- ⭐ 判别器**保留**：口径没变，续训即可 ---------------------------------
        self.ckpt_manipulator = None

        self.run_name = "v33s6_ampcoef4_noairtime_keepdisc_from12500"


__all__ = [
    "AMP_V33S6_DISC_REWARD_COEF",
    "AMP_V33S6_STYLE_LERP",
    "G1AmpWalkV33S6PPORunnerCfg",
]
