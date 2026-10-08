"""V19a 的算法配置：**让判别器"敢说话"**（AMP 三件套）+ 保持 λ/c/quad 形式不变。

起点：V18（`log_quad` + `wasserstein` + `MSELoss` + λ0.75 + c1.0 + 10 帧 670 维）。
⚠️ `log_quad` / `wasserstein` **不是我这里设的** —— 它们由 **V8** 设定，被 V9→V18 全链继承
   （`agent.yaml` 实测：`amp_reward_form: log_quad`、`discriminator_reward_type: wasserstein`）。

为什么要动判别器（V18 @2000 轮 × 12 帧 AMP 的**实测证据**）
---------------------------------------------------------
`Loss/amp_total_loss` 的逐项分解（系数按 `wasabi.py::update` 的
`getattr(self, k + "_coef", 1.0)` 还原）：

    discriminator_loss            × 1.0     = 0.1527    ( 37% )
    discriminator_gradient_penalty× 5.0     = 0.2095    ( 51% ) ★ 主导项
    discriminator_weight_decay    × 3.0e-4  = 0.0319    (  8% )
    logit_weight_decay            × 1.0     = 0.0153    (  4% ) ⚠️ 配置写的 0.04 从未生效
    ─────────────────────────────────────────────────
    合计                                     0.4094  == TB 的 Loss/amp_total_loss ✅

  ① **`gradient_penalty` 占了判别器损失的一半，而它的 `tolerance = 0`** ⇒ 它惩罚**任何**
     输入梯度（`mean‖∂d/∂x‖²`，x 是 670 维未归一化状态）。实测该量为 0.042
     （⇒ 平均 ‖∂d/∂x‖ ≈ 0.19）⇒ 判别器被**训练成一个非常平的函数**。
     这直接解释了"风格项没有梯度"：策略动一点，d 几乎不动。
  ② `weight_decay`（`Σw² = 106.5`）只占 **8%**，且判别器权重 RMS = 0.0094 **低于随机初始化**
     ⇒ 权重被整体缩小，输出幅度被压到 `d ∈ [−0.62, +0.62]`（MSE 目标却是 ±1）。
     ⚠️ 更正：此前"`weight_decay ≈ 105 ≫ discriminator_loss ≈ 0.1`，判别器被压死"的说法是
     **错的** —— 那是把**未乘系数**的 `Σw²` 和**已乘系数**的 loss 直接比了。乘上系数后
     wd 只占 8%，**不是**主因。
  ③ `logit_weight_decay`：代码里该项的系数取键 `logit_weight_decay_coef`（**不存在**）
     ⇒ 落到 `getattr(..., 1.0)` 的默认值 **1.0**（而配置里的
     `discriminator_logit_weight_decay_coef = 0.04` **从未被使用**）。
     它专门惩罚**输出层**权重 ⇒ 实测输出层 `‖w‖₂ = 0.125`、`RMS = 0.0055`。

本版改的四处
------------
| # | 键 | V18 | **V19a** | 依据 |
|---|---|---|---|---|
| ① | `discriminator_gradient_penalty_coef` | 5.0 | **1.0** | 占比 51% 的"压平"力，降到 1/5 |
| ② | `discriminator_weight_decay_coef` | 3.0e-4 | **1.0e-5** | 占比 8% → 0.3%（正则交给 GP 就够） |
| ③ | `discriminator_logit_weight_decay_coef` | 0.04 | **0.0** | 该键的真实系数是 **1.0**（键名 bug）⇒ 用 0 显式关掉 |
| ④ | `amp_reward_eps`（`−log(1−D)` 的上端限幅） | 1e-6 | **0.05** | 判别器修好后 `d` 会向 ±1 靠 ⇒ 不限幅时上端会冲到 13.8 而淹没任务项；限到 **3.0** 保住量级 |

**不动**：`h = 10`、`stride = 1`（用户指定）、`log_quad` 形式、`λ = 0.75`、`c = 1.0`、
`MSELoss`(±1)、`AdamW lr 1e-4`、5 clip 数据集、三个跟踪项。

⚠️ 与本轮同时进行的 `feet_air_time.soft: 0.3 → 0.4`（env 侧）**互相独立**、且可从
不同探针分别读出（AMP 侧看 `Step/amp_d`/`amp_style_std`/`Loss/discriminator_*`；
步态侧看 `Step/air_time_single_stance` / `air_time_in_band_frac` /
`Episode_Reward/feet_air_time`）⇒ 事后仍能归因。

后续（V19b 预案，待用户点头）
-----------------------------
* **GP 继续放松**：`gradient_penalty_coef 1.0 → 0.0` 或 `discriminator_gradient_torlerance
  0.0 → 0.15`（若 V19a 后 `amp_d` 仍钉在 ±0.6 不动 ⇒ 说明 GP 还是主要约束）。
* **h / stride**：离线扫描的结论是"时间结构类退化需要 h ≈ 8–15"（h=2 完全不可分），
  本轮保持 10 帧；若要再测，`stride = 2`（同 h 覆盖更长物理窗口）比加 h 更划算。
* **quad**：`c` 或 D 的曲率 —— 历史证据（v9g/v9h：`c` ×4 后 `amp_d` 几乎不动）表明
  **权重不是杠杆**，故本轮不动；若 V19a 后判别器有分辨力了，再测才有效。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v18_ppo_cfg import (
    DISC_REWARD_COEF_V18,
    STYLE_LERP_V18,
    G1AmpWalkV18PPORunnerCfg,
)

#: ① 梯度惩罚系数：5.0 → 1.0（占比 51% 的"压平"力）
GP_COEF_V19A = 1.0
#: ② 判别器权重衰减：3.0e-4 → 1.0e-5
WD_COEF_V19A = 1.0e-5
#: ③ 输出层权重衰减：**实际按 1.0 生效**（键名 bug）⇒ 显式关掉
LOGIT_WD_COEF_V19A = 0.0
#: ④ `−log(1 − D)` 的上端限幅：1e-6 → 0.05（奖励上界 13.8 → 3.0）
AMP_REWARD_EPS_V19A = 0.05


@configclass
class G1AmpWalkV19APPORunnerCfg(G1AmpWalkV18PPORunnerCfg):
    """v19a：V18 全部 + 判别器"三件套" + 风格奖励上端限幅。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- ① ② ③ 判别器的正则/梯度惩罚 ------------------------------------
        self.algorithm.discriminator_gradient_penalty_coef = GP_COEF_V19A
        self.algorithm.discriminator_weight_decay_coef = WD_COEF_V19A
        self.algorithm.discriminator_logit_weight_decay_coef = LOGIT_WD_COEF_V19A

        # ---- ④ 风格奖励 `−log(1 − D)` 的上端限幅 -----------------------------
        self.algorithm.amp_reward_eps = AMP_REWARD_EPS_V19A

        # ---- 风格混合：与 V18 逐字相同（λ 0.75 / c 1.0 / log_quad / wasserstein）----
        self.algorithm.style_lerp = STYLE_LERP_V18
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V18

        # ---- ⭐ 热启动：从 V18 的 checkpoint 载入 **actor/critic/优化器/normalizer**，
        #      但**丢掉判别器及其优化器**（口径变了 ⇒ 必须重学；670 维虽兼容，但
        #      带着被 GP 压扁的权重起步等于没改）。
        self.ckpt_manipulator = "instinctlab.utils.ckpt_manipulators:drop_amp_discriminator"

        self.run_name = (
            f"v19a_warmV18_gp{GP_COEF_V19A}_wd{WD_COEF_V19A:g}"
            f"_logitwd{LOGIT_WD_COEF_V19A:g}_epse{AMP_REWARD_EPS_V19A:g}"
            f"_airsoft0.4_style_c{DISC_REWARD_COEF_V18}_lerp{STYLE_LERP_V18}"
        )


__all__ = [
    "AMP_REWARD_EPS_V19A",
    "GP_COEF_V19A",
    "LOGIT_WD_COEF_V19A",
    "WD_COEF_V19A",
    "G1AmpWalkV19APPORunnerCfg",
]
