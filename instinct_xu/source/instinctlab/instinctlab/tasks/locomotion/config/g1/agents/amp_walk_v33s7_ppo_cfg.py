"""V33-S7 的算法配置：**只修复判别器口径**（回归 AMP_mjlab 的正则），**权重一律不动**。

## 为什么（本轮最重要的失效根因，见 `instinct/docs/v33_tracking_gait_analysis.md` §5.7）

与参考实现逐项对照（左 = `instinct/repos/InstinctMJ` 的 parkour `AmpAlgoCfg`；右 = 我们 V33 实运行）：

| 项 | mjlab 侧 | 我们（旧） | 判定 |
|---|---|---|---|
| 判别器结构 / 损失 / 奖励形 | [1024,512] / `MSELoss`（**expert→+1, actor→−1**）/ `quad` | 同 / 同 / `bounded_quad` | ✅ 同 |
| 优化器（lr 1e-4、betas、epochs 5×4、entropy、max_grad_norm） | — | 逐项相同 | ✅ 同 |
| 有效风格权重 `(1−λ)·coef` | 0.25（加性混合） | 0.25（lerp 0.75） | ✅ **等效同**（故“权重太小”不成立） |
| `discriminator_gradient_penalty_coef` | **5.0** | **10.0** | ❌ 2× |
| `discriminator_weight_decay_coef`（loss 级） | **3e-4** | **0.0**（V19a 关） | ❌ |
| `discriminator_logit_weight_decay_coef` | **0.04** | **0.0**（V19a 关） | ❌ |
| 优化器分组 wd | 无（单一 3e-4） | **trunk 1e-3 / head 1e-2** | ❌ **head 极强** |

**实测：判别器从未到达设计工作点**（`MSELoss` 的目标是 ±1）：

| 量 | 设计目标 | S5 实测（末点） |
|---|---|---|
| `amp_d_ref`（参考批） | **→ +1** | **+0.216**（整段 0.0~0.44 **漂移**） |
| `amp_d`（活体批） | **→ −1** | −0.724 |
| `Loss/discriminator_loss` | → 0 | **0.103**，**1000 轮不降** |
| `amp_style_reward` | 两端分化（≈0 或 ≈0.8+） | **长期挤在 0.25**（0.203~0.270） |
| `disc_grad/angvel ÷ pos` | 判步态应以**位姿**为主 | **17×**（几乎只看角速度） |

⇒ **根因链**：
1. **正则把 logit 压死在错误区间**（head wd 1e-2、loss 级 wd/logit_wd 被关、GP 2×）
   ⇒ `d` 到不了 ±1 ⇒ 风格奖励常年落在 `bounded_quad` 的**斜坡中段 ~0.25**、`d_ref` 一直在漂
   ⇒ **风格信号信噪比低**；
2. ⇒ (a) AMP 没有能力塑造步态；(b) **抬权重＝放大低信噪比信号 ⇒ 必崩**
   （S6 `coef=4` 悬崖崩、V28 渐变崩，机制一致）；
3. ⇒ 所以正确的第一步不是调权重，而是**让判别器真的学会把参考与活体分开到 ±1**。

## 本臂唯一变量：判别器侧 5 个超参（全部向 mjlab 对齐）

| 项 | 我们（旧） | **S7** | mjlab |
|---|---|---|---|
| `discriminator_gradient_penalty_coef` | 10.0 | **5.0** | 5.0 |
| `discriminator_weight_decay_coef` | 0.0 | **3.0e-4** | 3e-4 |
| `discriminator_logit_weight_decay_coef` | 0.0 | **0.04** | 0.04 |
| 优化器 trunk wd | 1e-3 | **3.0e-4** | 3e-4（无分组，等价于两侧同值） |
| 优化器 head wd | **1e-2** | **3.0e-4** | 同上 |

**明确不动**：`discriminator_reward_coef = 1.0`、`style_lerp = 0.75`、奖励表（= S5：`air_time=1.0` /
`track_lin=3.0` / `track_ang=1.5`）、AMP 口径（396 / 无白化 / raw GP / 同专家集）、
`amp_expert_independent_sampling=True`、`schedule="fixed"`（adaptive 留作后续单独变量）。

## 起点

* **actor/critic 从 S5 的 `model_13500.pt` 热启**（AMP 口径没变、actor obs 378 不变 ⇒ 权重可直接加载）；
* **判别器从零重学**（正则变了 ⇒ 旧权重的校准不可复用）⇒ `drop_amp_discriminator`。

## 预登记判据（**先看判别器能否干净分开** —— 这是“风格权重是否有意义”的前置条件）

1. `Step/amp_d_ref` → **≥ +1.0**（现 0.216）；`Step/amp_d` → **≤ −1.0**（现 −0.724）。
2. `Loss/discriminator_loss` 明显下降（现 0.103 且不降）；`Loss/discriminator_gradient_penalty` 不爆。
3. `Step/amp_style_reward` **两端分化**（不再长期挤 0.25）、`amp_style_std` 稳定且不塌。
4. 红线：`Mean episode length ≥ 500`、`Step/action_abs_mean ≤ 0.62`、`tracking/vxy_rmse ≤ 0.35`。
5. 若判据 1~3 成立而步态仍无变化 ⇒ 说明信号内容才是下一个瓶颈 ⇒ 再上
   任务条件化判别器（命令 + 绝对速度通道）或镜像增广。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v27_ppo_cfg import DROP_DISC
from .amp_walk_v31_ppo_cfg import G1AmpWalkV31PPORunnerCfg
from .amp_walk_v33s_ppo_cfg import AMP_V33S_EXPERT_BUFFER

#: 判别器正则（回到 `InstinctMJ .../parkour/config/g1/rl_cfg.py::AmpAlgoCfg` 的值）
AMP_V33S7_GP_COEF = 5.0
AMP_V33S7_DISC_WD = 3.0e-4
AMP_V33S7_DISC_LOGIT_WD = 0.04
AMP_V33S7_OPT_WD_TRUNK = 3.0e-4
AMP_V33S7_OPT_WD_HEAD = 3.0e-4
#: 风格占比：与 S5 逐字相同（0.75）—— 本臂**不动权重**
AMP_V33S7_STYLE_LERP = 0.75


@configclass
class G1AmpWalkV33S7PPORunnerCfg(G1AmpWalkV31PPORunnerCfg):
    """v33-S7：无白化 + raw GP + 396 + 同专家集 + **mjlab 判别器正则** + 判别器重学。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- 判别器输入口径：与 S3/S4/S5 逐字相同（防御性 pop，确保没有 normalizer 残留）----
        dk = dict(self.algorithm.discriminator_kwargs)
        dk.pop("normalizer_class_name", None)
        dk.pop("normalizer_kwargs", None)
        self.algorithm.discriminator_kwargs = dk
        self.algorithm.amp_grad_pen_space = "raw"
        self.algorithm.amp_normalizer_expert_only = True
        # ---- 396 维专家集（维度必须与环境侧的 anchor 重力通道一致）--------------
        self.algorithm.amp_expert_buffer_path = AMP_V33S_EXPERT_BUFFER
        # ---- 权重：**全不动**（coef 保持父链默认 1.0；λ = S5 的 0.75）------------
        self.algorithm.style_lerp = AMP_V33S7_STYLE_LERP
        self.algorithm.discriminator_reward_coef = 1.0
        # ---- ⭐ 本臂唯一变量：判别器正则（5 项全部向 mjlab 对齐）-----------------
        self.algorithm.discriminator_gradient_penalty_coef = AMP_V33S7_GP_COEF
        self.algorithm.discriminator_weight_decay_coef = AMP_V33S7_DISC_WD
        self.algorithm.discriminator_logit_weight_decay_coef = AMP_V33S7_DISC_LOGIT_WD
        self.algorithm.discriminator_optimizer_trunk_weight_decay = AMP_V33S7_OPT_WD_TRUNK
        self.algorithm.discriminator_optimizer_head_weight_decay = AMP_V33S7_OPT_WD_HEAD
        # ---- 判别器正则变了 ⇒ **必须重学**（actor/critic/优化器照常热启）----------
        self.ckpt_manipulator = DROP_DISC

        self.run_name = "v33s7_mjlabdisc_keepreward_fork13500"


__all__ = [
    "AMP_V33S7_DISC_LOGIT_WD",
    "AMP_V33S7_DISC_WD",
    "AMP_V33S7_GP_COEF",
    "AMP_V33S7_OPT_WD_HEAD",
    "AMP_V33S7_OPT_WD_TRUNK",
    "AMP_V33S7_STYLE_LERP",
    "G1AmpWalkV33S7PPORunnerCfg",
]
