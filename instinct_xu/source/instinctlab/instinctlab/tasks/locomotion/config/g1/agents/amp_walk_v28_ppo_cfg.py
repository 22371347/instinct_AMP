"""V28 的算法配置：**恢复判别器正则（回到 V10b 健康期）+ 放大 AMP 风格项权重**。

## 为什么是这两件（本轮核查得到的事实）

1. **风格项当前权重 = 0.25**：`r = λ·r_task + (1−λ)·coef·style`，λ = `style_lerp` = 0.75、
   `coef` = `discriminator_reward_coef` = **1.0**（V15 就把它从 0.25 提到 1.0 了，
   不是基类默认的 0.1）。
2. **风格奖励的跨状态方差只有 0.058**（V27 实测 `amp_style_std`）
   ⇒ 风格项对策略的作用量 ≈ 0.25 × 0.058 = **0.0145/步**，比任务项（O(1)）小 **10~70 倍**
   ⇒ **判别器再准也推不动步态**。
3. **判别器的抗饱和正则是在 V19a 被拆掉的**（那一轮的动机是“让 `total_loss` 可读”）：
   `GP 5.0→1.0`、`weight_decay 3.0e-4→1.0e-5`。
   而**正则全在的 V10b 期是历史最佳**：`amp_style_std = 0.166`（V27 是 0.058）。
4. `discriminator_logit_weight_decay_coef` **不动**（保持 0.0）：该键有**键名 bug**
   ——`wasabi.py` 把 `losses["logit_weight_decay"]` 的系数取成
   `getattr(self, "logit_weight_decay_coef", 1.0)`（**不存在 ⇒ 默认 1.0**），
   所以一旦把 gate 打开，`Σw_last² ≈ 0.33` 会以**系数 1.0** 进入总损失（比判别器损失大 2~300×）
   ⇒ 会把最后一层压到 0 ⇒ `d≈0` ⇒ `quad≈0.875` ⇒ **风格奖励又变成常数**。
   ⇒ 走“先修 bug 再启用”的路（留到 V29），本轮不碰。

## 本轮变量（3 个数字 + 1 个机械动作）

| 项 | V27 | **V28** | 理由 |
|---|---|---|---|
| `discriminator_gradient_penalty_coef` | 1.0 | **5.0** | 回到 V10b（正则被拆前的健康期）|
| `discriminator_weight_decay_coef` | 1.0e-5 | **3.0e-4** | 同上 |
| `discriminator_reward_coef` | 1.0 | **4.0** | 风格项权重 0.25 → **1.0**（与任务项同量级）|
| `ckpt_manipulator` | drop disc（继承）| drop disc | 换系数后旧判别器无意义，从零重学 |

**不动**：`style_lerp`（0.75 ⇒ 任务项权重不被削弱）、去白化、σ=0.20 + 300 轮退火、
奖励表、AMP 观测（13body × 8frame）、`action_scale=1.0`。

## 这是一次“判别式”实验（结果直接分叉下一步）

| 观测 | 判定 | 下一步 |
|---|---|---|
| `amp_style_std` 明显 >0.058 **且** 步态向参考靠（`single_stance`↑、`hop_frac`↓）| 风格信号**有信息**，之前只是被权重压住 | V29：只调系数（coef 回 2.0 微调）|
| `amp_style_std` 仍 ≈0.05 或判别器照样贴 ±1，步态不动 | 判别器仍是常数输出 ⇒ 风格项**无信息** | V29：结构对齐（专家批解耦 / 判别器输入口径 / 修 logit_wd bug）|
| 步态变差（`Mean reward` <+20 / `hf_acc` 抬头）| 风格项在把策略带向噪声 | 立刻回退 coef 到 2.0 或 1.0 |

⚠️ 注意：`amp_style_std` 随 σ 变化（σ=0.57 时被判据噪声垫到 0.09~0.17；σ=0.20 时 0.05~0.06），
**只与同为 σ=0.20 的 V27 比较**。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v27_ppo_cfg import DROP_DISC, G1AmpWalkV27PPORunnerCfg

#: 风格项系数。有效风格压力 = (1 − style_lerp)·coef = (1−0.75)×4.0 = **1.0**（V27 是 0.25）。
#: 若行为侧出现退化（`hf_acc` 抬头 / `Mean reward` 掉到 <+20），回退到 2.0（权重 0.5）。
AMP_COEF_V28 = 4.0


#: 判别器正则（恢复 V10b：GP 5.0 / weight_decay 3.0e-4）
GP_COEF_V28 = 5.0
DISC_WD_COEF_V28 = 3.0e-4


@configclass
class G1AmpWalkV28PPORunnerCfg(G1AmpWalkV27PPORunnerCfg):
    """v28：V27（去白化 + σ=0.20 + 判别器重置）+ 判别器正则恢复（GP 5 / wd 3e-4）+ 风格项 ×4。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- ① 判别器抗饱和正则：回到 V19a 拆掉之前的 V10b 值 ----------------
        self.algorithm.discriminator_gradient_penalty_coef = GP_COEF_V28
        self.algorithm.discriminator_weight_decay_coef = DISC_WD_COEF_V28
        # `discriminator_logit_weight_decay_coef` 保持 0.0（键名 bug，见模块 docstring）
        # --------------------------------------------------------------------

        # ---- ② 风格项权重 ×4（0.25 → 1.0）-----------------------------------
        self.algorithm.discriminator_reward_coef = AMP_COEF_V28
        # --------------------------------------------------------------------

        # ---- 机械动作（继承自 V27，显式写出便于单看这份配置就能读全）----------
        self.ckpt_manipulator = DROP_DISC
        # --------------------------------------------------------------------

        self.run_name = "v28_ampcoef4_gp5_wd3e-4_dropdisc_4096"


__all__ = [
    "AMP_COEF_V28",
    "DISC_WD_COEF_V28",
    "DROP_DISC",
    "GP_COEF_V28",
    "G1AmpWalkV28PPORunnerCfg",
]
