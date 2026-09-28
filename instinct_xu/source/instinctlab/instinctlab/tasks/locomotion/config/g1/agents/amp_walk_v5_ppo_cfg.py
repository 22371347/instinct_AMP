"""AMP 行走教师 **v5** 的 PPO runner 配置 —— 只把风格奖励的**函数形式**换掉。

v5 相对 v4 只改 **1 个字段**：`discriminator_reward_type: "quad" → "log"`。
镜像损失 / 站立关风格 / 奖励集 / 观测 / σ / 判别器超参**全部不动**。

为什么要改（实测证据）
----------------------
`wasabi.py` 的 `quad` 形式把判别器的**原始线性输出** $d$ 映射成奖励：

$$r_{\\text{quad}} = \\mathrm{clamp}\\!\\left(1 - \\tfrac{(d-1)^2}{4},\\ 0\\right)$$

而我们的判别器用 `MSELoss` + 目标 $\\pm 1$，所以 $d$ **被强行钉在 $\\pm 1$**。
v3c/v4 两条 run 的实测曲线（`instinct/tools/cloud_amp_diag.sh` 反解）：

| iter | $r_{amp}$ | 反解 $d$ | $\\mathcal{L}_{disc}$ |
|---|---|---|---|
| 0 | 0.763 | +0.03 | 0.595 |
| 500 | 0.122 | **−0.875** | 0.033 |
| 2500 | 0.070 | −0.929 | 0.017 |
| 7000 | 0.093 | −0.905 | 0.015 |

⇒ **判别器在第 500 轮就把二分类做完了，之后 6500 轮一直停在 $d\\approx-0.91$。**

两个后果：

1. **量级塌陷**：同一个 $d=-0.90$，`quad` 给 **0.093**，`log` 给
   $-\\log(1-\\sigma(d)) = \\mathbf{0.341}$ —— 相差 **3.7 倍**。
   ⭐ 0.34 正好落在 AMP_mjlab 的 0.3~0.4 区间。
   我们的源码自己就写着警告：`"Copied from WASABI, not sure if this is correct."`
   `"This assumes disc is the output of a direct linear layer."`
   —— 它是给**无界**判别器输出设计的，配 `MSELoss(±1)` 恰好是最坏组合。
2. ⭐ **死区**：$d \\le -1$ 时 `quad` 的 $r \\equiv 0$ **且
   $\\partial r/\\partial d = -0.5(d-1) = 0$** ⇒ 这些样本**既不进奖励、也不给梯度**。
   均值 $d=-0.91$ 意味着**相当大一部分样本（粗估 25~40%）落在死区**，
   这正是"无论移动还是静止，风格值都差不多"的直接来源：
   它不再是"哪个状态更像人"的**排序信号**，而是一个钉死的常数。
   `log` 的梯度是 $\\partial r/\\partial d = \\sigma(d) = 0.29$，**处处非零**。

⚠️ **刻意不改 `discriminator_reward_coef`（保持 0.1）**：这样"奖励形式"是唯一变量。
   但要清楚它的**后果**：风格项从 $0.025\\times0.093 = 0.0023$/步 变成
   $0.025\\times0.341 = 0.0085$/步（≈3.7 倍），实测 AMP 占净回报量级会从 **29% → ~70%**。
   若希望"只改形状、不改力度"，把 `discriminator_reward_coef` 同步降到 **0.027** 即可
   （一行，见 `DISC_REWARD_COEF_KEEP_MAGNITUDE_V5`）。

⚠️ **判别器训练逻辑一个字没动**：判别器的损失函数、优化器、输入、正则全部继承 v4
   ⇒ 这属于**奖励构成**改动，所以可以 `--resume` 续训（不需要重训）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v4_ppo_cfg import G1AmpWalkV4PPORunnerCfg

#: v5 的奖励形式：标准 AMP 的 `-log(1 - sigmoid(d))`
V5_DISCRIMINATOR_REWARD_TYPE = "log"
#: 若只想改"形状"而不改"总力度"，把 `discriminator_reward_coef` 设成这个值。
#: 推导：log 在 $d\\approx-0.91$ 处给 0.341、quad 给 0.093 ⇒ $0.1\\times0.093/0.341 \\approx 0.027$。
#: **默认不启用**（保持 0.1，让"奖励形式"成为唯一变量）。
DISC_REWARD_COEF_KEEP_MAGNITUDE_V5 = 0.027


@configclass
class G1AmpWalkV5PPORunnerCfg(G1AmpWalkV4PPORunnerCfg):
    """v5：v4 全部配置 + 风格奖励形式 `quad` → `log`。"""

    def __post_init__(self):
        super().__post_init__()

        # --- 唯一改动：风格奖励的函数形式 ---------------------------------
        # `wasabi.py::WasabiAlgoMixin.compute_auxiliary_reward` 里：
        #   quad → clamp(1 - (d-1)^2/4, 0)          ← 饱和后塌到 0，且有梯度死区
        #   log  → -log(1 - sigmoid(d))             ← 饱和后仍留 ~0.31，梯度处处非零
        self.algorithm.discriminator_reward_type = V5_DISCRIMINATOR_REWARD_TYPE

        # ⚠️ 判别器自身超参**一个都没动**（`discriminator_loss_func` /
        #    `discriminator_reward_coef` / 优化器 / 正则 / `MSELoss` 目标 ±1）。
        #    所以这是"奖励构成"改动，`--resume` 续训即可。

        self.run_name = "v5_logamp_mirror0.1_nostyleonstand"


__all__ = [
    "DISC_REWARD_COEF_KEEP_MAGNITUDE_V5",
    "V5_DISCRIMINATOR_REWARD_TYPE",
    "G1AmpWalkV5PPORunnerCfg",
]
