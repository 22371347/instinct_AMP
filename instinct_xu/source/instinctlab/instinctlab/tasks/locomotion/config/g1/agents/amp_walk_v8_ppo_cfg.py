"""AMP 行走教师 **v8** 的 PPO runner 配置 —— 风格奖励换成 LLP 论文形式。

v8 相对 v7 只改 **风格奖励的函数形式**（用户指定，参照原 LLP 论文）：

    r_amp = -log(1 - D),      D = quad(d) = clamp(1 - (d-1)^2 / 4, 0)

其中 `d` 是判别器的**原始线性输出**（`MSELoss` + 目标 ±1）。
它在数学上有闭式解：

    1 - D = (d-1)^2 / 4   ⇒   r = -log(1/4) - 2*ln|d-1| = **1.3863 - 2*ln|d-1|**

| d | D = quad | **r = -log(1-D)** | （对比 v5-v7 的 `log` 形式）|
|---|---|---|---|
| +0.03（iter0）| 0.763 | **1.44** | 0.68 |
| 0 | 0.750 | 1.386 | 0.693 |
| -0.5 | 0.438 | 0.575 | 0.475 |
| -0.9（饱和）| 0.098 | **0.103** | 0.341 |
| -1.0 | 0 | **0** | 0.313 |
| **= +1.0** | **1.0** | **+inf ⚠️** | 0.693 |

⚠️⚠️ **必须夹住 $D$**：$d=1$ 时 $D$ 恰好 = 1 ⇒ $-\log 0 = +\infty$，
而 `MSELoss` 的**参考侧目标就是 +1**，不是罕见事件。
实现里夹到 `1-1e-6` ⇒ 奖励上界 `-log(1e-6) ≈ 13.8`，有限且不产生 NaN。
（见 `instinctlab/utils/wrappers/instinct_rl/algorithms.py` 的 `amp_reward_form`。）

✅ 用它的理由（比 `log` 形式更好的一点）：
$\partial r/\partial d = -2/(d-1)$，在饱和区 $d\approx-0.9$ 处是 **+2.0**，
而 `log` 形式只有 $\sigma(d)=0.29$ ⇒ **梯度强约 7 倍且不消失**。

⚠️ 但**它并不解决"奖励量级塌到 0"**：$d\le-1$ 时 $r$ 仍 →0，$d=-0.9$ 时只有 0.103
（≈ `quad` 的 0.093，比 `log` 的 0.341 低 **3.3 倍**）。
⇒ **风格项的实际压力会再降 ~3.3 倍**（叠加 v7 已把系数减半）。
若想保持与 v7 相当的风格压力，把 `discriminator_reward_coef` 从 0.05 提到
**0.166**（= 0.05 × 0.341/0.103）。**本版默认不改**（保持 0.05），留给你按曲线决定。

⚠️ 实现方式：**不改上游 `instinct_rl`**。父类用 `discriminator_reward_type="wasserstein"`
（那是它唯一**原样返回 d** 的分支），塑形在我们自己的桥接层做。
判别器自身的损失/优化器/输入**一律没动** ⇒ 本改动属于**奖励构成** ⇒ **可续训**。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v7_ppo_cfg import DISC_REWARD_COEF_V7, G1AmpWalkV7PPORunnerCfg

#: 保持 v7 的 AMP 系数（0.05）。若要补偿新形式的量级下降，改成 0.166。
DISC_REWARD_COEF_V8 = DISC_REWARD_COEF_V7
#: 若想维持与 v7 等效的风格压力（见模块 docstring 的推导）
DISC_REWARD_COEF_V8_STYLE_MATCHED = 0.166


@configclass
class G1AmpWalkV8PPORunnerCfg(G1AmpWalkV7PPORunnerCfg):
    """v8：v7 全部配置 + 风格奖励换成 LLP 论文形式 `r = -log(1 - quad(d))`。"""

    def __post_init__(self):
        super().__post_init__()

        # --- 唯一实质改动：风格奖励的函数形式 -----------------------------
        # ① 让父类原样返回判别器原始输出 d（`elif ... == "wasserstein": reward = disc`）
        self.algorithm.discriminator_reward_type = "wasserstein"
        # ② 在我们自己的桥接层把 d 塑形成 -log(1 - quad(d))
        self.algorithm.amp_reward_form = "log_quad"
        self.algorithm.amp_reward_eps = 1.0e-6

        # ⚠️ 判别器自身的损失/优化器/输入**一个字都没改**（仍是 MSELoss + 目标 ±1）
        #    ⇒ 这属于**奖励构成**改动 ⇒ 可以 `--resume` 续训。

        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V8
        self.run_name = f"v8_logquad_ampc{DISC_REWARD_COEF_V8}"


__all__ = [
    "DISC_REWARD_COEF_V8",
    "DISC_REWARD_COEF_V8_STYLE_MATCHED",
    "G1AmpWalkV8PPORunnerCfg",
]
