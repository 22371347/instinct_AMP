"""V33-S2 的算法配置：**与 S0 逐字相同，只把白化强度换回 mjlab 原值**（`eps: 0.1 → 0.01`）。

## 为什么单独开一臂（用户 2026-10-06 决定）

`EmpiricalNormalization.eps` 是 σ 的下限，**同时**是"白化放大倍数"的上限 `1/eps`：

| 臂 | `eps` | 放大上限 | 依据 |
|---|---|---|---|
| **S0 / S1** | 0.1 | 10× | 本仓库 V23/V24/V26 三次实测：`eps=0.01`（100×）⇒ 系统性偏移被放大到 sigmoid 全饱和 ⇒ `style_std` 0.003/0.008/**0.0008**（判别器死）。V30 修掉真正的根因（专家批=单点 → 全数据集 i.i.d.）后，把白化重新拿回来试，但**保守**取 0.1 |
| **S2** | **0.01** | 100× | 严格对齐 mjlab。测"白化强度本身"是不是那个失效因子 |

⇒ **S2 − S0 = 只差一个 `eps`**（环境侧逐字继承 S0：396 通道 / 指令分布 / 奖励表 / 脚手架全不动）。
⇒ 若 S0（0.1）活着而 S2（0.01）死掉 ⇒ 坐实"V23/V24/V26 的死因是放大倍数，不是白化本身"；
   若两臂都活 ⇒ 说明 V30 的专家批修复确实把这条失效模式消掉了，可以放心用 mjlab 原值；
   若两臂都死 ⇒ 白化在本口径下就是不可用，S0 也应回退（那就要回 A-cold 的那条路）。

S2 排在 sweep **最后**跑，方便把训练过程中新冒出来的调整思路一并并进这一臂（用户原话）。

⚠️ 判据（与 S0 相同的"白化三联征"）：前 200 iters 内 `Step/amp_style_std` 不得塌到 < 0.03、
`Step/amp_style_zero_frac` 不得 > 0.5、`Loss/discriminator_loss` 不得掉到 < 1e-4。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v33s_ppo_cfg import G1AmpWalkV33SPPORunnerCfg

#: ⭐ 本臂的唯一变量：白化 σ 下限回到 mjlab 原值（放大上限 100×）
AMP_V33S2_NORMALIZER_EPS = 0.01
#: 统计量累积上限（`None` = 永不冻结，与 mjlab 一致）
AMP_V33S2_NORMALIZER_UNTIL = None


@configclass
class G1AmpWalkV33S2PPORunnerCfg(G1AmpWalkV33SPPORunnerCfg):
    """v33-S2：S0 的白化(eps=0.1) → **eps=0.01**，其余（含 396 专家集 / normalized GP / drop-disc）不动。"""

    def __post_init__(self):
        super().__post_init__()
        # 只换 `normalizer_kwargs`（`normalizer_class_name` / hidden_sizes / nonlinearity 保持 S0 的值）
        self.algorithm.discriminator_kwargs = {
            **self.algorithm.discriminator_kwargs,
            "normalizer_kwargs": {
                "eps": AMP_V33S2_NORMALIZER_EPS,
                "until": AMP_V33S2_NORMALIZER_UNTIL,
            },
        }
        self.run_name = "v33s2_white001_normgp_grav396_side015_fork11500"


__all__ = [
    "AMP_V33S2_NORMALIZER_EPS",
    "AMP_V33S2_NORMALIZER_UNTIL",
    "G1AmpWalkV33S2PPORunnerCfg",
]
