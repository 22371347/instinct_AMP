"""V33 A-cold 的算法配置：**旧 AMP 口径 + 冷启动判别器**（S0 的参照系）。

## 为什么需要这一臂

我们把 fork 点（V31 `model_11500.pt`）的判别器**永久丢掉**了（S 臂改了输入维度 ⇒ 旧判别器
物理上不可复用）。于是"旧口径"这条路再也没有连续的历史曲线可比 —— 想要知道
"白化 + 重力通道"到底带来了什么，必须有一个**同起点、同指令分布、同预算**的旧口径对照。

⇒ A-cold = 与 S0 **逐字相同**，只把三件事退回旧口径：

| # | 项 | **S0** | **A-cold** |
|---|---|---|---|
| ① | 判别器输入白化 | `EmpiricalNormalization(eps=0.1)` | **无**（不接 normalizer）|
| ② | GP 作用空间 | `normalized` | **`raw`**（与 ① 配对）|
| ③ | AMP 状态 | 396（含 anchor 投影重力）| **390**（旧口径）|
| ④ | 专家数据集 | `/root/amp_expert_states_396.pt` | **`/root/amp_expert_states.pt`**（V30 那份，继续用）|

**共有**（不是变量）：fork = V31 `model_11500.pt`、actor/critic 热启动、**判别器从零重学**
（`drop_amp_discriminator`）、指令分布（`side_vel_min=0.15`）、`flat_orientation_l2=−4`、
GP 系数 10.0 / expert-only、优化器分组 wd、`style_lerp=0.75`、`bounded_quad`、σ=0.20+退火。

⇒ 所以 **S0 − A-cold = "① ② ③ 三件结构改动之和"**（不能拆单项，这是用户确认过的取舍：
优先结构级改动、避免反复调参）。

## 判据

A-cold 的作用是**参照系**，不是"要赢"：它给出"在新指令分布 + 判别器重学"条件下，
旧口径能达到的 `amp_style_std` / `style_reward` / `amp_d` / `grav_bx_mean` /
`reward` / `ep_len` 基线。S0 与 S1 的所有比较都对着它做。

不退化闸门（与 S0 相同）：`Mean reward ≥ 30`、`episode length ≥ 300`、`action_abs_mean ≤ 0.6`。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v27_ppo_cfg import DROP_DISC
from .amp_walk_v30_ppo_cfg import EXPERT_BUFFER_V30
from .amp_walk_v31_ppo_cfg import G1AmpWalkV31PPORunnerCfg


@configclass
class G1AmpWalkV33AColdPPORunnerCfg(G1AmpWalkV31PPORunnerCfg):
    """v33 A-cold：旧 AMP 口径（390 / 无白化 / raw GP）+ 判别器冷启动。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- ① 明确"没有白化"（防御性 pop：即使父链哪天带上了 normalizer 也退回旧口径）----
        dk = dict(self.algorithm.discriminator_kwargs)
        dk.pop("normalizer_class_name", None)
        dk.pop("normalizer_kwargs", None)
        self.algorithm.discriminator_kwargs = dk
        # ---- ② GP 回到原始空间（无白化时必须与 raw 配对）-----------------------
        self.algorithm.amp_grad_pen_space = "raw"
        # ---- 无 normalizer ⇒ 该 flag 实际不起作用（`forward` 按 `normalizer is None` 短路）----
        self.algorithm.amp_normalizer_expert_only = True
        # ---- ③ 390 维专家数据集（V30 那份，**不需要**重导）---------------------
        self.algorithm.amp_expert_buffer_path = EXPERT_BUFFER_V30
        # ---- 机械动作：判别器（与其优化器）从零重学 -----------------------------
        self.ckpt_manipulator = DROP_DISC

        self.run_name = "v33acold_nodisc_nowhite_rawgp_390_side015_fork11500"


__all__ = ["G1AmpWalkV33AColdPPORunnerCfg"]
