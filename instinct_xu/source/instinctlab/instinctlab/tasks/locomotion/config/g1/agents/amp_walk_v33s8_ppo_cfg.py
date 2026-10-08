"""V33-S8（方案 S8-a）：**抬风格分量 + 自适应 LR 配对**。

## 为什么要"配对"（而不是像 S6/V28 那样只抬分量）

来自本轮的结构分析（`instinct/docs/v33_tracking_gait_analysis.md` §5.5 / §5.7 / §5.8）：

1. **实测风格项只占奖励的 2.2%**（风格 `(1−λ)·coef·style` = 0.25×1.0×0.25 = 0.063/步；
   任务加权和 ≈ 2.8/步）⇒ 风格项**没有能力**参与步态塑形（λ 在 0.70↔0.75 之间调几乎空转）。
2. **但单独抬分量必崩**（两次先例，机制一致）：
   * **S6**：S4 配置 + `coef=4` ⇒ **150 轮悬崖式崩溃**（`action_abs_mean` 1.5→2.76、`ep_len` → 1.9），
     被守护判据②止损；
   * **V28**：`coef=4`（当时还叠加"奖励形状无上界"）⇒ 10470 轮渐变崩溃。
   共同机制：**风格奖励是"非平稳 + 对抗性"信号**（判别器同时在适应策略）⇒ 占大份额后
   优势函数被它的噪声主导，而 `schedule="fixed"` **无法自动降 LR** ⇒ 一个坏批次即毁。
3. **mjlab 用的正是 `schedule="adaptive"` + `desired_kl=0.01`**（`instinct_rl/instinct_rl/algorithms/ppo.py:236-253`
   实现已存在，只是我们从未启用 —— agent.yaml 里 `schedule: fixed`，`desired_kl` 形同虚设）。
4. **S7 的负结果**（只把判别器正则回归 mjlab）确认："判别器工作点"不是根因（均衡点没变），
   且用户回放判定 S7 的类人风格**恶化** ⇒ 本臂不从 S7 续。

## 本臂两个变量（其余全部沿用 S5 / S7）

| # | 项 | S7（上一臂） | **S8** |
|---|---|---|---|
| ① | `discriminator_reward_coef` | 1.0 | **4.0**（风格占比 2.2% → **10.6%**） |
| ② | `algorithm.schedule` | `fixed` | **`adaptive`**（`desired_kl = 0.01`） |

* 判别器正则：**沿用 S7 的 mjlab 值**（GP 5.0 / loss 级 wd 3e-4 / logit_wd 0.04 / trunk=head=3e-4）
  —— S7 的负结果只说明"它不改变均衡点"，没有证据说它有害。
* 奖励表 = **S5**（`air_time=1.0` / `track_lin=3.0` / `track_ang=1.5` / `flat=−4` / `arms=0`）；
  `style_lerp = 0.75`（不动）。
* AMP 口径不变（396 / 无白化 / raw GP / 同专家集）⇒ **判别器保留续训**（`ckpt_manipulator=None`），
  避免冷启动扰动（S8 要检验的是"分量 + 稳定性"，不是判别器重学）。

## 起点

**fork = S5 的 `model_13500.pt`**（用户判定 S7 风格恶化 ⇒ 回到已知最好的一版：
`ep_len` 689、倾角 15°、`vxy_rmse` 0.324、角速度跟踪最好）。

## 预登记判据

1. **稳定性（本臂的首要目标）**：**不得出现 S6 那种悬崖** ——
   `Mean episode length` 不跌破 300、`action_abs_mean ≤ 0.7`、`Mean reward` 不跌破 20；
   同时观察 `Train/learning_rate`（自适应机制应把它压下去）与 KL。
2. **风格是否真的开始塑形**（行为以用户 MuJoCo 回放为准）：
   `amp_style_reward` 分布拉开、`amp_style_std` 上升且稳定、`disc_grad` 通道分布变化、
   `air_time_single_stance` / `in_band_frac` / `tilt` / `foot_sep` 的趋势。
3. **不退化闸门**：`tracking/vxy_rmse ≤ 0.40`（S5 是 0.324）、`ep_len ≥ 500`。
4. 若 ① 仍崩 ⇒ 说明"抬分量"这条路在当前 PPO 配置下不可行 ⇒ 转
   **任务条件化判别器 / 镜像增广 / 相位时钟**等结构改动（而不是继续调权重）。
5. 若 ① 稳但风格无变化 ⇒ 说明 10.6% 仍不够 ⇒ 下次把 `coef` 提到 8 或把任务项整体降权。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v31_ppo_cfg import G1AmpWalkV31PPORunnerCfg
from .amp_walk_v33s_ppo_cfg import AMP_V33S_EXPERT_BUFFER

#: ⭐ 变量①：风格项权重 ×4（有效风格权重 (1−λ)·coef：0.25 → **1.0**）
AMP_V33S8_DISC_REWARD_COEF = 4.0
#: 风格占比（不动）= S5 的 0.75
AMP_V33S8_STYLE_LERP = 0.75
#: ⭐ 变量②：自适应 LR（mjlab 口径）
AMP_V33S8_SCHEDULE = "adaptive"
AMP_V33S8_DESIRED_KL = 0.01
#: 判别器正则：沿用 S7（= mjlab `AmpAlgoCfg` 的值）
AMP_V33S8_GP_COEF = 5.0
AMP_V33S8_DISC_WD = 3.0e-4
AMP_V33S8_DISC_LOGIT_WD = 0.04
AMP_V33S8_OPT_WD_TRUNK = 3.0e-4
AMP_V33S8_OPT_WD_HEAD = 3.0e-4


@configclass
class G1AmpWalkV33S8PPORunnerCfg(G1AmpWalkV31PPORunnerCfg):
    """v33-S8：S7 的判别器口径 + **coef 4.0** + **adaptive LR** + 判别器保留续训。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- 判别器输入口径：与 S3~S7 逐字相同（防御性 pop，确保没有 normalizer 残留）----
        dk = dict(self.algorithm.discriminator_kwargs)
        dk.pop("normalizer_class_name", None)
        dk.pop("normalizer_kwargs", None)
        self.algorithm.discriminator_kwargs = dk
        self.algorithm.amp_grad_pen_space = "raw"
        self.algorithm.amp_normalizer_expert_only = True
        self.algorithm.amp_expert_buffer_path = AMP_V33S_EXPERT_BUFFER
        # ---- 风格占比：不动（S5 的 0.75）------------------------------------------
        self.algorithm.style_lerp = AMP_V33S8_STYLE_LERP
        # ---- ⭐ 变量①：风格分量 ×4（风格占比 2.2% → 10.6%）------------------------
        self.algorithm.discriminator_reward_coef = AMP_V33S8_DISC_REWARD_COEF
        # ---- ⭐ 变量②：自适应 LR（与变量①**配对**，对抗"抬分量必崩"的机制）--------
        self.algorithm.schedule = AMP_V33S8_SCHEDULE
        self.algorithm.desired_kl = AMP_V33S8_DESIRED_KL
        # ---- 判别器正则：沿用 S7（mjlab 值；S7 证明它不改变判别器均衡点）----------
        self.algorithm.discriminator_gradient_penalty_coef = AMP_V33S8_GP_COEF
        self.algorithm.discriminator_weight_decay_coef = AMP_V33S8_DISC_WD
        self.algorithm.discriminator_logit_weight_decay_coef = AMP_V33S8_DISC_LOGIT_WD
        self.algorithm.discriminator_optimizer_trunk_weight_decay = AMP_V33S8_OPT_WD_TRUNK
        self.algorithm.discriminator_optimizer_head_weight_decay = AMP_V33S8_OPT_WD_HEAD
        # ---- ⭐ 判别器**保留**：AMP 口径没变（变的是奖励分量与 PPO 调度）-----------
        self.ckpt_manipulator = None

        self.run_name = "v33s8_coef4_adaptive_keepdisc_from13500"


__all__ = [
    "AMP_V33S8_DESIRED_KL",
    "AMP_V33S8_DISC_LOGIT_WD",
    "AMP_V33S8_DISC_REWARD_COEF",
    "AMP_V33S8_DISC_WD",
    "AMP_V33S8_GP_COEF",
    "AMP_V33S8_OPT_WD_HEAD",
    "AMP_V33S8_OPT_WD_TRUNK",
    "AMP_V33S8_SCHEDULE",
    "AMP_V33S8_STYLE_LERP",
    "G1AmpWalkV33S8PPORunnerCfg",
]
