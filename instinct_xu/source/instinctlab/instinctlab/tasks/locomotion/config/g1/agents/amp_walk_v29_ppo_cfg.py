"""V29 的算法配置：**判别器向 AMP_mjlab 对齐**（系数逐字回退到 V27）。

## 改动一览（细节与理由见 `amp_walk_v29_env_cfg.py` 的模块 docstring）

| # | 项 | V27 | V29 |
|---|---|---|---|
| ④ | `discriminator_gradient_penalty_coef` | 1.0 | **1.0（回退 V28 的 5.0）** |
| ④ | `discriminator_weight_decay_coef` | 1.0e-5 | **1.0e-5（回退 V28 的 3.0e-4）** |
| ④ | `discriminator_reward_coef` | 1.0 | **1.0（回退 V28 的 4.0）** |
| ③ | `amp_reward_form` | `"log_quad"`（无上界）| **`"bounded_quad"`**（mjlab 形状，∈[0,1]）|
| ③ | `amp_quad_k` | 0.125 | **0.25**（mjlab 的曲率；零点 = `d ≤ −1`）|
| ② | `amp_expert_independent_sampling` | 不存在（同相位）| **`True`**（参考批独立随机重采 `(T,B)`）|
| ① | AMP 观测历史帧 | 8（1560）| **2（390）** ← 在 env cfg 里设 |
| — | `ckpt_manipulator` | drop disc | drop disc（判别器输入维度变了 ⇒ **必须**重学）|

**有效风格权重 (1−0.75)×1.0 = 0.25 与 V27 完全相同** —— 本轮**刻意不动权重**（V15 证明"抬权重"无效，
V28 证明"抬权重"有害）。

## 判据速查

* ① 生效：启动日志 `Linear(in_features=390)`（V27 是 1560）。
* ② 生效：`amp_d` 不再在 ~20 轮内贴到 ±1（判别器失去了"相位/时序"这条捷径）。
* ③ 生效：`amp_style_reward ∈ [0,1]`（V27 是 [0,3] 且饱和时恒 0.693）；
  `amp_style_zero_frac` 是"`d ≤ −1` 死区占比"，长期 1.0 = 风格项死掉。
* **防退化闸门**：`Mean reward ≥ 30`、`Mean episode length ≥ 300`、`action_abs_mean ≤ 0.6`、
  `hf_acc_all ≈44`；wrapper 里有"`episode length < 50` 连续 20 轮 ⇒ 自动停训"的守护。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v27_ppo_cfg import DROP_DISC, G1AmpWalkV27PPORunnerCfg

# ---- ④ 系数：把 V28 的三项改动全部回滚到 V27（= V19A/V18 的历史值）------------
GP_COEF_V29 = 1.0
DISC_WD_COEF_V29 = 1.0e-5
AMP_COEF_V29 = 1.0
# ---- ③ 风格奖励形状：对齐 AMP_mjlab（有界；`d ≤ −1` 时恰为 0）----------------
AMP_REWARD_FORM_V29 = "bounded_quad"
AMP_QUAD_K_V29 = 0.25
# ---- ② 数据管线：参考批与策略批解耦（独立随机重采 (T,B)）--------------------
EXPERT_INDEP_V29 = True


@configclass
class G1AmpWalkV29PPORunnerCfg(G1AmpWalkV27PPORunnerCfg):
    """v29：V27（去白化 + σ=0.20 + 判别器重置）+ mjlab 对齐（输入 2 帧 / 专家批解耦 / 有界奖励形状）。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- ④ 回退 V28 的判别器系数（显式写回，便于单看这份配置就能读全）----
        self.algorithm.discriminator_gradient_penalty_coef = GP_COEF_V29
        self.algorithm.discriminator_weight_decay_coef = DISC_WD_COEF_V29
        self.algorithm.discriminator_reward_coef = AMP_COEF_V29
        # `discriminator_logit_weight_decay_coef` 保持 0.0（键名 bug，启用前必须先修）

        # ---- ③ 风格奖励形状：mjlab 的 `clamp(1 − 0.25(d−1)², 0, 1)` ----------
        self.algorithm.amp_reward_form = AMP_REWARD_FORM_V29
        self.algorithm.amp_quad_k = AMP_QUAD_K_V29
        # 形状是 `bounded_quad` ⇒ 仍需要原始 d ⇒ `discriminator_reward_type` 必须是
        # "wasserstein"（继承自 V27 的链，verifier 里会断言）。

        # ---- ② 专家批解耦（在 `init_storage` 里接到 amp storage 上）----------
        self.algorithm.amp_expert_independent_sampling = EXPERT_INDEP_V29

        # ---- 机械动作：判别器输入维度变了（1560→390）⇒ 必须从零重学 ----------
        self.ckpt_manipulator = DROP_DISC

        self.run_name = "v29_mjlabalign_amp2frame_indepref_boundedquad_dropdisc_4096"


__all__ = [
    "AMP_COEF_V29",
    "AMP_QUAD_K_V29",
    "AMP_REWARD_FORM_V29",
    "DISC_WD_COEF_V29",
    "DROP_DISC",
    "EXPERT_INDEP_V29",
    "GP_COEF_V29",
    "G1AmpWalkV29PPORunnerCfg",
]
