"""V19c 的算法配置：**与 V19b 逐字相同**（AMP 侧一个参数都不动）。

原因：本轮的全部变量都在 env 侧（运动门控 / `action_rate` 降权 / 站立姿态加强），
     所以 AMP 口径必须保持原样才能归因：
       `amp_quad_k = 0.125`、`gradient_penalty_coef = 1.0`、`weight_decay = 1e-5`、
       `logit_weight_decay = 0.0`、`amp_reward_eps = 0.05`、`log_quad` + `wasserstein`、
       λ = 0.75、c = 1.0。

起点：**热启动 V19b 的最新 checkpoint，并保留判别器**
      ⇒ `ckpt_manipulator = None`（判别器输入仍是 670 维、`k` 仍是 0.125 ⇒ 直载兼容）。
      为什么不丢判别器：V19b 的判别器已经**几乎完全分开**（`discriminator_loss ≈ 0.0055`、
      `d ≈ −0.986 / d_ref ≈ +0.985`），丢掉它等于把这份分辨力重新学一遍，而本轮
      根本没改任何影响判别器的东西。

监测（本轮主指标）
------------------
* ⭐ `Step/air_time_motion_gate_off_frac`（**新探针**）：被"实测未移动"门控关掉的比例。
  它应 > 0（说明门控在工作）；若长期 ≈ 0 说明阈值太松。
* ⭐ 站立行为：`Episode_Reward/track_default_pose_on_stand` 应更负（罚得更多、姿态更回位），
  同时 `Step/amp_stand_frac`（被门控判为站立的 env 比例）不应异常变化。
* `Episode_Reward/action_rate_l2`：权重减半后**数值应明显变小**（−1.5 → 约 −0.9/步）；
  要看的原始量是 `Episode_Reward/action_rate_l2 / 0.06` —— 若它继续爬过 ~18–20 而步态不涨，
  说明抖动回来了 ⇒ 把权重调回 −0.10 或改用"只罚高频分量"的形式。
* 步态本身不能退步：`Step/air_time_in_band_frac` / `air_time_hop_frac` /
  `Episode_Reward/feet_air_time`（运动门控会**压低**这一项，因为部分步被关掉了
  ⇒ 要同时看 `in_band_frac` 而不是只看它的绝对值）。
* 守住 `Episode_Reward/track_lin_vel_xy_exp ≥ 1.0`、`Step/tilt_deg_mean ≤ 12°`。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v18_ppo_cfg import (
    DISC_REWARD_COEF_V18,
    STYLE_LERP_V18,
)
from .amp_walk_v19a_ppo_cfg import (
    AMP_REWARD_EPS_V19A,
    GP_COEF_V19A,
    LOGIT_WD_COEF_V19A,
    WD_COEF_V19A,
)
from .amp_walk_v19b_ppo_cfg import QUAD_K_V19B, G1AmpWalkV19BPPORunnerCfg


@configclass
class G1AmpWalkV19CPPORunnerCfg(G1AmpWalkV19BPPORunnerCfg):
    """v19c：AMP 口径与 V19b 完全相同；只换 env 侧奖励结构，热启动 V19b。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- AMP 侧：全部重申为 V19b 的值（= 本轮不动）---------------------------
        self.algorithm.amp_quad_k = QUAD_K_V19B                       # 0.125
        self.algorithm.discriminator_gradient_penalty_coef = GP_COEF_V19A        # 1.0
        self.algorithm.discriminator_weight_decay_coef = WD_COEF_V19A            # 1e-5
        self.algorithm.discriminator_logit_weight_decay_coef = LOGIT_WD_COEF_V19A  # 0.0
        self.algorithm.amp_reward_eps = AMP_REWARD_EPS_V19A           # 0.05
        self.algorithm.style_lerp = STYLE_LERP_V18                    # 0.75
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V18  # 1.0

        # ---- 起点：热启动 V19b（**保留判别器**）---------------------------------
        self.ckpt_manipulator = None

        self.run_name = f"v19c_warmV19b_motiongate_ar0.06_standpose0.5_quadk{QUAD_K_V19B:g}"


__all__ = ["G1AmpWalkV19CPPORunnerCfg"]
