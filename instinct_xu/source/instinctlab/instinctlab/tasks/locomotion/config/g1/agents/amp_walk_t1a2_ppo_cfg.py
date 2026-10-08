"""**T1a2 的 PPO runner**：T1a（= S8 判别器口径 + 风格归零 + 镜像损失）之上，
把**探索与网络**两项调回 T0 的口径。

============================================================================
T1a2 相对 T1a 的改动（全部是"回到 T0"，不引入新设计）
============================================================================
| 项 | T1a | **T1a2（= T0）** | 为什么 |
|---|---|---|---|
| `policy.init_noise_std` | 0.35（±20°） | **1.0**（±57°） | T0 靠大探索找到行走解；0.35 探索自由度差一个量级 |
| `clip_min_std` | 0.02 | **1e-12** | T1a 的 0.02 让 σ 无法收敛到小值 |
| `clip_max_std` | 0.2（+300 轮生效） | **None（关闭）** | 该上限把 σ **硬压到 ≤0.2**，是探索不足的头号嫌疑 |
| `actor/critic_hidden_dims` | [512,256,128] | **[256,128,128]** | 网络结构对齐 T0；顺带把 3.7 s/轮 拉回 ~2 s/轮 |
| `entropy_coef` | 0.005 | **0.008** | 探索奖励系数对齐 T0 |

**不动**（用户点名的核心部分）：
* 风格项 `style_lerp = 1.0`（贡献恰为 0）—— 判别器仍在训练，供 T1b 热启动；
* 策略层镜像损失 `mirror_loss`（coef 0.1、`OMNI_LOCO_JOINT_ORDER` 29 关节）；
* S8 的判别器配方（Wasserstein / GP 5.0 / AdamW / coef 4.0 / adaptive LR / 专家集）；
* 冷启动 + 不设轮次上限（`save_interval=500`）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_t1a_ppo_cfg import G1AmpWalkT1aPPORunnerCfg

#: 对齐 T0 的探索/网络口径（见模块 docstring 表格）
AMP_T1A2_INIT_NOISE_STD = 1.0
AMP_T1A2_CLIP_MIN_STD = 1e-12
AMP_T1A2_ENTROPY_COEF = 0.008
AMP_T1A2_HIDDEN_DIMS = (256, 128, 128)


@configclass
class G1AmpWalkT1a2PPORunnerCfg(G1AmpWalkT1aPPORunnerCfg):
    """T1a2：T1a + 探索/网络调回 T0（其余逐字沿用 T1a）。"""

    def __post_init__(self):
        super().__post_init__()

        # --- ① 探索口径 = T0 ---------------------------------------------------
        self.policy.init_noise_std = AMP_T1A2_INIT_NOISE_STD
        self.algorithm.clip_min_std = AMP_T1A2_CLIP_MIN_STD
        #: `None` ⇒ 不钳上限（`algorithms.py:257` 明确说明 None 为 no-op）
        self.algorithm.clip_max_std = None
        self.algorithm.clip_max_std_schedule = None

        # --- ② 网络结构与探索系数 = T0 -----------------------------------------
        self.policy.actor_hidden_dims = list(AMP_T1A2_HIDDEN_DIMS)
        self.policy.critic_hidden_dims = list(AMP_T1A2_HIDDEN_DIMS)
        self.algorithm.entropy_coef = AMP_T1A2_ENTROPY_COEF

        # --- ③ run_name（覆盖 T1a 的）-----------------------------------------
        self.run_name = "t1a2_t0impl_t0explore_mirror0p1"
