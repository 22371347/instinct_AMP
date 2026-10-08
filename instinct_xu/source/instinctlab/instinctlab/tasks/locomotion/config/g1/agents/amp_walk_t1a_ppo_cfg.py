"""T1a 的 PPO runner 配置：**S8 的判别器/PPO 口径 + 风格项归零 + 策略层镜像损失**。

============================================================================
T1a ↔ T1b 的"单变量"关系（本文件的核心设计）
============================================================================
总奖励的混合公式是 `r = λ·task + (1−λ)·coef·style`（λ = `style_lerp`）：

| | `style_lerp` | 效果 |
|---|---|---|
| **T1a（本文件）** | **1.0** | 风格项恰好归零 ⇒ **总奖励 = 纯任务项**；判别器仍在训练但不参与奖励 |
| **T1b** | 0.75 | 把风格项打开 ⇒ **只需改这一个数**（从 T1a 的 ckpt 续训即可） |

⇒ 这样 T1b 是真正的单变量增量，"风格项是否又把高速行走弄没"这个问题不会被别的
差异污染。

镜像损失（交付要求②的"对称能力"）：
* 用 **策略层** `mirror_loss`（`L = coeff·‖μ_π(M_obs·o) − M_act·μ_π(o)‖²`），
  系数 0.1，关节名用 **articulation 序**（`OMNI_LOCO_JOINT_ORDER`）。
* **刻意不开**数据层 `symmetric_augmentation_*`：我们的参考 clip 混了前向/后向/
  侧移/弧线，镜像参考会产生**错误标签**（v4 的结论，AMP_mjlab 也是关的）；
  而策略层镜像损失不受参考数据内容影响 ⇒ 安全。
* 该机制在 V4/V9d 上用过（`amp_walk_v4_ppo_cfg.py` / `amp_walk_v9d_env_cfg.py`），
  不是新代码；算子有单测（`instinct/tools/test_mirror_ops.py`）。

其余超参（判别器正则 / coef 4.0 / adaptive LR / 专家集 / 无白化 raw GP）
**逐字沿用 S8**，不做任何顺手调整。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ..omni_loco_env_cfg import OMNI_LOCO_JOINT_ORDER
from .amp_walk_v33s8_ppo_cfg import G1AmpWalkV33S8PPORunnerCfg

#: 镜像损失系数（= AMP_mjlab `mirror_loss_coeff`，与 v4 一致）
AMP_T1A_MIRROR_LOSS_COEFF = 0.1
#: T1a 的风格项权重：**1.0 ⇒ 风格项归零**（T1b 改回 0.75 即可打开）
AMP_T1A_STYLE_LERP = 1.0
#: 不设轮次上限（跑到人工叫停；本项目惯例）
AMP_T1A_MAX_ITERATIONS = 1_000_000_000


@configclass
class G1AmpWalkT1aPPORunnerCfg(G1AmpWalkV33S8PPORunnerCfg):
    """T1a：S8 口径 + `style_lerp=1.0`（风格归零）+ 镜像损失 ON。"""

    def __post_init__(self):
        super().__post_init__()

        # --- ① 风格项归零（总奖励 = 纯任务项）--------------------------------
        self.algorithm.style_lerp = AMP_T1A_STYLE_LERP

        # --- ② 策略层镜像损失 ON --------------------------------------------
        self.algorithm.use_mirror_loss = True
        # ⚠️ 系数必须落在 `mirror_coef` 这个名字上（`PPO.update` 用
        #    `getattr(self, k + "_coef", 1.0)` 取系数，`losses` 里的键是 `"mirror"`）。
        self.algorithm.mirror_loss_coeff = AMP_T1A_MIRROR_LOSS_COEFF
        # ⚠️ 必须是 **articulation 序**（OMNI_LOCO_JOINT_ORDER），与 obs/动作同序；
        #    不传会直接抛 "use_mirror_loss=True 但没给 mirror_joint_names"。
        self.algorithm.mirror_joint_names = list(OMNI_LOCO_JOINT_ORDER)

        # --- ③ 不设轮次上限 + 冷启动 -----------------------------------------
        self.max_iterations = AMP_T1A_MAX_ITERATIONS
        self.save_interval = 500
        self.resume = False
        self.load_run = None

        self.run_name = "t1a_t0rewards_stylelerp1p0_mirror0p1"
