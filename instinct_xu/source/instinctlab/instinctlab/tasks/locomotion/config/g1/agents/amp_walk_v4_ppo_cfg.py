"""AMP 行走教师 **v4** 的 PPO runner 配置 —— AMP 侧两项修正（仿 AMP_mjlab v5）。

v4 相对 v3 只改 **AMP 侧**，环境 / 观测 / 动作 / 奖励集 / σ 起点**全部不动**：

1. **镜像损失（mirror loss）** —— `use_mirror_loss=True`, `mirror_loss_coeff=0.1`
   AMP_mjlab v5 用它**替代**手写的落脚对称惩罚，其 docstring 明写目的是
   "消除策略的系统性左右偏置（自转、单腿偏置），从根因上解决落脚不对称"。
   我们此前**完全没有**这个机制（全仓库 grep `mirror` 零命中）。
   算子实现在 `instinctlab/utils/mirror.py`，单测 `instinct/tools/test_mirror_ops.py`
   （云端 20/20 通过，含"默认姿态是不动点"这条关键判据）。

2. **`disable_style_on_stand=True`**（`stand_speed_threshold=0.1`）——
   站立/低速指令下**直接丢掉风格项**。这是 AMP_mjlab 与我们的**唯一一行配置差异**
   （他们是 `True`，我们默认 `False`），而低速正是参考数据最不一致的工况
   （7 个 clip 的 `left_elbow` 均值范围 0.130~1.183，差 55° 以上；
   且参考集里**没有 idle clip**）。

⚠️ **刻意不做**的：
* `symmetric_augmentation_*`（数据层镜像增广）：我们的 clip 混了前向/后向/侧移/弧线，
  镜像会产生**错误标签** —— 这个顾虑依然成立，且 AMP_mjlab 也是 `False`。
  mirror loss 是**策略层**约束，不受参考数据内容影响，所以安全。

⚠️ **已知近似**：mirror loss 在**归一化之后**的观测上施加（storage 存的就是归一化 obs），
   约束的是"策略作为其真实输入的函数"的等变性，而非物理量的等变性。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ..omni_loco_env_cfg import OMNI_LOCO_JOINT_ORDER
from .amp_walk_v3_ppo_cfg import G1AmpWalkV3PPORunnerCfg

#: 镜像损失系数（= AMP_mjlab `mirror_loss_coeff`）
MIRROR_LOSS_COEFF_V4 = 0.1
#: 站立判定的指令速度阈值（= AMP_mjlab `stand_speed_threshold`）
STAND_SPEED_THRESHOLD_V4 = 0.1


@configclass
class G1AmpWalkV4PPORunnerCfg(G1AmpWalkV3PPORunnerCfg):
    """v4：v3 全部超参 + 镜像损失 + 站立关风格。"""

    def __post_init__(self):
        super().__post_init__()

        # --- 1. 镜像损失 -------------------------------------------------
        self.algorithm.use_mirror_loss = True
        # ⚠️ 系数在算法侧必须落在 `mirror_coef` 这个名字上（`PPO.update` 用
        #    `getattr(self, k + "_coef", 1.0)` 取系数，`losses` 里的键是 `"mirror"`）。
        self.algorithm.mirror_loss_coeff = MIRROR_LOSS_COEFF_V4
        # ⚠️ 必须是 **articulation 序**（即 OMNI_LOCO_JOINT_ORDER），不是 npz 序。
        #    算子按关节名求左右置换，顺序本身不影响正确性，但必须与 obs/动作同序。
        self.algorithm.mirror_joint_names = list(OMNI_LOCO_JOINT_ORDER)

        # --- 2. 站立/低速关掉风格项 --------------------------------------
        self.algorithm.disable_style_on_stand = True
        self.algorithm.stand_speed_threshold = STAND_SPEED_THRESHOLD_V4

        self.run_name = f"v4_mirror{MIRROR_LOSS_COEFF_V4}_nostyleonstand"
