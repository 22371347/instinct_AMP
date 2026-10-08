"""V9d 的 PPO/算法配置：**V9c + `use_mirror_loss = True`**（唯一改动）。

去掉镜像损失会把策略推向"单脚/不对称跳"（V9b 姿态不平衡、V9c 单脚跳），
所以把它恢复回来；其余（σ 收敛用的 entropy 0.0025 / 上限 0.3 / 初值 0.2、
AMP 权重 0.25、手臂偏差归零）全部沿 V9c。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v9c_ppo_cfg import G1AmpWalkV9cPPORunnerCfg


@configclass
class G1AmpWalkV9dPPORunnerCfg(G1AmpWalkV9cPPORunnerCfg):
    """v9d：v9c + 镜像损失（恢复左右等变约束）。"""

    def __post_init__(self):
        super().__post_init__()

        # ⭐ 唯一改动：恢复镜像损失（`mirror_joint_names` 从 v4 的 cfg 继承而来）
        self.algorithm.use_mirror_loss = True

        self.run_name = self.run_name.replace("_nomirror", "_mirror")


__all__ = ["G1AmpWalkV9dPPORunnerCfg"]
