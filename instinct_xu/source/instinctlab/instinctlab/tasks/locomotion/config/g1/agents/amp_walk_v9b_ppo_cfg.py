"""V9b（消融）：**V9 + 关闭镜像损失**，其余一字不改。

## 为什么做这个消融

用户观察到：V9（带镜像损失）在 ~7500 轮时仍不会迈步，而是采用"与参考数据集相差较远的
跳跃姿态"。怀疑方向之一是镜像损失把策略推向了**左右对称**的运动，而"双脚同相跳"正好是
镜像约束的**平凡解**：

- 跳跃时 `M_obs(o) ≈ o`（状态本身左右对称）⇒ 镜像损失**几乎不受约束**；
- 行走时 `M_obs(o)` 是"另一条腿在前"的**另一个状态** ⇒ `μ(M o) = M μ(o)` 是一个
  **非平凡约束**，策略必须真的学会左右对称。

⇒ 镜像损失并不"奖励跳跃"，但它**对行走收税更重**，因此可能让"跳"这条路径更便宜。
这也正是本消融要验证的：**去掉镜像损失后，是否更早出现原地交替迈步。**

## 单变量

只改 `algorithm.use_mirror_loss = True → False`；`noise_std` / `clip_max_std` /
奖励 / 指令集 / 数据集 / 观测布局全部与 v9 相同，便于逐轮对比。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v9_ppo_cfg import G1AmpWalkV9PPORunnerCfg


@configclass
class G1AmpWalkV9bPPORunnerCfg(G1AmpWalkV9PPORunnerCfg):
    """v9b = v9 − 镜像损失（唯一变量）。"""

    def __post_init__(self):
        super().__post_init__()

        # ⭐ 唯一改动：关掉镜像损失（`compute_losses` 里 `losses["mirror"]` 不再产生）
        self.algorithm.use_mirror_loss = False

        self.run_name = self.run_name.replace("_mirror", "_nomirror")


__all__ = ["G1AmpWalkV9bPPORunnerCfg"]
