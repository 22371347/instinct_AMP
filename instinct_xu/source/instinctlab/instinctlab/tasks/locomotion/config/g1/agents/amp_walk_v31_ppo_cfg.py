"""V31 的算法配置：与 V30 的**判别器配方逐字相同**（mjlab 原始配方）。

本轮唯一变量在 env 侧：`flat_orientation_l2` 权重 −2 → −4（见 `amp_walk_v31_env_cfg.py`）。
算法侧只做一件事：

* **取消 `drop_amp_discriminator`** —— V30 因为"AMP 输入口径变了（134/670 → 390）"才必须丢判别器
  重学；V31 的 AMP 口径**完全没变**（390 维、同 anchor、同 13 体、同 2 帧、同专家数据集）
  ⇒ 保留判别器与它的优化器，从 V30 最新 ckpt **无缝续训**，这样"权重 −2→−4"才是干净的单变量。

⇒ 这样 V31 的变量清单里**只有** `flat_orientation_l2.weight` 一项。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v30_ppo_cfg import G1AmpWalkV30PPORunnerCfg


@configclass
class G1AmpWalkV31PPORunnerCfg(G1AmpWalkV30PPORunnerCfg):
    """v31：判别器配方 = V30（mjlab 原始配方）；只取消 drop-disc，保留判别器续训。"""

    def __post_init__(self):
        super().__post_init__()
        # ⭐ V30 为了"判别器口径变了"设了 DROP_DISC（丢掉判别器重学）；
        #    V31 口径没变 ⇒ 置 None = actor/critic/优化器/**判别器**/观测 normalizer 全部续训。
        self.ckpt_manipulator = None


__all__ = ["G1AmpWalkV31PPORunnerCfg"]
