"""V10b 的算法配置：**与 V10 逐字相同**（本版唯一变量是数据集 + 指令分档）。"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v10_ppo_cfg import G1AmpWalkV10PPORunnerCfg


@configclass
class G1AmpWalkV10bPPORunnerCfg(G1AmpWalkV10PPORunnerCfg):
    """v10b：探索量级/风格权重/镜像与 v10 完全一致，只换 run_name。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = (
            "v10b_ns0.6_maxstd0.7_ent0.005_fixedlr_lerp0.75_ampc0.25_rew21_aug_cmd3_clip2"
        )


__all__ = ["G1AmpWalkV10bPPORunnerCfg"]
