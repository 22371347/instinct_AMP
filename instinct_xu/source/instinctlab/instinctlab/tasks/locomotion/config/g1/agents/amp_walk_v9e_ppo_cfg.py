"""V9e 的 PPO/算法配置：在 **v9d**（镜像损失 ON + v9c 的 σ/AMP）基础上**算法侧一字不改**。

唯一"动作"是把 `run_name` 对上环境侧（v9e 的改动全在奖励集与指令分布），
避免出现"算法配置的名字"和"环境的名字"不一致的糊涂账。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v9d_ppo_cfg import G1AmpWalkV9dPPORunnerCfg


@configclass
class G1AmpWalkV9ePPORunnerCfg(G1AmpWalkV9dPPORunnerCfg):
    """v9e：= v9d 的算法设置（镜像 ON / σ 收敛参数 / AMP 权重 0.25）+ 只改 run_name。"""

    def __post_init__(self):
        super().__post_init__()

        # --- 算法侧不做任何数值改动，只把"这是哪一版"写清楚 -------------
        self.run_name = (
            "v9e_ns0.2_maxstd0.3_ent0.0025_ampc0.25_noarms_mirror_airtime_halfstand"
        )


__all__ = ["G1AmpWalkV9ePPORunnerCfg"]
