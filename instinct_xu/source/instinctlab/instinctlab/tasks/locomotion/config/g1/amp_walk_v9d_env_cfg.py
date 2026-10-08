"""V9d：**V9c + 恢复镜像损失**（V9c 已把 `joint_deviation_arms` 归零、σ 也已收敛）。

## 为什么恢复镜像损失（用户实测结论）

| 版本 | 镜像损失 | σ 走势 | 实测步态 |
|---|---|---|---|
| V9（参照） | ✅ 开 | 0.25 → **0.39（顶在上限）** | 对称的原地跳 |
| **V9b** | ❌ 关 | 0.25 → **0.39（同样顶在上限）** | **单脚跳 + 姿态不平衡，且不会移动** |
| **V9c** | ❌ 关 | 0.20 → **0.21（收敛！）** | 单脚跳**会移动**，但仍是跳 |

⇒ 结论：**镜像损失不能移除**。它约束的是"左右等变性"，
正是**阻止策略退化成"单脚/不对称跳"**的那一项；去掉之后策略在 ~430 轮就掉进了
不对称的局部最优（V9b 姿态不平衡、V9c 单脚跳）。

## V9d = V9c + 镜像损失（单变量）

保留 V9c 里已被证明有效的两处参数（实测：σ 从"顶天花板"变成**收敛在 0.21**；
AMP 从"占 11% 和不痛不痒"变成 `amp_d −0.79 → −0.45`、`r_amp 0.23 → 0.65`）：

- `policy.init_noise_std = 0.2`
- `algorithm.entropy_coef = 0.0025`（σ 收敛的真正推手）
- `algorithm.clip_max_std = 0.3`
- `algorithm.discriminator_reward_coef = 0.25`（AMP 才能塑造步态）
- `rewards.joint_deviation_arms.weight = 0.0`（限位已不再触发；该项吃 11.6% 回报）
- ⭐ **新增：`algorithm.use_mirror_loss = True`**（`mirror_joint_names` 从 v4 继承，
  无需重复设置）

⚠️ 观测/指令/数据集与 V9/V9c 逐字相同 ⇒ 直接复用 V9 的 teacher 校验器。

⏱️ **判读建议**：V9b/V9c 只跑到 ~430 轮就被判"学不会"其实**太早**（V9 跑了 7600+ 轮）。
本轮至少跑到 3000~5000 轮再比步态。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v9c_env_cfg import G1AmpWalkV9cEnvCfg, G1AmpWalkV9cEnvCfg_PLAY

#: run_name = 日志目录前缀（把关键超参写进目录名，便于事后对账）
AMP_WALK_V9D_RUN_NAME = (
    "G1AmpWalkV9dMVPPlane_obs378_rew15_ar0.02_ns0.2_maxstd0.3_ent0.0025_ampc0.25_noarms_mirror"
)


@configclass
class G1AmpWalkV9dEnvCfg(G1AmpWalkV9cEnvCfg):
    """训练用：与 V9c 的环境**逐字相同**，只改 run_name（区分布局）。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V9D_RUN_NAME


@configclass
class G1AmpWalkV9dEnvCfg_PLAY(G1AmpWalkV9cEnvCfg_PLAY):
    """回放用：与训练同奖励/同观测。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V9D_RUN_NAME


__all__ = [
    "AMP_WALK_V9D_RUN_NAME",
    "G1AmpWalkV9dEnvCfg",
    "G1AmpWalkV9dEnvCfg_PLAY",
]
