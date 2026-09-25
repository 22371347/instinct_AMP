"""AMP 行走教师 **v3** 配置：v2 奖励配方 + 把 `action_rate_l2` 提到 −0.02。

为什么是 v3（对 v2 退步的修正）
------------------------------
v2（12 项，与 AMP_mjlab 配方对齐）**实测全面退步**，已停训：

| 可比指标 | v1 @5000 | v2 @7330 |
|---|---|---|
| 回合长度（步） | 604.7 | 560.0 |
| 触地终止比例 | 0.5498 | **0.6772** |
| 脚距均值 | 0.3518 | **0.4616** |
| **动作噪声 σ** | 0.7557 | **1.5800** |

⭐ **首要病灶**：`Policy/mean_noise_std` **全程单调上升** `0.766 → 1.580`
（实测 `max = 1.57997`，234 个数据点）。`action_scale = 1.0` 下这是
**±90° 的关节目标噪声** —— 策略在**主动放大**探索噪声。

**机制**：`action_rate_l2 = mean((Δa)²)` 是二次型、**每一步每个关节都有梯度**的
**密集**信号；而 v2 保留的 exp 核跟踪奖励**在误差小时梯度趋近 0**。
v2 删掉 9 个密集惩罚项、又把 `action_rate_l2` 削到 1/5，等于**抽走了奖励集里
绝大部分密集梯度**。PPO 的 σ 是可训练参数，梯度信号变弱时它会**膨胀**
（方差大 → 探索多 → 梯度估计更稳），这是 PPO 的经典行为。

⇒ v3 的两处改动（**其余全部继承 v2**）：

1. **本文件**：`action_rate_l2` `−0.01 → −0.02`，把密集梯度补回来
   （不回到 v1 的 −0.05：那个确实也偏强）。
2. **agent cfg**（`agents/amp_walk_v3_ppo_cfg.py`）：`init_noise_std` `1.0 → 0.3`，
   把探索噪声起点从 **±57° 降到 ~±17°**。

⭐ **为什么 v3 从零训练而不是续训**：跳跃是**在大探索噪声下才成立的吸引子**
（支撑腿自稳、对 ±57° 目标噪声鲁棒；平滑交替步态依赖精确摆动腿时序，会被噪声摧毁）。
从 σ≈0.3 起步，策略**根本不会进入**那个吸引子；从已跳跃的 checkpoint 续训则很可能
困在局部最优里出不来。这正是「局部收敛」的机制解释。

本次**不动**的东西（刻意保持单变量归因）
--------------------------------------
- **不加**「步宽过宽」惩罚。现状：`feet_lateral_width` 只在 `sep < 0.145` 时罚
  （实测 −0.0014/s），对**过宽结构性没有约束** —— 所以脚距 0.352→0.462 这个
  **第二个病灶本轮不解决**，留待单独一轮。
- 不改速度/角速度跟踪的 `std`（仍 0.5，与学长同族教师保持形状一致）。
- 不动 `G1FlatRewardsCfg` 与 v1/v2 的任何文件。

接口契约**完全不变**
------------------
动作 29D / 顺序 / `scale=1.0` / default offset / `clip=None`、观测 626/99/670、
时序 50 Hz、执行器 —— 全部继承，与 `INTERFACE_CONTRACT.md` 逐项相同。
所以 checkpoint / ONNX / MuJoCo 回放 / **蒸馏 teacher 侧接口**全部不受影响。
"""

from __future__ import annotations

from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .amp_walk_v2_env_cfg import (
    AMP_WALK_V2_RUN_NAME,
    G1AmpWalkV2EnvCfg,
    G1AmpWalkV2EnvCfg_PLAY,
    G1AmpWalkV2RewardsCfg,
)

#: v3 的 run_name（= 日志目录前缀）。`ar` = action_rate，`ns` = noise std。
AMP_WALK_V3_RUN_NAME = "G1AmpWalkV3Plane_obs626_rew12_ar0.02_ns0.3"

#: v3 相对 v2 的改动（唯一的事实来源，供日志/文档引用）
AMP_WALK_V3_CHANGES: tuple[tuple[str, float, float], ...] = (
    ("rewards.action_rate_l2", -0.01, -0.02),
    ("policy.init_noise_std", 1.0, 0.3),
    ("algorithm.clip_min_std", 0.05, 0.015),
)


@configclass
class G1AmpWalkV3RewardsCfg(G1AmpWalkV2RewardsCfg):
    """v2 的 12 项配方，只把 `action_rate_l2` 从 −0.01 提到 −0.02。

    `func` 逐字沿用基类，只动 `weight`。动机见模块 docstring。
    """

    def __post_init__(self):
        super().__post_init__()
        # 补回密集梯度：v1 用 −0.05 时 σ 稳在 0.756；v2 削到 −0.01 后 σ 涨到 1.580。
        # 取中间值 −0.02，兼顾「足够密集的信号」与「不过度惩罚动作变化」。
        self.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)


@configclass
class G1AmpWalkV3EnvCfg(G1AmpWalkV2EnvCfg):
    """训练用：场景 / 观测 / 事件 / 终止 / 时序 / 动作 / 命令 **全部继承 v2**，
    只替换 `rewards`。"""

    rewards: G1AmpWalkV3RewardsCfg = G1AmpWalkV3RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V3_RUN_NAME


@configclass
class G1AmpWalkV3EnvCfg_PLAY(G1AmpWalkV2EnvCfg_PLAY):
    """Play 用：继承 v2 PLAY 的全部差异，只替换 `rewards`。"""

    rewards: G1AmpWalkV3RewardsCfg = G1AmpWalkV3RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V3_RUN_NAME


__all__ = [
    "AMP_WALK_V2_RUN_NAME",
    "AMP_WALK_V3_CHANGES",
    "AMP_WALK_V3_RUN_NAME",
    "G1AmpWalkV3EnvCfg",
    "G1AmpWalkV3EnvCfg_PLAY",
    "G1AmpWalkV3RewardsCfg",
]
