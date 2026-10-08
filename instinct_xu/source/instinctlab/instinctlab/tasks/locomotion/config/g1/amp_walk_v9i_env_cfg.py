"""V9i：**只有"探索量级"这一个变量**（奖励集与 V9h 逐字相同）。

## 为什么开这一版

仓库里那份**不含 AMP 的行走配方**（`G1OmniLocoPPORunnerCfg`）在**同一机器人、同一 `action_scale=1.0`、
同一时序契约**（dt 0.005 / decimation 4 / 指令 10 s 保持）下能从零学会走路，而它是：

| 字段 | 参考配方 | 我们 V9h | **V9i** |
|---|---|---|---|
| `init_noise_std` | **1.0**（±57°） | 0.3（±17°） | **0.6（±34°）** |
| `entropy_coef` | **0.008** | 0.0025 | **0.005** |
| `clip_max_std` | 无上限 | 0.4 | **0.7（有界！）** |

⇒ **我们比"能学会走路的配方"少 1.7~3.3 倍探索幅度。**

## 支撑这个方向的实测证据（V9h @2050）

- `Mean action noise std` **稳在 0.31~0.32（峰值 0.33）**，不再像 v9f 那样自己掉到 0.23
  ⇒ 策略**主动想要更大的噪声**；
- `Mean step amp_d = −0.583`、`amp_d_ref = +0.592` ⇒ **gap = 1.18**（起始 1.51），
  判别器**区分得很开、没有失效**，但 `d_actor` 650 轮钉死在 −0.58 不动
  ⇒ 瓶颈不在判别器，而在**策略没往参考动作那片区域探索过去**；
- `Episode_Reward/feet_air_time` 从 0.0026 单调爬到 **0.0384**（单脚支撑在增加）⇒ 方向对，只差"跨过去"。

## 唯一变量

**奖励集 20 项、`style_lerp 0.75`、`discriminator_reward_coef 0.25`、`schedule="fixed"`、
镜像 ON、指令分布、观测/数据集 —— 全部与 V9h 逐字相同**，只改上面那三个"探索"旋钮。

⇒ V9h vs V9i = **同奖励、同 AMP 设置，唯一差别是探索强度**，
1500 轮即可做 A/B，一次性回答"到底是不是探索不够"。

## 判读

| 判据 | 期望 |
|---|---|
| `Episode_Reward/feet_air_time` | > **0.1 /s**（V9h 现 0.034）⇒ 真交替支撑 |
| `vx_high` RMSE | 继续下降（V9h 已从 v9f 的 0.594 → **0.237**） |
| `Episode_Termination/base_contact` | < 0.5（V9h 现 ~0.52） |
| `amp_d` | 若向 0 靠近 ⇒ 探索把策略推向了参考动作那片区域 |
| `action_abs_max` | ⚠️ 监控（V9h 已到 11.8，σ 变大后可能更大） |
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v9h_env_cfg import (
    AMP_WALK_V9H_RUN_NAME,
    G1AmpWalkV9hEnvCfg,
    G1AmpWalkV9hEnvCfg_PLAY,
)

AMP_WALK_V9I_RUN_NAME = AMP_WALK_V9H_RUN_NAME.replace("V9h", "V9i").replace(
    "_ns0.3_", "_ns0.6_maxstd0.7_ent0.005_"
)


@configclass
class G1AmpWalkV9iEnvCfg(G1AmpWalkV9hEnvCfg):
    """训练用：**奖励集与 V9h 完全相同**，只换 run_name（探索旋钮在 agent cfg 里）。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V9I_RUN_NAME


@configclass
class G1AmpWalkV9iEnvCfg_PLAY(G1AmpWalkV9hEnvCfg_PLAY):
    """回放用：与训练同奖励/同分布。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V9I_RUN_NAME


__all__ = [
    "AMP_WALK_V9I_RUN_NAME",
    "G1AmpWalkV9iEnvCfg",
    "G1AmpWalkV9iEnvCfg_PLAY",
]
