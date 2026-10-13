"""V9i：**只有"探索量级"这一个变量**（奖励集与 V9h 逐字相同）。

## ⚠️ 本层对**今天**仍有影响的部分

* 本层**环境侧零改动**（与 V9h 逐字相同，只换 `run_name`）⇒ 无环境侧遗产；三个探索旋钮在 agent cfg。
* ⭐ **本层 agent cfg（`agents/amp_walk_v9i_ppo_cfg.py`）是 V10 agent cfg 的父类**（V10 直接 `import` 它）
  ⇒ 本层定的"**探索量级**"经 V10 一路下传；今天是否仍生效 → 下表 ② 的「算法侧」指针。
* ⚠️ **易误读点**：`__init__.py` 里 V9i 注册段注释那组探索旋钮值与本文件 `run_name` /
  `EFFECTIVE_REWARDS.md` 的「算法侧」**不一致** ⇒ **以 `EFFECTIVE_REWARDS.md`（= 源码常量）为准**。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | 环境侧零改动：20 项奖励 / `style_lerp` / `discriminator_reward_coef` / `schedule="fixed"` / 镜像损失 ON / 指令分布 / 观测 / 数据集 —— 全与 V9h **逐字相同** | —（本层无环境侧改动）；今天已被 T1a2 白名单整体覆盖 |
| ② | 三个"探索"旋钮：`policy.init_noise_std` 0.3→**0.6**、`algorithm.clip_max_std` 0.4→**0.7**、`algorithm.entropy_coef` 0.0025→**0.005**〔算法侧〕| 见 `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V9i-G1-v0` 的**「算法侧」节**（是否仍生效，对照 `...V10...` / `...T1l...` 同节）|

## 推理与判读（已归档，勿在本文件重复）

* **参照配方**：仓库里**不含 AMP** 的行走配方 `G1OmniLocoPPORunnerCfg` 在**同机器人、同 `action_scale=1.0`、
  同时序契约**下能从零学会走路，而它用 `init_noise_std=1.0`（±57°）+ `entropy_coef=0.008`、上限无
  ⇒ **V9h 比"能学会走路的配方"少 1.7~3.3 倍探索幅度** → 该版 run 日志
* **为什么取 0.6**（非参考的 1.0、也非先写的 0.5）：真正的坑是 **σ 会失控**（`entropy_coef=0.008` + 无上限
  ⇒ v8 实测 σ 涨到 **1.04 rad**）⇒ 用"初值 0.6 + 硬上限 0.7 + entropy 0.005"把 σ 关在 **0.3→0.7** 有界区间
  → 该版 run 日志
* **高 σ 两个副作用与应对**：① 限位（V9h `dof_pos_limits` 仅 −0.0207/s，很干净；升高后盯住）；
  ② 不稳定（盯 `base_contact`）。备用手段（**本版不做**）：动作项 `clip` → ±3 —— ⚠️ 这会改**动作契约**
  ⇒ 必须同步 `INTERFACE_CONTRACT.md` 与 MuJoCo 回放工具（`g1_mujoco_play.py`）→ 该版 run 日志
* **支撑证据（V9h @2050）**：σ 稳在 0.31~0.32（不再自降 ⇒ 策略**主动想要更大噪声**）；`amp_d = −0.583` /
  `amp_d_ref = +0.592`（**gap = 1.18**，判别器区分得开、没失效），但 `d_actor` 650 轮钉死 ⇒ 瓶颈不在判别器，
  而在**策略没探索过去**；`feet_air_time` 0.0026→**0.0384** ⇒ 方向对、只差"跨过去" → §A2 / §A6
* **设计 + 判读**：V9h vs V9i = **同奖励、同 AMP 设置，唯一差别是探索强度** ⇒ 1500 轮 A/B 一次性回答
  "是不是探索不够"；判据 `feet_air_time` > **0.1/s**（现 0.034）、`vx_high` RMSE 继续降（0.594→0.237）、
  `base_contact` < 0.5（现 ~0.52）、`amp_d` 向 0 靠近、⚠️ 监控 `action_abs_max`（已 11.8）→ 该版 run 日志

⚠️ 其余逐字不动：V9h 的 20 项奖励 / AMP 设置 / 指令分布 / 观测 / 数据集。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V9i-G1-v0`
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
