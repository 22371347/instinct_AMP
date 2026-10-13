"""V9d：**V9c + 恢复镜像损失**（V9c 已把 `joint_deviation_arms` 归零、σ 也已收敛）。

## ⚠️ 本层对**今天**仍有影响的部分

* ⭐ **镜像损失是今天唯一仍生效的改动**：`use_mirror_loss=True` 一路延续到 T1l。
  ⚠️ 但"镜像损失拦不住单侧步态"这一**归因已被推翻**（根因在命令分布、不在策略）→ `EXPERIMENTS.md` §A1。
* 本层登记的其余算法参数（σ 组 / `ampc` / `arms`）**今天全部已被覆盖**（见下表）。

## 本层改动一览（V9d = V9c + 镜像损失，单变量）

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | ⭐ 新增 `algorithm.use_mirror_loss = True`（`mirror_joint_names` 从 v4 继承）| ✅ **仍为 True**（经 T1a 链一路延续）|
| ② | 保留 V9c 的 σ 组：`init_noise_std 0.2` / `entropy_coef 0.0025` / `clip_max_std 0.3` | **已被覆盖** → 今天 `1.0 / 0.008 / None` |
| ③ | 保留 V9c 的 `discriminator_reward_coef 0.25`、`joint_deviation_arms 0.0` | **已被覆盖** → 今天 `4.0`、`-0.1` |

## 推理与判读（已归档，勿在本文件重复）

* ⭐ **恢复镜像损失的依据（V9/V9b/V9c 三臂对照：只有开镜像是唯一到达"对称步态"的臂）** → `EXPERIMENTS.md` §A1（原归因已改写）
* V9c 的 σ 收敛（0.20→0.21）与 `amp_d -0.79→-0.45`、`r_amp 0.23→0.65` 实测 → 见该版 run 日志
* ⏱️ 判读建议「≥3000~5000 轮再比步态；~430 轮判"学不会"太早」→ 见该版 run 日志

⚠️ 观测/指令/数据集与 V9/V9c 逐字相同（直接复用 V9 的 teacher 校验器）。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V9d-G1-v0`（算法侧见同节）
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
