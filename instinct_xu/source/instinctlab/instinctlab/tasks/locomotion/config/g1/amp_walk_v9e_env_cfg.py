"""V9e：在 v9d 之上做两项针对性改动 —— ① **复原** `feet_air_time`（15 → 16 项）
② 原地指令占比减半（18% → 9%）。

## ⚠️ 本层对**今天**仍有影响的部分

* ⭐ **复原的 `feet_air_time` 今天仍在**：T1l = `+1` / `threshold=0.5` / `func=feet_air_time_positive_biped`
  （= 本层从基类复原的那条；中途 V18 换成目标带版，T1a2 又换回本条）。
* ⭐ **`zero_frac_in_low = 0.3` 今天仍生效**（V18 又设回 0.3）；但配套 `prob_low` 已被 V18 由 0.30 改为
  **0.20** ⇒ 本层"18% → 9%"是**当时口径**，不要当今天的值。
* （算法侧）σ 参数 / `ampc 0.25` / `arms=0` 已被 T1a2 覆盖；但**镜像损失本身今天仍在**（T1a 链 ON）。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | **复原** `feet_air_time`（奖励集 15 → 16 项）：基类 `G1FlatRewardsCfg` 原定义（`+1.0`、`threshold=0.5`、`.*_ankle_roll_link`）| ✅ 仍生效（同 func / 同 threshold）|
| ② | 低速档里"精确 (0,0,0)"比例 `zero_frac_in_low` **0.6 → 0.3**（`prob_low` 仍 0.30）| ✅ 仍 = **0.3**；⚠️ `prob_low` 已被 V18 改为 **0.20** |
| ③ | （算法侧）镜像损失 ON、σ = `0.2 / 0.0025 / 0.3`、`ampc 0.25`、`arms=0` | ❌ 已被 T1a2 覆盖；镜像损失**今天仍在** |

**① 的语义（逐字读源码，今天仍适用）**：单脚支撑 = `sum(in_contact)==1`（**恰好一只脚着地**）；
奖励 = 单脚支撑时长 `clamp(max=0.5)`，再乘零指令门控 `(||cmd_xy|| > 0.1)`。
⇒ 对症三点：(a) 双脚同时着地/离地（对称跳跃）拿 0 分；(b) 奖励"单脚支撑持续时间"；(c) 零指令不给分（与 ② 同向）。
⚠️ 该项是 v2 采用 AMP_mjlab 12 项配方时被整项删掉、这里**复原**（非新设计）；
⚠️ **已知局限（今天仍成立）**：**单脚跳**（只用一条腿）在支撑相仍满足 `single_stance` ⇒ 本项**不**抑制单脚跳。

## 推理与判读（已归档，勿在本文件重复）

* ⚠️ **勘误（2026-10-08，务必照此口径）**：「镜像损失拦不住单侧步态」—— 该说法**已被 V9h 实测推翻**
  （2026-09-29：V9h@4770 镜像损失只有 0.0368，步态却是"单脚侧身跳着走"）。**根因不是**"策略访问不到镜像态
  （off-distribution）"，而是 **V9h 当时只给了前向速度指令**（纯 `(0, 0.7)`，`vy=wz=0`）⇒ 状态分布集中在
  "直行流形"上、**没有镜像场景的计算空间 ⇒ 该约束无法正常生效**；侧向/转向是 **V10 的"③ 指令扩展"**才引入的。
  → `docs/amp_walk/EXPERIMENTS.md` §A1（含旁证：**不能**拿 0.0368 反推"策略已左右对称"）
* 旧 `feet_air_time_positive_biped` 的**两个结构性漏洞**（短到几毫秒也给分 / 越长越给）→ §B2
* ② 的算术（零指令 env 里 `track_lin_vel_xy_exp` 变成"别动才给分"，而 `track_default_pose_on_stand` 门控 ≈1
  ⇒ 每 5 个 env 就有 1 个在教"站住 + 保持默认位姿"）；契约事实：`rel_standing_envs` 仍 0.0、
  `zero_frac_in_low` **不参与** `_validate_teacher_contract` ⇒ 改它不会 fail-fast → 见该版 run 日志
* 「从已塌陷 ckpt 续训基本出不来自吸引子」→ §C4；"单脚侧身跳着走"→ §C3
* ⏱️ 判读建议（≥3000 轮再比步态；看 `vx_*_vxy_rmse` / `Episode_Reward/feet_air_time`（>0 = 真的单脚交替）/ `base_contact`）→ 见该版 run 日志

⚠️ 其余逐字不动：`track_default_pose_on_stand` / `track_base_height` / `body_ang_vel_xy` / `track_ang_vel_z`
   （本轮"单变量优先"只验"迈步奖励 + 减少原地指令"）；观测 / 指令范围 / 数据集与 v9 逐字相同。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V9e-G1-v0`
"""

from __future__ import annotations

import instinctlab.tasks.locomotion.mdp as locomotion_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .amp_walk_v6_env_cfg import G1AmpWalkV6RewardsCfg
from .amp_walk_v9c_env_cfg import G1AmpWalkV9cEnvCfg, G1AmpWalkV9cEnvCfg_PLAY

#: run_name = 日志目录前缀（把关键改动写进目录名，便于事后对账）
AMP_WALK_V9E_RUN_NAME = (
    "G1AmpWalkV9eMVPPlane_obs378_rew16_ar0.02_ns0.2_maxstd0.3_ent0.0025_ampc0.25"
    "_noarms_mirror_airtime_halfstand"
)

#: 低速档里"精确 (0,0,0)"的比例（v9/v9c 是 0.6 ⇒ 18% 的 env 被命令原地不动）⇒ 减半
AMP_WALK_V9E_ZERO_FRAC_IN_LOW = 0.3
#: `feet_air_time` 的权重与单脚支撑时长上限（逐字沿用基类 `G1FlatRewardsCfg`）
AMP_WALK_V9E_AIR_TIME_WEIGHT = 1.0
AMP_WALK_V9E_AIR_TIME_THRESHOLD = 0.5


@configclass
class G1AmpWalkV9eRewardsCfg(G1AmpWalkV6RewardsCfg):
    """v6 的 15 项 + **复原** `feet_air_time` ⇒ 16 项。

    为什么类属性与 `__post_init__` 各写一次：v2 当初是用 `None`（还是 `delattr`）关掉该项
    取决于它的实现细节，而 `@configclass` 的子类字段是在**实例化时**生效、
    `__post_init__` 在**之后**调用 ⇒ 两边都写一次，两种关法都能被我覆盖回来（防坑）。
    """

    feet_air_time: RewTerm = RewTerm(
        func=locomotion_mdp.feet_air_time_positive_biped,
        weight=AMP_WALK_V9E_AIR_TIME_WEIGHT,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": AMP_WALK_V9E_AIR_TIME_THRESHOLD,
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # 覆盖父类可能的 `None` / `delattr`（见类 docstring）
        self.feet_air_time = RewTerm(
            func=locomotion_mdp.feet_air_time_positive_biped,
            weight=AMP_WALK_V9E_AIR_TIME_WEIGHT,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "threshold": AMP_WALK_V9E_AIR_TIME_THRESHOLD,
            },
        )


@configclass
class G1AmpWalkV9eEnvCfg(G1AmpWalkV9cEnvCfg):
    """训练用：v9c/v9d 的环境 + `feet_air_time` + 原地指令减半。"""

    rewards: G1AmpWalkV9eRewardsCfg = G1AmpWalkV9eRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # ⭐ 必须在 super() 之后：v9/v9c 的 `apply_amp_walk_v9_mvp_command_ranges()` 会
        #    把 `zero_frac_in_low` 写回 0.6，这里最后覆盖成 0.3。
        self.commands.base_velocity.zero_frac_in_low = AMP_WALK_V9E_ZERO_FRAC_IN_LOW
        # 父类会写回 v9c 的 RUN_NAME，所以必须在它之后再写一次。
        self.run_name = AMP_WALK_V9E_RUN_NAME


@configclass
class G1AmpWalkV9eEnvCfg_PLAY(G1AmpWalkV9cEnvCfg_PLAY):
    """回放用：与训练同奖励/同指令分布。"""

    rewards: G1AmpWalkV9eRewardsCfg = G1AmpWalkV9eRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.zero_frac_in_low = AMP_WALK_V9E_ZERO_FRAC_IN_LOW
        self.run_name = AMP_WALK_V9E_RUN_NAME


__all__ = [
    "AMP_WALK_V9E_AIR_TIME_THRESHOLD",
    "AMP_WALK_V9E_AIR_TIME_WEIGHT",
    "AMP_WALK_V9E_RUN_NAME",
    "AMP_WALK_V9E_ZERO_FRAC_IN_LOW",
    "G1AmpWalkV9eEnvCfg",
    "G1AmpWalkV9eEnvCfg_PLAY",
    "G1AmpWalkV9eRewardsCfg",
]
