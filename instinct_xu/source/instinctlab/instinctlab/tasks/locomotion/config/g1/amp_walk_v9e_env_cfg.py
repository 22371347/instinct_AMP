"""V9e：在 **v9d**（= v9c 的 σ/AMP 参数 + **镜像损失 ON**）之上，按分析结论做两项针对性改动。

## 相对 v9d 的两处改动

### ① 复原 `feet_air_time`（奖励集 15 → 16 项）⭐ 修补"没有任何一项奖励迈步"的缺口

用基类 `G1FlatRewardsCfg` 的原定义（权重 +1.0、`threshold=0.5 s`、`.*_ankle_roll_link`），
函数是 `locomotion_mdp.feet_air_time_positive_biped`，语义（逐字读源码）：

```
single_stance = (sum(in_contact) == 1)                     # 必须**恰好一只脚着地**
reward = min over feet of in_mode_time（单脚支撑时）否则 0
reward = clamp(reward, max=0.5)                            # 单脚支撑越久越接近 0.5
reward *= (||cmd_xy|| > 0.1)                               # 站立/零指令不给分
```

⇒ 三个正好对症的性质：
1. **双脚同时着地/同时离地（= 对称跳跃）拿 0 分** ⇒ 直接打击 v9 的"原地跳"退化解；
2. 奖励"**单脚支撑持续时间**" ⇒ 鼓励真正的交替支撑相；
3. 零指令门控 ⇒ 与下面的②同向（"不动"这件事不再被奖励，也不再从这一项拿分）。

⚠️ 已知局限：**单脚跳**（只用一条腿）在支撑相仍然满足 `single_stance` ⇒ 本项**不**抑制单脚跳；
   ~~那一条由**镜像损失**负责~~ ⚠️ **该说法已被 V9h 实测推翻（2026-09-29）**：V9h@4770 镜像损失
   只有 0.0368，步态却是"单脚侧身跳着走" ⇒ **镜像损失拦不住单侧步态**。机理：它约束的是
   `π(M(o)) ≈ M(π(o))`（策略**函数**的对称性），而单侧步态访问到的状态其镜像态**从未被访问**
   ⇒ 约束可以在"没去过的状态"上被满足。⇒ 真正的补丁是 V9j 的
   `feet_swing_time_penalty`（罚"某只脚腾空过久"），见 `rewards_amp.py`。
⚠️ 该项当初在 v2 采用 AMP_mjlab 12 项配方时被整项删掉，这里**复原**，不是新设计。

### ② 原地（精确 0）指令占比减半：18% → 9%

`prob_low` 仍是 0.30，只把低速档里"精确 (0,0,0)"的比例 `zero_frac_in_low` 0.6 → **0.3**：
- 减半前：0.30 × 0.6 = **18%** 的 env 被命令"原地不动"。在这些 env 里
  `track_lin_vel_xy_exp` 变成"**别动才给分**"（动一下就掉），而 `track_default_pose_on_stand`
  的门控 `exp(−cmd_speed/0.25)` ≈ 1（**完全生效**）⇒ 等于每 5 个 env 就有 1 个在教
  "站住 + 保持默认位姿"；
- 减半后：**9%**，前向指令（≥0.1 m/s）的覆盖从 ~70% 提到 ~79%。
- ⚠️ 契约校验只断言 `rel_standing_envs == 0.0`（我们不动它），`zero_frac_in_low` **不参与校验**
  ⇒ 改它不会 fail-fast（已核对 `_validate_teacher_contract`）。

## 保持不动的（= v9d 的全部内容）

- **镜像损失 ON**（`use_mirror_loss=True`）：防止不对称退化（v9b 的单脚跳）。
- **σ 参数**：`init_noise_std 0.2` / `entropy_coef 0.0025` / `clip_max_std 0.3`
  （实测 σ 从"顶在天花板 0.39"变成**收敛在 0.21**）。
- **AMP 权重** `discriminator_reward_coef = 0.25`（实测 amp_d −0.80→−0.45、r_amp 0.22→0.65）。
- `joint_deviation_arms = 0`（限位已不再触发）；观测/指令范围/数据集与 v9 逐字相同。
- `track_default_pose_on_stand` / `track_base_height` / `body_ang_vel_xy` / `track_ang_vel_z`
  **本轮不动**（保留"单变量优先"：这一轮只验证"迈步奖励 + 减少原地指令"）。

⏱️ 判读建议：≥3000 轮再比步态；关键指标 = `vx_*_vxy_rmse`（跟踪误差）、
`Episode_Reward/feet_air_time`（>0 说明真的在单脚支撑交替）、`base_contact`（摔倒率）。
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
