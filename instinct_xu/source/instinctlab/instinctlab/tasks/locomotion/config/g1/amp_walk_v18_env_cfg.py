"""V18：**步态奖励改造 + 数据集/指令收敛到"方案 A"**（用户 2026-10-01 指定）。

基准 = V17（从零 / 10 帧判明器 / `flat_orientation_l2 −2` / `feet_air_time 4.0`）。

## 本轮改的三组（都在用户本条指令范围内）

### ① 步态奖励换成 **目标时长带**：`feet_air_time_target_biped`（新函数，见 `mdp/rewards_amp.py`）

旧版 `feet_air_time_positive_biped` 的核 = `clamp(min(支撑脚接触时长, 摆动脚腾空时长), max=0.6)`，
即**按秒计费**，于是有两个结构性漏洞：

1. **短到几毫秒也给分** ⇒ 高频小碎步照样刷分（**V17 实测核值只有 0.005~0.009 s**，
   即单支撑相位仅几毫秒 —— 这与用户观察到的"小幅高频跳走"完全一致）；
2. **越长越给**（0.6 s 封顶）⇒ "单脚一直踩着、另一只脚吊着"能长期拿满
   （V9h 的"单脚侧身跳"、V14 从零的**单脚跳**都是这条捷径）。

新版：单支撑相位时长 `d` 落在 **[0.4, 1.0] s** 给满额 1.0，两侧**线性衰竭**（尺度 0.3 s）：

| d | 0.01 s | 0.2 s | **0.4 ~ 1.0 s** | ≥ 1.3 s |
|---|---|---|---|---|
| 旧版 | 0.01 | 0.20 | 0.4 ~ 0.6 | 0.6（封顶，跳着刷） |
| **新版** | **0** | 0.33 | **1.0** | **0** |

⇒ **同时反对碎步与反对"单脚长时间支撑/跳"**，且**上限就是 1.0**（无法靠踩久刷分）
⇒ V14 那个"跳着刷分"的坑从**结构上**消失 ⇒ 本版保持 V17 的权重 **4.0**（不必再靠压低权重防跳）。
同样的权重下本项每步收益从 ~0.03 抬到 **~1.2/步**（混合后），与速度跟踪（1.06/步）同量级 ——
这是**有意**的：不给步态项话语权就压不住碎步。

### ② `action_rate_l2` −0.06 → **−0.10**（用户认可）

V17 的抖动在持续变大（−0.406 → −0.441/步），该项是唯一直接约束关节抖动的位置。

### ③ `feet_swing_time_penalty` 权重 → **0.0**（避免与①打架）

它的作用是"罚腾空 > 0.6 s"，而①的上界在 1.0 s ⇒ 0.6~1.0 s 区间里一个给满分、一个在罚
⇒ 两项对撞。①已经完整覆盖"过长"这一侧 ⇒ 把它置 0（**保留该 term 只为日志**，权重 0 不进奖励）。

### ④ 数据集/指令 = **方案 B：5 clip + 前后左右/停止/斜向 四档指令**（用户 2026-10-01 定稿）

直接上全向方案（而不是先做"1 clip + 前进/停止"的方案 A）的理由 —— **避免再动三个跟踪项**：

| | 方案 A（前进/停止） | **方案 B（前后左右/停止/斜向）** |
|---|---|---|
| 指令 `ωz` | 恒 0 | 有真实指令（∈±0.8） |
| `track_ang_vel_z_exp`(0.8) | ⚠️ **退化成"几乎恒定白拿分"**（≈0.4~0.5/步）⇒ 必须降权 | ✅ **有效激励，不用改** |
| `track_lin_vel_xy_exp`(3.0) | 需另议 | 需另议（与指令集无关） |
| 最终目标 | 只是中间步骤 | 就是目标（全向行走） |

⇒ 用 B 可以 **三个跟踪项一个都不动**（3.0 / 0.8 / 1.0 全保持），这正是用户要的。
⇒ 数据集与指令**成套**取自 V13（已设计好、但当时没跑）：

| 档 | 概率 | 采样 |
|---|---|---|
| 静止/低速 | 0.20（30% 精确 0） | `‖v‖ ≤ 0.1`、`ωz=0` |
| 前后向 | 0.25（`prob_lin_fwd=0.6`） | 直线前向 `vx∈(0.5,0.9)`、`ωz=0`；其余 `vy=0` 的均匀 |
| 纯侧向 | 0.25 | `vx≡0`、`vy∈(−0.5,0.5)`、**`ωz≡0`**（`side_ang_vel_range`） |
| 持续后退 | 0.10 | `vy=0`、`vx∈(−0.5,−0.3)`、`ωz≈0` |
| 斜向/转向 | 0.20（余量=父类均匀） | `(vx,vy,ωz)` 三轴随机 |

`ranges = ((-0.5, 1.0), (-0.5, 0.5), (-0.8, 0.8))`。参考数据集 = **5 clip**：直行前进 /
弧线走 /
左侧移 /
右侧移 /
后退走（`amp_walk_v13_motion_selection.yaml`）。

⚠️ **风险（必须监控）**：从零学**全向**步态（侧移是交叉跨步、后退是反向交替）比"只会向前"难得多，
而每个模态只有 **1 个 clip** 的参考密度 ⇒ 侧移/后退可能欠拟合。早期判据：
- `Episode_Reward/track_lin_vel_xy_exp` 的总量（若从 V17 的 1.5 掉到 1.0 以下 ⇒ 多模态没跟上）；
- `Episode_Reward/track_ang_vel_z_exp`（有 wz 指令 ⇒ 看它是否开始下降，即真的在跟转向）；
- 回放时用 `--cmd 0 0.3 0`（侧移）/ `--cmd -0.3 0 0`（后退）单独验证（由用户跑 MuJoCo）。
- ⚠️ 侧移的"交叉跨步"与 `feet_crossing_cost`/`feet_lateral_width_cost` 不冲突：两项都带
  `gate_lateral=|vx|/‖v‖`（纯侧移时门控为 0）—— V13 设计时专门处理过。

## ⚠️ 本轮是"3 组"变量（不是单变量），判读时按组归因

| 组 | 看什么 | 期望 |
|---|---|---|
| ① 目标带步态 | `Step/air_time_single_stance`（新探针，单支撑时长均值）、`air_time_in_band_frac`、`Episode_Reward/feet_air_time` | 前者从 V17 的 ~0.007 s **升到 0.4~1.0 s**；后两者显著 > 0 |
| ②−③ 抖动 | `action_rate_l2`、`dof_acc_l2` | 抖动不再单调恶化 |
| ④ 数据集/指令收敛 | `Loss/discriminator_loss` / `Step/amp_d` / `amp_style_reward` 的**变化率** | 判别器"重新有话说"（loss 下降、`amp_d` 不再钉死） |
| 全局 | `Mean reward`、`track_lin_vel_xy_exp`、回放步态 | 不能为了步态牺牲前进能力 |

## 其余逐字不动

10 帧判别器（670 维）、其余 18 项奖励（含 `flat_orientation_l2 −2`、`track_default_pose_on_stand −0.3`、
`joint_pos_limits_margin −1.5`）、镜像增广/损失、σ 上限 0.6、固定 LR、λ 0.75 / c 1.0、**从零**。
"""

from __future__ import annotations

import os

import instinctlab.tasks.locomotion.mdp as locomotion_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .amp_walk_env_cfg import MOTION_NAME
from .amp_walk_v17_env_cfg import (
    G1AmpWalkV17EnvCfg,
    G1AmpWalkV17EnvCfg_PLAY,
    G1AmpWalkV17RewardsCfg,
    G1AmpWalkV17TeacherEnv,
)

# ============================================================================
# ④ 方案 B：数据集（5 clip）+ 指令（前后左右 / 停止 / 斜向 四档）
# ============================================================================

#: 方案 B 的参考数据集：5 clip（直行 / 弧线 / 侧移×2 / 后退）——取自 V13（已设计未跑）
AMP_WALK_V18_MOTION_SELECTION_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "motions",
    "amp_walk",
    "amp_walk_v13_motion_selection.yaml",
)

#: 指令范围：`vx∈(−0.5,1.0)`（前后）· `vy∈(−0.5,0.5)`（侧向）· `ωz∈(−0.8,0.8)`（转向）
AMP_WALK_V18_COMMAND_RANGES = ((-0.5, 1.0), (-0.5, 0.5), (-0.8, 0.8))
#: ① 静止/低速档
AMP_WALK_V18_PROB_LOW = 0.20
AMP_WALK_V18_LOW_VEL = 0.1
AMP_WALK_V18_ZERO_FRAC_IN_LOW = 0.3
#: ② 前后向档（`vy=0`、`ωz=0`）
AMP_WALK_V18_PROB_LIN = 0.25
AMP_WALK_V18_PROB_LIN_FWD = 0.6
AMP_WALK_V18_STRAIGHT_LIN_VEL_RANGE = (0.5, 0.9)      # 贴着直行 clip 的 0.6~1.0 下沿
AMP_WALK_V18_STRAIGHT_ANG_VEL_RANGE = (0.0, 0.0)
#: ③ 持续后退档（`vy=0`、`vx<0`）
AMP_WALK_V18_PROB_BACK = 0.10
AMP_WALK_V18_BACK_LIN_VEL_RANGE = (-0.5, -0.3)        # 后退 clip 是 0.55 m/s
AMP_WALK_V18_BACK_ANG_VEL_RANGE = (0.0, 0.0)
#: ④ 纯侧向档（`vx=0`，且 `ωz` 也钉零 ⇒ 严格"纯侧移"）
AMP_WALK_V18_PROB_SIDE = 0.25
AMP_WALK_V18_SIDE_ANG_VEL_RANGE = (0.0, 0.0)
#: 余量 0.20 = 斜向/转向档（父类均匀采样 `(vx,vy,ωz)`）
#:
#: ⚠️ **三个跟踪项（`track_lin_vel_xy_exp 3.0` / `track_ang_vel_z_exp 0.8` /
#: `track_base_height 1.0`）全部保持不动** —— 方案 B 里 `ωz` 有真实指令，
#: `track_ang_vel_z_exp` 就是有效激励（不像方案 A 那样退化成恒定白拿分）。

# ============================================================================
# ① 步态奖励：目标时长带
# ============================================================================

#: 单支撑相位时长的"好区间"（秒）——人类常速步行 ≈ 0.35~0.45 s
AMP_WALK_V18_AIR_TIME_GOOD_LO = 0.4
AMP_WALK_V18_AIR_TIME_GOOD_HI = 1.0
#: 区间外线性衰竭的尺度（秒）：0.4−0.3=0.1 s 及以下得 0；1.0+0.3=1.3 s 及以上得 0
AMP_WALK_V18_AIR_TIME_SOFT = 0.3
#: 权重（保持 V17 的 4.0：新形式已无"踩久刷分"漏洞，不必靠权重压制）
AMP_WALK_V18_AIR_TIME_WEIGHT = 4.0

#: ② 关节抖动（V11 起是 −0.06）
AMP_WALK_V18_ACTION_RATE_WEIGHT = -0.10

AMP_WALK_V18_RUN_NAME = (
    "G1AmpWalkV18MVPPlane_obs378_rew21_amp10frame_flat2.0_airband_ar0.10_5clip_omnidir_scratch"
)


def _apply_v18_dataset_commands(cfg) -> None:
    """把数据集与指令设成**方案 B**（5 clip + 前后左右/停止/斜向 四档）——逐字取自 V13。"""
    cfg.scene.motion_reference.motion_buffers[MOTION_NAME].filtered_motion_selection_filepath = (
        AMP_WALK_V18_MOTION_SELECTION_FILE
    )
    cmd = cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = AMP_WALK_V18_COMMAND_RANGES[0]
    cmd.ranges.lin_vel_y = AMP_WALK_V18_COMMAND_RANGES[1]
    cmd.ranges.ang_vel_z = AMP_WALK_V18_COMMAND_RANGES[2]

    cmd.prob_low = AMP_WALK_V18_PROB_LOW
    cmd.low_vel = AMP_WALK_V18_LOW_VEL
    cmd.zero_frac_in_low = AMP_WALK_V18_ZERO_FRAC_IN_LOW

    cmd.prob_side = AMP_WALK_V18_PROB_SIDE
    cmd.side_ang_vel_range = AMP_WALK_V18_SIDE_ANG_VEL_RANGE

    cmd.prob_lin = AMP_WALK_V18_PROB_LIN
    cmd.prob_lin_fwd = AMP_WALK_V18_PROB_LIN_FWD
    cmd.straight_lin_vel_range = AMP_WALK_V18_STRAIGHT_LIN_VEL_RANGE
    cmd.straight_ang_vel_range = AMP_WALK_V18_STRAIGHT_ANG_VEL_RANGE

    cmd.prob_back = AMP_WALK_V18_PROB_BACK
    cmd.back_lin_vel_range = AMP_WALK_V18_BACK_LIN_VEL_RANGE
    cmd.back_ang_vel_range = AMP_WALK_V18_BACK_ANG_VEL_RANGE


@configclass
class G1AmpWalkV18RewardsCfg(G1AmpWalkV17RewardsCfg):
    """V17 的 21 项（含 `flat_orientation_l2 −2`）+ ① 目标带步态 + ② 抖动 + ③ 关掉 swing 罚。

    ①②③ 都只改 `weight` / 换 `func`，其余项的 `func`/`params` 一律原样保留。
    """

    def __post_init__(self):
        super().__post_init__()

        # ① 步态：`feet_air_time_positive_biped`（按秒计费）→ `feet_air_time_target_biped`（目标带）
        self.feet_air_time = RewTerm(
            func=locomotion_mdp.feet_air_time_target_biped,
            weight=AMP_WALK_V18_AIR_TIME_WEIGHT,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "good_lo": AMP_WALK_V18_AIR_TIME_GOOD_LO,
                "good_hi": AMP_WALK_V18_AIR_TIME_GOOD_HI,
                "soft": AMP_WALK_V18_AIR_TIME_SOFT,
            },
        )

        # ② 关节抖动：−0.06 → −0.10
        base_ar = self.action_rate_l2
        self.action_rate_l2 = RewTerm(
            func=base_ar.func,
            weight=AMP_WALK_V18_ACTION_RATE_WEIGHT,
            params=base_ar.params,
        )

        # ③ 关掉"腾空 > 0.6 s"的罚（与①的上界重叠、会打架）；保留 term 只为日志
        base_sw = self.feet_swing_time_penalty
        self.feet_swing_time_penalty = RewTerm(
            func=base_sw.func,
            weight=0.0,
            params=base_sw.params,
        )


@configclass
class G1AmpWalkV18EnvCfg(G1AmpWalkV17EnvCfg):
    """训练用：V17 + 步态目标带 + 抖动加大 + 方案 A（1 clip / 前进+停止）。"""

    rewards: G1AmpWalkV18RewardsCfg = G1AmpWalkV18RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_v18_dataset_commands(self)
        self.run_name = AMP_WALK_V18_RUN_NAME


@configclass
class G1AmpWalkV18EnvCfg_PLAY(G1AmpWalkV17EnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广。"""

    rewards: G1AmpWalkV18RewardsCfg = G1AmpWalkV18RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_v18_dataset_commands(self)
        self.run_name = AMP_WALK_V18_RUN_NAME


class G1AmpWalkV18TeacherEnv(G1AmpWalkV17TeacherEnv):
    """契约期望值只写在 env 类上。

    ⚠️ 本版**指令范围扩到全向**（V13 的四档）⇒ 必须把它声明成期望值，否则
    `_validate_teacher_contract` 会在启动时直接报 `command ranges ... != ...`。
    （V13 当年正是漏了这一步 —— 它的 TeacherEnv 沿用了 V10b 的期望值，一旦真跑就会启动即报错。）
    """

    expected_command_ranges = AMP_WALK_V18_COMMAND_RANGES


__all__ = [
    "AMP_WALK_V18_ACTION_RATE_WEIGHT",
    "AMP_WALK_V18_AIR_TIME_GOOD_HI",
    "AMP_WALK_V18_AIR_TIME_GOOD_LO",
    "AMP_WALK_V18_AIR_TIME_SOFT",
    "AMP_WALK_V18_AIR_TIME_WEIGHT",
    "AMP_WALK_V18_BACK_LIN_VEL_RANGE",
    "AMP_WALK_V18_COMMAND_RANGES",
    "AMP_WALK_V18_LOW_VEL",
    "AMP_WALK_V18_MOTION_SELECTION_FILE",
    "AMP_WALK_V18_PROB_BACK",
    "AMP_WALK_V18_PROB_LIN",
    "AMP_WALK_V18_PROB_LIN_FWD",
    "AMP_WALK_V18_PROB_LOW",
    "AMP_WALK_V18_PROB_SIDE",
    "AMP_WALK_V18_RUN_NAME",
    "AMP_WALK_V18_SIDE_ANG_VEL_RANGE",
    "AMP_WALK_V18_STRAIGHT_LIN_VEL_RANGE",
    "AMP_WALK_V18_ZERO_FRAC_IN_LOW",
    "G1AmpWalkV18EnvCfg",
    "G1AmpWalkV18EnvCfg_PLAY",
    "G1AmpWalkV18RewardsCfg",
    "G1AmpWalkV18TeacherEnv",
]
