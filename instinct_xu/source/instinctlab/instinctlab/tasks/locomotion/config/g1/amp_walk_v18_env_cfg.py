"""V18：步态奖励改造 + 数据集/指令收敛到「方案 B」（用户 2026-10-01 指定）。基准 = V17。

## ⚠️ 本层对**今天**仍有影响的部分（其余已被后层覆盖，见下表）

* ⭐ **参考数据集 = 5 clip 仍在使用**：`amp_walk_v13_motion_selection.yaml`
  （直行前进 / 弧线走 / 左侧移 / 右侧移 / 后退走）—— 即今天仍在跑的参考集。
* ⭐ **指令分档结构**（低 / 前后 / 纯侧向 / 持续后退 / 斜向）由本层的「方案 B」定下，
  今天仍沿用（范围后来被改窄为 `(-0.5,1.0) / (-0.5,0.5) / (-0.8,0.8)`）。
* `feet_swing_time_penalty` 置 **0.0**（避免与①对撞）—— **今天仍是 0** ✅

## 本层改动一览（⚠️ 本轮是「3 组变量」，非单变量）

| # | 改动 | 现状（T1k/T1l 实际生效）|
|---|---|---|
| ① | `feet_air_time` 换成**目标时长带** `feet_air_time_target_biped`（单支撑相位 ∈ [0.4,1.0] s 给满额，两侧线性衰竭 0.3 s，**上限 1.0 ⇒ 无法靠踩久刷分**）；权重保持 4.0 | **已被 T1a2 覆盖** → 换回 `feet_air_time_positive_biped`（`threshold=0.5`），权重 1.0 |
| ② | `action_rate_l2` −0.06 → **−0.10**（补密集梯度、治抖动）| **已被覆盖** → −0.05（flat 基类）|
| ③ | `feet_swing_time_penalty` → **0.0** | ✅ 仍为 0 |
| ④ | 数据集/指令 = **方案 B**（5 clip + 五档指令）。选 B 不选 A 的理由：**避免再动三个跟踪项** —— 方案 A 下 `ωz` 恒 0 会让 `track_ang_vel_z_exp` 退化成「几乎恒定白拿分」 | 参考集 ✅ 沿用；指令范围已改窄 |

**推理与判读已归档，勿在本文件重复**：
* 旧 `feet_air_time_positive_biped` 的**两个结构性漏洞** → `docs/amp_walk/EXPERIMENTS.md` §B2
  （短到几毫秒也给分 / 越长越给 ⇒ 单脚吊着能拿满）
* 本层监控清单（`Step/air_time_single_stance` 应从 ~0.007 s 升到 0.4~1.0 s、
  `air_time_in_band_frac`、判别器「重新有话说」）见该版 run 日志

⚠️ 其余逐字不动：10 帧判别器（670 维）、其他 18 项奖励、镜像增广/损失、σ 上限 0.6、
   固定 LR、λ 0.75 / c 1.0、**从零**。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V18-G1-v0`
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
