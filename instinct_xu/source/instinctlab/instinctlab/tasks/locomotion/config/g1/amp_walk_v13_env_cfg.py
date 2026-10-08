"""V13：**全量测试** —— 指令集向侧向/后向拓展 + 补进对应数据集（用户 2026-09-30 指定）。

## ① 指令分档：3 档 → **4 档**（用户指定）

| 档 | 概率 | 采样 | 依据 |
|---|---|---|---|
| **静止/低速** | 0.20 | `‖v‖ ≤ 0.1`、`ωz=0`，其中 30% 精确 `(0,0,0)` | "0.1 m/s 内的静止指令" |
| **前后向** | 0.25+0.10 | `vy = 0`、`ωz = 0`：前向 0.5~0.9 / 均匀 `vx∈(−0.5,1.0)` / 后退 (−0.5,−0.3) | "(−0.5,1.0) 前后向" |
| **纯侧向** | 0.25 | `vx = 0`、`vy ∈ (−0.5,0.5)`、`ωz = 0`（新增 `side_ang_vel_range`） | "(−0.5,0.5) 侧向" |
| **斜向/转向** | 余量 0.20 | `(vx,vy,ωz)` 在 `ranges` 内**三轴全随机** | "xyz 均随机取" |

`ranges = ((−0.5,1.0), (−0.5,0.5), (−0.8,0.8))` —— 与数据集包络对照：
参考速度包络实测 `vx∈[−1.07,+1.69]`、`vy∈[−0.48,+0.55]`、`ωz∈[−0.77,+1.0]`
⇒ `vy=±0.5`、`ωz=−0.8` 属**轻度外插**（静态契约校验器会给出 ⚠️ 提示，可接受）。

为什么"前后向档"要拆成三段：均匀采样下**后退会被前向范围稀释**（后退只占 `(−0.5,1.0)` 的 1/3），
策略容易用"双脚同时蹬地后跳"蹭 `track_lin_vel` 而不做交替步 ⇒ `prob_back` 给它专属的**持续**配额
（`back_lin_vel_range=(−0.5,−0.3)`、`ωz=0`）。这是 `DirectionalVelocityCommand` 里已实现好的机制。

## ② 数据集：2 clip → **5 clip**（每个模态至少 1 个）

见 `motions/amp_walk/amp_walk_v13_motion_selection.yaml`：直行 + 弧线 + 侧移×2 + 后退。
⚠️ 用户提到的 `walk_backward_loop_003__A143` 在本仓库不存在（原始 csv/npz 都没有），
按用户指示改用 `walk_backward_loop_001__A022`（0.55 m/s、步姿正常）。

## ③ 奖励/算法：**逐字沿用 V12**（含 2 帧判别器、`track_default_pose_on_stand=−1`）

**一项都不动**，所以 V13 的变量只有"指令集 + 数据集"两件套（它们必须同时改：
指令要的模态若没有参考密度，判别器会把合法姿态判成"不像参考"而抑制 —— 这正是
V10b 文档里警告过的"任务要的 vs 风格要的打架")。

侧移**交叉跨步**与两项横向奖励不冲突：`feet_crossing_cost` / `feet_lateral_width_cost`
都带 `gate_lateral`（`w_dir=|vx|/‖v‖`）⇒ **纯侧移时门控为 0**（原设计就是为了保护侧步）。
"""

from __future__ import annotations

import os

from isaaclab.utils import configclass

from .amp_walk_env_cfg import MOTION_NAME
from .amp_walk_v12_env_cfg import (
    G1AmpWalkV12EnvCfg,
    G1AmpWalkV12EnvCfg_PLAY,
    G1AmpWalkV12TeacherEnv,
)

AMP_WALK_V13_RUN_NAME = "G1AmpWalkV13MVPPlane_obs378_rew21_amp2frame_cmd4_clip5_omnidir"

#: 5-clip 选择文件（直行 + 弧线 + 侧移×2 + 后退）
AMP_WALK_V13_MOTION_SELECTION_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "motions",
    "amp_walk",
    "amp_walk_v13_motion_selection.yaml",
)

#: 指令范围：`vx∈(−0.5,1.0)`（前后）· `vy∈(−0.5,0.5)`（侧向）· `ωz∈(−0.8,0.8)`（转向）
AMP_WALK_V13_COMMAND_RANGES = ((-0.5, 1.0), (-0.5, 0.5), (-0.8, 0.8))

#: ① 静止/低速档（用户："0.1 m/s 内的静止指令"）
AMP_WALK_V13_PROB_LOW = 0.20
AMP_WALK_V13_LOW_VEL = 0.1
AMP_WALK_V13_ZERO_FRAC_IN_LOW = 0.3
#: ② 前后向档（`vy=0`、`ωz=0`）：纯侧向之外都归它 + `prob_back`
AMP_WALK_V13_PROB_LIN = 0.25
AMP_WALK_V13_PROB_LIN_FWD = 0.6
AMP_WALK_V13_STRAIGHT_LIN_VEL_RANGE = (0.5, 0.9)      # 贴着直行 clip 的 0.6~1.0 下沿
AMP_WALK_V13_STRAIGHT_ANG_VEL_RANGE = (0.0, 0.0)
AMP_WALK_V13_PROB_BACK = 0.10
AMP_WALK_V13_BACK_LIN_VEL_RANGE = (-0.5, -0.3)        # 后退 clip 是 0.55 m/s
AMP_WALK_V13_BACK_ANG_VEL_RANGE = (0.0, 0.0)
#: ③ 纯侧向档（`vx=0`）：新增 `side_ang_vel_range` 把 `ωz` 也钉在 0（严格"纯侧移"）
AMP_WALK_V13_PROB_SIDE = 0.25
AMP_WALK_V13_SIDE_ANG_VEL_RANGE = (0.0, 0.0)
#: ④ 斜向/转向档 = 剩余概率（= 1 − 0.20 − 0.25 − 0.25 − 0.10 = 0.20，父类均匀采样）


def _apply_v13(cfg) -> None:
    # ① 数据集：2 clip → 5 clip
    cfg.scene.motion_reference.motion_buffers[MOTION_NAME].filtered_motion_selection_filepath = (
        AMP_WALK_V13_MOTION_SELECTION_FILE
    )
    # ② 指令：四档
    cmd = cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = AMP_WALK_V13_COMMAND_RANGES[0]
    cmd.ranges.lin_vel_y = AMP_WALK_V13_COMMAND_RANGES[1]
    cmd.ranges.ang_vel_z = AMP_WALK_V13_COMMAND_RANGES[2]

    cmd.prob_low = AMP_WALK_V13_PROB_LOW
    cmd.low_vel = AMP_WALK_V13_LOW_VEL
    cmd.zero_frac_in_low = AMP_WALK_V13_ZERO_FRAC_IN_LOW

    cmd.prob_side = AMP_WALK_V13_PROB_SIDE
    cmd.side_ang_vel_range = AMP_WALK_V13_SIDE_ANG_VEL_RANGE

    cmd.prob_lin = AMP_WALK_V13_PROB_LIN
    cmd.prob_lin_fwd = AMP_WALK_V13_PROB_LIN_FWD
    cmd.straight_lin_vel_range = AMP_WALK_V13_STRAIGHT_LIN_VEL_RANGE
    cmd.straight_ang_vel_range = AMP_WALK_V13_STRAIGHT_ANG_VEL_RANGE

    cmd.prob_back = AMP_WALK_V13_PROB_BACK
    cmd.back_lin_vel_range = AMP_WALK_V13_BACK_LIN_VEL_RANGE
    cmd.back_ang_vel_range = AMP_WALK_V13_BACK_ANG_VEL_RANGE


@configclass
class G1AmpWalkV13EnvCfg(G1AmpWalkV12EnvCfg):
    """训练用：V12（2 帧判别器 + 站姿权重）之上，只改指令分档与数据集。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_v13(self)
        self.run_name = AMP_WALK_V13_RUN_NAME


@configclass
class G1AmpWalkV13EnvCfg_PLAY(G1AmpWalkV12EnvCfg_PLAY):
    """回放用：与训练同指令/同数据集/同奖励。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_v13(self)
        self.run_name = AMP_WALK_V13_RUN_NAME


class G1AmpWalkV13TeacherEnv(G1AmpWalkV12TeacherEnv):
    """契约期望值只能写在 **env 类**上（V13 的指令范围已扩到全向）。"""

    expected_command_ranges = AMP_WALK_V13_COMMAND_RANGES


__all__ = [
    "AMP_WALK_V13_BACK_ANG_VEL_RANGE",
    "AMP_WALK_V13_BACK_LIN_VEL_RANGE",
    "AMP_WALK_V13_COMMAND_RANGES",
    "AMP_WALK_V13_LOW_VEL",
    "AMP_WALK_V13_MOTION_SELECTION_FILE",
    "AMP_WALK_V13_PROB_BACK",
    "AMP_WALK_V13_PROB_LIN",
    "AMP_WALK_V13_PROB_LIN_FWD",
    "AMP_WALK_V13_PROB_LOW",
    "AMP_WALK_V13_PROB_SIDE",
    "AMP_WALK_V13_RUN_NAME",
    "AMP_WALK_V13_SIDE_ANG_VEL_RANGE",
    "AMP_WALK_V13_STRAIGHT_ANG_VEL_RANGE",
    "AMP_WALK_V13_STRAIGHT_LIN_VEL_RANGE",
    "AMP_WALK_V13_ZERO_FRAC_IN_LOW",
    "G1AmpWalkV13EnvCfg",
    "G1AmpWalkV13EnvCfg_PLAY",
    "G1AmpWalkV13TeacherEnv",
]
