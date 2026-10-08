"""V10b：在 V10 基础上按用户指令**收敛数据集 + 明确指令分档**，并装上"门控口径"探针。

## 相对 V10 的两处实质改动

### ① 数据集：17 个 clip → **2 个**（用户指定）

⚠️ **重要发现**：仓库里实际生效的是 `amp_walk_motion_selection.yaml` —— **17 个 clip**
（`amp_walk_env_cfg.py::MOTION_SELECTION_FILE` 指向它，而且**没有任何版本 cfg 覆盖过它**），
但 v9 MVP 的文档却写着"只留 1 个 clip" ⇒ **文档与实际长期不符**。
17 个里混着后退×2、侧移×2、原地转身×2、慢跑×3、弧线×2、步态切换×1
⇒ 判别器的"专家分布"被摊成 17 种形态，**"走"这一模态密度很稀**
⇒ 与实测"gap 稳定 ≈1.1、但 `d_actor` 卡死在 −0.56 不动"的现象吻合。

本版 = 用户指定的 **2 个同族"走"clip**：`walk_forward_loop_002__A022`（直行）
+ `arc_walk_left_loop_001__A029`（弧线/转向）。见 `motions/amp_walk/amp_walk_v10_motion_selection.yaml`。

### ② 指令分档：明确成"静止 / 前进 / 斜向"三档（用户指定）

| 档 | 概率 | 采样 |
|---|---|---|
| **低速（静止）** | `prob_low=0.30`（其中 30% 精确 (0,0,0)） | `‖v‖ ≤ 0.1`、`wz=0` |
| **纯前进** | `prob_lin=0.35`（`prob_lin_fwd=1.0`） | `vx ∈ (0.4, 0.7)`、`vy=0`、`wz=0` |
| **前向斜向/弧线** | 余量 `0.35`（父类均匀斜向） | `vx ∈ (0,0.7)`、`vy ∈ ±0.35`、`wz ∈ ±0.4` |

- **删掉** `prob_side`（纯侧向）与 `prob_back`（后退）：用户要的指令集里没有它们，
  而参考集也没有侧移/后退 clip ⇒ 留着只会让"任务要的"和"风格要的"打架。
- 全部取值都在数据集包络内（vy∈[−0.48,0.55]、wz∈[−44°,57°]）。

## ③ 同时装上"门控口径"探针（回答"AMP 静止不参考门控是否正确"）

`disable_style_on_stand` 的门控读的是**策略观测里的 `velocity_command` 切片**，而不是
`command_manager`。**实测 `amp_stand_frac ≈ 0.049`，而按分档设计应为 ≈0.40**
（0.30 低速 + 0.70×0.143 前向低速段），**三个 run 都一致** ⇒ 必须证明切片对不对：

| 探针 | 位置 | 读的量 |
|---|---|---|
| `Mean step cmd_slice_vx/vy/wz`、`cmd_slice_speed` | **桥接层**（切片） | `_command_slice` 指到的那 3 个数 |
| `Step/cmd_speed_mean`、`cmd_low_frac_true` | **环境层**（真值） | `command_manager.get_command()` |

⇒ 若 `cmd_slice_*` 与真值均值吻合 ⇒ 门控实现正确（差异只可能来自"低概率档没按设计采样"）；
若明显不吻合 ⇒ `_command_slice` 错位，必须修（同时会解释 `action_abs_max` 曾出现 11.8 rad 的怪值）。
"""

from __future__ import annotations

import os

from isaaclab.utils import configclass

from .amp_walk_env_cfg import MOTION_NAME
from .amp_walk_v9_env_cfg import G1AmpWalkV9TeacherEnv
from .amp_walk_v10_env_cfg import G1AmpWalkV10EnvCfg, G1AmpWalkV10EnvCfg_PLAY

AMP_WALK_V10B_RUN_NAME = "G1AmpWalkV10bMVPPlane_obs378_rew21_explore_aug_cmd3_clip2"

#: 2-clip 选择文件（直行 + 弧线）
AMP_WALK_V10B_MOTION_SELECTION_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "motions", "amp_walk", "amp_walk_v10_motion_selection.yaml"
)

#: 指令范围：前向 + 侧向(±) + 转向(±)（与前向斜向档配套）
AMP_WALK_V10B_COMMAND_RANGES = ((0.0, 0.7), (-0.35, 0.35), (-0.4, 0.4))
#: 分档概率：低速 0.30 / 纯前进 0.35 / 余 0.35 为均匀斜向；无纯侧向、无后退
AMP_WALK_V10B_PROB_LOW = 0.30
AMP_WALK_V10B_PROB_LIN = 0.35
AMP_WALK_V10B_PROB_LIN_FWD = 1.0
AMP_WALK_V10B_ZERO_FRAC_IN_LOW = 0.3
#: 纯前进档的巡航速度（贴着 walk clip 的速度带 0.6~1.0 m/s 的下沿，且不超过指令上限）
AMP_WALK_V10B_STRAIGHT_LIN_VEL_RANGE = (0.4, 0.7)


def _apply_v10b(cfg) -> None:
    # ① 数据集收敛到 2 个 clip
    cfg.scene.motion_reference.motion_buffers[MOTION_NAME].filtered_motion_selection_filepath = (
        AMP_WALK_V10B_MOTION_SELECTION_FILE
    )
    # ② 指令分档：静止 / 前进 / 斜向
    cmd = cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = AMP_WALK_V10B_COMMAND_RANGES[0]
    cmd.ranges.lin_vel_y = AMP_WALK_V10B_COMMAND_RANGES[1]
    cmd.ranges.ang_vel_z = AMP_WALK_V10B_COMMAND_RANGES[2]
    cmd.prob_low = AMP_WALK_V10B_PROB_LOW
    cmd.low_vel = 0.1
    cmd.zero_frac_in_low = AMP_WALK_V10B_ZERO_FRAC_IN_LOW
    cmd.prob_side = 0.0
    cmd.prob_lin = AMP_WALK_V10B_PROB_LIN
    cmd.prob_lin_fwd = AMP_WALK_V10B_PROB_LIN_FWD
    cmd.prob_back = 0.0
    cmd.straight_lin_vel_range = AMP_WALK_V10B_STRAIGHT_LIN_VEL_RANGE
    cmd.straight_ang_vel_range = (0.0, 0.0)
    cmd.back_lin_vel_range = (0.0, 0.0)
    cmd.back_ang_vel_range = (0.0, 0.0)


@configclass
class G1AmpWalkV10bEnvCfg(G1AmpWalkV10EnvCfg):
    """训练用：V10（探索/增广/交替约束）+ 2-clip 数据集 + 三档指令。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_v10b(self)
        self.run_name = AMP_WALK_V10B_RUN_NAME


@configclass
class G1AmpWalkV10bEnvCfg_PLAY(G1AmpWalkV10EnvCfg_PLAY):
    """回放用：与训练同数据集/同指令分布。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_v10b(self)
        self.run_name = AMP_WALK_V10B_RUN_NAME


class G1AmpWalkV10bTeacherEnv(G1AmpWalkV9TeacherEnv):
    """⚠️ 契约期望值只能写在 **env 类**上（cfg 上的同名属性对 `_validate_teacher_contract` 无效）。"""

    expected_command_ranges = AMP_WALK_V10B_COMMAND_RANGES


__all__ = [
    "AMP_WALK_V10B_COMMAND_RANGES",
    "AMP_WALK_V10B_MOTION_SELECTION_FILE",
    "AMP_WALK_V10B_PROB_LIN",
    "AMP_WALK_V10B_PROB_LOW",
    "AMP_WALK_V10B_RUN_NAME",
    "AMP_WALK_V10B_STRAIGHT_LIN_VEL_RANGE",
    "AMP_WALK_V10B_ZERO_FRAC_IN_LOW",
    "G1AmpWalkV10bEnvCfg",
    "G1AmpWalkV10bEnvCfg_PLAY",
    "G1AmpWalkV10bTeacherEnv",
]
