"""V10b：在 V10 基础上按用户指令**收敛数据集（17 → 2 clip）+ 明确指令分档（三档）**，并装上"门控口径"探针。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— 数据集与指令分档均已被 V18 覆盖**（见下表）。
* 仍"活着"的两条**事实/方法**：① 本层查实"**仓库里实际生效的参考集与文档长期不符**"
  （`amp_walk_env_cfg.py::MOTION_SELECTION_FILE` 指向 17-clip 的 `amp_walk_motion_selection.yaml`，
  且当时**没有任何版本覆盖过它**）⇒ 读"参考集是几个 clip"**必须看实际覆盖链，不能信文档**；
  ② 门控口径"桥接层切片 vs 环境层真值"的只读探针做法。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | 数据集 **17 clip → 2 clip**（`walk_forward_loop_002__A022` 直行 + `arc_walk_left_loop_001__A029` 弧线 → `amp_walk_v10_motion_selection.yaml`）| ❌ **已被 V18 覆盖** → **5 clip**（`amp_walk_v13_motion_selection.yaml`：直行 / 弧线 / 左·右侧移 / 后退）|
| ② | 指令分档明确为**三档**（低速 `prob_low=0.30`（其中 30% 精确 (0,0,0)）/ 纯前进 `prob_lin=0.35`（`fwd=1.0`）/ 余 0.35 前向斜向；范围 `(0,0.7) / ±0.35 / ±0.4`），并**删掉** `prob_side` / `prob_back`（置 0）| ❌ **已被 V18 覆盖** → 五档 · 范围 `((-0.5,1.0),(-0.5,0.5),(-0.8,0.8))`；**纯侧向 / 后退已重新加回**（V13/V18 `prob_side=0.25`、V18 `prob_back=0.10`）|
| ③ | 门控口径探针（`Mean step cmd_slice_vx/vy/wz`、`cmd_slice_speed` vs `Step/cmd_speed_mean`、`cmd_low_frac_true`）| 代码侧；本层提出的疑点**至今未闭环** → `docs/amp_walk/EXPERIMENTS.md` §附待办 **#7** |

## 推理与判读（已归档，勿在本文件重复）

* ⭐ 数据集侧的判据：17 个 clip 混着后退 ×2 / 侧移 ×2 / 原地转身 ×2 / 慢跑 ×3 / 弧线 ×2 / 步态切换 ×1
  ⇒ 专家分布被摊成 17 种形态、"走"这一模态密度很稀 ⇒ 与实测"gap 稳定 ≈1.1、但 `d_actor` 卡死在 −0.56
  不动"吻合 → 见该版 run 日志；其一般化结论（**数据集覆盖面决定 AMP 会压制什么**）→ §D1
* ⚠️ 删 `prob_side` / `prob_back` 的理由（"参考集没有侧移 / 后退 clip ⇒ 留着只会让任务要的和风格要的打架"）
  **已被 V18 推翻**（改用 5 clip 覆盖全向）→ §D1
* 门控疑点：`amp_stand_frac` 实测 ≈ **0.049**，而按分档设计应 ≈ 0.40（三个 run 都一致）
  ⇒ "低概率档没按设计采样"或"门控读的量不对"；**未查** → §附待办 **#7**
* 包络自检：全部指令取值都在数据集包络内（`vy ∈ [−0.48,0.55]`、`wz ∈ [−44°,57°]`）→ 该版 run 日志

⚠️ 其余逐字不动：V10 的探索 / 增广 / 交替约束、21 项奖励结构、观测口径。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V10b-G1-v0`
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
