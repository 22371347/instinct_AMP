"""AMP 行走教师 **v3** 配置：v2 奖励配方 + 把 `action_rate_l2` 提到 −0.02。

为什么是 v3（对 v2 退步的修正）
------------------------------
v2（12 项，与 AMP_mjlab 配方对齐）**实测全面退步**，已停训：

| 可比指标 | v1 @5000 | v2 @7330 |
|---|---|---|
| 回合长度（步） | 604.7 | 560.0 |
| 触地终止比例 | 0.5498 | **0.6772** |
| 脚距均值 | 0.3518 | **0.4616** |
| **动作噪声 σ** | 0.7557 | **1.5800** |

⭐ **首要病灶**：`Policy/mean_noise_std` **全程单调上升** `0.766 → 1.580`
（实测 `max = 1.57997`，234 个数据点）。`action_scale = 1.0` 下这是
**±90° 的关节目标噪声** —— 策略在**主动放大**探索噪声。

**机制**：`action_rate_l2 = mean((Δa)²)` 是二次型、**每一步每个关节都有梯度**的
**密集**信号；而 v2 保留的 exp 核跟踪奖励**在误差小时梯度趋近 0**。
v2 删掉 9 个密集惩罚项、又把 `action_rate_l2` 削到 1/5，等于**抽走了奖励集里
绝大部分密集梯度**。PPO 的 σ 是可训练参数，梯度信号变弱时它会**膨胀**
（方差大 → 探索多 → 梯度估计更稳），这是 PPO 的经典行为。

⇒ v3 的两处改动（**其余全部继承 v2**）：

1. **本文件**：`action_rate_l2` `−0.01 → −0.02`，把密集梯度补回来
   （不回到 v1 的 −0.05：那个确实也偏强）。
2. **agent cfg**（`agents/amp_walk_v3_ppo_cfg.py`）：`init_noise_std` `1.0 → 0.3`，
   把探索噪声起点从 **±57° 降到 ~±17°**。

⭐ **为什么 v3 从零训练而不是续训**：跳跃是**在大探索噪声下才成立的吸引子**
（支撑腿自稳、对 ±57° 目标噪声鲁棒；平滑交替步态依赖精确摆动腿时序，会被噪声摧毁）。
从 σ≈0.3 起步，策略**根本不会进入**那个吸引子；从已跳跃的 checkpoint 续训则很可能
困在局部最优里出不来。这正是「局部收敛」的机制解释。

本次**不动**的东西（刻意保持单变量归因）
--------------------------------------
- **不加**「步宽过宽」惩罚。现状：`feet_lateral_width` 只在 `sep < 0.145` 时罚
  （实测 −0.0014/s），对**过宽结构性没有约束** —— 所以脚距 0.352→0.462 这个
  **第二个病灶本轮不解决**，留待单独一轮。
- 不改速度/角速度跟踪的 `std`（仍 0.5，与学长同族教师保持形状一致）。
- 不动 `G1FlatRewardsCfg` 与 v1/v2 的任何文件。

数据集与指令分布（2026-09-25 追加，按用户要求对齐 AMP_mjlab）
------------------------------------------------------------
依据 AMP_mjlab 的实测经验：**运动数据过多会导致风格混乱**（判别器的风格分布在过大的
参考集上被摊平，反而拉不住步态）。所以：

1. **参考数据集 17 → 7 clip**：换成 `~/rsl_project/AMP_mjlab/motion_data_csv/amp_selected.txt`
   里**未注释**的 7 个（6 行走 + 1 慢跑）。选择清单：
   `motions/amp_walk/amp_walk_v3_motion_selection.yaml`（逐条对应关系与
   「用了哪个 clip 代替、为什么」都在那个文件里写了）。
2. **指令分布换成 AMP_mjlab v4 的**（`env_cfgs.py:266-290`，逐字段对照表见
   `apply_amp_walk_v3_command_ranges` 的 docstring）：
   `prob_low 0.20`（含 60% 精确 (0,0,0)）/ `prob_side 0.20` / `prob_lin 0.40`
   （其中 `prob_lin_fwd 0.7`）/ `prob_back 0.10` / 余 `0.10` 均匀斜向；
   范围 vx(−0.7, 2.0) / vy(±1.0) / wz(±1.57)；重采样 (6, 15) s。
   ⇒ 这就是「原地 / 纵向 / 横向 / 斜向 / 后退」的分档结构。

⚠️ **两项刻意不照抄 AMP_mjlab**（照抄会报错或失效，不是保守）：
`rel_standing_envs` 仍 0.0（我们的实现有 fail-fast，站立语义交给 `prob_low`）、
`heading_command` 仍 False（开了 `wz` 会被航向环重算，方向分档直接失效）。

⚠️ **一个已知的覆盖缺口**：精选 7 clip 的实测 |wz| 上限 ≈ 1.13，而指令范围是 ±1.57
（照抄 AMP_mjlab）。vx/vy 刚好覆盖，**wz 超出**。若大角速度命令下步态劣化，先怀疑这里。

接口契约**完全不变**
------------------
动作 29D / 顺序 / `scale=1.0` / default offset / `clip=None`、观测 626/99/670、
时序 50 Hz、执行器 —— 全部继承，与 `INTERFACE_CONTRACT.md` 逐项相同。
所以 checkpoint / ONNX / MuJoCo 回放 / **蒸馏 teacher 侧接口**全部不受影响。
"""

from __future__ import annotations

import copy
import os

from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .amp_walk_env_cfg import (
    G1AmpWalkCommandsCfg,
    G1AmpWalkTeacherEnv,
    MOTION_DIR,
    MOTION_NAME,
)
from .amp_walk_v2_env_cfg import (
    AMP_WALK_V2_RUN_NAME,
    G1AmpWalkV2EnvCfg,
    G1AmpWalkV2EnvCfg_PLAY,
    G1AmpWalkV2RewardsCfg,
)

#: v3 的 run_name（= 日志目录前缀）。`ar` = action_rate，`ns` = noise std。
AMP_WALK_V3_RUN_NAME = "G1AmpWalkV3Plane_obs626_rew12_ar0.02_ns0.3"

#: v3 的 AMP 参考数据集 = AMP_mjlab 精选 7 clip（6 行走 + 1 慢跑）。
#: 动机与逐条对应见该 yaml 的注释；要恢复 17 clip 只需把本常量改回
#: `MOTION_SELECTION_FILE`。
AMP_WALK_V3_MOTION_SELECTION_FILE = os.path.join(
    MOTION_DIR, "amp_walk_v3_motion_selection.yaml"
)

#: v3 的指令范围 = AMP_mjlab v4 的 `env_cfgs.py:285-290`。
AMP_WALK_V3_COMMAND_RANGES = ((-0.7, 2.0), (-1.0, 1.0), (-1.57, 1.57))


def apply_amp_walk_v3_command_ranges(cmd_cfg) -> None:
    """按 AMP_mjlab v4 写回 v3 的指令**分档分布**与**范围**。

    对照 `~/rsl_project/AMP_mjlab/src/tasks/amp_loco/config/g1/env_cfgs.py:266-290`：

    | 字段 | v1 | **v3**（= AMP_mjlab v4）|
    |---|---|---|
    | `prob_low`（含 60% 精确 (0,0,0)）| 0.15 | **0.20** |
    | `prob_side`（纯侧向 $v_x\\equiv 0$）| 0.25 | **0.20** |
    | `prob_lin`（纯前后 $v_y\\equiv 0$）| 0.25 | **0.40** |
    | `prob_lin_fwd`（其中持续直线前向占比）| 0.6 | **0.7** |
    | `prob_back`（专属持续后退）| 0.10 | 0.10 |
    | （余下）均匀斜向 | 0.25 | **0.10** |
    | `straight_lin_vel_range` | (0.5, 1.2) | **(0.4, 2.0)** |
    | `back_lin_vel_range` | (−1.0, −0.5) | **(−0.7, −0.4)** |
    | `lin_vel_x` | (−1.0, 1.2) | **(−0.7, 2.0)** |
    | `lin_vel_y` | ±0.5 | **±1.0** |
    | `ang_vel_z` | ±0.8 | **±1.57** |
    | `resampling_time_range` | (10, 10) | **(6, 15)** |

    ⇒ 分档概率和 = 0.20+0.20+0.40+0.10 = **0.90**，余 **0.10** 为均匀斜向（与
    AMP_mjlab 注释「low 0.20 + side 0.20 + lin 0.40(fwd 70%) + back 0.10 + diag 0.10」**逐位一致**）。

    ⚠️ **两项刻意不照抄**（照抄会直接报错 or 失效，不是保守）：
    - `rel_standing_envs` 仍为 **0.0**（AMP_mjlab 是 0.05）：
      我们的 `DirectionalVelocityCommand.__init__` 有 **fail-fast 断言**，
      因为站立语义已交给 `prob_low` 分支，两套机制叠加会互相打架。
    - `heading_command` 仍为 **False**（AMP_mjlab 是 True，且 `rel_heading_envs=0.25`）：
      开了之后 `wz` 会被**航向环重算**，我们手工采样的方向分档会**直接失效**。
    """
    cmd_cfg.resampling_time_range = (6.0, 15.0)
    cmd_cfg.rel_standing_envs = 0.0      # ⚠️ 不能改（有 fail-fast）
    cmd_cfg.rel_heading_envs = 0.0       # ⚠️ 同上
    cmd_cfg.heading_command = False      # ⚠️ 开了方向分档就失效
    cmd_cfg.ranges.lin_vel_x = AMP_WALK_V3_COMMAND_RANGES[0]
    cmd_cfg.ranges.lin_vel_y = AMP_WALK_V3_COMMAND_RANGES[1]
    cmd_cfg.ranges.ang_vel_z = AMP_WALK_V3_COMMAND_RANGES[2]
    cmd_cfg.ranges.heading = (-1.57, 1.57)

    # 分档概率（和 = 0.90，余 0.10 为均匀斜向）
    cmd_cfg.prob_low = 0.20
    cmd_cfg.low_vel = 0.1
    cmd_cfg.zero_frac_in_low = 0.6      # 低速档里 60% 给精确 (0,0,0)
    cmd_cfg.prob_side = 0.20
    cmd_cfg.prob_lin = 0.40
    cmd_cfg.prob_lin_fwd = 0.7
    cmd_cfg.straight_lin_vel_range = (0.4, 2.0)
    cmd_cfg.straight_ang_vel_range = (-0.1, 0.1)
    cmd_cfg.prob_back = 0.10
    cmd_cfg.back_lin_vel_range = (-0.7, -0.4)
    cmd_cfg.back_ang_vel_range = (-0.1, 0.1)


def _apply_v3_motion_selection(env_cfg) -> None:
    """把参考动作的「选择清单」换成 v3 的精選 7 clip。

    ⚠️ **必须 deepcopy**：`motion_reference.motion_buffers` 是 dict，而
    `@configclass` 的默认值在多实例间可能共享 → 直接改会**连带改到 v1/v2 的配置**。
    """
    env_cfg.scene.motion_reference = copy.deepcopy(env_cfg.scene.motion_reference)
    env_cfg.scene.motion_reference.motion_buffers[
        MOTION_NAME
    ].filtered_motion_selection_filepath = AMP_WALK_V3_MOTION_SELECTION_FILE


@configclass
class G1AmpWalkV3CommandsCfg(G1AmpWalkCommandsCfg):
    """v3 的指令配置：继承 v1 的类，只改分档概率与范围（见 `apply_amp_walk_v3_command_ranges`）。"""

    def __post_init__(self):
        super().__post_init__()
        apply_amp_walk_v3_command_ranges(self.base_velocity)

#: v3 相对 v2 的改动（唯一的事实来源，供日志/文档引用）
AMP_WALK_V3_CHANGES: tuple[tuple[str, float, float], ...] = (
    ("rewards.action_rate_l2", -0.01, -0.02),
    ("policy.init_noise_std", 1.0, 0.3),
    ("algorithm.clip_min_std", 0.05, 0.015),
)


@configclass
class G1AmpWalkV3RewardsCfg(G1AmpWalkV2RewardsCfg):
    """v2 的 12 项配方，只把 `action_rate_l2` 从 −0.01 提到 −0.02。

    `func` 逐字沿用基类，只动 `weight`。动机见模块 docstring。
    """

    def __post_init__(self):
        super().__post_init__()
        # 补回密集梯度：v1 用 −0.05 时 σ 稳在 0.756；v2 削到 −0.01 后 σ 涨到 1.580。
        # 取中间值 −0.02，兼顾「足够密集的信号」与「不过度惩罚动作变化」。
        self.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)


@configclass
class G1AmpWalkV3EnvCfg(G1AmpWalkV2EnvCfg):
    """训练用：场景 / 观测 / 事件 / 终止 / 时序 / 动作 **全部继承 v2**。

    本类替换三样：
    1. `rewards` —— v2 的 12 项配方，`action_rate_l2` 提到 −0.02；
    2. `commands` —— 指令分布换成 AMP_mjlab v4 的（含精选 7-clip 数据集）；
    3. 参考动作的选择清单 —— 17 clip → 精选 7 clip（在 `__post_init__` 里 deepcopy 后改）。
    """

    rewards: G1AmpWalkV3RewardsCfg = G1AmpWalkV3RewardsCfg()
    commands: G1AmpWalkV3CommandsCfg = G1AmpWalkV3CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        # ⚠️ 父类 G1AmpWalkEnvCfg.__post_init__ 会把 ranges 写回 v1 的值，
        #    所以必须在它之后再写一次 v3 的。
        apply_amp_walk_v3_command_ranges(self.commands.base_velocity)
        _apply_v3_motion_selection(self)
        self.run_name = AMP_WALK_V3_RUN_NAME


@configclass
class G1AmpWalkV3EnvCfg_PLAY(G1AmpWalkV2EnvCfg_PLAY):
    """Play 用：继承 v2 PLAY 的全部差异，同样替换 rewards / commands / 数据集。"""

    rewards: G1AmpWalkV3RewardsCfg = G1AmpWalkV3RewardsCfg()
    commands: G1AmpWalkV3CommandsCfg = G1AmpWalkV3CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_amp_walk_v3_command_ranges(self.commands.base_velocity)
        _apply_v3_motion_selection(self)
        self.run_name = AMP_WALK_V3_RUN_NAME


class G1AmpWalkV3TeacherEnv(G1AmpWalkTeacherEnv):
    """v3 的契约校验器：只把「指令范围」与「指令保持时长」的**期望值**改成 v3 的。

    其余全部与 v1 逐字相同：动作空间 / joint 顺序 / `scale` / 时序 / 执行器 /
    default offset / 观测 626 / 高程图 527 射线 / 站立比 / 不允许课程。

    ⚠️ 为什么必须有这个子类：基类把许多期望值放在类属性里（所以子类只需覆盖元组），
    但「指令保持时长」原先**写死在方法体里**（`omni_loco_env_cfg.py` 的
    `_validate_teacher_contract`）。已把它改成类属性 `expected_command_hold`
    （默认值不变，对 OmniLoco/v1/v2 行为完全不变），这里就能正常声明 v3 的值。
    """

    expected_command_ranges = AMP_WALK_V3_COMMAND_RANGES
    expected_command_hold = (6.0, 15.0)


__all__ = [
    "AMP_WALK_V2_RUN_NAME",
    "AMP_WALK_V3_CHANGES",
    "AMP_WALK_V3_COMMAND_RANGES",
    "AMP_WALK_V3_MOTION_SELECTION_FILE",
    "AMP_WALK_V3_RUN_NAME",
    "G1AmpWalkV3CommandsCfg",
    "G1AmpWalkV3EnvCfg",
    "G1AmpWalkV3EnvCfg_PLAY",
    "G1AmpWalkV3RewardsCfg",
    "G1AmpWalkV3TeacherEnv",
    "apply_amp_walk_v3_command_ranges",
]
