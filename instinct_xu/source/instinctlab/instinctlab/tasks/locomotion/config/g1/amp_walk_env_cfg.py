"""AMP 行走专家策略任务（G1 29-DoF，privileged teacher）。

相对 OmniLoco 的**唯一接口差异**
--------------------------------
actor 观测在 623 维基础上 **增加 3 维 `base_lin_vel`** → **626 维**。

其余全部继承、不改动：

| 项 | 值 | 来源 |
|---|---|---|
| 动作 | 29 维位置控制，`scale=1.0`，`use_default_offset=True` | `G1OmniLocoActionsCfg` |
| 时序 | `dt=0.005`，`decimation=4`，`step_dt=0.02` | `G1FlatEnvCfg.__post_init__` |
| 高程图 | 527 射线（31×17，3.0×1.6 m，res 0.1） | `G1OmniLocoSceneCfg` |
| 执行器 | `beyondmimic_g1_29dof_actuators` | `G1OmniLocoEnvCfg.__post_init__` |
| 奖励 / 事件 / 终止 | 与 OmniLoco 相同 | `G1FlatRewardsCfg` 等 |
| Critic 观测 | 含 `base_lin_vel`（本就对称） | `G1FlatCriticObsCfg` |
| 速度指令 | **方向分档**（`DirectionalVelocityCommand`） | `tasks/locomotion/mdp/commands.py` |

为什么 actor 要加 `base_lin_vel`
--------------------------------
本任务是**特权（privileged）教师**，允许使用真值量测；`base_lin_vel` 在真机上不可直接观测，
但在仿真里可精确获得。加上它后：
1. 与 `AMP_mjlab` 的 actor 输入对齐（其 actor 也含 base 线速度）；
2. 为后续蒸馏到学生策略提供明确的“特权信息边界”；
3. `G1OmniLocoEnvCfg` 的 fail-fast 契约校验**只允许 623 维**，因此必须新建任务
   （`G1OmniLocoTeacherEnv._validate_teacher_contract` 会直接 `RuntimeError`）。

本文件对应 `PLAN_amp_walk.md` §6 的**步骤②③**：
- ② 任务骨架（actor obs 626），
- ③ 速度指令换成方向分档（否则「纯侧向」命令采不出来，见 `Plan` §3.5）。

**尚未接入 AMP**（步骤⑥）。
"""

from __future__ import annotations

import math

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.locomotion.mdp as locomotion_mdp

from .flat_env_cfg import G1FlatCriticObsCfg
from .omni_loco_env_cfg import (
    G1OmniLocoCommandsCfg,
    G1OmniLocoEnvCfg,
    G1OmniLocoSceneCfg,
    G1OmniLocoTeacherEnv,
)

# ============================================================================
# 冻结的接口契约常量
# ============================================================================

AMP_WALK_TASK_ID = "Instinct-Amp-Walk-G1-v0"

#: actor（= 教师）观测项顺序。**顺序即契约**，与 AMP_WALK_OBSERVATION_DIMS 一一对应。
AMP_WALK_OBSERVATION_TERMS = (
    "height_scan",          # 527  —— 局部高程图（特权）
    "velocity_command",     #   3  —— (vx, vy, wz) 指令
    "base_lin_vel",         #   3  —— ★ 本任务新增：base 线速度（特权）
    "projected_gravity",    #   3
    "base_ang_vel",         #   3
    "joint_pos_rel",        #  29
    "joint_vel",            #  29
    "last_action",          #  29
)
AMP_WALK_OBSERVATION_DIMS = (527, 3, 3, 3, 3, 29, 29, 29)
AMP_WALK_OBSERVATION_SIZE = sum(AMP_WALK_OBSERVATION_DIMS)     # == 626
AMP_WALK_SCAN_RAYS = 527

#: 速度指令范围 (vx, vy, wz)。上限取自 PLAN §3.4 实测的数据集包络并留余量：
#: 数据集 vx∈[-1.07,1.69]、vy∈[-0.48,0.55]、wz∈[-44.2°,57.0°]。
#: **不得超过包络**，否则该档位缺乏参考密度（判别器会抑制对应姿态）。
AMP_WALK_COMMAND_RANGES = ((-1.0, 1.2), (-0.5, 0.5), (-0.8, 0.8))


def apply_amp_walk_command_ranges(cmd_cfg) -> None:
    """把速度指令范围与方向分档参数写回命令配置。

    抽成函数是因为命令配置有**两个**构造入口（`G1AmpWalkCommandsCfg.__post_init__`
    与 `G1AmpWalkEnvCfg.__post_init__`），而父类两处都会把 ranges 覆盖成 OmniLoco 的。
    集中一处可保证无论哪个后执行，结果都一致。
    """
    cmd_cfg.resampling_time_range = (10.0, 10.0)
    cmd_cfg.rel_standing_envs = 0.0      # 站立语义交给 prob_low 分支
    cmd_cfg.rel_heading_envs = 0.0
    cmd_cfg.heading_command = False      # 否则 wz 会被航向环重算，采样失效
    cmd_cfg.ranges.lin_vel_x = AMP_WALK_COMMAND_RANGES[0]
    cmd_cfg.ranges.lin_vel_y = AMP_WALK_COMMAND_RANGES[1]
    cmd_cfg.ranges.ang_vel_z = AMP_WALK_COMMAND_RANGES[2]
    cmd_cfg.ranges.heading = (-math.pi, math.pi)


# ============================================================================
# 速度指令配置（步骤③）
# ============================================================================


@configclass
class G1AmpWalkCommandsCfg(G1OmniLocoCommandsCfg):
    """用方向分档指令替换 OmniLoco 的 `UniformVelocityCommand`。

    动机见 `tasks/locomotion/mdp/commands.py` 与 PLAN §3.5：三轴独立均匀采样下
    $v_x$ 恰为 0 的概率为零 ⇒ 学不出「左右直移」。
    """

    base_velocity = locomotion_mdp.DirectionalVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=locomotion_mdp.DirectionalVelocityCommandCfg.Ranges(
            lin_vel_x=AMP_WALK_COMMAND_RANGES[0],
            lin_vel_y=AMP_WALK_COMMAND_RANGES[1],
            ang_vel_z=AMP_WALK_COMMAND_RANGES[2],
            heading=(-math.pi, math.pi),
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        apply_amp_walk_command_ranges(self.base_velocity)


# ============================================================================
# 观测配置
# ============================================================================


@configclass
class G1AmpWalkPolicyObsCfg(ObsGroup):
    """actor 特权观测契约：527 + 3 + 3 + 3 + 3 + 29 + 29 + 29 = **626**。

    与 `LocoTeacherPolicyObsCfg`（623）逐项一致，仅在 `velocity_command` 之后插入
    `base_lin_vel`。

    噪声策略：`height_scan` / `velocity_command` / `base_lin_vel` / `last_action`
    **不加噪**；`projected_gravity` / `base_ang_vel` / `joint_pos_rel` / `joint_vel`
    沿用 OmniLoco 的噪声幅度（保持与学长基线可比）。
    `base_lin_vel` 是特权量、且 Critic 组本就不加噪，故此处同样不加噪。
    """

    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )
    velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    # NOTE: `instinct_mdp` 并没有转发 `base_lin_vel`（见 envs/mdp/observations/expanded.py
    # 的 import 列表），必须从 isaaclab 的 mdp 取——与 G1FlatCriticObsCfg 的写法一致。
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    projected_gravity = ObsTerm(
        func=instinct_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    base_ang_vel = ObsTerm(func=instinct_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
    last_action = ObsTerm(func=instinct_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class G1AmpWalkObservationsCfg:
    """actor = 626 维特权观测；critic 沿用 flat 的（本就含 base_lin_vel）。"""

    policy: G1AmpWalkPolicyObsCfg = G1AmpWalkPolicyObsCfg()
    critic: G1FlatCriticObsCfg = G1FlatCriticObsCfg()


# ============================================================================
# 环境配置
# ============================================================================


@configclass
class G1AmpWalkEnvCfg(G1OmniLocoEnvCfg):
    """AMP 行走教师训练配置。

    场景 / 动作 / 奖励 / 事件 / 终止 / 时序 **全部继承 OmniLoco**（= flat 基线），
    只替换观测组与速度指令项。
    """

    commands: G1AmpWalkCommandsCfg = G1AmpWalkCommandsCfg()
    observations: G1AmpWalkObservationsCfg = G1AmpWalkObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # ⚠️ 父类 G1OmniLocoEnvCfg.__post_init__ 会把命令 ranges 重新设成 OmniLoco 的值，
        #    因此必须在其之后“再”写回我们的方向分档参数。
        apply_amp_walk_command_ranges(self.commands.base_velocity)
        # 父类 __post_init__ 已经把 run_name 设成 OmniLoco 的，这里必须覆盖。
        self.run_name = "G1AmpWalkPlane_obs626_scale1"


@configclass
class G1AmpWalkEnvCfg_PLAY(G1AmpWalkEnvCfg):
    scene: G1OmniLocoSceneCfg = G1OmniLocoSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# ============================================================================
# 环境类：复用学长那套 fail-fast 契约校验，只改期望值
# ============================================================================


class G1AmpWalkTeacherEnv(G1OmniLocoTeacherEnv):
    """把 OmniLoco 的契约校验器「改参数复用」。

    `_validate_teacher_contract` 的期望值全部是**类属性**，所以子类只需覆盖元组即可
    获得同样的严格检查（动作空间、joint 顺序、scale、时序、执行器、命令范围、
    default offset 语义探针……）。
    """

    expected_policy_terms = AMP_WALK_OBSERVATION_TERMS
    expected_policy_dims = AMP_WALK_OBSERVATION_DIMS
    expected_scan_rays = AMP_WALK_SCAN_RAYS
    # 步骤③ 起用方向分档指令，命令范围不再是 OmniLoco 的，必须同步期望值
    expected_command_ranges = AMP_WALK_COMMAND_RANGES
    allow_curriculum = False

    def _validate_teacher_contract(self):
        super()._validate_teacher_contract()
        print(
            f"[AMP-WALK PASS] policy={sum(self.expected_policy_dims)} "
            f"(height_scan={self.expected_scan_rays} + base_lin_vel), action=29, scale=1.0, "
            f"cmd_ranges={self.expected_command_ranges} (directional)"
        )
