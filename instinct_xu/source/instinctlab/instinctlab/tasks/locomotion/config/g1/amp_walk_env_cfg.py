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
import os

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.locomotion.mdp as locomotion_mdp
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear

from .flat_env_cfg import G1FlatCriticObsCfg, G1_CFG
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
# 参考动作传感器（步骤④）
# ============================================================================

MOTION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions", "amp_walk")
MOTION_SELECTION_FILE = os.path.join(MOTION_DIR, "amp_walk_motion_selection.yaml")
MOTION_NAME = "amp_walk"

#: AMP 判别器的状态历史长度（与官方 parkour 实现一致）。
AMP_STATE_HISTORY = 10
#: 判别器状态向量的**项名**。`amp_policy` 与 `amp_reference` 必须逐项同构。
AMP_STATE_TERMS = (
    "projected_gravity",   # (10, 3) —— 未 flatten
    "joint_pos_rel",       # (290,)
    "joint_vel_rel",       # (290,)
    "base_lin_vel",        # (30,)
    "base_ang_vel",        # (30,)
)
AMP_STATE_DIMS = ((AMP_STATE_HISTORY, 3), (290,), (290,), (30,), (30,))
AMP_STATE_SIZE = sum(math.prod(d) for d in AMP_STATE_DIMS)     # == 670


@configclass
class AmpWalkMotionCfg(AmassMotionCfgBase):
    """AMP 行走参考动作（① 转换产出的 instinctlab 格式 npz）。

    ⚠️ 行走是**循环步态**，两项设置很关键：
    - `motion_start_from_middle_range = (0.0, 0.0)`：**关闭 RSI 随机起点**。
      随机起点会破坏步态相位连续性，对循环行走有害（官方 Vault 的 PLAY 配置
      也正是这么关掉的）。
    - `motion_bin_length_s = None`：不切 bin，整段连续播放。
    """

    path = MOTION_DIR
    retargetting_func = None
    filtered_motion_selection_filepath = MOTION_SELECTION_FILE
    supported_file_endings = ["retargeted.npz"]
    motion_start_from_middle_range = (0.0, 0.0)
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    motion_target_framerate = 50.0
    # 参考集里既有前向也有后退 clip，故用 frontbackward 估计速度
    velocity_estimation_method = "frontbackward"
    motion_bin_length_s = None


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    robot_model_path=G1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/torso_link",
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_roll_link",
        "right_wrist_roll_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    # 不做镜像增广：本任务的转身行为对称性未被验证，增广可能引入错误标签
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    frame_interval_s=0.0,
    update_period=0.02,
    num_frames=1,
    visualizing_robot_offset=(0.0, 2.0, 0.0),
    visualizing_robot_from="reference_frame",
    motion_buffers={
        MOTION_NAME: AmpWalkMotionCfg(),
    },
    mp_split_method="Even",
)


@configclass
class G1AmpWalkSceneCfg(G1OmniLocoSceneCfg):
    """OmniLoco 平面场景 + 参考机器人 + 运动参考传感器。

    继承了 527 射线的高程图，并新增：
    - `robot_reference`：可视化用的参考（真值）机器人，由传感器驱动；
    - `motion_reference`：从 npz 提供参考状态（`*_reference_as_state` 观测项的来源）。

    ⚠️ 参考机器人不参与碰撞/接触（`contact_forces` 的 prim_path 是
    `{ENV_REGEX_NS}/Robot/.*`，不会匹配到 `RobotReference`）。
    """

    robot_reference = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference")
    motion_reference: MotionReferenceManagerCfg = motion_reference_cfg


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
class AmpPolicyStateObsCfg(ObsGroup):
    """判别器输入（**策略侧**）：真实机器人的状态（机器人本体系）。

    与 `AmpReferenceStateObsCfg` **逐项同构**（项名 / 形状 / 缩放都一致），
    否则 WASABI 在 `init_storage` 会因为两侧 `get_subobs_size` 不等而断言失败。

    设计来源：上游 parkour（Hiking in the Wild）官方实现
    `instinct/repos/InstinctLab/.../tasks/parkour/config/parkour_env_cfg.py`。
    状态向量 = 投影重力 + 关节位置（相对 default）+ 关节速度（相对 default，0.05 缩放）
    + base 线速度 + base 角速度，各 10 帧历史。

    ⚠️ 两个 mdp 模块的**分工必须分清**（很容易写错，`check_amp_walk_task.py` 会查）：
    - 机器人自身状态：`base_lin_vel` / `joint_vel_rel` **只能**从 `isaaclab.envs.mdp` 取
      （`instinctlab.envs.mdp` 并没有转发这两个）；
    - `projected_gravity` / `base_ang_vel` / `joint_pos_rel` 则两个模块都有，
      这里沿用 OmniLoco 的写法（`instinct_mdp`）。

    注意：**保持默认 `enable_corruption=False`** —— 判别器输入不应加噪声。
    """

    concatenate_terms = False

    projected_gravity = ObsTerm(
        func=instinct_mdp.projected_gravity,
        history_length=AMP_STATE_HISTORY,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_pos_rel = ObsTerm(
        func=mdp.joint_pos_rel,
        history_length=AMP_STATE_HISTORY,
        flatten_history_dim=True,
        params={"asset_cfg": SceneEntityCfg(name="robot", preserve_order=True)},
    )
    joint_vel_rel = ObsTerm(
        func=mdp.joint_vel_rel,
        scale=0.05,
        history_length=AMP_STATE_HISTORY,
        flatten_history_dim=True,
        params={"asset_cfg": SceneEntityCfg(name="robot", preserve_order=True)},
    )
    base_lin_vel = ObsTerm(
        func=mdp.base_lin_vel,
        history_length=AMP_STATE_HISTORY,
        flatten_history_dim=True,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_ang_vel = ObsTerm(
        func=instinct_mdp.base_ang_vel,
        history_length=AMP_STATE_HISTORY,
        flatten_history_dim=True,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class AmpReferenceStateObsCfg(ObsGroup):
    """判别器输入（**参考侧**）：来自 `motion_reference` 的运动参考状态。

    与 `AmpPolicyStateObsCfg` 项名、形状、缩放逐项一致，只把数据源
    从 `robot` 换成 `motion_reference`。

    ⚠️ `*_reference_as_state` 系列定义在 **`instinctlab.envs.mdp`**
    （`envs/mdp/observations/reference_as_state.py`），**不在** `isaaclab.envs.mdp`。
    注意 `projected_gravity_reference_as_state` 是个 `ManagerTermBase` 子类
    （不是普通函数），IsaacLab 会自动实例化。
    """

    concatenate_terms = False

    projected_gravity = ObsTerm(
        func=instinct_mdp.projected_gravity_reference_as_state,
        history_length=AMP_STATE_HISTORY,
        params={"asset_cfg": SceneEntityCfg("motion_reference")},
    )
    joint_pos_rel = ObsTerm(
        func=instinct_mdp.joint_pos_rel_reference_as_state,
        history_length=AMP_STATE_HISTORY,
        flatten_history_dim=True,
        params={"asset_cfg": SceneEntityCfg("motion_reference")},
    )
    joint_vel_rel = ObsTerm(
        func=instinct_mdp.joint_vel_rel_reference_as_state,
        scale=0.05,
        history_length=AMP_STATE_HISTORY,
        flatten_history_dim=True,
        params={"asset_cfg": SceneEntityCfg("motion_reference")},
    )
    base_lin_vel = ObsTerm(
        func=instinct_mdp.base_lin_vel_reference_as_state,
        history_length=AMP_STATE_HISTORY,
        flatten_history_dim=True,
        params={"asset_cfg": SceneEntityCfg("motion_reference")},
    )
    base_ang_vel = ObsTerm(
        func=instinct_mdp.base_ang_vel_reference_as_state,
        history_length=AMP_STATE_HISTORY,
        flatten_history_dim=True,
        params={"asset_cfg": SceneEntityCfg("motion_reference")},
    )


@configclass
class G1AmpWalkObservationsCfg:
    """actor = 626 维特权观测；critic 沿用 flat 的（本就含 base_lin_vel）。

    `amp_policy` / `amp_reference` 是为判别器准备的额外观测组，
    **不参与 actor/critic 的输入**，在步骤⑥ 接入 WASABI 后才被使用。
    """

    policy: G1AmpWalkPolicyObsCfg = G1AmpWalkPolicyObsCfg()
    critic: G1FlatCriticObsCfg = G1FlatCriticObsCfg()
    amp_policy: AmpPolicyStateObsCfg = AmpPolicyStateObsCfg()
    amp_reference: AmpReferenceStateObsCfg = AmpReferenceStateObsCfg()


# ============================================================================
# 环境配置
# ============================================================================


@configclass
class G1AmpWalkEnvCfg(G1OmniLocoEnvCfg):
    """AMP 行走教师训练配置。

    奖励 / 事件 / 终止 / 时序 / 动作 **全部继承 OmniLoco**（= flat 基线）；
    替换的只有：场景（+参考机器人+运动参考传感器）、观测组、速度指令项。
    """

    scene: G1AmpWalkSceneCfg = G1AmpWalkSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: G1AmpWalkCommandsCfg = G1AmpWalkCommandsCfg()
    observations: G1AmpWalkObservationsCfg = G1AmpWalkObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # ⚠️ 父类 G1OmniLocoEnvCfg.__post_init__ 会把命令 ranges 重新设成 OmniLoco 的值，
        #    因此必须在其之后“再”写回我们的方向分档参数。
        apply_amp_walk_command_ranges(self.commands.base_velocity)
        # 父类 __post_init__ 已经把 run_name 设成 OmniLoco 的，这里必须覆盖。
        self.run_name = "G1AmpWalkPlane_obs626_directional"


@configclass
class G1AmpWalkEnvCfg_PLAY(G1AmpWalkEnvCfg):
    scene: G1AmpWalkSceneCfg = G1AmpWalkSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        motion_reference=motion_reference_cfg.replace(debug_vis=True),
    )

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
