"""AMP 行走专家策略（G1 29-DoF，privileged teacher）—— **整条奖励链的根**（本文件定义 `G1AmpWalkRewardsCfg`）。

## ⚠️ 本层对**今天**仍有影响的部分

* ⭐ `G1AmpWalkRewardsCfg`（AMP_mjlab v5 迁移的奖励集）**是今天所有臂的奖励集根** ⇒ 各项今天实际生效值见
  `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-T1l-G1-v0`（**别读本文件的字面值**）。
* ⭐ **本文件 = V5**（`run_name = "G1AmpWalkPlane_obs626_directional_rew5"`，**不是 V33**）；后层全靠 `super().__post_init__()` 逐层覆盖。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | actor 观测 623 → **626**（+3 `base_lin_vel`，特权量） | **已被 V9 覆盖** → 378 维 |
| ② | 速度指令 → **方向分档**（`DirectionalVelocityCommand`） | ✅ 仍用（范围被 V18 等改窄）|
| ③ | 奖励集 → **AMP_mjlab v5 迁移版** | ✅ 仍为整条链的根 |
| ④ | 额外挂 `robot_reference` + `motion_reference`（判别器参考态） | ✅ 仍用 |
| ⑤ | "**尚未接入 AMP 算法**（步骤⑥）" | **已被后层覆盖** → AMP 早已接入 |

## 其余全部继承、不改动（今天仍成立）

| 项 | 值 | 来源 |
|---|---|---|
| 动作 | 29 维位置控制，`scale=1.0`，`use_default_offset=True` | `G1OmniLocoActionsCfg` |
| 时序 | `dt=0.005`，`decimation=4`，`step_dt=0.02` | `G1FlatEnvCfg.__post_init__` |
| 高程图 | 527 射线（31×17，3.0×1.6 m，res 0.1） | `G1OmniLocoSceneCfg` |
| 执行器 | `beyondmimic_g1_29dof_actuators` | `G1OmniLocoEnvCfg.__post_init__` |
| 事件 / 终止 | 与 OmniLoco 相同 | `G1FlatEventsCfg` / `G1FlatTerminationsCfg` |
| Critic 观测 | 含 `base_lin_vel`（本就对称） | `G1FlatCriticObsCfg` |
| 速度指令 | 方向分档（`DirectionalVelocityCommand`） | `tasks/locomotion/mdp/commands.py` |
| 奖励函数本体 | 一行不改（只覆盖权重 + 新增 5 项） | `tasks/locomotion/mdp/rewards_amp.py` |

## 推理与判读（已归档，勿在本文件重复）

* 为什么 actor 要加 `base_lin_vel`（对齐 mjlab / 界定蒸馏特权边界 / `G1OmniLocoEnvCfg` 只允许 623 维的 fail-fast 契约）→ 见该版 run 日志与 `PLAN_amp_walk.md` §6
* 本文件对应 `PLAN_amp_walk.md` §6 的**步骤②③④⑤**（② 任务骨架 / ③ 方向分档指令 / ④ AMP 观测组 670 维 / ⑤ 奖励集迁移）

⚠️ **本文件里的 `RewTerm(...)` 字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-T1l-G1-v0`"""

from __future__ import annotations

import math
import os

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.locomotion.mdp as locomotion_mdp
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear

from .flat_env_cfg import G1FlatCriticObsCfg, G1FlatRewardsCfg, G1_CFG
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

# ⚠️ 上面两行是 **history=10** 的字面量（`290 = 10×29`、`30 = 10×3`），只为文档/自检用，
#    运行时判别器的输入维度由 obs 群自动推断（`wasabi.py` 的 `actor_state_key="amp_policy"`）。
#    改帧数时这两行**不会**自动跟着变：
#      * `history=2`（论文 (s_t, s_{t+1})）= 2×67 = **134**，见 `amp_walk_v12_env_cfg.py`
#       （V12 在 env cfg 的 `__post_init__` 里新建 obs 群覆盖 history，不动本基线）。


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
# 奖励配置（步骤⑤：奖励集从 AMP_mjlab v5 迁移）
# ============================================================================

#: 步宽地板（m）—— **本仓库实测标定**，不是照抄 AMP_mjlab 的 0.15。
#: 出处：`instinct/tools/calibrate_foot_sep.py` → `tools/amp_walk_foot_sep_stats.json`
#:   直线行走 4 段 / 2186 帧：sep min = 0.1387、p1 = 0.1455、中位 = 0.2140
#: `sep_floor` 取 p1 向下取整，保证健康步态几乎零惩罚。
AMP_WALK_SEP_FLOOR = 0.145
#: 行走速档门控上限（m/s）：vx_cmd ≤ 0.8 全量塑形、0.8~1.0 线性过渡、≥1.0 豁免。
#: 依据：实测慢跑 clip 摆动腿合法内收（`jog_forward_loop_003__A021` sep 低到 0.040），
#: 而行走 clip 从不低于 0.1387。AMP_mjlab v5 同样取 1.0。
AMP_WALK_WALK_MAX_LIN_VEL = 1.0


@configclass
class G1AmpWalkRewardsCfg(G1FlatRewardsCfg):
    """`G1FlatRewardsCfg` → AMP_mjlab v5 奖励集。

    只做两类操作，**不改任何奖励函数本体，也不改基类**：

    1. **覆盖权重**（下面 3 个项；`func` / `params` 与基类逐字一致，只动 `weight`）
       —— 基类配置和 OmniLoco/Vault 任务完全不受影响，因为改的是子类字段；
    2. **新增 5 项**（高度跟踪 / root 角速度 / 低速回 default / 步宽 / 真交叉），
       实现全部在 `tasks/locomotion/mdp/rewards_amp.py`。

    迁移对照与**刻意偏离**的说明见 `rewards_amp.py` 的模块 docstring；要点：
    - `self_collisions` **不迁移**（PhysX 不支持多对多 contact 过滤；且本任务已有
      `base_contact` 终止 −200 的更强兜底）；
    - `feet_crossing` **额外加了方向门控**（参考动作的侧移 clip 是交叉式侧步，
      实测 21% 帧 sep < 0，无条件惩罚会压制侧移）。
    """

    # --- 1. 覆盖既有项的权重（来自 AMP_mjlab v5）---------------------------
    # 速度跟踪 1.0 → 1.5；转向跟踪 1.0 → 1.2。
    # 注意 `std` 保持本仓库的 0.5（不照搬 AMP_mjlab 的 1.0 / 3.14）：
    # 奖励函数的**形状**与同族 OmniLoco/Vault 教师保持一致，便于日后对照与蒸馏。
    track_lin_vel_xy_exp = RewTerm(
        func=locomotion_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=locomotion_mdp.track_ang_vel_z_world_exp,
        weight=1.2,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # 足部滑移惩罚 −0.1 → −0.25（AMP_mjlab 的 foot_slip）。
    # ⚠️ 两边实现不同（AMP 用 foot site 速度 + 指令门控；本仓库用 isaaclab 的
    # `contact_slide`，body 速度、无门控），量级不可严格换算，此处只按迁移表对齐权重。
    feet_slide = RewTerm(
        func=instinct_mdp.contact_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # --- 2. 新增项 ---------------------------------------------------------
    # 高度保持（AMP_mjlab `track_root_height`，权重 1.5 / std 0.3）。
    # 基类里的 `base_height_l2` 是被注释掉的，所以这里是**新写**的 exp 形式。
    track_base_height = RewTerm(
        func=locomotion_mdp.track_base_height_exp,
        weight=1.5,
        params={"std": 0.3},
    )
    # root 横滚/俯仰角速度抑制（AMP_mjlab `body_ang_vel_xy_l2`，权重 0.7 / std 3.14）。
    # 名字带 l2 但那条其实是 **exp 正奖励**，本项保持一致语义。
    body_ang_vel_xy = RewTerm(
        func=locomotion_mdp.track_root_ang_vel_xy_exp,
        weight=0.7,
        params={"std": 3.14},
    )
    # 低速/静止回 default 站姿 + 关节速度阻尼（AMP_mjlab，权重 −1.0）。
    # `gate_tau=0.25` 是平滑门控：指令 → 0 时权重 → 1，指令变大指数衰减。
    # `prob_low` 档会给出精确 (0,0,0) 指令 → 门控 = 1，与分档采样天然对齐。
    # `waist_yaw_joint: 3.0` 是 AMP_mjlab v3.4c 加的：防止腰部锁在 90° 的假吸引子。
    track_default_pose_on_stand = RewTerm(
        func=locomotion_mdp.track_default_pose_on_stand,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "pos_scale": 4.0,
            "vel_scale": 0.5,
            "gate_tau": 0.25,
            "pos_joint_weights": {"waist_yaw_joint": 3.0},
        },
    )
    # 步宽下限（AMP_mjlab `feet_lateral_width`，权重 −10.0）。
    # `max_cost=0.10` 单帧封顶，防深度交叉的奖励尖峰（−10 × 0.10 = 单步最多 −1.0）。
    feet_lateral_width = RewTerm(
        func=locomotion_mdp.feet_lateral_width_cost,
        weight=-10.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "sep_floor": AMP_WALK_SEP_FLOOR,
            "gate_lateral": True,
            "max_cost": 0.10,
            "walk_max_lin_vel": AMP_WALK_WALK_MAX_LIN_VEL,
            "walk_gate_width": 0.2,
        },
    )
    # 真交叉惩罚（AMP_mjlab `feet_crossing`，权重 −20.0）。
    # ⚠️ `gate_lateral=True` 是对 AMP_mjlab 的**有意偏离**：
    #    本仓库参考动作的侧移 clip 本身就是交叉式侧步（grapevine），实测 21% 帧
    #    sep < 0。不加方向门控会与 AMP 风格先验正面对撞、压制侧移能力。
    feet_crossing = RewTerm(
        func=locomotion_mdp.feet_crossing_cost,
        weight=-20.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "gate_lateral": True,
            "max_cost": 0.10,
            "walk_max_lin_vel": AMP_WALK_WALK_MAX_LIN_VEL,
            "walk_gate_width": 0.2,
        },
    )


# ============================================================================
# 环境配置
# ============================================================================


@configclass
class G1AmpWalkEnvCfg(G1OmniLocoEnvCfg):
    """AMP 行走教师训练配置。

    事件 / 终止 / 时序 / 动作 **全部继承 OmniLoco**（= flat 基线）；
    替换的是：场景（+参考机器人+运动参考传感器）、观测组、速度指令项、**奖励集**。
    """

    scene: G1AmpWalkSceneCfg = G1AmpWalkSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: G1AmpWalkCommandsCfg = G1AmpWalkCommandsCfg()
    observations: G1AmpWalkObservationsCfg = G1AmpWalkObservationsCfg()
    rewards: G1AmpWalkRewardsCfg = G1AmpWalkRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # ⚠️ 父类 G1OmniLocoEnvCfg.__post_init__ 会把命令 ranges 重新设成 OmniLoco 的值，
        #    因此必须在其之后“再”写回我们的方向分档参数。
        apply_amp_walk_command_ranges(self.commands.base_velocity)
        # 父类 __post_init__ 已经把 run_name 设成 OmniLoco 的，这里必须覆盖。
        self.run_name = "G1AmpWalkPlane_obs626_directional_rew5"


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
        # --- 步骤⑤ 的附带诊断（只在日志里，不做断言）----------------------
        # 1) 脚部参考 body：`feet_lateral_sep` 优先用 LL_FOOT/LR_FOOT，
        #    找不到时静默退回 `*_ankle_roll_link`（横向偏移为 0，数值几乎一致，
        #    但 sep 的标定量是前者，所以这里必须把实际用的哪一套打出来）。
        robot = self.scene["robot"]
        body_names = list(robot.body_names)
        has_foot_bodies = all(b in body_names for b in ("LL_FOOT", "LR_FOOT"))
        # 2) 奖励项数量：确认 `G1AmpWalkRewardsCfg` 真的生效（而不是继承了 OmniLoco 的）
        terms = getattr(self.reward_manager, "active_terms", None)
        n_rewards = len(terms) if terms is not None else -1
        # 指令分布诊断：`zero_frac_in_low` **不参与契约断言**（只有 `rel_standing_envs`
        # 被断言为 0.0），所以"原地指令减半"这种改动只能靠打印核对 —— 直接打出来。
        _cmd = self.cfg.commands.base_velocity
        print(
            f"[AMP-WALK PASS] policy={sum(self.expected_policy_dims)} "
            f"(height_scan={self.expected_scan_rays} rays; 逐项与顺序见该臂的 [T1?]/[V?] 打印), "
            f"action=29, scale=1.0, "
            f"cmd_ranges={self.expected_command_ranges} (directional), "
            f"cmd_low/zero={getattr(_cmd, 'prob_low', float('nan')):.2f}/"
            f"{getattr(_cmd, 'zero_frac_in_low', float('nan')):.2f}, "
            f"foot_bodies={'LL_FOOT/LR_FOOT' if has_foot_bodies else 'ankle_roll_link(FALLBACK)'}, "
            f"rewards={n_rewards}"
        )
        print(f"[AMP-WALK INFO] 奖励项（{n_rewards}）：{', '.join(sorted(terms)) if terms else '?'}")
        # ⭐ 逐项权重：**跨版本核对用**（同一行、按名字排序 ⇒ 直接 diff 两轮日志即可看出
        #    到底改了哪些项的权重，不必再靠读源码猜继承链）：
        #      grep 'AMP-WALK WEIGHTS' /root/train_amp_walk_v1[12]_4096.log
        if terms:
            try:
                _pairs = ", ".join(
                    f"{n}={float(self.reward_manager.get_term_cfg(n).weight):g}"
                    for n in sorted(terms)
                )
                print(f"[AMP-WALK WEIGHTS] {_pairs}")
            except Exception as _e:  # noqa: BLE001 —— 纯诊断，绝不能因为它影响启动
                print(f"[AMP-WALK WEIGHTS] (打印失败: {type(_e).__name__}: {_e})")
        # --- 数据层镜像增广的状态 + 关节顺序自检（V10 起启用；只打印，不断言）----------
        # 增广的**标签正确性**完全依赖 "`symmetric_augmentation_joint_mapping` 的索引口径
        # 与 motion_reference buffer 的关节顺序一致"。我们的映射是用
        # `build_joint_mirror(OMNI_LOCO_JOINT_ORDER)` 现算的（与镜像损失同一套代码），
        # 而 `OMNI_LOCO_JOINT_ORDER` 又由 OmniLoco 的契约断言等于动作项的关节顺序
        # ⇒ 这里直接把"实际关节顺序 == OMNI_LOCO_JOINT_ORDER"打出来做交叉验证。
        # ⚠️ 注意：运行时**必须从 `scene.sensors` 拿**（`scene.motion_reference` 这种属性
        #    访问方式在 InteractiveScene 上并不一定存在 ⇒ 早先的探针因此误报 OFF）。
        _sensors = getattr(self.scene, "sensors", None)
        _ref = _sensors.get("motion_reference") if isinstance(_sensors, dict) else None
        if _ref is None:
            _ref = getattr(self.scene, "motion_reference", None)
        _ref_cfg = getattr(_ref, "cfg", None)
        _jm = getattr(_ref_cfg, "symmetric_augmentation_joint_mapping", None)
        try:
            _cond = bool(_ref._symmetric_augmentation_conditions_met) if _ref is not None else None
        except Exception as _exc:
            _cond = f"?({_exc!r})"
        try:
            _mask = float(_ref._env_symmetric_augmentation_mask.float().mean())
        except Exception:
            _mask = None
        try:
            from .omni_loco_env_cfg import OMNI_LOCO_JOINT_ORDER

            _order_ok = tuple(robot.joint_names) == tuple(OMNI_LOCO_JOINT_ORDER)
            _order_msg = "一致" if _order_ok else "❌不一致（增广标签会错！）"
        except Exception as _exc:  # pragma: no cover - 诊断路径不该拖垮启动
            _order_msg = f"无法核对（{_exc!r}）"
        print(
            f"[AMP-WALK INFO] 数据层镜像增广：映射={'已设' if _jm is not None else '未设'}，"
            f"运行时条件={_cond}，被镜像 env 比例={_mask}；"
            f"joint 顺序 vs OMNI_LOCO_JOINT_ORDER：{_order_msg}"
        )
        if not has_foot_bodies:
            # 这是**预期**情况，不是异常：`LL_FOOT`/`LR_FOOT` 在 URDF 里是 ankle_roll_link
            # 的固定子连杆，Isaac Sim 的 URDF 导入器（`merge_fixed_joints=True`）会把它们
            # 合并进父连杆。云端 Kit 日志已直接印证：
            #   "link LR_FOOT has body properties (mass, inertia, or collisions)
            #    and is being merged into right_ankle_roll_link"
            # 因此 `*_ankle_roll_link` 就是脚部参考点，`sep` 只在 yaw 系下取 y 分量，
            # 而 LR_FOOT→ankle_roll_link 的横向偏移为 0 → `sep_floor=0.145`（在 LL_FOOT 上
            # 标定）可以原样使用。真实数值由 `Step/foot_sep_mean` 在线自检。
            print(
                "[AMP-WALK INFO] robot.body_names 无 LL_FOOT/LR_FOOT（URDF 导入器已把固定"
                "脚连杆合并进 ankle_roll_link，属预期）；feet_lateral_sep 使用 "
                "*_ankle_roll_link。sep_floor=0.145 照旧，数值由 Step/foot_sep_mean 在线校验。"
            )
