import os

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

import instinctlab.tasks.shadowing.beyondmimic.beyondmimic_env_cfg as beyondmimic_cfg
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_REV10_PELVIS_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuators,
)
from instinctlab.monitors import (
    ActuatorMonitorTerm,
    MonitorTermCfg,
    RewardSumMonitorTerm,
    ShadowingJointReferenceMonitorTerm,
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear

#: ⭐⭐ 机器人本体 = **官方 rev1.0（pelvis 根）**，与下面的 `VAULT_FK_URDF` **成对**：
#: 之前本体用的是 popsicle(torso 根)，而 FK 已改成 rev10(pelvis 根) ⇒ 两者不同源，导致
#:   ① 三个腰关节**符号相反**（实测：`waist_*=+0.3` 时躯干旋转方向相反）；
#:   ② 观测/奖励里的 **base 也不同源**（机器人 root=torso vs 参考 base=pelvis）⇒
#:      `base_position_imitation_gauss` 基准直接差 torso−pelvis = **0.0440 m**。
#: 换成 rev10 后：root=pelvis ⇒ 与参考 base 一致、腰符号一致 ⇒ 内部完全自洽。
#: ⚠️ 该改动使旧 ckpt（popsicle 训练）**不再可用**，需从零重训。
G1_CFG = G1_29DOF_REV10_PELVIS_CFG

#: ⭐⭐ FK 的**浮动基座 = URDF 的根 link**（`MotionReferenceManager._initialize_robot_kinematics`
#: 用 `pk.build_chain_from_urdf` 建链） —— 而本仓所有 retarget 数据集的 `base_pos_w`
#: 记的是 **pelvis**，不是 torso_link：
#:   · 实测（`g1_speed_vault_retargeted.npz` 抽 6 帧）：把 base 当 **torso** 用，
#:     全模型最低点 = **-0.027~-0.037 m（穿地）**；当 **pelvis** 用则恰好 **+0.000 m**；
#:   · 我们已有的 34 个 AMP 数据集同样以 pelvis 为 base（base_z 均值 0.772）；
#:   · popsicle 资产的默认站姿 torso_z - pelvis_z = **0.0440 m**。
#: `g1_29dof_torsobase_popsicle.urdf` 的根是 **torso_link** ⇒ 直接用它做 FK 会把
#: pelvis 的轨迹张冠李戴到 torso 上，**参考整体偏 4.4 cm**（翻越这种精确足部接触
#: 任务里足以把奖励/终止条件带偏）。
#: ⇒ 改用仓内**同源但以 pelvis 为根**的官方 URDF 做 FK：两版 URDF 的关节原点逐项
#:   一致（如 `left_hip_pitch_joint` origin 均为 `0 0.064452 -0.1027`），只是根不同，
#:   关节/链接命名完全齐备，替换后 `link_of_interests` 与 `isaac_joint_names` 都能对上。
_G1_RESOURCES_DIR = os.path.dirname(os.path.dirname(G1_CFG.spawn.asset_path))
VAULT_FK_URDF = os.path.join(_G1_RESOURCES_DIR, "official_rev10", "g1_29dof_rev_1_0.urdf")
MOTION_NAME = "MimicKitSpeedVault"
MOTION_DIR = os.path.join(os.path.dirname(__file__), "motions", "vault")

VAULT_MOTION_FILE = "g1_speed_vault_retargeted.npz"
MOTION_SELECTION_FILE = os.path.join(MOTION_DIR, "g1_vault_only_motion_selection.yaml")
VAULT_BOX_HIDDEN_POS = (1000.0, 1000.0, -100.0)

with np.load(os.path.join(MOTION_DIR, VAULT_MOTION_FILE)) as _vault_npz:
    _size = _vault_npz["obstacle_size"]
    _pos = _vault_npz["obstacle_pos"]
    VAULT_BOX_SIZE: tuple[float, float, float] = (float(_size[0]), float(_size[1]), float(_size[2]))
    VAULT_BOX_POS: tuple[float, float, float] = (float(_pos[0]), float(_pos[1]), float(_pos[2]))


@configclass
class VaultMotionCfg(AmassMotionCfgBase):
    """Converted MimicKit G1 vault and flat locomotion motions.

    数据集契约（`g1_speed_vault_retargeted.npz`，实测）：
      · `framerate = 50 Hz`、737 帧 = 14.74 s；base 从 x=-6.52 跑到 x=+5.56
        （含 0.25 m 抬升，即翻越段）；
      · 障碍箱 `obstacle_pos=(1.7, 0.5, 0.055)`、`obstacle_size=(0.4, 10.0, 0.6)`
        ⇒ 与下面 `VAULT_BOX_POS` **完全一致**，即参考轨迹与箱子已在同一世界系下
        对齐（**不需要再做任何平移**；管理器只额外叠加 `env_origins`）；
      · npz 只有 `joint_names / joint_pos / base_pos_w / base_quat_w`：
        加载器走 `_read_retargetted_motion_file`，把 29 关节**按名**重排到
        articulation 序，`joint_vel / base_lin_vel / base_ang_vel` 由
        `velocity_estimation_method` 数值估计；npz 里的 `velocity_command`
        本管线**不读取**（BeyondMimic 系奖励用参考自身速度，不用外部指令）。
    """

    path = MOTION_DIR
    retargetting_func = None
    filtered_motion_selection_filepath = MOTION_SELECTION_FILE
    #: 收紧到本数据集后缀（默认列表还含 poses/stageii/soma.csv，会把同目录其它
    #: 文件一并扫进来）；与 AMP 臂的写法一致。
    supported_file_endings = ["retargeted.npz"]
    #: 数据集 50 fps == 环境控制频率（sim.dt=0.005 × decimation=4 = 50 Hz）
    #: ⇒ 插值退化为 no-op；写出来便于核对（默认值恰好也是 50）。
    motion_target_framerate = 50.0
    #: bin 生效时该项会被 binning 覆盖（见 `AmassMotionCfg` 的 docstring），
    #: 这里与 `perceptive_vault_cfg.py` 的训练口径保持一致。
    motion_start_from_middle_range = [0.0, 0.0]
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontbackward"
    #: 1 s 一个 bin ⇒ 允许从动作中段起（BeyondMimic 风格，训练恢复能力）；
    #: 置 None 则严格从运动起点起（PLAY 变体就是这么做的）。
    motion_bin_length_s = 1.0
    #: 与 `perceptive_vault_cfg.py` 训练口径一致（单 clip 下与 "independent" 等价）。
    env_starting_stub_sampling_strategy = "concat_motion_bins"


motion_reference_cfg = MotionReferenceManagerCfg(
    # ⚠️ 这里必须是**机器人的 articulation 根 prim**（报错原文：
    #   "For robot motion reference, only accept articulated prims for now"）。
    #   popsicle(torso 根) 时代写 `Robot/torso_link` 恰好就是根；换 rev10(pelvis 根) 后
    #   `Robot/torso_link` 变成子 link ⇒ 必须改为 `Robot/pelvis`。
    prim_path="{ENV_REGEX_NS}/Robot/pelvis",
    # ⭐ 必须用 **pelvis 为根**的 URDF（见上面 `VAULT_FK_URDF` 的说明）：
    #   数据集 base_pos_w 是 pelvis 的位姿，而 FK 的浮动基座 = URDF 根 link。
    #   用 popsicle（torso 根）会整体偏 4.4 cm。
    robot_model_path=VAULT_FK_URDF,
    #: 参考模型同样用 rev10（root=pelvis）⇒ 镜像 prim 也是 pelvis。
    reference_prim_path="/World/envs/env_.*/RobotReference/pelvis",
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    frame_interval_s=0.0,
    update_period=0.02,
    num_frames=1,
    visualizing_robot_offset=(0.0, 2.0, 0.0),
    visualizing_robot_from="reference_frame",
    motion_buffers={
        MOTION_NAME: VaultMotionCfg(),
    },
    # ⚠️ 单 clip 任务：**不能**用 "Even"（那是给 AMP 臂多 clip 跨进程切分用的）。
    #    本任务只有 1 个运动文件，用 "None"（= 不跨进程切分，每个进程被动加载整段），
    #    与已跑通的 `perceptive_vault_cfg.py` 的 `VAULT_MOTION_REFERENCE_CFG` 一致。
    mp_split_method="None",
)


def _vault_box_cfg(debug_vis: bool = False) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/VaultBox",
        spawn=sim_utils.MeshCuboidCfg(
            size=VAULT_BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.3) if not debug_vis else (0.0, 0.6, 0.9)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=VAULT_BOX_POS),
    )


def reset_vault_box_by_motion(
    env,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("vault_box"),
    motion_buffer_name: str = MOTION_NAME,
    vault_motion_file: str = VAULT_MOTION_FILE,
    active_pos: tuple[float, float, float] = VAULT_BOX_POS,
    hidden_pos: tuple[float, float, float] = VAULT_BOX_HIDDEN_POS,
):
    """Show the vault box only for envs that sampled the vault motion."""

    if asset_cfg.name not in env.scene.rigid_objects:
        return

    vault_box = env.scene[asset_cfg.name]
    motion_reference = env.scene[motion_ref_cfg.name]
    motion_buffer = motion_reference.motion_buffers[motion_buffer_name]
    assigned_ids = motion_buffer.env_ids_to_assigned_ids(env_ids).to(motion_buffer.buffer_device)
    motion_ids = motion_buffer._assigned_env_motion_selection[assigned_ids].detach().cpu().tolist()
    current_motion_files = [motion_buffer._all_motion_files[motion_id] for motion_id in motion_ids]

    is_vault_motion = torch.tensor(
        [os.path.basename(motion_file) == vault_motion_file for motion_file in current_motion_files],
        dtype=torch.bool,
        device=env.device,
    )

    root_state = vault_box.data.default_root_state[env_ids].clone()
    root_state[:, :3] = torch.tensor(hidden_pos, dtype=root_state.dtype, device=env.device)
    root_state[:, :3] += env.scene.env_origins[env_ids]
    if is_vault_motion.any():
        root_state[is_vault_motion, :3] = torch.tensor(active_pos, dtype=root_state.dtype, device=env.device)
        root_state[is_vault_motion, :3] += env.scene.env_origins[env_ids[is_vault_motion]]

    root_state[:, 7:] = 0.0
    vault_box.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    vault_box.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def _sync_link_observation_terms(env_cfg):
    if env_cfg.observations.policy.__dict__.get("link_pos", None) is not None:
        env_cfg.observations.policy.link_pos.params["asset_cfg"].body_names = (
            env_cfg.scene.motion_reference.link_of_interests
        )
    if env_cfg.observations.policy.__dict__.get("link_rot", None) is not None:
        env_cfg.observations.policy.link_rot.params["asset_cfg"].body_names = (
            env_cfg.scene.motion_reference.link_of_interests
        )
    if hasattr(env_cfg.observations, "critic") and env_cfg.observations.critic is not None:
        if env_cfg.observations.critic.__dict__.get("link_pos", None) is not None:
            env_cfg.observations.critic.link_pos.params["asset_cfg"].body_names = (
                env_cfg.scene.motion_reference.link_of_interests
            )
        if env_cfg.observations.critic.__dict__.get("link_rot", None) is not None:
            env_cfg.observations.critic.link_rot.params["asset_cfg"].body_names = (
                env_cfg.scene.motion_reference.link_of_interests
            )


def _apply_vault_task_settings(env_cfg, debug_vis: bool = False):
    env_cfg.scene.vault_box = _vault_box_cfg(debug_vis=debug_vis)
    env_cfg.scene.robot.actuators = beyondmimic_g1_29dof_actuators
    env_cfg.actions.joint_pos.scale = beyondmimic_action_scale
    env_cfg.events.push_robot = None
    env_cfg.events.bin_fail_counter_smoothing = None
    env_cfg.events.reset_vault_box = EventTermCfg(
        func=reset_vault_box_by_motion,
        mode="reset",
        params={
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            "asset_cfg": SceneEntityCfg("vault_box"),
            "motion_buffer_name": MOTION_NAME,
            "vault_motion_file": VAULT_MOTION_FILE,
            "active_pos": VAULT_BOX_POS,
            "hidden_pos": VAULT_BOX_HIDDEN_POS,
        },
    )
    env_cfg.curriculum.beyond_adaptive_sampling = None

    env_cfg.terminations.base_pos_too_far.params["distance_threshold"] = 0.5
    env_cfg.terminations.base_pos_too_far.params["height_only"] = False
    env_cfg.terminations.link_pos_too_far.params["distance_threshold"] = 0.5
    env_cfg.terminations.link_pos_too_far.params["height_only"] = False
    env_cfg.terminations.link_pos_too_far.params["reference_cfg"].body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]

    # Vaulting may legitimately touch the obstacle with the hands/wrists.
    env_cfg.rewards.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
        (
            r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
            r"(?!left_wrist_roll_link$)(?!left_wrist_pitch_link$)(?!left_wrist_yaw_link$)"
            r"(?!right_wrist_roll_link$)(?!right_wrist_pitch_link$)(?!right_wrist_yaw_link$).+$"
        )
    ]

    _sync_link_observation_terms(env_cfg)

    assert len(list(env_cfg.scene.motion_reference.motion_buffers.keys())) == 1


@configclass
class G1BeyondMimicVaultEnvCfg(beyondmimic_cfg.BeyondMimicEnvCfg):
    """G1 BeyondMimic vault environment using the MimicKit speed-vault clip."""

    scene: beyondmimic_cfg.BeyondMimicSceneCfg = beyondmimic_cfg.BeyondMimicSceneCfg(
        num_envs=4096,
        env_spacing=5.0,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg,
    )

    def __post_init__(self):
        super().__post_init__()
        _apply_vault_task_settings(self, debug_vis=False)

        self.run_name = "G1BeyondMimic_MimicKitSpeedVault_noPush"


@configclass
class G1BeyondMimicVaultEnvCfg_PLAY(G1BeyondMimicVaultEnvCfg):
    """G1 BeyondMimic vault environment for playing/evaluation."""

    scene: beyondmimic_cfg.BeyondMimicSceneCfg = beyondmimic_cfg.BeyondMimicSceneCfg(
        num_envs=1,
        env_spacing=5.0,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=motion_reference_cfg.replace(
            debug_vis=True,
        ),
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=[4.0, 1.5, 1.2],
        lookat=[1.0, 0.5, 0.4],
        origin_type="asset_root",
        asset_name="robot",
    )

    def __post_init__(self):
        super().__post_init__()
        _apply_vault_task_settings(self, debug_vis=True)

        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 3
            self.scene.terrain.terrain_generator.num_cols = 3
            self.scene.terrain.terrain_generator.curriculum = False

        self.scene.motion_reference.symmetric_augmentation_joint_mapping = None
        self.scene.motion_reference.visualizing_marker_types = ["relative_links"]
        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None

        self.episode_length_s = 6000.0
        self.scene.motion_reference.motion_buffers[MOTION_NAME].motion_start_from_middle_range = [0.0, 0.0]
        self.scene.motion_reference.motion_buffers[MOTION_NAME].motion_bin_length_s = None

        for term in self.terminations.__dict__.values():
            if term is None:
                continue
            if "print_reason" in term.params:
                term.params["print_reason"] = True

        for cmd in self.commands.__dict__.values():
            cmd.debug_vis = True

        self.monitors.shoulder_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names="left_shoulder_roll.*"),
                "torque_plot_scale": 1e-2,
                "joint_power_plot_scale": 1e-1,
            },
        )
        self.monitors.waist_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names="waist_roll.*"),
                "torque_plot_scale": 1e-2,
                "joint_power_plot_scale": 1e-1,
            },
        )
        self.monitors.knee_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names="left_knee.*"),
                "torque_plot_scale": 1e-2,
                "joint_power_plot_scale": 1e-1,
            },
        )
        self.monitors.reward_sum = MonitorTermCfg(
            func=RewardSumMonitorTerm,
        )
        self.monitors.reference_stat_case = MonitorTermCfg(
            func=ShadowingJointReferenceMonitorTerm,
            params=dict(
                reference_cfg=SceneEntityCfg(
                    "motion_reference",
                    joint_names=[
                        "left_hip_pitch.*",
                    ],
                ),
            ),
        )
