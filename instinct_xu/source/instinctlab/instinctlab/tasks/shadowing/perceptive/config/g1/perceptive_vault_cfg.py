import math
import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.utils.noise import UniformNoiseCfg

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg as perceptual_cfg
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_LINKS,
    G1_29DOF_REV10_PELVIS_CFG,
    beyondmimic_g1_29dof_actuators,
)
from instinctlab.tasks.shadowing.beyondmimic.config.g1 import beyondmimic_vault_cfg as vault_cfg
from instinctlab.sensors import get_link_prim_targets


# Match BeyondMimic: no proprio history stack (history_length=0).
TEACHER_PROPRIO_HISTORY_LENGTH = 0

#: 机器人本体 = 官方 rev1.0（pelvis 根），与 `beyondmimic_vault_cfg.VAULT_FK_URDF`
#: **成对**使用（见该文件里 `G1_CFG` 的说明）：root=pelvis ⇒ 与参考动作 base 一致、
#: 腰关节符号一致，避免"腰反相 + base 偏 4.4 cm"。
G1_CFG = G1_29DOF_REV10_PELVIS_CFG
VAULT_CAMERA_PITCH_DEG = 34
VAULT_CAMERA_FOVY_DEG = 70
VAULT_CAMERA_DEPTH_RANGE = (0.0, 3.0)
# Student depth resolution. PHP uses 87x58; lowered here for VRAM with large num_envs
# while keeping a similar aspect ratio (~1.5) and staying above the old 18x32 pipeline.
VAULT_CAMERA_WIDTH = 48
VAULT_CAMERA_HEIGHT = 32
# Match policy step_dt (sim.dt=0.005, decimation=4). Parent perceptive camera is 1/60.
VAULT_CAMERA_UPDATE_PERIOD = 0.02
VAULT_BOX_RAYCAST_PATH = "/World/envs/env_.*/VaultBox/.*"
VAULT_BOX_PRIM_EXPR = "/World/envs/env_.*/VaultBox"
# Match climb: torso-centered 3.0 x 0.8 m at 0.1 m -> 31 x 9 = 279. The 0.7 x 0.7
# PHP grid only reaches ±0.35 m and misses the box (front x≈1.5) during approach.
VAULT_HEIGHT_SCAN_SIZE = (3.0, 0.8)
VAULT_HEIGHT_SCAN_RESOLUTION = 0.1
VAULT_HEIGHT_SCAN_DIM = 31 * 9
# Clips walk x∈[-6.5, 5.6] and the 10 m-wide box (y∈[-4.5, 5.5]); ±10 m cell.
VAULT_ENV_SPACING = 20.0
# Teacher tracking termination (also used by distill DAgger mask).
TEACHER_TRACKING_DISTANCE_THRESHOLD = 0.5
VAULT_BOX_RANDOMIZATION_ENABLED = os.environ.get("PERCEPTIVE_VAULT_BOX_RANDOMIZATION", "1") != "0"
VAULT_BOX_RANDOMIZE_POSE_RANGE = {
    "x": (
        -float(os.environ.get("PERCEPTIVE_VAULT_BOX_RANDOMIZE_X", "0.05")),
        float(os.environ.get("PERCEPTIVE_VAULT_BOX_RANDOMIZE_X", "0.05")),
    ),
    "y": (
        -float(os.environ.get("PERCEPTIVE_VAULT_BOX_RANDOMIZE_Y", "0.05")),
        float(os.environ.get("PERCEPTIVE_VAULT_BOX_RANDOMIZE_Y", "0.05")),
    ),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    # PHP: obstacle yaw randomized within ±45°.
    "yaw": (
        -float(os.environ.get("PERCEPTIVE_VAULT_BOX_RANDOMIZE_YAW", str(math.radians(45)))),
        float(os.environ.get("PERCEPTIVE_VAULT_BOX_RANDOMIZE_YAW", str(math.radians(45)))),
    ),
}
# Keep BeyondMimic/PHP single-frame reference (num_frames=1, frame_interval_s=0.0)
# and vault link_of_interests (includes wrist roll/pitch for hand contact tracking).
# Do not inherit shadowing's 10-frame future reference stack or shortened link list.
VAULT_MOTION_REFERENCE_CFG = vault_cfg.motion_reference_cfg.replace(
    update_period=0.02,
    visualizing_robot_offset=(2.0, 0.0, 0.0),
    visualizing_robot_from="reference_frame",
    visualizing_marker_types=["relative_links", "links"],
    mp_split_method="None",
)
VAULT_MOTION_REFERENCE_CFG.motion_buffers[
    vault_cfg.MOTION_NAME
].filtered_motion_selection_filepath = vault_cfg.MOTION_SELECTION_FILE
# Number of tracked bodies for critic / rewards (must match link_of_interests).
VAULT_NUM_LINK_OF_INTERESTS = len(VAULT_MOTION_REFERENCE_CFG.link_of_interests)


def _append_unique(items: list[str], item: str):
    if item not in items:
        items.append(item)


def global_pelvis_position(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Pelvis/root position in each environment's world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def global_pelvis_velocity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Pelvis/root linear velocity in the world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w


def _apply_vault_camera_settings(camera_cfg):
    pitch_rad = math.radians(VAULT_CAMERA_PITCH_DEG)
    fovy_rad = math.radians(VAULT_CAMERA_FOVY_DEG)
    camera_cfg.offset.pos = (
        0.04764571478 + 0.0039635 - 0.0042 * math.cos(pitch_rad),
        0.015,
        0.46268178553 - 0.044 + 0.0042 * math.sin(pitch_rad) + 0.016,
    )
    camera_cfg.offset.rot = (
        math.cos(pitch_rad / 2),
        0.0,
        math.sin(pitch_rad / 2),
        0.0,
    )
    camera_cfg.pattern_cfg.vertical_aperture = 2 * math.tan(fovy_rad / 2)
    # Square pixels: aperture aspect must match image aspect (48/32), so fovx ≈ 92.8°.
    # Leaving horizontal_aperture at the parent 87° would keep non-square pixels.
    camera_cfg.pattern_cfg.horizontal_aperture = camera_cfg.pattern_cfg.vertical_aperture * (
        VAULT_CAMERA_WIDTH / VAULT_CAMERA_HEIGHT
    )
    camera_cfg.pattern_cfg.width = VAULT_CAMERA_WIDTH
    camera_cfg.pattern_cfg.height = VAULT_CAMERA_HEIGHT
    camera_cfg.update_period = VAULT_CAMERA_UPDATE_PERIOD
    camera_cfg.noise_pipeline["normalize"].depth_range = VAULT_CAMERA_DEPTH_RANGE
    # Native PHP resolution; drop InstinctLab crop/resize to 18x32.
    camera_cfg.noise_pipeline.pop("crop_and_resize", None)
    for mesh_prim_path in get_link_prim_targets(G1_29DOF_LINKS):
        _append_unique(camera_cfg.mesh_prim_paths, mesh_prim_path)
    _append_unique(camera_cfg.mesh_prim_paths, VAULT_BOX_RAYCAST_PATH)


def enable_vault_camera(env_cfg, debug_vis: bool = False):
    """Restore the vault depth camera (needed by the student / distill env)."""
    camera_cfg = perceptual_cfg.PerceptiveShadowingSceneCfg().camera
    _apply_vault_camera_settings(camera_cfg)
    camera_cfg.debug_vis = debug_vis
    env_cfg.scene.camera = camera_cfg
    # Re-enable camera extrinsics randomization if the parent event exists.
    if getattr(env_cfg.events, "randomize_ray_offsets", None) is None:
        env_cfg.events.randomize_ray_offsets = perceptual_cfg.EventsCfg().randomize_ray_offsets


def _vault_box_cfg(debug_vis: bool = False) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/VaultBox",
        spawn=sim_utils.MeshCuboidCfg(
            size=vault_cfg.VAULT_BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.3) if not debug_vis else (0.0, 0.6, 0.9)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=vault_cfg.VAULT_BOX_POS),
    )


def vault_box_cfg(debug_vis: bool = False) -> RigidObjectCfg:
    """Create the vault obstacle for composed perceptive environments."""
    return _vault_box_cfg(debug_vis=debug_vis)


def reset_vault_box_pose_by_motion_randomized(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("vault_box"),
    active_pos: tuple[float, float, float] = vault_cfg.VAULT_BOX_POS,
    pose_range: dict[str, tuple[float, float]] | None = None,
    # Kept for backward-compatible EventTermCfg params from older configs.
    motion_ref_cfg: SceneEntityCfg | None = None,
    motion_buffer_name: str | None = None,
    vault_motion_file: str | None = None,
    hidden_pos: tuple[float, float, float] | None = None,
):
    """Place the vault box for every env. Buffer is vault-only (incl. speed variants)."""
    _ = (motion_ref_cfg, motion_buffer_name, vault_motion_file, hidden_pos)
    if asset_cfg.name not in env.scene.rigid_objects:
        return

    vault_box = env.scene[asset_cfg.name]
    root_state = vault_box.data.default_root_state[env_ids].clone()
    root_state[:, :3] = torch.tensor(active_pos, dtype=root_state.dtype, device=env.device)
    root_state[:, :3] += env.scene.env_origins[env_ids]

    pose_range = pose_range or {}
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, dtype=root_state.dtype, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    root_state[:, :3] += rand_samples[:, :3]
    quat_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    root_state[:, 3:7] = math_utils.quat_mul(quat_delta, root_state[:, 3:7])
    root_state[:, 7:] = 0.0

    vault_box.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    vault_box.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


@configclass
class PhpTeacherPolicyObsCfg(ObsGroupCfg):
    """PHP privileged teacher actor observations (heightmap + global state, no depth)."""

    joint_pos_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"})
    joint_vel_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"})
    position_ref = ObsTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "position_ref_command"},
        noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
    )
    rotation_ref = ObsTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "rotation_ref_command"},
        noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    )
    height_scan = ObsTermCfg(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=[-1.0, 1.0],
    )
    global_pelvis_pos = ObsTermCfg(func=global_pelvis_position)
    global_pelvis_vel = ObsTermCfg(func=global_pelvis_velocity)
    base_lin_vel = ObsTermCfg(
        func=mdp.base_lin_vel,
        noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
    )
    projected_gravity = ObsTermCfg(
        func=mdp.projected_gravity,
        noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
    )
    base_ang_vel = ObsTermCfg(
        func=mdp.base_ang_vel,
        noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
    )
    joint_pos = ObsTermCfg(
        func=mdp.joint_pos_rel,
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
    )
    joint_vel = ObsTermCfg(
        func=mdp.joint_vel_rel,
        noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
    )
    last_action = ObsTermCfg(func=mdp.last_action, history_length=TEACHER_PROPRIO_HISTORY_LENGTH)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class PhpTeacherCriticObsCfg(ObsGroupCfg):
    """Teacher critic obs in explicit order, matching distill teacher obs_format."""

    joint_pos_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"})
    joint_vel_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"})
    position_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "position_ref_command"})
    rotation_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "rotation_ref_command"})
    height_scan = ObsTermCfg(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=[-1.0, 1.0],
    )
    global_pelvis_pos = ObsTermCfg(func=global_pelvis_position)
    global_pelvis_vel = ObsTermCfg(func=global_pelvis_velocity)
    base_lin_vel = ObsTermCfg(func=mdp.base_lin_vel, history_length=TEACHER_PROPRIO_HISTORY_LENGTH)
    projected_gravity = ObsTermCfg(func=mdp.projected_gravity, history_length=TEACHER_PROPRIO_HISTORY_LENGTH)
    base_ang_vel = ObsTermCfg(func=mdp.base_ang_vel, history_length=TEACHER_PROPRIO_HISTORY_LENGTH)
    joint_pos = ObsTermCfg(func=mdp.joint_pos_rel, history_length=TEACHER_PROPRIO_HISTORY_LENGTH)
    joint_vel = ObsTermCfg(func=mdp.joint_vel_rel, history_length=TEACHER_PROPRIO_HISTORY_LENGTH)
    last_action = ObsTermCfg(func=mdp.last_action, history_length=TEACHER_PROPRIO_HISTORY_LENGTH)
    link_pos = ObsTermCfg(
        func=instinct_mdp.link_pos_b,
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=VAULT_MOTION_REFERENCE_CFG.link_of_interests,
                preserve_order=True,
            ),
        },
    )
    link_rot = ObsTermCfg(
        func=instinct_mdp.link_tannorm_b,
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=VAULT_MOTION_REFERENCE_CFG.link_of_interests,
                preserve_order=True,
            ),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class PhpTeacherObservationsCfg(perceptual_cfg.ObservationsCfg):
    """Teacher env observations: privileged policy + standard critic."""

    policy: PhpTeacherPolicyObsCfg = PhpTeacherPolicyObsCfg()
    critic: PhpTeacherCriticObsCfg = PhpTeacherCriticObsCfg()


def _apply_perceptive_vault_settings(env_cfg, debug_vis: bool = False):
    env_cfg.scene.vault_box = _vault_box_cfg(debug_vis=debug_vis)
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.robot.actuators = beyondmimic_g1_29dof_actuators
    # PHP: action scale = 1 for all experts (no BeyondMimic effort/stiffness heuristics).
    env_cfg.actions.joint_pos.scale = 1.0
    # Privileged teacher does not use depth; drop the camera to save compute.
    env_cfg.scene.camera = None
    env_cfg.events.randomize_ray_offsets = None

    # Standard RayCaster only supports one mesh; use MultiMesh so the scan sees VaultBox.
    env_cfg.scene.height_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=VAULT_HEIGHT_SCAN_RESOLUTION,
            size=list(VAULT_HEIGHT_SCAN_SIZE),
        ),
        debug_vis=debug_vis,
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr=VAULT_BOX_PRIM_EXPR,
                track_mesh_transforms=True,
            ),
        ],
    )
    if getattr(env_cfg.observations.critic, "height_scan", None) is not None:
        env_cfg.observations.critic.height_scan.clip = [-1.0, 1.0]
    # Match BeyondMimic: critic proprio is also single-frame.
    for term_name in ("base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", "last_action"):
        term = getattr(env_cfg.observations.critic, term_name, None)
        if term is not None:
            term.history_length = TEACHER_PROPRIO_HISTORY_LENGTH

    # PHP Table V: friction on robot bodies (vault-only override; shared EventsCfg keeps InstinctLab defaults).
    if getattr(env_cfg.events, "physics_material", None) is not None:
        env_cfg.events.physics_material.params["static_friction_range"] = (0.4, 1.3)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.1)

    # PHP Table V: root velocity perturbations every U[1, 3] s (vault-only; not in shared EventsCfg).
    if debug_vis:
        env_cfg.events.push_robot = None
        env_cfg.events.bin_fail_counter_smoothing = None
        env_cfg.curriculum.beyond_adaptive_sampling = None
    else:
        env_cfg.events.push_robot = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1),
                    "z": (-0.05, 0.05),
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.1, 0.1),
                },
            },
        )
    env_cfg.scene.motion_reference.motion_buffers[
        vault_cfg.MOTION_NAME
    ].motion_start_from_middle_range = [0.0, 0.0]
    env_cfg.scene.motion_reference.motion_buffers[vault_cfg.MOTION_NAME].motion_bin_length_s = (
        None if debug_vis else 1.0
    )
    env_cfg.scene.motion_reference.motion_buffers[
        vault_cfg.MOTION_NAME
    ].env_starting_stub_sampling_strategy = "independent" if debug_vis else "concat_motion_bins"
    env_cfg.events.reset_vault_box = EventTermCfg(
        func=reset_vault_box_pose_by_motion_randomized,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("vault_box"),
            "active_pos": vault_cfg.VAULT_BOX_POS,
            "pose_range": VAULT_BOX_RANDOMIZE_POSE_RANGE
            if (VAULT_BOX_RANDOMIZATION_ENABLED and not debug_vis)
            else {},
        },
    )

    env_cfg.terminations.base_pos_too_far.params["distance_threshold"] = TEACHER_TRACKING_DISTANCE_THRESHOLD
    env_cfg.terminations.base_pos_too_far.params["height_only"] = False
    env_cfg.terminations.link_pos_too_far.params["distance_threshold"] = TEACHER_TRACKING_DISTANCE_THRESHOLD
    env_cfg.terminations.link_pos_too_far.params["height_only"] = False
    env_cfg.terminations.link_pos_too_far.params["reference_cfg"].body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]

    # PHP Table IV reward weights.
    env_cfg.rewards.rewards.base_position_imitation_gauss.weight = 1.0
    env_cfg.rewards.rewards.base_rot_imitation_gauss.weight = 1.0
    env_cfg.rewards.rewards.undesired_contacts.weight = -0.5
    env_cfg.rewards.rewards.applied_torque_limits_by_ratio = None
    env_cfg.rewards.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
        (
            r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
            r"(?!left_wrist_roll_link$)(?!left_wrist_pitch_link$)(?!left_wrist_yaw_link$)"
            r"(?!right_wrist_roll_link$)(?!right_wrist_pitch_link$)(?!right_wrist_yaw_link$).+$"
        )
    ]
    env_cfg.terminations.illegal_reset_contact.params["sensor_cfg"].body_names = (
        env_cfg.rewards.rewards.undesired_contacts.params["sensor_cfg"].body_names
    )

    env_cfg.observations.critic.link_pos.params["asset_cfg"].body_names = (
        env_cfg.scene.motion_reference.link_of_interests
    )
    env_cfg.observations.critic.link_rot.params["asset_cfg"].body_names = (
        env_cfg.scene.motion_reference.link_of_interests
    )


@configclass
class G1PerceptiveVaultEnvCfg(perceptual_cfg.PerceptiveShadowingEnvCfg):
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=4096,
        env_spacing=VAULT_ENV_SPACING,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=VAULT_MOTION_REFERENCE_CFG,
    )
    observations: PhpTeacherObservationsCfg = PhpTeacherObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_perceptive_vault_settings(self, debug_vis=False)
        self.run_name = "g1PerceptiveVaultPhpTeacher_concatMotionBins"


@configclass
class G1PerceptiveVaultEnvCfg_PLAY(G1PerceptiveVaultEnvCfg):
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=1,
        env_spacing=VAULT_ENV_SPACING,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=VAULT_MOTION_REFERENCE_CFG.replace(debug_vis=True),
    )
    viewer: ViewerCfg = ViewerCfg(
        # Keep the play camera in the world frame so the user can freely orbit,
        # pan, and zoom without the viewer snapping back to the robot every step.
        eye=[7.0, -9.0, 4.5],
        lookat=[0.0, 0.5, 0.8],
        origin_type="world",
    )

    def __post_init__(self):
        super().__post_init__()
        _apply_perceptive_vault_settings(self, debug_vis=True)

        self.episode_length_s = 6000.0
        self.scene.motion_reference.symmetric_augmentation_joint_mapping = None
        self.scene.motion_reference.visualizing_marker_types = ["relative_links"]
        self.scene.motion_reference.motion_buffers[vault_cfg.MOTION_NAME].motion_start_from_middle_range = [0.0, 0.0]
        self.scene.motion_reference.motion_buffers[vault_cfg.MOTION_NAME].motion_bin_length_s = None

        self.terminations.base_pos_too_far = None
        self.terminations.base_pg_too_far = None
        self.terminations.link_pos_too_far = None

        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.physics_material = None
        self.events.push_robot = None  # type: ignore
        self.events.reset_robot.params["randomize_pose_range"]["x"] = [0.0] * 2
        self.events.reset_robot.params["randomize_pose_range"]["y"] = [0.0] * 2
        self.events.reset_robot.params["randomize_pose_range"]["z"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_pose_range"]["roll"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_pose_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_pose_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["x"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["y"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["z"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["roll"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_joint_pos_range"] = (0.0, 0.0)
