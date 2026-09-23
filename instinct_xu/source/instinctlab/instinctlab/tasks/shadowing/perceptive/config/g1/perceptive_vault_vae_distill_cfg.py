import os
from collections.abc import Sequence

import isaaclab.envs.mdp as mdp
import numpy as np
import torch
from isaaclab.envs import ViewerCfg
from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg as perceptual_cfg
from instinctlab.assets.unitree_g1 import G1_29DOF_TORSOBASE_POPSICLE_CFG

from . import perceptive_vault_cfg as vault_env_cfg

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG
DISTILL_PROPRIO_HISTORY_LENGTH = 0
VAULT_SKILL_COMMAND_X_RANGE = (0.0, 4.0)
# Match agent loss_curriculum_iterations / PHP K/2.
STUDENT_TERMINATION_CURRICULUM_ITERS = 10000
STUDENT_NUM_STEPS_PER_ENV = 24
_VAULT_APPROACH_SPEED_CACHE: dict[tuple[int, int, str, float, float], torch.Tensor] = {}
# Cached per-motion (T, 3) velocity_command arrays loaded from npz, or None if absent.
_VAULT_VELOCITY_COMMAND_CACHE: dict[tuple[int, int, str], torch.Tensor | None] = {}


def _first_command_frame(command: torch.Tensor, frame_dim: int) -> torch.Tensor:
    if command.shape[-1] == frame_dim:
        return command[:, 0] if command.ndim == 3 else command
    if command.ndim != 2 or command.shape[-1] % frame_dim != 0:
        raise ValueError(
            f"Expected command with last dim {frame_dim} or {frame_dim}*N, got shape {tuple(command.shape)}"
        )
    return command.reshape(command.shape[0], -1, frame_dim)[:, 0]


def single_frame_velocity_command(env, command_name: str) -> torch.Tensor:
    return _first_command_frame(mdp.generated_commands(env, command_name), frame_dim=3)


def single_frame_generated_command(env, command_name: str, frame_dim: int) -> torch.Tensor:
    return _first_command_frame(mdp.generated_commands(env, command_name), frame_dim=frame_dim)


def _load_motion_velocity_command(motion_buffer, motion_id: int, device: torch.device) -> torch.Tensor | None:
    """Load baked velocity_command (T, 3) from the motion npz, cached per motion_id."""
    cache_key = (id(motion_buffer), int(motion_id), str(device))
    if cache_key in _VAULT_VELOCITY_COMMAND_CACHE:
        return _VAULT_VELOCITY_COMMAND_CACHE[cache_key]

    filepath = motion_buffer._all_motion_files[int(motion_id)]
    try:
        with np.load(filepath, allow_pickle=False) as raw:
            if "velocity_command" not in raw.files:
                _VAULT_VELOCITY_COMMAND_CACHE[cache_key] = None
                return None
            command = torch.as_tensor(raw["velocity_command"], device=device, dtype=torch.float32)
    except OSError:
        _VAULT_VELOCITY_COMMAND_CACHE[cache_key] = None
        return None

    if command.ndim != 2 or command.shape[-1] != 3:
        raise ValueError(
            f"velocity_command in {filepath} must have shape (T, 3), got {tuple(command.shape)}"
        )
    _VAULT_VELOCITY_COMMAND_CACHE[cache_key] = command
    return command


def load_motion_velocity_command(motion_buffer, motion_id: int, device: torch.device) -> torch.Tensor | None:
    """Public wrapper for loading a cached baked velocity command."""
    return _load_motion_velocity_command(motion_buffer, motion_id, device)


def _baked_velocity_command(
    motion_reference,
    motion_buffer_name: str,
) -> torch.Tensor | None:
    """Sample per-env velocity_command from baked npz frames at the current motion time.

    Returns None if any assigned motion is missing the velocity_command key.
    """
    motion_buffer = motion_reference.motion_buffers[motion_buffer_name]
    env_ids = motion_reference.ALL_INDICES
    assigned_ids = motion_buffer.env_ids_to_assigned_ids(env_ids).to(motion_buffer.buffer_device)
    motion_ids = motion_buffer._assigned_env_motion_selection[assigned_ids]
    start_time_s = motion_buffer._motion_buffer_start_time_s[assigned_ids]
    # Absolute motion time matches AmassMotion.fill_motion_data indexing.
    sample_time_s = start_time_s + motion_reference._timestamp[env_ids].to(motion_buffer.buffer_device)

    command = torch.zeros(len(env_ids), 3, device=motion_reference.device)
    for motion_id in torch.unique(motion_ids).tolist():
        baked = _load_motion_velocity_command(motion_buffer, motion_id, motion_reference.device)
        if baked is None:
            return None
        env_mask = motion_ids == motion_id
        framerate = motion_buffer._all_motion_sequences.framerate[motion_id].to(motion_buffer.buffer_device)
        frame_idx = torch.round(sample_time_s[env_mask] * framerate).to(torch.long)
        frame_idx = torch.clamp(frame_idx, min=0, max=baked.shape[0] - 1)
        command[env_mask.to(motion_reference.device)] = baked[frame_idx.to(motion_reference.device)]
    return command


def _motion_approach_speed(
    motion_reference,
    motion_buffer_name: str,
    x_range: tuple[float, float],
) -> torch.Tensor:
    motion_buffer = motion_reference.motion_buffers[motion_buffer_name]
    env_ids = motion_reference.ALL_INDICES
    assigned_ids = motion_buffer.env_ids_to_assigned_ids(env_ids).to(motion_buffer.buffer_device)
    motion_ids = motion_buffer._assigned_env_motion_selection[assigned_ids]

    approach_speed = torch.zeros(len(env_ids), device=motion_reference.device)
    for motion_id in torch.unique(motion_ids).tolist():
        cache_key = (id(motion_buffer), int(motion_id), str(motion_reference.device), x_range[0], x_range[1])
        if cache_key not in _VAULT_APPROACH_SPEED_CACHE:
            base_pos_w = motion_buffer._all_motion_sequences.base_pos_w[motion_id].to(motion_reference.device)
            base_quat_w = motion_buffer._all_motion_sequences.base_quat_w[motion_id].to(motion_reference.device)
            base_lin_vel_w = motion_buffer._all_motion_sequences.base_lin_vel_w[motion_id].to(motion_reference.device)
            base_lin_vel_b = math_utils.quat_apply_inverse(base_quat_w, base_lin_vel_w)
            horizontal_speed = torch.linalg.norm(base_lin_vel_b[:, :2], dim=-1)
            speed_mask = (base_pos_w[:, 0] < x_range[0]) & (horizontal_speed > 0.2)
            if not speed_mask.any():
                speed_mask = horizontal_speed > 0.2
            _VAULT_APPROACH_SPEED_CACHE[cache_key] = (
                torch.median(horizontal_speed[speed_mask])
                if speed_mask.any()
                else torch.zeros((), device=motion_reference.device)
            )

        env_mask = (motion_ids == motion_id).to(motion_reference.device)
        approach_speed[env_mask] = _VAULT_APPROACH_SPEED_CACHE[cache_key]
    return approach_speed


def vault_velocity_command(
    env,
    command_name: str,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    motion_buffer_name: str = vault_env_cfg.vault_cfg.MOTION_NAME,
    x_range: tuple[float, float] = VAULT_SKILL_COMMAND_X_RANGE,
    enabled: bool = True,
) -> torch.Tensor:
    """Planar high-level velocity command for the student.

    Prefer per-frame ``velocity_command`` baked into the motion npz. If absent,
    fall back to reference base lin vel, optionally with a constant approach-speed
    override inside the vault x-window when enabled=True.
    """
    motion_reference = env.scene[reference_cfg.name]
    baked = _baked_velocity_command(motion_reference, motion_buffer_name)
    if baked is not None:
        return baked

    command = single_frame_velocity_command(env, command_name).clone()
    command[..., 2] = 0.0
    if not enabled:
        return command

    ref_x = (
        motion_reference.data.base_pos_w[
            motion_reference.ALL_INDICES,
            motion_reference.aiming_frame_idx,
            0,
        ]
        - env.scene.env_origins[:, 0]
    )
    vault_mask = (ref_x >= x_range[0]) & (ref_x <= x_range[1])
    if not vault_mask.any():
        return command

    approach_speed = _motion_approach_speed(motion_reference, motion_buffer_name, x_range)
    forward_command = torch.zeros_like(command)
    forward_command[..., 0] = approach_speed
    mask_shape = (command.shape[0],) + (1,) * (command.ndim - 1)
    return torch.where(vault_mask.reshape(mask_shape).expand_as(command), forward_command, command)


def relax_tracking_termination_threshold(
    env,
    env_ids: Sequence[int],
    start_threshold: float = vault_env_cfg.TEACHER_TRACKING_DISTANCE_THRESHOLD,
    end_threshold: float = 1.0,
    num_steps_per_env: int = STUDENT_NUM_STEPS_PER_ENV,
    curriculum_iterations: int = STUDENT_TERMINATION_CURRICULUM_ITERS,
    term_names: tuple[str, ...] = ("base_pos_too_far", "link_pos_too_far"),
) -> dict[str, float]:
    """PHP: linearly relax student tracking terminations from 0.5 m to 1.0 m.

    Uses the same schedule horizon as the DAgger→PPO loss curriculum. DAgger mask
    stays at the teacher threshold (0.5 m) and is intentionally not updated here.
    """
    del env_ids  # curriculum API always provides env_ids; threshold is global.
    if curriculum_iterations <= 0:
        progress = 1.0
    else:
        iteration = env.common_step_counter // max(int(num_steps_per_env), 1)
        progress = min(1.0, max(0.0, float(iteration) / float(curriculum_iterations)))
    threshold = start_threshold + (end_threshold - start_threshold) * progress
    # Play configs disable tracking terminations; skip missing terms instead of crashing.
    active_terms = set(env.termination_manager.active_terms)
    for name in term_names:
        if name not in active_terms:
            continue
        env.termination_manager.get_term_cfg(name).params["distance_threshold"] = threshold
    return {"tracking_distance_threshold": threshold}


def distillation_loss_mask(env) -> torch.Tensor:
    """Disable DAgger when the student is outside the teacher's termination regime."""
    threshold = vault_env_cfg.TEACHER_TRACKING_DISTANCE_THRESHOLD
    expert_base_terminated = instinct_mdp.pos_far_from_ref(
        env,
        asset_cfg=SceneEntityCfg("robot"),
        reference_cfg=SceneEntityCfg("motion_reference"),
        distance_threshold=threshold,
        check_at_keyframe_threshold=-1,
        print_reason=False,
        height_only=False,
    )
    expert_link_terminated = instinct_mdp.link_pos_far_from_ref(
        env,
        asset_cfg=SceneEntityCfg("robot"),
        reference_cfg=SceneEntityCfg(
            "motion_reference",
            body_names=[
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
            preserve_order=True,
        ),
        distance_threshold=threshold,
        in_base_frame=False,
        check_at_keyframe_threshold=-1,
        height_only=False,
        print_reason=False,
    )
    return (~(expert_base_terminated | expert_link_terminated)).to(dtype=torch.float32).unsqueeze(-1)


@configclass
class CommandsCfg(perceptual_cfg.CommandCfg):
    base_lin_vel_ref_command = instinct_mdp.BaseLinVelRefCommandCfg(current_state_command=False)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyObsCfg(ObsGroupCfg):
        # PHP student: single-frame depth (no history stack).
        depth_image = ObsTermCfg(
            func=instinct_mdp.visualizable_image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "distance_to_image_plane_noised",
            },
        )

        velocity_command = ObsTermCfg(
            func=vault_velocity_command,
            params={"command_name": "base_lin_vel_ref_command"},
        )
        projected_gravity = ObsTermCfg(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=DISTILL_PROPRIO_HISTORY_LENGTH,
        )
        base_ang_vel = ObsTermCfg(
            func=mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            history_length=DISTILL_PROPRIO_HISTORY_LENGTH,
        )
        joint_pos = ObsTermCfg(
            func=mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            history_length=DISTILL_PROPRIO_HISTORY_LENGTH,
        )
        joint_vel = ObsTermCfg(
            func=mdp.joint_vel_rel,
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
            history_length=DISTILL_PROPRIO_HISTORY_LENGTH,
        )
        last_action = ObsTermCfg(func=mdp.last_action, history_length=DISTILL_PROPRIO_HISTORY_LENGTH)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticObsCfg(ObsGroupCfg):
        velocity_command = ObsTermCfg(
            func=vault_velocity_command,
            params={"command_name": "base_lin_vel_ref_command"},
        )
        joint_pos_ref = ObsTermCfg(
            func=single_frame_generated_command,
            params={"command_name": "joint_pos_ref_command", "frame_dim": 29},
        )
        joint_vel_ref = ObsTermCfg(
            func=single_frame_generated_command,
            params={"command_name": "joint_vel_ref_command", "frame_dim": 29},
        )
        position_ref = ObsTermCfg(
            func=single_frame_generated_command,
            params={"command_name": "position_b_ref_command", "frame_dim": 3},
        )
        rotation_ref = ObsTermCfg(
            func=single_frame_generated_command,
            params={"command_name": "rotation_ref_command", "frame_dim": 6},
        )
        # PHP student critic is privileged (height scan + motion), not depth.
        height_scan = ObsTermCfg(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=[-1.0, 1.0],
        )
        link_pos = ObsTermCfg(
            func=instinct_mdp.link_pos_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=vault_env_cfg.VAULT_MOTION_REFERENCE_CFG.link_of_interests,
                    preserve_order=True,
                )
            },
        )
        link_rot = ObsTermCfg(
            func=instinct_mdp.link_tannorm_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=vault_env_cfg.VAULT_MOTION_REFERENCE_CFG.link_of_interests,
                    preserve_order=True,
                )
            },
        )
        base_lin_vel = ObsTermCfg(
            func=mdp.base_lin_vel,
            history_length=DISTILL_PROPRIO_HISTORY_LENGTH,
        )
        projected_gravity = ObsTermCfg(
            func=mdp.projected_gravity,
            history_length=DISTILL_PROPRIO_HISTORY_LENGTH,
        )
        base_ang_vel = ObsTermCfg(
            func=mdp.base_ang_vel,
            history_length=DISTILL_PROPRIO_HISTORY_LENGTH,
        )
        joint_pos = ObsTermCfg(
            func=mdp.joint_pos_rel,
            history_length=DISTILL_PROPRIO_HISTORY_LENGTH,
        )
        joint_vel = ObsTermCfg(
            func=mdp.joint_vel_rel,
            history_length=DISTILL_PROPRIO_HISTORY_LENGTH,
        )
        last_action = ObsTermCfg(func=mdp.last_action, history_length=DISTILL_PROPRIO_HISTORY_LENGTH)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class TeacherObsCfg(vault_env_cfg.PhpTeacherPolicyObsCfg):
        """Same privileged actor obs as the PHP teacher checkpoint."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyObsCfg = PolicyObsCfg()
    critic: CriticObsCfg = CriticObsCfg()
    teacher: TeacherObsCfg = TeacherObsCfg()

    @configclass
    class DistillMaskObsCfg(ObsGroupCfg):
        mask = ObsTermCfg(func=distillation_loss_mask)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    distill_mask: DistillMaskObsCfg = DistillMaskObsCfg()


@configclass
class DistillCurriculumCfg(perceptual_cfg.CurriculumCfg):
    """Student curriculum: no adaptive sampling; linear termination relaxation."""

    beyond_adaptive_sampling = None
    relax_tracking_termination = CurriculumTermCfg(
        func=relax_tracking_termination_threshold,
        params={
            "start_threshold": vault_env_cfg.TEACHER_TRACKING_DISTANCE_THRESHOLD,
            "end_threshold": 1.0,
            "num_steps_per_env": STUDENT_NUM_STEPS_PER_ENV,
            "curriculum_iterations": STUDENT_TERMINATION_CURRICULUM_ITERS,
        },
    )


@configclass
class G1PerceptiveVaultVaeDistillEnvCfg(vault_env_cfg.G1PerceptiveVaultEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    curriculum: DistillCurriculumCfg = DistillCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Teacher cfg removes the camera; student needs depth sensing back.
        vault_env_cfg.enable_vault_camera(self, debug_vis=False)
        # Try 4x default parallel envs (4096 -> 16384), matching PHP scale.
        self.scene.num_envs = 16384
        # PHP: disable adaptive sampling for the student (uniform / even skill sampling).
        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None
        self.scene.motion_reference.motion_buffers[
            vault_env_cfg.vault_cfg.MOTION_NAME
        ].env_starting_stub_sampling_strategy = "independent"
        # PHP: start at teacher threshold 0.5 m; curriculum linearly relaxes to 1.0 m.
        # DAgger mask remains fixed at TEACHER_TRACKING_DISTANCE_THRESHOLD (0.5 m).
        self.terminations.base_pos_too_far.params["distance_threshold"] = (
            vault_env_cfg.TEACHER_TRACKING_DISTANCE_THRESHOLD
        )
        self.terminations.link_pos_too_far.params["distance_threshold"] = (
            vault_env_cfg.TEACHER_TRACKING_DISTANCE_THRESHOLD
        )
        self.run_name = "g1PerceptiveVaultVaeDistill"


@configclass
class G1PerceptiveVaultVaeDistillEnvCfg_PLAY(G1PerceptiveVaultVaeDistillEnvCfg):
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=1,
        env_spacing=vault_env_cfg.VAULT_ENV_SPACING,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=vault_env_cfg.VAULT_MOTION_REFERENCE_CFG.replace(debug_vis=True),
    )
    viewer: ViewerCfg = ViewerCfg(
        # Keep the viewer in world coordinates so orbit/pan/zoom does not
        # snap back to the robot on every simulation step.
        eye=[7.0, -9.0, 4.5],
        lookat=[0.0, 0.5, 0.8],
        origin_type="world",
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 6000.0
        self.scene.motion_reference.symmetric_augmentation_joint_mapping = None
        self.scene.motion_reference.visualizing_marker_types = ["relative_links"]
        self.scene.motion_reference.motion_buffers[vault_env_cfg.vault_cfg.MOTION_NAME].motion_start_from_middle_range = [
            0.0,
            0.0,
        ]
        self.scene.motion_reference.motion_buffers[vault_env_cfg.vault_cfg.MOTION_NAME].motion_bin_length_s = None
        self.scene.motion_reference.motion_buffers[
            vault_env_cfg.vault_cfg.MOTION_NAME
        ].env_starting_stub_sampling_strategy = "independent"
        self.curriculum.beyond_adaptive_sampling = None
        self.curriculum.relax_tracking_termination = None
        self.events.bin_fail_counter_smoothing = None
        getattr(self.events, "reset_vault_box").params["pose_range"] = {}

        self.scene.camera.debug_vis = True
        self.observations.policy.depth_image.params["debug_vis"] = True

        self.terminations.base_pos_too_far = None
        self.terminations.base_pg_too_far = None
        self.terminations.link_pos_too_far = None

        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.physics_material = None
        setattr(self.events, "push_robot", None)
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
