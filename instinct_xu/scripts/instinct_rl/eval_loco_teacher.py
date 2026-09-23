"""Evaluate one explicit unified Loco teacher checkpoint and issue its manifest on pass."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, help="Exact model_*.pt path; latest-checkpoint lookup is forbidden.")
parser.add_argument("--task", default="Instinct-Unified-OmniLoco-G1-v0")
parser.add_argument("--mode", choices=("fixed", "random", "both"), default="both")
parser.add_argument("--fixed_replicas", type=int, default=64)
parser.add_argument("--random_samples", type=int, default=2048)
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--transient", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default="/tmp/loco_teacher_a5_eval.json")
parser.add_argument("--manifest", default=None, help="Manifest path; defaults beside the checkpoint.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from instinct_rl.runners import OnPolicyRunner

from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat
from isaaclab_tasks.utils import parse_env_cfg

import instinctlab.tasks  # noqa: F401
from instinctlab.tasks.locomotion.config.g1.agents.omni_loco_ppo_cfg import G1OmniLocoPPORunnerCfg
from instinctlab.tasks.locomotion.config.g1.omni_loco_env_cfg import (
    OMNI_LOCO_COMMAND_RANGES,
    OMNI_LOCO_OBSERVATION_DIMS,
    OMNI_LOCO_OBSERVATION_TERMS,
)
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper


FIXED_COMMANDS = (
    ("C0_stand", (0.0, 0.0, 0.0)),
    ("C1_vx_pos", (1.0, 0.0, 0.0)),
    ("C2_vx_neg", (-0.5, 0.0, 0.0)),
    ("C3_vy_pos", (0.0, 0.5, 0.0)),
    ("C4_vy_neg", (0.0, -0.5, 0.0)),
    ("C5_wz_pos", (0.0, 0.0, 1.0)),
    ("C6_wz_neg", (0.0, 0.0, -1.0)),
    ("C7_combined", (1.0, 0.5, 1.0)),
)
GATES = {"vxy_rmse_max": 0.20, "wz_rmse_max": 0.20, "fall_rate_max_exclusive": 0.05}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_structure(checkpoint: str) -> tuple[dict, dict]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = state.get("model_state_dict", {})
    actor_weights = []
    for name, value in model.items():
        if name.startswith("actor.") and name.endswith(".weight") and value.ndim == 2:
            actor_weights.append((int(name.split(".")[1]), name, value))
    actor_weights.sort()
    normalizer = state.get("policy_normalizer_state_dict")
    normalizer_dim = None if normalizer is None or "_mean" not in normalizer else int(normalizer["_mean"].shape[-1])
    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from tensors(item)

    finite = all(bool(torch.isfinite(value).all()) for value in tensors(state))
    checks = {
        "checkpoint_exists": True,
        "actor_layers_found": bool(actor_weights),
        "actor_input_623": bool(actor_weights) and int(actor_weights[0][2].shape[1]) == 623,
        "actor_output_29": bool(actor_weights) and int(actor_weights[-1][2].shape[0]) == 29,
        "normalizer_present": normalizer is not None,
        "normalizer_input_623": normalizer_dim == 623,
        "all_checkpoint_tensors_finite": finite,
    }
    details = {
        "actor_first_weight": None if not actor_weights else [actor_weights[0][1], list(actor_weights[0][2].shape)],
        "actor_last_weight": None if not actor_weights else [actor_weights[-1][1], list(actor_weights[-1][2].shape)],
        "normalizer_dim": normalizer_dim,
        "iteration": int(state.get("iter", -1)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Checkpoint structural validation failed: {checks}; details={details}")
    return state, {"checks": checks, "details": details, "pass": True}


def _direct_command(env, commands: torch.Tensor):
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term.vel_command_b[:] = commands
    term.is_standing_env[:] = False
    term.is_heading_env[:] = False
    term.time_left[:] = args.duration + 2.0


def _new_accumulator(num_envs: int, num_joints: int, device: str):
    return {
        "vxy_sq": torch.zeros(num_envs, 2, device=device),
        "wz_sq": torch.zeros(num_envs, device=device),
        "count": torch.zeros(num_envs, device=device),
        "action_sum": torch.zeros(num_envs, device=device),
        "action_sq": torch.zeros(num_envs, device=device),
        "action_count": torch.zeros(num_envs, device=device),
        "action_abs_max": torch.zeros(num_envs, device=device),
        "action_per_joint_abs_max": torch.zeros(num_envs, num_joints, device=device),
        "action_rate_sq": torch.zeros(num_envs, device=device),
        "action_rate_count": torch.zeros(num_envs, device=device),
        "joint_sq": torch.zeros(num_envs, num_joints, device=device),
        "joint_count": torch.zeros(num_envs, device=device),
        "joint_limit_min_normalized_margin": torch.full((num_envs, num_joints), torch.inf, device=device),
        "joint_limit_violation_count": torch.zeros(num_envs, device=device),
        "joint_limit_sample_count": torch.zeros(num_envs, device=device),
    }


def _accumulate(acc, valid, velocity_error, yaw_error, action, previous_action, joint_error, joint_limit_margin):
    ids = valid.nonzero(as_tuple=False).flatten()
    if ids.numel() == 0:
        return
    acc["vxy_sq"][ids] += velocity_error[ids].square()
    acc["wz_sq"][ids] += yaw_error[ids].square()
    acc["count"][ids] += 1
    acc["action_sum"][ids] += action[ids].sum(dim=1)
    acc["action_sq"][ids] += action[ids].square().sum(dim=1)
    acc["action_count"][ids] += action.shape[1]
    acc["action_abs_max"][ids] = torch.maximum(acc["action_abs_max"][ids], action[ids].abs().amax(dim=1))
    acc["action_per_joint_abs_max"][ids] = torch.maximum(
        acc["action_per_joint_abs_max"][ids], action[ids].abs()
    )
    acc["action_rate_sq"][ids] += (action[ids] - previous_action[ids]).square().sum(dim=1)
    acc["action_rate_count"][ids] += action.shape[1]
    acc["joint_sq"][ids] += joint_error[ids].square()
    acc["joint_count"][ids] += 1
    acc["joint_limit_min_normalized_margin"][ids] = torch.minimum(
        acc["joint_limit_min_normalized_margin"][ids], joint_limit_margin[ids]
    )
    acc["joint_limit_violation_count"][ids] += (joint_limit_margin[ids] < 0.0).sum(dim=1)
    acc["joint_limit_sample_count"][ids] += joint_limit_margin.shape[1]


def _summarize(
    acc,
    selection,
    fall,
    nan_env,
    joint_names,
    commands,
    final_position,
    initial_position,
    final_heading,
    initial_heading,
):
    ids = selection.nonzero(as_tuple=False).flatten()
    count = acc["count"][ids].sum().clamp_min(1.0)
    joint_count = acc["joint_count"][ids].sum().clamp_min(1.0)
    action_count = acc["action_count"][ids].sum().clamp_min(1.0)
    action_rate_count = acc["action_rate_count"][ids].sum().clamp_min(1.0)
    vxy_sum = acc["vxy_sq"][ids].sum(dim=0)
    joint_rmse = torch.sqrt(acc["joint_sq"][ids].sum(dim=0) / joint_count)
    action_mean = acc["action_sum"][ids].sum() / action_count
    action_variance = acc["action_sq"][ids].sum() / action_count - action_mean.square()
    displacement = final_position[ids, :2] - initial_position[ids, :2]
    yaw_drift = wrap_to_pi(final_heading[ids] - initial_heading[ids])
    selected_commands = commands[ids]
    theta_0 = initial_heading[ids]
    theta_1 = theta_0 + selected_commands[:, 2] * args.duration
    angular = selected_commands[:, 2]
    safe_angular = torch.where(angular.abs() < 1.0e-6, torch.ones_like(angular), angular)
    expected_displacement = torch.stack(
        (
            (
                selected_commands[:, 0] * (torch.sin(theta_1) - torch.sin(theta_0))
                + selected_commands[:, 1] * (torch.cos(theta_1) - torch.cos(theta_0))
            )
            / safe_angular,
            (
                selected_commands[:, 0] * (torch.cos(theta_0) - torch.cos(theta_1))
                + selected_commands[:, 1] * (torch.sin(theta_1) - torch.sin(theta_0))
            )
            / safe_angular,
        ),
        dim=1,
    )
    straight_displacement = args.duration * torch.stack(
        (
            selected_commands[:, 0] * torch.cos(theta_0) - selected_commands[:, 1] * torch.sin(theta_0),
            selected_commands[:, 0] * torch.sin(theta_0) + selected_commands[:, 1] * torch.cos(theta_0),
        ),
        dim=1,
    )
    expected_displacement = torch.where(
        (angular.abs() < 1.0e-6).unsqueeze(1), straight_displacement, expected_displacement
    )
    displacement_error = displacement - expected_displacement
    cos_0 = torch.cos(theta_0)
    sin_0 = torch.sin(theta_0)
    lateral_displacement = -sin_0 * displacement[:, 0] + cos_0 * displacement[:, 1]
    expected_final_heading = wrap_to_pi(theta_1)
    final_yaw_error = wrap_to_pi(final_heading[ids] - expected_final_heading)
    joint_limit_margin = acc["joint_limit_min_normalized_margin"][ids].amin(dim=0)
    joint_limit_samples = acc["joint_limit_sample_count"][ids].sum().clamp_min(1.0)
    result = {
        "replicas": int(ids.numel()),
        "vx_rmse": float(torch.sqrt(vxy_sum[0] / count)),
        "vy_rmse": float(torch.sqrt(vxy_sum[1] / count)),
        "vxy_rmse": float(torch.sqrt(vxy_sum.sum() / count)),
        "wz_rmse": float(torch.sqrt(acc["wz_sq"][ids].sum() / count)),
        "fall_rate": float(fall[ids].float().mean()),
        "nan_rate": float(nan_env[ids].float().mean()),
        "action_mean": float(action_mean),
        "action_std": float(torch.sqrt(torch.clamp_min(action_variance, 0.0))),
        "action_abs_max": float(acc["action_abs_max"][ids].max()),
        "action_rate_rmse": float(torch.sqrt(acc["action_rate_sq"][ids].sum() / action_rate_count)),
        "mean_displacement_xy": [float(value) for value in displacement.mean(dim=0)],
        "displacement_error_rmse": float(torch.sqrt(torch.mean(torch.sum(displacement_error.square(), dim=1)))),
        "world_y_drift_rmse": float(torch.sqrt(torch.mean(displacement[:, 1].square()))),
        "initial_yaw_frame_lateral_drift_rmse": float(torch.sqrt(torch.mean(lateral_displacement.square()))),
        "yaw_drift_rmse": float(torch.sqrt(torch.mean(yaw_drift.square()))),
        "final_yaw_tracking_error_rmse": float(torch.sqrt(torch.mean(final_yaw_error.square()))),
        "joint_position_error_rmse": {
            name: float(value) for name, value in zip(joint_names, joint_rmse)
        },
        "action_per_joint_abs_max": {
            name: float(value) for name, value in zip(joint_names, acc["action_per_joint_abs_max"][ids].amax(dim=0))
        },
        "joint_limit_min_normalized_margin": {
            name: float(value) for name, value in zip(joint_names, joint_limit_margin)
        },
        "joint_limit_violation_rate": float(acc["joint_limit_violation_count"][ids].sum() / joint_limit_samples),
    }
    result["pass"] = (
        result["vxy_rmse"] <= GATES["vxy_rmse_max"]
        and result["wz_rmse"] <= GATES["wz_rmse_max"]
        and result["fall_rate"] < GATES["fall_rate_max_exclusive"]
        and result["nan_rate"] == 0.0
    )
    return result


def _evaluate_batch(env, policy, commands: torch.Tensor, active_count: int, joint_names):
    obs, _ = env.reset()
    all_commands = torch.zeros(env.num_envs, 3, device=env.device)
    all_commands[:active_count] = commands.to(env.device)
    _direct_command(env, all_commands)
    obs, _ = env.get_observations()

    robot = env.unwrapped.scene["robot"]
    initial_position = robot.data.root_pos_w.clone()
    initial_heading = robot.data.heading_w.clone()
    last_valid_position = initial_position.clone()
    last_valid_heading = initial_heading.clone()
    fall = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    nan_env = torch.zeros_like(fall)
    alive = torch.arange(env.num_envs, device=env.device) < active_count
    previous_action = torch.zeros(env.num_envs, 29, device=env.device)
    full = _new_accumulator(env.num_envs, 29, env.device)
    steady = _new_accumulator(env.num_envs, 29, env.device)
    num_steps = int(round(args.duration / env.unwrapped.step_dt))
    transient_steps = int(round(args.transient / env.unwrapped.step_dt))

    for step in range(num_steps):
        _direct_command(env, all_commands)
        obs, _ = env.get_observations()
        with torch.inference_mode():
            action = policy(obs)
        current_nan = ~torch.isfinite(action).all(dim=1) | ~torch.isfinite(obs).all(dim=1)
        nan_env |= current_nan & alive
        action = torch.nan_to_num(action)
        _, _, dones, _ = env.step(action)
        done = dones.bool()

        velocity_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
        velocity_error = velocity_yaw[:, :2] - all_commands[:, :2]
        yaw_error = robot.data.root_ang_vel_w[:, 2] - all_commands[:, 2]
        joint_error = robot.data.joint_pos - robot.data.default_joint_pos
        limits = robot.data.soft_joint_pos_limits
        limit_range = (limits[..., 1] - limits[..., 0]).clamp_min(1.0e-6)
        joint_limit_margin = torch.minimum(
            robot.data.joint_pos - limits[..., 0], limits[..., 1] - robot.data.joint_pos
        ) / limit_range
        state_nan = (
            ~torch.isfinite(velocity_error).all(dim=1)
            | ~torch.isfinite(yaw_error)
            | ~torch.isfinite(joint_error).all(dim=1)
        )
        nan_env |= state_nan & alive
        valid = alive & ~done & ~current_nan & ~state_nan
        _accumulate(full, valid, velocity_error, yaw_error, action, previous_action, joint_error, joint_limit_margin)
        if step >= transient_steps:
            _accumulate(
                steady, valid, velocity_error, yaw_error, action, previous_action, joint_error, joint_limit_margin
            )
        last_valid_position[valid] = robot.data.root_pos_w[valid]
        last_valid_heading[valid] = robot.data.heading_w[valid]
        fall |= done & alive
        alive &= ~done
        previous_action.copy_(action)

    return {
        "full": full,
        "steady": steady,
        "fall": fall,
        "nan_env": nan_env,
        "initial_position": initial_position,
        "commands": all_commands,
        "final_position": last_valid_position,
        "initial_heading": initial_heading,
        "final_heading": last_valid_heading,
    }


def _fixed_evaluation(env, policy, joint_names):
    commands = torch.tensor([command for _, command in FIXED_COMMANDS], dtype=torch.float32).repeat_interleave(
        args.fixed_replicas, dim=0
    )
    result = _evaluate_batch(env, policy, commands, len(commands), joint_names)
    cases = {}
    for index, (case_name, command) in enumerate(FIXED_COMMANDS):
        selection = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        selection[index * args.fixed_replicas : (index + 1) * args.fixed_replicas] = True
        case = {"command": list(command)}
        for window in ("full", "steady"):
            case[window] = _summarize(
                result[window], selection, result["fall"], result["nan_env"], joint_names, result["commands"],
                result["final_position"], result["initial_position"], result["final_heading"], result["initial_heading"]
            )
        # A5 keeps the full 0--10 s window for diagnosis, while the formal
        # tracking gate applies to the 1--10 s steady-state window.
        case["pass"] = case["steady"]["pass"]
        cases[case_name] = case
    return {"replicas_per_command": args.fixed_replicas, "cases": cases, "pass": all(c["pass"] for c in cases.values())}


def _random_commands(count: int):
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    columns = [
        torch.empty(count).uniform_(low, high, generator=generator) for low, high in OMNI_LOCO_COMMAND_RANGES
    ]
    return torch.stack(columns, dim=1)


def _random_evaluation(env, policy, joint_names):
    commands = _random_commands(args.random_samples)
    result = _evaluate_batch(env, policy, commands, args.random_samples, joint_names)
    active = torch.arange(env.num_envs, device=env.device) < args.random_samples
    report = {}
    for window in ("full", "steady"):
        report[window] = _summarize(
            result[window], active, result["fall"], result["nan_env"], joint_names, result["commands"],
            result["final_position"], result["initial_position"], result["final_heading"], result["initial_heading"]
        )

    bins = {}
    commands_device = commands.to(env.device)
    for axis_index, (axis, limits) in enumerate(zip(("vx", "vy", "wz"), OMNI_LOCO_COMMAND_RANGES)):
        edges = torch.linspace(limits[0], limits[1], 4, device=env.device)
        for bin_index, bin_name in enumerate(("low", "mid", "high")):
            if bin_index == 2:
                selection = (commands_device[:, axis_index] >= edges[bin_index]) & (
                    commands_device[:, axis_index] <= edges[bin_index + 1]
                )
            else:
                selection = (commands_device[:, axis_index] >= edges[bin_index]) & (
                    commands_device[:, axis_index] < edges[bin_index + 1]
                )
            padded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            padded[: args.random_samples] = selection
            bins[f"{axis}_{bin_name}"] = _summarize(
                result["steady"], padded, result["fall"], result["nan_env"], joint_names, result["commands"],
                result["final_position"], result["initial_position"], result["final_heading"], result["initial_heading"]
            )
    report["command_bins_steady"] = bins
    report["num_commands"] = args.random_samples
    # The full window includes the commanded acceleration transient and is
    # reported for diagnosis.  A5 admission uses steady-state performance.
    report["pass"] = report["steady"]["pass"]
    return report


def _runtime_actuators(robot):
    result = {}
    for group_name, actuator in robot.actuators.items():
        group = {"class": type(actuator).__name__, "joint_names": list(actuator._joint_names)}
        for parameter in ("stiffness", "damping", "armature", "effort_limit_sim", "velocity_limit_sim"):
            value = getattr(actuator, parameter)
            group[parameter] = [float(item) for item in value[0].detach().cpu()]
        result[group_name] = group
    return result


def _normalizer_manifest(state):
    normalizer = state["policy_normalizer_state_dict"]
    result = {}
    for name, value in normalizer.items():
        if isinstance(value, torch.Tensor):
            flat = value.detach().cpu().flatten()
            result[name] = float(flat[0]) if flat.numel() == 1 else [float(item) for item in flat]
        else:
            result[name] = value
    return result


def main():
    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Explicit checkpoint does not exist: {checkpoint}")
    if args.duration != 10.0 or args.transient != 1.0:
        print("[WARN] A5 release requires duration=10.0 and transient=1.0; this run cannot issue a manifest.")
    state, structure = _checkpoint_structure(checkpoint)

    fixed_count = len(FIXED_COMMANDS) * args.fixed_replicas if args.mode in ("fixed", "both") else 0
    random_count = args.random_samples if args.mode in ("random", "both") else 0
    num_envs = max(fixed_count, random_count)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)
    env_cfg.seed = args.seed
    env_cfg.commands.base_velocity.debug_vis = False
    env = gym.make(args.task, cfg=env_cfg)
    env = InstinctRlVecEnvWrapper(env)

    agent_cfg = G1OmniLocoPPORunnerCfg()
    agent_cfg.device = args.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.device)
    joint_names = list(env.unwrapped.scene["robot"].joint_names)

    report = {
        "task": args.task,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_structure": structure,
        "evaluation": {
            "duration_s": args.duration,
            "steady_window_s": [args.transient, args.duration],
            "gate_window": "steady",
            "control_hz": 1.0 / env.unwrapped.step_dt,
            "linear_velocity_frame": "gravity-aligned yaw frame",
            "yaw_rate_frame": "world z (same as training reward)",
            "gates": GATES,
        },
    }
    if args.mode in ("fixed", "both"):
        report["fixed"] = _fixed_evaluation(env, policy, joint_names)
    if args.mode in ("random", "both"):
        report["random"] = _random_evaluation(env, policy, joint_names)

    complete_protocol = (
        args.mode == "both"
        and args.fixed_replicas >= 64
        and args.random_samples >= 2048
        and args.duration == 10.0
        and args.transient == 1.0
    )
    report["protocol_complete"] = complete_protocol
    report["pass"] = complete_protocol and report["fixed"]["pass"] and report["random"]["pass"]

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"[A5] report={output} pass={report['pass']}")

    if report["pass"]:
        manifest_path = os.path.abspath(args.manifest or os.path.join(os.path.dirname(checkpoint), "loco_expert_manifest.json"))
        robot = env.unwrapped.scene["robot"]
        manifest = {
            "schema_version": 1,
            "expert": "unified_omni_loco_g1_plane",
            "checkpoint": checkpoint,
            "checkpoint_sha256": report["checkpoint_sha256"],
            "checkpoint_iteration": structure["details"]["iteration"],
            "observation": {
                "total_dim": 623,
                "terms": [
                    {"name": name, "dim": dim}
                    for name, dim in zip(OMNI_LOCO_OBSERVATION_TERMS, OMNI_LOCO_OBSERVATION_DIMS)
                ],
                "height_scan": {"dim": 527, "grid": [31, 17], "resolution_m": 0.1, "size_m": [3.0, 1.6]},
            },
            "action": {
                "dim": 29,
                "type": "JointPositionAction",
                "scale": 1.0,
                "use_default_offset": True,
                "joint_order": joint_names,
                "default_joint_position": [float(value) for value in robot.data.default_joint_pos[0].detach().cpu()],
            },
            "timing": {"physics_dt": 0.005, "decimation": 4, "control_dt": 0.02, "control_hz": 50.0},
            "command": {
                "terms": ["vx", "vy", "wz"],
                "ranges": {axis: list(bounds) for axis, bounds in zip(("vx", "vy", "wz"), OMNI_LOCO_COMMAND_RANGES)},
                "resampling_time_s": 10.0,
                "heading_command": False,
                "standing_injection": False,
                "curriculum": False,
            },
            "terrain": "plane",
            "actuators": _runtime_actuators(robot),
            "policy_normalizer": _normalizer_manifest(state),
            "evaluation_report": output,
            "evaluation_pass": True,
        }
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        report["manifest"] = manifest_path
        with open(output, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(f"[A5 PASS] manifest={manifest_path}")
    else:
        print("[A5 FAIL] Manifest was not issued.")

    env.close()
    if not report["pass"]:
        raise RuntimeError(f"Loco teacher did not pass A5; see {output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
