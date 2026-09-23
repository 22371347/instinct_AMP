"""Headless, paired ablations for the G1 vault vision student.

The policy input is modified *after* the environment computes observations.
This lets an external speed command replace the NPZ-derived command without
changing the teacher, checkpoint, reward, or physics configuration. Physical
crossing/upright metrics are used instead of reference-tracking reward.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


@dataclass(frozen=True)
class Case:
    name: str
    depth: str = "normal"
    command: str = "npz"
    box_x: float = 0.0
    box_y: float = 0.0
    box_yaw_deg: float = 0.0


CASES = {
    case.name: case
    for case in (
        Case("baseline"),
        Case("frozen_depth", depth="frozen"),
        Case("far_depth", depth="far"),
        Case("box_near", box_x=-0.25),
        Case("box_far", box_x=0.25),
        Case("box_yaw30", box_yaw_deg=30.0),
        Case("external_speed", command="constant_1mps"),
        Case("external_speed_box_near", command="constant_1mps", box_x=-0.25),
    )
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path(__file__).resolve().parents[2]
    / "logs/instinct_rl/g1_perceptive_vault_vae_distill"
    / "20260918_174642_g1PerceptiveVaultVaeDistill_vault_student_teacher8000/model_20000.pt",
)
parser.add_argument(
    "--teacher-logdir",
    type=Path,
    default=Path(__file__).resolve().parents[2]
    / "logs/instinct_rl/g1_perceptive_vault_shadowing"
    / "20260918_161738_g1PerceptiveVaultPhpTeacher_concatMotionBins_vault_teacher_resume7000_from20260918_112019",
)
parser.add_argument("--cases", type=str, default=",".join(CASES))
parser.add_argument("--replicates", type=int, default=4)
parser.add_argument("--duration", type=float, default=14.5)
parser.add_argument("--output", type=Path, default=Path("/tmp/vault_student_ablation.json"))
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

selected_names = [name.strip() for name in args.cases.split(",") if name.strip()]
unknown = sorted(set(selected_names) - CASES.keys())
if unknown:
    parser.error(f"Unknown cases: {unknown}; available: {list(CASES)}")
if not selected_names or args.replicates < 1 or args.duration <= 0:
    parser.error("Select at least one case, positive replicates, and positive duration")
if not args.checkpoint.is_file() or not (args.teacher_logdir / "model_8000.pt").is_file():
    parser.error("Student checkpoint or teacher model_8000.pt is missing")
teacher_models = sorted(
    args.teacher_logdir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[-1])
)
if teacher_models[-1].name != "model_8000.pt":
    parser.error(
        f"The distillation loader selects the newest teacher checkpoint, currently {teacher_models[-1].name}; "
        "this evaluation expects model_8000.pt"
    )

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.utils.io import load_yaml  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from instinct_rl.runners import OnPolicyRunner  # noqa: E402

import instinctlab.tasks  # noqa: E402,F401
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper  # noqa: E402


def policy_slices(env: InstinctRlVecEnvWrapper) -> dict[str, slice]:
    # The wrapper exposes the same ordered terms it uses to flatten policy
    # observations; ObservationManager has no group_obs_term_names attribute.
    obs_format = env.get_obs_format()["policy"]
    offset = 0
    result = {}
    for name, shape in obs_format.items():
        width = math.prod(shape)
        result[name] = slice(offset, offset + width)
        offset += width
    if result["depth_image"].stop - result["depth_image"].start != 1536:
        raise ValueError(f"Unexpected depth shape: {obs_format['depth_image']}")
    if result["velocity_command"].stop - result["velocity_command"].start != 3:
        raise ValueError("Velocity command is not three-dimensional")
    return result


def main() -> None:
    cases = [CASES[name] for name in selected_names]
    assigned = [case for case in cases for _ in range(args.replicates)]
    num_envs = len(assigned)
    task = "Instinct-Perceptive-Vault-VaeDistill-G1-Play-v0"

    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=num_envs)
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = num_envs
    # Remove all display/debug elements; no actor observations are removed.
    env_cfg.scene.robot_reference = None
    env_cfg.scene.motion_reference.debug_vis = False
    env_cfg.scene.motion_reference.visualizing_marker_types = []
    env_cfg.scene.camera.debug_vis = False
    env_cfg.scene.height_scanner.debug_vis = False
    env_cfg.observations.policy.depth_image.params["debug_vis"] = False
    env_cfg.events.randomize_ray_offsets = None
    # Avoid automatic reset: follow the same physical trajectory until the
    # chosen time (14.5 s is shorter than the 14.74 s NPZ clip).
    env_cfg.terminations.dataset_exhausted = None
    env_cfg.terminations.illegal_reset_contact = None
    env_cfg.terminations.out_of_border = None

    agent_cfg = load_yaml(args.checkpoint.parent / "params/agent.yaml")
    agent_cfg["algorithm"]["teacher_logdir"] = str(args.teacher_logdir.resolve())
    agent_cfg["device"] = "cuda:0"

    raw_env = gym.make(task, cfg=env_cfg)
    env = InstinctRlVecEnvWrapper(raw_env)
    try:
        print(f"[eval] Environment ready: {num_envs} envs, cases={selected_names}", flush=True)
        runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device="cuda:0")
        print(f"[eval] Loading student checkpoint: {args.checkpoint}", flush=True)
        runner.load(str(args.checkpoint.resolve()))
        policy = runner.get_inference_policy(device=env.device)
        sl = policy_slices(env)
        print(f"[eval] Checkpoint loaded; policy slices: {sl}", flush=True)
        obs, _ = env.reset()
        if obs.shape[0] != num_envs or obs.shape[1] != max(s.stop for s in sl.values()):
            raise ValueError(f"Unexpected policy observation shape: {obs.shape}; slices={sl}")

        box = env.unwrapped.scene["vault_box"]
        origins = env.unwrapped.scene.env_origins
        box_base = box.data.default_root_state.clone()
        box_base[:, :3] += origins
        box_xy = box_base[:, :2].clone()
        # Small, *paired* obstacle variations make replicates distinct while
        # preserving identical initial conditions between ablation cases.
        jitter = [
            (i % args.replicates - (args.replicates - 1) / 2)
            / max((args.replicates - 1) / 2, 1)
            for i in range(num_envs)
        ]
        yaw_rad = torch.tensor(
            [math.radians(case.box_yaw_deg + 3.0 * jitter[i]) for i, case in enumerate(assigned)],
            device=env.device,
        )
        for i, case in enumerate(assigned):
            box_base[i, 0] += case.box_x + 0.025 * jitter[i]
            box_base[i, 1] += case.box_y + 0.02 * jitter[i]
            box_xy[i, 0] += case.box_x + 0.025 * jitter[i]
            box_xy[i, 1] += case.box_y + 0.02 * jitter[i]
        zero = torch.zeros(num_envs, device=env.device)
        box_base[:, 3:7] = math_utils.quat_mul(
            math_utils.quat_from_euler_xyz(zero, zero, yaw_rad), box_base[:, 3:7]
        )
        env_ids = torch.arange(num_envs, device=env.device)
        box.write_root_pose_to_sim(box_base[:, :7], env_ids=env_ids)
        box.write_root_velocity_to_sim(torch.zeros_like(box_base[:, 7:]), env_ids=env_ids)
        print(f"[eval] Obstacles configured; rolling out {args.duration:.2f} s", flush=True)

        frozen_depth = obs[:, sl["depth_image"]].clone()
        modes = [case.depth for case in assigned]
        commands = [case.command for case in assigned]
        frozen_ids = torch.tensor([i for i, mode in enumerate(modes) if mode == "frozen"], device=env.device)
        far_ids = torch.tensor([i for i, mode in enumerate(modes) if mode == "far"], device=env.device)
        speed_ids = torch.tensor(
            [i for i, mode in enumerate(commands) if mode == "constant_1mps"], device=env.device
        )
        baseline_ids = torch.tensor(
            [i for i, case in enumerate(assigned) if case.name == "baseline"], device=env.device
        )

        dt = env.unwrapped.step_dt
        steps = round(args.duration / dt)
        if steps * dt >= 14.74:
            raise ValueError("Duration must end before the 14.74 s reference clip ends")
        crossed = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        crossing_time = torch.full((num_envs,), float("nan"), device=env.device)
        max_local_x = torch.full((num_envs,), -float("inf"), device=env.device)
        min_height = torch.full((num_envs,), float("inf"), device=env.device)
        min_height_after_cross = torch.full((num_envs,), float("inf"), device=env.device)
        last_upright_count = torch.zeros(num_envs, device=env.device)
        last_window_count = 0
        all_done = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        action_delta_sum = 0.0
        image_delta_sum = 0.0
        counterfactual_count = 0
        up_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(num_envs, 1)

        with torch.inference_mode():
            for step in range(steps):
                policy_obs = obs.clone()
                if len(frozen_ids):
                    policy_obs[frozen_ids, sl["depth_image"]] = frozen_depth[frozen_ids]
                if len(far_ids):
                    policy_obs[far_ids, sl["depth_image"]] = 1.0
                if len(speed_ids):
                    policy_obs[speed_ids, sl["velocity_command"]] = torch.tensor(
                        [1.0, 0.0, 0.0], device=env.device
                    )

                # At exactly the same robot state, change only depth and see
                # whether the student action changes near the obstacle.
                if len(baseline_ids) and step % 10 == 0:
                    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w - origins
                    near = baseline_ids[
                        (robot_pos[baseline_ids, 0] > 0.0) & (robot_pos[baseline_ids, 0] < 2.5)
                    ]
                    if len(near):
                        probe = policy_obs[near].clone()
                        far_probe = probe.clone()
                        far_probe[:, sl["depth_image"]] = 1.0
                        normal_action = policy(probe)
                        far_action = policy(far_probe)
                        action_delta_sum += float(
                            (normal_action - far_action).square().mean(dim=1).sqrt().sum()
                        )
                        image_delta_sum += float(
                            (probe[:, sl["depth_image"]] - 1.0).abs().mean(dim=1).sum()
                        )
                        counterfactual_count += len(near)

                actions = policy(policy_obs)
                obs, _, dones, _ = env.step(actions)
                all_done |= dones.reshape(-1).bool()

                robot = env.unwrapped.scene["robot"]
                root_pos = robot.data.root_pos_w - origins
                root_up = math_utils.quat_apply(robot.data.root_quat_w, up_axis)[:, 2]
                delta_xy = root_pos[:, :2] - (box_xy - origins[:, :2])
                local_x = torch.cos(yaw_rad) * delta_xy[:, 0] + torch.sin(yaw_rad) * delta_xy[:, 1]
                newly_crossed = (~crossed) & (local_x > 0.8)
                crossing_time[newly_crossed] = (step + 1) * dt
                crossed |= newly_crossed
                max_local_x = torch.maximum(max_local_x, local_x)
                min_height = torch.minimum(min_height, root_pos[:, 2])
                min_height_after_cross = torch.where(
                    crossed, torch.minimum(min_height_after_cross, root_pos[:, 2]), min_height_after_cross
                )
                if (step + 1) * dt >= args.duration - 1.0:
                    last_upright_count += (root_pos[:, 2] > 0.6) & (root_up > 0.7)
                    last_window_count += 1
                if step == 0 or step % 100 == 99 or step == steps - 1:
                    print(
                        f"step={step + 1}/{steps} t={(step + 1) * dt:.2f}s "
                        f"crossed={int(crossed.sum())}/{num_envs} "
                        f"mean_x={float(root_pos[:, 0].mean()):.2f}",
                        flush=True,
                    )

        final_root = env.unwrapped.scene["robot"].data.root_pos_w - origins
        final_up = math_utils.quat_apply(env.unwrapped.scene["robot"].data.root_quat_w, up_axis)[:, 2]
        final_delta = final_root[:, :2] - (box_xy - origins[:, :2])
        final_local_x = torch.cos(yaw_rad) * final_delta[:, 0] + torch.sin(yaw_rad) * final_delta[:, 1]
        stable_fraction = last_upright_count / max(last_window_count, 1)
        success = (
            crossed
            & (final_local_x > 0.8)
            & (final_root[:, 2] > 0.6)
            & (final_up > 0.7)
            & (stable_fraction >= 0.8)
            & ~all_done
        )

        def finite_or_none(value: torch.Tensor) -> float | None:
            result = float(value)
            return result if math.isfinite(result) else None

        rows = []
        for i, case in enumerate(assigned):
            rows.append(
                {
                    "case": case.name,
                    "replicate": i % args.replicates,
                    "box_offset_x_m": case.box_x + 0.025 * jitter[i],
                    "box_offset_y_m": case.box_y + 0.02 * jitter[i],
                    "box_yaw_deg": case.box_yaw_deg + 3.0 * jitter[i],
                    "success": bool(success[i]),
                    "crossed": bool(crossed[i]),
                    "crossing_time_s": finite_or_none(crossing_time[i]),
                    "max_local_x_m": float(max_local_x[i]),
                    "final_local_x_m": float(final_local_x[i]),
                    "final_root_height_m": float(final_root[i, 2]),
                    "final_up_z": float(final_up[i]),
                    "last_second_upright_fraction": float(stable_fraction[i]),
                    "min_root_height_m": float(min_height[i]),
                    "min_root_height_after_cross_m": finite_or_none(min_height_after_cross[i]),
                    "auto_reset": bool(all_done[i]),
                }
            )
        summary = {}
        for case in cases:
            members = [row for row in rows if row["case"] == case.name]
            summary[case.name] = {
                "success": sum(row["success"] for row in members),
                "crossed": sum(row["crossed"] for row in members),
                "total": len(members),
                "mean_final_local_x_m": sum(row["final_local_x_m"] for row in members) / len(members),
                "mean_final_root_height_m": sum(row["final_root_height_m"] for row in members) / len(members),
            }
        result = {
            "checkpoint": str(args.checkpoint.resolve()),
            "teacher_logdir": str(args.teacher_logdir.resolve()),
            "seed": args.seed,
            "duration_s": steps * dt,
            "box_size_m": [0.4, 10.0, 0.6],
            "paired_replicate_jitter": "x +/-0.025 m, y +/-0.02 m, yaw +/-3 deg, shared across cases",
            "success_definition": "center crosses box far face by 0.6 m, then upright >=80% of last second and at end",
            "case_definitions": {case.name: vars(case) for case in cases},
            "summary": summary,
            "counterfactual_same_state": {
                "baseline_normal_vs_all_far_depth_action_rmse": (
                    action_delta_sum / counterfactual_count if counterfactual_count else None
                ),
                "mean_depth_abs_difference_from_far": (
                    image_delta_sum / counterfactual_count if counterfactual_count else None
                ),
                "samples": counterfactual_count,
            },
            "rollouts": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("RESULT_JSON=" + json.dumps({"summary": summary, "counterfactual": result["counterfactual_same_state"]}))
        print(f"Saved {args.output.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print("[eval] Evaluation aborted before writing results:", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
