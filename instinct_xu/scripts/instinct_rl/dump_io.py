"""Dump cloud (IsaacLab) play-env IO so it can be compared against the local MuJoCo replay.

Purpose
-------
Feed-the-same-observation test in its strongest form: record, on the cloud, for every
control step, the **policy observation** and the **policy action** together with the raw
robot state that produced them.  Locally we can then

  (a) rebuild the 1500-d observation from the cloud's own qpos/qvel and compare it
      segment by segment with the cloud's observation  -> validates obs construction
      (layout / order / scale / history direction),
  (b) run the local exported ONNX on the cloud's own observation and compare with the
      cloud's action  -> validates the policy chain end-to-end on real states.

Usage (on the cloud, from /root/gpufree-data/instinct_xu)
--------------------------------------------------------
    python scripts/instinct_rl/dump_io.py \
        --task Instinct-Amp-Walk-Pelvis-G1-Play-v0 \
        --load_run <run_dir> --load_checkpoint model_2000.pt \
        --steps 300 --out /root/dump_io_p1_2000.npz --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Dump play-env IO for cross validation.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_instinct_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import collections
import numpy as np
import torch

import gymnasium as gym  # noqa: E402  (import order dictated by Isaac Sim)

from isaaclab.utils.io import load_yaml  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402

import instinctlab.tasks  # noqa: F401,E402
from instinct_rl.runners import OnPolicyRunner  # noqa: E402
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper  # noqa: E402


def log(*parts):
    """直接写 fd 1，避免任何缓冲/包装；顺便保证异常也不会静默。"""
    msg = " ".join(str(p) for p in parts) + "\n"
    os.write(1, msg.encode("utf-8", "replace"))


def describe_obs_manager(env):
    """只读 observation_manager 的**原始属性**，不碰任何 cfg 对象（避免 repr 卡死）。"""
    om = env.unwrapped.observation_manager
    fmt = env.get_obs_format()
    names = {g: list(v) for g, v in fmt.items()}
    dims = {g: [int(np.prod(d)) for d in v.values()] for g, v in fmt.items()}
    log("=" * 70)
    log("[fmt] policy observation layout (authoritative, in order):")
    for group, terms in names.items():
        log(f"  group '{group}'  total={sum(dims[group])}")
        for name, dim in zip(terms, dims[group]):
            log(f"    {name:24s} dim={dim}")
    log("[fmt] active_terms      =", om.active_terms)
    log("[fmt] group_obs_term_dim=", om.group_obs_term_dim)
    log("=" * 70)
    return names, dims


def describe_actions(env, robot):
    """⚠️ 实测：`action_manager.get_term()/get_term_cfg()` 在 IsaacLab 5.1 里
    会 100% CPU 卡住（未查清具体原因）⇒ 只读内部私有属性，不要碰它们。"""
    am = env.unwrapped.action_manager
    log("[action] active_terms =", am.active_terms)
    log("[action] _term_names(内部) =", getattr(am, "_term_names", None))
    names = None
    for attr in ("_joint_names", "joint_names"):
        val = getattr(am, attr, None)
        if val is not None:
            names = list(val)
            break
    log("[action] joint order(若可得) =", names)
    return names


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "instinct_rl", agent_cfg.experiment_name))
    if args_cli.load_run is not None:
        resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)
    else:
        raise RuntimeError("--load_run is required for dumping.")
    print(f"[INFO] checkpoint: {resume_path}", flush=True)
    log("[1/4] env cfg parsed", args_cli.task, "ckpt=", resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = InstinctRlVecEnvWrapper(env)

    names, dims = describe_obs_manager(env)
    robot = env.unwrapped.scene["robot"]
    log("[joint_names(articulation order)]", list(robot.data.joint_names))
    log("[body_names]", list(robot.data.body_names))
    log("[default_joint_pos]", list(robot.data.default_joint_pos[0].cpu().numpy().round(6)))
    log("[init_state.joint_pos]", list(robot.data.joint_pos[0].cpu().numpy().round(6)))
    describe_actions(env, robot)

    cmd_terms = env.unwrapped.command_manager.active_terms
    log("[command terms]", cmd_terms)
    log("[command] get_command(base_velocity) =",
        list(env.unwrapped.command_manager.get_command("base_velocity")[0].cpu().numpy().round(5))
        if "base_velocity" in cmd_terms else None)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    log("[2/4] ckpt loaded")
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    log("[3/4] inference policy ready; starting", args_cli.steps, "steps")

    rec = collections.defaultdict(list)
    obs, _ = env.get_observations()
    log("[3/4] obs shape =", tuple(obs.shape))

    def add(name, value):
        rec[name].append(np.asarray(value, dtype=np.float64).copy())

    for t in range(args_cli.steps):
        with torch.inference_mode():
            act = policy(obs)
            add("obs", obs[0].cpu().numpy())
            add("act", act[0].cpu().numpy())
            add("jpos", robot.data.joint_pos[0].cpu().numpy())
            add("jvel", robot.data.joint_vel[0].cpu().numpy())
            add("dpos", robot.data.default_joint_pos[0].cpu().numpy())
            add("rpos", robot.data.root_pos_w[0].cpu().numpy())
            add("rquat", robot.data.root_quat_w[0].cpu().numpy())
            add("rlin", robot.data.root_lin_vel_b[0].cpu().numpy())
            add("rang", robot.data.root_ang_vel_b[0].cpu().numpy())
            try:
                add("grav", robot.data.projected_gravity_b[0].cpu().numpy())
            except AttributeError:
                pass
            for cname in cmd_terms:
                try:
                    add(f"cmd_{cname}", env.unwrapped.command_manager.get_command(cname)[0].cpu().numpy())
                except Exception:  # noqa: BLE001
                    pass
            obs, _, _, _ = env.step(act)
        if t % 20 == 0:
            log(f"  step {t:4d} root_z={rec['rpos'][-1][2]:.4f} "
                f"act_absmax={np.abs(rec['act'][-1]).max():.3f}")

    out = {k: np.array(v) for k, v in rec.items()}
    out["joint_names"] = np.array(list(robot.data.joint_names))
    out["body_names"] = np.array(list(robot.data.body_names))
    for group in names:
        out[f"fmt_{group}_names"] = np.array(names[group])
        out[f"fmt_{group}_dims"] = np.array(dims[group])
    np.savez(args_cli.out, **out)
    log(f"[OK] wrote {args_cli.out} with keys: {sorted(out)}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
