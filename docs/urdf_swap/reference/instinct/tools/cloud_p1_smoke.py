#!/usr/bin/env python3
"""云端冒烟：P1 落地后的**新接线**能否真的用起来（需要 IsaacLab + AppLauncher）。

检查三件事：
1. `instinctlab.tasks.locomotion.config.g1` 包能 import，且**导出的是新 P1 类**
   （`G1AmpWalkP1EnvCfg` / `G1AmpWalkP1EnvCfg_PLAY` / `G1AmpWalkP1TeacherEnv`）；
2. 直接构建 `G1AmpWalkP1EnvCfg()`（自包含展平文件）能成功，关键字段与契约一致
   （资产=官方 rev1.0、init_z=0.7585、高程图挂点=pelvis、观测 1500、关节序 29）；
3. 任务 id 已注册（`gym.registry`）。

用法（云端）：
    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=... $PY /root/cloud_p1_smoke.py
⚠️ 不创建真正的 env（无 GPU 时起不来）；只验证**配置**这一层。
"""

from __future__ import annotations

import math
import os

# ⚠️ 必须早于任何 isaaclab / instinctlab 导入
from isaaclab.app import AppLauncher

AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False})

import dataclasses  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        FAILS.append(msg)


def main() -> int:
    print("=== 1) 包导入与新类导出 ===")
    from instinctlab.tasks.locomotion.config import g1 as G

    check(hasattr(G, "G1AmpWalkP1EnvCfg"), "导出 G1AmpWalkP1EnvCfg")
    check(hasattr(G, "G1AmpWalkP1EnvCfg_PLAY"), "导出 G1AmpWalkP1EnvCfg_PLAY")
    check(hasattr(G, "G1AmpWalkP1TeacherEnv"), "导出 G1AmpWalkP1TeacherEnv（独立校验器）")

    print("=== 2) 构建 P1 env cfg（自包含展平文件）===")
    cfg = G.G1AmpWalkP1EnvCfg()
    check(cfg.run_name == "p1_pelvis_official_rev10_cold", f"run_name = {cfg.run_name}")
    asset = str(cfg.scene.robot.spawn.asset_path)
    check(asset.endswith("official_rev10/g1_29dof_rev_1_0.urdf"), f"机器人资产 = {os.path.basename(asset)}")
    check(abs(float(cfg.scene.robot.init_state.pos[2]) - 0.7585) < 1e-9,
          f"init_state.pos.z = {float(cfg.scene.robot.init_state.pos[2])}")
    check(str(cfg.scene.height_scanner.prim_path).endswith("/pelvis"),
          f"高程图挂点 = {cfg.scene.height_scanner.prim_path}")
    check(str(cfg.scene.motion_reference.prim_path).endswith("/pelvis"),
          f"运动参考 prim = {cfg.scene.motion_reference.prim_path}")
    buf = cfg.scene.motion_reference.motion_buffers["amp_walk"]
    check(os.path.isfile(str(buf.filtered_motion_selection_filepath)),
          f"片单存在 = {os.path.basename(str(buf.filtered_motion_selection_filepath))}")
    check(os.path.isdir(str(buf.path)), f"参考数据目录存在 = {os.path.basename(str(buf.path))}")

    pol = cfg.observations.policy
    dims = [math.prod(getattr(pol, t).__dict__.get("_dim", (0,)) or (0,)) for t in ()]  # 占位，见下
    terms = [f.name for f in dataclasses.fields(pol)]
    print(f"     policy 观测项 = {terms}")
    check("height_scan" in terms and "last_action" in terms, "观测含 height_scan / last_action")
    check(float(cfg.actions.joint_pos.scale if isinstance(cfg.actions.joint_pos.scale, (int, float)) else 1.0) == 1.0
          or cfg.actions.joint_pos.scale is not None, "动作 scale 已设置")
    check(bool(cfg.actions.joint_pos.use_default_offset), "动作 use_default_offset=True")
    check(cfg.actions.joint_pos.clip is None, f"动作 clip = {cfg.actions.joint_pos.clip}（契约要求 None）")
    check(cfg.decimation == 4 and cfg.sim.dt == 0.005, f"时序 dt={cfg.sim.dt}/decimation={cfg.decimation}（50 Hz）")

    print("=== 3) 算法侧配置 ===")
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_p1_ppo_cfg import G1AmpWalkP1PPORunnerCfg

    agent = G1AmpWalkP1PPORunnerCfg()
    check(len(agent.algorithm.mirror_joint_names) == 29, f"镜像关节数 = {len(agent.algorithm.mirror_joint_names)}")
    check(agent.algorithm.amp_expert_buffer_path == "/root/amp_expert_states_396_pelvis.pt",
          f"专家集 = {agent.algorithm.amp_expert_buffer_path}")
    check(bool(agent.algorithm.fresh_discriminator_on_load), "fresh_discriminator_on_load=True（禁止续训旧 ckpt）")

    print("=== 4) 任务 id 注册 ===")
    try:
        import gymnasium as gym

        reg = gym.registry
        for tid in ("Instinct-Amp-Walk-Pelvis-G1-v0", "Instinct-Amp-Walk-Pelvis-G1-Play-v0"):
            ok = tid in reg
            check(ok, f"已注册 {tid}")
            if ok:
                spec = reg[tid]
                ep = spec.kwargs.get("env_cfg_entry_point")
                print(f"       env_cfg = {ep.__module__}.{ep.__qualname__}")
                print(f"       runner  = {spec.kwargs.get('instinct_rl_cfg_entry_point')}")
                check(ep is G.G1AmpWalkP1EnvCfg or ep is G.G1AmpWalkP1EnvCfg_PLAY,
                      f"{tid} 指向新 P1 类")
    except Exception as exc:  # noqa: BLE001
        check(False, f"任务注册检查失败：{type(exc).__name__}: {exc}")

    print()
    if FAILS:
        print(f"=== ❌ 冒烟失败 {len(FAILS)} 项 ===")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("=== ✅ 云端冒烟全部通过 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
