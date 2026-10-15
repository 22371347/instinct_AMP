#!/usr/bin/env python3
"""**启动前核验**：把 V19a 的最终生效配置打印出来（不建 Isaac Sim 场景，秒级返回）。

为什么需要：`@configclass` 的继承链很长（V19a ← V18 ← V17 ← V16 ← V15 ← V14 ← V12 ← V11
← V10b ← V10 ← V9j … ← V8），像 `amp_reward_form` / `discriminator_reward_type`
这类键是 **V8 设的、后面 10 代都没改**。光看文件很容易误判当前生效值
（本轮就踩过：以为基线是 `raw`+`quad`，实测是 **`log_quad`+`wasserstein`**）。

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
    source /root/.llp_env.sh && unset PYTHONPATH && \
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v19a_cfg.py

⚠️ 必须先建 `SimulationApp`：`amp_walk_env_cfg.py` 里有 `from isaaclab.envs import mdp`，
   而 Isaac Sim 的模块**必须在 `SimulationApp` 实例化之后**才能导入（否则
   `ModuleNotFoundError: No module named 'isaaclab.envs'`）。本脚本自带 bootstrap，
   启动约 1~2 分钟。
"""

from __future__ import annotations

EXPECT_ALGO = {
    # 与 V18 相同的部分
    "amp_reward_form": "log_quad",
    "discriminator_reward_type": "wasserstein",
    "discriminator_loss_func": "MSELoss",
    "discriminator_reward_coef": 1.0,
    "style_lerp": 0.75,
    # ⭐ V19a 的四项改动
    "discriminator_gradient_penalty_coef": 1.0,
    "discriminator_weight_decay_coef": 1.0e-5,
    "discriminator_logit_weight_decay_coef": 0.0,
    "amp_reward_eps": 0.05,
}
EXPECT_ENV = {
    "feet_air_time.weight": 4.0,
    "feet_air_time.params.soft": 0.4,
    "feet_air_time.params.good_lo": 0.4,
    "feet_air_time.params.good_hi": 1.0,
    "action_rate_l2.weight": -0.10,
    "flat_orientation_l2.weight": -2.0,
    "feet_swing_time_penalty.weight": 0.0,
    "track_lin_vel_xy_exp.weight": 3.0,
    "track_ang_vel_z_exp.weight": 0.8,
    "track_base_height.weight": 1.0,
}
EXPECT_CKPT = "instinctlab.utils.ckpt_manipulators:drop_amp_discriminator"


def main() -> int:
    bad: list[str] = []

    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v19a_ppo_cfg import (
        G1AmpWalkV19APPORunnerCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.amp_walk_v19a_env_cfg import (
        G1AmpWalkV19AEnvCfg,
        G1AmpWalkV19ATeacherEnv,
    )

    runner = G1AmpWalkV19APPORunnerCfg()
    algo = runner.algorithm
    print("========== 算法侧 ==========")
    for k, want in EXPECT_ALGO.items():
        got = getattr(algo, k, "<missing>")
        ok = (got == want)
        print(f"  {'OK ' if ok else 'XX '} {k:<44} = {got!r:<24} (期望 {want!r})")
        if not ok:
            bad.append(f"algo.{k}: {got!r} != {want!r}")
    print(f"  --  discriminator_optimizer_kwargs = {algo.discriminator_optimizer_kwargs}")
    print(f"  --  discriminator_gradient_torlerance = {algo.discriminator_gradient_torlerance}")
    print(f"  --  ckpt_manipulator = {runner.ckpt_manipulator!r}")
    if runner.ckpt_manipulator != EXPECT_CKPT:
        bad.append(f"ckpt_manipulator: {runner.ckpt_manipulator!r} != {EXPECT_CKPT!r}")
    print(f"  --  run_name = {runner.run_name}")

    env = G1AmpWalkV19AEnvCfg()
    r = env.rewards
    print("\n========== 环境侧（奖励）==========")
    got_env = {
        "feet_air_time.weight": r.feet_air_time.weight,
        "feet_air_time.params.soft": r.feet_air_time.params.get("soft"),
        "feet_air_time.params.good_lo": r.feet_air_time.params.get("good_lo"),
        "feet_air_time.params.good_hi": r.feet_air_time.params.get("good_hi"),
        "action_rate_l2.weight": r.action_rate_l2.weight,
        "flat_orientation_l2.weight": r.flat_orientation_l2.weight,
        "feet_swing_time_penalty.weight": r.feet_swing_time_penalty.weight,
        "track_lin_vel_xy_exp.weight": r.track_lin_vel_xy_exp.weight,
        "track_ang_vel_z_exp.weight": r.track_ang_vel_z_exp.weight,
        "track_base_height.weight": r.track_base_height.weight,
    }
    for k, want in EXPECT_ENV.items():
        got = got_env[k]
        ok = (got == want)
        print(f"  {'OK ' if ok else 'XX '} {k:<38} = {got!r:<12} (期望 {want!r})")
        if not ok:
            bad.append(f"env.{k}: {got!r} != {want!r}")
    print(f"  --  feet_air_time.func = {r.feet_air_time.func}")
    print(f"  --  track_base_height.params = {r.track_base_height.params}")

    print("\n========== 环境侧（数据集 / 指令 / 历史帧）==========")
    cmd = env.commands.base_velocity
    for lo, hi, tag in ((cmd.ranges.lin_vel_x, None, "lin_vel_x"),
                        (cmd.ranges.lin_vel_y, None, "lin_vel_y"),
                        (cmd.ranges.ang_vel_z, None, "ang_vel_z")):
        print(f"  --  ranges.{tag} = {lo}")
    print(f"  --  prob_low/side/lin/back = {cmd.prob_low}/{cmd.prob_side}/{cmd.prob_lin}/{cmd.prob_back}")
    print(f"  --  prob_lin_fwd = {cmd.prob_lin_fwd}; side_ang_vel_range = {cmd.side_ang_vel_range}")
    mb = env.scene.motion_reference.motion_buffers
    for k, buf in mb.items():
        print(f"  --  motion_buffers[{k!r}].filtered_motion_selection_filepath = "
              f"{getattr(buf, 'filtered_motion_selection_filepath', '<n/a>')}")
    print(f"  --  env.run_name = {env.run_name}")

    # 合约：指令范围必须与 TeacherEnv 声明的期望值一致（否则启动即报错）
    exp = G1AmpWalkV19ATeacherEnv.expected_command_ranges
    got_rng = (
        tuple(cmd.ranges.lin_vel_x), tuple(cmd.ranges.lin_vel_y), tuple(cmd.ranges.ang_vel_z)
    )
    ok = (tuple(tuple(x) for x in got_rng) == tuple(tuple(x) for x in exp))
    print(f"\n  {'OK ' if ok else 'XX '} 合约 expected_command_ranges = {exp}  "
          f"(cmd.ranges = {got_rng})")
    if not ok:
        bad.append(f"contract: {got_rng} != {exp}")
    print(f"  --  expected_command_hold = {G1AmpWalkV19ATeacherEnv.expected_command_hold}")

    print("\n========== 结论 ==========")
    if bad:
        for b in bad:
            print(f"  ❌ {b}")
        return 1
    print("  ✅ 全部符合预期（V19a = V18 + GP 1.0 / wd 1e-5 / logit_wd 0.0 / eps 0.05 / air-band soft 0.4）")
    return 0


if __name__ == "__main__":
    import sys

    # ⚠️ Isaac Sim 的模块（`isaaclab.envs` 等）必须先有 `SimulationApp` 才能导入。
    #    `check_only=True` 不建场景，但依然要实例化 app。
    from isaaclab.app import AppLauncher

    _launcher = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False})
    _app = _launcher.app
    try:
        rc = main()
    finally:
        _app.close()
    sys.exit(rc)
