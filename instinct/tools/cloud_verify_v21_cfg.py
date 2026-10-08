#!/usr/bin/env python3
"""**启动前核验（V21）**：打印最终生效配置 + **独立复算判别器输入维度**（期望 195 × 8 = 1560）。

要点：
1. 逐项复算每帧维度（`len(asset_cfg.body_names) × 每体维度`）并断言 **195/帧 × 8 帧 = 1560**，
   且两侧（policy/reference）逐项同构；
2. ⚠️ 断言 `observations.amp_policy/amp_reference` 的**类型**是 V20 的刚体组 ——
   祖先链的 `_patch_amp_history()` 会用**基类 5 项模板**把它冲掉（V20 踩过这个坑），
   一旦被覆盖，判别器输入会悄悄退回 670 维；
3. 断言 13 体在**参考侧**（`motion_reference.link_of_interests`）与**机器人侧**（`G1_29DOF_LINKS`）都能解析；
4. 奖励/算法应与 V19d 逐字相同（本轮不改奖励），`ckpt_manipulator is None`（从零）。

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
    source /root/.llp_env.sh && unset PYTHONPATH && \
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v21_cfg.py
结果（含异常）落在 `/root/v21_verify_result.txt`（Isaac Sim 会吞 stdout/stderr，故以文件为准）。
"""

from __future__ import annotations

PER_BODY_DIM = {"body_pos_b": 3, "body_ori_b": 6, "body_lin_vel_b": 3, "body_ang_vel_b": 3}
EXPECT_N_BODIES = 13
EXPECT_FRAME_SIZE = 195
EXPECT_HISTORY = 8
EXPECT_TOTAL = 1560
EXPECT_ANCHOR = ("torso_link",)

EXPECT_ALGO = {
    "amp_reward_form": "log_quad",
    "discriminator_reward_type": "wasserstein",
    "discriminator_loss_func": "MSELoss",
    "discriminator_reward_coef": 1.0,
    "style_lerp": 0.75,
    "amp_quad_k": 0.125,
    "discriminator_gradient_penalty_coef": 1.0,
    "discriminator_weight_decay_coef": 1.0e-5,
    "discriminator_logit_weight_decay_coef": 0.0,
    "amp_reward_eps": 0.05,
}
EXPECT_ENV = {
    "feet_air_time.weight": 4.0,
    "feet_air_time.params.tau": 0.08,
    "feet_air_time.params.motion_threshold": 0.15,
    "feet_air_time.params.motion_ang_scale": 0.5,
    "feet_air_time.params.stand_penalty": 0.5,
    "feet_air_time.params.good_lo": 0.4,
    "feet_air_time.params.good_hi": 1.0,
    "action_rate_l2.weight": -0.06,
    "flat_orientation_l2.weight": -2.0,
    "track_default_pose_on_stand.weight": -0.5,
    "track_lin_vel_xy_exp.weight": 3.0,
    #: ⭐ V21 唯一一项奖励改动：对齐 AMP_mjlab（0.8 → 1.0；**std 仍 0.5**，见 env cfg 注释）
    "track_ang_vel_z_exp.weight": 1.0,
    "track_base_height.weight": 1.0,
}


def _term_dims(group) -> dict[str, tuple[int, int, int]]:
    out: dict[str, tuple[int, int, int]] = {}
    for name, per_body in PER_BODY_DIM.items():
        term = getattr(group, name)
        n_bodies = len(term.params["asset_cfg"].body_names)
        hist = term.history_length
        out[name] = (hist, n_bodies * per_body, hist * n_bodies * per_body)
    return out


def main() -> int:
    bad: list[str] = []

    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v21_ppo_cfg import (
        G1AmpWalkV21PPORunnerCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.amp_walk_v20_env_cfg import (
        AmpV20PolicyStateObsCfg,
        AmpV20ReferenceStateObsCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.amp_walk_v21_env_cfg import (
        AMP_V20_BODY_NAMES,
        AMP_V21_STATE_SIZE,
        G1AmpWalkV21EnvCfg,
        G1AmpWalkV21TeacherEnv,
    )

    env = G1AmpWalkV21EnvCfg()
    runner = G1AmpWalkV21PPORunnerCfg()

    print("========== ① 判别器输入表征（195 维/帧 × 8 帧）==========")
    pol = env.observations.amp_policy
    ref = env.observations.amp_reference
    for tag, obj, cls in (("amp_policy", pol, AmpV20PolicyStateObsCfg),
                          ("amp_reference", ref, AmpV20ReferenceStateObsCfg)):
        ok = isinstance(obj, cls)
        print(f"  {'OK ' if ok else 'XX '} {tag} 类型 = {type(obj).__name__}（期望 {cls.__name__}）")
        if not ok:
            bad.append(f"{tag} 类型 {type(obj).__name__} != {cls.__name__}（被父类覆盖了）")

    d_pol, d_ref = _term_dims(pol), _term_dims(ref)
    frame_total = 0
    for name in PER_BODY_DIM:
        hist, per_frame, total = d_pol[name]
        h_r, f_r, t_r = d_ref[name]
        ok = (hist == h_r == EXPECT_HISTORY) and (per_frame == f_r) and (total == t_r)
        if name in ("body_pos_b", "body_ori_b"):
            a_pol = getattr(pol, name).params["anchor_cfg"].body_names
            a_ref = getattr(ref, name).params["anchor_cfg"].body_names
            ok = ok and a_pol == a_ref == EXPECT_ANCHOR
        print(f"  {'OK ' if ok else 'XX '} {name:<16} 帧数={hist} 每帧={per_frame:>4} "
              f"合计={total:>4}   (reference: 帧数={h_r} 每帧={f_r} 合计={t_r})")
        if not ok:
            bad.append(f"term {name}: policy{hist, per_frame, total} != reference{h_r, f_r, t_r}")
        frame_total += per_frame
    print(f"  --  单帧合计 = {frame_total}（期望 {EXPECT_FRAME_SIZE}）；"
          f"判别器输入 = {frame_total * EXPECT_HISTORY}（期望 {EXPECT_TOTAL}）")
    if frame_total != EXPECT_FRAME_SIZE:
        bad.append(f"frame size {frame_total} != {EXPECT_FRAME_SIZE}")
    if frame_total * EXPECT_HISTORY != EXPECT_TOTAL:
        bad.append("total != 1560")
    if AMP_V21_STATE_SIZE != EXPECT_TOTAL:
        bad.append(f"AMP_V21_STATE_SIZE={AMP_V21_STATE_SIZE} != {EXPECT_TOTAL}")

    n_bodies = len(AMP_V20_BODY_NAMES)
    print(f"  -- 被观测刚体数 = {n_bodies}（期望 {EXPECT_N_BODIES}）")
    if n_bodies != EXPECT_N_BODIES:
        bad.append(f"n_bodies {n_bodies} != {EXPECT_N_BODIES}")
    print("  --  13 体 =", ", ".join(AMP_V20_BODY_NAMES))

    print("\n========== ② 刚体名字在两侧都能解析 ==========")
    interests = tuple(env.scene.motion_reference.link_of_interests)
    miss_ref = [b for b in AMP_V20_BODY_NAMES if b not in interests]
    print(f"  {'OK ' if not miss_ref else 'XX '} 参考侧 link_of_interests({len(interests)}) "
          f"覆盖 13 体；缺失 = {miss_ref or '无'}")
    if miss_ref:
        bad.append(f"reference link_of_interests 缺少 {miss_ref}")
    if EXPECT_ANCHOR[0] not in interests:
        bad.append(f"reference link_of_interests 缺少 anchor {EXPECT_ANCHOR[0]}")
    try:
        from instinctlab.assets.unitree_g1 import G1_29DOF_LINKS

        miss_rob = [b for b in (*AMP_V20_BODY_NAMES, *EXPECT_ANCHOR) if b not in G1_29DOF_LINKS]
        print(f"  {'OK ' if not miss_rob else 'XX '} 机器人侧 G1_29DOF_LINKS({len(G1_29DOF_LINKS)}) "
              f"覆盖 13 体 + anchor；缺失 = {miss_rob or '无'}")
        if miss_rob:
            bad.append(f"robot links 缺少 {miss_rob}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️  无法导入 G1_29DOF_LINKS（{exc}）⇒ 跳过该检查")

    print("\n========== ③ 算法侧（应与 V19d 逐字相同）==========")
    algo = runner.algorithm
    for k, want in EXPECT_ALGO.items():
        got = getattr(algo, k, "<missing>")
        ok = (got == want)
        print(f"  {'OK ' if ok else 'XX '} {k:<44} = {got!r:<16} (期望 {want!r})")
        if not ok:
            bad.append(f"algo.{k}: {got!r} != {want!r}")
    ok = runner.ckpt_manipulator is None
    print(f"  {'OK ' if ok else 'XX '} ckpt_manipulator = {runner.ckpt_manipulator!r}（期望 None = 从零）")
    if not ok:
        bad.append(f"ckpt_manipulator={runner.ckpt_manipulator!r} != None")
    print(f"  --  run_name = {runner.run_name}")
    print(f"  --  env.run_name = {env.run_name}")

    print("\n========== ④ 环境侧奖励（V19d 基线，V21 不改）==========")
    r = env.rewards
    got_env = {
        "feet_air_time.weight": r.feet_air_time.weight,
        "feet_air_time.params.tau": r.feet_air_time.params.get("tau"),
        "feet_air_time.params.motion_threshold": r.feet_air_time.params.get("motion_threshold"),
        "feet_air_time.params.motion_ang_scale": r.feet_air_time.params.get("motion_ang_scale"),
        "feet_air_time.params.stand_penalty": r.feet_air_time.params.get("stand_penalty"),
        "feet_air_time.params.good_lo": r.feet_air_time.params.get("good_lo"),
        "feet_air_time.params.good_hi": r.feet_air_time.params.get("good_hi"),
        "action_rate_l2.weight": r.action_rate_l2.weight,
        "flat_orientation_l2.weight": r.flat_orientation_l2.weight,
        "track_default_pose_on_stand.weight": r.track_default_pose_on_stand.weight,
        "track_lin_vel_xy_exp.weight": r.track_lin_vel_xy_exp.weight,
        "track_ang_vel_z_exp.weight": r.track_ang_vel_z_exp.weight,
        "track_base_height.weight": r.track_base_height.weight,
    }
    for k, want in EXPECT_ENV.items():
        got = got_env[k]
        ok = (got == want)
        print(f"  {'OK ' if ok else 'XX '} {k:<40} = {got!r:<10} (期望 {want!r})")
        if not ok:
            bad.append(f"env.{k}: {got!r} != {want!r}")
    print(f"  --  feet_air_time.func = {r.feet_air_time.func}")

    cmd = env.commands.base_velocity
    exp = G1AmpWalkV21TeacherEnv.expected_command_ranges
    got_rng = (tuple(cmd.ranges.lin_vel_x), tuple(cmd.ranges.lin_vel_y), tuple(cmd.ranges.ang_vel_z))
    ok = (tuple(tuple(x) for x in got_rng) == tuple(tuple(x) for x in exp))
    print(f"\n  {'OK ' if ok else 'XX '} 合约 expected_command_ranges = {exp}")
    if not ok:
        bad.append(f"contract: {got_rng} != {exp}")

    print("\n========== 结论 ==========")
    if bad:
        for b in bad:
            print(f"  ❌ {b}")
        return 1
    print("  ✅ 全部符合预期：判别器输入 = 13 刚体 × 15 × 8 帧 = 1560；"
          "奖励/算法与 V19d 逐字相同；起点从零")
    return 0


if __name__ == "__main__":
    import os
    import sys
    import traceback

    _fd = os.open("/root/v21_verify_result.txt", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(_fd, 1)
    os.dup2(_fd, 2)
    print("===== V21 启动前核验 =====", flush=True)

    from isaaclab.app import AppLauncher

    _app = None
    rc = 1
    _err: str | None = None
    try:
        _launcher = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False})
        _app = _launcher.app
        rc = main()
    except BaseException:  # noqa: BLE001
        _err = traceback.format_exc()
        print("核验脚本抛异常：", flush=True)
        print(_err, flush=True)
    finally:
        if _app is not None:
            try:
                _app.close()
            except BaseException:  # noqa: BLE001
                pass
    if _err is not None:
        rc = 1
    print(f"RC={rc}", flush=True)
    sys.exit(rc)
