#!/usr/bin/env python3
"""**启动前核验**：把 V20 的最终生效配置打印出来，并**独立复算判别器输入维度**。

为什么需要：`@configclass` 的继承链很长（V20 ← V19d ← V19c ← V19b ← V18 ← … ← V8），
像 `amp_reward_form` / `discriminator_reward_type` / `amp_quad_k` 这类键是**很多代以前设的**，
看文件极易误判。本轮真正的"开关"是 **env 侧的 amp_policy/amp_reference 两组观测**
（表征），所以这里除了打印算法/奖励，还会：

1. 逐项独立复算每帧维度（`len(asset_cfg.body_names) × 每体维度`）并断言
   **195/帧 × 2 帧 = 390**，且两侧（policy/reference）逐项同构；
2. 断言那 13 个刚体在**参考侧**（`motion_reference.link_of_interests`，其 `body_ids`
   的索引空间）与**机器人侧**（`G1_29DOF_LINKS`）都能解析 —— 这两条一旦不满足，
   建场景时会直接抛 `resolve_matching_names` 的错。

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
    source /root/.llp_env.sh && unset PYTHONPATH && \
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v20_cfg.py

⚠️ 必须先建 `SimulationApp`（`amp_walk_env_cfg.py` 里有 `from isaaclab.envs import mdp`），
   否则 `ModuleNotFoundError: No module named 'isaaclab.envs'`。启动约 1~2 分钟。
"""

from __future__ import annotations

#: 每体维度（与 `amp_body_state.py` 的实现一一对应）
PER_BODY_DIM = {"body_pos_b": 3, "body_ori_b": 6, "body_lin_vel_b": 3, "body_ang_vel_b": 3}
EXPECT_N_BODIES = 13
EXPECT_FRAME_SIZE = 195
EXPECT_TOTAL = 390
EXPECT_HISTORY = 2
EXPECT_ANCHOR = ("torso_link",)

EXPECT_ALGO = {
    # AMP 奖励口径（V19b 起未变）
    "amp_reward_form": "log_quad",
    "discriminator_reward_type": "wasserstein",
    "discriminator_loss_func": "MSELoss",
    "discriminator_reward_coef": 1.0,
    "style_lerp": 0.75,
    "amp_quad_k": 0.125,
    # 判别器正则（V19a 起未变）
    "discriminator_gradient_penalty_coef": 1.0,
    "discriminator_weight_decay_coef": 1.0e-5,
    "discriminator_logit_weight_decay_coef": 0.0,
    "amp_reward_eps": 0.05,
}
EXPECT_ENV = {
    # V19d 的奖励基线（V20 不改任何一项）
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
    "track_ang_vel_z_exp.weight": 0.8,
    "track_base_height.weight": 1.0,
}


def _term_dims(group) -> dict[str, tuple[int, int, int]]:
    """{term: (帧数, 每帧维度, 该 term 总维度)}——**从 cfg 现场复算**，不看声明常量。"""
    out: dict[str, tuple[int, int, int]] = {}
    for name, per_body in PER_BODY_DIM.items():
        term = getattr(group, name)
        n_bodies = len(term.params["asset_cfg"].body_names)
        hist = term.history_length
        out[name] = (hist, n_bodies * per_body, hist * n_bodies * per_body)
    return out


def main() -> int:
    bad: list[str] = []

    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v20_ppo_cfg import (
        G1AmpWalkV20PPORunnerCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.amp_walk_v20_env_cfg import (
        AMP_V20_BODY_NAMES,
        AMP_V20_STATE_SIZE,
        AmpV20PolicyStateObsCfg,
        AmpV20ReferenceStateObsCfg,
        G1AmpWalkV20EnvCfg,
        G1AmpWalkV20TeacherEnv,
    )

    env = G1AmpWalkV20EnvCfg()
    runner = G1AmpWalkV20PPORunnerCfg()

    # ------------------------------------------------------------------ ① 表征
    print("========== ① 判别器输入表征（本轮唯一变量）==========")
    pol = env.observations.amp_policy
    ref = env.observations.amp_reference
    # ⚠️ 最致命的一类错：祖先链的 `_patch_amp_history` 会用**基类 5 项模板**覆盖这两组
    #    （见 `amp_walk_v20_env_cfg.py` 的踩坑记录）⇒ 判别器输入会悄悄退回 670 维。
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
        bad.append("total != 390")
    if AMP_V20_STATE_SIZE != EXPECT_TOTAL:
        bad.append(f"AMP_V20_STATE_SIZE={AMP_V20_STATE_SIZE} != {EXPECT_TOTAL}")

    n_bodies = len(AMP_V20_BODY_NAMES)
    print(f"  -- 被观测刚体数 = {n_bodies}（期望 {EXPECT_N_BODIES}）")
    if n_bodies != EXPECT_N_BODIES:
        bad.append(f"n_bodies {n_bodies} != {EXPECT_N_BODIES}")
    print("  --  13 体 =", ", ".join(AMP_V20_BODY_NAMES))

    # 两组项名必须完全一致（WASABI `init_storage` 会断言两侧等长）
    names_pol = tuple(n for n in PER_BODY_DIM if hasattr(pol, n))
    names_ref = tuple(n for n in PER_BODY_DIM if hasattr(ref, n))
    ok = names_pol == names_ref == tuple(PER_BODY_DIM)
    print(f"  {'OK ' if ok else 'XX '} 两组项名一致 = {names_pol}")
    if not ok:
        bad.append(f"term names policy={names_pol} reference={names_ref}")

    # ------------------------------------------------- ② 名字可解析性（两侧）
    print("\n========== ② 刚体名字在两侧都能解析 ==========")
    interests = tuple(env.scene.motion_reference.link_of_interests)
    miss_ref = [b for b in AMP_V20_BODY_NAMES if b not in interests]
    print(f"  {'OK ' if not miss_ref else 'XX '} 参考侧 link_of_interests({len(interests)}) "
          f"覆盖 13 体；缺失 = {miss_ref or '无'}")
    if miss_ref:
        bad.append(f"reference link_of_interests 缺少 {miss_ref}")
    print(f"  {'OK ' if EXPECT_ANCHOR[0] in interests else 'XX '} 参考侧含 anchor"
          f" {EXPECT_ANCHOR[0]}")
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

    # ------------------------------------------------------------------ ③ 算法
    print("\n========== ③ 算法侧（应与 V19d 逐字相同）==========")
    algo = runner.algorithm
    for k, want in EXPECT_ALGO.items():
        got = getattr(algo, k, "<missing>")
        ok = (got == want)
        print(f"  {'OK ' if ok else 'XX '} {k:<44} = {got!r:<16} (期望 {want!r})")
        if not ok:
            bad.append(f"algo.{k}: {got!r} != {want!r}")
    ok = runner.ckpt_manipulator is None
    print(f"  {'OK ' if ok else 'XX '} ckpt_manipulator = {runner.ckpt_manipulator!r}"
          f"（期望 None = **从零**）")
    if not ok:
        bad.append(f"ckpt_manipulator={runner.ckpt_manipulator!r} != None")
    print(f"  --  run_name = {runner.run_name}")
    print(f"  --  env.run_name = {env.run_name}")

    # ------------------------------------------------------------------ ④ 奖励
    print("\n========== ④ 环境侧奖励（V19d 基线，V20 不改）==========")
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

    # ------------------------------------------------------- ⑤ 指令契约（继承）
    cmd = env.commands.base_velocity
    exp = G1AmpWalkV20TeacherEnv.expected_command_ranges
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
    print("  ✅ 全部符合预期：判别器输入 = 13 刚体 × 15 × 2 帧 = 390；"
          "奖励/算法与 V19d 逐字相同；起点从零")
    return 0


if __name__ == "__main__":
    import os
    import sys
    import traceback

    # ⚠️ Isaac Sim 起来后会接管 stdout/stderr：实测 `print()`/traceback 会**丢失**
    #    （脚本抛异常时日志只剩最后一行 print、退出码还可能是 0）⇒ 本脚本以**文件**为
    #    唯一事实来源。这里直接把**文件描述符 1/2** 重定向到文件（比替换 `sys.stdout`
    #    更彻底；替换成自定义对象会破坏 `faulthandler.enable()` 需要的 `fileno`）。
    _fd = os.open("/root/v20_verify_result.txt", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(_fd, 1)
    os.dup2(_fd, 2)

    print("===== V20 启动前核验 =====", flush=True)

    from isaaclab.app import AppLauncher

    _launcher = None
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
