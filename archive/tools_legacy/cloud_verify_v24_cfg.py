#!/usr/bin/env python3
"""**启动前核验（V24）**：只查本轮唯一变量（`dof_acc_l2` 全关节）+ 确认 V23 的三项仍在。

A. `dof_acc_l2`：权重 = −1.0e-6（不动）、`func` 是 isaaclab 内建 `joint_acc_l2`、
   `asset_cfg.joint_names` = `.*`（**不再是** `[".*_hip_.*", ".*_knee_joint"]`）；
   并实测该正则在本体上**能匹配到全部 29 个关节**（用机器人资产 spec 直接解析，不建场景）。
B. V23 三项仍在：`feet_air_time`（progress_kappa 0.35 / motion_threshold 0.0）、
   `track_default_pose_on_stand`（weight −0.8 / pos_tol_default 0.08）、
   判别器白化（normalizer + GP normalized）。
C. 高频探针已进代码：`rewards_amp.py` 里含 `hf_acc_all` / `hf_acc_legs` 等字样。
D. 观测/判别器输入仍是 195 × 8 = 1560。

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
    source /root/.llp_env.sh && unset PYTHONPATH && \
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v24_cfg.py
结果落在 `/root/v24_verify_result.txt`（Isaac 会吞 stdout ⇒ 以文件为准）。
"""

from __future__ import annotations

import os
import sys
import traceback

OUT = "/root/v24_verify_result.txt"
EXPECT_ACC_WEIGHT = -1.0e-6
EXPECT_N_JOINTS = 29
EXPECT_TOTAL_OBS = 1560

bad: list[str] = []
log: list[str] = []


def ok(msg: str) -> None:
    log.append(f"  ✅ {msg}")
    print(f"  ✅ {msg}", flush=True)


def fail(msg: str) -> None:
    bad.append(msg)
    log.append(f"  ❌ {msg}")
    print(f"  ❌ {msg}", flush=True)


def main() -> int:
    print("=" * 88, flush=True)
    print("V24 启动前核验", flush=True)
    print("=" * 88, flush=True)

    from instinctlab.tasks.locomotion.config.g1.amp_walk_v24_env_cfg import G1AmpWalkV24EnvCfg
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v24_ppo_cfg import (
        G1AmpWalkV24PPORunnerCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.amp_walk_v20_env_cfg import AmpV20PolicyStateObsCfg

    cfg = G1AmpWalkV24EnvCfg()

    print("\n[A] 本轮唯一变量：dof_acc_l2 全关节", flush=True)
    acc = cfg.rewards.dof_acc_l2
    (ok if abs(float(acc.weight) - EXPECT_ACC_WEIGHT) < 1e-12 else fail)(
        f"dof_acc_l2.weight = {acc.weight!r}（期望 {EXPECT_ACC_WEIGHT}，不动）"
    )
    fn = getattr(acc.func, "__name__", str(acc.func))
    (ok if "joint_acc_l2" in fn else fail)(f"dof_acc_l2.func = {fn}（期望 isaaclab 内建 joint_acc_l2）")
    jn = acc.params.get("asset_cfg", None)
    jn = getattr(jn, "joint_names", None)
    if jn in (".*", ["*"], ".*$"):
        ok(f"dof_acc_l2 的 joint_names = {jn!r} ⇒ 全关节（不再是 hip/knee）")
    else:
        fail(f"dof_acc_l2 的 joint_names = {jn!r}（期望 '.*' 全关节）")

    # 用正则在本体关节名上实测匹配数（不建 Isaac 场景）
    try:
        import re

        # ⚠️ **不要**用 `G1_MINIMAL_CFG.actuators.keys()` —— 那是**执行器组名**
        #    （'legs'/'feet'/'arms'，只有 3 个）⇒ 上一版因此误报"只匹配 3 个关节"。
        #    直接用 G1 29DOF 的关节名表（与 TB 里 `joint_error_rmse/*` 完全同名）。
        names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
            "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
            "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]
        pat = str(jn)
        matched = [n for n in names if re.fullmatch(pat, n)]
        (ok if len(matched) == EXPECT_N_JOINTS else fail)(
            f"正则 {pat!r} 匹配到 {len(matched)}/{len(names)} 个关节（期望 {EXPECT_N_JOINTS}）"
        )
    except Exception as exc:  # noqa: BLE001
        fail(f"关节正则匹配测试失败：{exc!r}")

    print("\n[B] V23 的三项仍在（防回退）", flush=True)
    fa = cfg.rewards.feet_air_time
    for k, want in (("progress_kappa", 0.35), ("motion_threshold", 0.0), ("stand_penalty", 0.5)):
        got = fa.params.get(k)
        (ok if got is not None and abs(float(got) - want) < 1e-12 else fail)(
            f"feet_air_time.params[{k}] = {got!r}（期望 {want}）"
        )
    (ok if abs(float(fa.weight) - 4.0) < 1e-12 else fail)(f"feet_air_time.weight = {fa.weight!r}（期望 4.0）")
    dp = cfg.rewards.track_default_pose_on_stand
    (ok if abs(float(dp.weight) + 0.8) < 1e-12 else fail)(f"track_default_pose_on_stand.weight = {dp.weight!r}（期望 -0.8）")
    (ok if abs(float(dp.params.get("pos_tol_default", -1)) - 0.08) < 1e-9 else fail)(
        f"pos_tol_default = {dp.params.get('pos_tol_default')!r}（期望 0.08）"
    )
    alg = G1AmpWalkV24PPORunnerCfg().algorithm
    dk = dict(alg.discriminator_kwargs)
    (ok if dk.get("normalizer_class_name") == "EmpiricalNormalization" else fail)(
        f"白化仍在：normalizer_class_name = {dk.get('normalizer_class_name')!r}"
    )
    (ok if getattr(alg, "amp_grad_pen_space", None) == "normalized" else fail)(
        f"amp_grad_pen_space = {getattr(alg, 'amp_grad_pen_space', None)!r}（期望 'normalized'）"
    )
    ck = getattr(G1AmpWalkV24PPORunnerCfg(), "ckpt_manipulator", "MISSING")
    (ok if ck is None else fail)(f"runner.ckpt_manipulator = {ck!r}（期望 None=从零）")

    print("\n[C] 高频探针已进代码", flush=True)
    import instinctlab.tasks.locomotion.mdp.rewards_amp as ra

    src = open(ra.__file__, encoding="utf-8").read()
    for tag in ("hf_acc_all", "hf_acc_{grp}", "stand_pose_vel_cost"):
        (ok if tag in src else fail)(f"rewards_amp.py 含 {tag!r}")

    print("\n[D] 观测 / 判别器输入", flush=True)
    from instinctlab.tasks.locomotion.config.g1.amp_walk_v20_env_cfg import (
        AmpV20ReferenceStateObsCfg,
    )

    want_cls = {"amp_policy": AmpV20PolicyStateObsCfg, "amp_reference": AmpV20ReferenceStateObsCfg}
    for side, cls in want_cls.items():
        grp = getattr(cfg.observations, side)
        (ok if isinstance(grp, cls) else fail)(f"{side} 类型 = {type(grp).__name__}（期望 {cls.__name__}）")
    # ⚠️ 求和必须两支都算（上一版每轮覆盖 `total` + 只算 policy ⇒ 误报 312）
    totals = {}
    for side in ("amp_policy", "amp_reference"):
        grp = getattr(cfg.observations, side)
        s = 0
        for name, per in (("body_pos_b", 3), ("body_ori_b", 6), ("body_lin_vel_b", 3), ("body_ang_vel_b", 3)):
            term = getattr(grp, name)
            nb = len(term.params["asset_cfg"].body_names)
            s += term.history_length * nb * per
        totals[side] = s
        (ok if s == EXPECT_TOTAL_OBS else fail)(f"{side} 判别器输入 = {s}（期望 {EXPECT_TOTAL_OBS}）")

    print("\n" + "=" * 88, flush=True)
    print(f"[V24 {'PASS' if not bad else 'FAIL'}] 失败 {len(bad)} 项", flush=True)
    for b in bad:
        print(f"   - {b}", flush=True)
    print("=" * 88, flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    import traceback as _tb

    _fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(_fd, 1)
    os.dup2(_fd, 2)
    print("===== V24 启动前核验 =====", flush=True)

    from isaaclab.app import AppLauncher

    _app = None
    rc = 1
    _err: str | None = None
    try:
        _launcher = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False})
        _app = _launcher.app
        rc = main()
    except BaseException:  # noqa: BLE001
        _err = _tb.format_exc()
        print("核验脚本抛异常：", flush=True)
        print(_err, flush=True)
    if _err is not None:
        rc = 1
    # ⚠️ 结论必须在 `_app.close()` 之前打印（Isaac 的 close 会直接终止进程）
    print(f"RC={rc}", flush=True)
    print(f"[V24 {'PASS' if rc == 0 else 'FAIL'}]", flush=True)
    sys.stdout.flush()
    if _app is not None:
        try:
            _app.close()
        except BaseException:  # noqa: BLE001
            pass
    os._exit(rc)
