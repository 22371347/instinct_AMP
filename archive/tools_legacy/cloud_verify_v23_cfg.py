#!/usr/bin/env python3
"""**启动前核验（V23）**：不建 Isaac 场景（秒级），断言三块改动的最终生效值。

核验内容
--------
A. 观测/判别器输入：13 体 × 195 维/帧 × **8 帧** = **1560**；两组类型正确（防祖先
   `_patch_amp_history` 覆盖，V20 踩过的坑）；anchor = torso_link。
B. 奖励结构（V23 两项）：
   * `feet_air_time`：`progress_kappa=0.35` / `progress_tau=0.3` / `motion_threshold=0.0`（废门已删）
     / `stand_penalty=0.5` / weight 4.0；且**函数签名**已含新 kwarg；
   * `track_default_pose_on_stand`：weight **−0.8**、`pos_tol_default=0.08`、`vel_scale=0.8`、
     逐关节 `pos_joint_weights`（膝/踝 2.0、髋 1.5、腰偏航 3.0）与 `pos_joint_tol`（肩/腰 0.05）；
   * 其余项抽查：`track_ang_vel_z_exp=1.0`（V21 对齐）、`action_rate_l2=−0.06`、`flat_orientation_l2=−2.0`。
C. 算法（白化）：`discriminator_kwargs` 含 `EmpiricalNormalization` + `eps=0.01`、
   `amp_normalizer_expert_only=True`、`amp_grad_pen_space="normalized"`、GP coef 1.0、`ckpt_manipulator is None`。
D. 数学自检：直接跑 `mdp/amp_reward_utils.py` 的进度门控 —— 踏步（±0.5 交替）不放行、真走放行、
   站立档噪声不放行（纯 torch，无 Isaac）。

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
    source /root/.llp_env.sh && unset PYTHONPATH && \
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v23_cfg.py
结果（含异常）落在 `/root/v23_verify_result.txt`（Isaac 会吞 stdout/stderr ⇒ 以文件为准）。
"""

from __future__ import annotations

import os
import sys
import traceback

OUT = "/root/v23_verify_result.txt"
PER_BODY_DIM = {"body_pos_b": 3, "body_ori_b": 6, "body_lin_vel_b": 3, "body_ang_vel_b": 3}
EXPECT_N_BODIES = 13
EXPECT_HISTORY = 8
EXPECT_TOTAL = 1560
EXPECT_ENV = {
    "feet_air_time.weight": 4.0,
    "feet_air_time.params.good_lo": 0.4,
    "feet_air_time.params.good_hi": 1.0,
    "feet_air_time.params.tau": 0.08,
    "feet_air_time.params.motion_threshold": 0.0,   # ⭐ V23：废门已删
    "feet_air_time.params.progress_kappa": 0.35,    # ⭐ V23：进度判据
    "feet_air_time.params.progress_tau": 0.3,
    "feet_air_time.params.progress_omega_kappa": 0.3,
    "feet_air_time.params.stand_penalty": 0.5,
    "track_default_pose_on_stand.weight": -0.8,     # ⭐ V23：0.5 → 0.8
    "track_default_pose_on_stand.params.pos_tol_default": 0.08,
    "track_default_pose_on_stand.params.vel_scale": 0.8,
    "action_rate_l2.weight": -0.06,
    "flat_orientation_l2.weight": -2.0,
    "track_lin_vel_xy_exp.weight": 3.0,
    "track_ang_vel_z_exp.weight": 1.0,
    "track_base_height.weight": 1.0,
}
EXPECT_ALGO = {
    "amp_normalizer_expert_only": True,
    "amp_grad_pen_space": "normalized",
    "discriminator_gradient_penalty_coef": 1.0,
    "amp_quad_k": 0.125,
    "amp_reward_form": "log_quad",
    "discriminator_reward_type": "wasserstein",
    "discriminator_loss_func": "MSELoss",
}

bad: list[str] = []
log: list[str] = []


def ok(msg: str) -> None:
    log.append(f"  ✅ {msg}")
    print(f"  ✅ {msg}", flush=True)


def fail(msg: str) -> None:
    bad.append(msg)
    log.append(f"  ❌ {msg}")
    print(f"  ❌ {msg}", flush=True)


def dig(obj, path: str):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):  # ⚠️ `RewTerm.params` 是 **dict** ⇒ 必须用下标取值
            cur = cur[part]
        else:
            cur = getattr(cur, part)
    return cur


def main() -> int:
    log.append("=" * 88)
    log.append("V23 启动前核验")
    log.append("=" * 88)

    # ---------------- A. 观测 ----------------
    from instinctlab.tasks.locomotion.config.g1.amp_walk_v23_env_cfg import (
        G1AmpWalkV23EnvCfg,
        AMP_V23_PROGRESS_KAPPA,
    )
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v23_ppo_cfg import (
        G1AmpWalkV23PPORunnerCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.amp_walk_v20_env_cfg import (
        AmpV20PolicyStateObsCfg,
        AmpV20ReferenceStateObsCfg,
    )

    cfg = G1AmpWalkV23EnvCfg()
    log.append("\n[A] 观测 / 判别器输入")
    for side in ("amp_policy", "amp_reference"):
        grp = getattr(cfg.observations, side)
        want_cls = AmpV20PolicyStateObsCfg if side == "amp_policy" else AmpV20ReferenceStateObsCfg
        if isinstance(grp, want_cls):
            ok(f"{side} 类型正确 = {type(grp).__name__}")
        else:
            fail(f"{side} 类型被覆盖：{type(grp).__name__}（期望 {want_cls.__name__}）")
        total = 0
        for name, per_body in PER_BODY_DIM.items():
            term = getattr(grp, name)
            nb = len(term.params["asset_cfg"].body_names)
            hist = term.history_length
            total += hist * nb * per_body
            if hist != EXPECT_HISTORY:
                fail(f"{side}.{name}.history_length = {hist}（期望 {EXPECT_HISTORY}）")
            if nb != EXPECT_N_BODIES:
                fail(f"{side}.{name} 刚体数 = {nb}（期望 {EXPECT_N_BODIES}）")
            # ⚠️ 只有 **pos/ori** 两项有 anchor（相对 torso_link）；lv/av 是刚体自身
            #    坐标系下的速度，**按设计没有 anchor**（见 `amp_body_state.py`）。
            if name in ("body_pos_b", "body_ori_b"):
                anchor = tuple(getattr(term.params.get("anchor_cfg", None), "body_names", ()))
                if anchor != ("torso_link",):
                    fail(f"{side}.{name}.anchor_cfg = {anchor}（期望 ('torso_link',)）")
        if total == EXPECT_TOTAL:
            ok(f"{side} 判别器输入 = {total}（13 体 × 195 × 8）")
        else:
            fail(f"{side} 判别器输入 = {total}（期望 {EXPECT_TOTAL}）")

    # ---------------- B. 奖励 ----------------
    log.append("\n[B] 奖励结构（V23 两项改动）")
    for path, want in EXPECT_ENV.items():
        try:
            got = dig(cfg.rewards, path)
        except Exception as exc:  # noqa: BLE001
            fail(f"取不到 rewards.{path}：{exc!r}")
            continue
        if isinstance(want, float) and isinstance(got, (int, float)):
            same = abs(float(got) - want) < 1e-9
        else:
            same = got == want
        (ok if same else fail)(f"rewards.{path} = {got!r}（期望 {want!r}）")

    # 逐关节表抽查
    pw = cfg.rewards.track_default_pose_on_stand.params["pos_joint_weights"]
    pt = cfg.rewards.track_default_pose_on_stand.params["pos_joint_tol"]
    for j, want in (("left_knee_joint", 2.0), ("right_ankle_pitch_joint", 2.0), ("left_hip_pitch_joint", 1.5), ("waist_yaw_joint", 3.0)):
        (ok if abs(float(pw.get(j, -1)) - want) < 1e-9 else fail)(f"pos_joint_weights[{j}] = {pw.get(j)!r}（期望 {want}）")
    for j, want in (("left_shoulder_roll_joint", 0.05), ("waist_yaw_joint", 0.05), ("left_ankle_roll_joint", 0.12), ("left_knee_joint", 0.08)):
        (ok if abs(float(pt.get(j, -1)) - want) < 1e-9 else fail)(f"pos_joint_tol[{j}] = {pt.get(j)!r}（期望 {want}）")

    # 函数签名含新 kwarg（防止只改了 cfg、函数没接上）
    import inspect
    import instinctlab.tasks.locomotion.mdp as mdp

    sig_fa = inspect.signature(mdp.feet_air_time_target_biped).parameters
    for k in ("progress_kappa", "progress_tau", "progress_omega_kappa", "motion_threshold", "stand_penalty"):
        (ok if k in sig_fa else fail)(f"feet_air_time_target_biped 签名含 {k}")
    sig_dp = inspect.signature(mdp.track_default_pose_on_stand).parameters
    for k in ("pos_joint_tol", "pos_tol_default", "vel_joint_tol", "vel_tol_default", "pos_joint_weights"):
        (ok if k in sig_dp else fail)(f"track_default_pose_on_stand 签名含 {k}")

    # ---------------- C. 算法（白化） ----------------
    log.append("\n[C] 算法 / 白化")
    alg = G1AmpWalkV23PPORunnerCfg().algorithm
    for path, want in EXPECT_ALGO.items():
        got = dig(alg, path)
        (ok if got == want else fail)(f"algorithm.{path} = {got!r}（期望 {want!r}）")
    dk = dict(alg.discriminator_kwargs)
    (ok if dk.get("normalizer_class_name") == "EmpiricalNormalization" else fail)(
        f"discriminator_kwargs.normalizer_class_name = {dk.get('normalizer_class_name')!r}"
    )
    nk = dict(dk.get("normalizer_kwargs") or {})
    (ok if abs(float(nk.get("eps", -1)) - 0.01) < 1e-9 else fail)(f"normalizer_kwargs.eps = {nk.get('eps')!r}（期望 0.01）")
    # ⚠️ `ckpt_manipulator` 是 **runner cfg** 的字段（不是 algorithm 的）
    ck = getattr(G1AmpWalkV23PPORunnerCfg(), "ckpt_manipulator", "MISSING")
    (ok if ck is None else fail)(f"runner.ckpt_manipulator = {ck!r}（期望 None=从零）")

    # ---------------- D. 数学自检（纯 torch） ----------------
    log.append("\n[D] amp_reward_utils 数学自检")
    import torch
    from instinctlab.tasks.locomotion.mdp.amp_reward_utils import ProgressGate

    g = ProgressGate(4, dt=0.02, tau=0.3)
    # 指令：env2 也是**前进 0.6**（它走的是反方向）—— 上一版把 env2 的指令写成 −0.6，
    # 于是"沿指令方向走 −0.6"反而是达标的 ⇒ 导致误报。
    cmd = torch.tensor([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for i in range(150):
        v = torch.tensor(
            [
                [0.5, 0.0] if i % 2 == 0 else [-0.5, 0.0],  # 踏步：±0.5 交替
                [0.6, 0.0],                                  # 真走
                [-0.6, 0.0],                                 # 指令前进却后退
                [0.2 if i % 2 == 0 else -0.2, 0.0],          # 站立档噪声
            ]
        )
        w_act = torch.tensor([0.0, 0.0, 0.0, 0.3])
        ema, prog, _ = g(v, cmd, w_act, torch.zeros(4), kappa=0.35, omega_kappa=0.3)
    log.append(f"     EMA={[round(float(x), 3) for x in ema]}  progress={[int(float(x)) for x in prog]}")
    (ok if float(prog[0]) == 0.0 else fail)("踏步（±0.5 交替）不放行")
    (ok if float(prog[1]) == 1.0 else fail)("真走（0.6）放行")
    (ok if float(prog[2]) == 0.0 else fail)("反向走不放行")
    (ok if float(prog[3]) == 0.0 else fail)("站立档噪声不放行（turning 未误放行）")

    log.append("\n" + "=" * 88)
    log.append(f"[V23 {'PASS' if not bad else 'FAIL'}] 失败 {len(bad)} 项")
    if bad:
        for b in bad:
            log.append(f"   - {b}")
    log.append("=" * 88)
    return 0 if not bad else 1


if __name__ == "__main__":
    # ⚠️ 与 cloud_verify_v21_cfg.py 同款引导：
    #   1) 先把 fd 1/2 重定向到结果文件 —— Isaac Sim 会吞 stdout/stderr，异常时
    #      退出码仍为 0 且没有 traceback（踩过）；
    #   2) `import isaaclab.envs` **必须**在 `AppLauncher` 之后（否则
    #      ModuleNotFoundError: No module named 'isaaclab.envs'）。
    import traceback

    _fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(_fd, 1)
    os.dup2(_fd, 2)
    print("===== V23 启动前核验 =====", flush=True)

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
    if _err is not None:
        rc = 1
    # ⚠️ **必须在 `_app.close()` 之前**打印结论：Isaac 的 close 会直接终止进程，
    #    写在后面的话根本到不了结果文件（踩过：结果文件末尾停在最后一个 ✅，
    #    导致启动脚本的 PASS 闸门永远失败）。
    print(f"RC={rc}", flush=True)
    print(f"[V23 {'PASS' if rc == 0 else 'FAIL'}]", flush=True)
    sys.stdout.flush()
    if _app is not None:
        try:
            _app.close()
        except BaseException:  # noqa: BLE001
            pass
    os._exit(rc)
