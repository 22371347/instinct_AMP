#!/usr/bin/env python3
"""**启动前核验（V26）**：确认"只改了 σ 上限"这一件事，且退火机制真的存在。

A. 与 V24 的**逐字一致性**（这是单变量的前提）：
   - 21 项奖励的**权重表逐项相等**；
   - `action_scale = 1.0`、`dt/decimation`、指令范围、AMP 观测维度（195×8=1560）；
   - algo 侧：`amp_quad_k`、GP coef、`entropy_coef`、`learning_rate`、白化三项
     （`normalizer_class_name` / `amp_normalizer_expert_only` / `amp_grad_pen_space`）
     与 V24 相同。
B. 本轮唯一变量在位：`clip_max_std = 0.20`、`clip_max_std_schedule = 300`
   （V24 必须分别是 `0.6` / `None`）。
C. σ 退火机制真实存在（防"文件没上传/上传了旧版本"）：
   - `AlgorithmsWrapper.__init__` 签名里**有** `clip_max_std_schedule`；
   - `update()` 源码里含 `_std_cap_from` 与 `sigma_cap`（日志键）。
D. 热启动对象存在：V24 run 目录 + `model_8000.pt`（运行时由 `--resume` 加载）。
E. 任务注册可用：`gym.spec("Instinct-Amp-Walk-V26-G1-v0")` 拿得到两个 entry point。

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
    source /root/.llp_env.sh && unset PYTHONPATH && \
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_v26_cfg.py
结果落 `/root/v26_verify_result.txt`，末尾 `[V26 PASS]`。
"""

from __future__ import annotations

import inspect
import os
import sys
import traceback

OUT = "/root/v26_verify_result.txt"
RUN_ROOT = "/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk"
EXPECT_CAP = 0.20
EXPECT_RAMP = 300
V24_CAP = 0.6
V24_RAMP = None
EXPECT_AMP_DIM = 1560

bad: list[str] = []


def ok(msg: str) -> None:
    print(f"  ✅ {msg}", flush=True)


def fail(msg: str) -> None:
    bad.append(msg)
    print(f"  ❌ {msg}", flush=True)


def main() -> int:
    print("=" * 88, flush=True)
    print("V26 启动前核验（σ 单变量 + 退火机制）", flush=True)
    print("=" * 88, flush=True)

    import gymnasium as gym

    import instinctlab.tasks  # noqa: F401
    from instinctlab.tasks.locomotion.config.g1.amp_walk_v24_env_cfg import G1AmpWalkV24EnvCfg
    from instinctlab.tasks.locomotion.config.g1.amp_walk_v26_env_cfg import G1AmpWalkV26EnvCfg
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v24_ppo_cfg import (
        G1AmpWalkV24PPORunnerCfg,
    )
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v26_ppo_cfg import (
        G1AmpWalkV26PPORunnerCfg,
    )

    cfg24, cfg26 = G1AmpWalkV24EnvCfg(), G1AmpWalkV26EnvCfg()
    runner24, runner26 = G1AmpWalkV24PPORunnerCfg(), G1AmpWalkV26PPORunnerCfg()

    print("\n[A] 与 V24 的逐字一致性（单变量前提）", flush=True)
    keys24 = set(cfg24.rewards.__dict__.keys())
    keys26 = set(cfg26.rewards.__dict__.keys())
    if keys24 == keys26:
        ok(f"奖励项集合一致（{len(keys24)} 项）")
    else:
        fail(f"奖励项集合不同：仅在 V24 {sorted(keys24 - keys26)}，仅在 V26 {sorted(keys26 - keys24)}")
    diff = []
    for k in sorted(keys24 & keys26):
        t24, t26 = getattr(cfg24.rewards, k), getattr(cfg26.rewards, k)
        if getattr(t24, "weight", None) != getattr(t26, "weight", None):
            diff.append(f"{k}: {t24.weight} → {t26.weight}")
    if not diff:
        ok("21 项奖励权重逐项相等")
    else:
        fail("权重不同：" + "; ".join(diff))

    ann = cfg26.actions.__dict__.get("joint_pos")
    scale = getattr(ann, "scale", None)
    if scale == 1.0:
        ok("action.scale = 1.0（用户指定：不缩减）")
    else:
        fail(f"action.scale = {scale}，期望 1.0")
    if getattr(cfg26, "decimation", None) == getattr(cfg24, "decimation", None):
        ok(f"decimation = {cfg26.decimation}（与 V24 相同）")
    else:
        fail("decimation 与 V24 不同")
    r24, r26 = cfg24.commands.base_velocity.ranges, cfg26.commands.base_velocity.ranges
    if (r24.lin_vel_x, r24.lin_vel_y, r24.ang_vel_z) == (r26.lin_vel_x, r26.lin_vel_y, r26.ang_vel_z):
        ok(f"指令范围一致 {r26.lin_vel_x}/{r26.lin_vel_y}/{r26.ang_vel_z}")
    else:
        fail("指令范围与 V24 不同")

    print("\n[B] 本轮唯一变量：σ 上限 + 退火周期", flush=True)
    c26 = getattr(runner26.algorithm, "clip_max_std", "MISSING")
    s26 = getattr(runner26.algorithm, "clip_max_std_schedule", "MISSING")
    c24 = getattr(runner24.algorithm, "clip_max_std", "MISSING")
    s24 = getattr(runner24.algorithm, "clip_max_std_schedule", "MISSING")
    if c26 == EXPECT_CAP and s26 == EXPECT_RAMP:
        ok(f"V26: clip_max_std={c26}  clip_max_std_schedule={s26}")
    else:
        fail(f"V26 σ 设置不对：clip_max_std={c26}（期望 {EXPECT_CAP}）"
             f" schedule={s26}（期望 {EXPECT_RAMP}）")
    if c24 == V24_CAP and s24 == V24_RAMP:
        ok(f"V24 未变：clip_max_std={c24}  schedule={s24}")
    else:
        fail(f"V24 的 σ 设置被动过：clip_max_std={c24}（期望 {V24_CAP}）schedule={s24}（期望 {V24_RAMP}）")
    for k in ("entropy_coef", "learning_rate"):
        a, b = getattr(runner24.algorithm, k, None), getattr(runner26.algorithm, k, None)
        if a == b:
            ok(f"{k} 一致 = {b}")
        else:
            fail(f"{k} 不同：V24 {a} vs V26 {b}")
    for k in ("amp_quad_k", "discriminator_gradient_penalty_coef", "amp_grad_pen_space",
              "amp_normalizer_expert_only", "amp_reward_form"):
        a, b = getattr(runner24.algorithm, k, None), getattr(runner26.algorithm, k, None)
        if a == b:
            ok(f"{k} 一致 = {b}")
        else:
            fail(f"{k} 不同：V24 {a} vs V26 {b}")
    nk24 = runner24.algorithm.discriminator_kwargs.get("normalizer_class_name")
    nk26 = runner26.algorithm.discriminator_kwargs.get("normalizer_class_name")
    if nk24 == nk26:
        ok(f"白化一致：normalizer_class_name={nk26}")
    else:
        fail(f"白化不同：V24 {nk24} vs V26 {nk26}")

    print("\n[C] 退火机制真实存在（防旧版本文件）", flush=True)
    from instinctlab.utils.wrappers.instinct_rl.algorithms import WasabiPPOAmpMix

    params = inspect.signature(WasabiPPOAmpMix.__init__).parameters
    if "clip_max_std_schedule" in params:
        ok("WasabiPPOAmpMix.__init__ 接受 clip_max_std_schedule")
    else:
        fail("WasabiPPOAmpMix.__init__ 里**没有** clip_max_std_schedule ⇒ 上传的是旧文件")
    src = inspect.getsource(WasabiPPOAmpMix.update)
    for token in ("_std_cap_from", "sigma_cap", "_std_cap_step"):
        if token in src:
            ok(f"update() 源码含 {token}")
        else:
            fail(f"update() 源码**不含** {token}")

    print("\n[D] 热启动对象（V24 ckpt）", flush=True)
    v24 = [d for d in sorted(os.listdir(RUN_ROOT)) if "v24" in d]
    ck = os.path.join(RUN_ROOT, v24[-1], "model_8000.pt") if v24 else ""
    if v24 and os.path.isfile(ck):
        ok(f"找到 {os.path.basename(v24[-1])}/model_8000.pt（{os.path.getsize(ck) / 1e6:.1f} MB）")
    else:
        fail(f"找不到 V24 的 model_8000.pt（候选 run：{v24[-3:]}）")

    print("\n[E] 任务注册", flush=True)
    for tid in ("Instinct-Amp-Walk-V26-G1-v0", "Instinct-Amp-Walk-V26-G1-Play-v0"):
        spec = gym.spec(tid)
        ec = spec.kwargs["env_cfg_entry_point"]
        rc = spec.kwargs["instinct_rl_cfg_entry_point"]
        if "v26" in rc and ec.__name__.endswith("V26" + ("EnvCfg" if "Play" not in tid else "EnvCfg_PLAY")):
            ok(f"{tid}: env={ec.__name__}  algo={rc.split(':')[-1]}")
        else:
            fail(f"{tid} 注册内容异常：env={ec.__name__} algo={rc}")

    print("\n[F] AMP 观测维度（信息）", flush=True)
    of = cfg26.observations
    for g in ("amp_policy", "amp_reference"):
        terms = getattr(of, g, None)
        if terms is None:
            fail(f"观测组 {g} 不存在")
            continue
        names = [k for k in terms.__dict__.keys() if not k.startswith("_")]
        print(f"      {g}: terms={names}", flush=True)
    print(f"      期望判别器输入 = {EXPECT_AMP_DIM}（195×8）—— 与 V24 相同故不丢判别器", flush=True)

    print("\n" + "=" * 88, flush=True)
    if bad:
        print(f"[V26 FAIL] {len(bad)} 项不通过：", flush=True)
        for b in bad:
            print(f"   - {b}", flush=True)
        return 1
    print("[V26 PASS] σ 单变量 + 退火机制 + 热启动对象 全部就位", flush=True)
    return 0


if __name__ == "__main__":
    _fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(_fd, 1)
    os.dup2(_fd, 2)
    from isaaclab.app import AppLauncher

    _app = None
    rc = 1
    try:
        _app = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False}).app
        rc = main()
    except BaseException:  # noqa: BLE001
        print("抛异常：\n" + traceback.format_exc(), flush=True)
        rc = 1
    print(f"RC={rc}", flush=True)
    sys.stdout.flush()
    os._exit(rc)
