#!/usr/bin/env python3
"""**D2 短测评：动作噪声对"高频抖动"的贡献**（有噪声 vs 只用均值动作）。

目的（验证一个假设）：V23/V24 的"高频小跳"里有多少来自 **σ 上的采样噪声**（σ 顶在
`clip_max_std=0.6`，`mean_noise_std≈0.59`），多少来自**策略本意**。

做法（不改任何训练代码）：把同一个 checkpoint 在同一环境里跑**两遍**——
  A) `noisy`：正常采样（`distribution.sample()`，= 训练时的行为）
  B) `mean` ：只用分布均值（`distribution.mean`，= 导出 ONNX / MuJoCo 回放时的行为）
对比 `env.extras["step"]` 里的探针（我们的 `_log_metric` 写的）与**动作空间**的二阶差分。

动作空间的解析对照：若 `a = a_mean + σ·ε`（ε 白噪声），则
    |Δ²a| = |a_t − 2a_{t−1} + a_{t−2}| 的噪声部分 = σ·|ε_t − 2ε_{t−1} + ε_{t−2}|
    ⇒ E|·| ≈ 0.7979·σ·√6 ≈ 1.954·σ      （σ=0.59 ⇒ ≈ **1.15 rad**）
把它与实测 `|Δ²a_sampled|`、`|Δ²a_mean|` 比较，就能定量拆出"噪声 vs 意图"。

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
    source /root/.llp_env.sh && unset PYTHONPATH && \
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_eval_noise_free.py \
        --task Instinct-Amp-Walk-V24-G1-v0 \
        --load_run /root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk/<run> \
        --checkpoint model_2000.pt --num_envs 128 --steps 600

结果落在 `/root/d2_eval_result.txt`（Isaac 会吞 stdout ⇒ 以文件为准），末尾有 `[D2 PASS]`。
"""

from __future__ import annotations

import argparse
import importlib
import os
import statistics
import sys
import traceback
from collections import defaultdict, deque

OUT = os.environ.get("D2_OUT", "/root/d2_eval_result.txt")  # 可用环境变量改，避免并发实例互相截断
WATCH = (
    "hf_acc_all", "hf_acc_legs", "hf_acc_arms", "hf_acc_waist",
    "air_time_hop_frac", "air_time_in_band_frac", "air_time_single_stance",
    "air_time_lift_frac", "air_time_progress_frac", "air_time_proj_ratio",
    "air_time_proj_ema", "air_time_stand_pen_mean",
    "stand_pose_pos_err", "stand_pose_vel_err", "stand_pose_active",
    "stand_pose_pos_cost", "stand_pose_vel_cost",
    "stand_pose_pos_err_legs", "stand_pose_pos_err_arms", "stand_pose_pos_err_waist",
    "stand_pose_vel_err_legs", "stand_pose_vel_err_arms", "stand_pose_vel_err_waist",
)


def _peak_to_mean_ratio(x):
    return 0.7978845608 * x  # E|N(0,1)|


def rollout(env, runner, alg, *, steps: int, noise: bool, warmup: int, tag: str, log):
    import torch

    ac = alg.actor_critic
    if noise:
        ac.act = ac.__class__.act.__get__(ac, ac.__class__)  # 还原原始采样
    else:
        def _act_mean(observations, **kwargs):  # 只用均值（等价 act_inference）
            ac.update_distribution(observations)
            return ac.distribution.mean

        ac.act = _act_mean

    obs, extras = env.get_observations()
    obs, critic_obs, obs_pack = runner._normalize_observation_pack(extras["observations"])

    acc: dict[str, list[float]] = defaultdict(list)
    j_mean: list[float] = []
    j_samp: list[float] = []
    hist_m = deque(maxlen=3)
    hist_s = deque(maxlen=3)

    with torch.inference_mode():
        for i in range(steps):
            if hasattr(alg, "act_with_obs_pack"):
                actions = alg.act_with_obs_pack(obs, critic_obs, obs_pack)
            else:
                actions = alg.act(obs, critic_obs)
            am = getattr(alg.transition, "action_mean", None)
            hist_m.append(None if am is None else am.detach().clone())
            hist_s.append(actions.detach().clone())
            if len(hist_m) == 3 and i >= warmup:
                j_mean.append(float((hist_m[0] - 2 * hist_m[1] + hist_m[2]).abs().mean()))
                j_samp.append(float((hist_s[0] - 2 * hist_s[1] + hist_s[2]).abs().mean()))

            # ⚠️ 故意**不**调用 process_env_step：wasabi 会往固定容量的 amp_storage 里塞
            # transition，跑 600 步（>> num_steps_per_env）必然 `Rollout buffer overflow`。
            # 我们只需要 env 探针 + 动作统计，不需要写 rollout buffer，所以整段跳过。
            obs, rewards, dones, infos = env.step(actions)
            obs, critic_obs, obs_pack = runner._normalize_observation_pack(infos["observations"])

            if i >= warmup:
                for k, v in infos.get("step", {}).items():
                    acc[k].append(float(v))
                for k in WATCH:  # 兜底：万一探针写在了 infos 顶层
                    if k not in acc and isinstance(infos.get(k), (int, float)):
                        acc[k].append(float(infos[k]))

    res = {k: (statistics.mean(v) if v else float("nan")) for k, v in acc.items()}
    res["_jerk_mean_action"] = statistics.mean(j_mean) if j_mean else float("nan")
    res["_jerk_sampled_action"] = statistics.mean(j_samp) if j_samp else float("nan")
    res["_sigma"] = float(ac.std.mean())
    lines = [f"\n===== 模式 {tag}（noise={noise}）====="]
    for k in WATCH:
        if k in res:
            lines.append(f"   {k:<32} {res[k]:+.4f}")
    lines.append(f"   {'[动作] |Δ²a_mean|':<32} {res['_jerk_mean_action']:.4f} rad   ← 意图 jerk")
    lines.append(f"   {'[动作] |Δ²a_sampled|':<32} {res['_jerk_sampled_action']:.4f} rad   ← 含噪 jerk")
    lines.append(f"   {'[动作] sigma（ckpt 里的）':<32} {res['_sigma']:.4f} rad")
    lines.append(f"   [未收录探针] {sorted(set(acc) - set(WATCH))[:12]}")
    print("\n".join(lines), flush=True)   # ← 直接落盘（log 列表只是给调用方归档）
    log.extend(lines)
    return res


def main(log) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--load_run", required=True, help="run 目录的**绝对路径**")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--drop_amp", action="store_true", help="丢掉 ckpt 里的判别器（AMP 输入维度变了时必须）")
    args = ap.parse_args()
    log.append(f"参数：{vars(args)}  结果文件={OUT}")

    import gymnasium as gym
    import torch

    import instinctlab.tasks  # noqa: F401  （注册 gym 任务）
    from instinct_rl.runners import OnPolicyRunner
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

    spec = gym.spec(args.task)
    env_cfg_cls = spec.kwargs["env_cfg_entry_point"]
    runner_cfg_path = spec.kwargs["instinct_rl_cfg_entry_point"]
    mod_name, cls_name = runner_cfg_path.split(":")
    runner_cfg_cls = getattr(importlib.import_module(mod_name), cls_name)

    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = 0

    agent_cfg = runner_cfg_cls()
    agent_cfg.seed = 0
    agent_cfg.device = "cuda:0"
    if args.drop_amp:
        agent_cfg.ckpt_manipulator = "instinctlab.utils.ckpt_manipulators:drop_amp_discriminator"
    cfg_dict = agent_cfg.to_dict()

    env = InstinctRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
    runner = OnPolicyRunner(env, cfg_dict, log_dir=None, device=agent_cfg.device)
    ckpt = os.path.join(args.load_run, args.checkpoint)
    print(f"载入 {ckpt}", flush=True)
    log.append(f"载入 {ckpt}")
    runner.load(ckpt)
    alg = runner.alg
    print(f"sigma(ckpt) = {float(alg.actor_critic.std.mean()):.4f}   "
          f"(clip_max_std = {cfg_dict.get('clip_max_std')})", flush=True)

    noisy = rollout(env, runner, alg, steps=args.steps, noise=True,
                    warmup=args.warmup, tag="A 有噪声（= 训练时的行为）", log=log)
    mean = rollout(env, runner, alg, steps=args.steps, noise=False,
                   warmup=args.warmup, tag="B 无噪声（只用均值 = ONNX/MuJoCo 回放）", log=log)

    lines = ["\n===== 对比（B/A，越小说明该项越依赖噪声）====="]
    keys = sorted({k for k in WATCH if k in noisy or k in mean})
    for k in keys:
        a, b = noisy.get(k, float("nan")), mean.get(k, float("nan"))
        if a == a and b == b:
            ratio = f"{b / a:.3f}" if a != 0 else "n/a"
            lines.append(f"   {k:<32} 有噪 {a:+.4f}   无噪 {b:+.4f}   B/A = {ratio}")
    sigma = noisy["_sigma"]
    pred = _peak_to_mean_ratio(sigma * (6 ** 0.5))
    jn, jm = noisy["_jerk_sampled_action"], mean["_jerk_mean_action"]
    lines.append(f"\n[动作空间分解] sigma={sigma:.4f} => 噪声对 |d2a| 的理论贡献 ~ {pred:.4f} rad")
    lines.append(f"   实测 有噪 {jn:.4f} / 无噪 {jm:.4f} => 噪声占比 ~ "
                 f"{(1 - jm / max(jn, 1e-9)) * 100:.0f}%")
    print("\n".join(lines), flush=True)
    log.extend(lines)
    return 0


if __name__ == "__main__":
    _log: list[str] = []
    _fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(_fd, 1)
    os.dup2(_fd, 2)
    print("===== D2：动作噪声对高频抖动的贡献 =====", flush=True)

    from isaaclab.app import AppLauncher

    _app = None
    rc = 1
    _err = None
    try:
        _app = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False}).app
        rc = main(_log)
    except BaseException:  # noqa: BLE001
        _err = traceback.format_exc()
        print("抛异常：\n" + _err, flush=True)
    if _err is not None:
        rc = 1
    print(f"RC={rc}", flush=True)
    print(f"[D2 {'PASS' if rc == 0 else 'FAIL'}]", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # ⚠️ 刻意**不**调 `_app.close()`：在这个环境里它会挂住不返回（结果已写完但进程不退，
    # 串行 driver 会永远等下去）。一次性诊断脚本直接退出即可，由 OS 回收资源。
    os._exit(rc)
