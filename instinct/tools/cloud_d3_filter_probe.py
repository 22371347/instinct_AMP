#!/usr/bin/env python3
"""**D3：探针去噪诊断** —— 三种"去噪"手段 vs **无噪声真值**的实测对照。

背景（D2 已证实）：V24 的 `hf_acc_all` 135 → 无噪声 13.1，`stand_pose_vel_err` 4.83 →
0.50，动作 jerk 1.158 → 0.084 rad。也就是说我们在训练日志里盯的"抖动"指标 ~90% 在
度量**我们自己灌进去的采样噪声**（σ≈0.57 rad）。

本脚本回答：**能不能把这些探针"去噪"回策略本意？** 三条候选路线全部实测：

| 路线 | 做什么 | 预期 |
|---|---|---|
| ① 时域滤波 | 对**已记录的探针序列**做 EMA / 中值滤波 | 只压方差，**不去偏差** ⇒ 拉不回真值 |
| ② σ² 标定扣除 | 在 3 个噪声幅度（σ, 0.5σ, 0.25σ）测同一探针，对 `y = a + b·σ²` 做外推到 σ→0 | **能**去掉噪声偏差（若噪声贡献确实是 σ² 型） |
| ③ 执行侧低通 | 对**采样动作**做一阶低通 `a ← a + α(a_sampled − a)` 再执行 | 降低等效噪声 σ·√(α/(2−α))，但是"改变行为"，不是纯诊断 |

另外做两个**动作空间**的统计诊断（只用一条含噪 rollout 就能反推意图）：
  - 白噪声签名：若 `a = m + σ·ε`（ε 白），则 `corr(Δa_t, Δa_{t+1}) = −0.5`（任何 lag≥2 ≈ 0）
  - 统计去噪：`Var(Δa_intent) ≈ Var(Δa_noisy) − 2σ²`，与真值 `Var(Δa_mean)` 对比

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
    source /root/.llp_env.sh && unset PYTHONPATH && \
    D3_OUT=/root/d3_v24.txt /opt/conda/envs/isaaclab/bin/python -u /root/cloud_d3_filter_probe.py \
        --task Instinct-Amp-Walk-V24-G1-v0 --load_run <run 目录绝对路径> \
        --checkpoint model_3500.pt --num_envs 128 --steps 600 \
        --modes noisy,mean,scale:0.5,scale:0.25,ema:0.5

结果落 `$D3_OUT`（默认 /root/d3_eval_result.txt），末尾 `[D3 PASS]`。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import traceback
from collections import defaultdict, deque

OUT = os.environ.get("D3_OUT", "/root/d3_eval_result.txt")

WATCH = (
    "hf_acc_all", "hf_acc_legs", "hf_acc_arms", "hf_acc_waist",
    "air_time_hop_frac", "air_time_in_band_frac", "air_time_single_stance",
    "air_time_lift_frac", "air_time_progress_frac", "air_time_proj_ratio",
    "air_time_proj_ema",
    "stand_pose_pos_err", "stand_pose_vel_err", "stand_pose_vel_cost",
    "stand_pose_pos_err_legs", "stand_pose_vel_err_legs",
    "stand_pose_vel_err_arms", "stand_pose_vel_err_waist",
)


# ---------------------------------------------------------------- 滤波器
def ema(xs, tau: float):
    """一阶 EMA，τ 以"样本数"为单位（α = 1/τ）。"""
    if not xs:
        return []
    a = 1.0 / max(tau, 1.0)
    out, s = [], xs[0]
    for v in xs:
        s = s + a * (v - s)
        out.append(s)
    return out


def median_filter(xs, k: int):
    if k < 3:
        return list(xs)
    h = k // 2
    n = len(xs)
    return [statistics.median(xs[max(0, i - h): min(n, i + h + 1)]) for i in range(n)]


def tail_mean(xs, frac: float = 0.25):
    if not xs:
        return float("nan")
    m = max(1, int(len(xs) * frac))
    return statistics.mean(xs[-m:])


def lsq_line(xs, ys):
    """最小二乘 y = a + b·x，返回 (a, b, R²)。"""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, float("nan"), float("nan")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


def acf_per_joint(da_vectors, lag: int) -> float:
    """对 29 维 Δa 时间序列逐关节算自相关，再对关节取均值。"""
    import numpy as np

    if len(da_vectors) <= lag + 2:
        return float("nan")
    A = np.stack([v for v in da_vectors])  # T x 29
    A = A - A.mean(axis=0, keepdims=True)
    num = (A[:-lag] * A[lag:]).sum(axis=0)
    den = (A * A).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    r = r[~np.isnan(r)]
    return float(r.mean()) if r.size else float("nan")


# ---------------------------------------------------------------- 模式
def parse_mode(spec: str):
    """'noisy' | 'mean' | 'scale:0.5' | 'ema:0.5' → (kind, param)"""
    if ":" in spec:
        k, v = spec.split(":", 1)
        return k, float(v)
    return spec, float("nan")


def sigma_eff(spec: str, sigma: float) -> float:
    kind, p = parse_mode(spec)
    if kind == "noisy" or kind == "mean":
        return sigma if kind == "noisy" else 0.0
    if kind == "scale":
        return sigma * p
    if kind == "abs":   # 目标绝对 σ：把采样噪声缩放到 σ=p
        return p
    if kind == "ema":
        return sigma * (p / (2.0 - p)) ** 0.5
    return float("nan")


def make_hook(ac, spec: str, state: dict, sigma: float):
    kind, p = parse_mode(spec)
    if kind == "noisy":
        return ac.__class__.act.__get__(ac, ac.__class__)

    def hook(observations, **kwargs):
        ac.update_distribution(observations)
        mean = ac.distribution.mean
        if kind == "mean":
            return mean
        sampled = ac.distribution.sample()
        if kind == "scale":
            return mean + p * (sampled - mean)
        if kind == "abs":
            return mean + (p / max(sigma, 1e-9)) * (sampled - mean)
        if kind == "ema":
            if state.get("af") is None:
                state["af"] = sampled.clone()
            else:
                state["af"] = state["af"] + p * (sampled - state["af"])
            return state["af"]
        raise ValueError(f"未知模式 {spec}")

    return hook


def rollout(env, runner, alg, *, steps: int, warmup: int, spec: str, log):
    import torch

    ac = alg.actor_critic
    state: dict = {}
    ac.act = make_hook(ac, spec, state, float(ac.std.mean()))

    obs, extras = env.get_observations()
    obs, critic_obs, obs_pack = runner._normalize_observation_pack(extras["observations"])

    series: dict[str, list[float]] = defaultdict(list)
    da_series: list = []
    hist_exec: deque = deque(maxlen=3)
    hist_mean: deque = deque(maxlen=3)
    prev_exec = None

    with torch.inference_mode():
        for i in range(steps):
            if hasattr(alg, "act_with_obs_pack"):
                actions = alg.act_with_obs_pack(obs, critic_obs, obs_pack)
            else:
                actions = alg.act(obs, critic_obs)
            am = getattr(alg.transition, "action_mean", None)
            if am is not None:
                am = am.detach()
                hist_mean.append(am.clone())
            hist_exec.append(actions.detach().clone())
            if len(hist_exec) == 3 and i >= warmup:
                series["jerk_exec"].append(
                    float((hist_exec[0] - 2 * hist_exec[1] + hist_exec[2]).abs().mean())
                )
                if len(hist_mean) == 3:
                    series["jerk_mean"].append(
                        float((hist_mean[0] - 2 * hist_mean[1] + hist_mean[2]).abs().mean())
                    )
            if prev_exec is not None and i >= warmup:
                da_series.append(((actions.detach() - prev_exec).mean(dim=0)).cpu().numpy())
            prev_exec = actions.detach().clone()

            # ⚠️ 不调 process_env_step（wasabi 的 amp_storage 会溢出；我们不需要 rollout buffer）
            obs, rewards, dones, infos = env.step(actions)
            obs, critic_obs, obs_pack = runner._normalize_observation_pack(infos["observations"])

            if i >= warmup:
                # ⭐ 奖励账目：env.step 返回的真实每步奖励 + 逐 env 标准差（不可预测分量）
                series["_env_reward"].append(float(rewards.mean()))
                series["_env_reward_std"].append(float(rewards.std()))
                rm = getattr(env.unwrapped, "reward_manager", None)
                if rm is not None and hasattr(rm, "_step_reward"):
                    with torch.no_grad():
                        sr = rm._step_reward.mean(dim=0)
                        names = list(getattr(rm, "_term_names", [])) or [
                            str(j) for j in range(sr.numel())
                        ]
                        for j, nm in enumerate(names):
                            if j < sr.numel():
                                series["R:" + str(nm)].append(float(sr[j]))
                for k, v in infos.get("step", {}).items():
                    series[k].append(float(v))
                for k in WATCH:
                    if k not in series and isinstance(infos.get(k), (int, float)):
                        series[k].append(float(infos[k]))

    return series, da_series, float(ac.std.mean())


def main(log) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--load_run", required=True, help="run 目录的**绝对路径**")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--modes", default="noisy,mean,scale:0.5,scale:0.25,ema:0.5")
    ap.add_argument("--drop_amp", action="store_true")
    ap.add_argument("--dump", default="/root/d3_series.json")
    args = ap.parse_args()
    specs = [s.strip() for s in args.modes.split(",") if s.strip()]
    print(f"参数：{vars(args)}\n模式：{specs}", flush=True)

    import gymnasium as gym
    import torch

    import instinctlab.tasks  # noqa: F401
    from instinct_rl.runners import OnPolicyRunner
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

    spec_info = gym.spec(args.task)
    env_cfg = spec_info.kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = 0
    mod_name, cls_name = spec_info.kwargs["instinct_rl_cfg_entry_point"].split(":")
    agent_cfg = getattr(importlib.import_module(mod_name), cls_name)()
    agent_cfg.seed = 0
    agent_cfg.device = "cuda:0"
    if args.drop_amp:
        agent_cfg.ckpt_manipulator = "instinctlab.utils.ckpt_manipulators:drop_amp_discriminator"
    cfg_dict = agent_cfg.to_dict()

    env = InstinctRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
    runner = OnPolicyRunner(env, cfg_dict, log_dir=None, device=agent_cfg.device)
    ckpt = os.path.join(args.load_run, args.checkpoint)
    print(f"载入 {ckpt}", flush=True)
    runner.load(ckpt)
    alg = runner.alg
    sigma = float(alg.actor_critic.std.mean())
    print(f"sigma(ckpt) = {sigma:.4f}   clip_max_std = {cfg_dict.get('clip_max_std')}", flush=True)

    # ---------------- 跑各模式 ----------------
    res: dict[str, dict] = {}
    for spec in specs:
        series, da_series, sig = rollout(
            env, runner, alg, steps=args.steps, warmup=args.warmup, spec=spec, log=log
        )
        res[spec] = {
            "sigma": sig,
            "sigma_eff": sigma_eff(spec, sigma),
            "series": series,
            "da": da_series,
        }
        mean_of = {
            k: statistics.mean(v)
            for k, v in series.items()
            if k not in ("jerk_exec", "jerk_mean") and v
        }
        print(
            f"\n== 模式 {spec}（σ_eff={sigma_eff(spec, sigma):.4f} rad）"
            f"  hf_acc_all={mean_of.get('hf_acc_all', float('nan')):.2f}"
            f"  vel_err={mean_of.get('stand_pose_vel_err', float('nan')):.3f}"
            f"  hop={mean_of.get('air_time_hop_frac', float('nan')):.3f}"
            f"  in_band={mean_of.get('air_time_in_band_frac', float('nan')):.3f}"
            f"  jerk_exec={statistics.mean(series.get('jerk_exec', [float('nan')])):.4f}"
            f"  jerk_mean={statistics.mean(series.get('jerk_mean', [float('nan')])):.4f}",
            flush=True,
        )

    truth = "mean" if "mean" in res else None

    # ---------------- ① 时域滤波能否去掉噪声？ ----------------
    print("\n\n" + "=" * 96, flush=True)
    print("① 时域滤波（对已记录的探针序列做 EMA / 中值）—— 能不能拉回到无噪声真值？", flush=True)
    print("=" * 96, flush=True)
    if truth is None:
        print("（没有跑 'mean' 模式，跳过真值对照）", flush=True)
    else:
        hdr = (f"{'探针':<26}{'真值(B)':>12}{'原始(A)':>12}{'EMA τ=5':>12}"
               f"{'EMA τ=20':>12}{'中值 k=11':>12}{'A 偏差':>12}")
        print(hdr, flush=True)
        for k in WATCH:
            a = res["noisy"]["series"].get(k)
            b = res[truth]["series"].get(k)
            if not a or not b:
                continue
            av, bv = statistics.mean(a), statistics.mean(b)
            e5 = tail_mean(ema(a, 5))
            e20 = tail_mean(ema(a, 20))
            m11 = tail_mean(median_filter(a, 11))
            print(f"{k:<26}{bv:>12.4f}{av:>12.4f}{e5:>12.4f}{e20:>12.4f}{m11:>12.4f}"
                  f"{av - bv:>12.4f}", flush=True)
        print("\n判读：'A 偏差' 列若远大于 0，而 EMA/中值列与 '原始(A)' 基本相同 ⇒ "
              "时域滤波只压方差、不去偏差，**无法**替代无噪声口径。", flush=True)

    # ---------------- ② σ² 标定扣除 ----------------
    print("\n" + "=" * 96, flush=True)
    print("② σ² 标定扣除：用 3 个噪声幅度拟合 y = a + b·σ_eff²，外推到 σ→0，与真值比", flush=True)
    print("=" * 96, flush=True)
    pure = [s for s in specs if parse_mode(s)[0] in ("noisy", "scale", "abs", "mean")]
    if len(pure) >= 3 and truth is not None:
        print(f"用于拟合的模式：{pure}（纯 σ 缩放的严格模式不含 ema，因为 ema 还带相位滞后）",
              flush=True)
        print(f"{'探针':<26}{'真值(B)':>12}{'σ→0 外推':>12}{'相对误差':>12}{'b(斜率)':>14}{'R²':>8}",
              flush=True)
        for k in WATCH:
            xs, ys = [], []
            for s in pure:
                v = res[s]["series"].get(k)
                if v:
                    xs.append(res[s]["sigma_eff"] ** 2)
                    ys.append(statistics.mean(v))
            if len(xs) < 3:
                continue
            a0, b0, r2 = lsq_line(xs, ys)
            bv = statistics.mean(res[truth]["series"][k])
            rel = (a0 - bv) / bv * 100 if bv else float("nan")
            print(f"{k:<26}{bv:>12.4f}{a0:>12.4f}{rel:>11.1f}%{b0:>14.4g}{r2:>8.3f}", flush=True)

    # ---------------- ③ 执行侧低通 ----------------
    print("\n" + "=" * 96, flush=True)
    print("③ 执行侧低通（对采样动作做 EMA 再执行）—— 能恢复多少？", flush=True)
    print("=" * 96, flush=True)
    print(f"{'模式':<14}{'σ_eff':>10}{'hf_acc_all':>13}{'vel_err':>11}{'hop_frac':>11}"
          f"{'in_band':>10}{'jerk_exec':>12}{'jerk_mean':>11}", flush=True)
    for s in specs:
        r = res[s]
        g = lambda k: statistics.mean(r["series"][k]) if r["series"].get(k) else float("nan")
        je = statistics.mean(r["series"]["jerk_exec"]) if r["series"].get("jerk_exec") else float("nan")
        jm = statistics.mean(r["series"]["jerk_mean"]) if r["series"].get("jerk_mean") else float("nan")
        print(f"{s:<14}{r['sigma_eff']:>10.4f}{g('hf_acc_all'):>13.2f}"
              f"{g('stand_pose_vel_err'):>11.3f}{g('air_time_hop_frac'):>11.3f}"
              f"{g('air_time_in_band_frac'):>10.3f}{je:>12.4f}{jm:>11.4f}", flush=True)

    # ---------------- 动作空间统计诊断 ----------------
    print("\n" + "=" * 96, flush=True)
    print("④ 动作空间：白噪声签名 + 统计去噪（只用一条含噪 rollout 反推意图）", flush=True)
    print("=" * 96, flush=True)
    import numpy as np

    for s in specs:
        da = res[s]["da"]
        if len(da) < 20:
            continue
        A = np.stack(da)  # T x 29
        var_da = float((A ** 2).mean())
        acfs = [acf_per_joint(da, lag) for lag in (1, 2, 3)]
        print(f"{s:<14} Var(Δa)={var_da:>9.5f}   acf(lag1,2,3)="
              f"{acfs[0]:>7.3f}{acfs[1]:>7.3f}{acfs[2]:>7.3f}", flush=True)
    if truth is not None and "noisy" in res:
        var_n = float((np.stack(res["noisy"]["da"]) ** 2).mean())
        var_m = float((np.stack(res[truth]["da"]) ** 2).mean())
        print(f"\n理论：白噪声下 Var(Δa)=2σ² ⇒ 2σ² = {2 * sigma ** 2:.5f}"
              f"（σ={sigma:.4f}）；白噪声签名 acf(lag1) 应 ≈ −0.5", flush=True)
        print(f"统计去噪：Var(Δa_intent) ≈ Var(Δa_noisy) − 2σ² = {var_n:.5f} − "
              f"{2 * sigma ** 2:.5f} = {var_n - 2 * sigma ** 2:.5f}", flush=True)
        print(f"         真值（mean 模式）Var(Δa_mean) = {var_m:.5f}"
              f" ⇒ 相对误差 {(var_n - 2 * sigma ** 2 - var_m) / max(var_m, 1e-12) * 100:+.1f}%",
              flush=True)

    # ---------------- ⑤ 奖励账目：噪声罚占多少？ ----------------
    print("\n" + "=" * 96, flush=True)
    print("⑤ 奖励账目：逐项加权奖励（每步、env 平均）—— 噪声各占多少？", flush=True)
    print("=" * 96, flush=True)
    if truth is not None:
        rk = sorted(k for k in res["noisy"]["series"] if k.startswith("R:"))
        print(f"{'奖励项':<34}{'含噪(A)':>12}{'无噪(B)':>12}{'噪声部分':>12}{'噪声占':>9}",
              flush=True)
        tot_n = tot_m = tot_d = 0.0
        for k in rk:
            a = res["noisy"]["series"].get(k)
            b = res[truth]["series"].get(k)
            if not a or not b:
                continue
            av, bv = statistics.mean(a), statistics.mean(b)
            d = av - bv
            tot_n += av
            tot_m += bv
            tot_d += d
            if abs(d) < 5e-3 and abs(av) < 5e-3:
                continue
            share = f"{d / av * 100:>7.0f}%" if abs(av) > 1e-6 else "     n/a"
            print(f"{k[2:]:<34}{av:>12.4f}{bv:>12.4f}{d:>12.4f}{share:>9}", flush=True)
        print(f"{'—— 逐项合计':<34}{tot_n:>12.4f}{tot_m:>12.4f}{tot_d:>12.4f}", flush=True)
        en = statistics.mean(res["noisy"]["series"].get("_env_reward", [float("nan")]))
        em = statistics.mean(res[truth]["series"].get("_env_reward", [float("nan")]))
        sn = statistics.mean(res["noisy"]["series"].get("_env_reward_std", [float("nan")]))
        sm = statistics.mean(res[truth]["series"].get("_env_reward_std", [float("nan")]))
        print(f"{'env.step 返回的每步奖励':<34}{en:>12.4f}{em:>12.4f}{en - em:>12.4f}", flush=True)
        print(f"{'逐 env 奖励标准差(不可预测)':<34}{sn:>12.4f}{sm:>12.4f}{sn - sm:>12.4f}", flush=True)
        print(f"\n⇒ 信噪比（|总奖励| / 逐env标准差）：含噪 {abs(en) / max(sn, 1e-9):.2f}"
              f"   无噪 {abs(em) / max(sm, 1e-9):.2f}", flush=True)
        if abs(en) > 1e-9:
            print(f"⇒ 噪声对总奖励的贡献占比：(A−B)/|A| = {(en - em) / abs(en) * 100:.0f}%", flush=True)

    # ---------------- 落盘序列（供离线复查） ----------------
    try:
        dump = {
            s: {
                "sigma_eff": res[s]["sigma_eff"],
                "series": {k: v for k, v in res[s]["series"].items()},
            }
            for s in specs
        }
        with open(args.dump, "w") as f:
            json.dump(dump, f)
        print(f"\n[序列已落盘] {args.dump}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[序列落盘失败] {e}", flush=True)

    return 0


if __name__ == "__main__":
    _log: list[str] = []
    _fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(_fd, 1)
    os.dup2(_fd, 2)
    print("===== D3：探针去噪诊断（滤波 / σ² 标定 / 执行侧低通）=====", flush=True)

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
    print(f"[D3 {'PASS' if rc == 0 else 'FAIL'}]", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)  # ⚠️ 不调 _app.close()：它会挂住不返回
