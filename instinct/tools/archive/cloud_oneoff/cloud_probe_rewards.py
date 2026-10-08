#!/usr/bin/env python3
"""打印任务**实际生效**的奖励项（名字 + 权重 + 总数），并列出被置 None 删掉的项。

用途：改奖励集后确认 `RewardManager` 真的按预期加载，而不是被继承覆盖掉。

用法：
  python cloud_probe_rewards.py --task Instinct-Amp-Walk-V2-G1-v0
"""
import argparse
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--task", required=True)
ap.add_argument("--expect", type=int, default=None, help="期望的生效项数，用于 PASS/FAIL")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import instinctlab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)

print("=" * 78)
print(f"task = {args.task}")
print("=" * 78)

# --- 1. 配置层面：哪些 term 被置 None（= 删除） ------------------------------
cfg_terms = dict(vars(cfg.rewards))
none_names = sorted(k for k, v in cfg_terms.items() if v is None)
live_in_cfg = sorted(k for k, v in cfg_terms.items() if v is not None)
print(f"\n[配置层] rewards 字段总数 = {len(cfg_terms)}")
print(f"[配置层] 被删除(None) {len(none_names)} 项: {none_names}")
print(f"[配置层] 保留        {len(live_in_cfg)} 项")

# --- 1b. 指令分布（分档概率 + 范围）-----------------------------------------
try:
    c = cfg.commands.base_velocity
    tot = float(c.prob_low) + float(c.prob_side) + float(c.prob_lin) + float(c.prob_back)
    print("\n[cmd] resampling_time_range =", c.resampling_time_range)
    print(f"[cmd] prob_low={c.prob_low}  prob_side={c.prob_side}  prob_lin={c.prob_lin}  "
          f"prob_lin_fwd={c.prob_lin_fwd}  prob_back={c.prob_back}")
    print(f"[cmd] 分档和 = {tot:.2f}（余 {1.0 - tot:.2f} 为均匀斜向）")
    print(f"[cmd] low_vel={c.low_vel}  zero_frac_in_low={c.zero_frac_in_low}")
    print(f"[cmd] straight_lin_vel_range={c.straight_lin_vel_range}  "
          f"back_lin_vel_range={c.back_lin_vel_range}")
    print(f"[cmd] ranges: vx={c.ranges.lin_vel_x}  vy={c.ranges.lin_vel_y}  wz={c.ranges.ang_vel_z}")
    print(f"[cmd] rel_standing_envs={c.rel_standing_envs}  heading_command={c.heading_command}")
except Exception as exc:  # noqa: BLE001
    print(f"\n[cmd] ⚠️ 读取失败: {type(exc).__name__}: {exc}")

# --- 1c. 参考动作数据集 ------------------------------------------------------
try:
    import yaml as _yaml

    for name, buf in cfg.scene.motion_reference.motion_buffers.items():
        path = getattr(buf, "filtered_motion_selection_filepath", None)
        n_sel = -1
        if path and os.path.isfile(path):
            with open(path) as fh:
                n_sel = len((_yaml.safe_load(fh) or {}).get("selected_files", []))
        print(f"\n[motion] buffer='{name}'")
        print(f"[motion] 选择清单 = {path}")
        print(f"[motion] selected_files = {n_sel} 个（-1 = 读不到文件）")
        print(f"[motion] supported_endings={getattr(buf, 'supported_file_endings', None)}")
except Exception as exc:  # noqa: BLE001
    print(f"\n[motion] ⚠️ 读取失败: {type(exc).__name__}: {exc}")

# --- 2. 运行层面：RewardManager 实际加载了哪些 -------------------------------
env = gym.make(args.task, cfg=cfg).unwrapped
rm = env.reward_manager
print(f"\n[运行层] reward_manager.active_terms 总数 = {len(rm.active_terms)}")

# 权重表：RewTermCfg 的 weight + func 名
rows = []
for name in sorted(rm.active_terms):
    cfg_term = cfg_terms.get(name)
    w = getattr(cfg_term, "weight", None)
    fn = getattr(getattr(cfg_term, "func", None), "__name__", "?")
    rows.append((name, w, fn))

print(f"\n{'序号':<5}{'奖励项':<34}{'权重':>12}  {'func'}")
print("-" * 90)
for i, (name, w, fn) in enumerate(rows):
    ws = f"{w:.6g}" if isinstance(w, (int, float)) else str(w)
    print(f"{i:<5}{name:<34}{ws:>12}  {fn}")

# --- 3. 结论 ---------------------------------------------------------------
n = len(rm.active_terms)
if args.expect is not None:
    verdict = "PASS" if n == args.expect else "FAIL"
    print(f"\n[REWARD-SET {verdict}] {args.task}: active_terms={n} (expect={args.expect})")
else:
    print(f"\n[REWARD-SET INFO] {args.task}: active_terms={n}")

# --- 4. agent cfg（策略/算法侧的关键超参）------------------------------------
try:
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg = load_cfg_from_registry(args.task, "instinct_rl_cfg_entry_point")
    pol = getattr(agent_cfg, "policy", None)
    alg = getattr(agent_cfg, "algorithm", None)
    print(f"\n[agent] class = {type(agent_cfg).__name__}")
    print(f"[agent] init_noise_std  = {getattr(pol, 'init_noise_std', '?')}"
          f"    <- 绝对 σ（rad）；scale=1.0 下即关节目标噪声幅度")
    print(f"[agent] clip_min_std    = {getattr(alg, 'clip_min_std', '?')}")
    print(f"[agent] actor_hidden    = {getattr(pol, 'actor_hidden_dims', '?')}")
    print(f"[agent] experiment_name = {getattr(agent_cfg, 'experiment_name', '?')}")
except Exception as exc:  # noqa: BLE001
    print(f"\n[agent] ⚠️ 读取 agent cfg 失败: {type(exc).__name__}: {exc}")

env.close()
simulation_app.close()
