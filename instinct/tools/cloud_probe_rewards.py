#!/usr/bin/env python3
"""打印任务**实际生效**的奖励项（名字 + 权重 + 总数），并列出被置 None 删掉的项。

用途：改奖励集后确认 `RewardManager` 真的按预期加载，而不是被继承覆盖掉。

用法：
  python cloud_probe_rewards.py --task Instinct-Amp-Walk-V2-G1-v0
"""
import argparse

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

env.close()
simulation_app.close()
