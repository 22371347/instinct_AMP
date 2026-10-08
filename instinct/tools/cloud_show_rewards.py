#!/usr/bin/env python3
"""打印指定 cfg 里若干奖励项的**完整实现细节**（func 全名 + 权重 + 全部 params）。

与 `cloud_dump_cfg_rewards.py`（只打权重/是否 None）的分工：本脚本用于**逐字复刻**
某个奖励项 —— 权重相同但 `func`/`params` 不同是"同名不同实现"事故的根源（2026-10-07
T1a vs T0 审计踩过），复刻时必须把 params 完整取出来。

用法（云端）：
    python3 /root/cloud_show_rewards.py <mod:Class> <项名> [项名...]
    # 例：
    python3 /root/cloud_show_rewards.py \
        instinctlab.tasks.locomotion.config.g1.omni_loco_env_cfg:G1OmniLocoEnvCfg \
        feet_air_time dof_pos_limits dof_acc_l2
"""

from __future__ import annotations

import sys

from isaaclab.app import AppLauncher  # noqa: E402

_APP = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False}).app

import importlib  # noqa: E402


def fname(o) -> str:
    if o is None:
        return "None"
    return f"{getattr(o, '__module__', '?')}.{getattr(o, '__qualname__', repr(o))}"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    mod_name, _, cls_name = sys.argv[1].partition(":")
    cfg = getattr(importlib.import_module(mod_name), cls_name)()
    print(f"# {sys.argv[1]}   run_name={getattr(cfg, 'run_name', '?')}")
    for n in sys.argv[2:]:
        v = getattr(cfg.rewards, n, None)
        print(f"\n## rewards.{n}")
        if v is None:
            print("   <None 已删除>")
            continue
        print(f"   weight = {getattr(v, 'weight', None)!r}")
        print(f"   func   = {fname(getattr(v, 'func', None))}")
        p = getattr(v, "params", {}) or {}
        if not p:
            print("   params = {}")
        for k, val in p.items():
            # 不截断：复刻需要完整信息
            print(f"   params[{k}] = {val!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
