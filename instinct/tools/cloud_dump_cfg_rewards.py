#!/usr/bin/env python3
"""**构建**指定的 env cfg 类，打印它 `rewards` 里每一项的生效状态。

与 `cloud_dump_rewards.py` 的分工
---------------------------------
* `cloud_dump_rewards.py`：读**已训练 run** 落盘的 `params/env.yaml`（事后权威快照）
* 本脚本：**现场构建 cfg**，看"现在这份代码"会生效什么（发车前预检）

为什么需要它：`= None` 表示该项**已被删除**（`ManagerBase` 会跳过 `None` 的 term），
这和"权重设成 0"是**两回事**。只看源码肉眼分辨不出，必须把 cfg 建出来看。

用法（云端）：
    python3 /root/cloud_dump_cfg_rewards.py \\
        instinctlab.tasks.locomotion.config.g1.amp_walk_v33_env_cfg:G1AmpWalkV33SEnvCfg

⚠️ 需要先起 `AppLauncher`（`isaaclab` 子包 / `pxr` 是 app 启动时才挂上的）。
"""

from __future__ import annotations

import sys

ARGS = sys.argv[1:] or ["instinctlab.tasks.locomotion.config.g1.amp_walk_v33_env_cfg:G1AmpWalkV33SEnvCfg"]

# ⚠️ 必须早于任何 instinctlab / isaaclab.envs 的导入
from isaaclab.app import AppLauncher  # noqa: E402

_APP = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False}).app

import dataclasses  # noqa: E402
import importlib  # noqa: E402


def dump(spec: str) -> None:
    mod_name, _, cls_name = spec.partition(":")
    try:
        cls = getattr(importlib.import_module(mod_name), cls_name)
    except Exception as exc:
        print(f"[FATAL] 无法导入 {spec}: {exc}")
        return

    print("=" * 78)
    print(f"# {spec}")
    try:
        cfg = cls()
    except Exception as exc:
        print(f"[FATAL] 构建失败: {type(exc).__name__}: {exc}")
        return

    rew = cfg.rewards
    try:
        names = [f.name for f in dataclasses.fields(rew)]
    except Exception:
        names = [k for k in vars(rew) if not k.startswith("_")]

    rows = []
    for n in names:
        v = getattr(rew, n, None)
        if v is None:
            rows.append((n, None))  # 已删除
        else:
            rows.append((n, getattr(v, "weight", "?")))

    alive = [r for r in rows if r[1] is not None]
    nz = [r for r in alive if isinstance(r[1], (int, float)) and r[1] != 0.0]
    print(f"# 字段共 {len(rows)} 项：已删除(None) {len(rows) - len(alive)} 项 / "
          f"存活 {len(alive)} 项（其中非零 {len(nz)} 项）")
    print(f"# run_name = {getattr(cfg, 'run_name', '?')}")

    def sort_key(item):
        _, w = item
        if w is None:
            return (0, 0.0, item[0])   # None 排最后
        return (1, -(abs(w) if isinstance(w, (int, float)) else 0.0), item[0])

    rows.sort(key=sort_key)
    for n, w in rows:
        if w is None:
            print(f"  -  {n:34s} <None 已删除>")
        else:
            mark = "  " if isinstance(w, (int, float)) and w != 0.0 else "× "
            ws = f"{w:+.6g}" if isinstance(w, (int, float)) else str(w)
            print(f"  {mark}{n:34s} {ws}")


def main() -> int:
    for spec in ARGS:
        dump(spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
