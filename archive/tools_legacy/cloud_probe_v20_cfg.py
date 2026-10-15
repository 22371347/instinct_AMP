#!/usr/bin/env python3
"""最小探针：逐步**落盘**（不经 stdout），定位 V20 cfg 在哪一步出问题。

背景：`cloud_verify_v20_cfg.py` 在云端跑到 `===== ① ...` 之后就没有任何输出了，
退出码却是 0（既不是异常、也不是被 kill）。stdout 在 Isaac Sim 起来之后可能被
Kit 接管/丢弃，所以这里所有输出都直接写文件并 flush + fsync。
"""

from __future__ import annotations

import os
import traceback

_F = open("/root/v20_probe.txt", "w", buffering=1)


def w(msg: str) -> None:
    _F.write(msg + "\n")
    _F.flush()
    try:
        os.fsync(_F.fileno())
    except Exception:  # noqa: BLE001
        pass


w("A: start")
try:
    from isaaclab.app import AppLauncher

    w("B: imported AppLauncher")
    _launcher = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False})
    _app = _launcher.app
    w("C: SimulationApp up")

    from instinctlab.tasks.locomotion.config.g1.amp_walk_v20_env_cfg import (
        AMP_V20_BODY_NAMES,
        AMP_V20_FRAME_SIZE,
        AMP_V20_STATE_SIZE,
        G1AmpWalkV20EnvCfg,
    )

    w(f"D: imported v20 env cfg (FRAME_SIZE={AMP_V20_FRAME_SIZE} STATE_SIZE={AMP_V20_STATE_SIZE})")

    env = G1AmpWalkV20EnvCfg()
    w("E: built G1AmpWalkV20EnvCfg")

    pol = env.observations.amp_policy
    ref = env.observations.amp_reference
    w(f"F: pol={type(pol).__name__} ref={type(ref).__name__}")

    total_pol = 0
    for name in ("body_pos_b", "body_ori_b", "body_lin_vel_b", "body_ang_vel_b"):
        t = getattr(pol, name)
        n = len(t.params["asset_cfg"].body_names)
        per = {"body_pos_b": 3, "body_ori_b": 6, "body_lin_vel_b": 3, "body_ang_vel_b": 3}[name]
        total_pol += t.history_length * n * per
        anchor = t.params.get("anchor_cfg")
        anchor_names = anchor.body_names if anchor is not None else None
        w(f"   POL {name:<16} hist={t.history_length} n={n} anchor={anchor_names} "
          f"flatten={t.flatten_history_dim} scale={t.scale} func={getattr(t.func, '__name__', t.func)}")
    total_ref = 0
    for name in ("body_pos_b", "body_ori_b", "body_lin_vel_b", "body_ang_vel_b"):
        t = getattr(ref, name)
        n = len(t.params["asset_cfg"].body_names)
        per = {"body_pos_b": 3, "body_ori_b": 6, "body_lin_vel_b": 3, "body_ang_vel_b": 3}[name]
        total_ref += t.history_length * n * per
        anchor = t.params.get("anchor_cfg")
        anchor_names = anchor.body_names if anchor is not None else None
        w(f"   REF {name:<16} hist={t.history_length} n={n} anchor={anchor_names} "
          f"flatten={t.flatten_history_dim} scale={t.scale} func={getattr(t.func, '__name__', t.func)}")
    w(f"G: policy 总维度={total_pol}  reference 总维度={total_ref}  "
      f"(期望 390/390)")

    interests = tuple(env.scene.motion_reference.link_of_interests)
    missing = [b for b in (*AMP_V20_BODY_NAMES, "torso_link") if b not in interests]
    w(f"H: 参考侧 link_of_interests({len(interests)}) 缺失={missing or '无'}")
    w("I: ALL DONE")
except Exception:  # noqa: BLE001
    w("EXC:\n" + traceback.format_exc())
finally:
    w("J: finally")
    try:
        _app.close()
        w("K: app closed")
    except Exception:  # noqa: BLE001
        w("EXC(close):\n" + traceback.format_exc())
