#!/usr/bin/env python3
"""指令扫描：本地闭环跑不同速度指令，看哪些会摔倒（判断"回放抽搐"是否由指令引起）。

用法：
    python instinct/tools/command_sweep.py [--steps 100]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", default="instinct/assets/g1_29dof_rev10_play.xml")
    ap.add_argument("--onnx", default="instinct/eval/p1_2000/actor.onnx")
    ap.add_argument("--norm", default="instinct/eval/p1_2000/policy_normalizer.npz")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--decimation", type=int, default=10)
    ap.add_argument("--cmds", nargs="*", default=None,
                    help="格式 'vx,vy,wz'；默认跑一组常用指令")
    ap.add_argument("--joint-order", default="p1",
                    choices=["p1", "isaaclab", "npz", "ctrl"],
                    help="观测/动作关节序（用错序来复现“抽搐”）")
    args = ap.parse_args(argv)

    import mujoco

    import g1_mujoco_play as T
    from joint_order import MUJOCO_ORDER, PELVIS_ORDER, mujoco_state_order_from_model

    cmd_list = ([tuple(float(x) for x in c.split(",")) for c in args.cmds] if args.cmds
                else [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.5, 0.0, 0.0), (0.8, 0.0, 0.0),
                      (1.0, 0.0, 0.0), (-0.5, 0.0, 0.0), (-0.6843, 0.0, -0.093),
                      (0.0, 0.0, 0.5), (0.0, 0.0, -0.5), (0.5, 0.0, 0.5)])

    m = mujoco.MjModel.from_xml_path(str(args.xml))
    d = mujoco.MjData(m)
    st_order = mujoco_state_order_from_model(m)
    key = m.key_qpos[0]
    dflt_st = key[7:7 + 29]
    st2ctrl = [st_order.index(nm) for nm in MUJOCO_ORDER]
    dflt_ctrl = dflt_st[st2ctrl]
    from joint_order import ORDER_CANDIDATES
    OBS_ORDER = {"p1": PELVIS_ORDER, "npz": __import__("joint_order").NPZ_ORDER,
                 "isaaclab": ORDER_CANDIDATES["isaaclab"], "ctrl": MUJOCO_ORDER}[args.joint_order]
    pol2ctrl = [OBS_ORDER.index(nm) if nm in OBS_ORDER else -1 for nm in MUJOCO_ORDER]
    print(f"[order] 观测/动作关节序 = {args.joint_order}")
    if -1 in pol2ctrl:
        print("   ⚠️ 该序与 ctrl 序的关节名集合不匹配，跳过")
        return 2
    pol = T.load_policy(str(args.onnx), "auto")
    norm = T.ObsNormalizer(str(args.norm))
    lay = T.AMP_WALK_LAYOUTS["amp1500"]
    T.apply_scan_layout(lay)
    T.set_scan_body("pelvis")

    print(f"{'指令 (vx,vy,wz)':>22s} {'末z':>7s} {'最低z':>7s} {'Δt摔倒':>7s} "
          f"{'Δxy(实际)':>10s} {'|action|max':>11s} {'末5步 rootΔz 抖动':>16s}")
    print("-" * 92)
    for cmdc in cmd_list:
        builder = T.AmpWalkObsBuilder(m, d, dflt_st.copy(), n_joint=29,
                                      torso_body_name="pelvis",
                                      obs_joint_order=args.joint_order, layout=lay)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        cmd = np.asarray(cmdc, float)
        last = np.zeros(29)
        zmin = 1.0
        fell = None
        amax = 0.0
        zs = []
        for t in range(args.steps):
            raw = builder.build(cmd, last)
            a = np.asarray(pol(norm(raw.astype(np.float32).reshape(1, -1)))).reshape(-1)
            last = a.copy()
            amax = max(amax, float(np.abs(a).max()))
            tgt = dflt_ctrl + a[pol2ctrl]
            for _ in range(args.decimation):
                d.ctrl[:] = tgt
                mujoco.mj_step(m, d)
            zs.append(d.qpos[2])
            zmin = min(zmin, d.qpos[2])
            if fell is None and d.qpos[2] < 0.55:
                fell = (t + 1) * 0.02
        xy = np.linalg.norm(d.qpos[:2])
        jitter = float(np.std(np.diff(zs[-5:]))) if len(zs) >= 6 else float("nan")
        ok = "✅" if fell is None else "❌"
        print(f"{str(tuple(round(v,3) for v in cmdc)):>22s} {d.qpos[2]:7.3f} {zmin:7.3f} "
              f"{(f'{fell:.2f}s' if fell else '—'):>7s} {xy:10.3f} {amax:11.3f} {jitter:16.4f}  {ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
