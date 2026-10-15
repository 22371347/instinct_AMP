#!/usr/bin/env python3
"""**动力学配置探测**：找出让本地 MuJoCo 最接近云端轨迹的积分器/步长组合。

背景：用云端动作开环驱动本地时，本地 ~0.6 s 就与云端发散、~1.2 s 摔倒
（云端一直稳定）⇒ 动力学层面有系统差异。本脚本用「开环跟随误差」+「多久摔倒」
作为打分，扫描若干关键配置：

    integrator ∈ {Euler(0), implicitfast(3)}
    timestep   ∈ {0.002, 0.005}
    solver     ∈ {Newton(2), PGS(0)}
    cone       ∈ {pyramidal(0), elliptic(1)}
    noslip     ∈ {0, 5}

用法：
    python instinct/tools/physics_probe.py [--npz ...] [--steps 60]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))


def quat_angle(q1, q2):
    return 2.0 * np.arccos(min(1.0, abs(float(np.dot(q1, q2)))))


def run_case(z, perm, args, *, integrator, timestep, solver, cone, noslip):
    import mujoco

    from joint_order import MUJOCO_ORDER, PELVIS_ORDER, mujoco_state_order_from_model

    m = mujoco.MjModel.from_xml_path(str(args.xml))
    m.opt.integrator = integrator
    m.opt.timestep = timestep
    m.opt.solver = solver
    m.opt.cone = cone
    m.opt.noslip_iterations = noslip
    d = mujoco.MjData(m)
    st_order = mujoco_state_order_from_model(m)
    key = m.key_qpos[0]
    st2ctrl = [st_order.index(nm) for nm in MUJOCO_ORDER]
    dflt_ctrl = key[7:7 + 29][st2ctrl]
    pol2ctrl = [PELVIS_ORDER.index(nm) for nm in MUJOCO_ORDER]
    dec = args.decimation
    # 均匀拆分 20 ms 控制周期
    nsub = max(1, int(round(args.step_dt / timestep / dec)))
    n = min(len(z["obs"]), args.steps)

    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[0:3] = z["rpos"][0]
    d.qpos[3:7] = z["rquat"][0]
    d.qpos[7:7 + 29] = z["jpos"][0][perm]
    # 线速度：云端 rlin 是**体坐标系** ⇒ 转到世界系（自由关节 qvel[0:3] 是世界系）
    w, x, y, zq = z["rquat"][0]
    R = np.array([
        [1 - 2 * (y * y + zq * zq), 2 * (x * y - w * zq), 2 * (x * zq + w * y)],
        [2 * (x * y + w * zq), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - w * x)],
        [2 * (x * zq - w * y), 2 * (y * zq + w * x), 1 - 2 * (x * x + y * y)],
    ])
    d.qvel[0:3] = R @ z["rlin"][0]
    d.qvel[3:6] = z["rang"][0]
    d.qvel[6:6 + 29] = z["jvel"][0][perm]
    mujoco.mj_forward(m, d)

    err20 = err40 = np.nan
    fell_at = None
    track = []
    for t in range(n):
        tgt = dflt_ctrl + z["act"][t][pol2ctrl]
        for _ in range(dec * nsub):
            d.ctrl[:] = tgt
            mujoco.mj_step(m, d)
        if t + 1 < n:
            e = float(np.linalg.norm(d.qpos[0:3] - z["rpos"][t + 1]))
            track.append(e)
            if t + 1 == 20:
                err20 = e
            if t + 1 == 40:
                err40 = e
        if fell_at is None and d.qpos[2] < 0.55:
            fell_at = t + 1
    return dict(err20=err20, err40=err40, fell=fell_at,
                zend=float(d.qpos[2]), xend=float(d.qpos[0]),
                maxerr=float(np.max(track)) if track else np.nan)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="instinct/eval/p1_2000/dump_io_p1_2000.npz")
    ap.add_argument("--xml", default="instinct/assets/g1_29dof_rev10_play.xml")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--decimation", type=int, default=10)
    ap.add_argument("--step-dt", type=float, default=0.02)
    args = ap.parse_args(argv)

    z = np.load(args.npz, allow_pickle=True)
    cloud_order = [str(x) for x in z["joint_names"]]
    import mujoco  # noqa: F401

    from joint_order import mujoco_state_order_from_model
    m0 = __import__("mujoco").MjModel.from_xml_path(str(args.xml))
    st_order = mujoco_state_order_from_model(m0)
    perm = [cloud_order.index(nm) for nm in st_order]

    print(f"模型原始: integrator={m0.opt.integrator} timestep={m0.opt.timestep} "
          f"solver={m0.opt.solver} cone={m0.opt.cone} noslip={m0.opt.noslip_iterations}")
    print(f"云端 z_end={z['rpos'][args.steps-1][2]:.4f} x_end={z['rpos'][args.steps-1][0]:+.3f}\n")
    hdr = f"{'integrator':>12s} {'dt':>6s} {'solver':>7s} {'cone':>10s} {'noslip':>6s} " \
          f"{'err@20':>8s} {'err@40':>8s} {'摔倒步':>7s} {'末z':>7s} {'末x':>8s}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for integ, dt, solver, cone, noslip in itertools.product(
            (0, 3), (0.002, 0.005), (2, 0), (0, 1), (0, 5)):
        if dt == 0.005 and solver == 0 and cone == 1:
            continue  # 抽样，减少组合
        try:
            r = run_case(z, perm, args, integrator=integ, timestep=dt,
                         solver=solver, cone=cone, noslip=noslip)
        except Exception as exc:  # noqa: BLE001
            print(f"{integ:>12d} {dt:>6.3f} {solver:>7d} {cone:>10d} {noslip:>6d}  ERR {exc}")
            continue
        rows.append((integ, dt, solver, cone, noslip, r))
        name = {0: "Euler", 3: "implicitfast"}[integ]
        sname = {0: "PGS", 2: "Newton"}[solver]
        cname = {0: "pyramidal", 1: "elliptic"}[cone]
        print(f"{name:>12s} {dt:>6.3f} {sname:>7s} {cname:>10s} {noslip:>6d} "
              f"{r['err20']:8.4f} {r['err40']:8.4f} "
              f"{(r['fell'] if r['fell'] else '-'):>7} {r['zend']:7.4f} {r['xend']:+8.3f}")
    rows.sort(key=lambda t: (t[5]["fell"] is None, -(t[5]["fell"] or 0), t[5]["err40"]))
    print("\n最接近云端的 3 个配置（越晚摔倒越好）：")
    for r in rows[:3]:
        integ, dt, solver, cone, noslip, res = r
        print(f"   integrator={ {0:'Euler',3:'implicitfast'}[integ] } dt={dt} "
              f"solver={ {{0:'PGS',2:'Newton'}}[solver] } cone={cone} noslip={noslip} "
              f"⇒ 摔倒步={res['fell']} err@40={res['err40']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
