#!/usr/bin/env python3
"""闭环长程对照：本地策略在本地 MuJoCo 里跑**云端同一条指令**，与云端的 10 s 轨迹对比。

回答两个问题：
  1. 本地闭环能撑多久（云端 500 步 = 10 s 全程 root_z 0.742~0.815 不摔）
  2. 与云端轨迹的偏差随时间的增长，以及接触参数标定（solref）能否改善

用法：
    python instinct/tools/closed_loop_vs_cloud.py [--solref-tc 0.005] [--steps 500]
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
    ap.add_argument("--npz", default="instinct/eval/p1_2000/dump_io_p1_long.npz")
    ap.add_argument("--xml", default="instinct/assets/g1_29dof_rev10_play.xml")
    ap.add_argument("--onnx", default="instinct/eval/p1_2000/actor.onnx")
    ap.add_argument("--norm", default="instinct/eval/p1_2000/policy_normalizer.npz")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--decimation", type=int, default=10)
    ap.add_argument("--solref-tc", type=float, default=None,
                    help="覆盖所有 geom 的接触时间常数（越小越硬）；不给则用模型原值")
    args = ap.parse_args(argv)

    import mujoco

    import g1_mujoco_play as T
    from joint_order import MUJOCO_ORDER, PELVIS_ORDER, mujoco_state_order_from_model

    z = np.load(args.npz, allow_pickle=True)
    cloud_order = [str(x) for x in z["joint_names"]]
    n = min(len(z["obs"]), args.steps)
    cmd = np.asarray(z[[k for k in z.files if k.startswith("cmd_")][0]][0], float)

    m = mujoco.MjModel.from_xml_path(str(args.xml))
    if args.solref_tc is not None:
        m.geom_solref[:, 0] = args.solref_tc
        m.geom_solref[:, 1] = 1.0
    d = mujoco.MjData(m)
    st = mujoco_state_order_from_model(m)
    key = m.key_qpos[0]
    dflt_st = key[7:7 + 29]
    st2ctrl = [st.index(nm) for nm in MUJOCO_ORDER]
    dflt_ctrl = dflt_st[st2ctrl]
    pol2ctrl = [PELVIS_ORDER.index(nm) for nm in MUJOCO_ORDER]

    pol = T.load_policy(str(args.onnx), "auto")
    norm = T.ObsNormalizer(str(args.norm))
    lay = T.AMP_WALK_LAYOUTS["amp1500"]
    T.apply_scan_layout(lay)
    T.set_scan_body("pelvis")
    builder = T.AmpWalkObsBuilder(m, d, dflt_st.copy(), n_joint=29,
                                 torso_body_name="pelvis", obs_joint_order="p1", layout=lay)

    # 初始化 = 云端 step0 状态（含线速度，体坐标→世界系）
    q0 = z["rquat"][0]
    w, x, y, zq = q0
    R = np.array([[1 - 2 * (y * y + zq * zq), 2 * (x * y - w * zq), 2 * (x * zq + w * y)],
                  [2 * (x * y + w * zq), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - w * x)],
                  [2 * (x * zq - w * y), 2 * (y * zq + w * x), 1 - 2 * (x * x + y * y)]])
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[0:3] = z["rpos"][0]
    d.qpos[3:7] = q0
    d.qpos[7:7 + 29] = z["jpos"][0][[cloud_order.index(nm) for nm in st]]
    d.qvel[0:3] = R @ z["rlin"][0]
    d.qvel[3:6] = z["rang"][0]
    d.qvel[6:6 + 29] = z["jvel"][0][[cloud_order.index(nm) for nm in st]]
    mujoco.mj_forward(m, d)

    last = np.zeros(29)
    fell = None
    err, zs = [], []
    for t in range(n):
        raw = builder.build(cmd, last)
        a = np.asarray(pol(norm(raw.astype(np.float32).reshape(1, -1)))).reshape(-1)
        last = a.copy()
        tgt = dflt_ctrl + a[pol2ctrl]
        for _ in range(args.decimation):
            d.ctrl[:] = tgt
            mujoco.mj_step(m, d)
        err.append(float(np.linalg.norm(d.qpos[0:3] - z["rpos"][t])))
        zs.append(float(d.qpos[2]))
        if fell is None and d.qpos[2] < 0.55:
            fell = (t + 1) * 0.02
    err = np.array(err)
    zs = np.array(zs)
    print(f"接触 solref 时间常数 = {args.solref_tc if args.solref_tc else '模型原值'}")
    print(f"  本地 root_z: 前 1 s 均值={zs[:50].mean():.4f}  全程最低={zs.min():.4f}  "
          f"末值={zs[-1]:.4f}")
    print(f"  云端 root_z: 全程 [{z['rpos'][:n,2].min():.4f},{z['rpos'][:n,2].max():.4f}]")
    print(f"  与云端位置偏差: 第1s均值={err[:50].mean()*1000:.1f}mm  "
          f"第2s={err[50:100].mean()*1000:.1f}mm  "
          f"第5s={err[250:300].mean()*1000:.1f}mm  末值={err[-1]*1000:.0f}mm")
    print(f"  摔倒时刻 = {f'{fell:.2f}s' if fell else '未摔（≥'+str(n*0.02)+'s）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
