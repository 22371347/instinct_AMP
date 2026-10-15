#!/usr/bin/env python3
"""**单步动力学误差**检测：在云端记录的每一个状态上，用云端动作走**一个控制周期**，
与云端的下一状态比较。

为什么用它：开环长时间跟随会被混沌放大，无法区分「动力学差一点」和「动力学差很多」。
逐步独立地做单步预测，误差不会被累积放大 ⇒ 得到动力学一致性的**无偏度量**。

输出：每一步的 Δroot位置 / Δ姿态 / Δq_max / Δq̇_max，以及「误差最大的关节」
（若集中在踝/膝 ⇒ 接触模型；若均匀分布 ⇒ 执行器/惯量）。

用法：
    python instinct/tools/one_step_dynamics.py [--npz ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))


def quat_angle(q1, q2):
    return 2.0 * np.arccos(min(1.0, abs(float(np.dot(q1, q2)))))


def rmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="instinct/eval/p1_2000/dump_io_p1_long.npz")
    ap.add_argument("--xml", default="instinct/assets/g1_29dof_rev10_play.xml")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--decimation", type=int, default=10)
    args = ap.parse_args(argv)

    import mujoco

    from joint_order import MUJOCO_ORDER, PELVIS_ORDER, mujoco_state_order_from_model

    z = np.load(args.npz, allow_pickle=True)
    cloud_order = [str(x) for x in z["joint_names"]]
    n = min(len(z["obs"]), args.steps) - 1

    m = mujoco.MjModel.from_xml_path(str(args.xml))
    d = mujoco.MjData(m)
    st = mujoco_state_order_from_model(m)
    key = m.key_qpos[0]
    perm = [cloud_order.index(nm) for nm in st]
    st2ctrl = [st.index(nm) for nm in MUJOCO_ORDER]
    dflt_ctrl = key[7:7 + 29][st2ctrl]
    pol2ctrl = [PELVIS_ORDER.index(nm) for nm in MUJOCO_ORDER]

    rows = []
    dq_per_joint = np.zeros(29)
    dqd_per_joint = np.zeros(29)
    for t in range(n):
        q, v = z["rquat"][t], z["rlin"][t]
        d.qpos[0:3] = z["rpos"][t]
        d.qpos[3:7] = q
        d.qpos[7:7 + 29] = z["jpos"][t][perm]
        d.qvel[0:3] = rmat(q) @ v            # qvel[0:3] 是世界系；云端 rlin 是体系
        d.qvel[3:6] = z["rang"][t]           # MuJoCo 自由关节角速度 = 体系
        d.qvel[6:6 + 29] = z["jvel"][t][perm]
        mujoco.mj_forward(m, d)
        tgt = dflt_ctrl + z["act"][t][pol2ctrl]
        for _ in range(args.decimation):
            d.ctrl[:] = tgt
            mujoco.mj_step(m, d)
        t1 = t + 1
        dpos = float(np.linalg.norm(d.qpos[0:3] - z["rpos"][t1]))
        dang = quat_angle(d.qpos[3:7], z["rquat"][t1])
        dq = np.abs(d.qpos[7:7 + 29] - z["jpos"][t1][perm])
        dqd = np.abs(d.qvel[6:6 + 29] - z["jvel"][t1][perm])
        rows.append((t1, dpos, dang, float(dq.max()), float(dqd.max()),
                     float(np.linalg.norm(d.qvel[6:] - z["jvel"][t1][perm]))))
        dq_per_joint += dq
        dqd_per_joint += dqd

    arr = np.array([r[1:] for r in rows])
    lab = ["Δroot位置(m)", "Δ姿态(rad)", "Δq_max(rad)", "Δq̇_max(rad/s)", "Δq̇_L2"]
    print(f"单步预测误差（{n} 个独立状态，每个走 1 个控制周期 20 ms；云端 {args.npz}）")
    print(f"{'指标':>14s} {'中位':>10s} {'均值':>10s} {'p95':>10s} {'最大':>10s}")
    for i, name in enumerate(lab):
        c = arr[:, i]
        print(f"{name:>14s} {np.median(c):10.4f} {c.mean():10.4f} "
              f"{np.percentile(c, 95):10.4f} {c.max():10.4f}")
    print(f"\nΔq 累计最大的 6 个关节（按名）：")
    order = np.argsort(-dq_per_joint)
    for k in order[:6]:
        print(f"   {st[k]:28s} Σ|Δq|={dq_per_joint[k]:8.3f}  "
              f"Σ|Δq̇|={dqd_per_joint[k]:9.2f}")
    print(f"Δq 最小的 4 个关节：")
    for k in order[-4:]:
        print(f"   {st[k]:28s} Σ|Δq|={dq_per_joint[k]:8.3f}  "
              f"Σ|Δq̇|={dqd_per_joint[k]:9.2f}")
    # 云的根高：确认云端没摔
    print(f"\n云端 root_z: min={z['rpos'][:n,2].min():.4f} max={z['rpos'][:n,2].max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
