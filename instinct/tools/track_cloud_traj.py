#!/usr/bin/env python3
"""**物理隔离实验**：本地 MuJoCo 用**云端的动作**开环驱动，看动力学能否跟随云端轨迹。

为什么需要它：观测链已经彻底证明与云端一致（布局 / 关节序 / 历史方向 / 默认姿 /
真值 Δ=0；观测差异只剩云端的 `Unoise` 噪声）。所以若本地仍摔倒，剩下的可能就是
**动力学/执行器层面**的差异。本脚本把「策略+观测」整条链**旁路掉**：

    init 本地状态 = 云端状态[0]
    for t:  ctrl = 默认姿 + 云端动作[t]（p1 序 → ctrl 序），跑 10 × mj_step
            与云端状态[t+1] 逐项比较（根位置 / 姿态 / 关节角 / 关节速度）

⇒ 若误差在几十步内保持很小 ⇒ 动力学等效，问题必然在闭环（观测/时序）；
   若误差立刻爆炸 ⇒ 动力学/执行器（增益、限幅、接触、摩擦、积分器）不一致。

用法：
    python instinct/tools/track_cloud_traj.py [--npz ...] [--steps 80]
    python instinct/tools/track_cloud_traj.py --closed-loop   # 改用本地策略闭环（对照）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))


def quat_angle(q1, q2):
    """两个 (w,x,y,z) 四元数之间的旋转角（rad）。"""
    d = abs(float(np.dot(q1, q2)))
    return 2.0 * np.arccos(min(1.0, d))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="instinct/eval/p1_2000/dump_io_p1_2000.npz")
    ap.add_argument("--xml", default="instinct/assets/g1_29dof_rev10_play.xml")
    ap.add_argument("--onnx", default="instinct/eval/p1_2000/actor.onnx")
    ap.add_argument("--norm", default="instinct/eval/p1_2000/policy_normalizer.npz")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--decimation", type=int, default=10)
    ap.add_argument("--closed-loop", action="store_true",
                    help="用本地策略闭环驱动（对比开环）")
    ap.add_argument("--command", type=float, nargs=3, default=None,
                    help="闭环时使用的速度指令（默认取云端记录的命令）")
    args = ap.parse_args(argv)

    import mujoco

    import g1_mujoco_play as T
    from joint_order import MUJOCO_ORDER, PELVIS_ORDER, mujoco_state_order_from_model

    z = np.load(args.npz, allow_pickle=True)
    cloud_order = [str(x) for x in z["joint_names"]]
    n = min(len(z["obs"]), args.steps)
    cmd = z[[k for k in z.files if k.startswith("cmd_")][0]]

    m = mujoco.MjModel.from_xml_path(str(args.xml))
    d = mujoco.MjData(m)
    st_order = mujoco_state_order_from_model(m)
    key = m.key_qpos[0]
    perm = [cloud_order.index(nm) for nm in st_order]        # 本地状态槽 ← 云端槽
    dflt_st = key[7:7 + 29]
    # 默认姿在 ctrl 序下
    st2ctrl = [st_order.index(nm) for nm in MUJOCO_ORDER]
    dflt_ctrl = dflt_st[st2ctrl]
    # 策略序 → ctrl 序
    pol2ctrl = [PELVIS_ORDER.index(nm) for nm in MUJOCO_ORDER]

    # 初始化：默认姿 + 云端初始根位姿
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[0:3] = z["rpos"][0]
    d.qpos[3:7] = z["rquat"][0]
    d.qpos[7:7 + 29] = z["jpos"][0][perm]
    d.qvel[3:6] = z["rang"][0]
    d.qvel[6:6 + 29] = z["jvel"][0][perm]
    mujoco.mj_forward(m, d)

    pol = norm = None
    if args.closed_loop:
        pol = T.load_policy(str(args.onnx), "auto")
        norm = T.ObsNormalizer(str(args.norm))
        lay = T.AMP_WALK_LAYOUTS["amp1500"]
        T.apply_scan_layout(lay)
        T.set_scan_body("pelvis")
        builder = T.AmpWalkObsBuilder(m, d, dflt_st.copy(), n_joint=29,
                                      torso_body_name="pelvis", obs_joint_order="p1",
                                      layout=lay)
        cmdc = np.asarray(args.command if args.command is not None else cmd[0], float)
        print(f"[闭环] 使用指令 {cmdc.tolist()}")
    else:
        print("[开环] 完全使用**云端动作**驱动（策略/观测全旁路）")

    print(f"{'step':>5} {'|Δ根位置|':>10s} {'Δ姿态(rad)':>11s} {'Δ关节角max':>11s} "
          f"{'Δ关节速max':>11s} {'本地root_z':>11s} {'云端root_z':>11s}")
    last = np.zeros(29)
    for t in range(n):
        if args.closed_loop:
            raw = builder.build(cmdc, last)
            a = np.asarray(pol(norm(raw.astype(np.float32).reshape(1, -1)))).reshape(-1)
            last = a.copy()
        else:
            a = z["act"][t]
        tgt = dflt_ctrl + a[pol2ctrl]
        for _ in range(args.decimation):
            d.ctrl[:] = tgt
            mujoco.mj_step(m, d)
        if t + 1 >= n:
            break
        tc = t + 1
        dpos = np.linalg.norm(d.qpos[0:3] - z["rpos"][tc])
        dang = quat_angle(d.qpos[3:7], z["rquat"][tc])
        dq = np.abs(d.qpos[7:7 + 29] - z["jpos"][tc][perm]).max()
        dv = np.abs(d.qvel[6:6 + 29] - z["jvel"][tc][perm]).max()
        if t < 12 or t % 10 == 0 or t + 2 >= n:
            print(f"{tc:5d} {dpos:10.4f} {dang:11.4f} {dq:11.4f} {dv:11.4f} "
                  f"{d.qpos[2]:11.4f} {z['rpos'][tc][2]:11.4f}")
    print(f"\n最终：本地 root_z={d.qpos[2]:.4f}（云端 {z['rpos'][n-1][2]:.4f}）  "
          f"本地 xy={np.round(d.qpos[:2],3).tolist()}（云端 {np.round(z['rpos'][n-1][:2],3).tolist()}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
