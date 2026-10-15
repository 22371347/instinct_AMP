#!/usr/bin/env python3
"""**接触参数标定**：用「单步动力学误差」当目标函数，扫 MuJoCo 接触设置。

背景（已实测）：
  · 策略链、观测量（布局/关节序/历史/尺度/归一化）与云端逐项一致；
  · 但「同一状态 + 同一动作走 20 ms」后，本地与云端的关节角差 **中位 0.0157 rad**，
    误差集中在 **踝 pitch/roll 与膝** ⇒ 签名指向**接触/地面反力模型**：
      - MuJoCo: solref=(0.02 s, ζ=1) 的**软接触** + pyramidal 锥 + impratio=1
      - PhysX : 隐式 PD + 高刚度接触 + TGS 求解
  · 脚部碰撞体是 **4 个 r=5 mm 的小球**（PhysX 用整只脚网格的凸包 → 分布式面接触）。

本脚本在**不落盘**的前提下逐组改 `geom_solref` / `opt.cone` / `opt.impratio` /
`opt.noslip_iterations`（内存态即可生效），用同一批云端状态算误差，挑最小者。

用法：
    python instinct/tools/contact_tune.py [--npz ...] [--states 200]
"""
from __future__ import annotations

import argparse
import itertools
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
    ap.add_argument("--states", type=int, default=150)
    ap.add_argument("--decimation", type=int, default=10)
    args = ap.parse_args(argv)

    import mujoco

    from joint_order import MUJOCO_ORDER, PELVIS_ORDER, mujoco_state_order_from_model

    z = np.load(args.npz, allow_pickle=True)
    cloud_order = [str(x) for x in z["joint_names"]]
    n = min(len(z["obs"]) - 1, args.states)

    def rmat(q):
        w, x, y, zq = q
        return np.array([
            [1 - 2 * (y * y + zq * zq), 2 * (x * y - w * zq), 2 * (x * zq + w * y)],
            [2 * (x * y + w * zq), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - w * x)],
            [2 * (x * zq - w * y), 2 * (y * zq + w * x), 1 - 2 * (x * x + y * y)],
        ])

    def evaluate(*, tc, cone, impratio, noslip, stiff_geom=True):
        m = mujoco.MjModel.from_xml_path(str(args.xml))
        m.opt.cone = cone
        m.opt.impratio = impratio
        m.opt.noslip_iterations = noslip
        if stiff_geom:
            # 地面(0) 与 所有脚部 geom 的接触刚度/阻尼直接改（solref=[timeconst, dampratio]）
            m.geom_solref[:, 0] = tc
            m.geom_solref[:, 1] = 1.0
        d = mujoco.MjData(m)
        st = mujoco_state_order_from_model(m)
        key = m.key_qpos[0]
        perm = [cloud_order.index(nm) for nm in st]
        st2ctrl = [st.index(nm) for nm in MUJOCO_ORDER]
        dflt_ctrl = key[7:7 + 29][st2ctrl]
        pol2ctrl = [PELVIS_ORDER.index(nm) for nm in MUJOCO_ORDER]
        dq, dqd, dpos = [], [], []
        for t in range(n):
            q = z["rquat"][t]
            d.qpos[0:3] = z["rpos"][t]
            d.qpos[3:7] = q
            d.qpos[7:7 + 29] = z["jpos"][t][perm]
            d.qvel[0:3] = rmat(q) @ z["rlin"][t]
            d.qvel[3:6] = z["rang"][t]
            d.qvel[6:6 + 29] = z["jvel"][t][perm]
            mujoco.mj_forward(m, d)
            tgt = dflt_ctrl + z["act"][t][pol2ctrl]
            for _ in range(args.decimation):
                d.ctrl[:] = tgt
                mujoco.mj_step(m, d)
            t1 = t + 1
            dq.append(np.abs(d.qpos[7:7 + 29] - z["jpos"][t1][perm]).max())
            v = d.qvel[6:6 + 29] - z["jvel"][t1][perm]
            dqd.append(float(np.linalg.norm(v)))
            dpos.append(float(np.linalg.norm(d.qpos[0:3] - z["rpos"][t1])))
        dq, dqd, dpos = map(np.array, (dq, dqd, dpos))
        return dict(dq=np.median(dq), dq_p90=np.percentile(dq, 90),
                    dqd=dqd.mean(), dpos=np.median(dpos))

    print(f"基线（原样模型）")
    base = evaluate(tc=0.02, cone=0, impratio=1.0, noslip=0, stiff_geom=False)
    print(f"   Δq中位={base['dq']:.5f}  Δq_p90={base['dq_p90']:.5f}  "
          f"Δq̇L2均值={base['dqd']:.4f}  Δroot中位={base['dpos']:.5f}\n")
    hdr = f"{'solref_tc':>10s} {'cone':>9s} {'impr':>6s} {'noslip':>7s} " \
          f"{'Δq中位':>10s} {'Δq_p90':>10s} {'Δq̇L2':>9s} {'Δroot':>9s}  {'vs基线':>8s}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for tc, cone, impr, noslip in itertools.product(
            (0.02, 0.005, 0.002, 0.001), (0, 1), (1.0, 10.0), (0,)):
        r = evaluate(tc=tc, cone=cone, impratio=impr, noslip=noslip)
        gain = (base["dq"] - r["dq"]) / base["dq"] * 100
        rows.append((tc, cone, impr, noslip, r, gain))
        print(f"{tc:10.4f} {('pyramidal' if cone == 0 else 'elliptic'):>9s} "
              f"{impr:6.1f} {noslip:7d} {r['dq']:10.5f} {r['dq_p90']:10.5f} "
              f"{r['dqd']:9.4f} {r['dpos']:9.5f}  {gain:+7.1f}%")
    rows.sort(key=lambda t: t[4]["dq"])
    best = rows[0]
    print(f"\n最接近 PhysX 的配置：solref_tc={best[0]} cone="
          f"{'pyramidal' if best[1] == 0 else 'elliptic'} impratio={best[2]} "
          f"⇒ Δq中位={best[4]['dq']:.5f}（基线 {base['dq']:.5f}，改善 {best[5]:+.1f}%）")
    # 单独看「只改脚型」的对照（把 4 球换成脚掌盒）
    print("\n提示：脚型对照（4 球 → 脚掌盒）需另生成 XML；见 patch_foot_collision.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
