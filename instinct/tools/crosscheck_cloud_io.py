#!/usr/bin/env python3
"""云端 IsaacLab ↔ 本地 MuJoCo 回放的**逐步对照**（消费云端 `dump_io.py` 的产物）。

策略链本身已证明逐位一致（`.pt` 与本地 `.onnx` 在同一观测下 |Δ|≤5e-7），
所以问题只可能是「**同一物理状态，两边算出来的观测不同**」。
本脚本把云端的 qpos/qvel 原样注入本地 MuJoCo（按名重排），用**本地工具自己的
观测构造器**连续算 1500 维观测，再与云端同一步的观测逐段/逐帧比对。

用法：
    python instinct/tools/crosscheck_cloud_io.py [--npz ...] [--steps 80]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from joint_order import (  # noqa: E402
    MUJOCO_ORDER, NPZ_ORDER, ORDER_CANDIDATES, PELVIS_ORDER,
    mujoco_state_order_from_model,
)

LAYOUT_NAMES = ["height_scan", "velocity_command", "projected_gravity",
                "base_ang_vel", "joint_pos_rel", "joint_vel", "last_action"]
LAYOUT_PER = {"height_scan": 279, "velocity_command": 3, "projected_gravity": 3,
              "base_ang_vel": 3, "joint_pos_rel": 29, "joint_vel": 29, "last_action": 29}

OFFSETS = {}
_o = 0
for _n in LAYOUT_NAMES:
    OFFSETS[_n] = _o
    _o += LAYOUT_PER[_n] * 4


def seg(v, name, frame=None):
    per, off = LAYOUT_PER[name], OFFSETS[name]
    if frame is None:
        return v[off: off + per * 4]
    return v[off + per * frame: off + per * (frame + 1)]


def corr(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def cand_order(c):
    if c == "p1":
        return PELVIS_ORDER
    if c == "npz":
        return NPZ_ORDER
    if c == "ctrl":
        return MUJOCO_ORDER
    return ORDER_CANDIDATES[c]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="instinct/eval/p1_2000/dump_io_p1_2000.npz")
    ap.add_argument("--onnx", default="instinct/eval/p1_2000/actor.onnx")
    ap.add_argument("--norm", default="instinct/eval/p1_2000/policy_normalizer.npz")
    ap.add_argument("--xml", default="instinct/assets/g1_29dof_rev10_play.xml")
    ap.add_argument("--steps", type=int, default=80, help="连续注入多少步（默认全部）")
    args = ap.parse_args(argv)

    import mujoco

    import g1_mujoco_play as T

    z = np.load(args.npz, allow_pickle=True)
    cloud_order = [str(x) for x in z["joint_names"]]
    n = min(len(z["obs"]), args.steps)
    cmd_key = [k for k in z.files if k.startswith("cmd_")][0]
    cmd = z[cmd_key]

    # ═══ A. 观测布局 ═══════════════════════════════════════════════════════
    cn = [str(x) for x in z["fmt_policy_names"]]
    cd = [int(x) for x in z["fmt_policy_dims"]]
    print("A. 观测量布局（云端 observation_manager 权威 vs 本地工具假设）")
    print(f"   云端: {cn}  {cd}  总={sum(cd)}")
    print(f"   本地: {LAYOUT_NAMES}  {[LAYOUT_PER[k]*4 for k in LAYOUT_NAMES]}  "
          f"总={sum(LAYOUT_PER[k]*4 for k in LAYOUT_NAMES)}")
    print("   ⇒ " + ("✅ 完全一致" if cn == LAYOUT_NAMES and
                    cd == [LAYOUT_PER[k] * 4 for k in LAYOUT_NAMES] else "❌ 不一致"))
    print(f"   云端关节序 = 训练侧 PELVIS_JOINT_ORDER："
          f"{'✅' if tuple(cloud_order) == tuple(PELVIS_ORDER) else '❌'}")

    # ═══ B. 默认关节角 ═════════════════════════════════════════════════════
    m = mujoco.MjModel.from_xml_path(str(args.xml))
    d = mujoco.MjData(m)
    key = m.key_qpos[0]
    st_order = mujoco_state_order_from_model(m)
    loc_default_by_name = dict(zip(st_order, key[7:7 + 29]))
    cd0 = z["dpos"][0]
    dmax = max(abs(loc_default_by_name[nm] - cd0[i]) for i, nm in enumerate(cloud_order))
    print(f"\nB. 默认关节角 Δmax = {dmax:.3e} rad " + ("✅ 一致" if dmax < 1e-6 else "❌ 不一致"))

    # ═══ C. 连续注入：本地构造观测 vs 云端观测 ═════════════════════════════
    lay = T.AMP_WALK_LAYOUTS["amp1500"]
    T.apply_scan_layout(lay)
    T.set_scan_body("pelvis")
    builder = T.AmpWalkObsBuilder(m, d, key[7:7 + 29].copy(), n_joint=29,
                                  torso_body_name="pelvis", obs_joint_order="p1", layout=lay)
    perm = [cloud_order.index(nm) for nm in st_order]   # 本地状态槽 ← 云端槽

    local_obs = np.zeros((n, 1500))
    for t in range(n):
        d.qpos[0:3] = z["rpos"][t]
        d.qpos[3:7] = z["rquat"][t]                # 都是 (w,x,y,z)
        d.qpos[7:7 + 29] = z["jpos"][t][perm]
        d.qvel[0:3] = 0.0
        d.qvel[3:6] = z["rang"][t]                 # MuJoCo 自由关节角速度 = 体坐标系
        d.qvel[6:6 + 29] = z["jvel"][t][perm]
        mujoco.mj_forward(m, d)
        last = z["act"][t - 1] if t > 0 else np.zeros(29)
        local_obs[t] = builder.build(np.asarray(cmd[t], float).reshape(-1), last)

    print(f"\nC. 连续注入 {n} 步：逐段/逐帧 |Δ|max（云端 obs vs 本地构造）")
    print(f"   {'段':22s} {'帧0(旧)':>10s} {'帧1':>10s} {'帧2':>10s} {'帧3(新)':>10s}")
    for s in LAYOUT_NAMES:
        row = []
        for f in range(4):
            dd = np.abs(np.stack([seg(local_obs[t], s, f) for t in range(n)])
                        - np.stack([seg(z["obs"][t], s, f) for t in range(n)]))
            row.append(float(dd.max()))
        flag = "✅" if max(row) < 1e-4 else ("~" if max(row) < 0.02 else "❌")
        print(f" {flag} {s:22s} " + " ".join(f"{v:10.5f}" for v in row))

    # ═══ D. 本地构造器的关节块**顺序**判定 ════════════════════════════════
    print("\nD. 本地构造器关节块顺序判定（与云端真值按关节名对齐，取最新帧）")
    truth = {"joint_pos_rel": z["jpos"] - z["dpos"],
             "joint_vel": z["jvel"],
             "last_action": np.vstack([np.zeros((1, 29)), z["act"][:-1]])}
    for s, tr in truth.items():
        for c in ("p1", "npz", "isaaclab", "ctrl"):
            idx = [cloud_order.index(nm) for nm in cand_order(c)]
            pred = tr[:n][:, idx]
            got = np.stack([seg(local_obs[t], s, 3) for t in range(n)])
            cs = [corr(got[:, j], pred[:, j]) for j in range(29)]
            print(f"   {s:15s} 本地块序={c:9s} 相关(均值)={np.nanmean(cs):+.4f} "
                  f"Δmax={np.abs(got - pred).max():.5f}")

    # ═══ E. height_scan 结构分析 ═══════════════════════════════════════════
    print("\nE. height_scan 结构（云端有起伏、本地常数 ⇒ 射线网格/对齐语义不同）")
    sc = np.stack([seg(z["obs"][t], "height_scan", 3) for t in range(n)])
    lsc = np.stack([seg(local_obs[t], "height_scan", 3) for t in range(n)])
    print(f"   云端 scan: mean={sc.mean():.4f} std={sc.std():.4f} "
          f"范围=[{sc.min():.4f},{sc.max():.4f}]")
    print(f"   本地 scan: mean={lsc.mean():.4f} std={lsc.std():.4f} "
          f"范围=[{lsc.min():.4f},{lsc.max():.4f}]")
    qw, qx, qy, qz = z["rquat"][:n].T
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    for rs in ((31, 9), (9, 31)):
        A = sc.reshape(n, *rs)
        pat = A.mean(axis=0)
        xs = (np.arange(rs[0]) - (rs[0] - 1) / 2) * 0.1
        ys = (np.arange(rs[1]) - (rs[1] - 1) / 2) * 0.1
        print(f"   reshape{rs}: 行均值span={np.ptp(pat.mean(axis=1)):.4f} "
              f"列均值span={np.ptp(pat.mean(axis=0)):.4f} "
              f"行斜率={np.polyfit(xs, pat.mean(axis=1), 1)[0]:+.4f}(应≈pitch) "
              f"列斜率={np.polyfit(ys, pat.mean(axis=0), 1)[0]:+.4f}(应≈-roll)")
    print(f"   姿态范围: pitch=[{pitch.min():+.4f},{pitch.max():+.4f}] "
          f"roll=[{roll.min():+.4f},{roll.max():+.4f}] rad")
    print(f"   scan 首射线 vs pitch 相关={corr(sc[:, 0], pitch):+.4f}  "
          f"末射线 vs pitch 相关={corr(sc[:, -1], pitch):+.4f}  "
          f"scan 全量 vs pitch 相关={corr(sc.reshape(-1), np.repeat(pitch, 279)):+.4f}")

    # ═══ F/G. 策略对照 ═════════════════════════════════════════════════════
    import onnxruntime as ort
    zn = np.load(args.norm, allow_pickle=True)
    mean, std = zn["mean"].reshape(1, -1), zn["std"].reshape(1, -1)
    eps = float(np.asarray(zn["eps"]).reshape(-1)[0]) if "eps" in zn.files else 1e-2
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inm = sess.get_inputs()[0].name

    def run_onnx(obs_mat):
        xn = ((obs_mat - mean) / (std + eps)).astype(np.float32)
        return np.concatenate([np.asarray(sess.run(None, {inm: xn[i:i + 1]})[0]).reshape(1, -1)
                               for i in range(len(xn))], axis=0)

    a_cloud = z["act"][:n]
    d_ref = np.abs(run_onnx(z["obs"][:n]) - a_cloud)
    print(f"\nF. 同一观测下 云端 .pt vs 本地 .onnx：|Δ|max={d_ref.max():.3e} "
          + ("✅ 一致" if d_ref.max() < 1e-3 else "❌ 不一致"))
    d_loc = np.abs(run_onnx(local_obs) - a_cloud)
    print(f"G. 用**本地构造的观测**驱动本地 onnx vs 云端动作："
          f"|Δ|max={d_loc.max():.4f} rad  mean={d_loc.mean():.5f}")
    per_step = d_loc.max(axis=1)
    print("   逐步 |Δ|max（前 24 步）: " + " ".join(f"{v:.2f}" for v in per_step[:24]))
    worst = int(np.argmax(per_step))
    print(f"   最差步 {worst}: |Δ|max={per_step[worst]:.3f}，逐关节 Δ="
          f"{np.round(d_loc[worst], 3).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
