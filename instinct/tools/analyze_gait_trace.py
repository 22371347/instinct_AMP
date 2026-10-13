"""回放 trace 取证：**策略走路时到底哪个关节不对**（腰？膝？）+ 每一项在此轨迹上的一阶拉力。

    # 1) 先采集 trace（见 g1_mujoco_play.py --trace-csv）
    python instinct/tools/g1_mujoco_play.py \
      --policy instinct/eval/vt3_12500/actor.onnx \
      --cmd 0.8 0 0 --duration 14 --no-view --no-realtime \
      --trace-csv instinct/eval/trace_t3_12500_vx08.csv
    # 2) 再分析
    python3 instinct/tools/analyze_gait_trace.py --csv instinct/eval/trace_t3_12500_vx08.csv

它回答四个问题（全部只用**实测**数字，不靠肉眼看图）：

  ① 速度跟踪到底差多少（vx/vy 均值 + RMSE + 侧漂）；
  ② **腰 vs 膝**：policy 的 waist_pitch / knee 分布 vs 参考（参考腰 +0.201、膝 0.383）；
  ③ 躯干**僵不僵**：由四元数差分算本体系 ‖ω_xy‖，与参考的 0.220 rad/s 对照
     （"没人付钱就别乱动" ⇒ 躯干被锁死的直接证据）；
  ④ `track_base_height` 在这条**真实轨迹**上值多少分、梯度多大 ⇒
     一阶拉力 = ∂R/∂z × ∂z/∂θ_waist（后者由 MJCF FK 实测），
     再与"总奖励幅度"比 ⇒ 判定"该项能不能顶起腰的姿态"。

⚠️ 与训练用的 IsaacLab 存在**回放保真度**差异（摩擦/接触求解器/随机化），
   所以数值只用于**横向对比**（参考 vs 策略），不用于绝对判分。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "instinct/tools")
from joint_order import NPZ_ORDER  # type: ignore  # noqa: E402

BASE = (
    "instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1"
    "/motions/amp_walk/"
)
SEL = BASE + "amp_walk_v13_motion_selection.yaml"

#: 我们的 default 站姿（MJCF keyframe）—— 也是 `use_default_offset=True` 的动作原点
DEFAULT_WAIST_PITCH = 0.0
DEFAULT_KNEE = 0.669
#: T2 的 `track_base_height` 靶（= `default_root_state[:,2]`）与 T3 的新靶
TARGET_T2 = 0.82
TARGET_T3 = 0.8025
STD = 0.3
#: `track_default_pose_on_stand` 的参数（T2 cfg 值）
STAND_POS_SCALE, STAND_VEL_SCALE, STAND_GATE_TAU, STAND_WEIGHT = 4.0, 0.5, 0.25, -1.0
#: T2 末期 12 项奖励之和（reward/s，用来把"拉力"换算成"占总奖励的比例"）
T2_TERM_SUM = 4.2626

#: 关注关节（名字与 npz/IsaacLab 一致）
WATCH = [
    "waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint",
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_elbow_joint", "right_elbow_joint",
]


def quat_mul(a, b):
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack(
        [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
         w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2],
        axis=-1,
    )


def _quat_conj(q):
    w, x, y, z = q.T
    return np.stack([w, -x, -y, -z], axis=-1)


def omega_body(q, dt):
    """由四元数差分得**本体系**角速度 ω_b ≈ 2·vec(q_{t+1} ⊗ q_t⁻¹)/dt（取短弧）。"""
    dq = quat_mul(q[1:], _quat_conj(q[:-1]))
    sign = np.sign(dq[:, 0:1])
    sign[sign == 0] = 1.0
    return 2.0 * (dq * sign)[:, 1:] / dt


def quat_to_mat(q):
    """(w,x,y,z) → R（行=B 系轴在世界系的表示）。"""
    w, x, y, z = q.T
    return np.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
         2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
         2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        axis=-1,
    ).reshape(-1, 3, 3)


def torso_lean_deg(q):
    """躯干**世界系**倾角：返回 (前倾°, 左倾°)。

    `waist_pitch` 是**相对骨盆**的关节角；肉眼看到的"腰立直"是**世界系**姿态 ——
    两者只有在骨盆俯仰也确定时才等价。骨盆俯仰是自由关节，必须单独量。
    """
    R = quat_to_mat(q)
    up = R[:, :, 2]                 # 躯干 up 轴（世界系）
    fwd_h = R[:, :2, 0]             # 躯干 forward 轴的水平分量
    left_h = R[:, :2, 1]
    fn = np.linalg.norm(fwd_h, axis=1, keepdims=True)
    ln = np.linalg.norm(left_h, axis=1, keepdims=True)
    fwd_h = fwd_h / np.maximum(fn, 1e-9)
    left_h = left_h / np.maximum(ln, 1e-9)
    fwd_deg = np.degrees(np.arctan2(np.sum(up[:, :2] * fwd_h, axis=1), up[:, 2]))
    left_deg = np.degrees(np.arctan2(np.sum(up[:, :2] * left_h, axis=1), up[:, 2]))
    return fwd_deg, left_deg


def fk_dz_dtheta_waist(xml: str) -> float:
    """MJCF FK：腰前倾 1 rad 会让**躯干离地高度**变化多少（脚保持贴地）。"""
    try:
        import mujoco
    except ImportError:
        return -0.0383          # 兜底：analyze_default_pose_height.py 的实测值
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    q0 = (m.key_qpos[0].copy() if m.nkey > 0 else m.qpos0.copy())
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint")
    if jid < 0:
        return -0.0383
    adr = m.jnt_qposadr[jid]
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    out = []
    for th in (0.0, 0.1, 0.2, 0.3):
        d.qpos[:] = q0
        d.qpos[adr] = th
        mujoco.mj_forward(m, d)
        foot_z = float(d.site_xpos[sid, 2]) if sid >= 0 else 0.0
        out.append(-(foot_z - 0.0))      # 脚回到地面所需的 base 位移
    dz = np.array(out)
    dz = dz - dz[0]                      # 以 θ=0 为基准
    th = np.array([0.0, 0.1, 0.2, 0.3])
    # 最小二乘斜率
    return float(np.polyfit(th, dz, 1)[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="g1_mujoco_play.py --trace-csv 的产物")
    ap.add_argument("--cmd", type=float, default=0.8, help="该 trace 的前向指令（用于参考分档/门控）")
    ap.add_argument("--drop-s", type=float, default=2.0, help="丢掉开头 N 秒（起步瞬态）")
    ap.add_argument("--xml", default="instinct/assets/g1_29dof_popsicle_play.xml")
    args = ap.parse_args()

    tr = np.genfromtxt(args.csv, delimiter=",", names=True)
    if tr.size == 0:
        raise SystemExit("trace 为空")
    dt = float(np.median(np.diff(np.atleast_1d(tr["t"]))))
    fps = 1.0 / dt
    keep = np.atleast_1d(tr["t"]) >= args.drop_s
    n_all, n_keep = tr.size, int(keep.sum())
    print(f"=== trace {Path(args.csv).name} ===")
    print(f"  {n_all} 步 @ {fps:.0f} Hz（{n_all * dt:.1f} s）；分析去掉前 {args.drop_s:.1f} s ⇒ 用 {n_keep} 步")

    cmd_vx = np.atleast_1d(tr["cmd_vx"])[keep]
    z = np.atleast_1d(tr["root_z"])[keep]
    x = np.atleast_1d(tr["root_x"])[keep]
    y = np.atleast_1d(tr["root_y"])[keep]
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)

    # ------------------------------------------------------------------ ①
    print("\n=== ① 速度跟踪（世界系，差分自 root 位移）===")
    m = float(np.mean(cmd_vx))
    print(f"  指令 vx = {m:+.3f} m/s")
    print(f"  实际 vx = {vx.mean():+.3f} m/s（效率 {100*vx.mean()/max(m,1e-9):.0f}%）"
          f"   vy = {vy.mean():+.3f} m/s（侧漂）")
    print(f"  RMSE   vx {np.sqrt(np.mean((vx-m)**2)):.3f}  vy {np.sqrt(np.mean(vy**2)):.3f} m/s")
    print(f"  躯干高度 z：p5 {np.percentile(z,5):.3f}  p50 {np.median(z):.3f}  "
          f"p95 {np.percentile(z,95):.3f}  （站姿 default 0.8025，参考中位 0.806）")

    # ------------------------------------------------------------------ ②
    ref_sel = yaml.safe_load(open(SEL))["selected_files"]
    ref_jp: dict[str, dict[str, np.ndarray]] = {"forward": {}, "all": {}}
    for s in ref_sel:
        d = np.load(BASE + s["file"])
        jp = np.asarray(d["joint_pos"], dtype=np.float64)
        is_fwd = "forward" in s["file"]
        for name in WATCH:
            if name not in NPZ_ORDER:
                continue
            col = jp[:, NPZ_ORDER.index(name)]
            ref_jp["all"].setdefault(name, []).append(col)
            if is_fwd:
                ref_jp["forward"].setdefault(name, []).append(col)
    ref_jp = {k: {n: np.concatenate(v) for n, v in d.items()} for k, d in ref_jp.items()}

    print("\n=== ② 腰 vs 膝（rad）===")
    print(f"  {'关节':30s} {'策略 p50':>9s} {'策略 p5':>8s} {'策略 p95':>9s} "
          f"{'参考fwd p50':>11s} {'参考all p50':>11s} {'default':>8s} {'Δ(fwd)':>8s}")
    for name in WATCH:
        col = f"j_{name}"
        if col not in tr.dtype.names:
            continue
        p = np.atleast_1d(tr[col])[keep]
        rf = ref_jp["forward"].get(name)
        ra = ref_jp["all"].get(name)
        if rf is None:
            continue
        rfm, ram = float(np.median(rf)), float(np.median(ra))
        dfault = DEFAULT_WAIST_PITCH if name == "waist_pitch_joint" else (
            DEFAULT_KNEE if name == "left_knee_joint" else float("nan"))
        print(f"  {name:30s} {np.median(p):9.3f} {np.percentile(p,5):8.3f} "
              f"{np.percentile(p,95):9.3f} {rfm:11.3f} {ram:11.3f} "
              f"{dfault:8.3f} {np.median(p)-rfm:+8.3f}")

    # ------------------------------------------------------------------ ②b
    q_pol = np.stack([np.atleast_1d(tr[f"q{c}"])[keep] for c in "wxyz"], axis=-1)
    print("\n=== ②b 躯干**世界系**倾角（肉眼看'立不立'看的是这个）===")
    fwd_p, left_p = torso_lean_deg(q_pol)
    ref_q = np.concatenate([np.asarray(np.load(BASE + s["file"])["base_quat_w"], np.float64)
                            for s in ref_sel])
    fwd_r, left_r = torso_lean_deg(ref_q)
    print(f"  前倾(+)/后倾(−)  策略 p50 {np.median(fwd_p):+6.2f}°  p5/p95 "
          f"{np.percentile(fwd_p,5):+6.2f}/{np.percentile(fwd_p,95):+6.2f}°   "
          f"参考 p50 {np.median(fwd_r):+6.2f}°")
    print(f"  左倾(+)/右倾(−)  策略 p50 {np.median(left_p):+6.2f}°  p5/p95 "
          f"{np.percentile(left_p,5):+6.2f}/{np.percentile(left_p,95):+6.2f}°   "
          f"参考 p50 {np.median(left_r):+6.2f}°")
    print("  ⚠️ 参考的 `base_quat_w` 若是 pelvis 而我们是 torso_link，此项只能看**量级**")

    # ------------------------------------------------------------------ ③
    w_pol = omega_body(q_pol, dt)
    wxy_p = np.linalg.norm(w_pol[:, :2], axis=1)
    ref_w = []
    for s in ref_sel:
        d = np.load(BASE + s["file"])
        ref_w.append(omega_body(np.asarray(d["base_quat_w"], np.float64), 1.0 / float(d["framerate"])))
    wxy_r = np.linalg.norm(np.concatenate(ref_w)[:, :2], axis=1)
    print("\n=== ③ 躯干有多'僵'（本体系 ‖ω_xy‖，rad/s）===")
    print(f"  策略 p50 {np.median(wxy_p):.3f}  p95 {np.percentile(wxy_p,95):.3f}   "
          f"参考 p50 {np.median(wxy_r):.3f}  p95 {np.percentile(wxy_r,95):.3f}")
    print(f"  ⇒ 代入 `body_ang_vel_xy`(std 3.14)：策略 {np.exp(-np.square(wxy_p)/3.14**2).mean():.4f}"
          f"  参考 {np.exp(-np.square(wxy_r)/3.14**2).mean():.4f}")

    # ------------------------------------------------------------------ ④
    dz_dth = fk_dz_dtheta_waist(args.xml)
    print("\n=== ④ `track_base_height` 在**这条真实轨迹**上的账 ===")
    for tgt, tag in ((TARGET_T2, "T2 靶 0.82"), (TARGET_T3, "T3 靶 0.8025")):
        e = tgt - z
        r = np.exp(-np.square(e) / STD**2)
        grad_z = 2.0 * e / STD**2 * r            # ∂r/∂z（reward per meter）
        print(f"  {tag:16s}: 得分 p50 {np.median(r):.4f} mean {r.mean():.4f}  "
              f"平均误差 {e.mean():+.4f} m  |e| p50 {np.median(np.abs(e)):.4f}")
        print(f"  {'':16s}  ∂r/∂z p50 = {np.median(grad_z):+.3f} /m"
              f"   ⇒ 权重 1.5 时 {1.5*np.median(grad_z):+.3f}/m（1 cm = {1.5*np.median(grad_z)*0.01:+.5f}/s）")
    print(f"  MJCF FK 实测 ∂z/∂θ_waist = {dz_dth:+.4f} m/rad"
          f"（前倾 ⇒ 躯干变矮，所以本项**反对**前倾）")
    e_t2 = TARGET_T2 - z
    g_waist = (1.5 * 2.0 * e_t2 / STD**2 * np.exp(-np.square(e_t2) / STD**2)).mean() * dz_dth
    print(f"  ⇒ 把腰前倾 0.2 rad 的代价 ≈ {abs(g_waist)*0.2:.5f} reward/s"
          f"（占总奖励幅度 {T2_TERM_SUM:.2f}/s 的 {abs(g_waist)*0.2/T2_TERM_SUM*100:.3f}%）")

    # ------------------------------------------------------------------ ⑤
    gate = float(np.exp(-abs(args.cmd) / STAND_GATE_TAU))
    print("\n=== ⑤ `track_default_pose_on_stand` 在 cmd={:.2f} 下的实际门控 ===".format(args.cmd))
    print(f"  gate = exp(−{args.cmd:.2f}/{STAND_GATE_TAU}) = {gate:.4f}（{100*gate:.1f}%）")
    dif = []
    for name in WATCH:
        col = f"j_{name}"
        if col not in tr.dtype.names:
            continue
        p = np.atleast_1d(tr[col])[keep]
        d0 = DEFAULT_WAIST_PITCH if name == "waist_pitch_joint" else (
            DEFAULT_KNEE if name == "left_knee_joint" else 0.0)
        dif.append(np.abs(p - d0))
    if dif:
        pos_cost = float(np.mean([np.mean(v) for v in dif]))
        print(f"  策略 29 关节 |q − q_default| 均值（上表关节）≈ {pos_cost:.3f} rad")
        print(f"  ⇒ 该项量级 ≈ |w|·gate·pos_scale·|q−q_d| ≈ "
              f"{abs(STAND_WEIGHT)*gate*STAND_POS_SCALE*pos_cost:.5f}/s（含 deadband 会更小）")
    print(f"  ⇒ 在 {args.cmd:.2f} m/s 行走时，'把关节拉回 default' 的力量只剩 {100*gate:.1f}%")

    print("\n=== 判读要点 ===")
    print("  * 腰：策略 p50 ≈ 0 而参考 +0.201 ⇒ 前倾**没有被奖励**（不是被某项顶直）")
    print("  * 膝：若策略 p50 明显 > 0.383 ⇒ 参考是'屈膝+前倾'的走法，我们退化成'直立腿'")
    print("  * ④⑤ 两项合起来若 ≪ 风格项量级 ⇒ 姿态问题是'没人付钱'，**不是** height 项")
    return 0


if __name__ == "__main__":
    sys.exit(main())
