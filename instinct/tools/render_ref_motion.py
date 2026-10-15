"""把 **AMP 参考动作数据集**（`*_retargeted.npz`）回放成 **mp4** + **关节曲线 PNG**。

    # 1) 只出视频（默认 = 前向 clip，循环 3 遍便于观看）
    python3 instinct/tools/render_ref_motion.py --clip forward --repeat 3

    # 2) 同时出"手臂四宫格"曲线图，并**叠加策略 trace**（同一张图看参考 vs 策略）
    python3 instinct/tools/render_ref_motion.py --clip forward --repeat 3 \
        --plot-out instinct/eval/ref_walk_forward_arms.png \
        --trace-csv /tmp/t5_0.5.csv

为什么要有这个工具
------------------
参考动作一直只是"数字"（`waist_pitch p50 +0.201` 之类），而**视觉对比**是最快的沟通方式；
而且它用的是**和我们训练/回放完全相同的那台机器人资产**（`g1_29dof_popsicle_play.xml`，
free joint 挂在 `torso_link`）⇒ 视频里看到的姿势差异**不是资产差异**造成的。

⚠️ 参考 npz 的 `base_*` 实测是**躯干**（`torso_link`）系（见 `analyze_gait_trace.py` 的标定
FK 测试）⇒ 直接把 `base_pos_w/base_quat_w` 写进 qpos 就是正确姿势，不需要换算。

⚠️ 关节按**名字**映射到 MJCF（`mujoco.mj_name2id`）⇒ 不依赖 npz 的关节顺序。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = (
    "instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1"
    "/motions/amp_walk/"
)
SEL = BASE + "amp_walk_v13_motion_selection.yaml"
XML = "instinct/assets/g1_29dof_popsicle_play.xml"

#: 曲线图里画的关节（手臂问题是当前主要疑点）
PLOT_JOINTS = (
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_elbow_joint",
    "right_elbow_joint",
)


def pick_clip(pattern: str) -> str:
    sel = yaml.safe_load(open(SEL))["selected_files"]
    hits = [s["file"] for s in sel if pattern in s["file"]]
    if not hits:
        raise SystemExit(f"在 {SEL} 里找不到含 {pattern!r} 的 clip；可选：{[s['file'] for s in sel]}")
    return hits[0]


def list_selection(sel_path: str) -> list[str]:
    """选型 yaml 里的全部 clip（按 yaml 顺序，**相对 yaml 自身目录**解析）。

    ⚠️ 早先的实现在调用处写 `BASE + f`（BASE = 固定的 `motions/amp_walk`），
    于是拿 `motions/amp_walk_pelvis/xxx.yaml` 当选择文件时，读到的仍然是
    **旧目录的 npz** —— 换根后会静默混用“新模型 + 旧数据”，结果全错。
    """
    import os

    sel = yaml.safe_load(open(sel_path))["selected_files"]
    d = os.path.dirname(os.path.abspath(sel_path))
    return [f if os.path.isabs(f := s["file"]) else os.path.join(d, f) for s in sel]


def _qmul(a, b):
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def _qconj(q):
    w, x, y, z = q.T
    return np.stack([w, -x, -y, -z], axis=-1)


def _qrot(q, v):
    """用四元数旋转向量 v（v 形状 (...,3)）。"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w2, x2, y2, z2 = w * w, x * x, y * y, z * z
    R = np.stack([
        w2 + x2 - y2 - z2, 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), w2 - x2 + y2 - z2, 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), w2 - x2 - y2 + z2,
    ], axis=-1).reshape(v.shape[:-1] + (3, 3))
    # ⚠️ 必须用省略号写法：`d_pos` 在单帧路径里是**不带 batch 维**的 (3,)，
    #    写死 "nij,nj->ni" 会报 "too many subscripts"。
    return np.einsum("...ij,...j->...i", R, v)


def torso_pose_from_pelvis(pelvis_pos, pelvis_quat, d_pos, d_quat):
    """已知 **pelvis** 的世界位姿 + 腰链相对位姿 `D`（pelvis 在 torso 系里）

    ⇒ 解出 torso（= 我们的 root）的世界位姿：

        P = T ∘ D   ⇒   T_quat = P_quat ⊗ D_quat⁻¹,  T_pos = P_pos − R(T_quat)·D_pos

    用途：参考 npz 的 `base_*` 到底挂在**躯干**还是**骨盆**一直没 100% 定死；
    用这个函数可以两种假设各渲一份、并把“脚掌离地间隙”打出来对比。
    """
    tq = _qmul(pelvis_quat, _qconj(d_quat))
    tp = pelvis_pos - _qrot(tq, d_pos)
    return tp, tq


def load_ref(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    z = np.load(path)
    jp = np.asarray(z["joint_pos"], dtype=np.float64)
    pos = np.asarray(z["base_pos_w"], dtype=np.float64)
    quat = np.asarray(z["base_quat_w"], dtype=np.float64)
    names = [str(x) for x in z["joint_names"]]
    fps = float(z["framerate"]) if "framerate" in z else 50.0
    return pos, quat, jp, names, fps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="forward", help="run 子串（默认 forward）")
    ap.add_argument("--npz", default=None, help="直接给 npz 路径（优先于 --clip）")
    ap.add_argument("--xml", default=XML)
    ap.add_argument("--out", default=None, help="mp4 路径（默认 instinct/eval/ref_<clip>.mp4）")
    ap.add_argument("--repeat", type=int, default=1, help="循环几遍（便于观看）")
    ap.add_argument("--stride", type=int, default=1, help="每 N 帧录一帧")
    ap.add_argument("--slowmo", type=float, default=1.0, help="慢放倍数（2.0 = 半速）")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--azimuth", type=float, default=145.0)
    ap.add_argument("--elevation", type=float, default=-18.0)
    ap.add_argument("--distance", type=float, default=3.0)
    ap.add_argument("--fixed-camera", action="store_true",
                    help="相机固定 lookat=[0,0,0.5]（与 g1_mujoco_play 的 viewer 一致）；"
                         "默认**跟踪躯干**，便于看原地/位移两种 clip")
    ap.add_argument("--no-align-yaw", dest="align_yaw", action="store_false", default=True,
                    help="默认把相机方位角加上机体的**中位偏航**（参考 clip 的 root 位移"
                         "不一定落在世界 +x，不对齐会看成'侧着走'）")
    ap.add_argument("--plot-out", default=None, help="额外输出关节曲线 PNG")
    ap.add_argument("--trace-csv", default=None, help="叠加策略 trace（g1_mujoco_play --trace-csv）")
    ap.add_argument("--base-body", choices=["torso", "pelvis"], default="torso",
                    help="npz 的 `base_*` 当成哪个体的世界位姿：torso（默认，旧结论）/ pelvis（对照）")
    ap.add_argument("--selection", default=None,
                    help="选型 yaml：批量渲染它列的**所有** clip（如 amp_walk_motion_selection.yaml=17 个、"
                         "amp_walk_v13_motion_selection.yaml=5 个）。给文件名即可（自动加上 motions/amp_walk/）")
    ap.add_argument("--outdir", default="instinct/eval/ref_clips", help="批量模式的输出目录")
    ap.add_argument("--flip-waist", dest="flip_waist", action="store_true",
                    help="把 waist_{pitch,roll,yaw} **取反**后再回放（对照实验：验证我们的 torsobase "
                         "URDF 把腰链父子顺序翻过来、导致这三个关节符号与标准 G1/参考数据相反）")
    ap.add_argument("--table", action="store_true", help="只打**定量表**（不渲染）：逐个 clip 的躯干/骨盆世界倾角、高度、脚隙、肩/肘/腰/膝")
    args = ap.parse_args()

    import mujoco
    from g1_mujoco_play import load_model  # 复用同一套"补棋盘地面"的加载逻辑

    # ---------------------------------------------------------------- clip 列表
    if args.npz:
        clips = [args.npz]
    elif args.selection:
        selp = args.selection if os.path.sep in args.selection else (BASE + args.selection)
        clips = list_selection(selp)   # ← 已在 list_selection 内按 yaml 目录解析成完整路径
        print(f"[sel] {os.path.basename(selp)}：{len(clips)} 个 clip")
    else:
        clips = [BASE + pick_clip(args.clip)]
    multi = len(clips) > 1

    model = load_model(args.xml, checkered_floor=True)
    data = mujoco.MjData(model)
    pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    torso_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    foot_sids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
                 for s in ("left_foot", "right_foot")]
    foot_sids = [s for s in foot_sids if s >= 0]
    #: ⚠️ 官方 rev1.0 的 MJCF **没有** `left_foot`/`right_foot` site ⇒ 退回用
    #: `*_ankle_roll_link` 的体心高度（同一套“脚底框”语义，只差一个常数偏移；
    #: 下面统一用“站姿基准”归零，所以两种来源的间隙口径一致）。
    foot_bids: list[int] = []
    if not foot_sids:
        foot_bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                     for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
        foot_bids = [b for b in foot_bids if b >= 0]
        if not foot_bids:
            raise SystemExit("模型里既没有 left_foot/right_foot site，也没有 ankle_roll link")

    def lowest_foot_z() -> float:
        """当前状态下“最低脚部参考点”的高度（site 优先，否则用踝体心）。"""
        if foot_sids:
            return float(min(data.site_xpos[s, 2] for s in foot_sids))
        return float(min(data.xpos[b, 2] for b in foot_bids))

    #: 站姿（keyframe）下脚参考点的离地偏移 ⇒ 把"支撑脚离地间隙"归零的基准
    data.qpos[:] = model.key_qpos[0].copy()
    mujoco.mj_forward(model, data)
    foot_off = lowest_foot_z()
    _src = "site" if foot_sids else "ankle_roll 体心"
    print(f"[ref] 站姿脚（{_src}）离地 = {100 * foot_off:+.2f} cm（间隙以此为基准归零）")

    # ---------------------------------------------------------------- 定量表模式
    if args.table:
        from analyze_gait_trace import torso_lean_deg  # 复用同一套"世界系倾角"

        JW = ("waist_pitch_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
              "left_elbow_joint", "right_elbow_joint", "left_knee_joint")
        print(f"\n{'clip':40s} {'waist°':>7s} {'torso倾°':>9s} {'pelvis倾°':>10s} "
              f"{'torsoZ':>7s} {'fitZ':>7s} {'脚隙p5':>8s} {'脚隙p50':>8s} "
              f"{'L肩p5/p50/p95':>20s} {'R肩p5/p50/p95':>20s} {'L肘p50':>7s} {'膝p50':>6s}")
        for c in clips:
            pos, quat, jp, names, fps = load_ref(c)
            n = len(jp)
            adr = np.asarray([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)]
                              for nm in names])
            tt, tp, zt, zp, cl = [], [], [], [], []
            for i in range(0, n, 4):
                p, q = pos[i], quat[i]
                if args.base_body == "pelvis":
                    data.qpos[0:3] = 0.0
                    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
                    data.qpos[adr] = jp[i]
                    mujoco.mj_forward(model, data)
                    p, q = torso_pose_from_pelvis(pos[i], quat[i],
                                                  data.xpos[pelvis_bid].copy(),
                                                  data.xquat[pelvis_bid].copy())
                data.qpos[0:3] = p
                data.qpos[3:7] = q
                data.qpos[adr] = jp[i]
                mujoco.mj_forward(model, data)
                # ⚠️ 胻干倾角必须读 **FK 后的 torso 刚体**姿态：
                #    换根（base=pelvis，模型 root=pelvis）后 `q` 与 pelvis 等价，
                #    再拿它当“躯干”会退化成“骨盆倾角”。
                tt.append(torso_lean_deg(np.asarray(data.xquat[torso_bid])[None, :])[0][0])
                tp.append(torso_lean_deg(np.asarray(data.xquat[pelvis_bid])[None, :])[0][0])
                zt.append(float(data.xpos[torso_bid, 2]))
                zp.append(float(data.xpos[pelvis_bid, 2]))
                cl.append(lowest_foot_z() - foot_off)
            g = lambda nm, q=None: np.percentile(jp[:, names.index(nm)], q if q else 50)  # noqa: E731
            js = lambda nm: "/".join(f"{v:+.0f}" for v in  # noqa: E731
                                     np.degrees(np.percentile(jp[:, names.index(nm)], [5, 50, 95])))
            print(f"{os.path.basename(c).replace('_retargeted.npz', ''):40s} "
                  f"{np.degrees(g('waist_pitch_joint')):+7.1f} "
                  f"{np.median(tt):+9.2f} {np.median(tp):+10.2f} "
                  f"{np.median(zt):7.3f} {np.median(zp):7.3f} "
                  f"{100*np.percentile(cl,5):+8.2f} {100*np.median(cl):+8.2f} "
                  f"{js('left_shoulder_pitch_joint'):>20s} {js('right_shoulder_pitch_joint'):>20s} "
                  f"{np.degrees(g('left_elbow_joint')):+7.1f} {np.degrees(g('left_knee_joint')):+6.1f}")
        print(f"\n（base={args.base_body}；角度单位 °；高度 m；脚隙 cm⇒越接近 0 越自洽）")
        return 0

    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.elevation, cam.distance = args.elevation, args.distance

    def render_clip(npz_path: str) -> str | None:
        pos, quat, jp, names, fps = load_ref(npz_path)
        if args.flip_waist:
            jp = jp.copy()
            for w in ("waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint"):
                if w in names:
                    jp[:, names.index(w)] *= -1.0
            print("[ref]   ⚠️ --flip-waist：waist_{pitch,roll,yaw} 已取反")
        n = len(jp)
        stem = os.path.basename(npz_path).replace("_retargeted.npz", "")
        print(f"\n[ref] {stem}：{n} 帧 @ {fps:.0f} Hz = {n / fps:.2f} s")
        dx, dy = float(np.ptp(pos[:, 0])), float(np.ptp(pos[:, 1]))
        print(f"[ref]   root 行程 Δx={dx:.3f} m  Δy={dy:.3f} m"
              + ("（原地踏步）" if max(dx, dy) < 0.15 else "（有位移）"))

        adr, missing = [], []
        for nm in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
            (adr.append(model.jnt_qposadr[jid]) if jid >= 0 else missing.append(nm))
        if missing:
            raise SystemExit(f"[ref] MJCF 里找不到这些关节：{missing}")
        adr = np.asarray(adr)

        # ---- base 体假设：pelvis ⇒ 先逐帧解出 torso（= 我们的 root）的世界位姿 ----
        base_pos, base_quat = pos.copy(), quat.copy()
        if args.base_body == "pelvis":
            for i in range(n):
                data.qpos[0:3] = 0.0
                data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
                data.qpos[adr] = jp[i]
                mujoco.mj_forward(model, data)
                tp, tq = torso_pose_from_pelvis(
                    pos[i], quat[i], data.xpos[pelvis_bid].copy(), data.xquat[pelvis_bid].copy())
                base_pos[i], base_quat[i] = tp, tq

        # 相机：按机体中位偏航对齐（参考 clip 的 root 位移方向不一定 = 世界 +x）
        _yaw = float(np.median(np.degrees(np.arctan2(
            2 * (base_quat[:, 0] * base_quat[:, 3] + base_quat[:, 1] * base_quat[:, 2]),
            1 - 2 * (base_quat[:, 2] ** 2 + base_quat[:, 3] ** 2)))))
        cam.azimuth = args.azimuth + (_yaw if args.align_yaw else 0.0)

        frames, clear = [], []
        for rep in range(max(1, args.repeat)):
            for i in range(0, n, max(1, args.stride)):
                data.qpos[0:3] = base_pos[i]
                data.qpos[3:7] = base_quat[i]
                data.qpos[adr] = jp[i]
                mujoco.mj_forward(model, data)
                clear.append(lowest_foot_z() - foot_off)
                if args.fixed_camera:
                    cam.lookat[:] = [0.0, 0.0, 0.5]
                else:
                    cam.lookat[:] = data.xpos[torso_bid] + np.array([0.0, 0.0, 0.05])
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render().copy())

        clear = np.asarray(clear)
        print(f"[ref]   ⭐ base={args.base_body}：支撑脚离地间隙 p5 {100*np.percentile(clear,5):+.2f} cm  "
              f"p50 {100*np.median(clear):+.2f} cm  min {100*clear.min():+.2f} cm"
              f"   ← 越接近 0 说明这个 base 假设越自洽")
        print(f"[ref]   yaw={_yaw:+.1f}°（相机方位已加此值）")

        if multi:
            out = os.path.join(args.outdir, f"{stem}__{args.base_body}.mp4")
        else:
            out = args.out or f"{args.outdir}/{stem}__{args.base_body}.mp4"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        eff_fps = max(1.0, fps / max(1e-9, args.slowmo))
        try:
            import imageio.v3 as iio

            iio.imwrite(out, np.asarray(frames), fps=eff_fps)
            print(f"[ref]   ✅ mp4：{out}（{len(frames)} 帧 @ {eff_fps:.0f} fps）")
        except Exception as e:  # noqa: BLE001 —— ffmpeg 缺失就退化成 GIF
            gif = os.path.splitext(out)[0] + ".gif"
            try:
                import imageio.v3 as iio

                iio.imwrite(gif, np.asarray(frames), fps=eff_fps, loop=0)
                print(f"[ref]   ⚠️ mp4 失败（{type(e).__name__}: {e}）⇒ GIF：{gif}")
            except Exception as e2:  # noqa: BLE001
                npy = os.path.splitext(out)[0] + ".npy"
                np.save(npy, np.asarray(frames))
                print(f"[ref]   ⚠️ GIF 也失败（{type(e2).__name__}）⇒ npy：{npy}")
        return (pos, quat, jp, names, fps)

    last = None
    for c in clips:
        last = render_clip(c)
    renderer.close()
    if last is None:
        return 0
    pos, quat, jp, names, fps = last
    n = len(jp)

    # ---------------- 关节曲线 PNG（可叠加策略 trace；仅单 clip 模式）----------------
    if args.plot_out and not multi:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t_ref = np.arange(n) / fps
        tr = None
        if args.trace_csv:
            tr = np.genfromtxt(args.trace_csv, delimiter=",", names=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
        fig.suptitle(
            f"Arm joints — reference ({os.path.basename(clips[-1])[:34]}) vs policy trace "
            f"{'(' + os.path.basename(args.trace_csv) + ')' if tr is not None else ''}", fontsize=12)
        for ax, nm in zip(axes.flat, PLOT_JOINTS):
            k = names.index(nm)
            v = np.degrees(jp[:, k])
            ax.plot(t_ref, v, lw=1.4, color="tab:blue", label=f"reference ({nm.split('_')[0]})")
            p5, p50, p95 = np.percentile(v, [5, 50, 95])
            ax.axhline(p50, color="tab:blue", ls="--", lw=1.0)
            ax.axhspan(p5, p95, color="tab:blue", alpha=0.12,
                       label=f"ref p5/p50/p95 = {p5:+.1f}/{p50:+.1f}/{p95:+.1f}°")
            if tr is not None and f"j_{nm}" in tr.dtype.names:
                vp = np.degrees(np.atleast_1d(tr[f"j_{nm}"]))
                tp = np.atleast_1d(tr["t"])
                ax.plot(tp, vp, lw=1.4, color="tab:orange", label="policy")
                q5, q50, q95 = np.percentile(vp, [5, 50, 95])
                ax.axhline(q50, color="tab:orange", ls="--", lw=1.0)
                ax.axhspan(q5, q95, color="tab:orange", alpha=0.10,
                           label=f"policy p5/p50/p95 = {q5:+.1f}/{q50:+.1f}/{q95:+.1f}°")
            ax.set_title(nm, fontsize=10)
            ax.set_xlabel("t (s)")
            ax.set_ylabel("deg")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(args.plot_out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[ref] ✅ 曲线图已写：{args.plot_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
