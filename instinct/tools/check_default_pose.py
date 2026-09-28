#!/usr/bin/env python3
"""核对 `track_default_pose_on_stand` 的 default pose 是否给了**错误的引导**。

背景
----
`track_default_pose_on_stand`（`tasks/locomotion/mdp/rewards_amp.py`）的实现是：

    abs_err = |joint_pos - asset.data.default_joint_pos|
    cost    = w · (4.0 · mean(abs_err) + 0.5 · mean(|joint_vel|))
    w       = exp(-cmd_speed / 0.25)        # cmd=0 ⇒ w=1

⇒ **它的目标姿态就是资产的 `default_joint_pos`**，即
`G1_29DOF_TORSOBASE_POPSICLE_CFG.init_state.joint_pos`。

若这个 default 是**半蹲**（`knee=+0.669` 等），而参考动作数据里站立/低速时的膝角
远小于它，那么「速度=0」时这条奖励就会把腿**拽向一个数据里不存在的姿态**
⇒ 与 AMP 判别器互相拉扯 ⇒ 出现异常腿姿。

本脚本做三件事（**全部离线：只加载 MJCF 做正向运动学 + matplotlib 画图，
不启动仿真、不跑策略、不打开 viewer**）：
  1. 交叉校验：MJCF 的 qpos0（= `build_g1_mjcf.py` 从 cfg 生成）
     **是否等于** cfg 里的 `init_state.joint_pos`（若不等于，说明两处已经漂了）；
  2. 逐关节对照：default vs 7 个精选参考 clip 的实测分布（均值 / min / max /
     最静止 5% 帧的均值），并标出 **default 落在数据范围之外** 的关节；
  3. 出图：default 姿态 vs 参考均值姿态的骨架图 + 逐关节偏差条形图。

用法：
    python instinct/tools/check_default_pose.py [--out-dir instinct/eval/default_pose_check]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MJCF = REPO / "instinct/assets/g1_29dof_popsicle_play.xml"
MOTION_DIR = REPO / "instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/motions/amp_walk"
SELECTION = MOTION_DIR / "amp_walk_v3_motion_selection.yaml"

#: ⭐ 奖励真正用的 default pose —— 出处：
#: `instinct_xu/.../assets/unitree_g1.py` `G1_29DOF_TORSOBASE_POPSICLE_CFG.init_state.joint_pos`
#: （约 558-571 行）。**其余 22 个关节都是 0.0。**
CFG_DEFAULT_PATTERNS: tuple[tuple[str, float], ...] = (
    (".*_hip_pitch_joint", -0.312),
    (".*_knee_joint", 0.669),
    (".*_ankle_pitch_joint", -0.363),
    (".*_elbow_joint", 0.600),
    ("left_shoulder_roll_joint", 0.200),
    ("left_shoulder_pitch_joint", 0.200),
    ("right_shoulder_roll_joint", -0.200),
    ("right_shoulder_pitch_joint", 0.200),
)
CFG_BASE_HEIGHT = 0.82

#: 腿部关节（用户报"速度=0 时腿部姿态异常"，优先看这几个）
LEG_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
)


def cfg_default_pose(joint_names: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    """按 `init_state.joint_pos` 的通配规则展开成 29 维向量（默认 0.0）。

    ⚠️ 这些键是 **IsaacLab 的正则**（`.*_knee_joint`），**不是 glob**！
       用 `fnmatch` 会把开头的 `.` 当字面点、于是 `.*_hip_pitch_joint` 一个都匹配不上
       （实测：通配规则全失配，`default` 列表全 0）⇒ 必须用 `re.fullmatch`。
    """
    import re

    out = np.zeros(len(joint_names), dtype=np.float64)
    hit = {p: 0 for p, _ in CFG_DEFAULT_PATTERNS}
    for i, name in enumerate(joint_names):
        for pat, val in CFG_DEFAULT_PATTERNS:
            if re.fullmatch(pat, name):
                out[i] = val
                hit[pat] += 1
                break
    return out, hit


def load_clips() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """返回 (poses [n_frames, 29], speeds [n_frames], 关节名)。只读精选 7 clip。"""
    files = []
    for line in SELECTION.read_text().splitlines():
        s = line.strip()
        if s.startswith("- file:"):
            files.append(s.split(":", 1)[1].strip())
    assert files, f"没能从 {SELECTION} 解析出 clip 清单"

    poses, speeds, names = [], [], None
    for f in files:
        npz = np.load(MOTION_DIR / f)
        q = np.asarray(npz["joint_pos"], dtype=np.float64)  # (T, 29)
        fps = float(npz["framerate"])
        if names is None:
            names = [str(x) for x in npz["joint_names"]]
        else:
            assert [str(x) for x in npz["joint_names"]] == names, f"{f} 关节顺序不一致"
        dq = np.diff(q, axis=0) * fps
        sp = np.concatenate([np.linalg.norm(dq, axis=1), [np.linalg.norm(dq[-1])]])
        poses.append(q)
        speeds.append(sp)
        print(f"  {f:52s} frames={q.shape[0]:5d} fps={fps:.0f}")
    return np.concatenate(poses), np.concatenate(speeds), names


def fk_positions(qpos_full: np.ndarray):
    """只做正向运动学（mj_kinematics），不步进、不渲染场景。"""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(MJCF))
    d = mujoco.MjData(m)
    d.qpos[:] = qpos_full
    mujoco.mj_kinematics(m, d)
    return m, d


def build_qpos(m, joint_names: list[str], base_height: float, q: np.ndarray) -> np.ndarray:
    import mujoco

    qpos = np.zeros(m.nq)
    qpos[0:3] = [0.0, 0.0, base_height]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, val in zip(joint_names, q):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"MJCF 里找不到关节 {name}")
        qpos[m.jnt_qposadr[jid]] = val
    return qpos


def skeleton_axes(ax, m, d, title: str, color: str) -> None:
    """把 body 树画成骨架（parent→child 连线）。"""
    pos = d.xpos.copy()
    for b in range(1, m.nbody):
        p = m.body_parentid[b]
        if p <= 0:
            continue
        ax.plot([pos[p, 0], pos[b, 0]], [pos[p, 1], pos[b, 1]], [pos[p, 2], pos[b, 2]],
                "-", color=color, lw=2.0, alpha=0.85)
    ax.scatter(pos[1:, 0], pos[1:, 1], pos[1:, 2], s=14, c=color, depthshade=False)
    ax.set_title(title, fontsize=10)
    ax.set_box_aspect((1, 1, 1.6))
    lim = 0.9
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, 1.5)
    ax.set_xlabel("x (fwd)"); ax.set_ylabel("y (left)"); ax.set_zlabel("z (up)")


def try_render(qpos_full: np.ndarray, out_png: Path, cam_azimuth: float, label: str) -> bool:
    """尝试用 mujoco 离屏渲染出**逼真图像**（仍然只是渲染静态姿态，不步进）。失败返回 False。"""
    try:
        import mujoco

        m = mujoco.MjModel.from_xml_path(str(MJCF))
        d = mujoco.MjData(m)
        d.qpos[:] = qpos_full
        mujoco.mj_kinematics(m, d)
        r = mujoco.Renderer(m, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
        cam.distance, cam.azimuth, cam.elevation = 2.6, cam_azimuth, -8.0
        cam.lookat[:] = [0.0, 0.0, 0.75]
        r.update_scene(d, camera=cam)
        img = r.render()
        r.close()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4.2, 4.2))
        plt.imshow(img)
        plt.axis("off")
        plt.title(label, fontsize=10)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"  ✅ 离屏渲染成功: {out_png.name}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ 离屏渲染不可用（{type(e).__name__}: {str(e)[:80]}），退回 matplotlib 骨架图")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "instinct/eval/default_pose_check"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import mujoco

    m = mujoco.MjModel.from_xml_path(str(MJCF))
    # ⚠️ 不能写 `m.jnt_type[j] in (mjJNT_HINGE, mjJNT_SLIDE)`：numpy 整数与 mjtJoint
    #    枚举直接比较会静默失败（实测取到 0 个关节）⇒ 一律转成 int 再比。
    n_jnt_free = int(mujoco.mjtJoint.mjJNT_FREE)
    mjcf_joint_names = []
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if nm is None or int(m.jnt_type[j]) == n_jnt_free:
            continue
        mjcf_joint_names.append(nm)
    print(f"MJCF: {MJCF.name}  关节数={len(mjcf_joint_names)}  nq={m.nq}  njnt={m.njnt}")
    assert len(mjcf_joint_names) == 29, f"期望 29 个铰链关节，实得 {len(mjcf_joint_names)}"

    # ---------- 1. cfg default vs MJCF 默认姿态 的交叉校验 ----------
    # ⚠️ 必须比 **keyframe**（`key_qpos[0]`），不是 `qpos0`！
    #    `g1_mujoco_play.py` 的默认姿态取自 keyframe（有则用），而本 MJCF 的
    #    `qpos0` 那几个关节是 0.000（未被使用）⇒ 拿 qpos0 比会得到假的 "两处已漂"。
    print("\n=== [1] 交叉校验：cfg 的 init_state.joint_pos  vs  MJCF 的默认姿态（keyframe）===")
    cfg_def, hits = cfg_default_pose(mjcf_joint_names)
    if m.nkey > 0:
        base_qpos = m.key_qpos[0]
        src = f"keyframe 0 ({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, 0)})"
    else:
        base_qpos = m.qpos0
        src = "qpos0（无 keyframe）"
    mjcf_def = np.array([base_qpos[m.jnt_qposadr[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] for n in mjcf_joint_names])
    print(f"  MJCF nkey={m.nkey} ⇒ 回放基准取自 {src}")
    print("  通配规则命中次数:", {k: v for k, v in hits.items()})
    n_zero = sum(1 for v in cfg_def if v == 0.0)
    print(f"  规则合计命中 {sum(hits.values())} 个关节；其余 {n_zero} 个关节按规则为 0.0")
    dmax = float(np.max(np.abs(cfg_def - mjcf_def)))
    print(f"  max|cfg_default - MJCF基准| = {dmax:.6f}  ⇒ "
          f"{'✅ 一致（回放不会被默认姿态偏置）' if dmax < 1e-6 else '❌ 两处已漂！回放会系统性偏置'}")
    for n, a, b in zip(mjcf_joint_names, cfg_def, mjcf_def):
        if abs(a - b) > 1e-6:
            print(f"     ⚠️ {n}: cfg={a:+.4f} mjcf={b:+.4f}  Δ={a-b:+.4f}")

    # ---------- 2. 参考动作的实际分布 ----------
    print("\n=== [2] 读取精选参考 clip ===")
    poses, speeds, ref_names = load_clips()
    missing = [n for n in mjcf_joint_names if n not in ref_names]
    if missing:
        raise KeyError(f"MJCF 里有、npz 里没有的关节: {missing}\nnpz 关节名: {ref_names}")
    order = [ref_names.index(n) for n in mjcf_joint_names]
    poses = poses[:, order]
    n_slow = max(1, int(0.05 * len(poses)))
    slow_idx = np.argsort(speeds)[:n_slow]
    ref_mean = poses.mean(axis=0)
    ref_slow = poses[slow_idx].mean(axis=0)
    ref_min, ref_max = poses.min(axis=0), poses.max(axis=0)

    print(f"\n  总计 {poses.shape[0]} 帧；最静止 5% = {n_slow} 帧")
    print("\n=== [3] 逐关节对照（按 |default − 参考均值| 降序）===")
    print(f"  {'关节':36s} {'default':>8s} {'clip均值':>9s} {'最静帧':>9s} "
          f"{'clip min':>9s} {'clip max':>9s}  {'default在范围内?':>14s}")
    rows = []
    for i, n in enumerate(mjcf_joint_names):
        rows.append((abs(cfg_def[i] - ref_mean[i]), n, cfg_def[i], ref_mean[i], ref_slow[i],
                     ref_min[i], ref_max[i]))
    outside = []
    for _, n, dv, rm, rs, lo, hi in sorted(rows, reverse=True):
        ok = (lo - 1e-9) <= dv <= (hi + 1e-9)
        if not ok:
            outside.append(n)
        flag = "✅" if ok else "❌ 范围外"
        print(f"  {n:36s} {dv:+8.3f} {rm:+9.3f} {rs:+9.3f} {lo:+9.3f} {hi:+9.3f}  {flag:>14s}")

    print(f"\n  ⚠️ default 落在参考数据范围之外的关节（{len(outside)} 个）: {outside}")
    print("\n  ⭐ 腿部重点对照（用户报『速度=0 时腿姿异常』）:")
    for n in LEG_JOINTS:
        i = mjcf_joint_names.index(n)
        print(f"     {n:32s} default={cfg_def[i]:+.3f}  clip均值={ref_mean[i]:+.3f} "
              f" 最静帧={ref_slow[i]:+.3f}  Δ(default-均值)={cfg_def[i]-ref_mean[i]:+.3f}")

    # 写 CSV
    csv = out / "default_vs_reference.csv"
    with csv.open("w") as fh:
        fh.write("joint,default,clip_mean,clip_slowest5pct,clip_min,clip_max,delta_default_minus_mean,default_outside_range\n")
        for _, n, dv, rm, rs, lo, hi in sorted(rows, reverse=True):
            ok = (lo - 1e-9) <= dv <= (hi + 1e-9)
            fh.write(f"{n},{dv:.5f},{rm:.5f},{rs:.5f},{lo:.5f},{hi:.5f},{dv-rm:+.5f},{0 if ok else 1}\n")
    print(f"\n  CSV 已写: {csv}")

    # ---------- 4. 出图 ----------
    print("\n=== [4] 出图 ===")
    # matplotlib 缺中文字体会刷屏告警 ⇒ 统一换英文标题（终端输出用中文）
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"

    q_def = build_qpos(m, mjcf_joint_names, CFG_BASE_HEIGHT, cfg_def)
    q_ref = build_qpos(m, mjcf_joint_names, CFG_BASE_HEIGHT, ref_mean)

    got_render = try_render(q_def, out / "render_default.png", 90.0, "default pose (reward target)")
    try_render(q_ref, out / "render_reference_mean.png", 90.0, "reference clips: mean pose")

    # 骨架对比图（始终生成，不依赖 GL）
    _, d_def = fk_positions(q_def)
    _, d_ref = fk_positions(q_ref)
    fig = plt.figure(figsize=(11, 4.6))
    for k, (d, ttl, col) in enumerate((
        (d_def, "default pose  (track_default_pose_on_stand target)", "tab:red"),
        (d_ref, "reference clips: mean pose  (AMP source)", "tab:blue"),
    )):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        skeleton_axes(ax, m, d, ttl, col)
    ax = fig.add_subplot(1, 3, 3)
    order2 = np.argsort(-np.abs(cfg_def - ref_mean))[:12]
    ax.barh([mjcf_joint_names[i] for i in order2][::-1],
            (cfg_def - ref_mean)[order2][::-1], color="tab:orange")
    ax.axvline(0, c="k", lw=0.8)
    ax.set_title("default - ref_mean (rad)\nTop12", fontsize=10)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    png = out / "default_vs_reference.png"
    plt.savefig(png, dpi=150)
    plt.close()
    print(f"  ✅ 骨架/条形图: {png}")
    if not got_render:
        print("  （离屏渲染不可用，只有 matplotlib 骨架图）")

    print("\n" + "=" * 72)
    print(f"结论速览：default 有 {len(outside)}/29 个关节落在参考数据范围之外")
    print(f"  最大偏差关节 top3: " + ", ".join(
        f"{n}({dv - rm:+.3f})" for _, n, dv, rm, _, _, _ in sorted(rows, reverse=True)[:3]))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
