#!/usr/bin/env python3
"""**数据集回放**：把 retarget 好的运动 npz（base + 29 关节）在本地 MuJoCo 里
做**运动学回放**并导出视频，供人工核对数据集本身是否正确。

为什么需要它
------------
`motions/vault/g1_speed_vault_retargeted.npz` 只有 `base_pos_w / base_quat_w /
joint_pos`（29 关节）+ 障碍物参数，**没有** body 级数据；IsaacLab 侧的
`MotionReferenceManager` 会用 URDF 做 FK 补出所有 link。要看"数据集到底长什么样"，
最快的办法就是按帧把 qpos 写进去、直接渲染 —— 不跑物理、不依赖 Isaac Sim。

约定（已实测核对）
------------------
* `base_pos_w/base_quat_w` = **根 body（torso_link）的世界位姿**，四元数 (w,x,y,z)；
* npz 的 29 个关节**按名字**映射到 MJCF 的 qpos 序（两套顺序不同，必须按名重排）；
* `obstacle_pos/obstacle_size` = IsaacLab `MeshCuboidCfg` 的**全尺寸**；
  MuJoCo 的 box `size` 是**半尺寸** ⇒ 除以 2。

用法
----
    # 默认：popsicle 资产 + vault 数据集 → instinct/eval/vault_motion_replay.mp4
    python instinct/tools/render_vault_motion.py
    # 只要关键帧 PNG（最快，用来看姿势对不对）
    python instinct/tools/render_vault_motion.py --stills 6 --outdir /tmp/vault_stills
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]   # instinct/tools/render_vault_motion.py → 仓库根
DEFAULT_NPZ = (REPO / "instinct_xu/source/instinctlab/instinctlab/tasks/shadowing"
               / "beyondmimic/config/g1/motions/vault/g1_speed_vault_retargeted.npz")
DEFAULT_XML = REPO / "instinct/assets/g1_29dof_popsicle_play.xml"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=str(DEFAULT_NPZ))
    ap.add_argument("--xml", default=str(DEFAULT_XML))
    ap.add_argument("--out", default=str(REPO / "instinct/eval/vault_motion_replay.mp4"))
    ap.add_argument("--fps", type=float, default=None, help="默认用 npz 的 framerate")
    ap.add_argument("--every", type=int, default=1, help="每 N 帧取一帧（抽帧）")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--cam-distance", type=float, default=4.5)
    ap.add_argument("--cam-azimuth", type=float, default=125.0)
    ap.add_argument("--cam-elevation", type=float, default=-12.0)
    ap.add_argument("--follow", action="store_true", default=True,
                    help="相机跟随根节点（默认开；12 m 的助跑+翻越太宽）")
    ap.add_argument("--no-follow", dest="follow", action="store_false")
    ap.add_argument("--no-box", dest="box", action="store_false", default=True,
                    help="不画障碍箱")
    ap.add_argument("--base-body", choices=["pelvis", "torso"], default="pelvis",
                    help="base_pos_w 指的是哪个 body：pelvis（默认，实测正确：最低点恰为 0.000"
                         "m）或 torso（会让全模型穿地 0.027~0.037 m，仅用于对照）")
    ap.add_argument("--stills", type=int, default=0,
                    help=">0 时只导出 N 张关键帧 PNG（不写 mp4）")
    ap.add_argument("--outdir", default="/tmp/vault_stills")
    args = ap.parse_args(argv)

    import mujoco

    z = np.load(args.npz, allow_pickle=True)
    fr = float(z["framerate"]) if args.fps is None else float(args.fps)
    base_p = z["base_pos_w"].astype(np.float64)
    base_q = z["base_quat_w"].astype(np.float64)
    jpos = z["joint_pos"].astype(np.float64)
    npz_names = [str(s) for s in z["joint_names"]]
    n = len(base_p)
    box_pos = z["obstacle_pos"].astype(np.float64)
    box_size = z["obstacle_size"].astype(np.float64)
    box_top = float(box_pos[2] + box_size[2] / 2)

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    # qpos 序（跳过 free joint）→ 按名映射到 npz 的关节列
    qpos_joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                        for j in range(1, m.njnt)]
    assert set(qpos_joint_names) <= set(npz_names), "MJCF 关节名不在 npz 里"
    col = [npz_names.index(nm) for nm in qpos_joint_names]

    print(f"[npz] {Path(args.npz).name}: {n} 帧 @ {fr:g} fps = {n / fr:.2f} s")
    print(f"[box] pos={box_pos.tolist()} size={box_size.tolist()} ⇒ 顶面 z={box_top:.3f} m")
    print(f"[xml] {Path(args.xml).name}  根 body = "
          f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, 1)}")
    print(f"[map] 关节按名重排：qpos[0]={qpos_joint_names[0]} ← npz 列 {col[0]}")

    renderer_w = max(int(m.vis.global_.offwidth), args.width)
    renderer_h = max(int(m.vis.global_.offheight), args.height)
    m.vis.global_.offwidth, m.vis.global_.offheight = renderer_w, renderer_h
    renderer = mujoco.Renderer(m, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = args.cam_distance, args.cam_azimuth, args.cam_elevation

    def set_frame(i: int):
        d.qpos[:] = 0.0
        d.qpos[0:3] = base_p[i]
        d.qpos[3:7] = base_q[i]
        d.qpos[7:7 + 29] = jpos[i][col]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        if args.base_body == "pelvis":
            # ⭐ 数据集 base_pos_w 是 **pelvis** 的位姿（实测：这样脚才恰好吃地），
            #    而 MjCF 的根是 torso_link ⇒ 平移修正一下（纯平移，不动姿态）。
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            d.qpos[0:3] += base_p[i] - d.xpos[bid]
            mujoco.mj_forward(m, d)
        cam.lookat[:] = (d.xpos[1] if args.follow else np.array([0.0, 0.0, 0.6]))

    def render_frame():
        renderer.update_scene(d, camera=cam)
        if args.box:
            scn = renderer.scene
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX,
                                    np.asarray(box_size / 2.0), box_pos,
                                    np.eye(3).reshape(-1), np.array([0.35, 0.35, 0.38, 1.0]))
                scn.ngeom += 1
        return renderer.render()

    frames_idx = list(range(0, n, args.every))
    # 逐帧统计（人工核对用）
    zmin_log, x_log = [], []
    for i in range(n):
        set_frame(i)
        zmin_log.append(float(d.xpos[:, 2].min()))
        x_log.append(float(base_p[i, 0]))
    print(f"[stat] 根高度 {base_p[:,2].min():.3f}~{base_p[:,2].max():.3f} m"
          f"（箱顶 {box_top:.3f}）⇒ 抬升 {base_p[:,2].max()-base_p[:,2].min():.3f} m")
    print(f"[stat] 全模型最低点 {min(zmin_log):.3f} m"
          + ("（<0 说明有关节穿地 ⚠️）" if min(zmin_log) < -0.02 else "  ✅ 无穿地"))
    above = np.nonzero(base_p[:, 2] > box_top)[0]
    if len(above):
        print(f"[stat] 根高于箱顶的帧区间 = [{above[0]}, {above[-1]}]"
              f"（t={above[0]/fr:.2f}~{above[-1]/fr:.2f} s）")
    # 踝高度：翻越时应明显抬离地面
    ank = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
           for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
    ank_h = []
    for i in range(n):
        set_frame(i)
        ank_h.append(min(float(d.xpos[b, 2]) for b in ank))
    print(f"[stat] 踝最低高度 {min(ank_h):.3f}~{max(ank_h):.3f} m"
          f"（峰值在 t={ank_h.index(max(ank_h))/fr:.2f} s）")

    if args.stills:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        idxs = np.linspace(0, n - 1, args.stills).astype(int)
        from PIL import Image
        for k, i in enumerate(idxs):
            set_frame(int(i))
            Image.fromarray(render_frame()).save(outdir / f"still_{k:02d}_f{i:04d}.png")
        print(f"[still] 已导出 {len(idxs)} 张 -> {outdir}  帧号={idxs.tolist()}")
        return 0

    import imageio

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(out, fps=fr / args.every, codec="libx264", quality=7,
                           macro_block_size=1, ffmpeg_log_level="error")
    for i in frames_idx:
        set_frame(i)
        w.append_data(render_frame())
    w.close()
    print(f"[record] 已保存 {len(frames_idx)} 帧 -> {out}"
          f"（{len(frames_idx)/ (fr/args.every):.2f} s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
