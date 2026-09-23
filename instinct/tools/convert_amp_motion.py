#!/usr/bin/env python3
"""AMP_mjlab 行走动作 → instinctlab 参考动作格式转换

背景
----
`instinct_xu` 的 `AmassMotionCfg`（通用 npz 加载器）读 `*_retargeted.npz`，需要字段：

    framerate, joint_names, joint_pos (T,29), base_pos_w (T,3), base_quat_w (T,4)

而 AMP_mjlab 的 `WalkandRun/*.npz` 是 mjlab 格式：

    fps, joint_pos (T,29), joint_vel, body_pos_w (T,30,3), body_quat_w (T,30,4),
    body_lin_vel_w, body_ang_vel_w

本脚本把后者转成前者，并**内置验证**。

已实测确认的事实（本脚本的验证会再次确认）
------------------------------------------
1. **关节列序 == MJCF hinge 顺序**（FK 误差 0.000000 m）
2. **四元数已是 wxyz**（AMP_mjlab 的 csv_to_npz 内已做 `[:, [3,0,1,2]]` 转换）→ 无需翻转
3. `body_pos_w[:, 0]` 是 root，对应 MJCF 的 `pelvis`（npz[i] == MJCF body[i+1]）
4. `fps = 50`，与 instinctlab 的 `motion_target_framerate` 一致 → 无需重采样
5. 关节名带 `_joint` 后缀，与 IsaacLab 命名一致 → 加载器会**按名字**重排

设计要点
--------
- 输出 `joint_names` = MJCF hinge 顺序（**直通，不重排**），因为加载器会按名字对齐到
  `isaac_joint_names`，我们保证名字正确即可
- 输出 `base_pos_w` / `base_quat_w` = **`--base-body`（默认 torso_link）经 FK 算出的世界位姿**，
  而不是直通 pelvis。原因：instinctlab 的 `MotionReferenceManagerCfg.prim_path` 指向
  `Robot/torso_link`，base 应与实际 articulation root 一致。
  ⚠️ 若运行时发现参考与机器人有恒定偏移，改用 `--base-body pelvis` 再转一次即可。

用法
----
    conda activate instinct
    python tools/convert_amp_motion.py --verify-only          # 只做验证
    python tools/convert_amp_motion.py                        # 转换全部
    python tools/convert_amp_motion.py --clips a b c          # 只转指定 clip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # .../llp/instinct
DEFAULT_XML = Path("/home/leo/rsl_project/AMP_mjlab/deploy/xmls/g1.xml")
DEFAULT_SRC = Path("/home/leo/rsl_project/AMP_mjlab/src/assets/motions/g1/amp/WalkandRun")
DEFAULT_DST = (
    ROOT.parent
    / "instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/motions/amp_walk"
)


# ---------------------------------------------------------------------------
# 模型侧信息
# ---------------------------------------------------------------------------

class G1Model:
    """从 MJCF 取出做 FK 与命名所需的全部索引。"""

    def __init__(self, xml: Path):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.hinge_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(m.njnt)
            if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
        ]
        self.hinge_joint_ids = [
            j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
        ]
        self.hinge_qpos_adr = [int(m.jnt_qposadr[j]) for j in self.hinge_joint_ids]
        free = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        if len(free) != 1:
            raise SystemExit(f"期望恰好 1 个 free joint（浮动基），实际 {len(free)} 个")
        self.free_qpos_adr = int(m.jnt_qposadr[free[0]])

        self.n_robot_bodies = m.nbody - 1          # 去掉 world
        self.body_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, m.nbody)
        ]

    def body_id(self, name: str) -> int:
        bid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise SystemExit(f"MJCF 里找不到 body: {name}（可用: {', '.join(self.body_names[:8])} …）")
        return bid

    def fk(self, base_pos, base_quat, joint_pos) -> tuple[np.ndarray, np.ndarray]:
        """按给定 base + 关节角做正运动学，返回 (xpos[nbody], xquat[nbody])。"""
        d, mj = self.data, self.mujoco
        d.qpos[:] = 0.0
        d.qpos[self.free_qpos_adr : self.free_qpos_adr + 3] = base_pos
        d.qpos[self.free_qpos_adr + 3 : self.free_qpos_adr + 7] = base_quat
        for k, adr in enumerate(self.hinge_qpos_adr):
            d.qpos[adr] = joint_pos[k]
        mj.mj_kinematics(self.model, d)
        return d.xpos.copy(), d.xquat.copy()


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------

def verify_clip(g1: G1Model, path: Path, frames: tuple[int, ...] = (0, -1), tol: float = 2e-3) -> dict:
    """验证：关节列序假设 + 四元数约定 + body 对齐。"""
    z = np.load(path)
    need = {"fps", "joint_pos", "body_pos_w", "body_quat_w"}
    missing = need - set(z.files)
    if missing:
        return {"ok": False, "msg": f"缺少字段 {sorted(missing)}"}

    jp, bp, bq = z["joint_pos"], z["body_pos_w"], z["body_quat_w"]
    if jp.shape[1] != len(g1.hinge_names):
        return {"ok": False, "msg": f"关节数 {jp.shape[1]} != MJCF {len(g1.hinge_names)}"}
    if bp.shape[1] != g1.n_robot_bodies:
        return {"ok": False, "msg": f"body 数 {bp.shape[1]} != MJCF {g1.n_robot_bodies}"}

    idx = [f if f >= 0 else len(jp) + f for f in frames]
    max_err = 0.0
    for f in idx:
        xpos, _ = g1.fk(bp[f, 0], bq[f, 0], jp[f])
        max_err = max(max_err, float(np.abs(xpos[1:] - bp[f]).max()))

    # 四元数范数（wxyz 应为单位四元数）
    qn = float(np.abs(np.linalg.norm(bq[:, 0], axis=-1) - 1.0).max())

    return {
        "ok": max_err < tol,
        "frames": len(jp),
        "fps": float(np.asarray(z["fps"]).reshape(-1)[0]),
        "max_fk_err": max_err,
        "max_quat_norm_err": qn,
        "msg": "OK" if max_err < tol else "关节序不匹配",
    }


# ---------------------------------------------------------------------------
# 转换
# ---------------------------------------------------------------------------

def convert_clip(g1: G1Model, path: Path, dst: Path, base_body: str, tol: float = 2e-3) -> dict:
    z = np.load(path)
    jp, bp, bq = z["joint_pos"], z["body_pos_w"], z["body_quat_w"]
    fps = float(np.asarray(z["fps"]).reshape(-1)[0])

    bid = g1.body_id(base_body)
    n = len(jp)
    base_pos = np.zeros((n, 3), dtype=np.float32)
    base_quat = np.zeros((n, 4), dtype=np.float32)

    max_err = 0.0
    for f in range(n):
        xpos, xquat = g1.fk(bp[f, 0], bq[f, 0], jp[f])
        max_err = max(max_err, float(np.abs(xpos[1:] - bp[f]).max()))
        base_pos[f] = xpos[bid]
        base_quat[f] = xquat[bid]

    if max_err >= tol:
        raise SystemExit(f"{path.name}: FK 校验失败（{max_err:.4f} m），拒绝输出")

    out_name = f"{path.stem}_retargeted.npz"
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst / out_name,
        framerate=np.float32(fps),
        joint_names=np.array(g1.hinge_names),          # MJCF/IsaacLab 命名，带 _joint 后缀
        joint_pos=jp.astype(np.float32),               # 列序 == joint_names（已验证）
        base_pos_w=base_pos,                           # FK 得到的 base_body 世界位置
        base_quat_w=base_quat,                         # wxyz，无需翻转
    )

    # 关节限位检查（宽松：仅报告越界幅度，不阻断）
    lo = g1.model.jnt_range[g1.hinge_joint_ids, 0]
    hi = g1.model.jnt_range[g1.hinge_joint_ids, 1]
    excess = float(max(0.0, (lo - jp).max(), (jp - hi).max()))

    return {
        "file": out_name,
        "frames": n,
        "fps": fps,
        "base_body": base_body,
        "max_fk_err": max_err,
        "limit_excess_rad": excess,
        "out_bytes": (dst / out_name).stat().st_size,
    }


def write_selection_yaml(dst: Path, files: list[str], path: Path) -> None:
    lines = ["selected_files:"]
    for f in files:
        lines.append(f"  - file: {f}")
        lines.append("    weight: 1.0")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 数据集统计（用于速度指令包络设计）
# ---------------------------------------------------------------------------

def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """wxyz -> 旋转矩阵（把本体系向量映到世界系）。"""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def body_frame_stats(path: Path) -> dict:
    """把轨迹换算到**本体系**，得到 vx/vy/wz 真实均值（单位 m/s、rad/s）。

    AMP_mjlab 的 clip 初始 yaw 是任意的（-100°~+170°），因此必须在每帧本体系下
    求速度，直接看世界系 vx/vy 会得出错误结论。
    """
    z = np.load(path)
    p, q = z["base_pos_w"], z["base_quat_w"]
    fps = float(z["framerate"])
    vf, vl, wz = [], [], []
    for t in range(len(p) - 1):
        v = _quat_to_mat(q[t]).T @ (p[t + 1] - p[t])     # 米/帧
        vf.append(v[0] * fps)                            # -> m/s
        vl.append(v[1] * fps)
        y0 = 2 * np.arctan2(q[t, 3], q[t, 0])
        y1 = 2 * np.arctan2(q[t + 1, 3], q[t + 1, 0])
        dy = (y1 - y0 + np.pi) % (2 * np.pi) - np.pi
        wz.append(dy * fps)                              # rad/s
    vx, vy, w = float(np.mean(vf)), float(np.mean(vl)), float(np.mean(wz))
    spd = float(np.hypot(vx, vy))
    return {
        "name": path.stem.replace("_retargeted", ""),
        "frames": len(p),
        "fps": fps,
        "height": float(p[:, 2].mean()),
        "vx": vx, "vy": vy, "wz": w, "speed": spd,
        "radius": (spd / abs(w)) if abs(w) > 0.05 else float("inf"),
    }


def print_stats(dst: Path) -> None:
    files = sorted(dst.glob("*_retargeted.npz"))
    if not files:
        raise SystemExit(f"{dst} 下没有 *_retargeted.npz")
    rows = [body_frame_stats(f) for f in files]
    print(f"{'clip':<34}{'高度m':>7}{'vx m/s':>9}{'vy m/s':>9}{'|v|':>7}{'wz°/s':>9}{'半径m':>8}")
    for r in rows:
        rad = f"{r['radius']:.2f}" if np.isfinite(r["radius"]) else " 直行"
        print(f"{r['name']:<34}{r['height']:>7.3f}{r['vx']:>9.3f}{r['vy']:>9.3f}"
              f"{r['speed']:>7.3f}{np.degrees(r['wz']):>9.1f}{rad:>8}")
    mv = np.array([[r["vx"], r["vy"], r["wz"]] for r in rows])
    print("\n--- 数据集体系速度包络（速度指令应落在其中，否则 AMP 无法提供风格）---")
    print(f"  vx : [{mv[:,0].min():+.3f}, {mv[:,0].max():+.3f}] m/s")
    print(f"  vy : [{mv[:,1].min():+.3f}, {mv[:,1].max():+.3f}] m/s")
    print(f"  wz : [{np.degrees(mv[:,2]).min():+.1f}, {np.degrees(mv[:,2]).max():+.1f}] °/s")
    print(f"  |v|: [{np.linalg.norm(mv[:,:2],axis=1).min():.3f}, "
          f"{np.linalg.norm(mv[:,:2],axis=1).max():.3f}] m/s")


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", default=str(DEFAULT_XML), help="G1 MJCF（用于 FK）")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="AMP_mjlab npz 源目录")
    ap.add_argument("--dst", default=str(DEFAULT_DST), help="输出目录")
    ap.add_argument("--clips", nargs="*", help="只处理这些 clip（不带 .npz）；缺省=全部")
    ap.add_argument("--base-body", default="torso_link", help="作为 base_pos_w/base_quat_w 的 body")
    ap.add_argument("--verify-only", action="store_true", help="只验证不输出")
    ap.add_argument("--stats", action="store_true", help="统计 --dst 里已转换动作的本体系速度")
    ap.add_argument("--no-yaml", action="store_true", help="不生成 selection yaml")
    args = ap.parse_args(argv)

    if args.stats:
        print_stats(Path(args.dst))
        return 0

    xml, src = Path(args.xml), Path(args.src)
    if not xml.is_file():
        raise SystemExit(f"找不到 MJCF: {xml}")
    if not src.is_dir():
        raise SystemExit(f"找不到源目录: {src}")

    files = sorted(src.glob("*.npz"))
    if args.clips:
        want = set(args.clips)
        files = [f for f in files if f.stem in want]
    if not files:
        raise SystemExit("没有可处理的 npz（检查 --src / --clips）")

    g1 = G1Model(xml)
    print(f"MJCF      : {xml}")
    print(f"  hinge   : {len(g1.hinge_names)} 个关节, robot bodies: {g1.n_robot_bodies}")
    print(f"  base    : {args.base_body}")
    print(f"源目录    : {src}  ({len(files)} 个 clip)")

    print("\n=== 验证 ===")
    bad = []
    for f in files:
        r = verify_clip(g1, f)
        tag = "✅" if r["ok"] else "❌"
        print(f"  {tag} {f.stem:<34} frames={r.get('frames','?'):>4} "
              f"fps={r.get('fps','?')} fk_err={r.get('max_fk_err', float('nan')):.6f} m")
        if not r["ok"]:
            bad.append((f.stem, r["msg"]))
    if bad:
        print("\n❌ 验证失败，中止：")
        for n, m in bad:
            print(f"   {n}: {m}")
        return 1

    if args.verify_only:
        print("\n✅ 全部 clip 验证通过（仅验证模式，未输出）")
        return 0

    print("\n=== 转换 ===")
    dst = Path(args.dst)
    out_files = []
    for f in files:
        r = convert_clip(g1, f, dst, args.base_body)
        out_files.append(r["file"])
        print(f"  ✅ {r['file']:<42} {r['frames']:>4}帧  {r['out_bytes']/1024:>7.1f} KB")

    if not args.no_yaml:
        yml = dst / "amp_walk_motion_selection.yaml"
        write_selection_yaml(dst, out_files, yml)
        print(f"\n  selection yaml -> {yml}")

    print(f"\n✅ 完成：{len(out_files)} 个 clip -> {dst}")
    print("   （加载器会按 joint_names 自动重排到 isaac_joint_names，速度由 estimate_velocity 估计）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
