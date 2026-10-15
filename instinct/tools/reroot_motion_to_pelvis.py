#!/usr/bin/env python3
"""把 AMP 行走参考动作 npz 的 base（root 位姿）从 **torso_link 换根到 pelvis**。

为什么需要
----------
我们的参考动作 npz（`*_retargeted.npz`）只有 4 个有效字段：

    framerate, joint_names(29), joint_pos(T,29), base_pos_w(T,3), base_quat_w(T,4)

其中 `base_*` 是 **torso_link 的世界位姿**（已取证：17 个 clip 的 FK 一致高度残差
`res_torso ≈ −0.02…+0.08 m`，而 `res_pelvis` 恒为正且 ≥10 个 clip 会要求
torso > 0.838 m 的物理上限 ⇒ 不可能）。

训练侧换根（`G1_29DOF_REV10_PELVIS_CFG`，官方 rev1.0，**pelvis 为根**）之后，
`MotionReferenceManager` 必须把 npz 的 base 当作**参考机器人根刚体**的位姿
（`prim_path` / `reference_prim_path` 都指向 `/Robot/pelvis`）
⇒ **npz 的 base 必须换成 pelvis 的世界位姿**，否则整个参考机器人会整体错位/错向。

怎么算（用官方 MJCF 做 FK，不做任何近似）
----------------------------------------
1. 设浮动基 = pelvis 在原点且姿态为单位四元数，把腰关节设成该帧的 θ，`mj_forward`
   后读到的 torso 位姿即 `T_torso←pelvis`（torso 在 pelvis 系下的位姿）；
2. 取逆得到 `T_pelvis←torso`；
3. `T_pelvis^world = T_torso^world ∘ T_pelvis←torso`。

⚠️ 因为官方模型（pelvis 根）与厂商约定一致，**不需要**任何关节符号取反：直接喂 npz 的
   关节角即可。本脚本自带**往返自检**（用 pelvis 位姿 + 关节角反算 torso，必须逐帧回到
   原始值）+ 脚底贴地统计（换根正确时最低脚底应≈ −0.2 cm 均匀贴地）。

用法
----
    conda activate instinct
    python tools/reroot_motion_to_pelvis.py                 # 转换 + 自检 + 打印统计
    python tools/reroot_motion_to_pelvis.py --dry-run       # 只自检不写文件
    python tools/reroot_motion_to_pelvis.py --src DIR --dst DIR
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # .../llp/instinct
DEFAULT_XML = ROOT / "assets/g1_29dof_rev10_play.xml"
DEFAULT_SRC = (
    ROOT.parent
    / "instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/motions/amp_walk"
)
DEFAULT_DST = DEFAULT_SRC.parent / "amp_walk_pelvis"


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """wxyz 四元数乘法（a ∘ b）。支持 (..., 4)。"""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用 wxyz 四元数旋转向量。"""
    return quat_mul(quat_mul(q, np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)), quat_conj(q))[
        ..., 1:
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--xml", default=str(DEFAULT_XML), help="官方 rev1.0 的**可回放** MJCF（含执行器/keyframe）")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="源 npz 目录（base = torso 位姿）")
    ap.add_argument("--dst", default=str(DEFAULT_DST), help="输出目录（base = pelvis 位姿）")
    ap.add_argument("--dry-run", action="store_true", help="只做自检与统计，不写文件")
    args = ap.parse_args(argv)

    import mujoco

    xml, src, dst = Path(args.xml), Path(args.src), Path(args.dst)
    if not xml.is_file():
        raise SystemExit(f"MJCF 不存在: {xml}")
    if not src.is_dir():
        raise SystemExit(f"源目录不存在: {src}")

    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    free_j = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    free_adr = int(m.jnt_qposadr[free_j])
    hinge = [
        (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j), int(m.jnt_qposadr[j]))
        for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    b_torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    b_pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if b_torso < 0 or b_pelvis < 0:
        raise SystemExit("MJCF 里找不到 torso_link / pelvis")
    print(f"MJCF       : {xml}")
    print(f"free joint : {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, free_j)} (qposadr={free_adr})")
    print(f"hinge 数   : {len(hinge)}")
    print(f"源/目标     : {src} → {dst}\n")

    files = sorted(src.glob("*_retargeted.npz"))
    if not files:
        raise SystemExit(f"{src} 下没有 *_retargeted.npz")
    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    print(f"{'clip':44s} {'帧':>5s} {'往返误差':>10s} {'骨盆z中位':>9s} {'最低脚底':>9s}")
    print("-" * 90)
    worst_rt, worst_clip = 0.0, ""
    all_clear = []
    for f in files:
        z = np.load(f)
        names = [str(n) for n in z["joint_names"]]
        col = {n: i for i, n in enumerate(names)}
        jp = np.asarray(z["joint_pos"], dtype=np.float64)
        bp = np.asarray(z["base_pos_w"], dtype=np.float64)
        bq = np.asarray(z["base_quat_w"], dtype=np.float64)
        T = len(jp)

        # --- ① T_pelvis←torso：pelvis 置于原点 + 单位姿态，腰关节 = θ，FK 读 torso ---
        pelvis_pos = np.empty((T, 3))
        pelvis_quat = np.empty((T, 4))
        rt_err = np.empty(T)
        lowest_foot = np.empty(T)
        for t in range(T):
            d.qpos[:] = m.qpos0
            d.qpos[free_adr : free_adr + 3] = 0.0
            d.qpos[free_adr + 3 : free_adr + 7] = (1.0, 0.0, 0.0, 0.0)
            for n, adr in hinge:
                d.qpos[adr] = jp[t, col[n]]
            mujoco.mj_forward(m, d)
            t_pos = d.xpos[b_torso].copy()
            t_quat = d.xquat[b_torso].copy()
            # pelvis 在 torso 系下 = inverse(T_torso←pelvis)
            pq = quat_conj(t_quat)
            pp = -quat_rot(pq, t_pos)
            # T_pelvis^world = T_torso^world ∘ T_pelvis←torso
            pelvis_pos[t] = bp[t] + quat_rot(bq[t], pp)
            pelvis_quat[t] = quat_mul(bq[t], pq)
            # --- ② 往返自检：用 pelvis 位姿 + θ 反算 torso，必须回到原值 ---
            d.qpos[free_adr : free_adr + 3] = pelvis_pos[t]
            d.qpos[free_adr + 3 : free_adr + 7] = pelvis_quat[t]
            mujoco.mj_forward(m, d)
            rt_err[t] = max(
                float(np.abs(d.xpos[b_torso] - bp[t]).max()),
                float(np.abs(d.xquat[b_torso] - bq[t]).max()),
            )
            lowest_foot[t] = min(
                float(d.xpos[b][2])
                for b in (
                    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"),
                    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"),
                )
            )

        all_clear.append(lowest_foot)
        stem = f.name.replace("_retargeted.npz", "")
        if rt_err.max() > worst_rt:
            worst_rt, worst_clip = float(rt_err.max()), stem
        print(
            f"{stem:44s} {T:5d} {rt_err.max():10.2e} "
            f"{np.median(pelvis_pos[:, 2]):9.4f} {np.median(lowest_foot):9.4f}"
        )

        if not args.dry_run:
            np.savez(
                dst / f.name,
                framerate=z["framerate"],
                joint_names=z["joint_names"],
                joint_pos=z["joint_pos"],
                base_pos_w=pelvis_pos.astype(z["base_pos_w"].dtype),
                base_quat_w=pelvis_quat.astype(z["base_quat_w"].dtype),
            )

    clear = np.concatenate(all_clear)
    print("-" * 90)
    print(f"往返自检最大误差 = {worst_rt:.2e}  ({worst_clip})   ← 应 ≈ 1e-7 量级（浮点噪声）")
    print(
        f"全部 {len(files)} 个 clip 的最低脚踝高度：p5={np.percentile(clear, 5):.4f} "
        f"中位={np.median(clear):.4f} p95={np.percentile(clear, 95):.4f} m"
    )
    print("  （脚踝体心距地面约 0.0368 m 是正常的“脚底框偏移”；**均匀、无系统性漂移**才是对的）")

    if not args.dry_run:
        for y in sorted(src.glob("*.yaml")):
            shutil.copy2(y, dst / y.name)
            print(f"已复制选择文件: {y.name}")
        print(f"\n✅ 已写出 {len(files)} 个换根 npz 到 {dst}")
    else:
        print("\n(--dry-run：未写文件)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
