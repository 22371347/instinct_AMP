#!/usr/bin/env python3
"""**核实 AMP-Walk 参考动作是否"原地"** —— 直接读 `*_retargeted.npz` 的 `base_pos_w`。

为什么这一步是判定的关键
------------------------
判别器的输入（AMP 状态 396 维）里的速度通道**不是数据里存的**，而是
`AmpWalkMotionCfg.velocity_estimation_method = "frontbackward"` 从**位置**有限差分估出来的
（转换脚本只输出 `joint_pos / base_pos_w / base_quat_w`，**没有** base 速度字段）。

而已实测：专家集里 anchor 的线速度 σ = 0.017（y 通道 σ 恒为 0）⇒ 若源数据的
`base_pos_w` 本来就不动，那么"专家不动"是数据天性（判别器没错，但风格项因此**无法
奖励前进**、反而在惩罚根位移）；若 `base_pos_w` 在动，则是**加载/差分链路**丢了位移。

判读口径
--------
`Δx = max(x) - min(x)` 是**行程范围**（一个循环 clip 的前后摆幅）。
* Δx ≈ 0（几厘米内）⇒ **原地踏步**（root 被烘焙成不动，位移靠引擎/指令给）；
* Δx 与步长/腿长同量级（数十厘米以上）⇒ **真实位移**。
⚠️ 循环 clip 的**净位移** `x[-1]-x[0]` 接近 0 是正常的（一个周期回到起点），
   所以要看 **Δx（行程）** 而不是净位移。

用法（云端，纯 numpy，不需要 Isaac）：
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_check_motion_root.py [motions 目录]
"""

from __future__ import annotations

import glob
import os
import statistics
import sys

import numpy as np

DEFAULT = (
    "/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab/tasks/locomotion"
    "/config/g1/motions/amp_walk"
)


def main() -> int:
    d = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    files = sorted(glob.glob(os.path.join(d, "*_retargeted.npz")))
    if not files:
        print(f"[FATAL] {d} 下没有 *_retargeted.npz")
        return 2
    print(f"# {len(files)} 个 clip @ {d}")
    print(f"  {'clip':44s} {'T':>5s} {'fps':>4s} {'Δx':>8s} {'Δy':>8s} {'Δz':>8s} "
          f"{'净Δx':>8s} {'|v|仿射':>9s}")
    rows = []
    for f in files:
        z = np.load(f)
        keys = set(z.files)
        T = int(z["joint_pos"].shape[0]) if "joint_pos" in keys else -1
        fps = float(z["framerate"]) if "framerate" in keys else float("nan")
        if "base_pos_w" not in keys:
            print(f"  {os.path.basename(f):44s}  缺 base_pos_w（键={sorted(keys)}）")
            continue
        bp = np.asarray(z["base_pos_w"], dtype=np.float64)
        d_xyz = [float(bp[:, i].max() - bp[:, i].min()) for i in range(3)]
        net_x = float(bp[-1, 0] - bp[0, 0])
        # 近似"平均速度"：用**轨迹总长度**除以时长（对循环/往返 clip 更合理）
        if T > 1 and fps > 0:
            path_len = float(np.abs(np.diff(bp[:, 0])).sum())
            v_abs = path_len / (T / fps)
        else:
            v_abs = float("nan")
        rows.append((os.path.basename(f).replace("_retargeted.npz", ""), T, fps,
                     d_xyz[0], d_xyz[1], d_xyz[2], net_x, v_abs))
    for r in rows:
        print(f"  {r[0]:44s} {r[1]:5d} {r[2]:4.0f} {r[3]:8.3f} {r[4]:8.3f} {r[5]:8.3f} "
              f"{r[6]:+8.3f} {r[7]:9.3f}")

    if rows:
        dx = [r[3] for r in rows]
        va = [r[7] for r in rows]
        print(f"\n# Δx（行程）  中位 {statistics.median(dx):.3f} m   最大 {max(dx):.3f} m   "
              f"最小 {min(dx):.3f} m")
        print(f"# |v|(轨迹法) 中位 {statistics.median(va):.3f} m/s  最大 {max(va):.3f} m/s")
        med = statistics.median(dx)
        if med < 0.05:
            print("\n判读：⛔ **源数据就是原地踏步**（Δx 中位 < 5 cm）⇒")
            print("      「专家不动」是数据天性 ⇒ 风格项结构上无法奖励前进、且在惩罚根位移。")
            print("      修法是**数据侧**（把根位移补回去，或改用带位移的数据集），")
            print("      而不是改奖励形状；也可先做「剔除根线速度维」的止血臂。")
        else:
            print("\n判读：✅ **源数据含真实位移**（Δx 中位 >= 5 cm）⇒")
            print("      「专家不动」是**加载/差分链路**丢了位移 ⇒ 应修")
            print("      `velocity_estimation_method`/`base_pos_w` 的传递，属于链路 bug。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
