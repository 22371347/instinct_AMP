#!/usr/bin/env python3
"""分析 AMP 参考动作数据集，核验「每个 clip 到底是什么运动」以及**是否覆盖指令范围**。

为什么需要它：AMP 风格先验只能教数据集里有的东西。如果指令范围（例如 vx 到 +1.2、
wz 到 ±0.8）超出数据集实际覆盖的速度，风格奖励就会和跟踪奖励**正面对撞**。
所以训练前必须用**数据**（不是文件名）确认覆盖面。

判据全部来自 clip 自身：把世界系速度投到**该帧的 yaw 坐标系**，
得到「前后 / 左右 / 偏航」三个分量。

用法：
    python instinct/tools/analyze_amp_motions.py
    python instinct/tools/analyze_amp_motions.py --dir <motions dir> --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

DEFAULT_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/motions/amp_walk"
)

#: 训练用的指令范围（= `AMP_WALK_COMMAND_RANGES`，`amp_walk_env_cfg.py:92`）
CMD_RANGES = ((-1.0, 1.2), (-0.5, 0.5), (-0.8, 0.8))


def yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """四元数 (w,x,y,z) → yaw（绕世界 z）。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def clip_stats(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    fps = float(np.asarray(d["framerate"]).reshape(-1)[0])
    pos = np.asarray(d["base_pos_w"], dtype=np.float64)
    quat = np.asarray(d["base_quat_w"], dtype=np.float64)
    t = pos.shape[0]
    dt = 1.0 / fps

    # 中心差分求世界系速度（首末用单侧差分）
    vel_w = np.empty_like(pos)
    vel_w[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
    vel_w[0] = (pos[1] - pos[0]) / dt
    vel_w[-1] = (pos[-1] - pos[-2]) / dt

    yaw = yaw_from_quat_wxyz(quat)
    # 世界 → 该帧 yaw 坐标系（只转 yaw，忽略 roll/pitch）
    c, s = np.cos(-yaw), np.sin(-yaw)
    vx_fwd = c * vel_w[:, 0] - s * vel_w[:, 1]     # 前后（+ 为前）
    vy_lat = s * vel_w[:, 0] + c * vel_w[:, 1]     # 左右（+ 为左）

    yaw_unwrap = np.unwrap(yaw)
    wx = np.empty_like(yaw_unwrap)
    wx[1:-1] = (yaw_unwrap[2:] - yaw_unwrap[:-2]) / (2.0 * dt)
    wx[0] = (yaw_unwrap[1] - yaw_unwrap[0]) / dt
    wx[-1] = (yaw_unwrap[-1] - yaw_unwrap[-2]) / dt

    def q(a: np.ndarray, p: float) -> float:
        return float(np.percentile(np.abs(a), p))

    return {
        "name": path.name.replace("_retargeted.npz", ""),
        "frames": t,
        "fps": fps,
        "dur_s": (t - 1) / fps,
        "vx_med": float(np.median(vx_fwd)),
        "vx_lo": float(np.percentile(vx_fwd, 5)),
        "vx_hi": float(np.percentile(vx_fwd, 95)),
        "vy_med": float(np.median(vy_lat)),
        "vy_abs_p90": q(vy_lat, 90),
        "wz_med": float(np.median(wx)),
        "wz_abs_p90": q(wx, 90),
    }


def classify(s: dict) -> str:
    """按实测分量给 clip 贴一个**数据驱动**的标签（与文件名互相印证）。"""
    tags = []
    vx, vy, wz = s["vx_med"], s["vy_med"], s["wz_med"]
    if vx > 0.25:
        tags.append("前向")
    elif vx < -0.25:
        tags.append("后向")
    if abs(vy) > 0.25 and abs(vy) > abs(vx):
        tags.append("侧向")
    if abs(wz) > 0.25:
        tags.append("转向")
    elif abs(vx) > 0.25 and abs(wz) > 0.08:
        tags.append("弧线")
    if abs(vx) < 0.25 and abs(vy) < 0.25 and abs(wz) < 0.25:
        tags.append("原地/微动")
    if s["vx_med"] > 0 and s["vx_med"] > 0.9:
        tags.append("慢跑档")
    return "+".join(tags) if tags else "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--csv", help="可选：把逐 clip 结果写成 csv")
    args = ap.parse_args()

    root = Path(args.dir)
    files = sorted(root.glob("*_retargeted.npz"))
    if not files:
        print(f"❌ {root} 下没有 *_retargeted.npz")
        return 2

    rows = [clip_stats(f) for f in files]
    rows.sort(key=lambda r: r["name"])

    print(f"数据集目录: {root}")
    print(f"clip 数量: {len(rows)}    总时长: {sum(r['dur_s'] for r in rows):.1f} s\n")
    print(f"{'clip':<38}{'帧':>5}{'fps':>6}{'时长':>7}"
          f"{'vx中位':>9}{'vx[5,95]':>16}{'vy中位':>9}{'|vy|p90':>9}{'|wz|p90':>9}  实测分类")
    print("-" * 130)
    for r in rows:
        vx_band = f"[{r['vx_lo']:.2f},{r['vx_hi']:.2f}]"
        print(
            f"{r['name']:<38}{r['frames']:>5}{r['fps']:>6.1f}{r['dur_s']:>7.2f}"
            f"{r['vx_med']:>9.3f}{vx_band:>16}"
            f"{r['vy_med']:>9.3f}{r['vy_abs_p90']:>9.3f}{r['wz_abs_p90']:>9.3f}  {classify(r)}"
        )

    # ---- 覆盖面核对：数据集实际到得了哪里 vs 指令范围 ----
    print("\n" + "=" * 130)
    print("覆盖面核对（数据集的实测极值 vs 训练指令范围）")
    print("=" * 130)
    vx_lo = min(r["vx_lo"] for r in rows)
    vx_hi = max(r["vx_hi"] for r in rows)
    vy_max = max(r["vy_abs_p90"] for r in rows)
    wz_max = max(r["wz_abs_p90"] for r in rows)

    def verdict(lo: float, hi: float, cmd_lo: float, cmd_hi: float) -> str:
        out = []
        if lo > cmd_lo + 1e-6:
            out.append(f"⚠️ 后退方向不足（数据 {lo:+.2f} > 指令下限 {cmd_lo:+.2f}）")
        if hi < cmd_hi - 1e-6:
            out.append(f"⚠️ 前进方向不足（数据 {hi:+.2f} < 指令上限 {cmd_hi:+.2f}）")
        return " / ".join(out) if out else "✅ 覆盖"

    print(f"vx（前后）: 数据实测 [{vx_lo:+.2f}, {vx_hi:+.2f}]   指令 {list(CMD_RANGES[0])}"
          f"   {verdict(vx_lo, vx_hi, *CMD_RANGES[0])}")
    print(f"vy（左右）: 数据实测 ±{vy_max:.2f}（p90）        指令 ±{CMD_RANGES[1][1]:.2f}"
          f"   {'✅ 覆盖' if vy_max >= CMD_RANGES[1][1] - 1e-6 else '⚠️ 侧向速度不足'}")
    print(f"wz（偏航）: 数据实测 ±{wz_max:.2f}（p90）        指令 ±{CMD_RANGES[2][1]:.2f}"
          f"   {'✅ 覆盖' if wz_max >= CMD_RANGES[2][1] - 1e-6 else '⚠️ 转向速率不足'}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + ["category"])
            w.writeheader()
            for r in rows:
                w.writerow({**r, "category": classify(r)})
        print(f"\n已写出 {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
