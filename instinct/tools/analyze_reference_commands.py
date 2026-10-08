#!/usr/bin/env python3
"""**参考动作的"指令支撑能力"核对**（纯 numpy，本地 CPU）：参考数据到底"能不能"走这些指令？

动机（2026-10-06，用户提出"左右移动效果差，先确认指令分布"）
--------------------------------------------------------------
指令是从配置里采样的（`prob_low/lin/side/back` + `ranges`），但**参考动作是否覆盖这些指令**
是另一件事。若某个档的指令范围超出参考动作的实际速度能力，策略就只能"想象"该怎么走
⇒ 该方向必然欠拟合，而且**加多少奖励权重都救不回来**（因为 style 项也在把策略往那个
不存在的模态上拉）。

做法
----
对每个 retargeted npz：用 `base_pos_w` 的一阶差分（× framerate）算基座世界速度，
再用 `base_quat_w` 旋到**机体坐标系**（wxyz 四元数）⇒ `vx_b`（前向）/ `vy_b`（**横向**）/
`|v_xy|`。横向分量就是"侧移能力"的直接读数。

用法
----
    conda activate instinct && python instinct/tools/analyze_reference_commands.py            # 用 v13 选择表（= 当前 5 clip）
    python instinct/tools/analyze_reference_commands.py --yaml <path> --detail
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
MOTION_DIR = (
    REPO
    / "instinct_xu"
    / "source"
    / "instinctlab"
    / "instinctlab"
    / "tasks"
    / "locomotion"
    / "config"
    / "g1"
    / "motions"
    / "amp_walk"
)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((q.shape[0], 3, 3))
    m[:, 0, 0] = 1 - 2 * (y * y + z * z); m[:, 0, 1] = 2 * (x * y - w * z); m[:, 0, 2] = 2 * (x * z + w * y)
    m[:, 1, 0] = 2 * (x * y + w * z); m[:, 1, 1] = 1 - 2 * (x * x + z * z); m[:, 1, 2] = 2 * (y * z - w * x)
    m[:, 2, 0] = 2 * (x * z - w * y); m[:, 2, 1] = 2 * (y * z + w * x); m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def clip_velocity(npz_path: Path) -> dict[str, float] | None:
    z = np.load(npz_path, allow_pickle=True)
    if "base_pos_w" not in z.files or "base_quat_w" not in z.files:
        return None
    fps = float(z["framerate"]) if "framerate" in z.files else 50.0
    pos, quat = z["base_pos_w"], z["base_quat_w"]
    v_w = np.diff(pos, axis=0) * fps                      # (T-1, 3) 世界系
    R = quat_to_mat(quat[:-1])                            # (T-1, 3, 3) 机体系→世界系
    v_b = np.einsum("nij,nj->ni", R.transpose(0, 2, 1), v_w)   # 旋到机体系
    vxy = np.linalg.norm(v_b[:, :2], axis=1)
    return dict(
        n=int(v_b.shape[0]),
        vx=v_b[:, 0].mean(), vy=v_b[:, 1].mean(),
        vx_abs=np.abs(v_b[:, 0]).mean(), vy_abs=np.abs(v_b[:, 1]).mean(),
        vy_abs_p95=float(np.quantile(np.abs(v_b[:, 1]), 0.95)),
        vxy=vxy.mean(), vxy_p95=float(np.quantile(vxy, 0.95)),
        # 横向占比：1.0 = 纯侧移，0.0 = 纯前后
        lat_frac=float(np.mean(np.abs(v_b[:, 1]) / np.maximum(vxy, 1e-6))),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default=str(MOTION_DIR / "amp_walk_v13_motion_selection.yaml"))
    ap.add_argument("--motions", default=str(MOTION_DIR))
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.yaml, encoding="utf-8"))
    files = [x["file"] for x in cfg["selected_files"]]
    print(f"选择表：{Path(args.yaml).name} ⇒ {len(files)} 个 clip")
    print(f"参考数据目录：{args.motions}\n")
    hdr = f"{'clip':<42}{'帧':>5}{'|vx_b|':>8}{'|vy_b|':>8}{'vy_p95':>8}{'|v_xy|':>8}{'横向占比':>9}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for f in files:
        p = Path(args.motions) / f
        st = clip_velocity(p) if p.is_file() else None
        if st is None:
            print(f"{f:<42}  （读不到）")
            continue
        rows.append((f, st))
        print(f"{f.replace('_retargeted.npz',''):<42}{st['n']:>5}{st['vx_abs']:>8.3f}"
              f"{st['vy_abs']:>8.3f}{st['vy_abs_p95']:>8.3f}{st['vxy']:>8.3f}{st['lat_frac']:>9.2f}")
    if rows:
        w = np.array([r[1]["n"] for r in rows], float)
        w /= w.sum()
        mix = {k: float(np.dot(w, [r[1][k] for r in rows])) for k in rows[0][1] if k != "n"}
        print("-" * len(hdr))
        print(f"{'（按帧数加权）':<42}{'':>5}{mix['vx_abs']:>8.3f}{mix['vy_abs']:>8.3f}"
              f"{mix['vy_abs_p95']:>8.3f}{mix['vxy']:>8.3f}{mix['lat_frac']:>9.2f}")
    print("\n对照（当前指令分布，来自 V18 的「方案 B」，全仓无后续覆盖）：")
    print("  前向 vx ∈ (−0.5, 1.0) m/s（其中 0.25 概率的直线档取 0.5~0.9）")
    print("  侧向 vy ∈ **(−0.5, 0.5)** m/s（纯侧向档占 **0.25**，且 ωz≡0）")
    print("  后退 vx ∈ (−0.5, −0.3)；静止 ‖v‖≤0.1（0.20）；斜向/转向余量 0.20")
    print("\n⇒ 若参考的 |vy_b|/p95 明显低于指令上限 0.5，说明**侧向指令范围超出参考能力**"
          "（策略只能「臆造」侧移）—— 这是「左右移动差」的一类根因。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
