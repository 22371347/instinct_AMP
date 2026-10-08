#!/usr/bin/env python3
"""**参考动作的躯干倾角基线**（纯 numpy，本地 CPU，零 GPU 成本）。

回答的问题
----------
策略在 V30 里躯干后仰 ~24°（`Step/tilt_deg_mean` 12.56°→23.86°，`grav_bx_mean`
−0.037→−0.366）。**这 24° 是"参考动作本来就这么站"，还是策略自己漂出来的？**
⇒ 直接把参考动作 npz 的 `base_quat_w` 换算成"躯干系重力"，算倾角分布对照。

口径（与训练侧 `Step/tilt_*` / `Episode_Reward/flat_orientation_l2` 严格同构）
------------------------------------------------------------------------
`tilt = atan2(‖g_xy‖, |g_z|)`，其中 `g = R^T·(0,0,-1)`（= `projected_gravity_b`）。
* 该量对**全局偏航/平移不敏感** ⇒ 不需要知道运动的世界朝向就能比较；
* `g_x` **有符号**：`g_x < 0` ⇒ 躯干顶朝 −x 方向倾 ⇒ **后仰**（G1 机体 x 向前）；
* `flat_orientation_l2 = Σ g_xy²`（策略侧再乘权重 −2），本脚本同时打印 `E[Σg_xy²]`
  以便和 TB 的 `Episode_Reward/flat_orientation_l2` 反推对照（注意 TB 那条有
  "按 max_episode_length_s 归一"的偏置，别做精确等式比较，只看量级）。

用法
----
    python instinct/tools/analyze_reference_posture.py            # 全部 clip
    python instinct/tools/analyze_reference_posture.py --yaml <path/to/selection.yaml>
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

MOTION_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "instinct_xu",
        "source",
        "instinctlab",
        "instinctlab",
        "tasks",
        "locomotion",
        "config",
        "g1",
        "motions",
        "amp_walk",
    )
)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(N,4) wxyz → (N,3,3) 旋转矩阵。"""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - w * z)
    m[:, 0, 2] = 2 * (x * z + w * y)
    m[:, 1, 0] = 2 * (x * y + w * z)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - w * x)
    m[:, 2, 0] = 2 * (x * z - w * y)
    m[:, 2, 1] = 2 * (y * z + w * x)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def tilt_stats(quat_w: np.ndarray) -> dict[str, float]:
    """→ 倾角均值/p95、Σg_xy² 均值、g_x 均值（符号 = 前后倾）、g_z 均值、帧数。"""
    R = quat_to_mat(quat_w)
    g = np.einsum("nij,j->ni", R.transpose(0, 2, 1), np.array([0.0, 0.0, -1.0]))
    tilt = np.degrees(np.arctan2(np.linalg.norm(g[:, :2], axis=1), np.abs(g[:, 2])))
    return {
        "n": int(quat_w.shape[0]),
        "tilt_mean": float(tilt.mean()),
        "tilt_p95": float(np.quantile(tilt, 0.95)),
        "gt20": float((tilt > 20).mean()),
        "gt30": float((tilt > 30).mean()),
        "sum_gxy2": float((g[:, 0] ** 2 + g[:, 1] ** 2).mean()),
        "gx": float(g[:, 0].mean()),
        "gz": float(g[:, 2].mean()),
    }


def read_selection(path: str) -> list[str]:
    """从 motion selection yaml 里粗读 clip 名（只认 `- xxx` / `xxx:` 这类行）。"""
    if not os.path.isfile(path):
        return []
    names = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^-?\s*([A-Za-z0-9_\.\-]+?)(?:_retargeted)?(?:\.npz)?\s*:?$", line)
        if m:
            names.append(m.group(1))
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=MOTION_DIR)
    ap.add_argument("--yaml", default=os.path.join(MOTION_DIR, "amp_walk_motion_selection.yaml"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*_retargeted.npz")))
    if not files:
        print(f"[ERROR] {args.dir} 下没有 *_retargeted.npz")
        return 1

    sel = read_selection(args.yaml)
    print(f"motion dir : {os.path.normpath(args.dir)}")
    print(f"selection  : {os.path.basename(args.yaml)} → {len(sel)} 行；样例 {sel[:6]}")
    print()
    hdr = (
        f"{'clip':<44} {'帧':>5} {'tilt_mean':>9} {'p95':>7} {'>20°':>6} {'>30°':>6} "
        f"{'E[Σg_xy²]':>9} {'g_x':>7} {'g_z':>7}  在用"
    )
    print(hdr)
    print("-" * len(hdr))

    agg_all, agg_sel = [], []
    for f in files:
        name = os.path.basename(f).replace("_retargeted.npz", "")
        d = np.load(f, allow_pickle=True)
        key = "base_quat_w" if "base_quat_w" in d.files else "link_quat_w"
        st = tilt_stats(d[key])
        used = any(s in name or name in s for s in sel) if sel else False
        (agg_sel if used else agg_all).append(st)
        print(
            f"{name:<44} {st['n']:>5} {st['tilt_mean']:>9.2f} {st['tilt_p95']:>7.2f} "
            f"{st['gt20']*100:>5.1f}% {st['gt30']*100:>5.1f}% "
            f"{st['sum_gxy2']:>9.4f} {st['gx']:>+7.3f} {st['gz']:>+7.3f}  {'✔' if used else ''}"
        )

    def agg(tag, rows):
        if not rows:
            print(f"\n[{tag}] 空")
            return
        w = np.array([r["n"] for r in rows], dtype=float)
        w = w / w.sum()
        mix = {k: float(np.dot(w, [r[k] for r in rows])) for k in rows[0] if k != "n"}
        print(
            f"\n[{tag}] {len(rows)} 个 clip，{int(sum(r['n'] for r in rows))} 帧 "
            f"⇒ tilt_mean={mix['tilt_mean']:.2f}°  p95={mix['tilt_p95']:.2f}°  "
            f">20°={mix['gt20']*100:.1f}%  >30°={mix['gt30']*100:.1f}%  "
            f"E[Σg_xy²]={mix['sum_gxy2']:.4f}  g_x={mix['gx']:+.3f}  g_z={mix['gz']:+.3f}"
        )

    agg("被选用 clip（按帧数加权）", agg_sel)
    if agg_all:
        agg("未选用 clip", agg_all)
    print(
        "\n对照（TB，尾段均值）：V27 tilt=12.56° p95=23.31° g_x=−0.037 ｜ "
        "V30 tilt=23.86° p95=41.31° g_x=−0.366"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
