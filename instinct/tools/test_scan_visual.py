#!/usr/bin/env python3
"""`height_scan` 扫描区域可视化的**离线几何单测**（不开窗、不跑策略）。

验证的是最容易算错的东西：那个橙色矩形框到底画在哪儿。
判据来自源码：`omni_loco_env_cfg.py:86`
    GridPatternCfg(resolution=0.1, size=(3.0, 1.6))  → 31x17 = 527
    prim_path = torso_link, ray_alignment = "yaw"

跑法：
    python instinct/tools/test_scan_visual.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import g1_mujoco_play as g  # noqa: E402

FAILS: list[str] = []
HALF_X = g.SCAN_SIZE_XY[0] / 2.0
HALF_Y = g.SCAN_SIZE_XY[1] / 2.0


def check(cond: bool, label: str, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(label)


def section(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(g.DEFAULT_XML))
    data = mujoco.MjData(model)
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, g.SCAN_TORSO_BODY)
    scn = mujoco.MjvScene(model, maxgeom=2000)

    def draw(pos, quat_wxyz) -> tuple[int, np.ndarray]:
        """返回 (线段数, 所有端点)。

        ⭐ 直接调 `scan_footprint_segments` 算几何，**不从 mjvGeom 反推**：
        MuJoCo 的 capsule 长轴是局部 **z** 轴（不是 x），且总长还要加两倍半径，
        反推很容易把测试自己写错 —— 那就不是在测几何了。
        """
        data.qpos[:3] = pos
        data.qpos[3:7] = quat_wxyz
        mujoco.mj_forward(model, data)
        segs = g.scan_footprint_segments(data, tid)
        pts = np.array([p for a, b, _ in segs for p in (a, b)])
        return len(segs), pts

    def yaw_quat(deg: float) -> np.ndarray:
        h = np.deg2rad(deg) / 2.0
        return np.array([np.cos(h), 0.0, 0.0, np.sin(h)])

    # ------------------------------------------------------------------ 1
    section("1. 射线数自洽：3.0/0.1+1 = 31, 1.6/0.1+1 = 17 → 527")
    check(g.SCAN_GRID_RAYS == 527, f"SCAN_GRID_RAYS == 527", f"got {g.SCAN_GRID_RAYS}")
    check(g.SCAN_ALIGNMENT == "yaw", "对齐方式 = yaw（与 ray_alignment 一致）")

    # ------------------------------------------------------------------ 2
    section("2. 原地不动、朝向 +x：矩形应是 x∈[-1.5,1.5], y∈[-0.8,0.8], 贴地")
    n, pts = draw([0, 0, 0.82], yaw_quat(0))
    check(n == 12, "画出 12 段（4 边框 + 5+2 内网格 + 1 前向标记）", f"got {n}")
    # 另一半：确认真的能填进 mjvScene（绘制路径本身也要能跑）
    scn.ngeom = 0
    filled = g.draw_scan_footprint(scn, data, tid)
    check(filled == 12 and scn.ngeom == 12, "draw_scan_footprint 已向 mjvScene 写入 12 个 geom",
          f"filled={filled} ngeom={scn.ngeom}")
    check(
        np.allclose([pts[:, 0].min(), pts[:, 0].max()], [-HALF_X, HALF_X + 0.3], atol=1e-6),
        f"x 范围 = [-{HALF_X}, {HALF_X}+0.3](前向标记)",
        f"got [{pts[:, 0].min():.3f}, {pts[:, 0].max():.3f}]",
    )
    check(
        np.allclose([pts[:, 1].min(), pts[:, 1].max()], [-HALF_Y, HALF_Y], atol=1e-6),
        f"y 范围 = ±{HALF_Y}",
        f"got [{pts[:, 1].min():.3f}, {pts[:, 1].max():.3f}]",
    )
    check(np.allclose(pts[:, 2], g.SCAN_DRAW_Z, atol=1e-9), f"z 全部 = {g.SCAN_DRAW_Z}（贴地）")

    # ------------------------------------------------------------------ 3
    section("3. 机器人平移 → 矩形整体跟随（且**不随身高变化**）")
    _, pts_a = draw([2.0, 3.0, 0.82], yaw_quat(0))
    _, pts_b = draw([2.0, 3.0, 1.30], yaw_quat(0))  # 只改高度
    cx = (pts_a[:, 0].min() + pts_a[:, 0].max() - 0.3) / 2.0
    cy = (pts_a[:, 1].min() + pts_a[:, 1].max()) / 2.0
    check(abs(cx - 2.0) < 1e-6 and abs(cy - 3.0) < 1e-6, "矩形中心跟到 (2,3)", f"got ({cx:.3f},{cy:.3f})")
    check(np.allclose(pts_a, pts_b), "躯干高度从 0.82 抬到 1.30 → 地面上的框**不变**")

    # ------------------------------------------------------------------ 4
    section("4. 转向 90°：前后轴应转到世界 y 轴（范围互换）")
    _, pts90 = draw([0, 0, 0.82], yaw_quat(90))
    ext_x = pts90[:, 0].max() - pts90[:, 0].min()
    ext_y = pts90[:, 1].max() - pts90[:, 1].min()
    check(abs(ext_x - (2 * HALF_Y)) < 1e-6, f"朝向 90° 后 x 跨度 = {2 * HALF_Y}（左右）", f"got {ext_x:.3f}")
    check(abs(ext_y - (2 * HALF_X + 0.3)) < 1e-5, f"y 跨度 = {2 * HALF_X}+0.3（前后+标记）", f"got {ext_y:.3f}")

    # ------------------------------------------------------------------ 5
    section("5. ⭐ 躯干俯仰/横滚时，框必须**保持水平**（ray_alignment='yaw' 的语义）")
    pitch30 = np.array([np.cos(np.deg2rad(15)), 0.0, np.sin(np.deg2rad(15)), 0.0])
    roll30 = np.array([np.cos(np.deg2rad(15)), np.sin(np.deg2rad(15)), 0.0, 0.0])
    _, pts_p = draw([0, 0, 0.82], pitch30)
    _, pts_r = draw([0, 0, 0.82], roll30)
    check(np.allclose(pts_p[:, 2], g.SCAN_DRAW_Z, atol=1e-9), "俯仰 30° → 框仍贴地（没被带倾斜）")
    check(np.allclose(pts_r[:, 2], g.SCAN_DRAW_Z, atol=1e-9), "横滚 30° → 框仍贴地")
    _, pts_flat = draw([0, 0, 0.82], yaw_quat(0))
    check(
        np.allclose(pts_p[:, :2], pts_flat[:, :2], atol=1e-6),
        "俯仰 30° 的 xy 投影与水平姿态**一致**（说明只取了 yaw）",
    )

    # ------------------------------------------------------------------ 6
    section("6. 边界安全：极端姿态（机体 x 轴竖直）不能崩")
    up = np.array([np.cos(np.deg2rad(45)), 0.0, np.sin(np.deg2rad(45)), 0.0])  # 机头朝天
    try:
        n_up, pts_up = draw([0, 0, 0.5], up)
        check(n_up == 12 and np.isfinite(pts_up).all(), "机头朝天时仍画出 12 段且数值有限")
    except Exception as exc:  # noqa: BLE001
        check(False, "机头朝天时不崩", f"抛了 {type(exc).__name__}: {exc}")

    print(f"\n{'=' * 74}")
    if FAILS:
        print(f"[SCAN-VISUAL FAIL] {len(FAILS)} 项失败：")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("[SCAN-VISUAL PASS] 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
