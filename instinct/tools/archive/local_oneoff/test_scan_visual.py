#!/usr/bin/env python3
"""`height_scan` 扫描区域可视化的**离线几何单测**（不开窗、不跑策略）。

验证的是最容易算错的东西：那个橙色矩形框到底画在哪儿。
⭐ 对 `g.AMP_WALK_LAYOUTS` 里的**每个布局**各跑一遍：

    amp626 (v1~v8): GridPatternCfg(0.1, (3.0, 1.6)) → 31x17 = 527，居中（前移 0）
    amp378 (v9)   : GridPatternCfg(0.1, (3.0, 0.8)) → 31x9  = 279，中心**前移 1.5 m**

判据来自源码：`omni_loco_env_cfg.py:86`；
`instinct_xu/.../config/g1/amp_walk_v9_env_cfg.py`（offset.pos=(1.5,0,20)）
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


def check(cond: bool, label: str, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(label)


def section(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def run_layout(layout_name: str) -> None:
    """对**单个布局**跑完全部几何断言（内部会切全局可视化几何）。

    断言里的期望值**全部由布局推导**，不写死 527/1.5 —— 这样以后新增布局时
    单测自动覆盖，不会出现“测试还在验旧布局”的假绿。
    """
    layout = g.AMP_WALK_LAYOUTS[layout_name]
    g.apply_scan_layout(layout)              # ⭐ 切布局：尺寸 / 前移量 / 射线数 / 网格
    half_x = g.SCAN_SIZE_XY[0] / 2.0
    half_y = g.SCAN_SIZE_XY[1] / 2.0
    fwd = float(g.SCAN_FORWARD)
    x_lo, x_hi = fwd - half_x, fwd + half_x
    n_rays = layout.scan_dim
    print(f"\n{'#' * 74}\n### 布局 {layout_name}：size={layout.scan_size} fwd={fwd:g} "
          f"→ {n_rays} 条射线\n{'#' * 74}")

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
    section(f"1. 射线数自洽：{layout.scan_size[0]}/0.1+1={layout.scan_nx}, "
            f"{layout.scan_size[1]}/0.1+1={layout.scan_ny} → {n_rays}")
    check(g.SCAN_GRID_RAYS == n_rays, f"SCAN_GRID_RAYS == {n_rays}", f"got {g.SCAN_GRID_RAYS}")
    check(g.SCAN_NY == layout.scan_ny and g.SCAN_NX == layout.scan_nx,
          f"SCAN_NY/NX == {layout.scan_ny}/{layout.scan_nx}", f"got {g.SCAN_NY}/{g.SCAN_NX}")
    check(g.SCAN_ALIGNMENT == "yaw", "对齐方式 = yaw（与 ray_alignment 一致）")

    # ------------------------------------------------------------------ 2
    section(f"2. 原地不动、朝向 +x：矩形应是 x∈[{x_lo:g},{x_hi:g}], y∈±{half_y:g}, 贴地")
    # 段数 = 4 边框 + 内网格 + 1 前向标记；内网格是固定 **0.5 m** 步长：
    #   某轴条数 = int(size/0.5) - 1（3.0→5；1.6→2；**0.8→0**，即窄扫描没有横网格线）
    n_grid_x = int(layout.scan_size[0] / 0.5) - 1
    n_grid_y = int(layout.scan_size[1] / 0.5) - 1
    n_expect = 4 + n_grid_x + n_grid_y + 1
    n, pts = draw([0, 0, 0.82], yaw_quat(0))
    check(n == n_expect,
          f"画出 {n_expect} 段（4 边框 + {n_grid_x}+{n_grid_y} 内网格 + 1 前向标记）", f"got {n}")
    # 另一半：确认真的能填进 mjvScene（绘制路径本身也要能跑）
    scn.ngeom = 0
    filled = g.draw_scan_footprint(scn, data, tid)
    check(filled == n_expect and scn.ngeom == n_expect,
          f"draw_scan_footprint 已向 mjvScene 写入 {n_expect} 个 geom",
          f"filled={filled} ngeom={scn.ngeom}")
    check(
        np.allclose([pts[:, 0].min(), pts[:, 0].max()], [x_lo, x_hi + 0.3], atol=1e-6),
        f"x 范围 = [{x_lo:g}, {x_hi:g}+0.3](前向标记)",
        f"got [{pts[:, 0].min():.3f}, {pts[:, 0].max():.3f}]",
    )
    check(
        np.allclose([pts[:, 1].min(), pts[:, 1].max()], [-half_y, half_y], atol=1e-6),
        f"y 范围 = ±{half_y:g}",
        f"got [{pts[:, 1].min():.3f}, {pts[:, 1].max():.3f}]",
    )
    check(np.allclose(pts[:, 2], g.SCAN_DRAW_Z, atol=1e-9), f"z 全部 = {g.SCAN_DRAW_Z}（贴地）")

    # ------------------------------------------------------------------ 3
    section("3. 机器人平移 → 矩形整体跟随（且**不随身高变化**）")
    _, pts_a = draw([2.0, 3.0, 0.82], yaw_quat(0))
    _, pts_b = draw([2.0, 3.0, 1.30], yaw_quat(0))  # 只改高度
    cx = (pts_a[:, 0].min() + pts_a[:, 0].max() - 0.3) / 2.0
    cy = (pts_a[:, 1].min() + pts_a[:, 1].max()) / 2.0
    check(abs(cx - (2.0 + fwd)) < 1e-6 and abs(cy - 3.0) < 1e-6,
          f"矩形中心跟到 (2,3) 的前移后位置 (2+{fwd:g}, 3)", f"got ({cx:.3f},{cy:.3f})")
    check(np.allclose(pts_a, pts_b), "躯干高度从 0.82 抬到 1.30 → 地面上的框**不变**")

    # ------------------------------------------------------------------ 4
    section("4. 转向 90°：前后轴应转到世界 y 轴（范围互换）")
    _, pts90 = draw([0, 0, 0.82], yaw_quat(90))
    ext_x = pts90[:, 0].max() - pts90[:, 0].min()
    ext_y = pts90[:, 1].max() - pts90[:, 1].min()
    check(abs(ext_x - (2 * half_y)) < 1e-6, f"朝向 90° 后 x 跨度 = {2 * half_y:g}（左右）", f"got {ext_x:.3f}")
    check(abs(ext_y - (2 * half_x + 0.3)) < 1e-5, f"y 跨度 = {2 * half_x:g}+0.3（前后+标记）", f"got {ext_y:.3f}")
    cy90 = (pts90[:, 1].min() + pts90[:, 1].max() - 0.3) / 2.0
    check(abs(cy90 - fwd) < 1e-6, f"90° 时框中心沿世界 y 前移 {fwd:g}", f"got {cy90:.3f}")

    # ------------------------------------------------------------------ 5
    section("5. ⭐ 躯干俯仰/横滚时，框必须**保持水平**（ray_alignment='yaw' 的语义，与布局无关）")
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
        check(n_up == n_expect and np.isfinite(pts_up).all(),
              f"机头朝天时仍画出 {n_expect} 段且数值有限", f"got n={n_up}")
    except Exception as exc:  # noqa: BLE001
        check(False, "机头朝天时不崩", f"抛了 {type(exc).__name__}: {exc}")

def main() -> int:
    for name in g.AMP_WALK_LAYOUTS:          # amp626 → amp378，逐布局校验
        run_layout(name)

    print(f"\n{'=' * 74}")
    if FAILS:
        print(f"[SCAN-VISUAL FAIL] {len(FAILS)} 项失败：")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print(f"[SCAN-VISUAL PASS] 全部通过（{len(g.AMP_WALK_LAYOUTS)} 种布局）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
