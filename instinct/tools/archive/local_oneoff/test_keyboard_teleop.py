#!/usr/bin/env python3
"""键盘遥控器（`g1_mujoco_play.KeyboardCommandController`）的离线单元测试。

**不开窗口、不跑策略** —— 只验证键位表与指令演进逻辑，
所以可以随时跑（CI 友好）。查看器本身的行为仍需人工开窗确认。

跑法：
    python instinct/tools/test_keyboard_teleop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import glfw  # noqa: E402
import numpy as np  # noqa: E402

import g1_mujoco_play as g  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(label)


def section(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def keys_for(action: str) -> list[int]:
    return sorted(k for k, a in g._KEY_ACTIONS.items() if a == action)


def main() -> int:
    # ------------------------------------------------------------------ 1
    section("1. 键位表：GLFW 键码与 ASCII 字符码**两种语义都必须命中**")
    expect = {
        "vx+": ("W", "w"), "vx-": ("S", "s"),
        "vy+": ("A", "a"), "vy-": ("D", "d"),
        "wz+": ("Q", "q"), "wz-": ("E", "e"),
        "zero": ("X", "x", " "), "reset": ("R", "r"),
        "step-": ("[",), "step+": ("]",),
    }
    for action, chars in expect.items():
        codes = keys_for(action)
        ok = True
        detail = []
        for ch in chars:
            code = ord(ch)
            hit = g._KEY_ACTIONS.get(code) == action
            ok &= hit
            detail.append(f"ord({ch!r})={code}{'✓' if hit else '✗'}")
        # GLFW 侧（字母/空格/括号的 glfw 码恰好等于大写 ASCII，这里显式验一遍）
        letter = chars[0].upper()
        gk = getattr(glfw, f"KEY_{letter}", None) if len(chars[0]) == 1 and chars[0].isalpha() else None
        if gk is not None:
            hit = g._KEY_ACTIONS.get(int(gk)) == action
            ok &= hit
            detail.append(f"glfw.KEY_{letter}={gk}{'✓' if hit else '✗'}")
        check(ok, f"{action:6s} <- {'/'.join(repr(c) for c in chars)}", "  ".join(detail))

    check(g._KEY_ACTIONS.get(int(glfw.KEY_SPACE)) == "zero", "空格 = zero (glfw.KEY_SPACE)")
    check(g._KEY_ACTIONS.get(int(glfw.KEY_LEFT_BRACKET)) == "step-", "glfw.KEY_LEFT_BRACKET = step-")
    check(g._KEY_ACTIONS.get(int(glfw.KEY_RIGHT_BRACKET)) == "step+", "glfw.KEY_RIGHT_BRACKET = step+")

    # ------------------------------------------------------------------ 2
    section("2. 指令演进：增量 / 清零 / 夹到训练范围")
    ctl = g.KeyboardCommandController([0.0, 0.0, 0.0], step=0.1, verbose=False)

    def cmd() -> np.ndarray:
        return ctl.poll()[0]

    for _ in range(3):
        ctl.on_key(ord("W"))
    check(np.allclose(cmd(), [0.3, 0.0, 0.0]), "W×3 -> vx=0.3", f"got {cmd()}")

    for _ in range(50):
        ctl.on_key(ord("W"))
    check(np.isclose(cmd()[0], g.CMD_LIMITS[0][1]), f"W×50 夹到上限 vx={g.CMD_LIMITS[0][1]}", f"got {cmd()[0]:.3f}")

    for _ in range(50):
        ctl.on_key(ord("S"))
    check(np.isclose(cmd()[0], g.CMD_LIMITS[0][0]), f"S×50 夹到下限 vx={g.CMD_LIMITS[0][0]}", f"got {cmd()[0]:.3f}")

    ctl.on_key(ord(" "))
    check(np.allclose(cmd(), [0.0, 0.0, 0.0]), "空格 -> 全零")

    ctl.on_key(ord("A"))
    check(np.isclose(cmd()[1], 0.1), "A -> vy=+0.1", f"got {cmd()[1]:.3f}")
    ctl.on_key(ord("D"))
    ctl.on_key(ord("D"))
    check(np.isclose(cmd()[1], -0.1), "D×2 -> vy=-0.1", f"got {cmd()[1]:.3f}")
    for _ in range(20):
        ctl.on_key(ord("D"))
    check(np.isclose(cmd()[1], g.CMD_LIMITS[1][0]), f"vy 夹到下限 {g.CMD_LIMITS[1][0]}", f"got {cmd()[1]:.3f}")

    ctl.on_key(ord("Q"))
    check(np.isclose(cmd()[2], 0.1), "Q -> wz=+0.1", f"got {cmd()[2]:.3f}")
    for _ in range(20):
        ctl.on_key(ord("E"))
    check(np.isclose(cmd()[2], g.CMD_LIMITS[2][0]), f"wz 夹到下限 {g.CMD_LIMITS[2][0]}", f"got {cmd()[2]:.3f}")

    # ------------------------------------------------------------------ 3
    section("3. step 预设与 [ / ] 调整")
    ctl2 = g.KeyboardCommandController([0.0, 0.0, 0.0], step=0.1, verbose=False)
    seen = [ctl2._step]
    for _ in range(3):
        ctl2.on_key(ord("]"))
        seen.append(ctl2._step)
    check(seen == [0.1, 0.2, 0.4, 0.8], "]==] 依次升到 0.2/0.4/0.8（封顶）", f"got {seen}")
    for _ in range(8):
        ctl2.on_key(ord("["))
    check(np.isclose(ctl2._step, min(g.KeyboardCommandController.STEP_PRESETS)), "[ 降到底部封顶", f"got {ctl2._step}")

    # step 影响增量大小
    ctl3 = g.KeyboardCommandController([0.0, 0.0, 0.0], step=0.4, verbose=False)
    check(np.isclose(ctl3._step, 0.4), "--cmd-step 0.4 对上预设")
    ctl3.on_key(ord("W"))
    check(np.isclose(ctl3.poll()[0][0], 0.4), "step=0.4 时 W 一次 -> vx=0.4")

    # ------------------------------------------------------------------ 4
    section("4. 复位标志：只生效一次（避免每步都复位）")
    ctl4 = g.KeyboardCommandController([0.0, 0.0, 0.0], verbose=False)
    check(ctl4.poll()[1] is False, "初始无复位请求")
    ctl4.on_key(ord("R"))
    check(ctl4.poll()[1] is True, "R -> 第一次 poll 拿到 True")
    check(ctl4.poll()[1] is False, "第二次 poll 自动清成 False")

    # ------------------------------------------------------------------ 5
    section("5. 未知键必须被安全忽略（不能抛异常）")
    ok = True
    for code in (0, 1, 999, -5, 12345):
        try:
            ctl4.on_key(code)
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"    code={code} 抛了 {type(exc).__name__}: {exc}")
    check(ok, "随机键码全部安全忽略")

    # ------------------------------------------------------------------ 6
    section("6. reset_state：真实 MJCF + 必须跟一次 mj_forward")
    import mujoco

    xml = g.DEFAULT_XML
    if not Path(xml).is_file():
        check(False, f"找不到 MJCF: {xml}")
    else:
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        # 先故意弄脏状态，再复位，确认真的回到初始
        data.qpos[:] = 1.234
        data.qvel[:] = 5.678
        g.reset_state(model, data)
        check(np.isclose(data.qpos[2], 0.82, atol=1e-3), "复位后 root z = 0.82", f"got {data.qpos[2]:.4f}")
        check(np.allclose(data.qvel, 0.0), "复位后 qvel 全零")
        # mj_forward 生效的判据：派生量 xpos 必须是有限且非全零
        check(np.isfinite(data.xpos).all() and np.abs(data.xpos).max() > 0.01,
              "派生量 data.xpos 已初始化（mj_forward 生效）",
              f"|xpos|max={np.abs(data.xpos).max():.3f}")
        # 再复位两次，确认幂等
        g.reset_state(model, data)
        g.reset_state(model, data)
        check(np.isclose(data.qpos[2], 0.82, atol=1e-3), "多次复位幂等")

    # ------------------------------------------------------------------ 汇总
    print(f"\n{'=' * 74}")
    if FAILS:
        print(f"[KEYBOARD-TELEOP FAIL] {len(FAILS)} 项失败：")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("[KEYBOARD-TELEOP PASS] 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
