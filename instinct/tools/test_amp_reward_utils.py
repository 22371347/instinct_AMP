#!/usr/bin/env python3
"""V23 两个新数学件的本地单测（**纯 torch，无 Isaac / 无 GPU**）。

被测对象：`instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/mdp/amp_reward_utils.py`
  * `ProgressGate`            —— 进度判据（方向投影 + EMA 低通）
  * `weighted_deadband_mean`  —— 逐关节死区 + 内部权重（按 Σw_j 归一）
  * `name_weight_row`         —— 按关节名构造权重/死区行

断言（T1）：踏步⇒不放行 / 真走⇒放行 / 反向⇒不放行 / **站立档噪声不得误放行** / 纯转向⇒放行 /
            有 ω 指令但跟不上⇒不放行 / reset 清零 / κ 相对指令标定 / 起步延迟量级。

用法：
    conda activate instinct && python instinct/tools/test_amp_reward_utils.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
UTILS_PY = (
    REPO
    / "instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/mdp/amp_reward_utils.py"
)

_spec = importlib.util.spec_from_file_location("amp_reward_utils_under_test", UTILS_PY)
assert _spec is not None and _spec.loader is not None
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  —— {detail}" if detail else ""))


N = 4  # 同时测 4 个 env，每个 env 走一种行为
DT, TAU, KAPPA, KW = 0.02, 0.3, 0.35, 0.3


def run(gate, v_act, cmd, w_act, w_cmd, steps=150):
    """跑 steps 步，返回 (EMA, progress, ratio) 的末值。"""
    out = None
    for _ in range(steps):
        out = gate(v_act, cmd, w_act, w_cmd, kappa=KAPPA, omega_kappa=KW)
    return out


def main() -> int:
    print("=" * 84)
    print("T0 载入被测模块")
    print("=" * 84)
    print(f"  {UTILS_PY}")
    check("ProgressGate 存在", hasattr(m, "ProgressGate"))
    check("weighted_deadband_mean 存在", hasattr(m, "weighted_deadband_mean"))
    check("name_weight_row 存在", hasattr(m, "name_weight_row"))

    print()
    print("=" * 84)
    print("T1 ProgressGate：方向投影 + EMA")
    print("=" * 84)
    g = m.ProgressGate(N, dt=DT, tau=TAU)
    # env0: 踏步原地（前后冲量交替，净位移 0），指令 = 前进 0.6
    # env1: 真走（v = 指令 0.6），指令 = 前进 0.6
    # env2: 反向走（v = −0.6），指令 = 前进 0.6
    # env3: 站立档噪声（指令 0，v 噪声 ±0.2，ω_act 噪声 ±0.3，ω_cmd = 0）
    cmd = torch.tensor([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.0, 0.0]])
    w_cmd = torch.tensor([0.0, 0.0, 0.0, 0.0])
    w_act = torch.tensor([0.0, 0.0, 0.0, 0.3])
    v_step = torch.tensor([[0.5, 0.0], [0.6, 0.0], [-0.6, 0.0], [0.2, 0.0]])
    v_empty = torch.tensor([[0.0, 0.0]] * N)
    ema = prog = ratio = None
    for i in range(150):
        v = v_step.clone() if i % 2 == 0 else v_empty.clone()  # env0: 前冲量
        v[0] = torch.tensor([0.5, 0.0]) if i % 2 == 0 else torch.tensor([-0.5, 0.0])  # env0: **前后交替摇摆**
        v[1] = torch.tensor([0.6, 0.0])  # env1 一直真走
        v[2] = torch.tensor([-0.6, 0.0])  # env2 一直反向
        v[3] = torch.tensor([0.2 if i % 2 == 0 else -0.2, 0.0])  # env3 站立噪声
        ema, prog, ratio = g(v, cmd, w_act, w_cmd, kappa=KAPPA, omega_kappa=KW)
    print(f"  EMA      = {[round(float(x), 4) for x in ema]}")
    print(f"  progress = {[int(float(x)) for x in prog]}   (env0 踏步±0.5 / env1 真走0.6 / env2 反向 / env3 站立噪声)")
    print(f"  ratio    = {[round(float(x), 3) for x in ratio]}")
    check("踏步（前后交替 ±0.5，净位移 0）⇒ 不放行", float(prog[0]) == 0.0, f"EMA={float(ema[0]):.4f}（线 {KAPPA * 0.6:.3f}）")
    check("真走 ⇒ 放行", float(prog[1]) == 1.0, f"EMA={float(ema[1]):.4f} ≈ v_cmd 0.6")
    check("反向走 ⇒ 不放行", float(prog[2]) == 0.0, f"EMA={float(ema[2]):.4f}")
    check(
        "站立档噪声（ω_cmd=0 但 ω_act 有噪声）⇒ 不放行（守住 turning 误放行）",
        float(prog[3]) == 0.0,
        f"EMA={float(ema[3]):.4f}",
    )

    # 不对称摇摆（边踏边前漂 0.1 m/s）：EMA ≈ 净漂移量，应仍低于门线
    g6 = m.ProgressGate(1, dt=DT, tau=TAU)
    c6 = torch.tensor([[0.6, 0.0, 0.0]])
    for i in range(150):
        vv = torch.tensor([[0.6, 0.0]]) if i % 2 == 0 else torch.tensor([[-0.4, 0.0]])  # 均值 +0.1
        _, p6, _ = g6(vv, c6, torch.zeros(1), torch.zeros(1), kappa=KAPPA)
    e6 = float(g6.ema[0])
    check("不对称摇摆（边踏边前漂 0.1 m/s）⇒ 仍不放行", float(p6[0]) == 0.0, f"EMA={e6:.4f} ≈ 净漂移 0.1")

    # 纯转向档：cmd_speed=0、ω_cmd=0.5、ω_act=0.5 ⇒ 放行；ω_act 跟不上 ⇒ 不放行
    g2 = m.ProgressGate(2, dt=DT, tau=TAU)
    cmd2 = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    w_act2 = torch.tensor([0.5, 0.05])
    e2, p2, _ = run(g2, torch.zeros(2, 2), cmd2, w_act2, cmd2[:, 2])
    check("纯转向档（ω_act 跟得上）⇒ 放行", float(p2[0]) == 1.0, f"ω_act=0.5 vs 0.3×0.5")
    check("有 ω 指令但跟不上 ⇒ 不放行", float(p2[1]) == 0.0, f"ω_act=0.05 < 0.15")

    # κ 相对指令：同 κ 下 cmd 0.3 与 1.0 的"达标线"不同
    g3 = m.ProgressGate(2, dt=DT, tau=TAU)
    cmd3 = torch.tensor([[0.3, 0.0, 0.0], [1.0, 0.0, 0.0]])
    v3 = torch.tensor([[0.2, 0.0], [0.2, 0.0]])
    e3, p3, r3 = run(g3, v3, cmd3, torch.zeros(2), torch.zeros(2))
    print(f"  κ 相对性：v_act=0.2 时  cmd=0.3 ⇒ progress={int(float(p3[0]))}（线 0.105）,  cmd=1.0 ⇒ progress={int(float(p3[1]))}（线 0.35）")
    check("κ 相对指令标定（0.2 m/s 在 0.3 指令下达标、在 1.0 指令下不达标）",
          float(p3[0]) == 1.0 and float(p3[1]) == 0.0)

    # reset
    g4 = m.ProgressGate(2, dt=DT, tau=TAU)
    run(g4, torch.tensor([[0.6, 0.0], [0.6, 0.0]]), torch.tensor([[0.6, 0.0, 0.0]] * 2),
        torch.zeros(2), torch.zeros(2))
    before = float(g4.ema[0])
    g4.reset_by_mask(torch.tensor([True, False]))
    after0, after1 = float(g4.ema[0]), float(g4.ema[1])
    check("reset_by_mask 只清指定 env", after0 == 0.0 and after1 > 0.5, f"{before:.3f} → [{after0:.3f}, {after1:.3f}]")

    # 起步延迟（τ=0.3 ⇒ 真走时要 ~0.1 s 才开门）
    g5 = m.ProgressGate(1, dt=DT, tau=TAU)
    v = torch.tensor([[0.6, 0.0]])
    c = torch.tensor([[0.6, 0.0, 0.0]])
    steps_to_open = None
    for i in range(100):
        _, p, _ = g5(v, c, torch.zeros(1), torch.zeros(1), kappa=KAPPA)
        if float(p[0]) == 1.0 and steps_to_open is None:
            steps_to_open = i + 1
    print(f"  起步延迟：真走后第 {steps_to_open} 步（{steps_to_open * DT:.2f} s）开门")
    check("起步延迟在 0.1~0.5 s（不会一直拖着不给分）", steps_to_open is not None and 5 <= steps_to_open <= 25)

    print()
    print("=" * 84)
    print("T2 weighted_deadband_mean：逐关节死区 + 内部权重")
    print("=" * 84)
    names = ["left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_knee_joint",
             "right_knee_joint", "left_ankle_roll_joint", "waist_yaw_joint"]
    err = torch.tensor([[0.93, 0.93, 0.60, 0.60, 0.12, 0.23]])  # V19d 实测量级
    w = m.name_weight_row(names, {"waist_yaw_joint": 3.0}, default=1.0)
    tol = m.name_weight_row(names, {"left_shoulder_roll_joint": 0.05}, default=0.08)
    cost = m.weighted_deadband_mean(err, w, tol)
    cost_x2 = m.weighted_deadband_mean(err, w * 2.0, tol)
    print(f"  权重行 = {[float(x) for x in w]}   死区行 = {[round(float(x), 2) for x in tol]}")
    print(f"  cost = {float(cost):.4f}    （权重整体 ×2 后 cost = {float(cost_x2):.4f}）")
    check("死区以下的部分不计（踝 0.12 < 0.08? 否；肩 0.93−0.05）", float(cost) > 0)
    check("权重整体缩放不改变 cost（分母 Σw_j 归一）", abs(float(cost) - float(cost_x2)) < 1e-6)

    inside = torch.full((1, len(names)), 0.03)
    check("全部落在死区内 ⇒ cost = 0", float(m.weighted_deadband_mean(inside, w, tol)) == 0.0)

    # 分组占比：按"腿 vs 臂"的内部权重，腿应占主导
    w_g = m.name_weight_row(names, {
        "left_shoulder_roll_joint": 1.0, "right_shoulder_roll_joint": 1.0,  # 臂
        "left_knee_joint": 2.0, "right_knee_joint": 2.0,                    # 膝
        "left_ankle_roll_joint": 2.0,                                       # 踝
        "waist_yaw_joint": 1.0,
    }, default=1.0)
    tol_g = m.name_weight_row(names, {"left_shoulder_roll_joint": 0.05}, default=0.08)
    pen = torch.relu(err - tol_g) * w_g
    arms = float(pen[0, :2].sum())
    legs = float(pen[0, 2:5].sum())
    print(f"  分组贡献（死区后）：臂 {arms:.3f} / 腿 {legs:.3f} ⇒ 腿占比 {100 * legs / (arms + legs):.0f}%")
    check("按内部权重可让腿在姿态项里占主导（>50%）", legs > arms)

    print()
    print("=" * 84)
    print("T3 name_weight_row")
    print("=" * 84)
    r = m.name_weight_row(["a", "b"], None, default=1.0)
    r2 = m.name_weight_row(["a", "b"], {"a": 0.5}, default=1.0)
    check("None ⇒ 全 default", [float(x) for x in r] == [1.0, 1.0])
    check("未列出的关节用 default；列出的用给定值", [float(x) for x in r2] == [0.5, 1.0])

    print()
    n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
    print("=" * 84)
    print(f"结果：{len(CHECKS) - n_fail}/{len(CHECKS)} 通过" + ("" if n_fail == 0 else f" —— {n_fail} 项失败"))
    print("=" * 84)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
