#!/usr/bin/env python3
"""**本地单测**：`DirectionalVelocityCommand` 纯侧向档的 `|vy|` 下限（`side_vel_min`）。

为什么要测
----------
用户要求：纯侧向指令**不要落在 `|vy| <= 0.1`** 区间。原因是所有"静止/低速"门控
（`track_default_pose_on_stand`、`disable_style_on_stand`、`feet_lateral_width/crossing` 的
`w_act`）统一用

        ‖(vx, vy)‖ + |ωz| <= low_vel (0.1)

判"这是立定"。若侧向档采到 `|vy| < 0.1`，这条指令会**同时**被要求"侧移"和"立定回位"
—— 监督信号自相矛盾。本测试把"侧向档永不落入静止门"钉成**不变量**。

为什么不直接 `import`
---------------------
`commands.py` 顶部 import 了 isaaclab（本地没有）。所以本测试从源码里
**按注释标记抽取"纯侧向"分支的真实代码块**再 `exec` 运行 ⇒ 测的是**真源码行**，
不是手抄副本（手抄副本测不出"源码改了忘了同步"这类问题）。

跑法：
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate instinct
    python instinct/tools/test_side_vel_min.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
COMMANDS_PY = (
    REPO / "instinct_xu" / "source" / "instinctlab" / "instinctlab" / "tasks" / "locomotion" / "mdp" / "commands.py"
)

LOW_VEL = 0.1  # 静止门控阈值（与 cfg 的 low_vel / stand_speed_threshold 同源）
LIN_VEL_Y = (-0.5, 0.5)  # cfg.ranges.lin_vel_y
ANG_VEL_Z = (-0.8, 0.8)  # cfg.ranges.ang_vel_z
PROB_LOW = 0.20
PROB_SIDE = 0.25

_results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    _results.append((bool(ok), msg))
    print(f"  {'✅' if ok else '❌'} {msg}")


# --------------------------------------------------------------------------------------
# 1) 从源码抽取"纯侧向"分支
# --------------------------------------------------------------------------------------
def extract_side_branch() -> str:
    src = COMMANDS_PY.read_text(encoding="utf-8")
    begin = src.index("# --- 2) 纯侧向")
    end = src.index("# --- 3) 纯前后")
    block = src[begin:end]
    assert block.strip(), "抽到的代码块是空的 —— 注释标记是不是被改了？"
    # 真实缩进是 8 空格（方法体内）⇒ 减 4 变成函数体（4 空格）
    body = textwrap.dedent(block)
    body = textwrap.indent(body, "    ")
    return "def side_branch(self, env_ids, m, n, dev, uni, r, p_low, p_side):\n" + body


def build_side_branch():
    ns = {"torch": torch}
    exec(compile(extract_side_branch(), "<commands.py::纯侧向分支>", "exec"), ns)  # noqa: S102
    return ns["side_branch"]


class _Hdr:
    def __init__(self, lin_vel_y, ang_vel_z):
        self.lin_vel_y = lin_vel_y
        self.ang_vel_z = ang_vel_z


class _Cfg:
    def __init__(self, side_vel_min, side_ang_vel_range=None):
        self.ranges = _Hdr(LIN_VEL_Y, ANG_VEL_Z)
        self.side_vel_min = side_vel_min
        if side_ang_vel_range is not None:
            self.side_ang_vel_range = side_ang_vel_range


class _Self:
    def __init__(self, cfg, n):
        self.cfg = cfg
        self.vel_command_b = torch.zeros(n, 3)


def run(side_vel_min, n=200_000, seed=0, m=None, side_ang_vel_range=None):
    """跑一次抽出来的侧向分支；`m` 默认全部落在侧向签内。"""
    branch = build_side_branch()
    cfg = _Cfg(side_vel_min, side_ang_vel_range)
    obj = _Self(cfg, n)
    dev = torch.device("cpu")
    env_ids = torch.arange(n, device=dev)
    if m is None:
        m = torch.rand(n, device=dev)  # 占位，下面覆盖成"一定落在侧向档"
    torch.manual_seed(seed)
    branch(
        obj,
        env_ids,
        m,
        n,
        dev,
        lambda k, lo, hi: torch.empty(k, device=dev).uniform_(lo, hi),  # 与源码里的 uni 一致
        cfg.ranges,
        PROB_LOW,
        PROB_LOW + PROB_SIDE,
    )
    return obj.vel_command_b


def in_side_bucket(n: int, seed: int = 1234) -> torch.Tensor:
    """构造一定落在 [p_low, p_side) 的签 m（并顺带验证桶宽 = prob_side）。"""
    torch.manual_seed(seed)
    u = torch.rand(n)
    return PROB_LOW + (PROB_SIDE) * u


def main() -> int:
    print("=" * 88)
    print("单测：DirectionalVelocityCommand 纯侧向档 side_vel_min")
    print(f"源码：{COMMANDS_PY.relative_to(REPO)}")
    print("=" * 88)
    n = 200_000
    m = in_side_bucket(n)

    # ------------------------------------------------------------------ 契约（文本级）
    print("\n[0] 配置字段契约")
    src = COMMANDS_PY.read_text(encoding="utf-8")
    check("side_vel_min: float = 0.0" in src, "`DirectionalVelocityCommandCfg` 里有 `side_vel_min: float = 0.0`")
    check("side_ang_vel_range: tuple[float, float] | None = None" in src, "`side_ang_vel_range` 字段未被破坏")

    # ------------------------------------------------------------------ 核心不变量
    print("\n[1] side_vel_min=0.15（V33 口径）：侧向档不再与「静止档」冲突")
    v = run(0.15, n=n, m=m)
    vy = v[:, 1]
    check(torch.all(v[:, 0] == 0.0), "vx ≡ 0（纯侧向）")
    check(bool((vy.abs() >= 0.15 - 1e-5).all()) and bool((vy.abs() <= 0.5 + 1e-6).all()),
          f"|vy| ∈ [0.15, 0.5]（实测 min={vy.abs().min():.4f}, max={vy.abs().max():.4f}）")
    gate = torch.norm(v[:, :2], dim=1) + v[:, 2].abs()
    n_stand = int((gate <= LOW_VEL).sum())
    check(n_stand == 0, f"⭐ 被判「静止」的样本数 = {n_stand}（必须为 0：‖(vx,vy)‖+|ωz| ≤ {LOW_VEL}）")
    neg = float((vy < 0).float().mean())
    check(abs(neg - 0.5) < 0.005, f"左右符号均衡：负向占比 {neg:.4f}（期望 ≈0.5）")
    check(abs(float(vy.abs().mean()) - 0.325) < 0.005, f"|vy| 均值 {float(vy.abs().mean()):.4f}（期望 ≈0.325）")
    check(float(v[:, 2].min()) < -0.75 and float(v[:, 2].max()) > 0.75,
          f"默认 ωz 沿用 ranges.ang_vel_z：实测 [{float(v[:, 2].min()):.3f}, {float(v[:, 2].max()):.3f}]")
    # 侧向档占比 = prob_side
    torch.manual_seed(99)
    mm = torch.rand(n)
    frac = float(((mm >= PROB_LOW) & (mm < PROB_LOW + PROB_SIDE)).float().mean())
    check(abs(frac - PROB_SIDE) < 0.005, f"侧向档占比 {frac:.4f}（期望 = prob_side = {PROB_SIDE}）")

    # ------------------------------------------------------------------ 向下兼容（逐字）
    print("\n[2] side_vel_min=0.0：必须与「加这个字段之前」逐字一致（兼容性回归）")
    v_new = run(0.0, n=n, seed=777, m=m)  # ⚠️ 必须传 m：否则只有少数样本落在侧向档
    torch.manual_seed(777)
    v_old = torch.empty(n).uniform_(*LIN_VEL_Y)  # 旧实现：直接 uni(k, lo, hi)
    check(torch.equal(v_new[:, 1], v_old), "同种子下与旧实现 **逐位相同**（torch.equal）")

    print("\n[3] side_vel_min=None：按 0 处理（getattr(...) or 0.0）")
    v_none = run(None, n=n, seed=777, m=m)
    check(torch.equal(v_none[:, 1], v_old), "None ⇒ 与旧实现逐位相同")

    print("\n[4] 误配 side_vel_min=0.7 > max|lin_vel_y|=0.5：安全回退，不崩、不产生超界值")
    v_bad = run(0.7, n=n, seed=777, m=m)
    check(torch.equal(v_bad[:, 1], v_old), "回退到旧实现（逐位相同，不会把 |vy| 抬到 0.7）")

    # ------------------------------------------------------------------ 兼容 side_ang_vel_range
    print("\n[5] side_angle_vel_range 兼容性：显式给定 0 时仍可用（V13 字段不受影响）")
    v_wz0 = run(0.15, n=1000, m=in_side_bucket(1000), side_ang_vel_range=(0.0, 0.0))
    check(bool((v_wz0[:, 2] == 0.0).all()), "side_ang_vel_range=(0.0,0.0) ⇒ ωz 全 0（未被 (0,0) 假值吞掉）")
    gate0 = torch.norm(v_wz0[:, :2], dim=1) + v_wz0[:, 2].abs()
    check(int((gate0 <= LOW_VEL).sum()) == 0, "⭐ 即使 ωz≡0，侧向档仍不被判静止（|vy| ≥ 0.15 > 0.1）")

    # ------------------------------------------------------------------ 汇总
    bad = [msg for ok, msg in _results if not ok]
    print("\n" + "=" * 88)
    print(f"结果：{len(_results) - len(bad)}/{len(_results)} 通过")
    for msg in bad:
        print(f"  ❌ {msg}")
    print("=" * 88)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
