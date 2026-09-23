#!/usr/bin/env python3
"""标定 AMP 行走任务的「两脚横向间距」(foot lateral separation) 基准。

为什么需要这个脚本
------------------
`AMP_mjlab` 的 `feet_lateral_width_cost` / `feet_crossing_cost` 里有一个硬阈值
`sep_floor`（两脚横向间距下限；低于它 = 「窄步」受罚）。那个数是在 **mjlab 的 G1 模型 +
`left_foot`/`right_foot` site** 上标定出来的（数据集 `walk_forward_loop_002__A022`
的中位 ≈0.199、p1 = 0.157，`sep_floor` 取 0.15）。

搬到 `instinct_xu` 后有三点必须重新确认，**不能照抄那个数**：

1. **foot 参考点不同**：IsaacLab 侧没有 site，只有 body。
   本仓库 G1 URDF 里有 `LL_FOOT`/`LR_FOOT` 两个 body，是 `*_ankle_roll_link` 下
   偏移 `(0.04, 0, -0.037)` 的脚底框 —— 与 mjlab 的 `left_foot`/`right_foot` 同源，
   但坐标轴定义（URDF body frame vs MJCF site frame）不一定逐点相同。
2. **base 帧不同**：mjlab 的 root 是 `pelvis`，instinct_xu 的 root 是 `torso_link`。
   `R_yaw` 取的是 root 的偏航，两者差一个 `waist_yaw_joint` 角。
3. **关节/地形不同**：我们的参考动作 npz 是转换后的 17 段（`*_retargeted.npz`）。

本脚本用**与运行时奖励函数完全相同的量**重新标定：

    sep(t) = [ R_yaw(q_root)^T · ( p_LL_FOOT(t) − p_LR_FOOT(t) ) ]_y

其中 `p_*` 由 MuJoCo 正运动学算出（输入 = 转换后 npz 的 `base_pos_w` / `base_quat_w`
/ `joint_pos`，即 `torso_link` 位姿 + 29 关节角）。

输出
----
- 终端报告：每个 clip 的 sep 分位数 + 全局汇总
- `--json`（默认 `tools/amp_walk_foot_sep_stats.json`）：给静态检查器做数值对账用

用法
----
    conda activate instinct
    python tools/calibrate_foot_sep.py                # 报告 + 写 JSON
    python tools/calibrate_foot_sep.py --no-json      # 只报告
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # .../llp/instinct
DEFAULT_XML = ROOT / "assets/g1_29dof_popsicle_play.xml"
DEFAULT_MOTION_DIR = (
    ROOT.parent
    / "instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/motions/amp_walk"
)
DEFAULT_JSON = ROOT / "tools/amp_walk_foot_sep_stats.json"

#: 「健康行走」标定集 —— **只取直线行走** 4 段。
#:
#: 依据（实测，见本脚本输出）：
#: - 侧移 2 段（`walk_sideway_*`）有 **~22~24%** 的帧 sep<0 —— 那是**交叉式侧步**
#:   （grapevine），是参考动作本身的风格，不是缺陷 → 不能拿它定地板，且**宽度/交叉
#:   项必须对纯侧向指令做方向门控**，否则会与 AMP 风格先验互相打架（这正是
#:   AMP_mjlab §"数据集覆盖面决定 AMP 会压制什么" 记录的坑）。
#: - 慢跑 clip（`jog_*`）摆动腿合法内收，sep 会低到 0.04~0.07 → 由
#:   `walk_max_lin_vel` 速档门控豁免，不参与地板标定。
#: - 弧线 / 原地转 / 原地踏步：非直线步态，不参与。
WALK_CLIPS = (
    "walk_forward_loop_002__A022",
    "walk_forward_loop_002__A024",
    "walk_backward_loop_001__A022",
    "walk_backward_loop_001__A024",
)

#: 交叉式侧步（仅用于报告，提醒读者「这些负 sep 是参考动作本身的风格」）
CROSS_STEP_CLIPS = (
    "walk_sideway_left_loop_002__A021",
    "walk_sideway_right_loop_001__A022",
)

#: 与运行时奖励函数一致的 foot body 候选（前者优先，后者为兜底）。
FOOT_BODY_PAIRS = (
    ("LL_FOOT", "LR_FOOT"),
    ("left_ankle_roll_link", "right_ankle_roll_link"),
)


def yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """从 (N,4) wxyz 四元数取偏航角（与 `isaaclab.utils.math.yaw_quat` 语义一致）。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def pick_foot_bodies(body_names: list[str]) -> tuple[str, str]:
    for left, right in FOOT_BODY_PAIRS:
        if left in body_names and right in body_names:
            return left, right
    raise SystemExit(f"找不到脚部 body，候选 {FOOT_BODY_PAIRS} 均不在模型中：{body_names}")


def clip_sep(g1, path: Path, left: str, right: str) -> dict:
    """算单个 clip 的 sep 序列。

    ⚠️ **必须按关节名重映射列序**：npz 的 `joint_pos` 列序 == `joint_names`
    （= AMP_mjlab MJCF 的 hinge 顺序，腿在前），而本仓库 MJCF 的 hinge 顺序是
    **腰在前**。直接按列喂进去会得到完全错误的姿态（实测会让 sep 从 +0.15 变成
    -0.25，凭空造出「交叉」）。这里是本项目第三次踩到「同名不同序」的坑。
    """
    z = np.load(path)
    need = {"joint_pos", "joint_names", "base_pos_w", "base_quat_w"}
    missing = need - set(z.files)
    if missing:
        raise SystemExit(f"{path.name}: 缺少字段 {sorted(missing)}（应先跑 convert_amp_motion.py）")

    jp_raw = np.asarray(z["joint_pos"], dtype=np.float64)
    names = [str(n) for n in z["joint_names"]]
    col = {n: i for i, n in enumerate(names)}
    if set(col) != set(g1.hinge_names):
        only_npz = sorted(set(col) - set(g1.hinge_names))
        only_mjcf = sorted(set(g1.hinge_names) - set(col))
        raise SystemExit(f"{path.name}: 关节名集合不一致\n  仅 npz: {only_npz}\n  仅 MJCF: {only_mjcf}")
    jp = jp_raw[:, [col[n] for n in g1.hinge_names]]         # ← 按名重排

    bp = np.asarray(z["base_pos_w"], dtype=np.float64)
    bq = np.asarray(z["base_quat_w"], dtype=np.float64)

    iL, iR = g1.body_names.index(left), g1.body_names.index(right)
    yaw = yaw_from_quat_wxyz(bq)
    sin_y, cos_y = np.sin(yaw), np.cos(yaw)

    sep = np.empty(len(jp), dtype=np.float64)
    for f in range(len(jp)):
        xpos, _ = g1.fk(bp[f], bq[f], jp[f])
        d = xpos[1:][iL] - xpos[1:][iR]                      # world 下 left - right
        # 把向量转回 root 的偏航系（等价于 quat_apply_inverse(yaw_quat(q), d)）
        sep[f] = -sin_y[f] * d[0] + cos_y[f] * d[1]

    return {
        "file": path.name,
        "frames": int(len(jp)),
        "sep": sep,
    }


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", default=str(DEFAULT_XML), help="G1 MJCF（用于 FK，须与 IsaacLab 资产同源）")
    ap.add_argument("--motion-dir", default=str(DEFAULT_MOTION_DIR), help="转换后的参考动作目录")
    ap.add_argument("--json", default=str(DEFAULT_JSON), help="统计结果输出路径")
    ap.add_argument("--no-json", action="store_true", help="不写 JSON")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from convert_amp_motion import G1Model                          # noqa: PLC0415

    xml, motion_dir = Path(args.xml), Path(args.motion_dir)
    if not xml.exists():
        raise SystemExit(f"MJCF 不存在: {xml}")
    g1 = G1Model(xml)
    left, right = pick_foot_bodies(g1.body_names)
    print(f"MJCF          : {xml}")
    print(f"foot body     : {left} / {right}   (body 数 {g1.n_robot_bodies}，关节数 {len(g1.hinge_names)})")
    print(f"参考动作目录  : {motion_dir}")

    files = sorted(motion_dir.glob("*_retargeted.npz"))
    if not files:
        raise SystemExit(f"{motion_dir} 下没有 *_retargeted.npz")

    print()
    print("=" * 96)
    print(f"{'clip':46s} {'帧':>6s} {'ps0':>8s} {'p1':>8s} {'ps5':>8s} {'中位':>8s} {'均值':>8s} {'<0 占比':>8s}")
    print("=" * 96)

    per_clip: dict[str, dict] = {}
    raw_sep: dict[str, np.ndarray] = {}
    walk_pool: list[np.ndarray] = []
    all_pool: list[np.ndarray] = []
    for f in files:
        stem = f.name.replace("_retargeted.npz", "")
        r = clip_sep(g1, f, left, right)
        s = r["sep"]
        raw_sep[stem] = s
        per_clip[stem] = {
            "frames": r["frames"],
            "min": float(s.min()),
            "p1": pct(s, 1),
            "p5": pct(s, 5),
            "median": float(np.median(s)),
            "mean": float(s.mean()),
            "max": float(s.max()),
            "neg_frac": float((s < 0).mean()),
        }
        all_pool.append(s)
        if stem in WALK_CLIPS:
            walk_pool.append(s)
        mark = " *" if stem in WALK_CLIPS else "  "
        print(
            f"{stem:46s} {r['frames']:6d} {s.min():8.4f} {pct(s, 1):8.4f} {pct(s, 5):8.4f} "
            f"{np.median(s):8.4f} {s.mean():8.4f} {(s < 0).mean():8.4f}{mark}"
        )
    print("=" * 96)
    print("* = 计入「健康行走」标定集")

    walk = np.concatenate(walk_pool)
    allsep = np.concatenate(all_pool)
    cross = np.concatenate(
        [raw_sep[c] for c in CROSS_STEP_CLIPS if c in raw_sep] or [np.array([np.nan])]
    )
    print()
    print(f"直线行走标定集（{len(walk_pool)} 段 / {len(walk)} 帧）")
    print(f"  min={walk.min():.4f}  p1={pct(walk,1):.4f}  p5={pct(walk,5):.4f}  "
          f"中位={np.median(walk):.4f}  均值={walk.mean():.4f}  max={walk.max():.4f}")
    print(f"  真交叉(sep<0)帧占比 = {(walk < 0).mean():.5f}")
    if np.isfinite(cross).any():
        print(f"交叉式侧步（{len(CROSS_STEP_CLIPS)} 段）")
        print(f"  min={cross.min():.4f}  中位={np.median(cross):.4f}  "
              f"真交叉(sep<0)帧占比 = {(cross < 0).mean():.4f}  ← 参考动作本身的风格")
    print(f"全部 clip（{len(all_pool)} 段 / {len(allsep)} 帧）")
    print(f"  min={allsep.min():.4f}  p1={pct(allsep,1):.4f}  p5={pct(allsep,5):.4f}  "
          f"中位={np.median(allsep):.4f}  max={allsep.max():.4f}")

    # sep_floor 的选取准则（与 AMP_mjlab 一致）：取「健康行走」的 p1 —— 健康步态基本
    # 不受罚，只有明显窄于数据集的步态才拿到梯度。向下取到 0.005 的整数倍便于阅读。
    p1w = pct(walk, 1)
    floor_suggest = float(np.floor(p1w / 0.005) * 0.005)
    print()
    print(f"→ 建议 sep_floor = {p1w:.4f} 的向下取整 = {floor_suggest:.3f} m")
    print("  （AMP_mjlab 在其模型上取 0.15；此处必须以本仓库实测为准）")

    if not args.no_json:
        out = Path(args.json)
        out.write_text(
            json.dumps(
                {
                    "xml": str(xml),
                    "motion_dir": str(motion_dir),
                    "foot_bodies": {"left": left, "right": right},
                    "walk_clips": list(WALK_CLIPS),
                    "healthy_walk": {
                        "n_clips": len(walk_pool),
                        "n_frames": int(len(walk)),
                        "min": float(walk.min()),
                        "p1": p1w,
                        "p5": pct(walk, 5),
                        "median": float(np.median(walk)),
                        "mean": float(walk.mean()),
                        "max": float(walk.max()),
                        "neg_frac": float((walk < 0).mean()),
                        "sep_floor_suggest": floor_suggest,
                    },
                    "all_clips": {
                        "n_clips": len(all_pool),
                        "n_frames": int(len(allsep)),
                        "min": float(allsep.min()),
                        "p1": pct(allsep, 1),
                        "p5": pct(allsep, 5),
                        "median": float(np.median(allsep)),
                        "max": float(allsep.max()),
                    },                    "per_clip": per_clip,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n统计已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
