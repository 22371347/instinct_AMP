#!/usr/bin/env python3
"""**回放布局回归测试**（`g1_mujoco_play.py` 的 `AMP_WALK_LAYOUTS`）。

    source /home/leo/miniconda3/etc/profile.d/conda.sh && conda activate instinct
    python3 instinct/tools/test_replay_layouts.py

为什么需要它
------------
布局表里一个数字写错，**不会报错** —— 只会让回放喂给策略一份错位的观测，
然后表现为"策略跑不动"，让人误判成策略本身的问题。这类错误必须靠断言兜住。

⚠️ 本测试**只查布局算术**（不需要 MuJoCo / isaaclab）。
   端到端（含实际拼出的观测值）请用回放脚本自己的自检：
       python3 instinct/tools/g1_mujoco_play.py --selftest-obs --obs-mode amp663
       python3 instinct/tools/g1_mujoco_play.py --selftest-obs --obs-mode amp1500
"""

from __future__ import annotations

import importlib.util
import sys

PLAY_PY = "instinct/tools/g1_mujoco_play.py"

#: 期望：布局名 → (总维度, **每帧**扫描维, 尾段维, 尾段帧数, 是否含 base_lin_vel, 扫描帧数)
EXPECT: dict[str, tuple[int, int, int, int, bool, int]] = {
    "amp626": (626, 527, 99, 1, True, 1),
    "amp378": (378, 279, 99, 1, True, 1),
    "loco623": (623, 527, 96, 1, False, 1),
    "amp663": (663, 279, 384, 4, False, 1),      # ⭐ T1o：279 + 96×4
    "amp1500": (1500, 279, 384, 4, False, 4),    # ⭐⭐ T2：279×4 + 96×4
}


def _load_play():
    """按路径加载回放脚本（其内部对 mujoco 的 import 都是惰性的）。"""
    sys.path.insert(0, "instinct/tools")          # 供 `joint_order` / `g1_params` 用
    spec = importlib.util.spec_from_file_location("g1_mujoco_play_mod", PLAY_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["g1_mujoco_play_mod"] = mod       # dataclass 需要它在 sys.modules 里
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    m = _load_play()
    layouts = m.AMP_WALK_LAYOUTS
    failures: list[str] = []

    print(f"布局表共 {len(layouts)} 项：{sorted(layouts)}")
    for name, (dim, scan, tail, nf, wlv, sf) in EXPECT.items():
        if name not in layouts:
            failures.append(f"{name} 不在 AMP_WALK_LAYOUTS 里")
            continue
        lay = layouts[name]
        got = (lay.obs_dim, lay.scan_dim, lay.tail_dim, lay.n_frames, lay.with_lin_vel,
               lay.scan_frames)
        ok = got == (dim, scan, tail, nf, wlv, sf)
        if not ok:
            failures.append(f"{name}: {got} != {(dim, scan, tail, nf, wlv, sf)}")
        print(
            f"  [{name}] obs_dim={lay.obs_dim} scan/帧={lay.scan_dim}×{lay.scan_frames} "
            f"tail={lay.tail_dim} n_frames={lay.n_frames} lin_vel={lay.with_lin_vel} "
            f"{'OK' if ok else 'FAIL'}"
        )

    # ---- 663 的专项不变量（这几点错一个就会静默错位）----
    t1o, ref = layouts.get("amp663"), layouts.get("amp378")
    if t1o is not None and ref is not None:
        checks = [
            ("扫描几何与 amp378 完全一致（9×31 / 3.0×0.8 / 前移 1.5）",
             (t1o.scan_ny, t1o.scan_nx, t1o.scan_size, t1o.scan_forward)
             == (ref.scan_ny, ref.scan_nx, ref.scan_size, ref.scan_forward)),
            ("`height_scan` **不参与**堆叠（`scan_frames=1`）", t1o.scan_frames == 1),
            ("尾段去掉 base_lin_vel（mjlab 里它只在 critic）", t1o.with_lin_vel is False),
            ("帧数 = 4（= mjlab 的 history_length）", t1o.n_frames == 4),
            ("尾段 = 96×4 = 384", t1o.tail_dim == 384),
            ("总维度 = 279+384 = 663", t1o.obs_dim == 663),
        ]
        print("\n663 专项不变量：")
        for desc, ok in checks:
            if not ok:
                failures.append(f"663 不变量失败：{desc}")
            print(f"  {'OK  ' if ok else 'FAIL'} {desc}")

    # ---- ⭐⭐ 1500（T2）的专项不变量：扫描块 279×4 + 尾段 96×4 ----
    t2 = layouts.get("amp1500")
    if t2 is not None and ref is not None:
        checks = [
            ("扫描几何与 amp378/amp663 完全一致（9×31 / 3.0×0.8 / 前移 1.5）",
             (t2.scan_ny, t2.scan_nx, t2.scan_size, t2.scan_forward)
             == (ref.scan_ny, ref.scan_nx, ref.scan_size, ref.scan_forward)),
            ("`height_scan` **也堆叠** 4 帧（用户：height_scan 也是 4 帧）", t2.scan_frames == 4),
            ("扫描块 = 279×4 = 1116", t2.scan_block_dim == 1116),
            ("尾段 = 96×4 = 384（无 base_lin_vel）",
             t2.tail_dim == 384 and t2.with_lin_vel is False),
            ("总维度 = 1116+384 = 1500", t2.obs_dim == 1500),
        ]
        print("\n1500（T2）专项不变量：")
        for desc, ok in checks:
            if not ok:
                failures.append(f"1500 不变量失败：{desc}")
            print(f"  {'OK  ' if ok else 'FAIL'} {desc}")

    # ---- 单帧布局必须零回归（n_frames=1 ⇒ tail_dim 与历史行为相同）----
    for name in ("amp626", "amp378", "loco623"):
        lay = layouts.get(name)
        if lay is None:
            continue
        if lay.n_frames != 1:
            failures.append(f"{name} 的 n_frames 应为 1（单帧布局），实际 {lay.n_frames}")
        if lay.scan_frames != 1:
            failures.append(f"{name} 的 scan_frames 应为 1（单帧布局），实际 {lay.scan_frames}")
        if lay.obs_dim != lay.scan_dim + lay.tail_dim:
            failures.append(f"{name} 的单帧 obs_dim 不等于 scan+tail")

    print()
    if failures:
        print(f"❌ 失败 {len(failures)} 项：" + "；".join(failures))
        return 1
    print("✅ 全部通过（端到端请再用 --selftest-obs --obs-mode amp663 跑一次）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
