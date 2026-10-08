#!/usr/bin/env python3
"""**本地单测**：判别器梯度归因的列索引（`wasabi.py::_disc_grad_index`）。

为什么要有它
------------
V32 那版把布局**硬编码**成"每帧 = [pos|ori|lin|ang]"（帧优先），但观测管理器是**按项拼接**的
（`flatten_history_dim=True` 只在**项内**展平历史）⇒ 真实布局是

    [body_pos_b | body_ori_b | body_lin_vel_b | body_ang_vel_b | anchor_gravity_b]
       2×39         2×78           2×39              2×39               2×3

即"**项优先、帧在项内**"。V33 改成从运行时 `obs_format` 段读（`get_obs_slice`）后，
本测试用**构造的段**把这条口径钉死：每个分组恰好覆盖自己那几维、互不重叠、合起来覆盖全部维度。

跑法：
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate instinct
    python instinct/tools/test_disc_grad_index.py
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "instinct_xu" / "source" / "instinct_rl"))

from instinct_rl.algorithms.wasabi import WasabiAlgoMixin  # noqa: E402
from instinct_rl.utils.utils import get_obs_slice  # noqa: E402

N_BODIES = 13
_results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    _results.append((bool(ok), msg))
    print(f"  {'✅' if ok else '❌'} {msg}")


def make_segments(with_gravity: bool, history: int = 2) -> OrderedDict:
    """与 `AmpV20PolicyStateObsCfg` / `AmpV33PolicyStateObsCfg` 的拼接顺序逐字一致。"""
    segs = OrderedDict()
    segs["body_pos_b"] = (history, 3 * N_BODIES)
    segs["body_ori_b"] = (history, 6 * N_BODIES)
    segs["body_lin_vel_b"] = (history, 3 * N_BODIES)
    segs["body_ang_vel_b"] = (history, 3 * N_BODIES)
    if with_gravity:
        segs["anchor_gravity_b"] = (history, 3)
    return segs


def build_index(segs, dim: int):
    obj = SimpleNamespace(
        _amp_obs_segments=segs,
        _AMP_DISC_GRAD_INDEX_CACHE={},
        _AMP_TERM_SHORT=WasabiAlgoMixin._AMP_TERM_SHORT,
        _AMP_TERM_BODY_DIMS=WasabiAlgoMixin._AMP_TERM_BODY_DIMS,
        _AMP_NUM_BODIES=WasabiAlgoMixin._AMP_NUM_BODIES,
    )
    return WasabiAlgoMixin._disc_grad_index(obj, dim, torch.device("cpu"))


def main() -> int:
    print("=" * 88)
    print("单测：判别器梯度归因索引（项优先布局）")
    print("=" * 88)

    for with_gravity, expect_dim, tag in ((False, 390, "V20/V31/A-cold（无重力通道）"), (True, 396, "V33-S（含重力通道）")):
        segs = make_segments(with_gravity)
        dim = int(sum(int(torch.tensor(v).prod()) for v in segs.values()))
        print(f"\n[{tag}] 段={list(segs)}  总维度={dim}")
        check(dim == expect_dim, f"总维度 = {expect_dim}（实测 {dim}）")

        by_term, by_part = build_index(segs, dim)
        print(f"     by_term  = { {k: int(v.numel()) for k, v in by_term.items()} }")
        print(f"     by_part  = { {k: int(v.numel()) for k, v in by_part.items()} }")

        # ---- 1) 各项的列索引与 `get_obs_slice` 完全一致 ------------------------
        ok_terms = True
        for name, v in segs.items():
            sl, _ = get_obs_slice(segs, name)
            short = WasabiAlgoMixin._AMP_TERM_SHORT[name]
            idx = by_term[short].tolist()
            if idx != list(range(sl.start, sl.stop)):
                ok_terms = False
                print(f"       ✗ {name}: {idx[:5]}... vs [{sl.start},{sl.stop})")
        check(ok_terms, "每个观测项的列索引 == `get_obs_slice` 给出的区间")

        # ---- 2) by_term 覆盖且不重叠 ------------------------------------------
        all_idx = torch.cat([v for v in by_term.values()])
        check(int(all_idx.numel()) == dim and torch.equal(torch.sort(all_idx)[0], torch.arange(dim)),
              f"by_term 恰好覆盖全部 {dim} 维、无重复（实测 {int(all_idx.numel())}）")

        # ---- 3) by_part = torso/legs/arms，且只覆盖 4 个逐刚体项 --------------
        part_idx = torch.cat([v for v in by_part.values()])
        body_size = sum(int(torch.tensor(segs[k]).prod()) for k in WasabiAlgoMixin._AMP_TERM_BODY_DIMS)
        check(int(part_idx.numel()) == body_size and torch.equal(torch.sort(part_idx)[0], torch.arange(body_size)),
              f"by_part 覆盖前 {body_size} 维（4 个逐刚体项）且不重叠")
        # 逐项检查：torso(1 体) : legs(6 体) : arms(6 体) = 1 : 6 : 6（按各自的每体维度加权）
        dims = WasabiAlgoMixin._AMP_TERM_BODY_DIMS
        hist = segs["body_pos_b"][0]
        w = {k: hist * v * N_BODIES for k, v in dims.items()}
        check(int(by_part["torso"].numel()) == sum(w.values()) // N_BODIES,
              f"torso 列数 = 每帧 15 维 × {hist} 帧 = {sum(w.values()) // N_BODIES}")
        check(int(by_part["legs"].numel()) == 6 * (sum(w.values()) // N_BODIES),
              f"legs 列数 = 6 × torso = {6 * (sum(w.values()) // N_BODIES)}")
        check(int(by_part["arms"].numel()) == 6 * (sum(w.values()) // N_BODIES),
              f"arms 列数 = 6 × torso = {6 * (sum(w.values()) // N_BODIES)}")

        # ---- 4) 重力通道不进 torso/legs/arms（它不是逐刚体项）----------------
        if with_gravity:
            g = by_term["gravity"].tolist()
            check(g == list(range(390, 396)), f"重力通道 = 最后 6 维 {g}")
            check(not set(g) & set(part_idx.tolist()), "重力通道不参与 torso/legs/arms 分组")

    # ---- 5) 未知口径（老版本 1560 = 8 帧）也不能崩，且不覆盖重力 -------------
    print("\n[兼容性] 老口径 8 帧（V21/V22 的 1560 维）")
    segs8 = make_segments(False, history=8)
    dim8 = 8 * 195
    by_term8, by_part8 = build_index(segs8, dim8)
    check(sorted(by_term8) == ["angvel", "linvel", "ori", "pos"], f"by_term 键 = {sorted(by_term8)}")
    check("gravity" not in by_part8 and "gravity" not in by_term8, "老口径没有 gravity 组，且不报错")
    tot8 = torch.cat([v for v in by_term8.values()])
    check(int(tot8.numel()) == dim8, f"覆盖 {dim8} 维")

    bad = [m for ok, m in _results if not ok]
    print("\n" + "=" * 88)
    print(f"结果：{len(_results) - len(bad)}/{len(_results)} 通过")
    for m in bad:
        print(f"  ❌ {m}")
    print("=" * 88)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
