#!/usr/bin/env python3
"""**本地单测**：`instinctlab.utils.ckpt_manipulators:reset_amp_normalizer`（纯 torch，无需 Isaac/GPU）。

为什么要有它
------------
这一臂（V32-B）的成败完全取决于"白化统计量真的被清零了"：
`EmpiricalNormalization.update()` 是**累计平均**，`count` 在 V31 结束时已是千万级
⇒ 若没清零，新样本对 μ/σ 的影响可忽略 ⇒ 整个实验臂会退化成"什么都没改"（**假阴性**）。
所以"重置函数有没有真的动到那 4 个 buffer、并且**没有**动别的键"必须被单测钉住。

跑法：
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate instinct
    python instinct/tools/test_amp_normalizer_reset.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

# ⚠️ 不能 `import instinctlab...`：`instinctlab/__init__.py` 会连坐 import isaaclab（本地没有）。
#    本模块只依赖 `typing` +（函数内）`torch` ⇒ 直接按**文件路径**加载，绕开包 __init__。
REPO = Path(__file__).resolve().parents[2]
_MODULE_PATH = (
    REPO / "instinct_xu" / "source" / "instinctlab" / "instinctlab" / "utils" / "ckpt_manipulators.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_llp_ckpt_manipulators", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_ckpt(dim: int = 390):
    """造一个结构与真实 checkpoint 同形的 dict（`discriminator` 下含 normalizer 的 4 个 buffer）。"""
    return {
        "model_state_dict": {"actor.weight": torch.zeros(3, 3)},
        "optimizer_state_dict": {"state": {}},
        "iter": 11500,
        "discriminator": {
            "model.0.weight": torch.randn(1024, dim),
            "model.0.bias": torch.randn(1024),
            "normalizer._mean": torch.full((1, dim), 3.5),
            "normalizer._var": torch.full((1, dim), 7.0),
            "normalizer._std": torch.full((1, dim), 2.645),
            "normalizer.count": torch.full((), 12345678, dtype=torch.long),
        },
        "discriminator_optimizer": {"state": {}},
    }


def main() -> int:
    _mod = _load_module()
    reset_amp_normalizer = _mod.reset_amp_normalizer
    AMP_NORMALIZER_BUFFER_DEFAULTS = _mod.AMP_NORMALIZER_BUFFER_DEFAULTS
    print(f"被测模块：{_MODULE_PATH.relative_to(REPO)}")
    print(f"被测函数：reset_amp_normalizer")
    print(f"buffer 默认值：{AMP_NORMALIZER_BUFFER_DEFAULTS}\n")

    ok = 0
    fail = 0

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  ✅ {name} {extra}")
        else:
            fail += 1
            print(f"  ❌ {name} {extra}")

    ckpt = _fake_ckpt()
    before = {k: v.clone() for k, v in ckpt["discriminator"].items()}
    out = reset_amp_normalizer(ckpt, {"model_state_dict": {}}, verbose=True)

    check("返回的是同一个对象（原地修改）", out is ckpt)

    disc = ckpt["discriminator"]
    for leaf, default in AMP_NORMALIZER_BUFFER_DEFAULTS.items():
        key = f"normalizer.{leaf}"
        v = disc[key]
        check(
            f"{key} 已重置为 {default}",
            bool(torch.all(v == default)),
            f"（实测 {v.reshape(-1)[0].item()}）",
        )
        check(f"{key} 形状/dtype 未变", v.shape == before[key].shape and v.dtype == before[key].dtype)

    check(
        "判别器权重（model.0.weight）**未被改动**",
        torch.equal(disc["model.0.weight"], before["model.0.weight"]),
    )
    check("判别器 bias 未被改动", torch.equal(disc["model.0.bias"], before["model.0.bias"]))
    check("discriminator_optimizer 仍在", "discriminator_optimizer" in ckpt)
    check("actor 权重未被改动", torch.equal(ckpt["model_state_dict"]["actor.weight"], torch.zeros(3, 3)))
    check("iter 未被改动", ckpt["iter"] == 11500)

    # count 必须是整数型（它是 torch.long）——用 full_like 保留 dtype
    check("normalizer.count 仍是整型", disc["normalizer.count"].dtype == torch.long)

    # 没有 discriminator 键时必须是 no-op（不抛异常）
    empty = {"model_state_dict": {}}
    try:
        ret = reset_amp_normalizer(empty, {}, verbose=False)
        check("没有 discriminator 键 ⇒ no-op 且不抛异常", ret is empty)
    except Exception as exc:  # noqa: BLE001
        check("没有 discriminator 键 ⇒ no-op 且不抛异常", False, f"（抛了 {exc!r}）")

    # 归一化效果自检：μ=0/std=1 时 normalize(x)=x/(1+eps)
    print(f"\n汇总：{ok} passed, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
