#!/usr/bin/env python3
"""系统性找出 instinctlab / instinct_rl 源码树里**所有**顶层第三方 import 并补齐。

为什么要系统性扫
----------------
逐个撞 ImportError 太慢（`pytorch_kinematics` → `lxml` → `arm_pytorch_utilities`
→ `pytorch_seed`，一轮一个）。这里一次把两棵源码树的顶层 import 全抽出来，
把**缺的**一次性装掉，并特别盯住 IsaacLab 侧的导入（Kit 相关模块名要排除，
因为不启动 Kit 时它们本来就 import 不了，那不是"缺依赖"）。

跑法（云上，普通 python 即可，不需要 Kit）：
    /opt/conda/envs/isaaclab/bin/python /root/cloud_dep_scan.py
"""
from __future__ import annotations

import ast
import importlib
import pathlib
import subprocess
import sys

ROOT = pathlib.Path("/root/gpufree-data/instinct_xu/source")
TREES = [ROOT / "instinctlab", ROOT / "instinct_rl"]

#: 不进 PyPI 的：本项目自己的包 / Kit 运行时（只在 Kit 内可用）
SKIP = {
    "instinctlab", "instinct_rl", "isaaclab", "isaacsim",
    "omni", "carb", "pxr", "usdrt", "warp", "omni_kit",
}
#: 已经是正确版本、绝不能动的
PROTECTED = {"torch", "torchvision", "numpy", "isaacsim", "isaaclab"}
#: 模块名 → PyPI 包名（不一致的那些）
PKG_ALIAS = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "git": "GitPython",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "serial": "pyserial",
    "OpenGL": "PyOpenGL",
    "google": "protobuf",
}

stdlib = set(sys.stdlib_module_names)


def collect() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for tree in TREES:
        if not tree.is_dir():
            continue
        for p in tree.rglob("*.py"):
            try:
                t = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for n in ast.walk(t):
                names: list[str] = []
                if isinstance(n, ast.Import):
                    names = [a.name for a in n.names]
                elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                    names = [n.module]
                for full in names:
                    top = full.split(".")[0]
                    if top in stdlib or top in SKIP or top.startswith("_"):
                        continue
                    found.setdefault(top, set()).add(str(p.relative_to(ROOT)))
    return found


def main() -> int:
    mods = collect()
    print(f"源码树里出现的第三方顶层模块（{len(mods)} 个）：")
    for m in sorted(mods):
        print(f"  {m:26s} ← {sorted(mods[m])[0]}")

    print("\n逐个探测导入：")
    missing: list[str] = []
    for m in sorted(mods):
        try:
            importlib.import_module(m)
            print(f"  OK   {m}")
        except ModuleNotFoundError as e:
            print(f"  MISS {m}  （缺 {e.name}）")
            missing.append(e.name.split(".")[0])
        except BaseException as e:  # noqa: BLE001
            print(f"  ERR  {m}: {type(e).__name__}: {str(e)[:70]}")

    if not missing:
        print("\n✅ 没有缺失")
        return 0

    print(f"\n需要补装（{len(missing)} 个）：{missing}")
    print("⚠️ 一律 --no-deps：不能动 Isaac Sim / Torch / NumPy")
    import torch
    import numpy

    before = (torch.__version__, numpy.__version__)
    for m in sorted(set(missing)):
        if m in PROTECTED:
            print(f"  ⛔ 跳过受保护模块 {m}")
            continue
        pkg = PKG_ALIAS.get(m, m)
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", pkg],
            capture_output=True, text=True,
        )
        print(f"  pip install --no-deps {pkg:24s} rc={r.returncode}"
              + ("" if r.returncode == 0 else f"  {(r.stderr or r.stdout)[-160:]}"))

    print("\n版本回归：")
    importlib.reload(torch)
    importlib.reload(numpy)
    after = (torch.__version__, numpy.__version__)
    print(f"  torch {before[0]} → {after[0]}   numpy {before[1]} → {after[1]}")
    if before != after:
        print("  ❌ 核心版本被改动了！需要立刻回滚")
        return 2
    print("  ✅ 核心版本未变")

    print("\n复查：")
    still: list[str] = []
    for m in sorted(mods):
        try:
            importlib.import_module(m)
        except BaseException as e:  # noqa: BLE001
            still.append(f"{m}: {type(e).__name__}: {str(e)[:60]}")
    for s in still:
        print(f"  MISS {s}")
    print("✅ 全部可导入" if not still else f"⚠️ 仍有 {len(still)} 个不可导入")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
