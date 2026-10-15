#!/usr/bin/env python3
"""**静态追踪**某个 config 类的继承链（纯本地，不需要 IsaacLab）。

用途
----
P1 的配置是一条深链，改/删/展平它之前必须先知道"到底有哪几层、每层多大"。
本脚本只做**文本级**解析：`from .<mod> import <Base>` + `class X(<Base>)`，
因此**无 IsaacLab 也能跑**（本地 pre-flight）。

用法
----
    # 环境侧链（默认从 P1 环境配置出发）
    python3 instinct/tools/trace_cfg_chain.py G1AmpWalkPelvisEnvCfg

    # 算法侧链
    python3 instinct/tools/trace_cfg_chain.py G1AmpWalkPelvisPPORunnerCfg

    # 指定搜索根 / 只追踪到本能匹配的层
    python3 instinct/tools/trace_cfg_chain.py G1AmpWalkPelvisEnvCfg --root instinct_xu/source

输出
----
每层一行：`序号. 文件:行号  类名(<基类>)  [行数]`，
最后给出**总行数**与"层数"，用于评估展平成本。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass

#: 上游（IsaacLab / 非本项目）类名 —— 追到这些就停
UPSTREAM_PREFIXES = (
    "ManagerBasedRLEnvCfg",
    "InteractiveSceneCfg",
    "RslRlOnPolicyRunnerCfg",
    "RslRlBaseRunnerCfg",
    "InstinctRlEnv",
    "object",
)

_CLS_RE = re.compile(r"^class\s+(\w+)\s*\(([^)]*)\)\s*:", re.M)
#: ⚠️ 必须同时支持两种写法：
#:   `from .amp_walk_t2_env_cfg import G1AmpWalkT2EnvCfg`
#:   `from .amp_walk_t2_env_cfg import (\n    T2_SCAN_RAYS,\n    G1AmpWalkT2EnvCfg,\n)`
_IMPORT_RE = re.compile(
    r"^from\s+\.(\w+)\s+import\s+(?:\(([^)]*)\)|([^\n(]+))",
    re.M,
)


@dataclass
class Layer:
    cls: str
    base: str
    path: str
    line: int
    n_lines: int


def _index(root: str) -> dict[str, str]:
    """模块名 → 文件路径（只索引本项目 config 目录，避免把上游也扫进来）。"""
    idx: dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".py"):
                idx.setdefault(fn[:-3], os.path.join(dirpath, fn))
    return idx


def _find_module_index(root: str, first: str) -> str:
    """从 `first` 反查它属于哪个目录（用于确定"同包"范围）。"""
    mod, _, _attr = first.partition(":")
    for dirpath, _d, filenames in os.walk(root):
        if f"{mod}.py" in filenames:
            return dirpath
    raise SystemExit(f"在 {root} 下找不到 {mod}.py")


def _build_class_index(root: str) -> dict[str, list[tuple[str, int]]]:
    """`类名 → [(文件, 行号), ...]`。全库扫描一次后按名字跟随基类，
    比"顺着 import 猜模块"稳（基类常常来自同包另一个模块，甚至被间接 re-export）。"""
    idx: dict[str, list[tuple[str, int]]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "logs")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                text = open(p, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            for m in _CLS_RE.finditer(text):
                line = text[: m.start()].count("\n") + 1
                idx.setdefault(m.group(1), []).append((p, line))
    return idx


def trace(root: str, entry: str, max_depth: int = 80) -> list[Layer]:
    mod, _, cls_name = entry.partition(":") if ":" in entry else ("", "", entry)
    idx = _build_class_index(root)

    def _pick(name: str, near: str | None) -> tuple[str, int] | None:
        cands = idx.get(name)
        if not cands:
            return None
        if len(cands) == 1 or near is None:
            return cands[0]
        # 多个定义：优先与上一层同目录
        same = [c for c in cands if os.path.dirname(c[0]) == os.path.dirname(near)]
        chosen = (same or cands)[0]
        if len(cands) > 1:
            print(f"[warn] 类名 {name} 有 {len(cands)} 处定义，取 {chosen[0]}")
        return chosen

    start = _pick(cls_name, None)
    if start is None:
        raise SystemExit(f"全库找不到类 {cls_name}")
    cur_path, cur_line = start

    layers: list[Layer] = []
    for _ in range(max_depth):
        text = open(cur_path, encoding="utf-8").read()
        n_lines = text.count("\n") + 1
        m = None
        for mm in _CLS_RE.finditer(text):
            if mm.group(1) == cls_name and text[: mm.start()].count("\n") + 1 == cur_line:
                m = mm
                break
        if m is None:
            for mm in _CLS_RE.finditer(text):
                if mm.group(1) == cls_name:
                    m = mm
                    break
        if m is None:
            print(f"[warn] {cur_path} 里没有 class {cls_name}")
            break
        base = m.group(2).strip().split(",")[0].strip()
        layers.append(Layer(cls_name, base, cur_path, cur_line, n_lines))

        if not base or base in UPSTREAM_PREFIXES or any(base.startswith(p) for p in UPSTREAM_PREFIXES):
            layers[-1].base = f"{base}  ← 上游，终止"
            break
        nxt = _pick(base, cur_path)
        if nxt is None:
            layers[-1].base = f"{base}  ← 全库未找到定义，终止"
            break
        cur_path, cur_line = nxt
        cls_name = base
    return layers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cls", help="类名（可带 `模块:` 前缀）")
    ap.add_argument("--root", default="instinct_xu/source", help="搜索根（默认 instinct_xu/source）")
    args = ap.parse_args()

    layers = trace(args.root, args.cls)
    print(f"=== {args.cls} 的继承链（{len(layers)} 层）===")
    total = 0
    seen_files: list[str] = []
    for i, L in enumerate(layers):
        rel = os.path.relpath(L.path, args.root)
        print(f"  {i:2d}. {rel}:{L.line}  {L.cls}({L.base})  [{L.n_lines} 行]")
        if L.path not in seen_files:
            seen_files.append(L.path)
            total += L.n_lines
    layers_only = [L for L in layers if "上游" not in L.base]
    print(f"    链上各层**文件**共 {len(seen_files)} 个、合计 {total} 行；其中继承层 {len(layers_only)} 个")
    print("    （展平 = 读这 %d 个文件，把最终生效值写进 1 个文件）" % len(seen_files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
