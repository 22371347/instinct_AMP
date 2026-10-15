#!/usr/bin/env python3
"""**配置快照 / 逐字段等价校验**（用于把深层继承链展平成独立配置）。

为什么要这个工具
----------------
P1 的环境配置是一条 **30+ 层**的继承链：

    amp_walk_pelvis → t2 → t1l → t1k → t1a2 → … → amp_walk → omni_loco → flat → ManagerBasedRLEnvCfg

手工"内联展开"极易漏项，而且**漏了不一定报错**（只会在训练/回放时表现为分布不对）。
本工具把**已解析的** dataclass 树递归序列化成规范化 JSON，于是"展平前后是否等价"
就变成一个可机检的断言：`diff` 为空即等价。

用法（**必须在有 IsaacLab 的环境里跑**，本机没有）
----------------------------------------------
    # 1. 导出参考快照（链式配置）
    python3 instinct/tools/dump_cfg_snapshot.py dump \
        --cls instinctlab.tasks.locomotion.config.g1:G1AmpWalkPelvisEnvCfg \
        --out /tmp/p1_chain.json

    # 2. 导出候选快照（展平后的独立配置）
    python3 instinct/tools/dump_cfg_snapshot.py dump \
        --cls instinctlab.tasks.locomotion.config.g1.amp_walk_p1_env_cfg:G1AmpWalkP1EnvCfg \
        --out /tmp/p1_flat.json

    # 3. 逐字段比对（只打印差异，退出码 = 差异数 > 0）
    python3 instinct/tools/dump_cfg_snapshot.py diff /tmp/p1_chain.json /tmp/p1_flat.json

规范化规则
----------
* dataclass            → ``{"__fields__": {名: 值, ...}}``（**字段声明序**，可复现）
* tuple / list         → ``{"__seq__": [...]}``（区分"序列"与"dataclass 字段表"）
* dict                 → 键排序的普通对象（键可能不是字符串 ⇒ 转成 ``repr``）
* 函数 / 类 / 方法      → ``"pycall:模块.限定名"``（**含 lambda**：module 为 ``__main__`` 时带源码）
* 基础类型              → 原样
* torch.Tensor/np.ndarray → ``{"__array__": [...], "shape": [...]}``（紧凑化，>256 元素折叠）
* set/frozenset        → 排序后的列表
* 其它未知对象          → ``{"__repr__": repr(v), "__type__": 类名}`` + 打一条 warning
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import importlib
import inspect
import json
import math
import sys
from typing import Any

#: 序列化时超过这个长度的数组只保留统计量（避免 JSON 里塞进 4096 个数的常量表）
_ARRAY_FOLD_LIMIT = 256

_WARNINGS: list[str] = []


def _qualname(obj: Any) -> str:
    """把可调用对象 / 类映射成稳定的字符串（跨进程可比较）。"""
    mod = getattr(obj, "__module__", None) or "?"
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", repr(obj))
    tag = f"pycall:{mod}.{name}"
    # lambda / 局部函数：把源码也带上，否则两个不同的 lambda 会看起来一样
    if name.endswith("<lambda>") or "<locals>" in name:
        try:
            tag += f"@{inspect.getsource(obj).strip()}"
        except (OSError, TypeError):
            tag += "@<source-unavailable>"
    return tag


def is_cfg_like(value: Any) -> bool:
    """是不是"鸭子类型的 cfg 对象"（有 `__dict__` 的配置对象，但不是 dataclass）。

    为什么需要：本地 `offline_stub_isaaclab.py` 下，未知的上游 cfg 是宽容鸭子对象；
    为了让"链式 vs 展平"的快照能对称比较，两边必须用**同一套字段提取规则**。

    ⚠️ **可调用对象默认不算 cfg**（与 `flatten_cfg._is_duck_cfg` 保持同步）：
    真实 IsaacLab 里 `ObsTerm.func` 可能是函数/偏函数/带 `__signature__` 的可调用对象，
    它们应当归一化成函数引用（`pycall:...`），不能被当成配置展开。
    stub 的宽容 cfg 靠 `is_stub_cfg` 标记区分。
    """
    if dataclasses.is_dataclass(value) or isinstance(value, type):
        return False
    if isinstance(value, enum.Enum):   # enum 成员也有 __dict__，但绝不是 cfg
        return False
    if not hasattr(value, "__dict__") or not isinstance(getattr(value, "__dict__", None), dict):
        return False
    if not vars(value):
        return False
    if callable(value) and not getattr(value, "is_stub_cfg", False):
        return False
    return True


def fields_of(value: Any) -> dict[str, Any]:
    """统一取字段：dataclass 用 `dataclasses.fields`，鸭子对象用 `vars()`。

    ⚠️ dataclass 实例还要**并入 `__dict__` 里的额外属性**：本地 stub 下有些基类携带
    的字段不会成为 dataclass 字段（它们以普通属性存在），不并入就会“链式有值、
    展平后丢失”，反而不对称。
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {f.name: getattr(value, f.name, None) for f in dataclasses.fields(value)}
        for k, v in vars(value).items():
            if k != "_args" and k not in out:
                out[k] = v
        return out
    return {k: v for k, v in vars(value).items() if k != "_args"}


def normalize(value: Any, _depth: int = 0, _path: str = "<root>") -> Any:
    """递归规范化。**保持可比较性**优先于可读性。"""
    if _depth > 64:
        return {"__truncated__": "max depth"}

    # ---- enum（**必须在基础类型之前**：IsaacLab 大量 `str/int` 子类枚举）---------
    if isinstance(value, enum.Enum):
        return f"enum:{type(value).__name__}.{value.name}"

    # ---- 基础类型 ----------------------------------------------------------
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value

    # ---- 类（不是实例）-----------------------------------------------------
    if isinstance(value, type):
        if dataclasses.is_dataclass(value):
            return _normalize_dataclass_class(value)
        return _qualname(value)

    # ---- slice（IsaacLab 里 `joint_ids=slice(None)` 是常见默认值）-----------
    if isinstance(value, slice):
        return {"__slice__": [value.start, value.stop, value.step]}

    # ---- 数组 --------------------------------------------------------------
    arr = _as_list(value)
    if arr is not None:
        payload, shape = arr
        if len(payload) > _ARRAY_FOLD_LIMIT:
            flat = [x for x in payload if isinstance(x, (int, float))]
            return {
                "__array__": "folded",
                "shape": shape,
                "len": len(payload),
                "min": min(flat) if flat else None,
                "max": max(flat) if flat else None,
                "sum": sum(flat) if flat else None,
                "head": _fold_head(flat, 8),
            }
        return {"__array__": payload, "shape": shape}

    # ---- dataclass 实例 ----------------------------------------------------
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _normalize_fields(value, _depth, _path)

    # ---- 鸭子类型 cfg 实例（必须在 callable 之前判断：宽容 cfg 是可调用的）----
    if is_cfg_like(value):
        return _normalize_fields(value, _depth, _path)

    # ---- 映射 --------------------------------------------------------------
    if isinstance(value, dict):
        return {
            "__dict__": {
                _key_repr(k): normalize(v, _depth + 1, f"{_path}.{_key_repr(k)}")
                for k, v in sorted(value.items(), key=lambda kv: _key_repr(kv[0]))
            }
        }

    # ---- 集合 --------------------------------------------------------------
    if isinstance(value, (set, frozenset)):
        return {"__set__": sorted(_key_repr(v) for v in value)}

    # ---- 序列 --------------------------------------------------------------
    if isinstance(value, (tuple, list)):
        return {
            "__seq__": [normalize(v, _depth + 1, f"{_path}[{i}]") for i, v in enumerate(value)]
        }

    # ---- 可调用（函数/方法）------------------------------------------------
    if callable(value):
        return _qualname(value)

    # ---- 兜底 --------------------------------------------------------------
    _WARNINGS.append(f"{_path}: 未识别的类型 {type(value).__name__} ⇒ 退化为 repr")
    return {"__repr__": repr(value), "__type__": type(value).__name__}


def _key_repr(key: Any) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, type) or callable(key):
        return _qualname(key)
    return f"{type(key).__name__}:{key!r}"


def _fold_head(flat: list, n: int) -> list:
    out = []
    for x in flat[:n]:
        out.append(round(float(x), 6) if isinstance(x, float) else x)
    return out


def _as_list(value: Any):
    """torch.Tensor / numpy.ndarray → (嵌套 list, shape)。不是数组则返回 None。"""
    mod = type(value).__module__
    if mod.startswith("torch"):
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return None
        if isinstance(value, torch.Tensor):
            t = value.detach().to("cpu")
            return t.tolist(), list(t.shape)
    if mod.startswith("numpy") or type(value).__name__ == "ndarray":
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            return None
        if isinstance(value, np.ndarray):
            return value.tolist(), list(value.shape)
    return None


def _normalize_fields(value: Any, depth: int, path: str) -> Any:
    """统一序列化：dataclass 实例与鸭子类型 cfg 实例走同一条路径（字段字典）。"""
    out = {}
    for name, v in fields_of(value).items():
        out[name] = normalize(v, depth + 1, f"{path}.{name}")
    return {"__fields__": out}


def _normalize_dataclass_class(cls: type) -> Any:
    """dataclass **类型**：记录字段名 + 默认值（用于比较"类默认值"是否一致）。"""
    out = {}
    for f in dataclasses.fields(cls):
        default = normalize(f.default) if f.default is not dataclasses.MISSING else "<MISSING>"
        if f.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
            default = f"<factory:{getattr(f.default_factory, '__qualname__', f.default_factory)}>"
        out[f.name] = default
    return {"__cls__": f"{cls.__module__}.{cls.__qualname__}", "__defaults__": out}


# ============================================================================
# 加载与比对
# ============================================================================


def load_object(spec: str) -> Any:
    """``"模块路径:名字"`` → 实例化后的对象（dataclass 直接调用，其它按需）。"""
    if ":" not in spec:
        raise SystemExit(f"--cls 需要 `模块:对象` 形式，收到：{spec!r}")
    mod_name, attr = spec.split(":", 1)
    if not mod_name.startswith("instinctlab") and not mod_name.startswith("isaaclab"):
        # 允许以文件路径形式给出（便于测我们自己的模块）
        pass
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr)
    if isinstance(obj, type) and dataclasses.is_dataclass(obj):
        return obj()
    if callable(obj) and not isinstance(obj, type):
        return obj()
    return obj


def _flatten_diff(a: Any, b: Any, path: str, out: list[str], limit: int) -> None:
    """把两棵规范化树压成"叶子路径 → 值"再比，输出人类可读的差异。"""
    if len(out) >= limit:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a), set(b)
        if set(a) != set(b):
            out.append(f"{path}: 键集不同 只A={sorted(ka - kb)[:5]} 只B={sorted(kb - ka)[:5]}")
        for k in sorted(ka & kb):
            _flatten_diff(a[k], b[k], f"{path}.{k}", out, limit)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: 长度 {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _flatten_diff(x, y, f"{path}[{i}]", out, limit)
        return
    if a != b:
        out.append(f"{path}: {_short(a)} != {_short(b)}")


def _short(v: Any, n: int = 120) -> str:
    s = json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + f"…(+{len(s) - n})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="导出配置快照 JSON")
    d.add_argument("--cls", required=True, help="`模块:类名`，如 instinctlab.tasks...:G1AmpWalkPelvisEnvCfg")
    d.add_argument("--out", required=True, help="输出 JSON 路径（`-` = stdout）")
    d.add_argument("--no-print", action="store_true", help="抑制被测配置自身 __post_init__ 的打印")

    f = sub.add_parser("diff", help="比较两份快照")
    f.add_argument("a", help="参考快照（链式）")
    f.add_argument("b", help="候选快照（展平）")
    f.add_argument("--limit", type=int, default=60, help="最多打印多少条差异")

    args = ap.parse_args()

    if args.cmd == "dump":
        if args.no_print:
            import contextlib
            import io

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                obj = load_object(args.cls)
            captured = buf.getvalue()
        else:
            obj = load_object(args.cls)
            captured = ""
        snap = normalize(obj)
        text = json.dumps(snap, ensure_ascii=False, indent=2, sort_keys=False, default=str)
        if args.out == "-":
            sys.stdout.write(text)
        else:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"[dump] {args.cls} → {args.out}（{len(text) / 1024:.1f} KB）")
        if captured:
            print("[dump] 被抑制的 __post_init__ 输出：")
            print("\n".join("    " + ln for ln in captured.strip().splitlines()))
        if _WARNINGS:
            print(f"[dump] ⚠️ {len(_WARNINGS)} 处退化为 repr：")
            for w in _WARNINGS[:10]:
                print("    " + w)
        return 0

    with open(args.a, encoding="utf-8") as fh1, open(args.b, encoding="utf-8") as fh2:
        sa, sb = json.load(fh1), json.load(fh2)
    diffs: list[str] = []
    _flatten_diff(sa, sb, "<root>", diffs, args.limit)
    if not diffs:
        print(f"[diff] ✅ 完全等价（{args.a} == {args.b}）")
        return 0
    print(f"[diff] ❌ 发现 {len(diffs)}{'+' if len(diffs) >= args.limit else ''} 处差异（上限 {args.limit}）：")
    for ln in diffs:
        print("  - " + ln)
    return min(len(diffs), 250)


if __name__ == "__main__":
    raise SystemExit(main())
