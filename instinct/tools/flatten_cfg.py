#!/usr/bin/env python3
"""把**深层继承链上的已解析配置**生成为**一份独立、自包含的配置文件**。

为什么需要生成器（而不是手写）
------------------------------
P1 的环境配置继承链有 **43 层 / 7852 行**（`pelvis→t2→t1l→t1k→t1a2→v33s→v33→v32b→
v31→v30→v29→v27→v26→v24→v23→v21→v19d→…→v2→amp_walk→omni_loco→flat`），
算法侧还有 15+ 层。人手"内联展开"一定会漏项，而且**漏了不一定报错**（只会在训练时
表现为分布不对）。因此这里走工程化路线：

    ① 实例化链式配置（在**有 IsaacLab 的云端**）；
    ② 递归遍历**已解析**的 dataclass 树，把每个嵌套 cfg 生成成一个 `@configclass` 类，
       叶级 term（`RewTerm`/`ObsTerm`/`Event`/…）直接生成**构造调用**，字面值写死；
    ③ 输出一个 `.py`，它只依赖上游 IsaacLab（`ManagerBasedRLEnvCfg` 等），**不 import
       任何本项目其它版本的 cfg**；
    ④ 用 `dump_cfg_snapshot.py` 对"链式"与"生成物"各导一份快照，**diff 必须为空**。

即：生成器保证**完整性**，快照 diff 保证**等价性**。

用法（云端）
------------
    PY=/opt/conda/envs/isaaclab/bin/python
    cd /root/gpufree-data/instinct_xu
    $PY instinct/tools/flatten_cfg.py generate \
        --cls instinctlab.tasks.locomotion.config.g1:G1AmpWalkPelvisEnvCfg \
        --out /tmp/p1_env_gen.py --root-name P1

    # 也可以一趟生成多个（各写各的文件）：
    #   --cls A:B,C:D

生成物自检（同一步内完成，失败即说明生成器有洞）
------------------------------------------------
    $PY instinct/tools/flatten_cfg.py verify \
        --chain instinctlab.tasks.locomotion.config.g1:G1AmpWalkPelvisEnvCfg \
        --flat-file /tmp/p1_env_gen.py --flat-attr G1AmpWalkP1EnvCfg

`verify` 等价于"dump 链 → dump 生成物 → diff"，只是省去手工搬 JSON。

⚠️ **验证协议（务必按这个来，否则会得出错误结论）**
================================================================
| 场景 | 本地 `offline_stub_isaaclab.py` | 真环境（云端 IsaacLab） |
|---|---|---|
| 生成**本地**产物 + `verify` | ✅ 有效（验"展平逻辑/完整性"） | — |
| 生成**真环境**产物 + `verify` | ❌ 无效 | ✅ **唯一权威** |

两条硬规则：

1. **`generate` 必须在"目标环境"里跑**——它依赖 `sys.modules` 里已导入的真实模块，
   并用 **exec 探测**（真跑一遍 `import …` + 属性链）来挑引用写法。本地 stub 与真环境的
   可达性不同（例：`import isaaclab.sensors.ray_caster.patterns` 在真环境**不可行**，
   只能用 `from …patterns import patterns_cfg`），所以**本地生成的产物不能直接拿去训练**。
2. **不要用本地 stub 去 `verify` 真环境产物**——差异必然出现，且全是 stub 自身局限：
   * 路径字段：产物把绝对路径改写成 `os.path.join(_THIS_DIR, …)`（为换机可用），
     换机后解析出的字符串自然不同；
   * 默认值：stub 没实现的字段默认值 ≠ 真值（实测 `history_length: null != 0`）；
   * 字段/内部类：真类比 stub 多（`TerrainImporterCfg.class_type`、`AssetBaseCfg.InitialStateCfg`）。

   ⇒ 本地 stub 的正确用法只有：**生成 + 校验它自己生成的产物**（做题面逻辑回归）。
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import importlib
import importlib.util
import os
import re
import sys
from typing import Any

#: 上游基类（生成物允许继承的"底"）：命中即停止向我们的链上找
_UPSTREAM_TOP = ("ManagerBasedRLEnvCfg", "InteractiveSceneCfg", "RslRlOnPolicyRunnerCfg", "RslRlBaseRunnerCfg")

#: 上游模块前缀：属于这些的实例可以**内联用它自己的类**，其余（本项目）必须被生成的类替换
_UPSTREAM_MODULES = ("isaaclab", "rsl_rl", "mjlab", "torch", "numpy", "gymnasium")

#: 环境命名空间正则（项目里 `reference_prim_path` 原本就是这么写的）。
#:
#: ⚠️ 为什么生成物要把 `{ENV_REGEX_NS}` 写成它（云端实测踩坑）：
#: IsaacLab `configclass.copy()` 会**重新执行 `__post_init__`**
#: （`isaaclab/sensors/sensor_base.py:59` 就是 `self.cfg = cfg.copy()`），而我们的生成类把
#: `prim_path` 字面量**写死在 `__post_init__`** 里 ⇒ 运行时已被 `InteractiveScene` 替换好的
#: `/World/envs/env_.*/Robot/pelvis` 会被**退回成占位符** ⇒ `SensorBase._initialize_impl` 的
#: `cfg.prim_path.rsplit("/", 1)[0]` 得到 `{ENV_REGEX_NS}/Robot` ⇒
#: `ValueError: Prim path ... is not global`。写成绝对正则后 copy 多少次都稳定
#: （`InteractiveScene` 的 `.format()` 对不含占位符的字符串是 no-op）。
ENV_NS = "/World/envs/env_.*"


def canon_env_ns(obj: Any) -> Any:
    """把嵌套结构里的 `ENV_NS` 归一回 `{ENV_REGEX_NS}`（等价校验专用）。

    生成物用绝对正则、链式用占位符，两者语义相同 ⇒ 比较前统一成同一写法，
    否则快照 diff 会为每个 prim_path 误报一处差异。
    """
    if isinstance(obj, str):
        return obj.replace(ENV_NS, "{ENV_REGEX_NS}")
    if isinstance(obj, dict):
        return {k: canon_env_ns(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canon_env_ns(v) for v in obj]
    return obj


def is_ours(v: Any) -> bool:
    """实例是否属于**本项目**的 cfg 类（⇒ 展平后不得再 import，必须生成替身类）。

    同时覆盖两种形态：真 dataclass，以及 stub 下的**鸭子类型 cfg 实例**
    （只有 `__dict__`，没有 dataclass 元信息）。
    """
    if isinstance(v, type):
        return False
    if not (dataclasses.is_dataclass(v) or _is_duck_cfg(v)):
        return False
    return not type(v).__module__.startswith(_UPSTREAM_MODULES)


def _is_duck_cfg(v: Any) -> bool:
    """鸭子类型 cfg：有非空 `__dict__` 的配置对象（不是 dataclass、不是类）。

    本地 stub（`offline_stub_isaaclab.py`）把未知的上游 cfg 做成这种对象，
    因此展平逻辑必须同样能处理它们（否则只能处理真 dataclass，本地就验不了）。

    ⚠️ **可调用对象默认不算 cfg**：真实 IsaacLab 里 `ObsTerm.func` 可能是函数/偏函数/
    带 `__signature__` 的可调用对象，它们必须当成**函数引用**（`pycall:...`）。
    stub 的宽容 cfg 是可调用的（支持装饰器用法），靠 `is_stub_cfg` 标记区分。
    """
    if dataclasses.is_dataclass(v) or isinstance(v, type):
        return False
    if isinstance(v, enum.Enum):       # enum 成员也有 __dict__，但绝不是 cfg
        return False
    d = getattr(v, "__dict__", None)
    if not (isinstance(d, dict) and d):
        return False
    if callable(v) and not getattr(v, "is_stub_cfg", False):
        return False
    return True


def _fields_of(v: Any) -> dict[str, Any]:
    """统一取字段：dataclass 用 `dataclasses.fields`（保持声明序），鸭子对象用 `vars()`。

    ⚠️ dataclass 实例要**并入 `__dict__` 里的额外属性**（同 `dump_cfg_snapshot.fields_of`）：
    本地 stub 下基类携带的字段可能只是普通属性，不并入就会在展平时**静默丢掉**。
    """
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        out = {f.name: getattr(v, f.name, None) for f in dataclasses.fields(v)}
        for k, val in vars(v).items():
            if k != "_args" and k not in out:
                out[k] = val
        return out
    return {k: val for k, val in vars(v).items() if k != "_args"}


class Unsupported(Exception):
    """遇到无法序列化的值（如 lambda / 局部函数）——必须让用户知道，不能静默降级。"""


# ============================================================================
# 发射器
# ============================================================================


class Emitter:
    def __init__(self, decorator: str = "isaaclab.utils:configclass", rel_base: str | None = None) -> None:
        self.decorator = decorator
        #: 装饰器名（生成物里统一用它标记每个 cfg 类；可换成 dataclasses.dataclass 自测）
        self.dec_name = decorator.partition(":")[2] or "configclass"
        #: 路径字面量的"基准目录" = 生成文件所在目录。绝对路径一律改写成
        #: `os.path.join(_THIS_DIR, <相对路径>)` —— 否则生成物会写死本机路径，
        #: 换一台机器（本地 → 云端）直接不可用。
        self.rel_base = os.path.abspath(rel_base) if rel_base else None
        self.uses_this_dir = False
        #: key → 别名（保留兼容；实际 import 语句存在 `import_lines`）
        self.imports: dict[str, str] = {}
        #: key → 完整 import 语句（**含 `as` 别名**）—— 生成物里真实写出的就是这些行
        self.import_lines: dict[str, str] = {}
        #: 需要在**模块级** import 的（类基类用到的）；其余放到 `__post_init__` 局部 import
        self.base_modules: set[str] = set()
        self.extra_imports: list[str] = []    # 手工追加的 import 行
        self.blocks: list[str] = []           # 已生成的类定义（**后序**：子类在前）
        self.names: dict[str, int] = {}       # 名字去重
        self.notes: list[str] = []            # 需要人看的东西（降级/可疑项）
        self.warnings: list[str] = []         # 生成期的警告（如 ref 反查失败走了兜底）

    # ---- 名字 / import ----------------------------------------------------
    def _alias_for(self, key: str, import_line: str) -> str:
        """为一条 import 语句分配别名（按 key 去重），并记下语句本身。"""
        if key not in self.import_lines:
            self.import_lines[key] = import_line.format(alias=f"_m{len(self.import_lines)}")
        return self.import_lines[key].rsplit(" as ", 1)[-1]

    @staticmethod
    def _resolve_from(mod: Any, path: str) -> Any:
        got = mod
        for attr in path.split("."):
            got = getattr(got, attr)
        return got

    def ref(self, obj: Any, where: str, ctx: str = "value") -> str:
        """可调用对象 / 类 → 源码表达式（自动选择**安全**的 import 写法）。

        云端实测踩了三个雷，所以这里用**三段式**（短→长）：

        * **A. 公开 re-export 路径**：在定义模块的**更短前缀**里找到同一对象（如
          `UrdfFileCfg` 在 `isaaclab.sim`）⇒ `import isaaclab.sim as _m3` +
          `_m3.UrdfFileCfg`。这是项目源码的写法，最安全。
          注意：必须比定义模块**短**，否则就退化成下面的 C。
        * **B. 镜像源码的 `from 父 import 子`**：有些名字只作为**模块属性**存在
          （`patterns.GridPatternCfg`），且子模块**按点分路径 import 不进来**
          （云端实测 `import isaaclab.sensors.ray_caster.patterns` ⇒
          `cannot import name 'patterns'`）⇒ 用
          `from isaaclab.sensors.ray_caster import patterns as _m9` ✓
        * **C. 兜底**：`import <定义模块> as _mN`（语义上总是对的）。
        """
        if isinstance(obj, type) or callable(obj):
            name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
            module = getattr(obj, "__module__", None)
            if name is None or module is None:
                raise Unsupported(f"{where}: 无法取名")
            if "<locals>" in name or "<lambda>" in name:
                # 局部函数 / lambda：把源码附上，让用户自己搬成模块级函数再重生成
                src = ""
                try:
                    import inspect

                    src = inspect.getsource(obj)
                except (OSError, TypeError):
                    src = "<源码不可得>"
                raise Unsupported(
                    f"{where}: 是局部函数或 lambda，无法自动序列化。\n"
                    f"        请把它提升为**模块级函数**后重新生成。原源码：\n{src}"
                )
            # ⭐ 候选逐个**真跑一次**（exec 探测），只采纳在当前环境里真的能执行的写法。
            #    理由：IsaacLab 里大量名字只作为"属性"存在（`ray_caster.patterns`），
            #    按点分路径 `import a.b.patterns` 会失败；而环境不同（本地 stub vs
            #    真实 IsaacLab）属性可达性也不同 —— 猜都不如跑一遍。
            tried: list[str] = []
            for key, import_line, attr_path in self._candidates(obj, module, name):
                if not attr_path:
                    continue
                if self._probe(import_line, attr_path, obj):
                    if ctx == "base":
                        self.base_modules.add(key)
                    return f"{self._alias_for(key, import_line)}.{attr_path}"
                tried.append(f"{import_line.format(alias='_m')} → _m.{attr_path}")
            raise Unsupported(
                f"{where}: 找不到**可执行**的引用（{module}.{name}）。\n"
                "        试过的写法（都跑不通）：\n" + "\n".join(f"        * {t}" for t in tried)
            )
        raise Unsupported(f"{where}: 不是可调用对象（{type(obj).__name__}）")

    def _probe(self, import_line: str, attr_path: str, obj: Any) -> bool:
        """把"import 语句 + 属性链"**实际执行**一遍，看拿到的还是不是同一个对象。

        这是生成期唯一的硬判据：能在**当前环境**里跑通，生成物换到同一环境
        （同机生成 → 同机训练）就一定跑得通。
        """
        src = f"{import_line.format(alias='_probe')}\n_result = _probe.{attr_path}"
        ns: dict[str, Any] = {}
        try:
            exec(compile(src, "<ref-probe>", "exec"), ns, ns)
        except Exception:
            return False
        return ns.get("_result") is obj

    def _candidates(self, obj: Any, module: str, name: str) -> list[tuple[str, str, str]]:
        """按"越贴近源码越优先"的顺序，产出所有可能的引用写法。

        * **A. 公开 re-export 前缀**（比定义模块短）：`import isaaclab.sim as _mN`
          + `_mN.UrdfFileCfg` —— 项目源码的写法，最稳。
        * **B. 镜像源码的 `from 父 import 子`**：`from isaaclab.sensors.ray_caster
          import patterns as _mN`。
        * **D. 任意可导入前缀 + 属性链**（含 C）：`import isaaclab.sensors.ray_caster
          as _mN` + `_mN.patterns.GridPatternCfg` —— 解决"名字只作为属性存在、按点分
          路径 import 不进来"的情况（云端实测 `import ...ray_caster.patterns` ⇒
          `cannot import name 'patterns'`）。
        * **C0. 严格反查**：`obj.__module__` 不可信时（类被 re-export / 挂到别处），
          在已导入模块里严格 `is` 反查，取点分最深者。
        """
        parts = module.split(".")
        head = name.split(".")[0]
        out: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()

        def add(key: str, import_line: str, attr_path: str) -> None:
            sig = (import_line, attr_path)
            if sig not in seen:
                seen.add(sig)
                out.append((key, import_line, attr_path))

        # ---- A. 公开 re-export（比定义模块短的前缀）---------------------------
        for i in range(1, len(parts)):
            cand = ".".join(parts[:i])
            try:
                mod = importlib.import_module(cand)
            except Exception:
                break
            if head in vars(mod):
                add(f"A:{cand}", f"import {cand} as {{alias}}", name)

        # ---- B. `from <父包> import <最后一段>` -------------------------------
        if len(parts) > 1:
            parent, seg = ".".join(parts[:-1]), parts[-1]
            try:
                importlib.import_module(parent)
                add(f"B:{parent}:{seg}", f"from {parent} import {seg} as {{alias}}", name)
            except Exception:
                pass

        # ---- D. 前缀 + 属性链（**长→短**：前缀越长，属性链越短、越贴近定义处）----
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            rest = ".".join(parts[i:] + [name])
            add(f"D{i}:{prefix}", f"import {prefix} as {{alias}}", rest)

        # ---- C0. 严格反查已导入模块 -----------------------------------------
        hits: list[str] = []
        for mod_name, mod in list(sys.modules.items()):
            d = getattr(mod, "__dict__", None)
            if not d or head not in d:
                continue
            try:
                if self._resolve_from(mod, name) is obj:
                    hits.append(mod_name)
            except AttributeError:
                continue
        for hit in sorted(hits, key=lambda m: (-m.count("."), m)):
            add(f"C0:{hit}", f"import {hit} as {{alias}}", name)

        # ---- C. 兜底：按 `__module__` 原文 -----------------------------------
        add(f"C:{module}", f"import {module} as {{alias}}", name)
        return out

    def unique(self, base: str) -> str:
        base = re.sub(r"[^0-9A-Za-z_]", "_", base)
        n = self.names.get(base, 0)
        self.names[base] = n + 1
        return base if n == 0 else f"{base}_{n}"

    # ---- 值 → 源码 --------------------------------------------------------
    def value(self, v: Any, where: str, *, allow_class: bool = True) -> str:
        """把任意已解析值转成源码表达式。"""
        # ⚠️ **枚举必须排在 str/int 之前**：IsaacLab 里大量枚举是 `class X(str, Enum)` /
        #    `(int, Enum)` 子类，`isinstance(v, str)` 也成立 ⇒ 先走字符串分支就会把
        #    `<DatasetExportMode.EXPORT_ALL: 1>` 这种 repr 写进源码（语法错误！）。
        if isinstance(v, enum.Enum):
            # 枚举类本身也是"对象" ⇒ 走同一套安全引用解析；成员名直接追在后面
            return f"{self.ref(type(v), where)}.{v.name}"
        # 基础
        if v is None:
            return "None"
        if v is True or v is False:
            return "True" if v else "False"
        if isinstance(v, (int, float, str)):
            if isinstance(v, str) and "{ENV_REGEX_NS}" in v:
                # ⭐ 占位符 → **绝对正则**（项目里 `reference_prim_path` 本来就是这种写法）。
                #    详情见文件头 `ENV_NS` 处：`configclass.copy()` 会重跑 `__post_init__`，
                #    写死的占位符会把运行时已替换好的路径退回成 `{ENV_REGEX_NS}` ⇒ 真机报错。
                v = v.replace("{ENV_REGEX_NS}", ENV_NS)
                self.notes.append(f"{where}: `{{ENV_REGEX_NS}}` → `{ENV_NS}`（绝对正则）")
            if isinstance(v, str) and self.rel_base and os.path.isabs(v) and os.path.exists(v):
                # ⭐ 绝对路径 → 相对生成文件自身目录（换机可用）。
                # ⚠️ 必须加 `os.path.exists` 守卫：像 `/World/ground`、`{ENV_REGEX_NS}/...`、
                #    Nucleus 资源路径这类"虚拟路径"在本地不存在，**不能**改写
                #    （改写后与链式值不等价，快照 diff 会立刻报错）。
                #    我们真正需要迁移的（motions 目录、URDF、片单）都真实存在于仓库里。
                self.uses_this_dir = True
                rel = os.path.relpath(v, self.rel_base)
                # ⚠️ 必须 `normpath`：`os.path.join` **不折叠** `..`，于是生成物里的字符串会是
                #    `/tmp/x/../../home/...` 这种形式 —— 文件访问没问题，但与链式配置的
                #    `/home/...` **字符串不等** ⇒ 快照 diff 会误报 5 处差异。
                return f"os.path.normpath(os.path.join(_THIS_DIR, {rel!r}))"
            return repr(v)
        if v is dataclasses.MISSING:
            self.extra_imports.append("from dataclasses import MISSING")
            return "MISSING"
        if isinstance(v, slice):
            # ⚠️ 真实 IsaacLab 里 `SceneEntityCfg.joint_ids/body_ids` 的**默认值就是
            #    `slice(None)`**（stub 里我写的是 None，所以本地跑不出来）⇒ 必须支持。
            return f"slice({v.start!r}, {v.stop!r}, {v.step!r})"
        # cfg 实例（真 dataclass 或 stub 下的鸭子对象）：
        #   * **上游**类（isaaclab/rsl_rl）→ 用它自己的类内联构造（可读、语义最贴近原样）；
        #   * **本项目**类 → 必须换成生成的类（展平后不得再 import 它们）。
        if dataclasses.is_dataclass(v) and not isinstance(v, type) or _is_duck_cfg(v):
            if is_ours(v):
                return self._gen_class(v, where) + "()"
            return self._inline_ctor(v, where)
        # 序列 / 映射 / 集合
        if isinstance(v, (tuple, list)):
            inner = ", ".join(self.value(x, f"{where}[{i}]") for i, x in enumerate(v))
            if len(v) == 1:
                inner += ","
            return f"({inner})" if isinstance(v, tuple) else f"[{inner}]"
        if isinstance(v, dict):
            parts = []
            for k, val in v.items():
                parts.append(f"{self.value(k, f'{where}.key')}: {self.value(val, f'{where}[{k!r}]')}")
            return "{" + ", ".join(parts) + "}" if parts else "{}"
        if isinstance(v, (set, frozenset)):
            return "{" + ", ".join(self.value(x, where) for x in sorted(v, key=repr)) + "}"
        # 类型 / 可调用（enum 已在函数开头处理）
        if isinstance(v, type) or callable(v):
            return self.ref(v, where)
        raise Unsupported(f"{where}: 不支持的类型 {type(v).__module__}.{type(v).__name__} = {v!r}")

    @staticmethod
    def _has_nested_dataclass(v: Any) -> bool:
        for _name, x in _fields_of(v).items():
            if dataclasses.is_dataclass(x) and not isinstance(x, type):
                return True
            if isinstance(x, (list, tuple)):
                if any(dataclasses.is_dataclass(e) and not isinstance(e, type) for e in x):
                    return True
            if isinstance(x, dict):
                if any(dataclasses.is_dataclass(e) and not isinstance(e, type) for e in x.values()):
                    return True
        return False

    def _inline_ctor(self, v: Any, where: str) -> str:
        """**上游** dataclass → 内联构造调用（用它自己的类）。

        ⚠️ **绝不能**在表达式尾部加 `# 注释`：内联构造会出现在其它表达式的内部
        （`params={"sensor_cfg": SceneEntityCfg(...)}`、元组元素等），注释会吞掉
        同一行后半段，生成出语法错误的文件。
        """
        cls_ref = self.ref(type(v), where)
        parts = []
        for name, val in _fields_of(v).items():
            # ⚠️ 用默认的 allow_class=True：内部若嵌着**本项目**的 cfg，仍需生成类替换
            parts.append(f"{name}={self.value(val, f'{where}.{name}')}")
        return f"{cls_ref}({', '.join(parts)})"

    def _gen_class(self, v: Any, where: str, top: bool = False) -> str:
        """为**本项目**的嵌套 cfg 生成一个独立的 cfg 类，返回其类名（调用点需再 `()`）。

        基类规则：若原类的 MRO 里有**上游**基类（如 `ObsGroup`），则继承它 ——
        保留 IsaacLab 里可能存在的 `isinstance` 检查；否则不继承（如奖励/事件容器）。
        """
        base_name = self.unique(self._name_for(v, where, top))
        try:
            base = upstream_base(type(v))
            base_ref = self.ref(base, f"{where}.__base__", ctx="base")
        except Unsupported:
            base = None
            base_ref = None
        ann = []
        fields = _fields_of(v)
        for name, val in fields.items():
            if is_ours(val):
                # 子 cfg：注解就是它的生成类（递归里先产出，名字已知）
                ann.append(f"    {name}: Any = None  # → {self._name_for(val, f'{where}.{name}')}")
            else:
                ann.append(f"    {name}: {self._scalar_annotation(val)} = None")
        doc = (
            f'    """`{type(v).__module__}.{type(v).__qualname__}` 的展平副本'
            f'（链上位置 `{where}`；字段值 = 链式配置的最终生效值）。"""'
        )
        body = ["", doc] + ann + ["", "    def __post_init__(self):"]
        if base is not None and hasattr(base, "__post_init__"):
            body.append("        super().__post_init__()")
        for name, val in fields.items():
            expr = self.value(val, f"{where}.{name}")
            body.append(f"        self.{name} = {expr}")
        if len(ann) == 0:  # 没有任何字段（如空 curriculum / monitors）
            body.append("        pass")
        head = f"class {base_name}({base_ref}):" if base_ref else f"class {base_name}:"
        lines = [f"@{self.dec_name}", head] + body
        self.blocks.append("\n".join(lines))
        return base_name

    @staticmethod
    def _name_for(v: Any, where: str, top: bool = False) -> str:
        """生成的类名。

        * 顶层分组（`top=True`）：用**属性路径**（`rewards` → `RewardsCfg`），可读性最好；
        * 嵌套 cfg：用**原类名**（去掉 `G1`/`G1AmpWalk` 前缀，`G1AmpWalkT2RewardsCfg` →
          `T2RewardsCfg`）—— 路径名会叠成一长串（`SpawnSceneRobotSpawnCfg`），不适合当类名。

        重名由 `unique()` 加数字后缀解决，溯源靠类 docstring 里的原类全名 + 链上位置。
        """
        if top:
            tail = "".join(p.capitalize() for p in re.split(r"[._\[\]]+", where) if p.isidentifier())
            return f"{re.sub(r'[^0-9A-Za-z]', '', tail) or 'Group'}Cfg"
        orig = re.sub(r"^(G1AmpWalk|G1)", "", type(v).__name__)
        core = orig[:-3] if orig.endswith("Cfg") else orig
        return f"{re.sub(r'[^0-9A-Za-z]', '', core) or 'Cfg'}Cfg"

    @staticmethod
    def _scalar_annotation(v: Any) -> str:
        """标量字段的注解：能确定就用真类型，否则 `Any`。

        ⚠️ 展平后的类必须**声明与原类同名的字段**（哪怕值是字面量），否则
        `dataclasses.fields()` 看到的结构与原类不同，快照 diff 会失败。
        """
        return {bool: "bool", int: "int", float: "float", str: "str"}.get(type(v), "Any")

    # ---- 输出 -------------------------------------------------------------
    def _inject_local_imports(self, block: str) -> str:
        """把该块里用到的**非基类** import，以**局部 import** 形式插到 `__post_init__` 开头。

        只插这个块**真正用到**的别名（按文本里出现的 `_mN` 反查），避免每个类都重复
        一大堆 import。
        """
        used = set(re.findall(r"\b_m\d+\b", block))
        alias_to_key = {line.rsplit(" as ", 1)[-1]: key for key, line in self.import_lines.items()}
        keys = [alias_to_key[a] for a in sorted(used) if a in alias_to_key and alias_to_key[a] not in self.base_modules]
        marker = "    def __post_init__(self):"
        if not keys or marker not in block:
            return block
        inject = "\n".join(
            f"        {self.import_lines[k]}  # 局部 import：避开导入期循环依赖" for k in keys
        )
        # ⚠️ 必须**每一处** `__post_init__` 都插：`top_code` 里可能塞了多个类
        #    （env + `_PLAY`），只插第一处的话第二个类就会 `NameError: _mN`。
        return block.replace(marker, f"{marker}\n{inject}")

    def render(self, header: str, top_code: str) -> str:
        dec_mod, _, dec_name = self.decorator.partition(":")
        lines = [header, "", "from __future__ import annotations", ""]
        lines.append("from typing import Any")
        if self.uses_this_dir:
            lines += [
                "import os",
                "",
                "#: 本文件所在目录（路径字面量都以它为基准，保证生成物换机可用）",
                "_THIS_DIR = os.path.dirname(os.path.abspath(__file__))",
            ]
        lines += self.extra_imports
        # ⚠️ 模块级只 import **类基类**用到的；字段值用到的改成 `__post_init__` 局部 import
        for key in sorted(self.base_modules, key=lambda k: self.import_lines[k]):
            lines.append(self.import_lines[key])
        lines.append(f"from {dec_mod} import {dec_name}")
        for b in self.blocks:
            lines += ["", "", self._inject_local_imports(b)]
        lines += ["", "", self._inject_local_imports(top_code), ""]
        return "\n".join(lines)


# ============================================================================
# 顶层生成
# ============================================================================


def upstream_base(cls: type, explicit: str | None = None) -> type:
    """找"上游"基类：MRO 里第一个非本项目的 cfg 基类。

    `explicit`（`模块:类名`）可强制指定，用于自测 / 上游基类不在 isaaclab 的情况。
    """
    if explicit:
        mod, _, attr = explicit.partition(":")
        return getattr(importlib.import_module(mod), attr)
    for b in cls.__mro__[1:]:
        if b is object:
            break
        if b.__module__.startswith(_UPSTREAM_MODULES):
            return b
    raise Unsupported(
        f"{cls.__qualname__} 的 MRO 里找不到上游基类（MRO={[b.__qualname__ for b in cls.__mro__]}）；"
        f"可用 `--base 模块:类名` 显式指定"
    )


def generate(cls_spec: str, root_name: str, decorator: str, explicit_base: str | None = None,
             rel_base: str | None = None, targets: list[tuple[str, str]] | None = None) -> str:
    """生成展平文件。

    * 单类：`generate("模块:类", "根类名", ...)`；
    * **多类同文件**：`targets=[("模块:类", "根类名"), ...]`（P1 环境侧要 env + `_PLAY` 同一文件，
      沿用仓库惯例）。多个目标共享同一个 Emitter ⇒ 公共子 cfg 只定义一次、
      重的自动加数字后缀，不会冲突。
    """
    targets = targets or [(cls_spec, root_name)]
    first = getattr(importlib.import_module(targets[0][0].partition(":")[0]), targets[0][0].partition(":")[2])
    base = upstream_base(first, explicit_base)
    em = Emitter(decorator=decorator, rel_base=rel_base)
    base_ref = em.ref(base, f"{targets[0][0]}.base", ctx="base")
    blocks: list[str] = []
    for spec, name in targets:
        blocks.append(_generate_one(em, spec, name, base_ref, base))
    header = _header(targets, len(em.blocks), base)
    if em.notes:
        print("[generate] ℹ️ 备注：")
        for n in dict.fromkeys(em.notes):
            print(f"          {n}")
    if em.warnings:
        # ⚠️ 生成期警告必须显式打出来：历史上"生成成功但云端运行炸"的地雷
        #    全都是这里悄悄吞掉的信息。
        print(f"[generate] ⚠️ 警告 {len(em.warnings)} 条：")
        for w in dict.fromkeys(em.warnings):
            print(f"          {w}")
    return em.render(header, "\n\n\n".join(blocks))


def _header(targets: list[tuple[str, str]], n_classes: int, base: type) -> str:
    lines = [
        '"""⭐ **自动生成**，不要手改结构 —— 只允许添加注释/文档。',
        "",
        "来源（继承链的**解析结果**）：",
        *[f"  * `{spec}` → `{name}`" for spec, name in targets],
        f"上游基类：`{base.__module__}.{base.__qualname__}`；另有 {n_classes} 个自动生成的子 cfg 类。",
        "",
        "重新生成（在有 IsaacLab 的机器上）：",
        "    python3 instinct/tools/flatten_cfg.py generate \\",
        f"        --cls {','.join(s for s, _ in targets)} \\",
        "        --out <本文件> --root-name <根类名>",
        "等价性校验（必须 diff 为空）：",
        "    python3 instinct/tools/flatten_cfg.py verify \\",
        f"        --chain {targets[0][0]} --flat-file <本文件> --flat-attr {targets[0][1]}",
        '"""',
    ]
    return "\n".join(lines)


def _generate_one(em: "Emitter", cls_spec: str, root_name: str, base_ref: str, base: type) -> str:
    mod_name, _, attr = cls_spec.partition(":")
    obs_cls = getattr(importlib.import_module(mod_name), attr)
    obj = obs_cls()                      # ← 链上所有 __post_init__ 在此依次生效

    # 顶层类：所有 dataclass 字段在 __post_init__ 里**字面重建**；
    # 标量字段（decimation/run_name/…）同样写死，保证不受上游 __post_init__ 副作用影响。
    fields = list(_fields_of(obj).items())
    group_fields, scalars = [], []
    for name, v in fields:
        if dataclasses.is_dataclass(v) and not isinstance(v, type) or _is_duck_cfg(v):
            group_fields.append((name, v))
        else:
            scalars.append((name, v))

    ann_lines, post_lines = [], []
    for name, v in group_fields:
        class_name = em._gen_class(v, f"{name}", top=True)   # 顶层字段一律生成独立类
        ann_lines.append(f"    {name}: Any = None  # → {class_name}")
        post_lines.append(f"        self.{name} = {class_name}()")
    for name, v in scalars:
        # 标量/字典：直接写值；dict 里可能含 dataclass（如 actuators）
        expr = em.value(v, name)
        if "\n" in expr:
            raise Unsupported(f"顶层字段 {name} 的表达式跨行，需要人工处理")
        ann_lines.append(f"    {name}: {em._scalar_annotation(v)} = None")
        post_lines.append(f"        self.{name} = {expr}")

    anchor = []
    if hasattr(obj, "run_name"):
        anchor = [
            "",
            "        # ---- 展平自检锚点：run_name（链式 vs 展平必须一致，diff 会抓）----",
            f"        self.run_name = {obj.run_name!r}",
        ]

    top_code = "\n".join(
        [
            f"@{em.dec_name}",
            f"class {root_name}({base_ref.split('  #')[0]}):",
            f'    """`{mod_name}:{attr}` 的**展平**版本：不继承本项目任何其它配置类。',
            "",
            f"    由 `instinct/tools/flatten_cfg.py` 从链式配置的**解析结果**生成，"
            f"上游基类 = `{base.__module__}.{base.__qualname__}` (链长 {len(obs_cls.__mro__)} 层)。",
            '    """',
            "",
            *ann_lines,
            "",
            "    def __post_init__(self):",
            f"        super().__post_init__()  # 只跑上游（{base.__qualname__}）的钩子",
            '        """全部字段按链式配置的**最终生效值**写死。"""',
            *post_lines,
            *anchor,
        ]
    )
    return top_code


# ============================================================================
# 校验：dump 链式 vs dump 生成物 → diff
# ============================================================================


def _load_flat(flat_file: str, attr: str) -> Any:
    path = os.path.abspath(flat_file)
    spec = importlib.util.spec_from_file_location("_flat_generated", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_flat_generated"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, attr)()


def verify(chain_spec: str, flat_file: str, flat_attr: str, limit: int) -> int:
    from dump_cfg_snapshot import normalize, _flatten_diff  # 同目录复用，避免第二份实现

    mod_name, _, attr = chain_spec.partition(":")
    chain_obj = getattr(importlib.import_module(mod_name), attr)()
    flat_obj = _load_flat(flat_file, flat_attr)

    a, b = normalize(chain_obj), normalize(flat_obj)
    # ⭐ 等价规范化：生成物把 `{ENV_REGEX_NS}` 写成了绝对正则 `ENV_NS`（见 `Emitter.value`），
    #    两者语义相同 ⇒ 比较前统一写法，否则每个 prim_path 都会误报一处差异。
    a, b = canon_env_ns(a), canon_env_ns(b)
    diffs: list[str] = []
    _flatten_diff(a, b, "<root>", diffs, limit)
    if not diffs:
        print(f"[verify] ✅ 展平等价：{chain_spec} == {flat_file}:{flat_attr}")
        return 0
    print(f"[verify] ❌ {len(diffs)} 处差异：")
    for d in diffs:
        print("  - " + d)
    return min(len(diffs), 250)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--cls", required=True, help="`模块:类名`（多个用逗号分隔）")
    g.add_argument("--out", required=True, help="输出文件；多个 --cls 时按 `_<类名>` 后缀区分")
    g.add_argument("--root-name", default="P1EnvCfg")
    g.add_argument("--decorator", default="isaaclab.utils:configclass")
    g.add_argument("--base", default=None, help="强制指定上游基类 `模块:类名`（默认自动找 isaaclab/rsl_rl）")
    g.add_argument("--no-app", action="store_true", help="不启 AppLauncher（仅纯 dataclass 时需要）")
    g.add_argument("--dry-run", action="store_true", help="只打印，不写文件")

    v = sub.add_parser("verify")
    v.add_argument("--chain", required=True)
    v.add_argument("--flat-file", required=True)
    v.add_argument("--flat-attr", required=True)
    v.add_argument("--limit", type=int, default=60)
    v.add_argument("--no-app", action="store_true")

    args = ap.parse_args()

    if args.cmd == "generate" and not args.no_app:
        # ⚠️ 必须早于任何 instinctlab / isaaclab.envs 导入
        from isaaclab.app import AppLauncher

        AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False})
    if args.cmd == "verify" and not args.no_app:
        from isaaclab.app import AppLauncher

        AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False})

    if args.cmd == "generate":
        specs = [s.strip() for s in args.cls.split(",") if s.strip()]
        names = [n.strip() for n in args.root_name.split(",") if n.strip()]
        rel_base = os.path.dirname(os.path.abspath(args.out))
        rc = 0
        if len(names) == len(specs) and len(specs) > 1:
            # ⭐ 多个类写进**同一个文件**（仓库惯例：env + `_PLAY` 同模块）
            try:
                code = generate(specs[0], names[0], args.decorator, args.base,
                                rel_base=rel_base, targets=list(zip(specs, names)))
            except Unsupported as e:
                print(f"[generate] ❌ {args.cls}: {e}")
                return 1
            if args.dry_run:
                sys.stdout.write(code)
            else:
                with open(args.out, "w", encoding="utf-8") as fh:
                    fh.write(code)
                print(f"[generate] ✅ {len(specs)} 个类 → {args.out}"
                      f"（{len(code.splitlines())} 行：{', '.join(names)}）")
            return 0

        outs = [args.out] if len(specs) == 1 else [
            re.sub(r"\.py$", f"_{s.partition(':')[2]}.py", args.out) for s in specs
        ]
        for spec, out in zip(specs, outs):
            name = names[0] if len(names) == 1 else f"{names[0]}{spec.partition(':')[2]}"
            try:
                code = generate(spec, name, args.decorator, args.base,
                                rel_base=os.path.dirname(os.path.abspath(out)))
            except Unsupported as e:
                print(f"[generate] ❌ {spec}: {e}")
                rc = 1
                continue
            if args.dry_run:
                sys.stdout.write(code)
            else:
                with open(out, "w", encoding="utf-8") as fh:
                    fh.write(code)
                print(f"[generate] ✅ {spec} → {out}（{len(code.splitlines())} 行，类 {name}）")
        return rc

    return verify(args.chain, args.flat_file, args.flat_attr, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
