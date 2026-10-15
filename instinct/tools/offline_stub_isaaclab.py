#!/usr/bin/env python3
"""**本地无 IsaacLab** 时用的宽容 stub —— 让 43 层配置链在纯 CPU 机器上也能实例化。

为什么可行
----------
需要验证的对象是**配置**（全是 dataclass 构造），不涉及 PhysX / 渲染 / GPU。
真正需要的 isaaclab API 面很小：**11 个模块、24 个名字**（见 `survey` 输出）。
所以本地装一层"宽容 stub"就能把整条链跑起来，从而：

* 把生成器（`flatten_cfg.py`）在**真实的 43 层链**上跑通，暴露全部结构性问题；
* 做**本地等价校验**（链 vs 展平物，两边都在 stub 下 ⇒ 字段集合对称）——
  这覆盖了"我自己的展平逻辑对不对"这块主要风险。

⚠️ **本地校验的局限（必须在云端复核）**：stub 里"未知类"是宽容鸭子类型，
其字段 = **我们代码显式传的那些**；真实 isaaclab 类可能还有别的字段/默认值。
因此本地 diff 为空 ⇒ "展平逻辑正确"；**不能**替代云端用真 isaaclab 跑的那次
`flatten_cfg.py verify`（那次才证明"上游形状假设"也成立）。

用法（作为运行器）
----------------
    python3 instinct/tools/offline_stub_isaaclab.py <目标脚本.py> <目标脚本的参数...>

例：
    export PYTHONPATH=$PWD/instinct_xu/source/instinctlab:$PYTHONPATH
    python3 instinct/tools/offline_stub_isaaclab.py instinct/tools/flatten_cfg.py generate \
        --cls instinctlab.tasks.locomotion.config.g1:G1AmpWalkPelvisEnvCfg \
        --out /tmp/p1_env_gen.py --root-name G1AmpWalkP1EnvCfg --no-app

也可作为库：`from offline_stub_isaaclab import install; install()`
"""

from __future__ import annotations

import copy
import abc
import dataclasses
import dataclasses as _dc
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import types
from dataclasses import MISSING, field
from typing import Any


def dataclass(cls=None, **kw):
    """本 stub 内部的 cfg 一律 `eq=False`。

    ⚠️ 必须：`dataclasses` 禁止"不可哈希的默认值"，而 cfg 实例作为默认值是常态。
    `eq=True`（默认）会把 `__hash__` 置为 None ⇒ 只要某个 cfg 实例被当作另一个
    cfg 的字段默认值，就会报 `ValueError: mutable default ... is not allowed`。
    IsaacLab 的 `configclass` 同样是 `eq=False`（实例按 id 哈希）。
    """
    kw.setdefault("eq", False)
    if cls is None:
        return lambda c: _dc.dataclass(c, **kw)
    return _dc.dataclass(cls, **kw)

#: 需要 stub 的顶层包
#: ⚠️ 必须包含 Omniverse/Kit 家族（`omni`/`carb`/`pxr`…）：`instinctlab` 的 envs/mdp
#: 一类的模块在 import 期就会 `import omni.log`、`from pxr import …`，本地没有这些。
#: 我们是**纯配置**验证，不跑仿真，所以替成宽容对象没有副作用。
_PREFIXES = (
    "isaaclab",
    "isaaclab_assets",
    "omni",
    "carb",
    "pxr",
    "usdrt",
    "warp",
    "isaacsim",
    # 本地没装、但链上 import 期就要用到的三方包（**只在本机确实缺失时**才 stub）
    "prettytable",
    "tensorboard",
    "gymnasium",
    "rsl_rl",
    "pytorch_kinematics",
    "trimesh",
    "h5py",
    "tqdm",
    "einops",
    "omegaconf",
    "hydra",
    "matplotlib",
    "pandas",
    "imageio",
    "torchvision",
    "cv2",
    "sklearn",
    "rich",
    "tabulate",
    "plotly",
)

#: 已缓存的动态对象（同名重复访问返回**同一个**对象，保证一致性）
_DYNAMIC: dict[tuple[str, str], Any] = {}


# ============================================================================
# 宽容基础对象
# ============================================================================


class _Replace:
    """描述符：让 `.replace(...)` 在**实例**与**类**上都成立。

    * 实例上：返回字段被覆盖的**副本**（对应真 isaaclab 的 `cfg.replace(...)`）；
    * 类上 （如模块级单例 `FRAME_MARKER_CFG`）：返回该类的一个**实例**。

    ⚠️ 不能只写在元类里：类属性查找会先在类自身的 MRO 里找到 `PermissiveCfg.replace`
    这个**未绑定的普通方法**，于是 `SomeCfg.replace(...)` 报
    `missing 1 required positional argument: 'self'`。用描述符才能两边都盖住。
    """

    def __get__(self, obj: Any, objtype: type | None = None):
        if obj is None:

            def _class_replace(**kw):
                return objtype(**kw)

            return _class_replace

        def _inst_replace(**kw):
            new = copy.copy(obj)
            new.__dict__ = dict(obj.__dict__)
            new.__dict__.update(kw)
            return new

        return _inst_replace


class _Copy:
    """描述符：`.copy()` 在实例上返回副本，在类上返回实例（同 `_Replace` 的道理）。"""

    def __get__(self, obj: Any, objtype: type | None = None):
        if obj is None:
            return lambda: objtype()

        def _inst_copy():
            new = copy.copy(obj)
            new.__dict__ = dict(obj.__dict__)
            return new

        return _inst_copy


class PermissiveCfg:
    """鸭子类型 cfg：接受任意位置/关键字参数，记录到 `__dict__`。

    记录方式与 `dataclasses.fields` **同构**（`_args` 单独存），这样
    `dump_cfg_snapshot.py` / `flatten_cfg.py` 能用同一套逻辑处理"真 dataclass"
    与"stub 鸭子对象"。
    """

    #: 真 isaaclab 的 cfg 都有 `.replace(...)` / `.copy()`，链上代码会用到
    replace = _Replace()
    copy = _Copy()

    #: 标记：告诉 `flatten_cfg.py` / `dump_cfg_snapshot.py` "我是 **cfg**（哪怕我可调用）"。
    #: 真实世界里 cfg 实例**不可调用**，而函数/偏函数可调用 ⇒ 没这个标记就会被误当函数引用。
    is_stub_cfg = True

    def __post_init__(self) -> None:
        """no-op ：我们的 config 类常写 `super().__post_init__()`，基类必须存在这个方法
        （否则报 `'super' object has no attribute '__post_init__'`）。"""
        return None

    def __init__(self, *args, **kwargs):
        if args:
            self.__dict__["_args"] = list(args)
        self.__dict__.update(kwargs)

    def __call__(self, *args, **kwargs):
        """让宽容实例也可调用 —— 为了兼容**装饰器**用法。

        ⚠️ 现实例子：`@wp.kernel(enable_backward=False)`（warp 内核）、
        `@torch.jit.script` 之类。`wp.kernel(...)` 在 stub 下是个宽容实例，紧接着还要
        被当成装饰器调用一次；装饰器语义就是"返回被装饰对象"，所以这里对"单个可调用
        参数"直接原样返回。
        """
        if len(args) == 1 and not kwargs and (callable(args[0]) or isinstance(args[0], type)):
            return args[0]
        return PermissiveCfg(*args, **kwargs)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "_args"}

    def __repr__(self) -> str:
        items = list(self.__dict__.items())[:4]
        inner = ", ".join(f"{k}={v!r}" for k, v in items)
        more = ", …" if len(self.__dict__) > 4 else ""
        return f"{type(self).__name__}({inner}{more})"


class _HybridMeta(abc.ABCMeta):
    """让"未知名字"既能当函数用、又能当类用、还能取嵌套属性。

    * `mdp.joint_pos_limits`      → 作为 `func=` 传引用（对象本身）
    * `mdp.UniformVelocityCommandCfg(...)` → 实例化
    * `mdp.UniformVelocityCommandCfg.Ranges(...)` → 嵌套类

    ⚠️ 必须继承 `abc.ABCMeta`：IsaacLab 里不少类是 ABC（`ActionTerm` 等），
    如果本元类只继承 `type`，那些类一旦和 hybrid 类做多重继承就会报
    `metaclass conflict`。
    """

    def __getattr__(cls, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        key = (cls.__module__, f"{cls.__qualname__}.{name}")
        if key not in _DYNAMIC:
            _DYNAMIC[key] = _make_hybrid(name, cls.__module__, qualname=f"{cls.__qualname__}.{name}")
        return _DYNAMIC[key]

    def replace(cls, **kwargs) -> Any:  # noqa: ARG003 - 兜底；实际由 _Replace 描述符接管
        """兜底：理论上到不了这里（`_Replace` 描述符优先级更高）。"""
        return cls(**kwargs)


def _make_hybrid(name: str, module: str, qualname: str | None = None) -> type:
    return _HybridMeta(
        name,
        (PermissiveCfg,),
        {"__module__": module, "__qualname__": qualname or name},
    )


class AutoAttrs(PermissiveCfg):
    """**自动生长**子属性的宽容对象。

    用途：像 `isaaclab_assets.G1_CFG` 这种"基线机器人配置"，链上会一路就地改写
    （`G1_CFG.copy()` → `cfg.spawn.asset_path = ...` → `init_state.pos = ...`），
    以及 `spawn.joint_drive.gains.stiffness = None` 这种**多层写入**。
    真对象有完整字段树，stub 没有 ⇒ 这里让任何缺失属性都按需生成一个同级对象，
    写进去之后就一直存在。
    """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        child = AutoAttrs()
        setattr(self, name, child)
        return child


#: 这些模块里"未知名字"默认做成**自动生长**的对象：链上会往里面多层写字段
#: （`spawn.joint_drive.gains.stiffness = ...`、`init_state.pos = ...`）。
_AUTO_PREFIXES = (
    "isaaclab.sim",
    "isaaclab.sim.schemas",
    "isaaclab.sim.spawners",
)


# ============================================================================
# 官方口径的真 dataclass（这些**必须**是真 dataclass：链上会 dataclasses.replace）
# ============================================================================


@dataclass
class RewardTermCfg:
    """对应 isaaclab.managers.RewardTermCfg。"""

    func: Any = None
    weight: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass
class ObservationTermCfg:
    func: Any = None
    params: dict = field(default_factory=dict)
    noise: Any = None
    clip: Any = None
    scale: Any = None
    history_length: int = None
    flatten_history_dim: bool = True


@dataclass
class EventTermCfg:
    func: Any = None
    mode: str = "startup"
    params: dict = field(default_factory=dict)
    interval_range_s: tuple = None
    is_global: bool = False


@dataclass
class TerminationTermCfg:
    func: Any = None
    params: dict = field(default_factory=dict)
    time_out: bool = False


@dataclass
class CurriculumTermCfg:
    func: Any = None
    params: dict = field(default_factory=dict)


@dataclass
class MonitorTermCfg:
    func: Any = None
    params: dict = field(default_factory=dict)


@dataclass
class SceneEntityCfg(metaclass=_HybridMeta):
    """⚠️ `joint_ids` / `body_ids` 的默认值必须是 `slice(None)` —— 与真实 IsaacLab 对齐。

    写成 `None` 会让本地生成的展平文件与云端真实值不同（云端实测会报
    `joint_ids: 不支持的类型 builtins.slice`）。
    """

    name: str = None
    joint_names: Any = None
    body_names: Any = None
    # ⚠️ 必须用 `default_factory`：`slice` **不可哈希**，直接当默认值会被 `dataclasses`
    #    拒绝（`ValueError: mutable default <class 'slice'> ...`）。
    joint_ids: Any = field(default_factory=lambda: slice(None))
    body_ids: Any = field(default_factory=lambda: slice(None))
    preserve_order: bool = False


@dataclass
class AdditiveUniformNoiseCfg:
    n_min: float = 0.0
    n_max: float = 0.0
    operation: str = "add"


@dataclass
class ObservationGroupCfg:
    concatenate_terms: bool = True
    enable_corruption: bool = False
    history_length: int = None
    history_ordering: str = "time"
    scale: Any = None


@dataclass
class InteractiveSceneCfg:
    num_envs: int = 1
    env_spacing: float = 2.0
    replicate_physics: bool = True
    clone_in_fabric: bool = False
    lazy_sync: bool = False


@dataclass
class SimulationCfg(metaclass=_HybridMeta):
    dt: float = 0.005
    render_interval: int = 4
    device: str = "cuda:0"
    gravity: tuple = (0.0, 0.0, -9.81)
    physics_material: Any = None
    render: Any = None
    diagnostics: Any = None


@dataclass
class ViewerCfg:
    eye: tuple = (7.5, 7.5, 7.5)
    lookat: tuple = (0.0, 0.0, 0.0)
    origin_type: str = "world"
    env_index: int = 0
    asset_name: str = None
    resolution: tuple = (1280, 720)


@dataclass
class _InitialStateCfg(metaclass=_HybridMeta):
    pos: tuple = (0.0, 0.0, 0.0)
    rot: tuple = (1.0, 0.0, 0.0, 0.0)
    lin_vel: tuple = (0.0, 0.0, 0.0)
    ang_vel: tuple = (0.0, 0.0, 0.0)
    joint_pos: dict = field(default_factory=dict)
    joint_vel: dict = field(default_factory=dict)


@dataclass
class ArticulationCfg(metaclass=_HybridMeta):
    prim_path: str = None
    spawn: Any = None
    init_state: Any = None
    actuators: Any = None
    soft_joint_pos_limit_factor: float = 1.0

    InitialStateCfg = _InitialStateCfg


@dataclass
class AssetBaseCfg(metaclass=_HybridMeta):
    prim_path: str = None
    spawn: Any = None
    init_state: Any = None

    # ⚠️ 用 `_HybridMeta`：真实 IsaacLab 的 `AssetBaseCfg` 还有 `InitialStateCfg`
    #    这类**内部类**字段类型（`AssetBaseCfg.InitialStateCfg(pos=…)`），
    #    stub 里没有 ⇒ 靠元类的 `__getattr__` 按需生长，避免本地校验**云端产物**时
    #    报 `AttributeError: type object 'AssetBaseCfg' has no attribute 'InitialStateCfg'`。


@dataclass
class ContactSensorCfg(metaclass=_HybridMeta):
    prim_path: str = None
    history_length: int = 1
    track_air_time: bool = False
    update_period: float = 0.0
    debug_vis: bool = False


@dataclass
class RayCasterCfg(metaclass=_HybridMeta):
    """⚠️ `metaclass=_HybridMeta`：真 isaaclab 的这些 cfg 有多个**嵌套 cfg 类**
    （`OffsetCfg` / `RaycastTargetCfg` / `Ranges` …），链上会写
    `MultiMeshRayCasterCfg.RaycastTargetCfg` 这种注解。挂在元类上就能自动兜住
    任意嵌套类名，不用一个个手写（且子类自动继承）。
    """

    @dataclass
    class OffsetCfg:
        pos: tuple = (0.0, 0.0, 0.0)
        rot: tuple = (1.0, 0.0, 0.0, 0.0)

    prim_path: str = None
    update_period: float = 0.0
    offset: Any = None
    ray_alignment: str = "base"
    pattern_cfg: Any = None
    debug_vis: bool = False


@dataclass
class MultiMeshRayCasterCfg(RayCasterCfg):
    mesh_prim_paths: Any = None


@dataclass
class GridPatternCfg:
    resolution: float = 0.1
    size: tuple = (1.0, 1.0)


@dataclass(init=False)
class TerrainImporterCfg(metaclass=_HybridMeta):
    prim_path: str = None
    terrain_type: str = "plane"
    collision_group: int = -1
    physics_material: Any = None
    terrain_generator: Any = None
    debug_vis: bool = False
    #: 真实 IsaacLab 里有这个字段，stub 早期版本漏了 ⇒ 校验**云端产物**时会报
    #: `unexpected keyword argument 'class_type'`（假报错，不是产物的问题）。
    class_type: Any = None

    def __init__(self, *args, **kwargs):
        # 宽容初始化：字段默认值先全部落进实例，再吸收任意 kwargs
        # （真实 cfg 就是"字段 + 关键字构造"，这样 dump 出来的结构才和真环境一致）
        for f in dataclasses.fields(type(self)):
            self.__dict__[f.name] = getattr(type(self), f.name, None)
        PermissiveCfg.__init__(self, *args, **kwargs)


@dataclass
class ManagerBasedRLEnvCfg(metaclass=_HybridMeta):
    """上游基类的"够用"版本：链上会读/写这些字段。"""

    decimation: int = 1
    episode_length_s: float = 10.0
    sim: SimulationCfg = field(default_factory=SimulationCfg)
    scene: Any = None
    observations: Any = None
    actions: Any = None
    rewards: Any = None
    terminations: Any = None
    events: Any = None
    curriculum: Any = None
    commands: Any = None
    monitors: Any = None
    viewer: ViewerCfg = field(default_factory=ViewerCfg)
    run_name: str = ""


#: `(模块, 名字) → 对象`；模块为 `None` 表示"任何模块里都叫这个名字"
_KNOWN: dict[tuple[str | None, str], Any] = {}


def _register() -> None:
    def put(module: str, **names: Any) -> None:
        for n, v in names.items():
            if isinstance(v, type):
                # ⭐ 必须把 `__module__` 改成**真实的 isaaclab 归属**：展平器靠它
                #   ①识别"上游基类"（生成物要 `class X(ManagerBasedRLEnvCfg)`）
                #   ②生成 `import isaaclab.envs as _m0` 这样的引用表达式。
                #   否则它们看起来像"本 stub 模块里的类"，生成物会引用不到。
                v.__module__ = module
                v.__qualname__ = n
            _KNOWN[(module, n)] = v
            _KNOWN[(None, n)] = v

    # ---- 术语/小组件（真 dataclass）----
    put(
        "isaaclab.managers",
        RewardTermCfg=RewardTermCfg,
        ObservationTermCfg=ObservationTermCfg,
        EventTermCfg=EventTermCfg,
        TerminationTermCfg=TerminationTermCfg,
        CurriculumTermCfg=CurriculumTermCfg,
        SceneEntityCfg=SceneEntityCfg,
        ObservationGroupCfg=ObservationGroupCfg,
    )
    put("isaaclab.utils.noise", AdditiveUniformNoiseCfg=AdditiveUniformNoiseCfg)
    put("isaaclab.scene", InteractiveSceneCfg=InteractiveSceneCfg)
    put("isaaclab.envs", ManagerBasedRLEnvCfg=ManagerBasedRLEnvCfg, ViewerCfg=ViewerCfg)
    put("isaaclab.envs.mdp", ViewerCfg=ViewerCfg)  # 保险：有的版本从这里也拿得到
    put(
        "isaaclab.assets",
        ArticulationCfg=ArticulationCfg,
        AssetBaseCfg=AssetBaseCfg,
    )
    put("isaaclab.assets.articulation", ArticulationCfg=ArticulationCfg)
    put("isaaclab.sensors", ContactSensorCfg=ContactSensorCfg, RayCasterCfg=RayCasterCfg)
    put("isaaclab.sensors.ray_caster", MultiMeshRayCasterCfg=MultiMeshRayCasterCfg)
    put("isaaclab.sensors.ray_caster.patterns", GridPatternCfg=GridPatternCfg)
    put("isaaclab.terrains", TerrainImporterCfg=TerrainImporterCfg)
    put("isaaclab.sim", SimulationCfg=SimulationCfg)
    put("isaaclab.utils.assets", ISAAC_NUCLEUS_DIR="/stub/Isaac/Nucleus")
    # ⚠️ `G1_CFG` 必须是**实例**（链上会 `.copy()` + 就地改写），且可自动生长子属性
    put(
        "isaaclab_assets",
        G1_CFG=AutoAttrs(
            prim_path="/World/envs/env_.*/Robot",
            spawn=AutoAttrs(asset_path="<stub>/g1.usd", activate_contact_sensors=True),
            init_state=AutoAttrs(pos=(0.0, 0.0, 0.8), rot=(1.0, 0.0, 0.0, 0.0)),
            actuators={},
            soft_joint_pos_limit_factor=0.9,
        ),
    )

    # 真 isaaclab 的**每个** cfg 类都有 `.replace()` / `.copy()`（链上到处在用，
    # 如 `G1_CFG.replace(prim_path=...)`）⇒ 给所有注册的类统一挂上描述符。
    # 同时补 no-op `__post_init__`：链上大量 `super().__post_init__()` 调用链
    # 最终会走到 stub 基类。
    seen: set[int] = set()
    for obj in list(_KNOWN.values()):
        if isinstance(obj, type) and id(obj) not in seen:
            seen.add(id(obj))
            setattr(obj, "replace", _Replace())
            setattr(obj, "copy", _Copy())
            if not hasattr(obj, "__post_init__"):
                setattr(obj, "__post_init__", lambda self: None)


# ============================================================================
# configclass（**关键**：IsaacLab 的 configclass 会把"没写注解但值是 cfg 的类属性"
# 变成 dataclass 字段；纯 dataclasses.dataclass 做不到，必须复刻这一点）
# ============================================================================


def _is_cfg_value(v: Any) -> bool:
    if v is MISSING:
        return True
    if isinstance(v, PermissiveCfg):
        return True
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return True
    if isinstance(v, (int, float, str, bool, type(None), tuple, list, dict, set)):
        return True
    return False


class _ConfigClassMarker:
    """标记：由本 stub 的 configclass 生成的类（便于识别/调试）。"""


def configclass(cls: type) -> type:
    """`isaaclab.utils.configclass` 的近似实现。

    ⚠️ 两个必须复刻的细节：
    1. IsaacLab 会把**没写注解、但值是 cfg 的类属性**也当成 dataclass 字段；
    2. 补注解时**必须保持类体里的声明顺序** —— 否则 `x = MISSING` 这种"无默认值的
       字段"会被排到有默认值的字段后面，`dataclasses` 直接报
       `non-default argument 'x' follows default argument`。
    """
    # ⚠️ 必须只取**本类自己的**注解：`getattr(cls, "__annotations__")` 会沿 MRO
    #    继承父类的注解字典，于是父类字段会被当成本类的"只有注解、没有赋值"⇒
    #    被判成**必需字段**，`SomeCfg()` 直接报 missing argument。
    ann = dict(cls.__dict__.get("__annotations__", {}))
    ordered: dict[str, Any] = {}
    for name, value in list(vars(cls).items()):
        if name.startswith("_"):
            continue
        if callable(value) or isinstance(value, (staticmethod, classmethod, property)):
            continue
        if name in ann:
            ordered[name] = ann[name]          # 原有注解，按类体出现顺序
        elif _is_cfg_value(value):
            ordered[name] = "Any"              # 补注解（保持同一位置）
    for name, t in ann.items():                # 只有注解、没有赋值的字段（罕见），原序追加
        ordered.setdefault(name, t)

    # ⚠️ `dataclasses` 禁止 list/dict/set 直接当默认值（"mutable default"），而 cfg 里
    #    `joint_names = [".*"]` 这种写法很常见 ⇒ 统一转成 `default_factory`。
    # ⭐ 更进一步：**类体里的 cfg 实例默认值必须每实例深拷贝**。
    #    判据：链上各层大量"取到实例里的 term 就地改写"（T1k 的 `term.weight = 1.5`、
    #    `_set_w()`、`term.params["std"] = ...`）。若默认值是共享对象，这些改写会污染
    #    类属性，一路传到后面的层，导致 T2 自己的 fail-fast 报
    #    `[T2] track_base_height 权重 1.0 != 1.5`（本地实测就是这样）。
    #    真实 IsaacLab 的 configclass 同样按实例复制 ⇒ 这里对齐，local/cloud 才一致。
    for name in list(ordered):
        value = cls.__dict__.get(name, MISSING)
        if value is MISSING or isinstance(value, (int, float, bool, str, type(None), tuple, frozenset)):
            continue
        if isinstance(value, _dc.Field):          # 已经处理过（list/dict/set 那一步）
            continue
        setattr(cls, name, field(default_factory=lambda v=copy.deepcopy(value): copy.deepcopy(v)))

    # ⚠️ **必需字段（无默认值）必须排在有默认值的字段之前**，否则 `dataclasses` 直接报
    #    `non-default argument 'x' follows default argument`。
    #    真实 isaaclab 里靠"字段声明顺序本来就合法"避免；但我们是**继承**它的基类，
    #    基类的字段顺序不一定配合子类 ⇒ 这里做一次稳定重排（两组内部保持原序）。
    #    绝大多数 cfg 都是**关键字**构造，重排无副作用；少数位置参数用法
    #    （如 `SceneEntityCfg("robot", body_names=...)`）依赖"第一个字段仍是第一个"，
    #    稳定排序保证这一点 ✓
    def _required(name: str) -> bool:
        if name not in cls.__dict__:
            return True                      # 只有注解、没有赋值 ⇒ 必需
        return cls.__dict__[name] is MISSING

    ordered = {**{n: t for n, t in ordered.items() if _required(n)},
               **{n: t for n, t in ordered.items() if not _required(n)}}

    cls.__annotations__ = ordered
    # ⚠️ 真实 `configclass` 会让**每个** cfg 类都带 `__post_init__`：链上大量
    #    `super().__post_init__()`，若某层父类没有这个方法，就会报
    #    `'super' object has no attribute '__post_init__'`。
    #    只在"整条 MRO 都没有"时注入 no-op —— 有的话**不能**覆盖，否则会挡住父类的钩子
    #    （链上很多设置就写在父类的 `__post_init__` 里）。
    #    必须在 `dataclasses` 之前注入：dataclass 只在 `hasattr(cls,'__post_init__')`
    #    为真时才会在生成的 `__init__` 里调用它。
    if not any("__post_init__" in vars(k) for k in cls.__mro__ if k is not object):
        cls.__post_init__ = lambda self: None
        print(f"[stub] {cls.__qualname__}: 注入 no-op __post_init__（供子类 super() 调用）")
    # ⚠️ 快照：`dataclasses` 在处理过程中会**删除**被它消费的类属性（把默认值搬到
    #    `__dataclass_fields__` 里），一旦它在最后一步抛 TypeError（字段顺序问题），
    #    这个类就已经被"改坏"了 —— 再重试时默认值全没，字段会变成**必需**。
    #    所以失败后必须先把类恢复原状再重试。
    snapshot = dict(vars(cls))
    try:
        out = _dc.dataclass(cls, eq=False)
    except TypeError as exc:
        # ⚠️ 继承来的字段位置**无法**在子类里重排：基类若把"有默认值的字段"排在前面，
        #    子类再把同名字段改成必需（`= MISSING`），就会报
        #    `non-default argument 'x' follows default argument`。
        #    真实 IsaacLab 靠"基类字段顺序本来就配合"避免；我们不掌握它每个类的细节，
        #    于是**逐级自愈**。若真有位置参数用法，会在**构造时**显式报错，不会静默出错。
        if "non-default argument" not in str(exc):
            raise
        _restore_class(cls, snapshot)
        print(f"[stub] {cls.__qualname__}: 字段顺序与基类冲突 ⇒ 改用 kw_only 构造")
        try:
            out = _dc.dataclass(cls, eq=False, kw_only=True)
        except TypeError as exc2:
            # 二级兜底：冲突发生在**基类**上时，`kw_only` 只作用于本类字段，救不了。
            # 那就彻底放弃自动生成的 `__init__`，换成我们自己的宽容版
            # （字段元信息仍在，`dataclasses.fields` / `replace` 照常可用）。
            if "non-default argument" not in str(exc2):
                raise
            _restore_class(cls, snapshot)
            print(f"[stub] {cls.__qualname__}: kw_only 也救不了（基类顺序冲突）⇒ 自建宽容 __init__")
            out = _dc.dataclass(cls, eq=False, init=False)
            _install_permissive_init(out)
    out.__configclass__ = True
    _install_flexible_init(out)
    return out


def _install_permissive_init(cls: type) -> None:
    """`init=False` 时用的自建 `__init__`：**任何字段都可省**，缺的填默认值，然后跑 `__post_init__`。

    语义：本 stub 只求"链上配置能建出来"，与真实 isaaclab 的差异由
    `dump_cfg_snapshot` 的**对称比较**兜住（链与展平物都在同一 stub 下）。
    """
    fields = _dc.fields(cls)

    def __init__(self, *args, **kwargs):
        if args:
            raise TypeError(f"{cls.__name__}() 在 stub 下只支持关键字参数（收到 {len(args)} 个位置参数）")
        for f in fields:
            if f.name in kwargs:
                value = kwargs.pop(f.name)
            elif f.default is not MISSING:
                value = f.default
            elif f.default_factory is not MISSING:
                value = f.default_factory()
            else:
                value = None                       # 必需但没给 ⇒ 填 None（仅 stub）
            object.__setattr__(self, f.name, value)
        for k, v in kwargs.items():                # 未知关键字（基类携带的字段）也记下
            object.__setattr__(self, k, v)
        post = getattr(self, "__post_init__", None)
        if post is not None:
            post()

    cls.__init__ = __init__


def _restore_class(cls: type, snapshot: dict) -> None:
    """把被 `dataclasses` 改坏的类恢复到快照状态（失败重试前必须做）。"""
    for k in list(vars(cls)):
        if k not in snapshot:
            type.__delattr__(cls, k)
    for k, v in snapshot.items():
        if k not in vars(cls) or vars(cls)[k] is not v:
            type.__setattr__(cls, k, v)


def _install_flexible_init(cls: type) -> None:
    """让 configclass 生成的 `__init__` **额外接受未知关键字**（写进 `__dict__`）。

    ⚠️ 为什么需要：当基类在 stub 里是宽容鸭子类型时（如
    `DirectionalVelocityCommandCfg(mdp.UniformVelocityCommandCfg)`），它携带的字段
    （`asset_name` 等）**不会**成为子类 dataclass 的字段，于是链上
    `SomeCfg(asset_name=..., ...)` 会报 `unexpected keyword argument`。
    真 isaaclab 里那些名字都是真字段 ⇒ 这里用"熟悉的进字段、陌生的进 __dict__"
    来等价兼容。
    """
    field_names = {f.name for f in _dc.fields(cls)}
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        known = {k: v for k, v in kwargs.items() if k in field_names}
        extra = {k: v for k, v in kwargs.items() if k not in field_names}
        orig_init(self, *args, **known)
        for k, v in extra.items():
            setattr(self, k, v)

    cls.__init__ = __init__


# ============================================================================
# import hook
# ============================================================================


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] not in _PREFIXES:
            return None
        # ⭐ 本机**真的**装了的包优先：否则会把 gymnasium/rsl_rl 等真包遮蔽掉，
        #    导致 `gym.register(...)` 之类的真实行为丢失（排查起来很痛）。
        #    `PathFinder` 只看磁盘，不会执行任何 __init__。
        if fullname.split(".")[0] not in ("isaaclab", "isaaclab_assets"):
            if importlib.machinery.PathFinder.find_spec(fullname, path) is not None:
                return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []          # 当作包，允许再往下 import
        mod.__file__ = f"<stub {spec.name}>"
        # ⚠️ 必须绑**闭包**而不是工厂函数本身：PEP 562 的模块 `__getattr__` 只收
        #    一个参数，绑错就变成"任何缺失属性都返回那个函数"。
        mod.__getattr__ = _module_getattr(spec.name)
        return mod

    def exec_module(self, mod) -> None:  # noqa: ARG002
        return


#: 已知的**子模块**名（小写包/模块）。stub 里遇到这些名字要返回"模块"而不是"类"：
#: 生成物会用 `from isaaclab import scene as _m1` / `from isaaclab.sensors.ray_caster
#: import patterns as _m9` 这两种写法（镜像源码、避开深层 import 的坑），
#: 如果 stub 把 `scene`/`patterns` 当成类，本地校验就会错。
_SUBMODULE_NAMES = frozenset(
    {
        "assets", "actuators", "app", "articulation", "cli", "envs", "markers", "managers",
        "mdp", "noise", "patterns", "ray_caster", "scene", "sensors", "sim", "spawners",
        "terrains", "utils", "from_files", "schemas", "assets_cfg", "sensors_cfg",
        "motion_files", "motion_reference", "observations", "events", "rewards", "actions",
        "commands", "wrappers", "instinct_rl", "amass_motion", "motion_reference_data",
        "motion_reference_manager", "rewards_amp", "amp_body_state", "regularizations",
    }
)


def _module_getattr(mod_name: str):
    def __getattr__(name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        hit = _KNOWN.get((mod_name, name)) or _KNOWN.get((None, name))
        if hit is not None:
            return hit
        # ⭐ 已知子模块名 ⇒ 交给 import 机制（我的 finder 会建出模块），保证与
        #    `import a.b` / `from a import b` 两种写法得到**同一个对象**
        if name in _SUBMODULE_NAMES:
            sub = importlib.import_module(f"{mod_name}.{name}")
            parent = sys.modules.get(mod_name)
            if parent is not None:
                setattr(parent, name, sub)
            return sub
        key = (mod_name, name)
        if key not in _DYNAMIC:
            # 未知名字 → 混合体（当函数当类都能用）；sim/spawner 类模块做成"可生长"的
            if mod_name.startswith(_AUTO_PREFIXES):
                _DYNAMIC[key] = _HybridMeta(
                    name, (AutoAttrs,), {"__module__": mod_name, "__qualname__": name}
                )
            else:
                _DYNAMIC[key] = _make_hybrid(name, mod_name)
            mod = sys.modules.get(mod_name)
            if mod is not None:
                setattr(mod, name, _DYNAMIC[key])   # 缓存进模块，保证同一对象
        return _DYNAMIC[key]

    return __getattr__


# ============================================================================
# 安装
# ============================================================================


def install(shim_instinctlab_envs: bool | None = None) -> None:
    """装 stub。**必须早于**任何 isaaclab / instinctlab.envs 的导入。

    `shim_instinctlab_envs=None`（默认）→ 读环境变量 `OFFLINE_STUB_KEEP_ENVS`
    （设为 1 则保留真 `instinctlab.envs`，用于排查"是不是 shim 的锅"）。
    """
    _register()
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _StubFinder())

    # `from isaaclab.utils import configclass` 必须是**真装饰器**。
    # ⚠️ 绝不能写成 `import isaaclab.utils.configclass` —— 那样 import 系统会把
    #    **子模块**绑到父包的 `configclass` 属性上，盖掉刚设进去的函数
    #    （现象：`@configclass` 报 `TypeError: 'module' object is not callable`）。
    utils_mod = importlib.import_module("isaaclab.utils")
    utils_mod.configclass = configclass
    for alias in ("isaaclab", "isaaclab.managers", "isaaclab.envs"):
        try:
            setattr(importlib.import_module(alias), "configclass", configclass)
        except ImportError:
            pass

    if shim_instinctlab_envs is None:
        keep = os.environ.get("OFFLINE_STUB_KEEP_ENVS", "0") in ("1", "true", "yes")
        shim_instinctlab_envs = not keep
    if shim_instinctlab_envs:
        _install_instinctlab_envs_shim()

    _disable_torchscript()


def _disable_torchscript() -> None:
    """把 `torch.jit.script` / `torch.compile` 变成恒等装饰器。

    ⚠️ 为什么必须关：链上（如 `instinctlab/utils/math.py`）用 `@torch.jit.script`
    装饰的函数里会调用 isaaclab 的函数；在 stub 下那些函数是宽容对象，TorchScript
    会在**import 期**就去编译并抛 `TypeError: __init__(): incompatible constructor
    arguments`（torch 内部调用栈构造失败）。
    本地只需要"配置对象能建出来"，JIT 与仿真无关 ⇒ 恒等化即可。
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return

    def _identity(*args, **kwargs):
        def deco(obj):
            return obj

        # 支持 @torch.jit.script 与 @torch.jit.script(...) 两种用法
        if len(args) == 1 and not kwargs and callable(args[0]) and not isinstance(args[0], type):
            return args[0]
        return deco

    torch.jit.script = _identity
    if hasattr(torch, "compile"):
        torch.compile = lambda obj=None, **kw: (obj if callable(obj) and obj is not None else (lambda o: o))


def _install_instinctlab_envs_shim() -> None:
    """把 `instinctlab.envs` 换成一个"只提供类名 + 保留子模块路径"的 shim。

    为什么**默认**要换掉：
    * 教师类继承 `instinctlab.envs.InstinctRlEnv`，但本地**从不实例化**它；
    * 真模块会拖进整套 env/manager 依赖（本地 `instinctlab` 环境缺 `prettytable`
      等三方包）⇒ 为了让 43 层链能 import，最省事、最确定的做法就是替掉这一层。

    ⚠️ 必须保留 `__path__`（指向真实目录）：链上大量 `import instinctlab.envs.mdp`，
    那是**真文件**，要走磁盘 import。
    """
    if "instinctlab.envs" in sys.modules and getattr(sys.modules["instinctlab.envs"], "_is_stub", False):
        return
    # ⚠️ 用 find_spec 定位目录，**不要** `import instinctlab` —— 它的 `__init__` 会
    #    连锁 import 整套 tasks（其中就包括我们要绕开的 envs），当场就炸。
    spec = importlib.util.find_spec("instinctlab")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("[stub] 找不到 instinctlab 包（PYTHONPATH 是否包含 .../source/instinctlab？）")
    envs_dir = os.path.join(next(iter(spec.submodule_search_locations)), "envs")
    if not os.path.isdir(envs_dir):
        raise RuntimeError(f"[stub] 目录不存在：{envs_dir}")

    shim = types.ModuleType("instinctlab.envs")
    shim.__path__ = [envs_dir]
    shim.__package__ = "instinctlab.envs"
    shim._is_stub = True
    shim.InstinctRlEnv = type("InstinctRlEnv", (PermissiveCfg,), {"__module__": "instinctlab.envs"})
    shim.InstinctLabRLEnvCfg = type(
        "InstinctLabRLEnvCfg", (ManagerBasedRLEnvCfg,), {"__module__": "instinctlab.envs"}
    )
    sys.modules["instinctlab.envs"] = shim
    print(f"[stub] instinctlab.envs → shim（子模块仍从 {envs_dir} import）")


# ============================================================================
# 宽容 __init__：真环境的字段比 stub 多，本地校验不该报假错
# ============================================================================


def _install_permissive_init(cls: Any) -> None:
    """把 dataclass 的**严格** `__init__` 换成宽容版。

    动机：stub 只实现了链上真正用到的字段，而**真实 IsaacLab 的 cfg 字段更多**；
    用真环境（云端）生成的展平文件会带这些多出来的字段 ⇒ 本地校验时报
    `unexpected keyword argument 'class_type'` 之类的**假错**（已验证：云端真环境
    的 `TerrainImporterCfg` / `ArticulationCfg` 都有 `class_type`）。

    宽容版 = **dataclass 默认值语义**（含 `default_factory`）+ 吸收任意额外 kwargs。
    """
    if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
        return
    if "_permissive_init" in cls.__dict__:
        return

    def __init__(self, *args, **kwargs):
        for f in dataclasses.fields(type(self)):
            if f.default is not dataclasses.MISSING:
                self.__dict__[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                self.__dict__[f.name] = f.default_factory()
        PermissiveCfg.__init__(self, *args, **kwargs)
        post = getattr(self, "__post_init__", None)
        if callable(post):
            post()

    __init__.__qualname__ = f"{cls.__qualname__}.__init__"
    cls.__init__ = __init__
    cls._permissive_init = True


for _cls in list(vars().values()) + list(_KNOWN.values()):
    _install_permissive_init(_cls)
del _cls


# ============================================================================
# 作为运行器
# ============================================================================


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    install()
    import runpy

    runpy.run_path(target, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
