#!/usr/bin/env python3
"""静态契约校验：`amp_walk_env_cfg.py` 及其注册项。

为什么需要这个工具
------------------
本机显存 4 GB，**跑不了 Isaac Sim**（最低 8 GB），因此无法用 `gym.make()` 做真实
冒烟测试。在云机器（RTX 4090）开机之前，唯一能做且**有意义**的验证就是
对配置模块做静态契约检查。

它检查的都是真正会出错的地方
----------------------------
A. 维度算术：`AMP_WALK_OBSERVATION_DIMS` 之和 == 626，且与 `_TERMS` 等长
B. 字段顺序：`G1AmpWalkPolicyObsCfg` 里 `ObsTerm` 的声明顺序 == `_TERMS`（**顺序即契约**）
C. 符号真实性：本文件用到的每个 `<alias>.<name>`，都必须在仓库其它文件中
   以同样形式出现过 —— 即“**不引入任何未经证明存在的符号**”。
   这条能抓住最隐蔽的一类 bug：`instinctlab.envs.mdp` 并**没有**转发
   `base_lin_vel`（只转发了 `base_ang_vel` 等），写 `instinct_mdp.base_lin_vel`
   在静态期看不出来，但到 Isaac Sim 里就是 `AttributeError`。
D. 注册一致性：任务 ID 全局唯一；`instinct_rl_cfg_entry_point` 指向的
   模块文件存在且确实定义了那个类。
E. 契约继承：`G1AmpWalkTeacherEnv` 覆盖的三个期望值 == 对应常量。

用法
----
    python tools/check_amp_walk_task.py
"""
from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent            # .../llp
INSTINCTLAB = REPO / "instinct_xu/source/instinctlab/instinctlab"
G1_DIR = INSTINCTLAB / "tasks/locomotion/config/g1"
TARGET = G1_DIR / "amp_walk_env_cfg.py"
COMMANDS_PY = INSTINCTLAB / "tasks/locomotion/mdp/commands.py"
MOTION_DIR = G1_DIR / "motions/amp_walk"
FLAT_CFG = G1_DIR / "flat_env_cfg.py"
REWARDS_AMP_PY = INSTINCTLAB / "tasks/locomotion/mdp/rewards_amp.py"
FOOT_SEP_STATS = REPO / "instinct/tools/amp_walk_foot_sep_stats.json"

EXPECTED_SIZE = 626
EXPECTED_AMP_STATE_SIZE = 670
#: 判别器状态向量里每个项的“单帧”维度（29 = G1 关节数）
AMP_BASE_DIMS = {
    "projected_gravity": 3,
    "joint_pos_rel": 29,
    "joint_vel_rel": 29,
    "base_lin_vel": 3,
    "base_ang_vel": 3,
}

#: 允许“仓库中无先例”的符号，但必须已经在代码里用 try/except 显式守卫。
#: 值 = 理由。这类符号会在输出里记 ⚠️ 而不是 ❌。
ALLOWED_NEW_SYMBOLS = {
    "isaaclab.envs.mdp.UniformVelocityCommand":
        "仓库内只见过 mdp.UniformVelocityCommandCfg；类那一半无先例，已在 commands.py 用 "
        "try/except ImportError 守卫并给出可读报错",
    "isaaclab.envs.mdp.joint_vel_rel":
        "IsaacLab 标准 API，且上游 parkour 官方实现用了它；但 instinctlab 未转发，本仓库内无先例",
}

#: 目标文件里模块级的字面量常量表（用于解析 `(AMP_STATE_HISTORY, 3)` 这类写法）
_CONSTS: dict[str, object] = {}

#: 步骤⑤ 奖励迁移契约。基类 `G1FlatRewardsCfg` 的这些权重**必须原样不变** ——
#: 用户要求「不改奖励函数与权重」，因为 OmniLoco / Vault / flat 三条链路已训好。
BASE_REWARD_WEIGHTS_UNCHANGED = {
    "termination_penalty": -200.0,
    "track_lin_vel_xy_exp": 1.0,
    "track_ang_vel_z_exp": 1.0,
    "feet_air_time": 1.0,
    "feet_slide": -0.1,
    "flat_orientation_l2": -1.0,
    "stand_still": -0.8,
    "dof_pos_limits": -1.0,
    "lin_vel_z_l2": -0.1,
    "action_rate_l2": -0.05,
}
#: AMP 行走子类**只允许**覆盖这三项的权重（AMP_mjlab v5 的表）。
AMP_WALK_REWARD_OVERRIDES = {
    "track_lin_vel_xy_exp": 1.5,
    "track_ang_vel_z_exp": 1.2,
    "feet_slide": -0.25,
}
#: 迁移过来的新项 → 期望权重（其余参数在下面单独对账）。
AMP_WALK_REWARD_NEW = {
    "track_base_height": 1.5,
    "body_ang_vel_xy": 0.7,
    "track_default_pose_on_stand": -1.0,
    "feet_lateral_width": -10.0,
    "feet_crossing": -20.0,
}
#: 直线行走标定集（与 `tools/calibrate_foot_sep.py` 的 `WALK_CLIPS` 一致）。
FOOT_SEP_WALK_CLIPS = (
    "walk_forward_loop_002__A022",
    "walk_forward_loop_002__A024",
    "walk_backward_loop_001__A022",
    "walk_backward_loop_001__A024",
)
#: 交叉式侧步 clip（实测 21% 帧 sep<0）—— 它们是 `gate_lateral=True` 的依据。
CROSS_STEP_CLIPS = (
    "walk_sideway_left_loop_002__A021",
    "walk_sideway_right_loop_001__A022",
)

#: `asset.data.<attr>` 中允许「本仓库无直接先例」的属性（值 = 理由）。
#: 这道守卫防的是 `AttributeError` —— 与 C 类同一个思路，但针对数据容器属性
#: （C 类只能查 `<模块>.<名字>`，查不到 `asset.data.xxx`）。
ALLOWED_DATA_ATTRS = {
    "root_ang_vel_b":
        "IsaacLab 标准属性：`isaaclab.envs.mdp.base_ang_vel` 的实现就是 `asset.data.root_ang_vel_b`，"
        "而 `instinctlab/envs/mdp/observations/expanded.py` 正是从那里转发的；"
        "仓库里只是没直接写过 `data.root_ang_vel_b` 而已",
}

_errors: list[str] = []
_warns: list[str] = []


def err(msg: str) -> None:
    _errors.append(msg)


def warn(msg: str) -> None:
    _warns.append(msg)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def literal(node: ast.AST):
    """把 AST 字面量（含一元负号 / tuple）转成 Python 值。

    先试 `literal_eval`；不成则在本模块的常量表 `_CONSTS` 下求值，
    以支持 `AMP_STATE_DIMS = ((AMP_STATE_HISTORY, 3), ...)` 这种引用。
    """
    try:
        return ast.literal_eval(node)
    except Exception:
        pass
    try:
        code = compile(ast.Expression(body=node), "<const>", "eval")
        return eval(code, {"__builtins__": {}}, dict(_CONSTS))  # noqa: S307 - 仅用于解析本仓库常量
    except Exception:
        return None


def module_consts(tree: ast.Module) -> dict[str, object]:
    """树里模块级的字面量常量（供上面的常量表求值）。"""
    out: dict[str, object] = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except Exception:
                pass
    return out


def import_aliases(tree: ast.Module) -> dict[str, str]:
    """别名 -> 模块路径。例如 `import instinctlab.envs.mdp as instinct_mdp` -> {...: 'instinctlab.envs.mdp'}"""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out[a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                out[a.asname or a.name] = f"{node.module}.{a.name}"
    return out


def class_defs(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}


def top_level_names(tree: ast.Module) -> set[str]:
    """模块顶层可导出名：def / class / `from x import a,b`。"""
    names: set[str] = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                names.add(a.asname or a.name)
    return names


def assigned_dict(tree: ast.Module, name: str):
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return literal(n.value)
        elif isinstance(n, ast.AnnAssign):
            if isinstance(n.target, ast.Name) and n.target.id == name:
                return literal(n.value)
    return None


def child_assignments(cls: ast.ClassDef) -> list[tuple[str, ast.Call]]:
    """类体内形如 `field = SomeCall(...)` 的赋值，按声明顺序。"""
    out = []
    for n in cls.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            if isinstance(n.value, ast.Call):
                out.append((n.targets[0].id, n.value))
    return out


def call_func_name(call: ast.AST) -> str:
    """Call 节点的被调函数名；非 Call 返回空串。"""
    if not isinstance(call, ast.Call):
        return ""
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return f"{f.value.id}.{f.attr}"
    return ""


def string_value(node: ast.AST | None) -> str | None:
    """取字符串字面量；f-string 只拼接其中的常量片段。

    仓库里的写法是 `f"{agents.__name__}.amp_walk_ppo_cfg:G1AmpWalkPPORunnerCfg"`，
    拼接常量片段后得到 `.amp_walk_ppo_cfg:G1AmpWalkPPORunnerCfg`（前面多个点），
    调用方用 lstrip('.') 处理即可。
    """
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)
        )
    return None


def _parent_map(t: ast.Module) -> dict:
    parents: dict = {}
    for p in ast.walk(t):
        for c in ast.iter_child_nodes(p):
            parents[c] = p
    return parents


def _nearest_for(node: ast.AST, parents: dict) -> ast.For | None:
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, ast.For):
            return cur
        cur = parents.get(cur)
    return None


def _loop_var_pairs(fornode: ast.For) -> tuple[str | None, list[str]]:
    """从 `for task_id, env_cfg in ((\"id\", Cfg), ...)` 里取出变量名与全部 id 字面量。"""
    if not (isinstance(fornode.target, ast.Tuple) and len(fornode.target.elts) == 2):
        return None, []
    first = fornode.target.elts[0]
    if not (isinstance(first, ast.Name) and isinstance(fornode.iter, ast.Tuple)):
        return None, []
    ids = [
        el.elts[0].value
        for el in fornode.iter.elts
        if isinstance(el, ast.Tuple) and el.elts and isinstance(el.elts[0], ast.Constant)
    ]
    return first.id, ids


def extract_registrations(py: Path) -> list[dict]:
    """提取一个文件里所有 gym.register(...)。

    仓库用的是「模块级 for 循环 + 元组表」的写法：
        for task_id, env_cfg in (("Instinct-...-v0", SomeCfg), ...):
            gym.register(id=task_id, ...)
    所以 `id` 是个 Name，必须靠**最近的**那个 for 循环才能解析回字面量
    （同一文件里会有多个都叫 `task_id` 的循环，靠全局名字表会串扰）。
    """
    try:
        t = parse(py)
    except SyntaxError as e:
        err(f"D: {py} 语法错误: {e}")
        return []
    parents = _parent_map(t)

    out: list[dict] = []
    for node in ast.walk(t):
        if not (isinstance(node, ast.Call) and call_func_name(node) == "gym.register"):
            continue
        kv = {k.arg: k.value for k in node.keywords}

        ids: list[str] = []
        idn = kv.get("id")
        if isinstance(idn, ast.Constant):
            ids = [idn.value]
        elif isinstance(idn, ast.Name):
            fornode = _nearest_for(node, parents)
            if fornode is not None:
                varname, loop_ids = _loop_var_pairs(fornode)
                if varname == idn.id:
                    ids = loop_ids

        ep_node = kv.get("entry_point")
        entry_point_is_name = isinstance(ep_node, ast.Name)
        entry_point = (ep_node.id if entry_point_is_name else (string_value(ep_node) or ""))

        rl_cfg = ""
        kwargs = kv.get("kwargs")
        if isinstance(kwargs, ast.Dict):
            for k, v in zip(kwargs.keys, kwargs.values):
                if isinstance(k, ast.Constant) and k.value == "instinct_rl_cfg_entry_point":
                    rl_cfg = (string_value(v) or "").lstrip(".")

        out.append(
            {
                "file": py,
                "ids": ids,
                "entry_point": entry_point,
                "entry_point_is_name": entry_point_is_name,
                "rl_cfg": rl_cfg,
            }
        )
    return out


# ---------------------------------------------------------------------------
# A. 维度算术
# ---------------------------------------------------------------------------

def check_dims(tree: ast.Module) -> None:
    terms = assigned_dict(tree, "AMP_WALK_OBSERVATION_TERMS")
    dims = assigned_dict(tree, "AMP_WALK_OBSERVATION_DIMS")
    if not terms or not dims:
        err("A: 找不到 AMP_WALK_OBSERVATION_TERMS / _DIMS 字面量定义")
        return
    print(f"A. 维度算术")
    print(f"   terms = {len(terms)} 项: {terms}")
    print(f"   dims  = {dims}")
    if len(terms) != len(dims):
        err(f"A: terms({len(terms)}) 与 dims({len(dims)}) 长度不一致")
        return
    if sum(dims) != EXPECTED_SIZE:
        err(f"A: dims 之和 = {sum(dims)}，期望 {EXPECTED_SIZE}")
    else:
        print(f"   ✅ sum = {sum(dims)} == {EXPECTED_SIZE}")


# ---------------------------------------------------------------------------
# B. 观测字段声明顺序
# ---------------------------------------------------------------------------

def check_obs_order(tree: ast.Module) -> None:
    terms = assigned_dict(tree, "AMP_WALK_OBSERVATION_TERMS") or ()
    classes = class_defs(tree)
    cls = classes.get("G1AmpWalkPolicyObsCfg")
    if cls is None:
        err("B: 找不到 G1AmpWalkPolicyObsCfg")
        return
    fields = [n for n, c in child_assignments(cls) if call_func_name(c) == "ObsTerm"]
    print(f"B. 观测字段顺序")
    print(f"   声明 = {tuple(fields)}")
    if tuple(fields) != tuple(terms):
        err(f"B: 声明顺序 != AMP_WALK_OBSERVATION_TERMS\n     声明: {tuple(fields)}\n     契约: {tuple(terms)}")
    else:
        print(f"   ✅ 声明顺序与契约完全一致")


# ---------------------------------------------------------------------------
# C. 符号真实性（不引入未经证明存在的符号）
# ---------------------------------------------------------------------------

def build_proven_symbols(exclude: Path) -> set[str]:
    """扫描仓库里所有*其它*任务配置，收集已出现过的 `<alias>.<name>` 形式。

    别名先被解析成真实模块路径（如 `isaaclab.envs.mdp` / `instinctlab.envs.mdp`），
    因此得到的是“模块路径 + 属性名”的**跨文件可比**结论。
    """
    proven: set[str] = set()
    roots = [INSTINCTLAB / "tasks", INSTINCTLAB / "envs", INSTINCTLAB / "assets"]
    for root in roots:
        for py in root.rglob("*.py"):
            if py == exclude:
                continue
            try:
                t = parse(py)
            except SyntaxError:
                continue
            aliases = import_aliases(t)
            for node in ast.walk(t):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    mod = aliases.get(node.value.id)
                    if mod:
                        proven.add(f"{mod}.{node.attr}")
    return proven


def self_authored_exports() -> set[str]:
    """本次新增/修改的模块自身的导出，也应算作「已被证明」——但附带一个前提检查。

    光“某个包下有文件定义了 X”并不能证明 `pkg.X` 可用；必须这个包的
    `__init__.py` 真的转发了它（`from .<mod> import *`）。所以这里顺手把
    转发关系一并验证，否则 C 类检查会退化成“宽松到没用”。
    """
    pkg_dir = COMMANDS_PY.parent
    mod = COMMANDS_PY.stem
    init_py = pkg_dir / "__init__.py"
    if not init_py.is_file():
        err(f"C: 缺少 {init_py.relative_to(INSTINCTLAB)}")
        return set()
    init_src = init_py.read_text(encoding="utf-8")
    if f"from .{mod} import *" not in init_src:
        err(f"C: {pkg_dir.name}/__init__.py 未转发 `from .{mod} import *` —— "
            f"则 `...{pkg_dir.name}.{mod}` 里的符号（如 DirectionalVelocityCommandCfg）实际不可用")
        return set()
    pkg = f"instinctlab.tasks.locomotion.{pkg_dir.name}"
    return {f"{pkg}.{n}" for n in top_level_names(parse(COMMANDS_PY))}


def module_exports(module_path: str) -> set[str] | None:
    """若该模块属于本仓库（`instinctlab.*`），返回它**实际**导出的名字集合；否则 None。

    这比早先“该符号在仓库别处出现过”的启发式更强：直接看定义/转发是否真的存在。
    跟随子包时必须存在显式 `from .<sub> import *`，否则“某子目录里有定义”
    并不能证明 `pkg.name` 可用。
    """
    if not module_path.startswith("instinctlab."):
        return None
    base = INSTINCTLAB / module_path[len("instinctlab."):].replace(".", "/")
    if base.is_dir():
        init = base / "__init__.py"
        if not init.is_file():
            return None
        init_src = init.read_text(encoding="utf-8")
        names: set[str] = set()
        for py in base.glob("*.py"):
            try:
                names |= top_level_names(parse(py))
            except SyntaxError:
                pass
        for sub in sorted(base.iterdir()):
            if sub.is_dir() and (sub / "__init__.py").is_file() \
                    and f"from .{sub.name} import *" in init_src:
                for py in sub.glob("*.py"):
                    try:
                        names |= top_level_names(parse(py))
                    except SyntaxError:
                        pass
        return names
    py = base.with_suffix(".py")
    if py.is_file():
        return top_level_names(parse(py))
    return None


def check_symbols() -> None:
    t = parse(TARGET)
    aliases = import_aliases(t)
    proven = build_proven_symbols(TARGET) | self_authored_exports()

    used: dict[str, list[str]] = {}
    for node in ast.walk(t):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            mod = aliases.get(node.value.id)
            if not mod:
                continue
            # 只审视“模块.名字”。末段非全小写的多半是被导入的**对象**
            # （如 `G1_CFG.spawn`），不属于“模块符号”范畴。
            if not mod.rsplit(".", 1)[-1].islower():
                continue
            used.setdefault(f"{mod}.{node.attr}", []).append(node.value.id)

    print("C. 符号真实性（本仓库模块直接查定义；外部模块查仓库先例）")
    for sym in sorted(used):
        mod, _, name = sym.rpartition(".")
        exports = module_exports(mod)
        if exports is not None:
            ok = name in exports
            print(f"   {'✅' if ok else '❌'} {sym}   [直接定义检查]")
            if not ok:
                err(f"C: `{mod}` 并未导出 `{name}`")
        elif sym in proven:
            print(f"   ✅ {sym}")
        elif sym in ALLOWED_NEW_SYMBOLS:
            print(f"   ⚠️  {sym}  (已允许：{ALLOWED_NEW_SYMBOLS[sym]})")
        else:
            print(f"   ❌ {sym}")
            err(f"C: 符号 `{sym}` 无法证明其存在")

    # 专项检查：不得经 instinct_mdp 访问它**没有**导出的名字。
    # （以前这里用子串判断，会被 `instinct_mdp.base_lin_vel_reference_as_state`
    #  这种正确用法误伤，所以改成基于 AST 的精确属性访问检查。）
    not_in_instinctlab = {"base_lin_vel", "joint_vel_rel"}
    misuse = sorted({
        node.attr
        for node in ast.walk(t)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and aliases.get(node.value.id) == "instinctlab.envs.mdp"
        and node.attr in not_in_instinctlab
    })
    if misuse:
        err(f"C: 不应经 instinct_mdp 访问 {misuse}（instinctlab.envs.mdp 未导出这些名字）")
    else:
        names = ", ".join(f"instinct_mdp.{n}" for n in sorted(not_in_instinctlab))
        print(f"   ✅ 未误用 {names}（二者应取自 isaaclab.envs.mdp）")


# ---------------------------------------------------------------------------
# D. 注册一致性
# ---------------------------------------------------------------------------

def check_registration() -> None:
    print("D. 注册一致性")

    all_regs: list[dict] = []
    for py in INSTINCTLAB.rglob("*.py"):
        all_regs += extract_registrations(py)

    id_to_files: dict[str, list[str]] = {}
    for r in all_regs:
        for i in r["ids"]:
            id_to_files.setdefault(i, []).append(str(r["file"].relative_to(INSTINCTLAB)))
    for k, v in id_to_files.items():
        if len(v) > 1:
            err(f"D: 任务 ID `{k}` 重复注册于 {v}")

    new_ids = sorted(i for i in id_to_files if "Amp-Walk" in i)
    print(f"   全部任务 ID: {sorted(id_to_files)}")
    print(f"   新增 Amp-Walk: {new_ids}")
    if len(new_ids) != 2:
        err(f"D: 期望注册 2 个 Amp-Walk 任务，实际 {len(new_ids)}")
    kinds = sorted(i_to_kind(i) for i in new_ids)
    if kinds != ["play", "train"]:
        err(f"D: Amp-Walk 注册应为 1 训练 + 1 Play，实际 {kinds}")
    else:
        print("   ✅ 新增 = 1 训练 + 1 Play")

    # entry_point 类 + agent 模块:类 必须真实存在（只对 G1 locomotion 包做深度检查，
    # 其它任务包（如 Vault）不在本次改动范围内）。
    for r in all_regs:
        target_dir = r["file"].parent
        if target_dir != G1_DIR:
            continue
        ep = r["entry_point"]
        if ep and r["entry_point_is_name"]:
            found = any(ep in class_defs(parse(p)) for p in target_dir.glob("*.py"))
            print(f"   entry_point {ep}: {'✅ 已定义' if found else '❌ 未定义'}")
            if not found:
                err(f"D: entry_point 类 `{ep}` 在 {target_dir.name}/ 下任何模块中都未定义")
        if r["rl_cfg"]:
            mod, _, klass = r["rl_cfg"].partition(":")
            modfile = target_dir / "agents" / f"{mod.split('.')[-1]}.py"
            if not modfile.is_file():
                err(f"D: agent 模块不存在: {modfile.relative_to(INSTINCTLAB)}")
            elif klass not in class_defs(parse(modfile)):
                err(f"D: {modfile.name} 中未定义类 `{klass}`")
            else:
                print(f"   agent cfg {modfile.name}:{klass} ✅")
    if not _errors:
        print("   ✅ ID 唯一；entry_point 类、agent 模块与类均存在")


def i_to_kind(task_id: str) -> str:
    return "play" if task_id.endswith("-Play-v0") else "train"


# ---------------------------------------------------------------------------
# E. 契约继承
# ---------------------------------------------------------------------------

def check_teacher_contract(tree: ast.Module) -> None:
    print("E. 契约继承")
    cls = class_defs(tree).get("G1AmpWalkTeacherEnv")
    if cls is None:
        err("E: 找不到 G1AmpWalkTeacherEnv")
        return
    overridden: dict[str, str] = {}
    for n in cls.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            if isinstance(n.value, ast.Name):
                overridden[n.targets[0].id] = n.value.id
    print(f"   TeacherEnv 覆盖: {overridden}")
    if overridden.get("expected_policy_terms") != "AMP_WALK_OBSERVATION_TERMS":
        err("E: expected_policy_terms 未指向 AMP_WALK_OBSERVATION_TERMS")
    if overridden.get("expected_policy_dims") != "AMP_WALK_OBSERVATION_DIMS":
        err("E: expected_policy_dims 未指向 AMP_WALK_OBSERVATION_DIMS")
    if "G1OmniLocoTeacherEnv" not in [ast.unparse(b) for b in cls.bases]:
        err("E: G1AmpWalkTeacherEnv 未继承 G1OmniLocoTeacherEnv")


# ---------------------------------------------------------------------------
# F. 配置接线
# ---------------------------------------------------------------------------

def _ann_fields(cls: ast.ClassDef) -> dict[str, str]:
    """类体内的带注解赋值：字段名 -> 注解文本。"""
    out: dict[str, str] = {}
    for n in cls.body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out[n.target.id] = ast.unparse(n.annotation)
    return out


def _bases_of(classes: dict[str, ast.ClassDef], name: str) -> list[str]:
    cls = classes.get(name)
    return [ast.unparse(b) for b in cls.bases] if cls else []


def check_wiring(tree: ast.Module) -> None:
    print("F. 配置接线")
    classes = class_defs(tree)

    env, env_play = classes.get("G1AmpWalkEnvCfg"), classes.get("G1AmpWalkEnvCfg_PLAY")
    obs = classes.get("G1AmpWalkObservationsCfg")
    if not (env and env_play and obs):
        err("F: 缺少 G1AmpWalkEnvCfg / _PLAY / G1AmpWalkObservationsCfg 之一")
        return

    checks = [
        ("G1AmpWalkEnvCfg 继承 G1OmniLocoEnvCfg",
         "G1OmniLocoEnvCfg" in [ast.unparse(b) for b in env.bases]),
        ("env.observations -> G1AmpWalkObservationsCfg",
         _ann_fields(env).get("observations") == "G1AmpWalkObservationsCfg"),
        ("env.commands     -> G1AmpWalkCommandsCfg",
         _ann_fields(env).get("commands") == "G1AmpWalkCommandsCfg"),
        ("env.scene        -> G1AmpWalkSceneCfg",
         _ann_fields(env).get("scene") == "G1AmpWalkSceneCfg"),
        ("obs.policy  -> G1AmpWalkPolicyObsCfg",
         _ann_fields(obs).get("policy") == "G1AmpWalkPolicyObsCfg"),
        ("obs.critic  -> G1FlatCriticObsCfg（本就含 base_lin_vel）",
         _ann_fields(obs).get("critic") == "G1FlatCriticObsCfg"),
        ("obs.amp_policy    -> AmpPolicyStateObsCfg",
         _ann_fields(obs).get("amp_policy") == "AmpPolicyStateObsCfg"),
        ("obs.amp_reference -> AmpReferenceStateObsCfg",
         _ann_fields(obs).get("amp_reference") == "AmpReferenceStateObsCfg"),
        ("_PLAY 继承 G1AmpWalkEnvCfg",
         "G1AmpWalkEnvCfg" in [ast.unparse(b) for b in env_play.bases]),
        ("_PLAY.scene 已改为 num_envs=1",
         "num_envs=1" in _play_scene_src(env_play)),
        ("G1AmpWalkSceneCfg 继承 G1OmniLocoSceneCfg（继承 527 射线高程图）",
         _bases_of(classes, "G1AmpWalkSceneCfg") == ["G1OmniLocoSceneCfg"]),
        ("scene 含 motion_reference 传感器",
         _ann_fields(classes["G1AmpWalkSceneCfg"]).get("motion_reference")
         == "MotionReferenceManagerCfg"),
    ]

    # 速度指令：必须用方向分档类，而**不是** UniformVelocityCommandCfg
    cmd_cls = classes.get("G1AmpWalkCommandsCfg")
    if cmd_cls is None:
        checks.append(("存在 G1AmpWalkCommandsCfg", False))
    else:
        bv = ""
        for n in cmd_cls.body:
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) \
                    and n.targets[0].id == "base_velocity":
                bv = call_func_name(n.value)
        checks.append(("base_velocity 用的是方向分档指令（非 UniformVelocityCommandCfg）",
                       bv.endswith("DirectionalVelocityCommandCfg")))
        checks.append(("G1AmpWalkCommandsCfg 继承 G1OmniLocoCommandsCfg",
                       "G1OmniLocoCommandsCfg" in [ast.unparse(b) for b in cmd_cls.bases]))
    for name, ok in checks:
        print(f"   {'✅' if ok else '❌'} {name}")
        if not ok:
            err(f"F: {name}")


def _play_scene_src(env_play: ast.ClassDef) -> str:
    for n in env_play.body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "scene":
            return ast.unparse(n.value)
    return ""


# ---------------------------------------------------------------------------
# G. 高程图射线数算术（独立复算 527）
# ---------------------------------------------------------------------------

def check_scan_rays(tree: ast.Module) -> None:
    print("G. 高程图射线数算术（独立复算）")
    t = parse(G1_DIR / "omni_loco_env_cfg.py")
    res = size = None
    for node in ast.walk(t):
        if isinstance(node, ast.Call) and call_func_name(node) == "patterns.GridPatternCfg":
            for kw in node.keywords:
                if kw.arg == "resolution":
                    res = literal(kw.value)
                if kw.arg == "size":
                    size = literal(kw.value)
    if res is None or size is None:
        err("G: 未能从 omni_loco_env_cfg.py 解析 GridPatternCfg(resolution, size)")
        return

    nx, ny = round(size[0] / res), round(size[1] / res)
    rays = (nx + 1) * (ny + 1)
    declared = assigned_dict(tree, "AMP_WALK_SCAN_RAYS")
    dims = assigned_dict(tree, "AMP_WALK_OBSERVATION_DIMS")
    height_scan_dim = dims[0] if dims else None

    print(f"   GridPatternCfg: resolution={res}, size={size}  ->  {nx + 1} x {ny + 1} = {rays}")
    print(f"   声明 AMP_WALK_SCAN_RAYS = {declared}；dims[0](height_scan) = {height_scan_dim}")
    if rays != declared:
        err(f"G: 独立算得射线数 {rays} != 声明值 {declared}")
    if height_scan_dim != rays:
        err(f"G: dims[0]={height_scan_dim} 与算得的射线数 {rays} 不一致")
    if rays != declared or height_scan_dim != rays:
        return
    print("   ✅ 独立复算 == 契约声明；且与 OmniLoco 的 527 一致")


def terminal_value(node: ast.AST):
    """把表达式化简为 Python 值，允许自然的算术。"""
    return literal(node)


# ---------------------------------------------------------------------------
# H. 速度指令包络 vs 参考动作实测包络
# ---------------------------------------------------------------------------

def _defaults_of(path: Path, class_name: str) -> dict:
    """取一个 configclass 里所有带字面量默认值的字段。"""
    cls = class_defs(parse(path)).get(class_name)
    if cls is None:
        return {}
    out = {}
    for n in cls.body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            v = literal(n.value)
            if v is not None:
                out[n.target.id] = v
    return out


def check_command_envelope(tree: ast.Module) -> None:
    """命令范围必须落在**参考动作实测包络**内。

    依据 AMP_mjlab 的实测教训（PLAN §3.4）：命令超出参考数据覆盖范围时，
    该档位缺乏参考密度，判别器会把对应姿态判为「不像参考」而抑制。
    这里直接拿①产出的 npz 重算包络，与②③的配置做数值对账。
    """
    print("H. 速度指令包络 vs 参考动作实测包络")

    # --- 1) 分档概率 ---------------------------------------------------------
    d = _defaults_of(COMMANDS_PY, "DirectionalVelocityCommandCfg")
    if not d:
        err(f"H: 未能解析 {COMMANDS_PY.name} 的默认值")
        return
    prob_keys = ("prob_low", "prob_side", "prob_lin", "prob_back")
    probs = sum(float(d.get(k, 0.0)) for k in prob_keys)
    print("   分档概率: " + ", ".join(f"{k}={d.get(k)}" for k in prob_keys))
    print(f"   之和 = {probs:.2f}，剩余 {1 - probs:.2f} 为均匀斜向")
    if probs > 1.0:
        err(f"H: 分档概率之和 {probs:.2f} > 1")
    elif probs <= 0.0:
        err("H: 分档概率之和为 0，方向分档失效")

    # --- 2) 数据集实测包络 ---------------------------------------------------
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import convert_amp_motion as cam  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        warn(f"H: 无法导入 convert_amp_motion（{e}），跳过包络对账")
        return
    files = sorted(MOTION_DIR.glob("*_retargeted.npz"))
    if not files:
        warn(f"H: {MOTION_DIR} 下没有 npz，跳过包络对账")
        return
    rows = [cam.body_frame_stats(f) for f in files]
    axes = {
        "lin_vel_x": [r["vx"] for r in rows],
        "lin_vel_y": [r["vy"] for r in rows],
        "ang_vel_z": [r["wz"] for r in rows],
    }
    env = {k: (min(v), max(v)) for k, v in axes.items()}
    print(f"   实测包络（{len(files)} 个 clip）:")
    for k, (lo, hi) in env.items():
        print(f"     {k:12s} [{lo:+.3f}, {hi:+.3f}]")

    # --- 3) 对账 ------------------------------------------------------------
    ranges = assigned_dict(tree, "AMP_WALK_COMMAND_RANGES") or ()
    to_check: list[tuple[str, str, tuple[float, float]]] = []
    for key, rng in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), ranges):
        to_check.append((f"ranges.{key}", key, tuple(rng)))
    # 分档专属范围：命名的轴是写死的，不做字符串猜测
    for key, axis in (
        ("straight_lin_vel_range", "lin_vel_x"),
        ("straight_ang_vel_range", "ang_vel_z"),
        ("back_lin_vel_range", "lin_vel_x"),
        ("back_ang_vel_range", "ang_vel_z"),
    ):
        if key in d:
            to_check.append((key, axis, tuple(d[key])))

    print("   对账：")
    for name, axis, (lo, hi) in to_check:
        if lo > hi:
            err(f"H: {name} 下界 {lo} > 上界 {hi}")
            continue
        dlo, dhi = env[axis]
        mag = max(abs(dlo), abs(dhi))
        inside_signed = lo >= dlo and hi <= dhi
        inside_mag = max(abs(lo), abs(hi)) <= mag + 1e-9
        if inside_signed:
            print(f"   ✅ {name:24s} [{lo:+.2f}, {hi:+.2f}] 完全落在 {axis} 包络内")
        elif inside_mag:
            print(f"   ⚠️  {name:24s} [{lo:+.2f}, {hi:+.2f}] 略超出 {axis} 单向包络 "
                  f"[{dlo:+.3f}, {dhi:+.3f}]，但未超幅值上限 {mag:.3f}")
            warn(f"H: {name} 略超出 {axis} 单向包络（属轻度外插，可接受但应知晓）")
        else:
            err(f"H: {name} [{lo:+.2f}, {hi:+.2f}] 超出 {axis} 幅值上限 {mag:.3f}，"
                f"该档位将缺乏参考密度")

    # --- 4) 前提假设 ---------------------------------------------------------
    src = COMMANDS_PY.read_text(encoding="utf-8")
    for cond, why in (
        ("heading_command=False", "heading_command 必须为 False，否则 wz 采样会被航向环重算"),
        ("rel_standing_envs=0", "rel_standing_envs 必须为 0，否则两套站立语义冲突"),
        ("vel_command_b", "必须依赖父类 vel_command_b 缓冲"),
        ("UniformVelocityCommand", "必须继承 IsaacLab 的 UniformVelocityCommand"),
    ):
        if cond not in src:
            err(f"H: {TARGET.name}/commands.py 缺少关键前提 “{cond}”（{why}）")
        else:
            print(f"   ✅ 包含前提 “{cond}”")


# ---------------------------------------------------------------------------
# I. AMP 判别器状态：amp_policy / amp_reference 必须逐项同构
# ---------------------------------------------------------------------------

def _obs_term_spec(cls: ast.ClassDef) -> dict[str, dict]:
    """提取一个 ObsGroup 里每个 ObsTerm 的关键参数。"""
    out: dict[str, dict] = {}
    for name, call in child_assignments(cls):
        if call_func_name(call) != "ObsTerm":
            continue
        spec: dict = {"func": "", "history": None, "flatten": False, "scale": None}
        for kw in call.keywords:
            if kw.arg == "func":
                spec["func"] = call_func_name(kw.value) if isinstance(kw.value, ast.Call) else (
                    kw.value.id if isinstance(kw.value, ast.Name) else ast.unparse(kw.value))
            elif kw.arg == "history_length":
                spec["history"] = literal(kw.value)
            elif kw.arg == "flatten_history_dim":
                spec["flatten"] = bool(literal(kw.value))
            elif kw.arg == "scale":
                spec["scale"] = literal(kw.value)
        out[name] = spec
    return out


def check_amp_state_isomorphism(tree: ast.Module) -> None:
    """校验判别器两侧状态向量**逐项同构**。

    为什么必须查：WASABI 在 `init_storage` 会断言 `amp_policy` 与 `amp_reference`
    两侧的 `get_subobs_size` 相等；不等就直接报错。而“不等”很容易因为
    改了 history_length / flatten / scale 而惄悄发生。
    """
    print("I. AMP 判别器状态同构（amp_policy vs amp_reference）")
    classes = class_defs(tree)

    # 1) 契约常量自洽
    terms = assigned_dict(tree, "AMP_STATE_TERMS")
    dims = assigned_dict(tree, "AMP_STATE_DIMS")
    size = assigned_dict(tree, "AMP_STATE_SIZE")
    hist = assigned_dict(tree, "AMP_STATE_HISTORY")
    if not terms or not dims:
        err("I: 找不到 AMP_STATE_TERMS / AMP_STATE_DIMS")
        return
    if not isinstance(hist, int) or hist <= 0:
        err(f"I: AMP_STATE_HISTORY 解析失败（{hist!r}）")
        return
    print(f"   AMP_STATE_TERMS = {terms}")
    print(f"   AMP_STATE_DIMS  = {dims}   (history={hist})")
    if len(terms) != len(dims):
        err(f"I: AMP_STATE_TERMS({len(terms)}) 与 AMP_STATE_DIMS({len(dims)}) 长度不一致")
        return
    if isinstance(size, int) and size != EXPECTED_AMP_STATE_SIZE:
        err(f"I: AMP_STATE_SIZE={size} 期望 {EXPECTED_AMP_STATE_SIZE}")

    # 按声明独立复算维度
    recomputed = []
    for t, d in zip(terms, dims):
        base = AMP_BASE_DIMS.get(t)
        if base is None:
            err(f"I: 未知状态项 `{t}`，请在 AMP_BASE_DIMS 中补充其单帧维度")
            return
        self_flat = (len(d) == 1)
        recomputed.append((base * hist,) if self_flat else (hist, base))
    if tuple(recomputed) != tuple(tuple(d) for d in dims):
        err(f"I: 独立复算的维度 {tuple(recomputed)} != 声明 {tuple(tuple(d) for d in dims)}")
    else:
        total = sum(math.prod(d) for d in dims)
        print(f"   ✅ 独立复算维度一致，合计 {total}")
        if isinstance(size, int) and size != total:
            err(f"I: AMP_STATE_SIZE={size} != 复算总和 {total}")

    # 2) 两侧逐项同构
    p_cls, ref_cls = classes.get("AmpPolicyStateObsCfg"), classes.get("AmpReferenceStateObsCfg")
    if not (p_cls and ref_cls):
        err("I: 找不到 AmpPolicyStateObsCfg / AmpReferenceStateObsCfg")
        return
    pol, ref = _obs_term_spec(p_cls), _obs_term_spec(ref_cls)

    if tuple(pol) != tuple(terms):
        err(f"I: amp_policy 声明项 {tuple(pol)} != AMP_STATE_TERMS {tuple(terms)}")
    if tuple(ref) != tuple(terms):
        err(f"I: amp_reference 声明项 {tuple(ref)} != AMP_STATE_TERMS {tuple(terms)}")
    if tuple(pol) != tuple(ref):
        err(f"I: 两侧项集/顺序不同：policy={tuple(pol)} reference={tuple(ref)}")
        return

    for name in pol:
        p, r = pol[name], ref[name]
        tag = f"{name}"
        for key in ("history", "flatten", "scale"):
            if p[key] != r[key]:
                err(f"I: 项 `{tag}` 的 {key} 不一致：policy={p[key]} reference={r[key]}")
            else:
                print(f"   ✅ {tag:18s} {key}={p[key]!r}")
        expected_ref = f"{p['func'].rsplit('.', 1)[-1]}_reference_as_state"
        if r["func"].rsplit(".", 1)[-1] != expected_ref:
            err(f"I: 项 `{tag}` 参考侧应使用 `{expected_ref}`，实际 `{r['func']}`")
        else:
            print(f"   ✅ {tag:18s} {p['func']}  ↔  {r['func']}")

    # 3) 参考动作目录确实存在
    n_npz = len(list(MOTION_DIR.glob("*_retargeted.npz")))
    print(f"   参考动作目录: {MOTION_DIR.relative_to(REPO)} （{n_npz} 个 npz）")
    if n_npz == 0:
        err("I: 参考动作目录里没有 *_retargeted.npz")
    src = TARGET.read_text(encoding="utf-8")
    if "motion_start_from_middle_range = (0.0, 0.0)" not in src:
        err("I: 行走是循环步态，必须关闭 RSI（motion_start_from_middle_range = (0.0, 0.0)）")
    else:
        print("   ✅ 已关闭 RSI 随机起点（循环步态必需）")
    if "motion_bin_length_s = None" not in src:
        err("I: 行走应设 motion_bin_length_s = None（不切 bin，整段连续播放）")
    else:
        print("   ✅ motion_bin_length_s = None")


# ---------------------------------------------------------------------------
# J. 奖励集迁移对账（步骤⑤：AMP_mjlab v5 → G1AmpWalkRewardsCfg）
# ---------------------------------------------------------------------------

def _reward_terms(path: Path, class_name: str) -> dict[str, dict]:
    """取一个 configclass 里的 `RewTerm(...)` 字段：{名: {weight, func, params}}。"""
    cls = class_defs(parse(path)).get(class_name)
    if cls is None:
        return {}
    out: dict[str, dict] = {}
    for name, call in child_assignments(cls):
        if call_func_name(call) != "RewTerm":
            continue
        spec: dict = {"weight": None, "func": "", "params": {}}
        for kw in call.keywords:
            if kw.arg == "weight":
                spec["weight"] = literal(kw.value)
            elif kw.arg == "func":
                v = kw.value
                if isinstance(v, ast.Call):
                    spec["func"] = call_func_name(v)
                elif isinstance(v, ast.Name):
                    spec["func"] = v.id
                else:
                    spec["func"] = ast.unparse(v)
            elif kw.arg == "params" and isinstance(kw.value, ast.Dict):
                for k, v in zip(kw.value.keys, kw.value.values):
                    key = string_value(k)
                    if key is not None:
                        spec["params"][key] = literal(v)
        out[name] = spec
    return out


def _num(spec: dict, key: str):
    v = spec["params"].get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _data_attr_precedents(exclude: Path) -> set[str]:
    """扫描仓库，收集所有 `...data.<attr>` 里出现过的属性名。"""
    out: set[str] = set()
    for py in INSTINCTLAB.rglob("*.py"):
        if py == exclude:
            continue
        try:
            t = parse(py)
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "data"
            ):
                out.add(node.attr)
    return out


def _used_data_attrs(path: Path) -> list[str]:
    return sorted({
        n.attr
        for n in ast.walk(parse(path))
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Attribute)
        and n.value.attr == "data"
    })


def check_reward_migration(tree: ast.Module) -> None:
    """把「奖励从 AMP_mjlab 迁移」这条要求变成**可执行的契约**。

    四件事：
    1. 证明**基类一个字没动**（否则会连带影响 OmniLoco/Vault 三条已训链路）；
    2. 证明迁移表逐项落地（覆盖项 / 新增项的权重）；
    3. 证明**没有**偷改其它项的权重（「不改权重」约束）；
    4. 把 `sep_floor` / `walk_max_lin_vel` 与①产出的**实测标定数据**做数值对账。
    """
    print("J. 奖励集迁移对账（步骤⑤，AMP_mjlab v5 → G1AmpWalkRewardsCfg）")

    base = _reward_terms(FLAT_CFG, "G1FlatRewardsCfg")
    if not base:
        err("J: 未能解析 G1FlatRewardsCfg（基类奖励集）")
        return

    # --- 1) 基类零改动 ------------------------------------------------------
    bad = []
    for name, want in BASE_REWARD_WEIGHTS_UNCHANGED.items():
        got = base.get(name, {}).get("weight")
        if got is None:
            bad.append(f"{name} 缺失")
        elif abs(float(got) - float(want)) > 1e-12:
            bad.append(f"{name}={got}（应为 {want}）")
    if bad:
        err("J: 基类 G1FlatRewardsCfg 的权重被动过！" + "; ".join(bad))
    else:
        print(f"   ✅ 基类 G1FlatRewardsCfg 的 {len(BASE_REWARD_WEIGHTS_UNCHANGED)} 项权重原样未动")
        print("      → OmniLoco / Vault / flat 三条已训链路不受影响")

    # --- 2) 子类接线 --------------------------------------------------------
    classes = class_defs(tree)
    sub_bases = _bases_of(classes, "G1AmpWalkRewardsCfg")
    if not any(b.startswith("G1FlatRewardsCfg") for b in sub_bases):
        err(f"J: G1AmpWalkRewardsCfg 未继承 G1FlatRewardsCfg（基类={sub_bases}）")
    env_fields = _ann_fields(classes.get("G1AmpWalkEnvCfg")) if "G1AmpWalkEnvCfg" in classes else {}
    if "rewards" not in env_fields or not env_fields["rewards"].startswith("G1AmpWalkRewardsCfg"):
        err(f"J: G1AmpWalkEnvCfg.rewards 未接到 G1AmpWalkRewardsCfg（实际 {env_fields.get('rewards')!r}）")
    else:
        print(f"   ✅ G1AmpWalkEnvCfg.rewards = {env_fields['rewards']}")

    sub = _reward_terms(TARGET, "G1AmpWalkRewardsCfg")
    if not sub:
        err("J: 未能解析 G1AmpWalkRewardsCfg")
        return

    # --- 3) 覆盖项 / 新增项的权重对账 ---------------------------------------
    for name, want in AMP_WALK_REWARD_OVERRIDES.items():
        spec = sub.get(name)
        if spec is None:
            err(f"J: 缺少覆盖项 '{name}'（应为 {want}）")
        elif spec["weight"] is None or abs(float(spec["weight"]) - want) > 1e-12:
            err(f"J: '{name}' 权重 {spec['weight']} != 迁移表期望 {want}")
        else:
            print(f"   ✅ 覆盖 {name:26s} weight={want:+g}（基类 {base.get(name, {}).get('weight'):+g}）")

    for name, want in AMP_WALK_REWARD_NEW.items():
        spec = sub.get(name)
        if spec is None:
            err(f"J: 缺少新增项 '{name}'（应为 {want}）")
        elif spec["weight"] is None or abs(float(spec["weight"]) - want) > 1e-12:
            err(f"J: '{name}' 权重 {spec['weight']} != 迁移表期望 {want}")
        else:
            print(f"   ✅ 新增 {name:26s} weight={want:+g}")

    # --- 4) 没有偷改其它项的权重 -------------------------------------------
    tampered = []
    for name, spec in sub.items():
        if name in AMP_WALK_REWARD_OVERRIDES or name in AMP_WALK_REWARD_NEW:
            continue
        base_w = base.get(name, {}).get("weight")
        if base_w is None:
            tampered.append(f"{name}（基类里没有这一项，却出现在子类）")
        elif spec["weight"] is None or abs(float(spec["weight"]) - float(base_w)) > 1e-12:
            tampered.append(f"{name}: {spec['weight']} != 基类 {base_w}")
    if tampered:
        err("J: 子类动了不该动的奖励项 —— " + "; ".join(tampered))
    else:
        others = [n for n in sub if n not in AMP_WALK_REWARD_OVERRIDES and n not in AMP_WALK_REWARD_NEW]
        print(f"   ✅ 其余 {len(others)} 个声明项权重与基类一致（未偷改）")

    # --- 5) 新项的 func 必须真的定义在 rewards_amp.py -----------------------
    if not REWARDS_AMP_PY.is_file():
        err(f"J: 缺少 {REWARDS_AMP_PY.relative_to(REPO)}")
    else:
        authored = {f"locomotion_mdp.{n}" for n in top_level_names(parse(REWARDS_AMP_PY))}
        init_src = (REWARDS_AMP_PY.parent / "__init__.py").read_text(encoding="utf-8")
        if "from .rewards_amp import *" not in init_src:
            err("J: locomotion/mdp/__init__.py 未转发 `from .rewards_amp import *`，新奖励项不可用")
        missing = [f"{n} → {sub[n]['func']}" for n in AMP_WALK_REWARD_NEW
                   if sub[n]["func"] not in authored]
        if missing:
            err("J: 新奖励项的 func 不在 mdp/rewards_amp.py 里定义：" + "; ".join(missing))
        else:
            print(f"   ✅ {len(AMP_WALK_REWARD_NEW)} 个新奖励项的 func 均在 mdp/rewards_amp.py 中定义")

    # --- 6) 步宽 / 交叉项的参数对账（vs 实测标定）---------------------------
    width = sub.get("feet_lateral_width", {}).get("params", {})
    cross = sub.get("feet_crossing", {}).get("params", {})
    for label, params in (("feet_lateral_width", width), ("feet_crossing", cross)):
        if params.get("gate_lateral") is not True:
            err(f"J: {label}.gate_lateral 必须为 True —— 实测侧移 clip 有 ~21% 帧 sep<0"
                "（交叉式侧步，属参考动作风格），关掉方向门控会与 AMP 风格先验对撞、压制侧移")
        if params.get("command_name") != "base_velocity":
            err(f"J: {label}.command_name 应为 'base_velocity'（实际 {params.get('command_name')!r}）")
        if params.get("walk_max_lin_vel") is None:
            warn(f"J: {label} 未设 walk_max_lin_vel，慢跑摆动腿内收会被误罚")

    worst = []
    for label, params in (("feet_lateral_width", width), ("feet_crossing", cross)):
        mc = params.get("max_cost")
        if isinstance(mc, (int, float)):
            worst.append((label, abs(AMP_WALK_REWARD_NEW[label]) * float(mc)))
    for label, v in worst:
        flag = "⚠️" if v > 2.5 else "✅"
        print(f"   {flag} {label:20s} 单帧最大惩罚 = |{AMP_WALK_REWARD_NEW[label]:g}| × "
              f"max_cost = {v:.2f}（硬封顶，与 AMP_mjlab 的 max_cost 一致）")

    # --- 7) sep_floor vs 实测标定 ------------------------------------------
    sep_floor = _num(sub.get("feet_lateral_width", {}), "sep_floor")
    walk_max = _num(sub.get("feet_lateral_width", {}), "walk_max_lin_vel")
    if sep_floor is None:
        err("J: feet_lateral_width 缺少 sep_floor")
        return
    if not FOOT_SEP_STATS.is_file():
        warn(f"J: 找不到标定文件 {FOOT_SEP_STATS.relative_to(REPO)}，跳过 sep_floor 对账"
             "（先跑 tools/calibrate_foot_sep.py）")
    else:
        stats = json.loads(FOOT_SEP_STATS.read_text(encoding="utf-8"))
        hw = stats["healthy_walk"]
        p1, lo = float(hw["p1"]), float(hw["min"])
        print(f"   实测（直线行走 {hw['n_clips']} 段 / {hw['n_frames']} 帧）: "
              f"min={lo:.4f}  p1={p1:.4f}  中位={float(hw['median']):.4f}  "
              f"脚部参考 body={stats['foot_bodies']['left']}/{stats['foot_bodies']['right']}")
        print(f"   配置 sep_floor = {sep_floor:.3f}")
        if sep_floor <= 0.0:
            err(f"J: sep_floor={sep_floor} 必须为正")
        elif sep_floor > p1 + 1e-9:
            err(f"J: sep_floor={sep_floor} 高于直线行走实测 p1={p1:.4f} → "
                f"健康步态会被误罚（AMP_mjlab 的教训：地板要放在健康下探值之下）")
        else:
            print(f"   ✅ sep_floor ≤ 直线行走 p1（健康步态被罚的帧 ≲1%）；"
                  f"最坏下探 {lo:.4f} ≥ sep_floor−max_cost 之下也不误伤")
        # 侧移 clip 的交叉占比 —— gate_lateral 的经验依据
        lat = [stats["per_clip"][c] for c in CROSS_STEP_CLIPS if c in stats["per_clip"]]
        if lat:
            neg = max(float(r["neg_frac"]) for r in lat)
            print(f"   交叉式侧步实测 sep<0 帧占比（最大）= {neg:.3f}")
            if neg < 0.05:
                warn("J: 侧移 clip 已几乎不交叉 → gate_lateral=True 的依据变弱，可考虑复核")
        for c in FOOT_SEP_WALK_CLIPS:
            r = stats["per_clip"].get(c)
            if r and float(r["neg_frac"]) > 0.0:
                err(f"J: 直线行走 clip {c} 出现 {r['neg_frac']:.4f} 帧 sep<0；"
                    f"若属实则 sep 的符号/左右定义可能反了")

    # --- 8) walk_max_lin_vel 必须高于直线行走速度 ---------------------------
    if walk_max is not None:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import convert_amp_motion as cam  # noqa: PLC0415
            speeds = [
                abs(cam.body_frame_stats(MOTION_DIR / f"{c}_retargeted.npz")["vx"])
                for c in FOOT_SEP_WALK_CLIPS
                if (MOTION_DIR / f"{c}_retargeted.npz").is_file()
            ]
        except Exception as e:  # pragma: no cover
            speeds = []
            warn(f"J: 无法读取直线行走速度（{e}），跳过 walk_max_lin_vel 对账")
        if speeds:
            top = max(speeds)
            print(f"   直线行走实测 |vx| 最大 = {top:.3f} m/s；配置 walk_max_lin_vel = {walk_max:.2f}")
            if walk_max <= top:
                err(f"J: walk_max_lin_vel={walk_max} ≤ 直线行走速度 {top:.3f} → 正常行走会被门控掉一部分")
            else:
                print("   ✅ 门控上限高于直线行走速度，正常行走不受速档豁免影响")

    # --- 9) 新奖励函数用到的 `asset.data.<attr>` 必须有仓库先例 --------------
    if REWARDS_AMP_PY.is_file():
        precedents = _data_attr_precedents(REWARDS_AMP_PY)
        used = _used_data_attrs(REWARDS_AMP_PY)
        unknown = [a for a in used if a not in precedents and a not in ALLOWED_DATA_ATTRS]
        allowed_used = [a for a in used if a in ALLOWED_DATA_ATTRS and a not in precedents]
        if unknown:
            err("J: rewards_amp.py 用到的 asset.data 属性在本仓库无先例（运行时可能 AttributeError）："
                + ", ".join(unknown))
        else:
            print(f"   ✅ {len(used)} 个 asset.data.* 属性均有仓库先例"
                  + (f"（{', '.join(allowed_used)} 走 allowlist）" if allowed_used else ""))


# ---------------------------------------------------------------------------

def main() -> int:
    if not TARGET.is_file():
        print(f"找不到 {TARGET}")
        return 1
    tree = parse(TARGET)
    _CONSTS.clear()
    _CONSTS.update(module_consts(tree))
    print(f"目标: {TARGET.relative_to(REPO)}\n" + "=" * 72)
    check_dims(tree)
    print("-" * 72)
    check_obs_order(tree)
    print("-" * 72)
    check_symbols()
    print("-" * 72)
    check_registration()
    print("-" * 72)
    check_teacher_contract(tree)
    print("-" * 72)
    check_wiring(tree)
    print("-" * 72)
    check_scan_rays(tree)
    print("-" * 72)
    check_command_envelope(tree)
    print("-" * 72)
    check_amp_state_isomorphism(tree)
    print("-" * 72)
    check_reward_migration(tree)
    print("=" * 72)

    for w in _warns:
        print(f"⚠️  {w}")
    if _errors:
        print(f"\n❌ 静态契约校验失败（{len(_errors)} 项）：")
        for e in _errors:
            print(f"   - {e}")
        return 1
    print("\n✅ 静态契约校验全部通过（注意：这不能替代云机器上的真实冒烟测试）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
