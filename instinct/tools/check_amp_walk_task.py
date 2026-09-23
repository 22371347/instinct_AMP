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
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent            # .../llp
INSTINCTLAB = REPO / "instinct_xu/source/instinctlab/instinctlab"
G1_DIR = INSTINCTLAB / "tasks/locomotion/config/g1"
TARGET = G1_DIR / "amp_walk_env_cfg.py"
COMMANDS_PY = INSTINCTLAB / "tasks/locomotion/mdp/commands.py"
MOTION_DIR = G1_DIR / "motions/amp_walk"

EXPECTED_SIZE = 626

#: 允许“仓库中无先例”的符号，但必须已经在代码里用 try/except 显式守卫。
#: 值 = 理由。这类符号会在输出里记 ⚠️ 而不是 ❌。
ALLOWED_NEW_SYMBOLS = {
    "isaaclab.envs.mdp.UniformVelocityCommand":
        "仓库内只见过 mdp.UniformVelocityCommandCfg；类那一半无先例，已在 commands.py 用 "
        "try/except ImportError 守卫并给出可读报错",
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
    """把 AST 字面量（含一元负号 / tuple）转成 Python 值。"""
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


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


def check_symbols() -> None:
    t = parse(TARGET)
    aliases = import_aliases(t)
    proven = build_proven_symbols(TARGET) | self_authored_exports()

    used: dict[str, list[str]] = {}
    for node in ast.walk(t):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            mod = aliases.get(node.value.id)
            if mod:
                used.setdefault(f"{mod}.{node.attr}", []).append(node.value.id)

    print("C. 符号真实性（每个 <module>.<name> 必须在仓库其它处已出现过）")
    for sym in sorted(used):
        if sym in proven:
            print(f"   ✅ {sym}")
        elif sym in ALLOWED_NEW_SYMBOLS:
            print(f"   ⚠️  {sym}  (已允许：{ALLOWED_NEW_SYMBOLS[sym]})")
        else:
            print(f"   ❌ {sym}")
            err(f"C: 符号 `{sym}` 在仓库其它文件中从未出现 —— 无法证明其存在")

    # 额外的高价值专项检查：instinctlab.envs.mdp 没有转发 base_lin_vel
    mdp_exports = set()
    for py in (INSTINCTLAB / "envs/mdp").rglob("*.py"):
        try:
            mdp_exports |= top_level_names(parse(py))
        except SyntaxError:
            pass
    inst_lin = "instinctlab.envs.mdp.base_lin_vel" in {
        f"instinctlab.envs.mdp.{n}" for n in mdp_exports
    }
    src = TARGET.read_text(encoding="utf-8")
    if "instinct_mdp.base_lin_vel" in src and not inst_lin:
        err("C: 使用了 instinct_mdp.base_lin_vel，但 instinctlab.envs.mdp 并未导出该名字")
    else:
        print("   ✅ base_lin_vel 取自 isaaclab 的 mdp（instinctlab 未转发，判断正确）")
    if "instinct_mdp.base_lin_vel" in src:
        err("C: 不应经 instinct_mdp 访问 base_lin_vel")
    else:
        print("   ✅ 未误用 instinct_mdp.base_lin_vel")


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
        ("obs.policy  -> G1AmpWalkPolicyObsCfg",
         _ann_fields(obs).get("policy") == "G1AmpWalkPolicyObsCfg"),
        ("obs.critic  -> G1FlatCriticObsCfg（本就含 base_lin_vel）",
         _ann_fields(obs).get("critic") == "G1FlatCriticObsCfg"),
        ("_PLAY 继承 G1AmpWalkEnvCfg",
         "G1AmpWalkEnvCfg" in [ast.unparse(b) for b in env_play.bases]),
        ("_PLAY.scene 已改为 num_envs=1",
         "num_envs=1" in _play_scene_src(env_play)),
        ("env 未覆盖 scene（必须继承 527 射线的高程图）",
         "scene" not in _ann_fields(env)),
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

def main() -> int:
    if not TARGET.is_file():
        print(f"找不到 {TARGET}")
        return 1
    tree = parse(TARGET)
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
