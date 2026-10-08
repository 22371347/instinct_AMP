#!/usr/bin/env python3
"""V30 启动前静态核验（**不依赖 isaaclab**，纯源码 AST）。

核验"V30 = V29 + **判别器四件事**"，且**没有偷改别的东西**：

  [A] V30 agent cfg：基类 = V29；`self.algorithm.*` 的改动**恰好**是
      {GP coef, gp_expert_only, weight_decay_coef, optimizer_trunk_weight_decay,
       optimizer_head_weight_decay, amp_expert_buffer_path}；
      且**没碰** `discriminator_reward_coef` / `style_lerp` / 优化器 lr / `discriminator_kwargs`
      （去白化）/ σ / `amp_reward_form` / `amp_quad_k`（V29 已对齐的必须原样）。
  [B] 常数：GP=10.0 · expert_only=True · trunk wd=1e-3 · head wd=1e-2 · buffer=/root/amp_expert_states.pt
  [C] V30 env cfg：基类 = V29；只改 amp_policy/amp_reference/run_name。
  [D] 库改动在位（本轮三个库文件）：wasabi 的 ①②③④、amp_storage 的 ④ 分支、wrapper 的探针里程碑。
  [E] 专家数据集存在且维度 == 390（`torch.load` 真读一次）。
  [F] 注册 / `drop_amp_discriminator` / V27 热启动 ckpt。
  [G] （信息）运行时导入尝试。

结果落 `/root/v30_verify_result.txt`，末尾 `[V30 PASS]`。
"""

from __future__ import annotations

import ast
import glob
import os
import re
import sys
import traceback

G1 = "/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1"
AG = f"{G1}/agents"
IR = "/root/gpufree-data/instinct_xu/source/instinct_rl/instinct_rl"
WRAP = ("/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab/utils/wrappers/"
        "instinct_rl/algorithms.py")
RUN_ROOT = "/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk"
BUF = "/root/amp_expert_states.pt"
RESULT_PATH = "/root/v30_verify_result.txt"

E_GP = "discriminator_gradient_penalty_coef"
E_GPO = "discriminator_gp_expert_only"
E_WD = "discriminator_weight_decay_coef"
E_WDT = "discriminator_optimizer_trunk_weight_decay"
E_WDH = "discriminator_optimizer_head_weight_decay"
E_BUF = "amp_expert_buffer_path"
MUST_NOT_TOUCH = (
    "discriminator_reward_coef", "style_lerp", "discriminator_optimizer_kwargs",
    "discriminator_kwargs", "discriminator_logit_weight_decay_coef",
    "amp_reward_form", "amp_quad_k", "clip_max_std", "clip_max_std_schedule",
    "amp_grad_pen_space", "amp_expert_independent_sampling",
)

_lines: list[str] = []
bad: list[str] = []


def out(msg: str = "") -> None:
    print(msg, flush=True)
    _lines.append(str(msg))


def ok(msg: str) -> None:
    out(f"  ✅ {msg}")


def fail(msg: str) -> None:
    bad.append(msg)
    out(f"  ❌ {msg}")


def read(p: str) -> str:
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def dotted(node) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def post_init_assigns(path: str, cls_suffix: str):
    tree = ast.parse(read(path))
    bases: list[str] = []
    deep: set[str] = set()
    shallow: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name.endswith(cls_suffix)):
            continue
        bases += [ast.unparse(b) for b in node.bases]
        for sub in ast.walk(node):
            if not (isinstance(sub, ast.Assign) and len(sub.targets) == 1):
                continue
            path_ = dotted(sub.targets[0])
            if path_ is None or not path_.startswith("self."):
                continue
            rest = path_[len("self."):]
            (deep if "." in rest else shallow).add(rest)
    return bases, deep, shallow


def literals(path: str) -> dict[str, object]:
    d: dict[str, object] = {}
    for node in ast.parse(read(path)).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                d[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:  # noqa: BLE001
                if isinstance(node.value, ast.Name):
                    d[node.targets[0].id] = node.value.id
    return d


def main() -> int:  # noqa: C901
    out("=" * 88)
    out("V30 静态核验：判别器 = mjlab 原始配方（GP λ=10 / 专家侧 / 优化器分组 wd / 离线专家批）")
    out("=" * 88)

    f_a = f"{AG}/amp_walk_v30_ppo_cfg.py"
    f_e = f"{G1}/amp_walk_v30_env_cfg.py"
    files = [f_a, f_e, f"{IR}/algorithms/wasabi.py", f"{IR}/storage/amp_storage.py", WRAP]
    for p in files:
        if not os.path.exists(p):
            fail(f"缺文件 {p}")
            out("")
            out("[V30 FAIL] 文件缺失")
            return 1

    # ---------------- [A] agent cfg 改动面 ----------------
    out("\n[A] V30 agent cfg 的改动面")
    bases, deep, shallow = post_init_assigns(f_a, "PPORunnerCfg")
    out(f"      基类: {bases}")
    out(f"      algorithm 侧: {sorted(deep)}")
    out(f"      self 侧: {sorted(shallow)}")
    if bases == ["G1AmpWalkV29PPORunnerCfg"]:
        ok("基类 = G1AmpWalkV29PPORunnerCfg")
    else:
        fail(f"基类意外: {bases}")
    expect = {f"algorithm.{x}" for x in (E_GP, E_GPO, E_WD, E_WDT, E_WDH, E_BUF)}
    if deep == expect:
        ok("algorithm 侧改动恰好 = ①②③④ 四项")
    else:
        fail(f"algorithm 侧改动集意外：期望 {sorted(expect)}，实际 {sorted(deep)}")
    src_a = read(f_a)
    touched = [x for x in MUST_NOT_TOUCH if re.search(rf"self\.algorithm\.{x}\s*=", src_a)]
    if touched:
        fail(f"不该动却动了: {touched}")
    else:
        ok("未碰风格强度/λ/优化器 lr/去白化/σ/形状（这些在 V29 已对齐，必须原样）")

    # ---------------- [B] 常数 ----------------
    out("\n[B] 常数（V30 生效值）")
    lit = literals(f_a)
    shown = {k: v for k, v in lit.items() if k.endswith("_V30")}
    out(f"      {shown}")
    for name, want in (("GP_COEF_V30", 10.0), ("GP_EXPERT_ONLY_V30", True),
                       ("OPT_WD_TRUNK_V30", 1.0e-3), ("OPT_WD_HEAD_V30", 1.0e-2),
                       ("EXPERT_BUFFER_V30", BUF)):
        got = lit.get(name)
        same = (got == want) if isinstance(want, (str, bool)) else (
            isinstance(got, (int, float)) and abs(got - want) < 1e-15)
        if same:
            ok(f"{name} = {got!r}")
        else:
            fail(f"{name} 期望 {want!r}，实际 {got!r}")

    # ---------------- [C] env cfg ----------------
    out("\n[C] V30 env cfg（应与 V29 逐字一致，只换 run_name）")
    bases_e, deep_e, shallow_e = post_init_assigns(f_e, "EnvCfg")
    out(f"      基类: {bases_e} / 带点: {sorted(deep_e)} / 裸属性: {sorted(shallow_e)}")
    if bases_e and all("V29" in b for b in bases_e):
        ok(f"env 基类来自 V29: {bases_e}")
    else:
        fail(f"env 基类意外: {bases_e}")
    allowed = {"run_name", "observations.amp_policy", "observations.amp_reference"}
    if (shallow_e | deep_e) <= allowed:
        ok("只改 amp_policy / amp_reference / run_name")
    else:
        fail(f"赋值面意外: {sorted((shallow_e | deep_e) - allowed)}")
    hit = [ln.strip() for ln in read(f_e).splitlines()
           if re.search(r"self\.(rewards|scene|events|commands)\b", ln) and "=" in ln
           and not ln.strip().startswith("#")]
    if hit:
        fail(f"env cfg 动了 env 组件: {hit[:3]}")
    else:
        ok("没动 rewards / scene / events / commands")

    # ---------------- [D] 库改动 ----------------
    out("\n[D] 库改动（本轮三个文件）")
    s_was = read(f"{IR}/algorithms/wasabi.py")
    checks = [
        (rf"{E_GP}\s*=\s*10\.0", "wasabi：GP 默认值含 10.0"),
        (rf"{E_GPO}=False", "wasabi：`__init__` 接受 gp_expert_only"),
        (rf"self\.{E_GPO}\s*=", "wasabi：gp_expert_only 已保存"),
        (r"expert_only: bool = False", "wasabi：GP 函数支持 expert_only"),
        (r"combined_states = reference_states", "wasabi：expert_only 分支（只对参考求 GP）"),
        (r"expert_only=self\.discriminator_gp_expert_only", "wasabi：GP 调用点已传 expert_only"),
        (rf"self\.{E_WDT}\s*=", "wasabi：优化器 trunk wd 已保存"),
        (rf'"weight_decay": _wd_trunk', "wasabi：优化器分组 weight_decay（trunk）"),
        (rf'"weight_decay": _wd_head', "wasabi：优化器分组 weight_decay（head）"),
        (r"self\.amp_storage\.expert_buffer = _buf", "wasabi：载入离线专家 buffer"),
    ]
    for pat, label in checks:
        (ok if re.search(pat, s_was) else fail)(label)
    s_st = read(f"{IR}/storage/amp_storage.py")
    for pat, label in ((r"self\.expert_buffer = None", "amp_storage：expert_buffer 属性"),
                       (r"if self\.expert_buffer is not None:", "amp_storage：专家 buffer 分支"),
                       (r"torch\.randint\(0, self\.expert_buffer\.shape\[0\]", "amp_storage：i.i.d. 抽行")):
        (ok if re.search(pat, s_st) else fail)(label)
    s_w = read(WRAP)
    for pat, label in ((r"_amp_probe_steps_per_iter", "wrapper：探针按迭代里程碑"),
                       (r'f"/root/amp_dim_probe_it\{_it:04d\}\.txt"', "wrapper：探针文件名带迭代号")):
        (ok if re.search(pat, s_w) else fail)(label)

    # ---------------- [E] 专家数据集 ----------------
    out("\n[E] 专家数据集（真读一次）")
    if os.path.exists(BUF):
        try:
            import torch
            obj = torch.load(BUF, map_location="cpu")
            arr = obj["states_pair"] if isinstance(obj, dict) else obj
            out(f"      {BUF}: {os.path.getsize(BUF) / 1e6:.1f} MB  shape={tuple(arr.shape)}")
            if tuple(arr.shape)[1] == 390:
                ok("维度 = 390（== AMP 状态 `195×2`）")
            else:
                fail(f"维度 {tuple(arr.shape)[1]} != 390")
            if tuple(arr.shape)[0] > 1000:
                ok(f"样本数 {tuple(arr.shape)[0]}（足够 i.i.d. 抽样）")
            else:
                fail(f"样本数过少: {tuple(arr.shape)[0]}")
            if isinstance(obj, dict) and "num_frames" in obj:
                out(f"      帧数={obj.get('num_frames')} clip={obj.get('layout')}")
        except Exception as exc:  # noqa: BLE001
            fail(f"读取失败: {exc!r}")
    else:
        fail(f"缺 {BUF}（需先跑 `cloud_export_amp_expert.py`）")

    # ---------------- [F] 注册 / manipulator / ckpt ----------------
    out("\n[F] 注册 / manipulator / 热启动 ckpt")
    init_src = read(f"{G1}/__init__.py")
    for tid in ("Instinct-Amp-Walk-V30-G1-v0", "Instinct-Amp-Walk-V30-G1-Play-v0",
                "Instinct-Amp-Walk-V29-G1-v0"):
        (ok if tid in init_src else fail)(f"注册 {tid}")
    if "amp_walk_v30_ppo_cfg:G1AmpWalkV30PPORunnerCfg" in init_src:
        ok("V30 entry_point 指向 V30 的 PPO cfg")
    else:
        fail("V30 entry_point 不对")
    mp = "/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab/utils/ckpt_manipulators.py"
    if os.path.exists(mp) and "def drop_amp_discriminator" in read(mp):
        ok("drop_amp_discriminator 存在")
    else:
        fail("drop_amp_discriminator 缺失")
    picked = None
    for d in sorted(glob.glob(f"{RUN_ROOT}/*v27*"), key=os.path.getmtime, reverse=True):
        cks = sorted(glob.glob(f"{d}/model_*.pt"),
                     key=lambda q: int(re.search(r"model_(\d+)\.pt", q).group(1)))
        if cks:
            picked = (d, cks[-1])
            break
    (ok if picked else fail)(f"{os.path.basename(picked[0])} / {os.path.basename(picked[1])}"
                             if picked else "找不到 V27 的 model_*.pt")

    # ---------------- [G] 运行时（信息）----------------
    out("\n[G] （信息）运行时导入尝试（失败不算不通过）")
    try:
        from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v30_ppo_cfg import (  # noqa: F401
            G1AmpWalkV30PPORunnerCfg,
        )
        alg = G1AmpWalkV30PPORunnerCfg().algorithm
        out(f"      运行时: GP={alg.discriminator_gradient_penalty_coef} "
            f"expert_only={getattr(alg, 'discriminator_gp_expert_only', None)} "
            f"wd(cfg)={alg.discriminator_weight_decay_coef} "
            f"wd_opt=({getattr(alg, 'discriminator_optimizer_trunk_weight_decay', None)},"
            f"{getattr(alg, 'discriminator_optimizer_head_weight_decay', None)}) "
            f"buf={getattr(alg, 'amp_expert_buffer_path', None)} "
            f"coef={alg.discriminator_reward_coef} form={alg.amp_reward_form}")
    except Exception as exc:  # noqa: BLE001
        out(f"      ⚠️ 运行时导入不可用（{type(exc).__name__}: {exc}）⇒ 以静态核验为准")

    out("")
    if bad:
        out(f"[V30 FAIL] {len(bad)} 项未通过：{bad}")
        return 1
    out("[V30 PASS] mjlab 判别器四件事就位；风格强度/λ/去白化/形状与 V29 逐字一致")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        out("=" * 88)
        out("[V30 FAIL] 核验脚本自身抛异常：")
        out(traceback.format_exc())
        rc = 2
    with open(RESULT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print(f"\n（结果已写入 {RESULT_PATH}）")
    sys.exit(rc)
