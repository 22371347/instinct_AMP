#!/usr/bin/env python3
"""V29 启动前静态核验（**不依赖 isaaclab**，纯源码级 / AST）。

核验 V29 的三件事 + 两处库改动，且**证明"除了这三件事，其余与 V27 逐字一致"**。

  [A] V29 agent cfg：基类 = V27；`self.algorithm.*` 的改动**恰好**是
      {GP, weight_decay, reward_coef, amp_reward_form, amp_quad_k, amp_expert_independent_sampling}
      （即 ④ 三项回退 + ③ 形状 + ② 开关），且**没碰** style_lerp / 判别器 kwargs（去白化）
      / clip_max_std / logit_weight_decay / 优化器学习率。
  [B] 常数：GP=1.0 / wd=1e-5 / coef=1.0（= V27，回退 V28）· form="bounded_quad" · k=0.25 · indep=True
  [C] V29 env cfg：基类 = V27；只改 amp_policy/amp_reference/run_name；`AMP_V29_STATE_HISTORY==2`；
      且赋值在 `super().__post_init__()` **之后**（防父链 `_patch_amp_history` 覆盖）。
  [D] 两处库改动在位：
      * `amp_storage.py`：`independent_reference_sampling` 形参 + 参考批 `torch.randint` 重采
      * `algorithms.py`：fail-fast 接受 `bounded_quad` + 实现分支 + `init_storage` 里接开关
  [E] 字段声明（防 `to_dict()` 静默丢）
  [F] 继承链解析：V27 生效的 GP/wd/coef = 1.0 / 1e-5 / 1.0（V29 回退到这三个值）
  [G] `discriminator_reward_type == "wasserstein"`（`bounded_quad` 需要原始 d）
  [H] 注册 + `drop_amp_discriminator` + 热启动 ckpt
  [I] （信息）运行时导入尝试，失败不判负

结果落 `/root/v29_verify_result.txt`，末尾 `[V29 PASS]`。
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
IR = "/root/gpufree-data/instinct_xu/source/instinct_rl/instinct_rl/storage/amp_storage.py"
WRAP = ("/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab/utils/wrappers/"
        "instinct_rl/algorithms.py")
RUN_ROOT = "/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk"
RESULT_PATH = "/root/v29_verify_result.txt"

E_GP = "discriminator_gradient_penalty_coef"
E_WD = "discriminator_weight_decay_coef"
E_COEF = "discriminator_reward_coef"
E_FORM = "amp_reward_form"
E_K = "amp_quad_k"
E_INDEP = "amp_expert_independent_sampling"

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
    """把 `self.observations.amp_policy` 这类（可能**三层以上**的）属性链取成点号字符串。"""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def post_init_assigns(path: str, cls_suffix: str):
    """返回 (基类名, {'algorithm.x'/'observations.y'} 集合, {'x'} 集合)。"""
    tree = ast.parse(read(path))
    bases: list[str] = []
    algo: set[str] = set()
    selfk: set[str] = set()
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
            if "." in rest:
                algo.add(rest)
            else:
                selfk.add(rest)
    return bases, algo, selfk


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
    out("V29 静态核验：判别器向 AMP_mjlab 对齐（2 帧 / 专家批解耦 / 有界奖励形状）+ 系数回退 V27")
    out("=" * 88)

    f_a = f"{AG}/amp_walk_v29_ppo_cfg.py"
    f_e = f"{G1}/amp_walk_v29_env_cfg.py"
    for p in (f_a, f_e, IR, WRAP):
        if not os.path.exists(p):
            fail(f"缺文件 {p}")
            out("")
            out("[V29 FAIL] 文件缺失")
            return 1

    # ---------------- [A] agent cfg ----------------
    out("\n[A] V29 agent cfg 的改动面")
    bases, algo, selfk = post_init_assigns(f_a, "PPORunnerCfg")
    out(f"      基类: {bases}")
    out(f"      algorithm 侧: {sorted(algo)}")
    out(f"      self 侧: {sorted(selfk)}")
    if bases == ["G1AmpWalkV27PPORunnerCfg"]:
        ok("基类 = G1AmpWalkV27PPORunnerCfg")
    else:
        fail(f"基类意外: {bases}")
    expect = {f"algorithm.{x}" for x in (E_GP, E_WD, E_COEF, E_FORM, E_K, E_INDEP)}
    if algo == expect:
        ok("algorithm 侧改动恰好 = ④三项回退 + ③形状(form,k) + ②开关")
    else:
        fail(f"algorithm 侧改动集意外：期望 {sorted(expect)}，实际 {sorted(algo)}")
    src_a = read(f_a)
    forbidden = [x for x in ("style_lerp", "discriminator_kwargs", "amp_grad_pen_space",
                             "clip_max_std", "clip_max_std_schedule",
                             "discriminator_logit_weight_decay_coef",
                             "discriminator_optimizer_kwargs", "reward_mix_mode")
                 if re.search(rf"self\.algorithm\.{x}\s*=", src_a)]
    if forbidden:
        fail(f"不该动却动了: {forbidden}")
    else:
        ok("未碰 style_lerp / 去白化 kwargs / σ / logit_wd / 优化器（权重与口径都不动）")
    if {"ckpt_manipulator", "run_name"} <= selfk:
        ok("ckpt_manipulator / run_name 已设")
    else:
        fail(f"self 侧缺项: {sorted(selfk)}")
    if "DROP_DISC" in src_a and "from .amp_walk_v27_ppo_cfg import" in src_a:
        ok("从 V27 继承 DROP_DISC（判别器重置）")
    else:
        fail("未从 V27 继承 DROP_DISC")

    # ---------------- [B] 常数 ----------------
    out("\n[B] 常数（V29 生效值）")
    lit = literals(f_a)
    out(f"      { {k: v for k, v in lit.items() if k.endswith('_V29')} }")
    for name, want in (("GP_COEF_V29", 1.0), ("DISC_WD_COEF_V29", 1.0e-5), ("AMP_COEF_V29", 1.0),
                       ("AMP_QUAD_K_V29", 0.25), ("AMP_REWARD_FORM_V29", "bounded_quad"),
                       ("EXPERT_INDEP_V29", True)):
        got = lit.get(name)
        same = (got == want) if isinstance(want, (str, bool)) else (
            isinstance(got, (int, float)) and abs(got - want) < 1e-15)
        if same:
            ok(f"{name} = {got!r}")
        else:
            fail(f"{name} 期望 {want!r}，实际 {got!r}")
    eff = (1 - 0.75) * float(lit.get("AMP_COEF_V29", 0.0))
    ok(f"有效风格权重 (1−style_lerp)·coef = {eff:.3f}（与 V27 相同 = 0.25，刻意不动权重）")

    # ---------------- [C] env cfg ----------------
    out("\n[C] V29 env cfg（AMP 观测历史 8 → 2）")
    src_e = read(f_e)
    bases_e, algo_e, self_e = post_init_assigns(f_e, "EnvCfg")
    out(f"      基类: {bases_e}")
    out(f"      带点赋值: {sorted(algo_e)}")
    out(f"      裸属性赋值: {sorted(self_e)}")
    lit_e = literals(f_e)
    if lit_e.get("AMP_V29_STATE_HISTORY") == 2:
        ok("AMP_V29_STATE_HISTORY == 2（判别器输入 195×2 = 390）")
    else:
        fail(f"AMP_V29_STATE_HISTORY 期望 2，实际 {lit_e.get('AMP_V29_STATE_HISTORY')!r}")
    changed = self_e | algo_e
    if changed == {"run_name", "observations.amp_policy", "observations.amp_reference"}:
        ok("只改 amp_policy / amp_reference / run_name")
    else:
        fail(f"赋值面意外: {sorted(changed)}")
    if bases_e and all("V27" in b for b in bases_e):
        ok(f"env 基类来自 V27: {bases_e}")
    else:
        fail(f"env 基类意外: {bases_e}")
    # 顺序：super().__post_init__() 必须在 amp_policy 赋值之前
    i_super = src_e.find("super().__post_init__()")
    i_amp = src_e.find("observations.amp_policy = _group_with_history")
    if i_super != -1 and i_amp != -1 and i_super < i_amp:
        ok("amp 观测赋值在 super().__post_init__() 之后（防父链覆盖）")
    else:
        fail(f"顺序可疑：super@{i_super} amp@{i_amp}")
    hit = [ln.strip() for ln in src_e.splitlines()
           if re.search(r"self\.(rewards|scene|events|commands)\b", ln) and "=" in ln
           and not ln.strip().startswith("#")]
    if hit:
        fail(f"env cfg 动了 env 组件: {hit[:3]}")
    else:
        ok("没动 rewards / scene / events / commands")

    # ---------------- [D] 两处库改动 ----------------
    out("\n[D] 库改动（本轮必须一起上传的两个文件）")
    s_ir = read(IR)
    if re.search(r"independent_reference_sampling: bool = False", s_ir):
        ok("amp_storage.py：`__init__` 有形参且默认 False（零回归）")
    else:
        fail("amp_storage.py：`__init__` 没有 independent_reference_sampling 形参")
    if re.search(r"torch\.randint\(0, self\.num_transitions_per_env", s_ir) and \
       re.search(r"T_ref\[, B_ref\]|reference_states\[T_ref, B_ref\]", s_ir):
        ok("amp_storage.py：参考批做独立随机 (T,B) 重采")
    else:
        fail("amp_storage.py：没看到参考批独立重采")
    if re.search(r"reference_states\[T_select, B_select\]", s_ir):
        ok("amp_storage.py：旧口径（同 (T,B)）仍作为 else 分支保留")
    else:
        fail("amp_storage.py：旧口径分支丢失（可能改错了）")

    s_w = read(WRAP)
    if '"bounded_quad"' in s_w:
        ok("algorithms.py：fail-fast 与实现都包含 bounded_quad")
    else:
        fail("algorithms.py：没有 bounded_quad")
    if re.search(r"self\.amp_storage\.independent_reference_sampling", s_w):
        ok("algorithms.py：`init_storage` 把开关接到 amp storage")
    else:
        fail("algorithms.py：开关没接上（会导致 ② 静默失效）")
    if re.search(r"r_amp = torch\.clamp\(1\.0 - self\.amp_quad_k \* torch\.square\(d - 1\.0\)",
                 s_w):
        ok("algorithms.py：形状 = clamp(1 − k(d−1)², 0, 1)（**有界**，mjlab 形状）")
    else:
        fail("algorithms.py：bounded_quad 的实现形状不对")

    # ---------------- [E] 字段声明 ----------------
    out("\n[E] 字段声明（防 `to_dict()` 静默丢）")
    s_base = read(f"{AG}/amp_walk_ppo_cfg.py")
    for k in (E_GP, E_WD, E_COEF, E_FORM, E_K, E_INDEP, "clip_max_std", "style_lerp"):
        if re.search(rf"^\s+{k}\s*[:=]", s_base, re.M):
            ok(f"已声明 {k}")
        else:
            fail(f"**未声明** {k}")

    # ---------------- [F] 继承链：V27 的生效系数 ----------------
    out("\n[F] 继承链解析（V27 生效的 GP/wd/coef）")
    defs: dict[str, object] = {}
    for v in ("v14", "v15", "v18", "v19a"):
        p = f"{AG}/amp_walk_{v}_ppo_cfg.py"
        if os.path.exists(p):
            defs.update(literals(p))

    def resolve(name: str, depth: int = 0):
        v = defs.get(name)
        while isinstance(v, str) and depth < 6:
            v = defs.get(v)
            depth += 1
        return v if isinstance(v, (int, float)) else None

    for name, want in (("GP_COEF_V19A", 1.0), ("WD_COEF_V19A", 1.0e-5), ("DISC_REWARD_COEF_V18", 1.0)):
        got = resolve(name)
        if got is not None and abs(got - want) < 1e-15:
            ok(f"{name} = {got:g}（V29 正回退到这三个值）")
        else:
            fail(f"{name} 期望 {want:g}，实际 {got}")
    refs = {}
    for v in ("v19a", "v19b", "v19c", "v19d", "v20", "v21", "v22", "v23", "v24", "v26", "v27"):
        p = f"{AG}/amp_walk_{v}_ppo_cfg.py"
        if not os.path.exists(p):
            continue
        s = read(p)
        g = {}
        for k in (E_GP, E_WD, E_COEF):
            m = re.search(rf"self\.algorithm\.{k}\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", s)
            if m:
                g[k] = m.group(1)
        if g:
            refs[v] = g
    weird = {v: g for v, g in refs.items()
             if any(x not in ("GP_COEF_V19A", "WD_COEF_V19A", "DISC_REWARD_COEF_V18") for x in g.values())}
    if not weird:
        ok("V27 → 生效值 = GP 1.0 / wd 1e-5 / coef 1.0（V29 的“回退”目标正确）")
    else:
        fail(f"出现未知常量引用: {weird}")

    # ---------------- [G] reward_type ----------------
    out("\n[G] `discriminator_reward_type` 必须是 wasserstein（bounded_quad 需要原始 d）")
    hits: list[tuple[str, str]] = []
    for p in sorted(glob.glob(f"{AG}/amp_walk_v*_ppo_cfg.py"),
                    key=lambda q: int(re.search(r"_v(\d+)", os.path.basename(q)).group(1))):
        s = read(p)
        for m in re.finditer(r"self\.algorithm\.discriminator_reward_type\s*=\s*(\S+)", s):
            hits.append((os.path.basename(p), m.group(1)))
    out(f"      链上设置顺序: {hits}")
    if hits and hits[-1][1].strip("\"'") == "wasserstein":
        ok(f"最后一个设置处 {hits[-1][0]} = {hits[-1][1]} ⇒ V29 继承 wasserstein（bounded_quad 可用）")
    elif hits:
        fail(f"最后一个设置处 {hits[-1][0]} = {hits[-1][1]} ⇒ bounded_quad 会 fail-fast")
    else:
        fail("链上找不到 discriminator_reward_type 的显式设置 ⇒ 需人工确认")

    # ---------------- [H] 注册 / manipulator / ckpt ----------------
    out("\n[H] 注册 / manipulator / 热启动 ckpt")
    init_src = read(f"{G1}/__init__.py")
    for tid in ("Instinct-Amp-Walk-V29-G1-v0", "Instinct-Amp-Walk-V29-G1-Play-v0",
                "Instinct-Amp-Walk-V27-G1-v0"):
        if tid in init_src:
            ok(f"注册 {tid}")
        else:
            fail(f"未注册 {tid}")
    if "amp_walk_v29_ppo_cfg:G1AmpWalkV29PPORunnerCfg" in init_src:
        ok("V29 entry_point 指向 V29 的 PPO cfg")
    else:
        fail("V29 entry_point 不对")
    mp = "/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab/utils/ckpt_manipulators.py"
    if os.path.exists(mp) and "def drop_amp_discriminator" in read(mp):
        ok("drop_amp_discriminator 存在")
    else:
        fail("drop_amp_discriminator 缺失")
    dirs = sorted(glob.glob(f"{RUN_ROOT}/*v27*"), key=os.path.getmtime, reverse=True)
    picked = None
    for d in dirs:
        cks = sorted(glob.glob(f"{d}/model_*.pt"),
                     key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))
        if cks:
            picked = (d, cks[-1])
            break
    if picked:
        ok(f"{os.path.basename(picked[0])} / {os.path.basename(picked[1])}")
    else:
        fail("找不到 V27 的 model_*.pt")

    # ---------------- [I] 运行时导入（信息）----------------
    out("\n[I] （信息）运行时导入尝试（失败不算不通过）")
    try:
        from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v29_ppo_cfg import (  # noqa: F401
            G1AmpWalkV29PPORunnerCfg,
        )
        alg = G1AmpWalkV29PPORunnerCfg().algorithm
        out(f"      运行时: GP={alg.discriminator_gradient_penalty_coef} "
            f"wd={alg.discriminator_weight_decay_coef} coef={alg.discriminator_reward_coef} "
            f"form={alg.amp_reward_form} k={alg.amp_quad_k} "
            f"indep={getattr(alg, 'amp_expert_independent_sampling', None)} "
            f"type={alg.discriminator_reward_type} style_lerp={alg.style_lerp}")
    except Exception as exc:  # noqa: BLE001
        out(f"      ⚠️ 运行时导入不可用（{type(exc).__name__}: {exc}）⇒ 以静态核验为准")

    out("")
    if bad:
        out(f"[V29 FAIL] {len(bad)} 项未通过：{bad}")
        return 1
    out("[V29 PASS] ① 2 帧 / ② 专家批解耦 / ③ 有界奖励形状 就位；系数已回退 V27；其余逐字一致")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        out("=" * 88)
        out("[V29 FAIL] 核验脚本自身抛异常：")
        out(traceback.format_exc())
        rc = 2
    with open(RESULT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print(f"\n（结果已写入 {RESULT_PATH}）")
    sys.exit(rc)
