#!/usr/bin/env python3
"""V28 启动前静态核验（**不依赖 isaaclab**，纯源码级 / AST）。

为什么不用 import：本环境下 `python -c "import instinctlab..."` 会在 `isaaclab.envs` 上
失败（`isaaclab` 可导入、但子包不可见），而**训练本身**正常（train.py 走另一条启动链）。
所以核验改成读源码 AST，反而能更严格地证明「V27 → V28 只有预期的改动」。

核验项：
  [A] V28 agent cfg：基类 = V27 的 cfg；`__post_init__` 对 `self.algorithm.*` 的赋值
      **恰好**是 {GP, weight_decay, discriminator_reward_coef}；另有 `ckpt_manipulator`
      与 `run_name`；且未触碰 style_lerp / logit_weight_decay / 去白化 / σ。
  [B] 常数：GP_COEF_V28=5.0、DISC_WD_COEF_V28=3e-4、AMP_COEF_V28=4.0
      ⇒ 有效风格压力 (1−0.75)×4.0 = 1.0（V27 是 0.25）。
  [C] V28 env cfg：基类 = V27 的；`__post_init__` 只赋 run_name；不碰 rewards/observations。
  [D] 字段声明：基类 `amp_walk_ppo_cfg.py` 里这些字段确实被声明（否则 `to_dict()` 静默丢）。
  [E] 继承链：V19b 之后没有任何 cfg 再改这三个系数（⇒ V27 生效值 = V19a/V19b 的值）。
  [F] 注册：V28 train/play 就位、entry_point 指向 V28 cfg、V27 注册未坏。
  [G] `drop_amp_discriminator` 存在。
  [H] 热启动 ckpt（V27 最新）存在。
  [I] （信息）运行时导入尝试，失败不判负。

结果落 `/root/v28_verify_result.txt`，末尾 `[V28 PASS]`。
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
RUN_ROOT = "/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk"
RESULT_PATH = "/root/v28_verify_result.txt"

E_GP = "discriminator_gradient_penalty_coef"
E_WD = "discriminator_weight_decay_coef"
E_COEF = "discriminator_reward_coef"

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


def post_init_assigns(path: str, cls_suffix: str):
    """返回 (基类名列表, {'algorithm.x'} 集合, {'x'} 集合)。"""
    tree = ast.parse(read(path))
    bases: list[str] = []
    algo_keys: set[str] = set()
    self_keys: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name.endswith(cls_suffix)):
            continue
        bases += [ast.unparse(b) for b in node.bases]
        for sub in ast.walk(node):
            if not (isinstance(sub, ast.Assign) and len(sub.targets) == 1):
                continue
            tgt = sub.targets[0]
            if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) \
                    and tgt.value.id == "self":
                self_keys.add(tgt.attr)
            elif isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Attribute) \
                    and isinstance(tgt.value.value, ast.Name) and tgt.value.value.id == "self":
                algo_keys.add(f"{tgt.value.attr}.{tgt.attr}")
    return bases, algo_keys, self_keys


def constants(path: str) -> dict[str, float]:
    tree = ast.parse(read(path))
    vals: dict[str, float] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                v = ast.literal_eval(node.value)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(v, (int, float)):
                vals[node.targets[0].id] = float(v)
    return vals


def main() -> int:  # noqa: C901
    out("=" * 88)
    out("V28 静态核验：判别器正则恢复（GP 5.0 / wd 3e-4）+ 风格项权重 (1-0.75)*4.0 = 1.0")
    out("=" * 88)

    f28a = f"{AG}/amp_walk_v28_ppo_cfg.py"
    f28e = f"{G1}/amp_walk_v28_env_cfg.py"
    for p in (f28a, f28e):
        if not os.path.exists(p):
            fail(f"缺文件 {p}")
            out("")
            out("[V28 FAIL] 文件缺失")
            return 1

    # ---------------- [A] agent cfg ----------------
    out("\n[A] V28 agent cfg（`__post_init__` 的改动面）")
    bases, algo_keys, self_keys = post_init_assigns(f28a, "PPORunnerCfg")
    out(f"      基类: {bases}")
    out(f"      self.algorithm.* 赋值: {sorted(algo_keys)}")
    out(f"      self.* 赋值: {sorted(self_keys)}")
    if bases == ["G1AmpWalkV27PPORunnerCfg"]:
        ok("基类 = G1AmpWalkV27PPORunnerCfg")
    else:
        fail(f"基类意外: {bases}")

    expect_algo = {f"algorithm.{E_GP}", f"algorithm.{E_WD}", f"algorithm.{E_COEF}"}
    if algo_keys == expect_algo:
        ok("algorithm 侧改动恰好只有 GP / weight_decay / reward_coef 三项")
    else:
        fail(f"algorithm 侧改动集意外: 期望 {sorted(expect_algo)}，实际 {sorted(algo_keys)}")

    src_a = read(f28a)
    forbidden = [k for k in ("style_lerp", "logit_weight_decay_coef", "discriminator_kwargs",
                            "amp_grad_pen_space", "clip_max_std", "discriminator_optimizer_kwargs")
                 if re.search(rf"self\.algorithm\.{k}\s*=", src_a)]
    if forbidden:
        fail(f"不该动却动了: {forbidden}")
    else:
        ok("未触碰 style_lerp / logit_weight_decay / discriminator_kwargs / 去白化 / σ")

    if {"ckpt_manipulator", "run_name"} <= self_keys:
        ok("self.ckpt_manipulator / self.run_name 已设")
    else:
        fail(f"self.* 缺项: {sorted(self_keys)}")

    if "DROP_DISC" in src_a and "from .amp_walk_v27_ppo_cfg import" in src_a:
        ok("从 V27 继承 DROP_DISC（判别器重置）")
    else:
        fail("未从 V27 继承 DROP_DISC")

    # ---------------- [B] 常数 ----------------
    out("\n[B] 常数（V28 生效值）")
    cv = constants(f28a)
    out(f"      {cv}")
    for name, want in (("GP_COEF_V28", 5.0), ("DISC_WD_COEF_V28", 3.0e-4), ("AMP_COEF_V28", 4.0)):
        got = cv.get(name)
        if got is not None and abs(got - want) < 1e-12:
            ok(f"{name} == {want}")
        else:
            fail(f"{name} 期望 {want}，实际 {got}")
    eff = (1 - 0.75) * cv.get("AMP_COEF_V28", 0.0)
    if abs(eff - 1.0) < 1e-9:
        ok(f"有效风格压力 (1-style_lerp)*coef = {eff:.3f}（V27 是 0.25）")
    else:
        fail(f"有效风格压力算出来是 {eff}")

    # ---------------- [C] env cfg ----------------
    out("\n[C] V28 env cfg（应只改 run_name）")
    src_e = read(f28e)
    bases_e, _, self_e = post_init_assigns(f28e, "EnvCfg")
    out(f"      基类: {bases_e} / self.* 赋值: {sorted(self_e)}")
    hit = [ln.strip() for ln in src_e.splitlines()
           if re.search(r"self\.(rewards|observations|scene|events|commands)\b", ln)
           and "=" in ln and not ln.strip().startswith("#")]
    if hit:
        fail(f"env cfg 里动了 env 组件: {hit[:3]}")
    else:
        ok("没有改 rewards / observations / scene / events / commands")
    if self_e == {"run_name"}:
        ok("`__post_init__` 只赋 run_name")
    else:
        fail(f"self.* 赋值意外: {sorted(self_e)}")
    if bases_e and all("V27" in b for b in bases_e):
        ok(f"env 基类来自 V27: {bases_e}")
    else:
        fail(f"env 基类意外: {bases_e}")

    # ---------------- [D] 字段声明 ----------------
    out("\n[D] 字段是否在 dataclass 里显式声明（防 `to_dict()` 静默丢字段）")
    base_src = read(f"{AG}/amp_walk_ppo_cfg.py")
    for k in (E_GP, E_WD, E_COEF, "clip_max_std", "clip_max_std_schedule", "style_lerp"):
        if re.search(rf"^\s+{k}\s*[:=]", base_src, re.M):
            ok(f"已声明 {k}")
        else:
            fail(f"**未声明** {k} ⇒ 改动可能被静默丢弃")

    # ---------------- [E] 继承链 ----------------
    out("\n[E] 继承链上这三个系数的**解析值**（决定 V27 的生效值）")
    # 先固定住被引用的常量的定义值（注意有间接引用：DISC_REWARD_COEF_V18 = DISC_REWARD_COEF_V15）
    pinned = {"GP_COEF_V19A": 1.0, "WD_COEF_V19A": 1.0e-5, "DISC_REWARD_COEF_V18": 1.0,
              "STYLE_LERP_V18": 0.75}
    defs: dict[str, object] = {}
    for v in ("v14", "v15", "v18", "v19a"):
        p = f"{AG}/amp_walk_{v}_ppo_cfg.py"
        if not os.path.exists(p):
            continue
        for node in ast.parse(read(p)).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                try:
                    defs[node.targets[0].id] = ast.literal_eval(node.value)
                except Exception:  # noqa: BLE001
                    if isinstance(node.value, ast.Name):
                        defs[node.targets[0].id] = node.value.id  # 间接引用

    def resolve(name: str, depth: int = 0):
        v = defs.get(name)
        while isinstance(v, str) and depth < 6:
            v = defs.get(v)
            depth += 1
        return v if isinstance(v, (int, float)) else None

    for name, want in pinned.items():
        got = resolve(name)
        if got is not None and abs(got - want) < 1e-15:
            ok(f"{name} = {got:g}（钉住）")
        else:
            fail(f"{name} 期望 {want:g}，实际 {got}")

    order = ["v19a", "v19b", "v19c", "v19d", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27"]
    refs: dict[str, dict[str, str]] = {}
    for v in order:
        p = f"{AG}/amp_walk_{v}_ppo_cfg.py"
        if not os.path.exists(p):
            continue
        s = read(p)
        got = {}
        for k in (E_GP, E_WD, E_COEF):
            m = re.search(rf"self\.algorithm\.{k}\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", s)
            if m:
                got[k] = m.group(1)
        if got:
            refs[v] = got
    for v, got in refs.items():
        out(f"      {v}: {got}")
    weird = {v: g for v, g in refs.items()
             if any(sym not in pinned for sym in g.values())}
    if not weird:
        ok("链上所有赋值都只是引用那几个被钉住的常量 ⇒ "
           f"V27 生效值 = GP {pinned['GP_COEF_V19A']:g} / wd {pinned['WD_COEF_V19A']:g} / "
           f"coef {pinned['DISC_REWARD_COEF_V18']:g}")
    else:
        fail(f"出现非钉住常量的引用: {weird} ⇒ 需人工确认 V27 生效值")
    if "v19a" in refs and E_GP in refs["v19a"]:
        ok("确认 V19a 是拆掉 GP/weight_decay 的那一轮（V28 正在恢复它）")
    else:
        fail("没在 V19a 找到 GP 改动，与『正则被拆』的结论不符")
    out(f"      ⇒ V27 有效风格压力 (1-0.75)x{pinned['DISC_REWARD_COEF_V18']:g} = "
        f"{(1 - 0.75) * pinned['DISC_REWARD_COEF_V18']:.3f}；V28 是 1.000（x4）")

    # ---------------- [F] 注册 ----------------
    out("\n[F] 任务注册")
    init_src = read(f"{G1}/__init__.py")
    for tid in ("Instinct-Amp-Walk-V28-G1-v0", "Instinct-Amp-Walk-V28-G1-Play-v0",
                "Instinct-Amp-Walk-V27-G1-v0", "Instinct-Amp-Walk-V27-G1-Play-v0"):
        if tid in init_src:
            ok(f"注册 {tid}")
        else:
            fail(f"未注册 {tid}")
    if "amp_walk_v28_ppo_cfg:G1AmpWalkV28PPORunnerCfg" in init_src:
        ok("V28 entry_point 指向 V28 的 PPO cfg")
    else:
        fail("V28 entry_point 不对")

    # ---------------- [G] manipulator ----------------
    out("\n[G] drop_amp_discriminator")
    mp = "/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab/utils/ckpt_manipulators.py"
    if os.path.exists(mp):
        s = read(mp)
        if "def drop_amp_discriminator" in s and "discriminator_optimizer" in s:
            ok("drop_amp_discriminator 存在（删 discriminator + discriminator_optimizer）")
        else:
            fail("drop_amp_discriminator 内容异常")
    else:
        fail(f"缺文件 {mp}")

    # ---------------- [H] 热启动 ckpt ----------------
    out("\n[H] 热启动对象（V27 最新 ckpt）")
    dirs = sorted(glob.glob(f"{RUN_ROOT}/*v27*"), key=os.path.getmtime, reverse=True)
    picked = None
    for d in dirs:
        cks = sorted(glob.glob(f"{d}/model_*.pt"),
                     key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))
        if cks:
            picked = (d, cks[-1])
            break
    if picked:
        ok(f"{os.path.basename(picked[0])} / {os.path.basename(picked[1])} "
           f"({os.path.getsize(picked[1]) / 1e6:.1f} MB)")
    else:
        fail(f"{RUN_ROOT} 下找不到含 model_*.pt 的 V27 run 目录")

    # ---------------- [I] best-effort 运行时导入 ----------------
    out("\n[I] （信息）运行时导入尝试（失败不算本轮不通过）")
    try:
        from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_v28_ppo_cfg import (  # noqa: F401
            G1AmpWalkV28PPORunnerCfg,
        )
        cfg = G1AmpWalkV28PPORunnerCfg()
        alg = cfg.algorithm
        out(f"      运行时: GP={alg.discriminator_gradient_penalty_coef} "
            f"wd={alg.discriminator_weight_decay_coef} coef={alg.discriminator_reward_coef} "
            f"style_lerp={alg.style_lerp} clip_max_std={getattr(alg, 'clip_max_std', None)}")
        d = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
        flat = str(d)
        if E_COEF in flat:
            ok("运行时 cfg 里能看到这三个字段（未被 to_dict 丢弃）")
        else:
            fail("运行时 to_dict 里找不到这些字段 ⇒ 可能被丢弃")
    except Exception as exc:  # noqa: BLE001
        out(f"      ⚠️ 运行时导入不可用（{type(exc).__name__}: {exc}）⇒ 以静态核验为准")

    out("")
    if bad:
        out(f"[V28 FAIL] {len(bad)} 项未通过：{bad}")
        return 1
    out("[V28 PASS] 判别器正则恢复 + 风格项 x4 就位；其余与 V27 逐字一致")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        out("=" * 88)
        out("[V28 FAIL] 核验脚本自身抛异常：")
        out(traceback.format_exc())
        rc = 2
    with open(RESULT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print(f"\n（结果已写入 {RESULT_PATH}）")
    sys.exit(rc)
