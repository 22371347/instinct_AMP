#!/usr/bin/env python3
"""审计：把每个 AMP-Walk 臂**实际生效**的奖励表 dump 成 markdown。

为什么需要它
------------
`config/g1/*env_cfg.py` 的注释**经常滞后或被中间继承层覆盖**，只看源码会读错。
实例（2026-10-08）：`track_base_height` 的 `std` 在 `amp_walk_v9h_env_cfg.py` 被
从 0.3 收紧到 0.2（并整体重建该项），之后经 V32B→V33Base→V33S→T1a→T1a2→T1j→T1k
一路继承；而 V33 源码字面仍写 0.3 ⇒ 我据此得出"std 已经是 0.3"的错误结论。

本脚本的做法是**实例化 cfg 对象**，即真正执行各层 `__post_init__`，
因此打印出来的是**运行期真正生效的值**，不是源码字面值。

用法（云端；本地无 isaaclab）
-----------------------------
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab \\
      && source /root/.llp_env.sh && unset PYTHONPATH \\
      && /opt/conda/envs/isaaclab/bin/python -u /root/audit_reward_table.py \\
         > /root/EFFECTIVE_REWARDS.md

⚠️ 必须走 `AppLauncher`（否则 Omniverse EULA 会读 stdin 挂死）。
⚠️ 远端禁用 `$0`；不要用 `pkill -f`。
⚠️ `AppLauncher` 会初始化 CUDA context（~几百 MB）⇒ **不要在训练高峰跑**，
   或错开 ckpt 落盘时刻；跑完进程退出即释放。

参数
----
    --match SUBSTR   只 dump 任务 id 含该子串的臂（默认全部，如 `--match T1`）
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib
import io

# ---- 铁律：先起 AppLauncher，再 import 任何 isaaclab / instinctlab ----------
from isaaclab.app import AppLauncher

AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False})

import gymnasium as gym  # noqa: E402

import instinctlab.tasks  # noqa: E402,F401  触发任务注册

TASK_SUBSTR = "Instinct-Amp-Walk"
#: 自检输出里值得保留的行（其余丢弃，否则文档会膨胀）
_KEEP_HINTS = ("PASS", "T1", "T0", "WHITELIST", "RESTORE", "REGRESS", "UNLOCK", "REWARD")


def _resolve_entry(entry):
    """注册时可能传类对象，也可能是 `module:Class` 字符串。"""
    if isinstance(entry, str):
        mod, _, cls = entry.partition(":")
        return getattr(importlib.import_module(mod), cls)
    return entry


def _short(v, max_len: int = 52) -> str:
    """截断显示。⚠️ 只加 `…`，不写"截掉多少"——那个数字是**字符数**，
    写成 `…(+113)` 会被误读成"113 项"（本工具第一版就犯过）。"""
    s = str(v)
    return s if len(s) <= max_len else f"{s[:max_len]}…"


def _fmt_dict(d: dict) -> str:
    """字典只显示**与众数不同的项**（30 关节 tol 表里真正有用的就是少数差异项）。"""
    items = list(d.items())
    if len(items) <= 4:
        return _short(d)
    from collections import Counter

    common, _ = Counter(d.values()).most_common(1)[0]
    odd = [(k, v) for k, v in items if v != common]
    if odd and len(odd) <= 8:
        inner = ", ".join(f"{k}:{v}" for k, v in odd)
        return f"{{{inner}}} (其余 {len(items) - len(odd)} 项={common})"
    return f"{{共 {len(items)} 项}}"


def _fmt_params(params) -> str:
    """把 params 压成一行；`SceneEntityCfg` 只留名字与作用域（最易踩坑的部分）。"""
    if not params:
        return "—"
    bits = []
    for k, v in params.items():
        jn = getattr(v, "joint_names", None)
        bn = getattr(v, "body_names", None)
        sn = getattr(v, "site_names", None)
        if jn is not None or bn is not None or sn is not None:
            parts = []
            nm = getattr(v, "name", None)
            if nm:
                parts.append(str(nm))
            if jn:
                parts.append(f"joints={_short(jn)}")
            if bn:
                parts.append(f"bodies={_short(bn)}")
            if sn:
                parts.append(f"sites={_short(sn)}")
            bits.append(f"{k}=({'|'.join(parts)})")
        elif isinstance(v, dict):
            bits.append(f"{k}={_fmt_dict(v)}")
        elif isinstance(v, (list, tuple)):
            if len(v) > 3:
                bits.append(f"{k}=[{', '.join(map(str, v[:2]))}…共{len(v)}项]")
            else:
                bits.append(f"{k}={_short(v)}")
        else:
            bits.append(f"{k}={_short(v)}")
    return ", ".join(bits)


def _trim_selftest(text: str, max_lines: int = 18) -> list[str]:
    """从 cfg 的自检打印里挑关键行（契约 / 断言 / 回读表）。"""
    out = []
    for ln in (text or "").splitlines():
        ln = ln.rstrip()
        if not ln.strip():
            continue
        if any(h in ln for h in _KEEP_HINTS):
            out.append(ln)
    return out[-max_lines:]


#: 算法侧要 dump 的字段名关键字（子串匹配）。
#: 为什么需要：`EFFECTIVE_REWARDS.md` 只覆盖**奖励**，而算法侧（λ / amp_coef /
#: 判别器 kwargs / σ / clip / hidden / mirror loss）**同样会被继承层覆盖** ——
#: 历史事故：把 `style_lerp=0.75` 当成"mjlab 权威值"直接用，实际它在我们的合并式
#: `aux=(1−λ)(style − r_task/c)` 下等于 **10× 强度**（§B3）。
_ALGO_HINTS = (
    "style_lerp", "amp_reward_coef", "discriminator_reward", "amp_quad_k",
    "disable_style_on_stand", "discriminator_kwargs", "grad_pen",
    "independent_reference_sampling", "normalizer", "confidence",
    "init_noise_std", "clip_min_std", "clip_max_std", "hidden_dims",
    "entropy_coef", "learning_rate", "mirror_loss", "num_steps_per_env",
    "max_iterations", "save_interval",
)


def _dump_algo(task_id: str) -> list[str]:
    """dump 算法侧（PPO / 判别器）的**实际生效值**。"""
    spec = gym.spec(task_id)
    assert spec is not None
    entry = spec.kwargs.get("instinct_rl_cfg_entry_point")
    if entry is None:
        return []
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            runner = _resolve_entry(entry)()
        except Exception as exc:  # noqa: BLE001 - 老臂的 agent cfg 可能已不可实例化
            return [f"**算法侧**：⚠️ 实例化失败 `{type(exc).__name__}: {exc}`", ""]
    rows = []
    for sec in ("policy", "algorithm"):
        obj = getattr(runner, sec, None)
        if obj is None:
            continue
        for k, v in sorted(vars(obj).items()):
            if k.startswith("_") or callable(v):
                continue
            if not any(h in k for h in _ALGO_HINTS):
                continue
            rows.append((f"{sec}.{k}", _short(repr(v), 110)))
    if not rows:
        return []
    out = ["**算法侧（实际生效）**", "", "| 字段 | 值 |", "|---|---|"]
    out += [f"| `{k}` | `{v}` |" for k, v in rows]
    out.append("")
    return out


def dump_one(task_id: str) -> str:
    spec = gym.spec(task_id)
    assert spec is not None, f"找不到任务：{task_id}"
    entry = _resolve_entry(spec.kwargs["env_cfg_entry_point"])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):  # 自检打印入缓冲，不污染 markdown
        cfg = entry()

    R = cfg.rewards
    try:
        names = [f.name for f in dataclasses.fields(R)]
    except Exception:  # pragma: no cover - configclass 不保证是 dataclass
        names = [k for k in vars(R) if not k.startswith("_")]

    rows, gone, n_on = [], [], 0
    for n in sorted(names):
        term = getattr(R, n, None)
        if term is None:
            gone.append(n)
            continue
        w = float(term.weight)
        if w != 0.0:
            n_on += 1
        rows.append(
            (
                n,
                w,
                _fmt_params(getattr(term, "params", {}) or {}),
                getattr(getattr(term, "func", None), "__qualname__", "?"),
            )
        )
    rows.sort(key=lambda r: (-abs(r[1]), r[0]))  # 按 |weight| 降序：先看最重的

    L = [
        f"### `{task_id}`",
        "",
        f"- **非零 {n_on}** / 存活 {len(rows)} / 已删除(None) {len(gone)}",
        f"- run_name: `{getattr(cfg, 'run_name', '?')}`",
        "",
        "| term | weight | params | func |",
        "|---|---:|---|---|",
    ]
    for n, w, p, fn in rows:
        L.append(f"| `{n}` | {w:+.4g} | {p} | `{fn}` |")
    if gone:
        L += ["", "已删除(None)：" + "、".join(f"`{g}`" for g in gone)]

    L += _dump_algo(task_id)

    keep = _trim_selftest(buf.getvalue())
    if keep:
        L += ["", "<details><summary>cfg 自检（关键行）</summary>", "", "```"]
        L += keep
        L += ["```", "", "</details>"]
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="", help="只 dump 任务 id 含该子串的臂")
    ap.add_argument("--include-play", action="store_true", help="也 dump `*-Play-*` 任务")
    args = ap.parse_args()

    ids = sorted(
        {
            s.id
            for s in gym.registry.values()
            if isinstance(s.id, str) and TASK_SUBSTR in s.id
        }
    )
    ids = [
        i
        for i in ids
        if args.match in i and (args.include_play or "Play" not in i)
    ]

    print("# AMP-Walk 各臂**实际生效**奖励表（自动生成）\n")
    print("> 由 `instinct/tools/audit_reward_table.py` **实例化 cfg 后 dump** ⇒")
    print("> 表中是运行期真正生效的值（含所有继承层覆盖），**不是源码字面值**。")
    print("> 重新生成方式见脚本 docstring。\n")
    print(f"匹配 `{args.match or '*'} `：共 {len(ids)} 个任务。\n")

    ok = fail = 0
    for tid in ids:
        try:
            print(dump_one(tid))
            ok += 1
        except Exception as exc:  # noqa: BLE001 - 老臂可能已不可实例化，记录即可
            fail += 1
            print(f"### `{tid}`\n")
            print(f"- ❌ 实例化失败：`{type(exc).__name__}: {exc}`\n")
    print(f"---\n\n成功 {ok} / 失败 {fail}")


if __name__ == "__main__":
    main()
