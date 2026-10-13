#!/usr/bin/env python3
"""P2 工具：把某个 cfg 的**模块 docstring 整体替换**为给定文件的内容。

为什么需要脚本（而不是编辑器替换）
----------------------------------
中文技术文档里混用了 `U+2212`（−）、全角引号（“”）、全角括号等字符，
基于**文本匹配**的替换会**静默失败**或匹配错位置（P2 第一轮就踩到）。
本脚本用 `ast` 定位**第一个 docstring 的行范围**，按**行号**替换 ⇒
1. 不依赖任何文本匹配，**不会因字符差异失败**；
2. **不可能碰到代码**（只重写 `node.lineno..node.end_lineno` 这几行）；
3. 新旧行数都会打印，便于审计。

用法
----
    python3 instinct/tools/rewrite_docstring.py <target.py> <new_docstring.txt> --check
    python3 instinct/tools/rewrite_docstring.py <target.py> <new_docstring.txt>

约定：`<new_docstring.txt>` 的内容**必须自带三引号**（即整段 `\"\"\"...\"\"\"`）。
建议先 `--check` 确认行范围，再实跑。

⚠️ 只做 docstring 替换。替换后必须：
   1. `get_errors` 通过；
   2. 云端重跑 `audit_reward_table.py`，与 `docs/amp_walk/_baseline_pre_p2.md` diff
      **零参数差异**（这是 P2 的验收标准）。
"""

from __future__ import annotations

import argparse
import ast
import pathlib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="要改的 .py")
    ap.add_argument("new", help="新 docstring 文本文件（自带三引号）")
    ap.add_argument("--check", action="store_true", help="只报告将替换的行范围")
    a = ap.parse_args()

    tgt = pathlib.Path(a.target)
    src = tgt.read_text(encoding="utf-8")
    tree = ast.parse(src)

    if not tree.body:
        raise SystemExit(f"{tgt}: 空文件")
    node = tree.body[0]
    ok = (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )
    if not ok:
        raise SystemExit(f"{tgt}: 第一个语句不是 docstring，拒绝改写（安全闸门）")

    start, end = node.lineno, node.end_lineno
    assert start is not None and end is not None

    new_doc = pathlib.Path(a.new).read_text(encoding="utf-8")
    if not new_doc.endswith("\n"):
        new_doc += "\n"
    body = new_doc.lstrip()
    if not body.startswith('"""'):
        raise SystemExit(f"{a.new}: 内容必须以 \"\"\" 开头（自带三引号）")

    new_lines = new_doc.splitlines()
    print(f"[范围] {tgt.name}: 第 {start}..{end} 行（旧 {end - start + 1} 行）")
    print(f"[新稿] {a.new}: {len(new_lines)} 行；首行 = {new_lines[0][:70]}")

    if a.check:
        print("[check] 未写文件。确认无误后去掉 --check 重跑。")
        return

    lines = src.splitlines(keepends=True)
    out = "".join(lines[: start - 1]) + new_doc + "".join(lines[end:])
    tgt.write_text(out, encoding="utf-8")
    print(f"[ok] 已写 {tgt}；新总行数 = {len(out.splitlines())}")


if __name__ == "__main__":
    main()
