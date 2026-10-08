#!/usr/bin/env python3
"""打印某个 run 实际落盘的 `params/env.yaml` 生效奖励权重表（name → weight）。

为什么需要它
------------
cfg 源码里有**继承 + `__post_init__` 覆盖**，肉眼看源码无法确定"训练时到底生效
了多少项、权重多少"。而 `params/env.yaml` 是 Hydra 在开跑那一刻 dump 的**快照**，
是唯一的权威事实来源。

用法（云端，本地也行）：
    python3 /root/cloud_dump_rewards.py <run_dir 或它的父目录> [过滤子串]

    # 例：只看最新的 T0 基座
    python3 /root/cloud_dump_rewards.py \
        /root/gpufree-data/instinct_xu/logs/instinct_rl/g1_unified_omni_loco

输出：按 |weight| 降序排列，`×` 标记已关闭项（weight == 0）。
"""

from __future__ import annotations

import glob
import os
import sys

try:
    import yaml
except Exception as exc:  # pragma: no cover
    print(f"[FATAL] 需要 PyYAML：{exc}")
    sys.exit(3)


def find_env_yaml(root: str, filt: str = "") -> list[str]:
    """在 root 下递归找 params/env.yaml；filt 非空时按子串过滤。"""
    hits = sorted(glob.glob(os.path.join(root, "**", "params", "env.yaml"), recursive=True))
    if filt:
        hits = [h for h in hits if filt in h]
    return hits


def load_env_yaml(path: str) -> dict:
    """读 OmegaConf dump 出来的 env.yaml。

    ⚠️ 它是 `OmegaConf.save()` 的产物，含 `!!python/tuple`（甚至
    `!!python/object:*`）标签 —— `yaml.safe_load` 会直接报
    `could not determine a constructor for the tag '...python/tuple'`。
    这里先注册 tuple 构造器安全加载；若还有别的标签，退回 `unsafe_load`
    （文件是训练进程自己写的，可信）。
    """

    class _L(yaml.SafeLoader):
        pass

    _L.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: tuple(loader.construct_sequence(node, deep=True)),
    )
    with open(path) as f:
        raw = f.read()
    try:
        return yaml.load(raw, Loader=_L) or {}
    except yaml.constructor.ConstructorError:
        return yaml.unsafe_load(raw) or {}


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    filt = sys.argv[2] if len(sys.argv) > 2 else ""

    hits = find_env_yaml(root, filt)
    if not hits:
        print(f"[FATAL] 在 {root} 下找不到 params/env.yaml（过滤={filt!r}）")
        print("  提示：确认 run 目录名，或用 ls -dt <root>/*/ 看一眼。")
        return 2

    # 多个命中时选最新的（按 mtime）
    p = max(hits, key=lambda h: os.path.getmtime(h))
    if len(hits) > 1:
        print(f"[note] 命中 {len(hits)} 份 env.yaml，取最新的一份")
    print(f"# env.yaml = {p}")
    print(f"# mtime    = {__import__('datetime').datetime.fromtimestamp(os.path.getmtime(p)):%Y-%m-%d %H:%M:%S}")

    with open(p) as f:
        d = load_env_yaml(p)
    rew = d.get("rewards", {}) or {}
    if not rew:
        print("[FATAL] env.yaml 里没有 `rewards` 段")
        return 2

    rows: list[tuple[str, object]] = []
    for k, v in rew.items():
        w = v.get("weight") if isinstance(v, dict) else None
        rows.append((str(k), w))

    def key(item: tuple[str, object]) -> tuple[float, str]:
        w = item[1]
        return (-(abs(w) if isinstance(w, (int, float)) else 0.0), item[0])

    rows.sort(key=key)
    nz = [r for r in rows if isinstance(r[1], (int, float)) and r[1] != 0.0]
    print(f"共 {len(rows)} 项，其中非零 {len(nz)} 项：")
    for k, w in rows:
        mark = "  " if isinstance(w, (int, float)) and w != 0.0 else "× "
        ws = f"{w:+.6g}" if isinstance(w, (int, float)) else str(w)
        print(f"  {mark}{k:34s} {ws}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
