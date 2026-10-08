#!/usr/bin/env python3
"""**通用 TB 标量读取器**：按 run 子串 + tag 正则，列出「尾段均值 / 极值 / 最新值 / 点数」。

为什么需要它
------------
`cloud_reward_accounting.py` 只把 `Episode_Reward/*` 汇总成会计表，AMP/探针 tag 只挑它
内置的那几个打印。做**姿态类**诊断（`Step/tilt_*`、`Step/grav_b*`、`Step/waist_*`）时
需要"按名字抓任意 tag 的时序统计"，这就是本脚本。

用法（云端）
------------
    /opt/conda/envs/isaaclab/bin/python /root/cloud_tb_tags.py v30 --re 'tilt|grav|waist'
    /opt/conda/envs/isaaclab/bin/python /root/cloud_tb_tags.py v30 --list          # 列全部 tag
    /opt/conda/envs/isaaclab/bin/python /root/cloud_tb_tags.py v27 --re 'tilt|grav' --tail 0.25
    /opt/conda/envs/isaaclab/bin/python /root/cloud_tb_tags.py v30 --re 'air_time' --since 10300

参数
----
    run        run 目录名 / 子串（取匹配中最新的那个）；也接受绝对路径
    --re       tag 正则（不区分大小写）；省略 = 全部
    --tail     尾段比例（默认 0.25）⇒ 统计最后 max(1, n*frac) 个记录点
    --since    只看 iteration/step >= 该值的点（用于"续训之后"的对比）
    --list     只列 tag 名，不统计
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

#: ⭐ 日志根由 agent 的 `experiment_name` 决定，**不止一个**：
#:   AMP-Walk 系列    ⇒ logs/instinct_rl/g1_amp_walk
#:   OmniLoco 基座(T0) ⇒ logs/instinct_rl/g1_unified_omni_loco
#: 所以默认扫 `logs/instinct_rl` 下**所有** run 根；
#: `LLP_LOG_ROOT` 可覆盖（冒号分隔多个），与 plot_training.py / pull_latest_onnx.sh 同口径。
_LOG_BASE = "/root/gpufree-data/instinct_xu/logs/instinct_rl"
_LOG_EXCLUDE = {"plots", "plots_full"}


def _log_roots() -> list[str]:
    env = os.environ.get("LLP_LOG_ROOT", "").strip()
    if env:
        return [p for p in env.split(":") if p]
    roots = sorted(
        d for d in glob.glob(os.path.join(_LOG_BASE, "*"))
        if os.path.isdir(d) and os.path.basename(d) not in _LOG_EXCLUDE
    )
    return roots or [os.path.join(_LOG_BASE, "g1_amp_walk")]


LOGROOTS = _log_roots()
#: 兼容旧引用（第一个根）
LOGROOT = LOGROOTS[0]


def resolve_run(arg: str) -> str | None:
    if os.path.isdir(arg):
        return arg
    cands: list[str] = []
    for root in LOGROOTS:
        cands += [
            d for d in glob.glob(os.path.join(root, "*"))
            if os.path.isdir(d) and arg in os.path.basename(d)
        ]
    return sorted(cands)[-1] if cands else None


def load_scalars(run_dir: str) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})  # 0 = 全部保留
    ea.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [(int(e.step), float(e.value)) for e in ea.Scalars(tag)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run 目录名 / 子串 / 绝对路径")
    ap.add_argument("--re", dest="pattern", default=None, help="tag 正则（不区分大小写）")
    ap.add_argument("--tail", type=float, default=0.25, help="尾段比例（默认 0.25）")
    ap.add_argument("--since", type=int, default=None, help="只看 step >= 该值的点")
    ap.add_argument("--list", action="store_true", help="只列 tag 名")
    args = ap.parse_args()

    run_dir = resolve_run(args.run)
    if run_dir is None:
        print(f"[ERROR] 找不到匹配 {args.run!r} 的 run（{LOGROOT}）")
        return 1
    print(f"run: {os.path.basename(run_dir)}")
    data = load_scalars(run_dir)

    pat = re.compile(args.pattern, re.IGNORECASE) if args.pattern else None
    tags = sorted(t for t in data if (pat is None or pat.search(t)))

    if args.list:
        print(f"共 {len(tags)} 个 tag：")
        for t in tags:
            print(f"  {t}")
        return 0

    rows = []
    for t in tags:
        xs = data[t]
        if args.since is not None:
            xs = [p for p in xs if p[0] >= args.since]
        if not xs:
            continue
        n = max(1, int(len(xs) * max(1e-9, args.tail)))
        seg = [v for _, v in xs[-n:]]
        rows.append((t, sum(seg) / len(seg), min(seg), max(seg), seg[-1], len(xs), xs[-1][0]))

    print(f"匹配 {len(rows)} 个 tag（tail={args.tail}，since={args.since}）")
    print(f"{'tag':<50} {'tail_mean':>12} {'min':>11} {'max':>11} {'last':>11} {'n':>6} {'step':>7}")
    for t, mean, lo, hi, last, n, step in rows:
        print(f"{t:<50} {mean:>+12.5f} {lo:>+11.5f} {hi:>+11.5f} {last:>+11.5f} {n:>6} {step:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
