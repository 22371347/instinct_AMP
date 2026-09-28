#!/usr/bin/env python3
"""V6 奖励集探针：确认 `Instinct-Amp-Walk-V6-G1-v0` = V5 + `joint_deviation_arms`。

⚠️ 需要 Kit（AppLauncher）但**不需要 GPU**，用 `--device cpu` 就能跑；
   它**只实例化配置**（`load_cfg_from_registry`），**不建仿真环境**，
   所以对正在跑的 GPU 训练干扰很小。

跑法（在云端 `/root/gpufree-data/instinct_xu` 目录内）：
    /opt/conda/envs/isaaclab/bin/python /root/cloud_probe_v6_rewards.py --headless --device cpu

退出码：0 = 断言通过；1 = 有断言失败。
"""
from __future__ import annotations

import argparse

# ⚠️ AppLauncher 必须最先（它会先起 Kit，之后才能 import 需要 pxr/carb 的模块）
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import instinctlab.tasks  # noqa: E402,F401  （触发任务注册）
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

TASK_V5 = "Instinct-Amp-Walk-V5-G1-v0"
TASK_V6 = "Instinct-Amp-Walk-V6-G1-v0"

_fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'✅' if cond else '❌'} {name}" + (f"   {detail}" if detail else ""), flush=True)
    if not cond:
        _fails.append(name)


def reward_terms(task_id: str):
    """返回 ({项名: weight}, env_cfg 实例, 奖励配置实例)。"""
    cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    if isinstance(cfg, type):  # 注册表里可能是类
        cfg = cfg()
    out: dict[str, float | None] = {}
    for name in vars(cfg.rewards):
        if name.startswith("_"):
            continue
        term = getattr(cfg.rewards, name)
        out[name] = None if term is None else float(term.weight)
    return out, cfg


def main() -> int:
    print("\n" + "=" * 72)
    print("V5 vs V6 奖励集探针")
    print("=" * 72)

    check(f"{TASK_V5} 已注册", gym.registry.get(TASK_V5) is not None)
    check(f"{TASK_V6} 已注册", gym.registry.get(TASK_V6) is not None)

    v5, _ = reward_terms(TASK_V5)
    v6, cfg6 = reward_terms(TASK_V6)

    print(f"\nV6 奖励集（{len(v6)} 项）：")
    for name in sorted(v6):
        print(f"  {name:<40} {v6[name]}")

    added = sorted(set(v6) - set(v5))
    removed = sorted(set(v5) - set(v6))
    changed = sorted(k for k in (set(v5) & set(v6)) if v5[k] != v6[k])

    print(f"\nV5 项数 = {len(v5)}   V6 项数 = {len(v6)}")
    print(f"新增    : {added}")
    print(f"删除    : {removed}")
    print(f"权重变化: {[(k, v5[k], v6[k]) for k in changed]}")

    print("\n断言：")
    check("项数 = 15（14 → 15）", len(v6) == 15, f"实际 {len(v6)}")
    check("只新增 joint_deviation_arms", added == ["joint_deviation_arms"], f"实际 {added}")
    check("没有任何项被删/改权重", (not removed) and (not changed), f"removed={removed} changed={changed}")
    check("joint_deviation_arms 权重 = -0.1（基类原值）", v6.get("joint_deviation_arms") == -0.1)

    term = getattr(cfg6.rewards, "joint_deviation_arms", None)
    if term is not None:
        names = list(term.params["asset_cfg"].joint_names)
        print(f"\n  arms joint_names ({len(names)} 个): {names}")
        check("覆盖 7 个手臂关节模式（左右各一套）", len(names) == 7, f"实际 {len(names)}")
        check("含 shoulder_pitch/roll/yaw + elbow",
              all(any(p.startswith(f".*_{j}") for p in names) for j in
                  ("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint")))
    else:
        check("joint_deviation_arms 存在", False)

    print("\n" + "=" * 72)
    print("结果：" + ("PASS" if not _fails else f"FAIL {len(_fails)} 项 -> {_fails}"))
    print("=" * 72, flush=True)
    return 0 if not _fails else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        simulation_app.close()
    raise SystemExit(code)
