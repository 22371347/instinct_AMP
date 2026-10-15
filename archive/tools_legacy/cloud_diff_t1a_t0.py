#!/usr/bin/env python3
"""**T1a vs T0 控制变量审计**：把两套 cfg 现场建出来，逐项对比并只打印**差异**。

为什么必须"建出来"而不是读源码
------------------------------
1. 同名奖励项**可能是不同实现**（如 OmniLoco 走 `G1FlatRewardsCfg` 的
   `mdp.track_lin_vel_xy_exp`，AMP-Walk 走 `locomotion_mdp.track_lin_vel_xy_axis_diag`
   "无损诊断版"）—— 只看名字会漏。
2. 继承链 + `__post_init__` 会层层覆盖（`init_noise_std` 就是被 AMP 链改过的例子）。
3. 观测噪声/`enable_corruption`/`clip` 分散在 obs cfg 里，源码里对不上号。

输出：每个主题一节，只列 **T0 与 T1a 不一致**的条目（一致的不打印，保持可读）。

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \\
    source /root/.llp_env.sh && \\
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_diff_t1a_t0.py
"""

from __future__ import annotations

# ⚠️ 必须早于 instinctlab / isaaclab.envs 的导入
from isaaclab.app import AppLauncher  # noqa: E402

_APP = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False}).app

import dataclasses  # noqa: E402
import importlib  # noqa: E402

SPECS = {
    "T0": (
        "instinctlab.tasks.locomotion.config.g1.omni_loco_env_cfg",
        "G1OmniLocoEnvCfg",
        "instinctlab.tasks.locomotion.config.g1.agents.omni_loco_ppo_cfg",
        "G1OmniLocoPPORunnerCfg",
    ),
    "T1a": (
        "instinctlab.tasks.locomotion.config.g1.amp_walk_t1a_env_cfg",
        "G1AmpWalkT1aEnvCfg",
        "instinctlab.tasks.locomotion.config.g1.agents.amp_walk_t1a_ppo_cfg",
        "G1AmpWalkT1aPPORunnerCfg",
    ),
}

SHOWN: list[str] = []


def field_names(obj) -> list[str]:
    """尽力列出字段名：dataclass → dict → __dict__。兼容 actuator 那种 dict。"""
    if obj is None:
        return []
    try:
        if dataclasses.is_dataclass(obj):
            return [f.name for f in dataclasses.fields(obj)]
    except Exception:
        pass
    if isinstance(obj, dict):
        return list(obj.keys())
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        return [k for k in d if not k.startswith("_")]
    return []


def fname(o) -> str:
    """`func` 的可读短名：模块末段 + 限定名（同名不同源一眼可见）。"""
    if o is None:
        return "None"
    m = str(getattr(o, "__module__", "")).split(".")[-1]
    q = getattr(o, "__qualname__", None)
    return f"{m}.{q}" if q else repr(o)


def r(v, n: int = 100) -> str:
    s = repr(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def crawl(obj, keys: list[str]) -> dict[str, str]:
    """取 attr 链（如 'joint_pos.scale'）→ repr。缺失记为 <无>。"""
    out = {}
    for k in keys:
        cur = obj
        for part in k.split("."):
            cur = getattr(cur, part, None) if cur is not None else None
        out[k] = fname(cur) if callable(cur) else r(cur, 80)
    return out


def diff_section(title: str, a: dict[str, str], b: dict[str, str]) -> None:
    keys = sorted(set(a) | set(b))
    rows = [(k, a.get(k, "-"), b.get(k, "-")) for k in keys if a.get(k, "-") != b.get(k, "-")]
    if not rows:
        return
    print(f"\n### {title}  （{len(rows)} 项差异 / 共 {len(keys)} 项）")
    for k, x, y in rows:
        print(f"  {k}")
        print(f"      T0 : {x}")
        print(f"      T1a: {y}")


def reward_snap(cfg) -> dict[str, tuple[str, str, str]]:
    out = {}
    for n in field_names(cfg.rewards):
        v = getattr(cfg.rewards, n, None)
        if v is None:
            out[n] = ("<None 已删除>", "-", "-")
            continue
        w = getattr(v, "weight", None)
        out[n] = (
            f"{w:+g}" if isinstance(w, (int, float)) else str(w),
            fname(getattr(v, "func", None)),
            r(getattr(v, "params", {}), 90),
        )
    return out


def obs_snap(cfg, group: str = "policy") -> dict[str, str]:
    g = getattr(cfg.observations, group, None)
    if g is None:
        return {"<无此组>": "-"}
    out = {"enable_corruption": r(getattr(g, "enable_corruption", None), 20),
           "concatenate_terms": r(getattr(g, "concatenate_terms", None), 20)}
    for n in field_names(g):
        t = getattr(g, n, None)
        if t is None or not hasattr(t, "func"):
            continue
        out[n] = (
            f"func={fname(t.func)} noise={r(getattr(t, 'noise', None), 46)} "
            f"clip={r(getattr(t, 'clip', None), 20)} params={r(getattr(t, 'params', {}), 60)}"
        )
    return out


def main() -> int:
    envs, ags = {}, {}
    for tag, (em, ec, am, ac) in SPECS.items():
        try:
            envs[tag] = getattr(importlib.import_module(em), ec)()
        except Exception as exc:
            print(f"[FATAL] 构建 {tag} env 失败: {type(exc).__name__}: {exc}")
            return 2
        try:
            ags[tag] = getattr(importlib.import_module(am), ac)()
        except Exception as exc:
            print(f"[FATAL] 构建 {tag} agent 失败: {type(exc).__name__}: {exc}")
            return 2

    e0, e1 = envs["T0"], envs["T1a"]
    a0, a1 = ags["T0"], ags["T1a"]

    print("=" * 78)
    print("T1a vs T0 控制变量审计（只列差异）")
    print("=" * 78)
    print(f"T0  run_name = {getattr(e0, 'run_name', '?')}")
    print(f"T1a run_name = {getattr(e1, 'run_name', '?')}")

    # ---- 1. 奖励：名字 → (权重, func, params) -----------------------------------
    r0, r1 = reward_snap(e0), reward_snap(e1)
    names = sorted(set(r0) | set(r1))
    print(f"\n### 奖励对照（T0 {len(r0)} 项 / T1a {len(r1)} 项，共 {len(names)} 个名字）")
    same_w, diff_f, diff_p = [], [], []
    for n in names:
        w0, f0, p0 = r0.get(n, ("<不存在>", "-", "-"))
        w1, f1, p1 = r1.get(n, ("<不存在>", "-", "-"))
        flag = []
        if w0 != w1:
            flag.append("权重")
        if f0 != f1:
            flag.append("**实现**")
        if p0 != p1:
            flag.append("参数")
        if flag:
            print(f"  {'/'.join(flag):<10s} {n}")
            print(f"      T0 : w={w0:>10s}  {f0}")
            print(f"          params={p0}")
            print(f"      T1a: w={w1:>10s}  {f1}")
            print(f"          params={p1}")
            (diff_f if f0 != f1 else (diff_p if p0 != p1 else same_w)).append(n)
    print(f"  ── 汇总：实现不同 {len(diff_f)} 项 {diff_f}")
    print(f"           仅参数不同 {len(diff_p)} 项 {diff_p}")
    print(f"           仅权重不同 {len(same_w)} 项 {same_w}")
    print(f"           完全一致 {len(names) - len(diff_f) - len(diff_p) - len(same_w)} 项")

    # ---- 2. 观测组（policy）---------------------------------------------------
    for grp in ("policy", "critic"):
        diff_section(f"观测组 `{grp}`", obs_snap(e0, grp), obs_snap(e1, grp))

    # ---- 3. 动作 / 指令 / 仿真 / 环境 ------------------------------------------
    diff_section(
        "动作",
        crawl(e0.actions.joint_pos, ["scale", "use_default_offset", "clip", "offset", "preserve_order"]),
        crawl(e1.actions.joint_pos, ["scale", "use_default_offset", "clip", "offset", "preserve_order"]),
    )
    diff_section(
        "指令",
        crawl(e0.commands.base_velocity, ["resampling_time_range", "rel_standing_envs", "rel_heading_envs",
                                          "heading_command", "ranges.lin_vel_x", "ranges.lin_vel_y",
                                          "ranges.ang_vel_z", "debug_vis"]),
        crawl(e1.commands.base_velocity, ["resampling_time_range", "rel_standing_envs", "rel_heading_envs",
                                          "heading_command", "ranges.lin_vel_x", "ranges.lin_vel_y",
                                          "ranges.ang_vel_z", "debug_vis"]),
    )
    diff_section(
        "仿真/回合",
        crawl(e0, ["sim.dt", "decimation", "episode_length_s", "num_envs"]),
        crawl(e1, ["sim.dt", "decimation", "episode_length_s", "num_envs"]),
    )
    diff_section("事件（域随机化）", _ev(e0), _ev(e1))
    diff_section("执行器（actuators）", _act(e0), _act(e1))

    # ---- 4. PPO 超参 ----------------------------------------------------------
    try:
        diff_section("PPO（agent 顶层）", _top(a0), _top(a1))
        diff_section("PPO（algorithm）", _algo(a0), _algo(a1))
        diff_section("PPO（policy）", _pol(a0), _pol(a1))
    except Exception as exc:
        print(f"\n[WARN] PPO 段失败（不影响前面的结论）: {type(exc).__name__}: {exc}")

    print("\n[完成]")
    return 0


def _ev(cfg) -> dict[str, str]:
    ev = getattr(cfg, "events", None)
    if ev is None:
        return {"<无 events>": "-"}
    out = {}
    for n in field_names(ev):
        v = getattr(ev, n, None)
        out[n] = "<None 关闭>" if v is None else f"{type(v).__name__} {r(getattr(v, 'params', {}), 60)}"
    return out


def _act(cfg) -> dict[str, str]:
    try:
        acts = cfg.scene.robot.actuators
    except Exception:
        return {"<无>": "-"}
    out = {}
    for n in field_names(acts):
        v = acts.get(n) if isinstance(acts, dict) else getattr(acts, n, None)
        out[n] = "<None>" if v is None else (
            f"{type(v).__name__} stiffness={r(getattr(v, 'stiffness', None), 46)} "
            f"damping={r(getattr(v, 'damping', None), 46)}"
        )
    return out


def _top(a) -> dict[str, str]:
    keys = ["max_iterations", "save_interval", "resume", "load_run", "experiment_name", "device",
            "num_steps_per_env", "clip_actions", "empirical_normalization", "logger"]
    return crawl(a, keys)


def _algo(a) -> dict[str, str]:
    alg = getattr(a, "algorithm", None)
    if alg is None:
        return {"<无>": "-"}
    out = {}
    for n in field_names(alg):
        v = getattr(alg, n, None)
        if isinstance(v, (dict, list, tuple)) and not isinstance(v, str):
            out[n] = r(v, 60)
        else:
            out[n] = r(v, 60)
    return out


def _pol(a) -> dict[str, str]:
    keys = ["init_noise_std", "noise_std_type", "actor_hidden_dims", "critic_hidden_dims",
            "activation", "actor_obs_normalization", "critic_obs_normalization"]
    return crawl(getattr(a, "policy", a), keys)


if __name__ == "__main__":
    raise SystemExit(main())
