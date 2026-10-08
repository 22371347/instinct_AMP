#!/usr/bin/env python3
"""**T1a2 云端配置校验**：把 cfg 建出来，断言"权重 + **实现/作用域** + 探索/网络 + 契约"。

为什么 T1a2 的校验必须新增"实现"一节
------------------------------------
T1a 的教训：**权重全对，但同名奖励的实现不同**（`feet_air_time` 换了带版、`dof_pos_limits`
换成全关节二次版、`dof_acc_l2` 作用域从髋膝变成全部 29 关节）⇒ "T0 等价"不成立，
归因失效，实测收敛到"站着不动"。所以本脚本把**实现**也当一等公民断言。

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \\
    source /root/.llp_env.sh && unset PYTHONPATH && \\
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_t1a2.py

⚠️ 必须先起 `AppLauncher`：`instinctlab.tasks` → `isaaclab.envs` → `utils.mesh`
    → `from pxr import Usd`，而 `pxr` 是 app 启动时才挂上的（2026-10-06/07 各踩一次）。
"""

from __future__ import annotations

import sys

from isaaclab.app import AppLauncher  # noqa: E402

_APP = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False}).app

import dataclasses  # noqa: E402

FAIL: list[str] = []


def chk(name: str, got, want, tol: float | None = 1e-12) -> None:
    if tol is not None and isinstance(got, (int, float)) and isinstance(want, (int, float)):
        ok = abs(float(got) - float(want)) <= float(tol)
    else:
        ok = got == want
    print(f"  {'OK  ' if ok else 'FAIL'} {name:58s} got={got!r}  want={want!r}")
    if not ok:
        FAIL.append(name)


def g(obj, name):
    if not hasattr(obj, name):
        FAIL.append(f"缺少属性 {name}")
        print(f"  FAIL 缺少属性 {name}（配置结构变了？）")
        return None
    return getattr(obj, name)


def main() -> int:
    from instinctlab.tasks.locomotion.config.g1.amp_walk_t1a2_env_cfg import (  # noqa: E402
        AMP_WALK_T1A2_RUN_NAME,
        G1AmpWalkT1a2EnvCfg,
        T1A2_T0_IMPL,
    )
    from instinctlab.tasks.locomotion.config.g1.amp_walk_t1a_env_cfg import (  # noqa: E402
        T1A_REWARD_WEIGHTS,
        T1A_ZEROED,
    )

    cfg = G1AmpWalkT1a2EnvCfg()

    # ---- ① 权重（与 T0 逐项相等）----------------------------------------------
    print("[1] 奖励权重（= T0 run 的 params/env.yaml）")
    for k, v in T1A_REWARD_WEIGHTS.items():
        chk(f"rewards.{k}", float(getattr(cfg.rewards, k).weight), float(v))
    for k in T1A_ZEROED:
        chk(f"rewards.{k} (显式置 0)", float(getattr(cfg.rewards, k).weight), 0.0)

    # ---- ② ⭐ 实现/作用域（T1a 漏掉的一步）------------------------------------
    print("\n[2] ⭐ 奖励实现与作用域（必须与 T0 逐字相同）")
    for name, (func, params) in T1A2_T0_IMPL.items():
        term = getattr(cfg.rewards, name)
        chk(f"rewards.{name}.func", getattr(term.func, "__qualname__", None), getattr(func, "__qualname__", None))
        got_p = getattr(term, "params", {}) or {}
        for key, want in params.items():
            if key in ("asset_cfg", "sensor_cfg"):
                got_scope = got_p.get(key)
                for attr in ("joint_names", "body_names", "name"):
                    w = getattr(want, attr, None)
                    if w is not None:
                        chk(f"  {name}.{key}.{attr}", getattr(got_scope, attr, None), w)
            else:
                chk(f"  {name}.params[{key}]", got_p.get(key), want)
    # 反向检查：不能是 T1a 那两个"带版/边距版"
    chk(
        "feet_air_time 不是 target_biped 版",
        "target_biped" in str(getattr(cfg.rewards, "feet_air_time").func),
        False,
    )
    chk(
        "dof_pos_limits 不是 margin 版",
        "margin" in str(getattr(cfg.rewards, "dof_pos_limits").func),
        False,
    )

    # ---- ③ 奖励集合计数（区分"已删除(None)"与"权重 0"）------------------------
    print("\n[3] 奖励集合")
    try:
        names = [f.name for f in dataclasses.fields(cfg.rewards)]
    except Exception:
        names = [k for k in vars(cfg.rewards) if not k.startswith("_")]

    def alive(k: str) -> bool:
        return getattr(cfg.rewards, k, None) is not None

    removed = sorted(k for k in names if not alive(k))
    live = [k for k in names if alive(k)]
    nz = [k for k in live if abs(float(getattr(cfg.rewards, k).weight)) > 0.0]
    chk("奖励字段总数", len(names), 22)
    chk("存活项数（None = 祖先链已删除）", len(live), 21)
    chk("非零项数（= T0 的 16 项）", len(nz), 16)
    chk("已删除项", removed, ["stand_still"])

    # ---- ④ 接口契约（交付硬约束）---------------------------------------------
    print("\n[4] 接口契约")
    chk("actions.joint_pos.scale（与越障专家对齐口径）", float(g(cfg.actions.joint_pos, "scale")), 1.0)
    chk("actions.joint_pos.use_default_offset", bool(g(cfg.actions.joint_pos, "use_default_offset")), True)
    chk("sim.dt", float(cfg.sim.dt), 0.005)
    chk("decimation", int(cfg.decimation), 4)
    chk("episode_length_s", float(cfg.episode_length_s), 20.0)
    chk("run_name", str(cfg.run_name), AMP_WALK_T1A2_RUN_NAME)
    _obs = cfg.observations.policy
    try:
        _flds = [f.name for f in dataclasses.fields(_obs)]
    except Exception:
        _flds = [k for k in vars(_obs) if not k.startswith("_")]
    # ⚠️ `ObsGroup` 里除观测项外还有 `enable_corruption`/`concatenate_terms` 等配置字段
    #    ⇒ 只数"带 func 的项"。（初版按字段总数断言 9、实测 13 —— 是断言写错了，
    #    不是配置错。378 维的最终确认由运行期 `[AMP-WALK PASS]` 横幅给出。）
    _terms = [k for k in _flds if hasattr(getattr(_obs, k, None), "func")]
    chk("policy 观测项数（8 项 ⇒ 279 + 3*3 + 29*2 = 378 维）", len(_terms), 8)
    chk("policy 含 base_lin_vel（V33S 教师特权）", "base_lin_vel" in _terms, True)
    print(f"      policy 项：{_terms}")

    # ---- ⑤ 算法：探索/网络回 T0 + 核心特性保留 --------------------------------
    print("\n[5] 算法（探索/网络 = T0；核心特性保留）")
    from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_t1a2_ppo_cfg import (  # noqa: E402
        AMP_T1A2_CLIP_MIN_STD,
        AMP_T1A2_ENTROPY_COEF,
        AMP_T1A2_HIDDEN_DIMS,
        AMP_T1A2_INIT_NOISE_STD,
        G1AmpWalkT1a2PPORunnerCfg,
    )

    acfg = G1AmpWalkT1a2PPORunnerCfg()
    alg, pol = acfg.algorithm, acfg.policy
    chk("policy.init_noise_std（= T0 的 1.0）", float(g(pol, "init_noise_std")), float(AMP_T1A2_INIT_NOISE_STD))
    chk("algorithm.clip_min_std（= T0）", float(g(alg, "clip_min_std")), float(AMP_T1A2_CLIP_MIN_STD))
    chk("algorithm.clip_max_std（关闭 = T0 无上限）", g(alg, "clip_max_std"), None)
    chk("algorithm.clip_max_std_schedule（关闭）", g(alg, "clip_max_std_schedule"), None)
    chk("policy.actor_hidden_dims（= T0）", list(g(pol, "actor_hidden_dims")), list(AMP_T1A2_HIDDEN_DIMS))
    chk("policy.critic_hidden_dims（= T0）", list(g(pol, "critic_hidden_dims")), list(AMP_T1A2_HIDDEN_DIMS))
    chk("algorithm.entropy_coef（= T0）", float(g(alg, "entropy_coef")), float(AMP_T1A2_ENTROPY_COEF))

    chk("style_lerp（风格项贡献 == 0）", float(g(alg, "style_lerp")), 1.0)
    chk("use_mirror_loss（交付要求②）", bool(g(alg, "use_mirror_loss")), True)
    chk("mirror_loss_coeff", float(g(alg, "mirror_loss_coeff")), 0.1)
    _mjn = g(alg, "mirror_joint_names")
    chk("mirror_joint_names 个数（29）", len(list(_mjn)) if _mjn else None, 29)
    chk("discriminator_reward_coef（沿用 S8）", float(g(alg, "discriminator_reward_coef")), 4.0)
    chk("schedule（沿用 S8）", str(g(alg, "schedule")), "adaptive")
    chk("amp_expert_buffer_path", str(g(alg, "amp_expert_buffer_path")), "/root/amp_expert_states_396.pt")
    chk("max_iterations（不设上限）", int(g(acfg, "max_iterations")), 1_000_000_000)
    chk("save_interval", int(g(acfg, "save_interval")), 500)
    chk("resume（冷启动）", bool(g(acfg, "resume")), False)
    chk("load_run（冷启动）", g(acfg, "load_run"), None)

    print()
    if FAIL:
        print(f"❌ T1a2 校验失败：{len(FAIL)} 项 —— **不要发车**")
        for f in FAIL:
            print(f"   - {f}")
        return 1
    print("✅ T1a2 校验全部通过 —— 可以发车")
    return 0


if __name__ == "__main__":
    sys.exit(main())
