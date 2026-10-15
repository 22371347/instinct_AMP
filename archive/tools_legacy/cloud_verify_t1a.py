#!/usr/bin/env python3
"""T1a 配置校验（云端运行：`/opt/conda/envs/isaaclab/bin/python /root/cloud_verify_t1a.py`）。

T1a = AMP-Walk 接口/特性骨架 + **T0(OmniLoco 基座) 等价奖励集** + 风格项归零
      + 策略层镜像损失 + 冷启动 + 无轮次上限。

设计原则：
  * **只断言"权威值"**（T0 run 的 params/env.yaml、既有默认值、交付要求），
    不写"复述实现"的循环断言；
  * 指令范围这类**与 S8 走同一继承路径**的量，只**打印**以留档（按构造必然与
    V33-S 一致），不对某个具体数字硬断言，避免把"我记错的值"固化进校验；
  * 任何一条断言失败 ⇒ 非零退出（**不要发车**）。

同时输出**生效奖励权重表**（发车前过目 + 留档）。
"""

from __future__ import annotations

import sys

# ============================================================================
# ⚠️ 必须先起 AppLauncher，再 import instinctlab / isaaclab
# ============================================================================
# `import instinctlab.tasks` 会走到 `isaaclab.envs` → `controllers` → `utils.mesh`
# → `from pxr import Usd`；而 `isaaclab` 的**子包路径与 USD(`pxr`) 都是 app 启动时
# 才挂上的**。不起 app 直接 import 会：
#     ModuleNotFoundError: No module named 'pxr'
# （2026-10-06 踩过一次、2026-10-07 又踩一次 —— 见 `cloud_verify_v33_cfg.py` 的
#  文件头注释。本脚本不建 sim，但**仍然必须包在 AppLauncher 里跑**。）
from isaaclab.app import AppLauncher  # noqa: E402

_APP = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False}).app

FAIL: list[str] = []


def chk(name: str, got, want, tol: float | None = 1e-12) -> None:
    if tol is not None and isinstance(got, (int, float)) and isinstance(want, (int, float)):
        ok = abs(float(got) - float(want)) <= float(tol)
    else:
        ok = got == want
    print(f"  {'OK  ' if ok else 'FAIL'} {name:56s} got={got!r}  want={want!r}")
    if not ok:
        FAIL.append(name)


def g(obj, name):
    """安全取属性：缺失记为失败，不抛异常（保证校验脚本总能跑完并给结论）。"""
    if not hasattr(obj, name):
        FAIL.append(f"缺少属性 {name}")
        print(f"  FAIL 缺少属性 {name}（配置结构变了？）")
        return None
    return getattr(obj, name)


# ============================================================================
# ① 奖励白名单（T1a = T0 等价集）
# ============================================================================
from instinctlab.tasks.locomotion.config.g1.amp_walk_t1a_env_cfg import (  # noqa: E402
    G1AmpWalkT1aEnvCfg,
    T1A_REWARD_WEIGHTS,
    T1A_ZEROED,
)

cfg = G1AmpWalkT1aEnvCfg()
print("[1] 奖励权重（T1a = T0 等价集；来源 = T0 run 的 params/env.yaml）")
for k, v in T1A_REWARD_WEIGHTS.items():
    chk(f"rewards.{k}", float(getattr(cfg.rewards, k).weight), float(v))
for k in T1A_ZEROED:
    chk(f"rewards.{k} (显式置 0)", float(getattr(cfg.rewards, k).weight), 0.0)

names = [k for k in vars(cfg.rewards) if not k.startswith("_")]

# ⚠️ 祖先链用 `= None` **删除**项（`ManagerBase` 跳过 None 的 term），它不是 0 权重。
#    2026-10-07：曾把 `stand_still`（V2 已置 None）误当"未归类"，两边都要能识别。
def _alive(k: str) -> bool:
    return getattr(cfg.rewards, k, None) is not None


alive = [k for k in names if _alive(k)]
deleted = sorted(k for k in names if not _alive(k))
n_on = [k for k in alive if abs(float(getattr(cfg.rewards, k).weight)) > 0.0]
chk("奖励字段总数（含已删除字段）", len(names), 22)
chk("存活奖励项数（None = 已删除、不参与奖励）", len(alive), 21)
chk("非零奖励项数（应 == T0 的 16 项）", len(n_on), 16)
chk("已删除项", deleted, ["stand_still"])

print("  生效奖励权重表：")
for k in sorted(names):
    if not _alive(k):
        print(f"      -  {k:34s} <None 已删除>")
        continue
    w = float(getattr(cfg.rewards, k).weight)
    print(f"      {'  ' if abs(w) > 0 else 'x '}{k:34s} {w:+.6g}")

# ============================================================================
# ② 接口契约（动作 / 频率）——交付硬约束
# ============================================================================
print("\n[2] 接口契约（动作 scale 与越障专家对齐口径 = 1.0）")
chk("actions.joint_pos.scale", float(g(cfg.actions.joint_pos, "scale")), 1.0)
chk("actions.joint_pos.use_default_offset", bool(g(cfg.actions.joint_pos, "use_default_offset")), True)
chk("sim.dt", float(cfg.sim.dt), 0.005)
chk("decimation", int(cfg.decimation), 4)
chk("episode_length_s", float(cfg.episode_length_s), 20.0)
chk("run_name", str(cfg.run_name), "G1AmpWalkT1aMVPPlane_obs378_rew16_t0equiv_sym")

print("  指令块（按构造与 V33-S/S8 一致，仅留档）：")
_cmd = cfg.commands.base_velocity
for nm in ("side_vel_min", "low_vel", "prob_low", "prob_side", "prob_lin", "prob_back", "prob_lin_fwd"):
    print(f"      {nm:22s} {g(_cmd, nm)!r}")
for nm in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
    print(f"      ranges.{nm:15s} {g(_cmd.ranges, nm)!r}")
print(f"      straight_lin_vel_range {g(_cmd, 'straight_lin_vel_range')!r}")
print(f"      straight_ang_vel_range {g(_cmd, 'straight_ang_vel_range')!r}")

# ============================================================================
# ③ 算法开关（风格归零 + 镜像损失 + 无上限 + 冷启动）
# ============================================================================
from instinctlab.tasks.locomotion.config.g1.agents.amp_walk_t1a_ppo_cfg import (  # noqa: E402
    AMP_T1A_MIRROR_LOSS_COEFF,
    AMP_T1A_STYLE_LERP,
    G1AmpWalkT1aPPORunnerCfg,
)

acfg = G1AmpWalkT1aPPORunnerCfg()
alg = acfg.algorithm
print("\n[3] 算法开关")
chk("style_lerp == 1.0（风格项恰好归零）", float(g(alg, "style_lerp")), float(AMP_T1A_STYLE_LERP))
chk("use_mirror_loss", bool(g(alg, "use_mirror_loss")), True)
chk("mirror_loss_coeff", float(g(alg, "mirror_loss_coeff")), float(AMP_T1A_MIRROR_LOSS_COEFF))
_mjn = g(alg, "mirror_joint_names")
chk("mirror_joint_names 个数（29 关节，articulation 序）", len(list(_mjn)) if _mjn else None, 29)
chk("max_iterations（不设上限）", int(g(acfg, "max_iterations")), 1_000_000_000)
chk("save_interval", int(g(acfg, "save_interval")), 500)
chk("resume（冷启动）", bool(g(acfg, "resume")), False)
chk("load_run（冷启动）", g(acfg, "load_run"), None)
# 沿用 S8：判别器仍在训练（T1b 只需把 style_lerp 翻回 0.75）
chk("discriminator_reward_coef（沿用 S8）", float(g(alg, "discriminator_reward_coef")), 4.0)
chk("schedule（沿用 S8）", str(g(alg, "schedule")), "adaptive")
chk("amp_expert_buffer_path", str(g(alg, "amp_expert_buffer_path")), "/root/amp_expert_states_396.pt")

# ============================================================================
print()
if FAIL:
    print(f"❌ T1a 校验失败：{len(FAIL)} 项 —— **不要发车**")
    for f in FAIL:
        print(f"   - {f}")
    sys.exit(1)
print("✅ T1a 校验全部通过 —— 可以发车")
