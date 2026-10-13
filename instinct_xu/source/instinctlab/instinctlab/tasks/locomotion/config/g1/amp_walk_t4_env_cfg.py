"""**T4 环境侧（消融臂）**：T3 之上**整项移除 `track_base_height`**，其余一字不动。

用户决策（2026-10-10）
----------------------
> "根据控制变量，我们和 AMP_mjlab 的最大区别就在于高度的跟踪是在 `torso_link`
>  还是在 `pelvis`。根据策略移动时躯干直立的表征，问题大概率出现在
>  `track_base_height` 和 `body_ang_vel_xy`。我计划先移除 `track_base_height`
>  然后从 11500 热启动续训。"

本臂 = **真消融**：`track_base_height` → **None**（字段保留 ⇒ 一行改回）。
非零奖励项 **12 → 11**（契约校验里的 `rewards={n}` 只打印、不断言，已核实）。

⚠️ 发车前写下的预期（事后不许改口）
-----------------------------------
`instinct/tools/analyze_gait_trace.py` 在同一条**真实轨迹**（T3@12500, vx=+0.8）上实测：

  * 该轨迹躯干高度 z p50 = **0.826 m**，比 T3 的新靶 0.8025 **高 2.4 cm**
  * T2 靶(0.82)：该项得分 **0.999**、∂r/∂z = −0.216/m；T3 靶(0.8025)：得分 0.993、−0.529/m
  * 把腰前倾 0.2 rad 的代价 = **0.0035 reward/s**，仅占全部 12 项幅度（4.26/s）的 **0.08%**
  * 且 T3 已经把该力的**方向整体翻转**（T2 是 +0.14/m 推高 → T3 是 −0.53/m 压低），
    12600 轮里躯干世界系倾角纹丝不动（**+0.81° vs 参考 +10.10°**，default 站姿是 0.00°）

⇒ **预判：移除本项同样不会改变躯干姿态。** 本臂的价值不是"期待它有效"，而是
  ① 把"height 项"这条腿**彻底钉死**（避免以后再把 12 h 花在冷启动它上）；
  ② 给后续 `pelvis`-root 冷启动臂留一个"**无高度项**"的对照基座。

⚠️ 移除的副作用（解读时必须记住）
----------------------------------
`track_base_height` 是 12 项里**唯一的防塌/防蹲正则**（terminations 只在 0.5 m 兜底）。
⇒ 若本臂出现"蹲走 / 重心过低 / 回合长度下降"，**不能**读成"高度项有害" ——
   那是"撤掉防塌保险"的代价。**判读只认一条：躯干世界系倾角有没有变化。**

本臂**不改**的东西
------------------
观测 1500 / 命令分布与范围 / 终止 3 项 / 事件与真 RSI / 参考集 / 动作语义
（`scale=1.0`、`use_default_offset=True`、`clip=None`）/ 其余 11 项奖励的 func 与 params。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_t3_env_cfg import (
    T3_HEIGHT_TARGET,
    T3_HEIGHT_WEIGHT,
    G1AmpWalkT3EnvCfg,
    G1AmpWalkT3EnvCfg_PLAY,
    G1AmpWalkT3TeacherEnv,
)

#: 本臂移除的项（`setattr(..., None)`，字段保留 ⇒ 一行改回）
T4_REMOVE_TERMS = ("track_base_height",)
#: 移除前 = T2/T3 的项数；移除后应 = 11
T4_EXPECT_NONZERO_BEFORE = 12
T4_EXPECT_NONZERO = 11


def _apply_t4(env_cfg) -> None:
    R = env_cfg.rewards

    # ---- 回读基线：必须是"T3 已生效"的状态（继承链漂移就 fail-fast）----------
    term = getattr(R, "track_base_height", None)
    if term is None:
        raise ValueError("[T4] 找不到 track_base_height（应继承自 T3 的已生效状态）")
    if abs(float(term.weight) - T3_HEIGHT_WEIGHT) > 1e-12:
        raise ValueError(f"[T4] 基线权重不是 T3 的 {T3_HEIGHT_WEIGHT}：{term.weight}")
    if abs(float((term.params or {}).get("target_height", -1.0)) - T3_HEIGHT_TARGET) > 1e-12:
        raise ValueError(
            f"[T4] 基线 target 不是 T3 的 {T3_HEIGHT_TARGET}：{(term.params or {}).get('target_height')}"
        )
    n_before = sum(1 for n in dir(R)
                   if getattr(getattr(R, n, None), "weight", None) not in (None, 0))
    if n_before != T4_EXPECT_NONZERO_BEFORE:
        raise ValueError(f"[T4] 移除前非零项 = {n_before}（预期 {T4_EXPECT_NONZERO_BEFORE}）")
    kept = sorted(n for n in dir(R)
                  if getattr(getattr(R, n, None), "weight", None) not in (None, 0)
                  and n not in T4_REMOVE_TERMS)

    # ---- 唯一改动：整项置 None ---------------------------------------------
    for name in T4_REMOVE_TERMS:
        setattr(R, name, None)

    # ---- 回读断言 ----------------------------------------------------------
    for name in T4_REMOVE_TERMS:
        if getattr(R, name, None) is not None:
            raise ValueError(f"[T4] {name} 未被移除：{getattr(R, name)}")
    n_after = sum(1 for n in dir(R)
                  if getattr(getattr(R, n, None), "weight", None) not in (None, 0))
    if n_after != T4_EXPECT_NONZERO:
        raise ValueError(f"[T4] 移除后非零项 = {n_after}（预期 {T4_EXPECT_NONZERO}）")
    for name in kept:                       # 其余 11 项必须一字不动
        if getattr(getattr(R, name, None), "weight", None) in (None, 0):
            raise ValueError(f"[T4] 非目标项被连带改动：{name}")

    print("[T4][消融] 整项移除 track_base_height：")
    print(
        f"    weight {float(term.weight)} / target {T3_HEIGHT_TARGET} / std "
        f"{(term.params or {}).get('std')} ⇒ **None**（字段保留，一行可改回）"
    )
    print(f"    非零奖励项 {n_before} → **{n_after}**；保留项：{' '.join(kept)}")
    print(
        "    ⚠️ 解读纪律：本臂**不再有防塌/防蹲正则**（终止只在 0.5 m 兜底）⇒ 出现「蹲走/回合变短」"
        "只能说明「撤掉保险」；判据只有一条 —— **躯干世界系倾角有没有变化**"
    )
    print("    ⚠️ 发车前预判（analyze_gait_trace.py 的实测）：该项在此轨迹上的姿态力只有 "
          "0.0035 reward/s（总幅度的 0.08%），且 T3 已把它的方向翻转而姿态纹丝不动 ⇒ "
          "**预计本臂也不变**")


@configclass
class G1AmpWalkT4EnvCfg(G1AmpWalkT3EnvCfg):
    """T4 = T3 − `track_base_height`（消融臂，非零项 12→11）。"""

    def __post_init__(self):
        # ⚠️ 必须先让 T3 的改动 + 其"非零项 == 12"断言跑完（此刻仍成立），再移除
        super().__post_init__()
        _apply_t4(self)
        self.run_name = "G1AmpWalkT4MVPPlane_obs1500_rew11_noh"


@configclass
class G1AmpWalkT4EnvCfg_PLAY(G1AmpWalkT3EnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_t4(self)
        self.run_name = "G1AmpWalkT4MVPPlane_obs1500_rew11_noh"


class G1AmpWalkT4TeacherEnv(G1AmpWalkT3TeacherEnv):
    """契约与 T3（= T2）**完全相同**（obs 1500 / 命令范围 / 保持时长 / 动作语义）⇒ 直接继承。

    ⚠️ 契约校验里的 `rewards={n}` 只**打印**、不断言（见 `amp_walk_env_cfg.py` 的
    `_validate_teacher_contract`），所以 12→11 不会拦车。
    """


__all__ = [
    "T4_EXPECT_NONZERO",
    "T4_REMOVE_TERMS",
    "G1AmpWalkT4EnvCfg",
    "G1AmpWalkT4EnvCfg_PLAY",
    "G1AmpWalkT4TeacherEnv",
]
