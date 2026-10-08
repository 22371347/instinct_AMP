"""**T1l**：把两项被"历史偏离"的参数改回 V33 / AMP_mjlab 的原值。

用户决策（2026-10-08）
----------------------
> ① "我也观察到了该现象，模型坚持不曲膝导致稳定性下降。std 放宽比较好。可以给到 0.3"
> ② "事实上我认为应该优先提升到1.5。因为我们本质上调整姿态不就是在追求速度跟踪
>     性能的优化吗？应该提升到1.5后再根据效果修改别的项"

## 两个改动都是"回归 V33 / mjlab"，不是新调参

| # | 项 | 当前 T1k | **T1l** | 偏离来源 | mjlab walk 值 |
|---|---|---|---|---|---|
| ① | `track_lin_vel_xy_exp` weight | 1.0 | **1.5** | T1a 白名单抄 T0 值（V33 本是 1.5）| **1.5** ✅ |
| ② | `track_base_height` **std** | **0.2** | **0.3** | **V9H** 把核宽 0.3→0.2 收紧 | **0.3** ✅ |

⚠️ **② 的实测值由 fail-fast 断言当场抓出**（本臂首跑 ~9 秒即崩，日志留痕）：
     `ValueError: [T1l] track_base_height.std = 0.2（应保持 mjlab 的 0.3）`
   来源：`amp_walk_v9h_env_cfg.py:78-79`
     `#: \`track_base_height\` 的核宽：0.3 → 0.2（跳 = 高度大幅振荡 ⇒ 收紧即定价）`
     `AMP_WALK_V9H_BASE_HEIGHT_STD = 0.2`
   ⇒ 该值经 V9H → … → V32B → **V33Base → V33S → T1a → T1a2 → T1j → T1k**
     一路继承，**期间从未恢复** ⇒ **用户"放宽 std"的判断是对的**。

## ② 的量化依据（价格/容差比）

target = 0.82 m（= 直立站姿高度）。屈膝站姿高度降 ~5 cm：

| std | 屈膝站姿奖励 | 与"伸直站姿"之差 |
|---|---|---|
| **0.2（当前 T1k）** | 1.5·exp(−(0.05/0.2)²) = 1.5·0.939 = **1.409** | **0.091/步** ⚠️ |
| **0.3（V33 / mjlab / 本臂）** | 1.5·exp(−(0.05/0.3)²) = 1.5·0.973 = **1.459** | **0.041/步** |
| 0.4（更激进，留给下一臂） | 1.5·exp(−(0.05/0.4)²) = 1.5·0.985 = **1.477** | 0.023/步 |

而 λ=0.975 时 **AMP 风格的绝对强度上限 = (1−λ)·D ≤ 0.025/步**
⇒ **std=0.2 时"屈膝"要付 0.091/步 = 风格的 3.6 倍 ⇒ 几乎不可能赢；
   std=0.3 时降到 0.041/步（仍是风格的 1.6 倍，但差距大幅缩小）。**
   ⇒ 这正是"模型坚持不曲膝"的价格来源。若 0.3 仍不够，下一臂可取 0.4
     （那一步才是**有意偏离 mjlab**）。

## 为什么可以同臂改两项（作用域可分离 ⇒ 仍能分别归因）

① 作用于**速度跟踪项**（全局，每步 ~1.0–1.3 → ×1.5）；
② 只改**高度项的核宽**（不碰它的 weight 1.5，也不碰其他任何项）。
⇒ 读数完全分离：看 `Episode_Reward/track_lin_vel_xy_exp` 与
   `Episode_Reward/track_base_height` 两条曲线即可判断各自贡献。

## 保持不变（守卫）

* `track_lin_vel_xy_exp` 的 **std 0.5 不动** —— 改它等于**换尺子**，
  会让 T1b→T1k 全序列的 `vx_*` 曲线失去横向可比性；且 0.5 更严格（灵敏度
  `w/std²` = 6.0，是 mjlab 1.5 的 4×），对速度跟踪有利。
* `track_ang_vel_z_exp` 仍是 **1.0**（V33 是 1.2）—— 同为 T1a 降下来的，**留作下一臂**。
* 原"三项草案"里的 `action_rate_l2`（当前 −0.05，mjlab −0.01）、
  `body_ang_vel_xy`（当前 0，mjlab +0.7）—— **本臂不动**。
* `track_base_height` 的 **weight 1.5** 不动（本就 = mjlab）。

⚠️ **算法侧不动**：λ = 0.975（= mjlab 风格强度），与 T1k/T1j/T1i 严格可比。

起点：**热启动自 T1a2 的 `model_7000.pt`**（与 T1j/T1i/T1k 同起点）。
非零奖励项：**17**（两项都只改数值，不改开关）

⚠️ 本文件不改动、不删除任何既有 cfg（非破坏性）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_t1k_env_cfg import G1AmpWalkT1kEnvCfg, G1AmpWalkT1kEnvCfg_PLAY

#: 相对 T1k 的**权重**改动（①）
T1L_CHANGES: dict[str, float] = {
    "track_lin_vel_xy_exp": 1.5,   # ← V33 / mjlab walk 值（T1a 白名单曾降到 1.0）
}
#: 改动前应有的基线值（fail-fast：若不符说明基线漂移）
T1L_EXPECT_BEFORE: dict[str, float] = {
    "track_lin_vel_xy_exp": 1.0,
}
#: 高度项核宽（②）：0.2（V9H 收紧值）→ 0.3（V33 / mjlab 值）
T1L_HEIGHT_STD_BEFORE = 0.2
T1L_HEIGHT_STD_AFTER = 0.3
#: 守卫：这些不得被顺手改动
T1L_GUARDS: dict[str, float] = {
    # ---- T1k 刚恢复的三项 ----
    "track_base_height": 1.5,        # 只改核宽，weight 保持 mjlab 值
    "feet_lateral_width": -1.0,
    "feet_crossing": -2.0,
    # ---- T1j 的解绑 ----
    "joint_deviation_torso": 0.0,
    "joint_deviation_knee": 0.0,
    # ---- 后置（本臂不动；mjlab 值分别是 −0.01 / +0.7）----
    "action_rate_l2": -0.05,
    "body_ang_vel_xy": 0.0,
    # ---- 其余 ----
    "flat_orientation_l2": -1.0,
    "joint_deviation_hip": -0.1,
    "joint_deviation_arms": -0.1,
    "dof_pos_limits": -1.0,
    "track_default_pose_on_stand": -0.8,
    "track_ang_vel_z_exp": 1.0,      # ⚠️ 仍是从 V33 的 1.2 降下来的；本臂不动（下一臂）
    "feet_slide": -0.1,
    "lin_vel_z_l2": -0.1,
    "feet_air_time": 1.0,
    "dof_acc_l2": -2.0e-7,
    "dof_torques_l2": -4.0e-6,
    "termination_penalty": -200.0,
}
T1L_EXPECT_NONZERO = 17


def _set_w(term, name: str, new_w: float) -> None:
    if term is None:
        raise ValueError(f"[T1l] 奖励项不存在：{name}")
    got = float(term.weight)
    want_before = T1L_EXPECT_BEFORE[name]
    if abs(got - want_before) > 1e-12:
        raise ValueError(f"[T1l] {name} 基线权重与预期不符：{got}（预期 {want_before}）")
    term.weight = float(new_w)
    if abs(float(term.weight) - float(new_w)) > 1e-12:
        raise ValueError(f"[T1l] {name} 未生效：{term.weight}")
    _p = getattr(term, "params", {}) or {}
    print(f"    {name:24s} {got:+.3f} → {float(term.weight):+.3f}"
          f"   std={_p.get('std')}（尺子不动）")


def _set_height_std(term, name: str, new_std: float) -> None:
    """把 `track_base_height` 的核宽从 V9H 的 0.2 改回 V33 / mjlab 的 0.3。"""
    p = getattr(term, "params", None)
    if not isinstance(p, dict) or "std" not in p:
        raise ValueError(f"[T1l] {name}.params 里没有 std：{p}")
    got = float(p["std"])
    if abs(got - T1L_HEIGHT_STD_BEFORE) > 1e-12:
        raise ValueError(
            f"[T1l] {name}.std 基线不符：{got}（预期 V9H 收紧后的 {T1L_HEIGHT_STD_BEFORE}）"
            " —— 说明该值又被改过，先查明再跑"
        )
    if abs(float(term.weight) - 1.5) > 1e-12:
        raise ValueError(f"[T1l] {name}.weight = {term.weight}（应保持 mjlab 的 1.5）")
    p["std"] = float(new_std)
    if abs(float(term.params["std"]) - float(new_std)) > 1e-12:
        raise ValueError(f"[T1l] {name}.std 未生效：{term.params['std']}")
    print(f"    {name:24s} std {got:.1f} → {float(term.params['std']):.1f}"
          f"   (V9H 收紧值 → V33/mjlab 值；weight {float(term.weight):.1f} 不动)")


def _apply_t1l(cfg) -> None:
    R = cfg.rewards
    print("[T1l][REGRESS TO V33/MJLAB] 把两项历史偏离改回 V33 / AMP_mjlab 原值：")
    for name, w in T1L_CHANGES.items():
        _set_w(getattr(R, name, None), name, w)
    _set_height_std(R.track_base_height, "track_base_height", T1L_HEIGHT_STD_AFTER)
    for name, want in T1L_GUARDS.items():
        term = getattr(R, name, None)
        got = None if term is None else float(term.weight)
        if got is None or abs(got - want) > 1e-12:
            raise ValueError(f"[T1l] 守卫失败：{name} = {got}（应为 {want}）")
    n_on = sum(1 for n in dir(R)
               if getattr(getattr(R, n, None), "weight", None) not in (None, 0))
    print(f"[T1l][REGRESS TO V33/MJLAB] 守卫通过；非零奖励项 = {n_on}"
          f"（预期 {T1L_EXPECT_NONZERO}）")
    if n_on != T1L_EXPECT_NONZERO:
        raise ValueError(f"[T1l] 非零奖励项数异常：{n_on}（应为 {T1L_EXPECT_NONZERO}）")


@configclass
class G1AmpWalkT1lEnvCfg(G1AmpWalkT1kEnvCfg):
    """T1l = T1k + `track_lin_vel_xy_exp` 1.0→1.5 + `track_base_height` std 0.2→0.3。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_t1l(self)
        self.run_name = "G1AmpWalkT1lMVPPlane_obs378_rew17_linvel1p5_hstd0p3"


@configclass
class G1AmpWalkT1lEnvCfg_PLAY(G1AmpWalkT1kEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_t1l(self)
        self.run_name = "G1AmpWalkT1lMVPPlane_obs378_rew17_linvel1p5_hstd0p3"
