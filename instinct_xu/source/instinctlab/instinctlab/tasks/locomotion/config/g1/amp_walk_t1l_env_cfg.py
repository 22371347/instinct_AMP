"""**T1l**：把两项"历史偏离"改回 V33 / AMP_mjlab 原值（本臂正在训练；2026-10-08 用户决策）。

## ⚠️ 本层对**今天**仍有影响的部分

* ⭐ **本层全部内容就是两个改动**（热启动自 T1a2 的 `model_7000.pt`；非零奖励项 17）：
  ① `track_lin_vel_xy_exp` weight **1.0 → 1.5**；② `track_base_height` **std 0.2 → 0.3**。
* ⭐ **② 是"隐藏覆盖"被当场抓出的一次**：std=0.2 由 **V9H** 引入，经 V33Base→V33S→T1a→T1a2→T1j→T1k
  一路继承、期间从未恢复 ⇒ 本臂首跑 ~9 秒即被 fail-fast 断言拦下
  （`ValueError: [T1l] track_base_height.std = 0.2（应保持 mjlab 的 0.3）`）。

## 本层改动一览

| # | 项 | 当前 T1k | **T1l（今天生效）** | 偏离来源 | mjlab walk 值 |
|---|---|---|---|---|---|
| ① | `track_lin_vel_xy_exp` weight | 1.0 | **1.5** ✅ | T1a 白名单抄 T0 值 | 1.5 |
| ② | `track_base_height` **std** | 0.2 | **0.3** ✅ | **V9H** 把核宽 0.3→0.2 收紧 | 0.3 |

## ⚠️ 关于 ② 的方向（务必按此口径；旧"价格表"是**错的**，已删）

`track_base_height` 的 target = 0.82 m = **"屈膝 38.3°"下的 torso 高度** ⇒ 该项**鼓励保持屈膝**；
**std 放宽对"不曲膝"是反向的**（放宽 = 允许高度偏离，含直腿抬高）。
⚠️ 原文"直腿拿满 1.5 / 屈膝亏 0.091 每步"**方向相反，作废** → `EXPERIMENTS.md` 勘误记录（二）。

## 推理与判读（已归档，勿在本文件重复）

* 用户决策原文与"两项可同臂改（作用域可分离、读数可分别归因）"的论证 → `EXPERIMENTS.md` §E3 / 见该版 run 日志
* "不曲膝"的推力候选重排（`action_rate_l2` / `joint_deviation_knee=0` / 参考集膝角）→ §勘误记录（三）、（二）
* 算法侧（λ=0.975 等）→ `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-T1l-G1-v0` 的「算法侧」节

⚠️ 本层守卫：`track_lin_vel_xy_exp` 的 **std 0.5 不动**（改它=换尺子、破坏 T1b→T1k 可比性）；
   `track_base_height` 的 weight 1.5 / `track_ang_vel_z_exp` 1.0 / 其余 15 项一律不动。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-T1l-G1-v0`
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
