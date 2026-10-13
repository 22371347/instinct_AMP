"""**T1k**：在 T1j 之上**恢复 mjlab 的三项机理**（高度项 + 足部两项小权重；起点 = 热启动 T1a2 的 `model_7000.pt`；2026-10-07）。

## ⚠️ 本层对**今天**仍有影响的部分

* ⭐⭐ **本层就是这三项的"当前生效层"**（**T1l** 逐字继承，只另改了两项）：
  `track_base_height = +1.5`、`feet_lateral_width = −1.0`、`feet_crossing = −2.0`；**非零奖励项 = 17**。
* ⭐⚠️ **一处"隐藏偏离"（本层不碰 `params`）**：`track_base_height` 的 **weight 1.5 是本层恢复的，
  但核宽 `std` 仍是 V9H 收紧的 0.2**（不是 mjlab/V5 的 0.3）
  ⇒ 这正是 **2026-10-08 被 T1l 的 fail-fast 断言当场抓出**的偏差（**T1l 已把它退回 0.3**）。
  ⚠️ `track_base_height` 的 target = `default_root_state[2]` = **0.82 m = "屈膝 38.3°"下的 torso 高度**
  ⇒ 该项**鼓励保持屈膝**（直腿会变高、反被罚）→ `docs/amp_walk/EXPERIMENTS.md` §P2 + 勘误记录（二）。

## 本层改动一览（相对 **T1j**）

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | `track_base_height` **0 → +1.5**（mjlab 值）⭐ 给"站立高度"定价 ⇒ 治蹲姿 | ✅ 仍 +1.5（**T1l 保留 weight，但把 `std` 0.2 → 0.3**）|
| ② | `feet_lateral_width` **0 → −1.0**（= **mjlab(−10) 的 1/10**；用户要求"以小值加入、便于后续消融对照曲线"）| ✅ 仍 −1.0 |
| ③ | `feet_crossing` **0 → −2.0**（= **mjlab(−20) 的 1/10**，同上）| ✅ 仍 −2.0 |
| ④ | 起点 = 热启动自 T1a2 的 `model_7000.pt`（与 T1j/T1i/T1b/T1c/T1d **同起点**）| 已不适用（今天由 T1l 的起点决定）|

## 推理与判读（已归档，勿在本文件重复）

* ⚠️ **本层的实测现象（重要）**：`Episode_Reward/action_rate_l2 ≈ −0.5`（T1j 及之前大多 ≈ −0.25）
  ⇒ 动作变化率更大（更碎/更抖，与"踝 roll 卡限位、用脚掌侧缘走"一致）；由此**撤销**"放松
  `action_rate_l2` 是治不曲膝第一候选"的说法 → `docs/amp_walk/EXPERIMENTS.md` **勘误记录（三）** + §B5
* "不曲膝"的机理与候选重排（`action_rate_l2` / `joint_deviation_knee`（T1j 解绑）/ 参考集膝角）
  → **勘误记录（二）**
* ⚠️ 守卫与 fail-fast（`T1K_CHANGES` / `T1K_EXPECT_BEFORE` / `T1K_GUARDS` / 非零项数 = 17）是
  "基线不漂移"的在线证据 → 该版 run 日志

⚠️ 其余逐字不动：T1j 的 `joint_deviation_torso/knee = 0`（解绑）、`body_ang_vel_xy = 0`、
   `flat_orientation_l2 = −1.0`、`dof_pos_limits` 标准版、`joint_deviation_hip/arms`、
   `track_lin_vel_xy_exp = 1.0`（⚠️ **T1l → 1.5**）、`track_ang_vel_z_exp = 1.0`。
   ⚠️ **算法侧本层不动**（λ = 0.975 = 1× mjlab 强度，便于与 T1j/T1i 严格可比）
   → 见 `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-T1k-G1-v0` 的「算法侧」节。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-T1k-G1-v0`
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_t1a2_env_cfg import G1AmpWalkT1a2EnvCfg, G1AmpWalkT1a2EnvCfg_PLAY

#: 相对 T1j 的**三处**改动（前两项是 T1j 已置 0 的，这里继续置 0）
T1K_CHANGES = {
    "joint_deviation_torso": 0.0,     # T1j 已解绑，继续
    "joint_deviation_knee": 0.0,      # T1j 已解绑，继续
    "track_base_height": 1.5,         # ← 本臂新增（mjlab）
    "feet_lateral_width": -1.0,       # ← 本臂新增（mjlab 的 1/10）
    "feet_crossing": -2.0,            # ← 本臂新增（mjlab 的 1/10）
}
#: 改动前应有的基线值（fail-fast：若不符说明基线漂移）
T1K_EXPECT_BEFORE = {
    "joint_deviation_torso": -0.1,
    "joint_deviation_knee": -0.05,
    "track_base_height": 0.0,
    "feet_lateral_width": 0.0,
    "feet_crossing": 0.0,
}
#: 守卫：这些不得被顺手改动
T1K_GUARDS = {
    "flat_orientation_l2": -1.0,
    "body_ang_vel_xy": 0.0,
    "joint_deviation_hip": -0.1,
    "joint_deviation_arms": -0.1,
    "dof_pos_limits": -1.0,
    "track_default_pose_on_stand": -0.8,
    "track_lin_vel_xy_exp": 1.0,
    "track_ang_vel_z_exp": 1.0,
    "lin_vel_z_l2": -0.1,
    "action_rate_l2": -0.05,
}
T1K_EXPECT_NONZERO = 17


def _set_w(term, name: str, new_w: float) -> None:
    if term is None:
        raise ValueError(f"[T1k] 奖励项不存在：{name}")
    got = float(term.weight)
    want_before = T1K_EXPECT_BEFORE[name]
    if abs(got - want_before) > 1e-12:
        raise ValueError(f"[T1k] {name} 基线权重与预期不符：{got}（预期 {want_before}）")
    term.weight = float(new_w)
    if abs(float(term.weight) - float(new_w)) > 1e-12:
        raise ValueError(f"[T1k] {name} 未生效：{term.weight}")
    _p = getattr(term, "params", {}) or {}
    _ac = _p.get("asset_cfg")
    extra = ""
    if name == "track_base_height":
        extra = f"   params={ {k: v for k, v in _p.items() if k != 'asset_cfg'} }"
    elif name.startswith("feet_"):
        extra = (f"   sep_floor={_p.get('sep_floor')} gate_lateral={_p.get('gate_lateral')}"
                 f" max_cost={_p.get('max_cost')}")
    print(f"    {name:24s} {got:+.3f} → {float(term.weight):+.3f}{extra}"
          + (f"   作用域={getattr(_ac, 'joint_names', None)!r}" if _ac is not None else ""))


def _apply_t1k(cfg) -> None:
    R = cfg.rewards
    print("[T1k][RESTORE MJLAB TERMS] 相对 T1j 恢复三项机理：")
    for name, w in T1K_CHANGES.items():
        _set_w(getattr(R, name, None), name, w)
    for name, want in T1K_GUARDS.items():
        term = getattr(R, name, None)
        got = None if term is None else float(term.weight)
        if got is None or abs(got - want) > 1e-12:
            raise ValueError(f"[T1k] 守卫失败：{name} = {got}（应为 {want}）")
    n_on = sum(1 for n in dir(R)
               if getattr(getattr(R, n, None), "weight", None) not in (None, 0))
    print(f"[T1k][RESTORE MJLAB TERMS] 守卫通过；非零奖励项 = {n_on}（预期 {T1K_EXPECT_NONZERO}）")
    if n_on != T1K_EXPECT_NONZERO:
        raise ValueError(f"[T1k] 非零奖励项数异常：{n_on}（应为 {T1K_EXPECT_NONZERO}）")


@configclass
class G1AmpWalkT1kEnvCfg(G1AmpWalkT1a2EnvCfg):
    """T1k = T1j + 高度项(+1.5) + 足部两项小权重(−1/−2)。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_t1k(self)
        self.run_name = "G1AmpWalkT1kMVPPlane_obs378_rew17_height_footsmall"


@configclass
class G1AmpWalkT1kEnvCfg_PLAY(G1AmpWalkT1a2EnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_t1k(self)
        self.run_name = "G1AmpWalkT1kMVPPlane_obs378_rew17_height_footsmall"
