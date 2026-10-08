"""**T1k**：在 T1j 的基础上**恢复 mjlab 的三项机理**（起点与 T1j 相同）。

相对 **T1j** 的差异（3 项奖励，其余逐字不动）
--------------------------------------------
    1. `track_base_height`    : 0      → **+1.5**   （mjlab 值）⭐ 给"站立高度"定价 ⇒ **治蹲姿**
    2. `feet_lateral_width`   : 0      → **−1.0**   （**mjlab(−10) 的 1/10** —— 用户要求"以小值加入，
                                                       便于后续消融对照曲线"，本臂先只做"脚手架"）
    3. `feet_crossing`        : 0      → **−2.0**   （**mjlab(−20) 的 1/10**，同上）

保持不变：`joint_deviation_torso/knee = 0`（T1j 的解绑）、`body_ang_vel_xy = 0`、
`flat_orientation_l2 = −1.0`、`dof_pos_limits` 标准版、`joint_deviation_hip/arms`、
`track_lin_vel_xy_exp = 1.0`、`track_ang_vel_z_exp = 1.0`。
⚠️ **算法侧不动**：λ = 0.975（= mjlab 风格强度），以便本臂与 T1j/T1i 严格可比。
   λ 的增强（0.95，即 2× mjlab）留作**下一臂 T1l 的单一变量**。

起点：**热启动自 T1a2 的 `model_7000.pt`**（与 T1j/T1i/T1b/T1c/T1d **同起点**）。

非零奖励项：T1a2/T1i = 16 ⇒ T1j = 14 ⇒ **T1k = 17**

⚠️ 本文件不改动、不删除任何既有 cfg（非破坏性）。
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
