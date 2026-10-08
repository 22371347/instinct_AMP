"""**T1j**：解除"**相对夹角的常数拉力**"（相对 T1i 的唯一改动）。

要验证的机制（2026-10-08 用户观察 + 代码核实）
--------------------------------------------
用户观察：AMP 风格行走时**大腿弯曲、但"大腿−骨盆"夹角保持常数**，于是**骨盆与上身一同倾斜**；
而且**只在前进指令时出现、静止时正常**。

代码核实（作用域）：
    * `joint_deviation_torso`  → `joint_names="waist_.*"` ⇒ **含 `waist_pitch`/`waist_roll`**
      ⇒ 把"**骨盆−腰**"的俯仰夹角往**常数 0** 拉（weight −0.1）
    * `joint_deviation_knee`   → `.*_knee_joint` ⇒ 把**膝角**往常数（默认屈膝）拉（−0.05）
    * `joint_deviation_hip`    → **只含 `hip_yaw`/`hip_roll`，不含 `hip_pitch`**（所以髋俯仰
      本身没有常角拉力；唯一覆盖它的是低速门控的 `track_default_pose_on_stand`）

历史笔记独立支持该机制：
    * `amp_walk_v14_env_cfg.py:27`：`waist_pitch/roll` 的 URDF 限位只有 **±0.52 rad**
      （除踝外最挤的两个）
    * `amp_walk_v3_env_cfg.py:227-237`：讨论过"歪斜是否来自腰部关节（行程只有 ±0.520）"
    * `amp_walk_v9_env_cfg.py:3`：v8 期间 "**`waist_pitch` 长期贴限位**"

⇒ 因果链：AMP 要匹配参考里**大腿/小腿的世界朝向**；而"腰−骨盆夹角""膝角"被往常数拉、
   `waist_pitch` 行程又只有 ±0.52 ⇒ 两者调和的**唯一便宜出路是"整体倾斜"**；而整体倾斜
   在奖励表里**几乎免费**（`body_ang_vel_xy` 被置 0 且它只罚角速度、`flat_orientation_l2`
   −1.0 太弱）⇒ 于是**前进时倾斜、静止时正常** ✓

本臂改动（**一个概念、相对 T1i 只有两项**）
----------------------------------------
    1. `joint_deviation_torso` : −0.1  → **0.0**（解除"骨盆−腰"常角拉力）
    2. `joint_deviation_knee`  : −0.05 → **0.0**（解除"膝角"常角拉力）

其余**逐字不动**（含 T1i 的 λ=0.975、`body_ang_vel_xy=0`、`flat_orientation_l2=−1.0`、
`joint_deviation_hip/arms`、`dof_pos_limits` 标准版、环境与观测契约）。
起点 = **热启动自 T1a2 的 `model_7000.pt`**（与 T1i/T1b/T1c/T1d 同起点）。

观测指标
--------
⚠️ 本臂**没有** margin 版的倾角探针（那属于 `joint_pos_limits_margin`）。
本臂用**所有臂都有**的等价代理指标：`Episode_Reward/flat_orientation_l2`
（实现为 $\sum(g_x^2+g_y^2)$，本仓库按**每步均值**记录）⇒ **越小 = 倾角越小**。
基线：T1i ≈ **−0.0353**（≈11°），T1f ≈ −0.078（≈16°）。

⚠️ 本文件不改动、不删除任何既有 cfg（非破坏性）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_t1a2_env_cfg import G1AmpWalkT1a2EnvCfg, G1AmpWalkT1a2EnvCfg_PLAY

#: 相对 T1i 的**唯一**改动：把两项常角拉力置 0
T1J_ZEROED = {
    "joint_deviation_torso": 0.0,
    "joint_deviation_knee": 0.0,
}
#: 守卫：这两项原本的权重（若与预期不符 ⇒ fail-fast，说明基线变了）
T1J_EXPECT_BEFORE = {
    "joint_deviation_torso": -0.1,
    "joint_deviation_knee": -0.05,
}


def _apply_t1j_no_constant_pull(cfg) -> None:
    R = cfg.rewards
    print("[T1j][NO CONSTANT PULL] 解除\"相对夹角的常数拉力\"：")
    for name, new_w in T1J_ZEROED.items():
        term = getattr(R, name, None)
        if term is None:
            raise ValueError(f"[T1j] 奖励项不存在：{name}")
        before = float(term.weight)
        if abs(before - T1J_EXPECT_BEFORE[name]) > 1e-12:
            raise ValueError(
                f"[T1j] {name} 的基线权重与预期不符：{before}（预期 {T1J_EXPECT_BEFORE[name]}）"
            )
        term.weight = float(new_w)
        if abs(float(term.weight) - float(new_w)) > 1e-12:
            raise ValueError(f"[T1j] {name} 未生效：{term.weight}")
        _p = getattr(term, "params", {}) or {}
        _ac = _p.get("asset_cfg")
        print(f"    {name:24s} {before:+.3f} → {float(term.weight):+.3f}"
              f"   作用域={getattr(_ac, 'joint_names', None)!r}")

    # 守卫：其余项一律不得被顺手改动（特别是全局倾角与角速度项）
    guards = {
        "flat_orientation_l2": -1.0,
        "body_ang_vel_xy": 0.0,
        "joint_deviation_hip": -0.1,
        "joint_deviation_arms": -0.1,
        "dof_pos_limits": -1.0,
        "track_default_pose_on_stand": -0.8,
        "track_lin_vel_xy_exp": 1.0,
        "track_ang_vel_z_exp": 1.0,
    }
    for name, want in guards.items():
        term = getattr(R, name, None)
        got = None if term is None else float(term.weight)
        if got is None or abs(got - want) > 1e-12:
            raise ValueError(f"[T1j] 守卫失败：{name} = {got}（应为 {want}）")
    n_on = sum(1 for n in dir(R)
               if getattr(getattr(R, n, None), "weight", None) not in (None, 0))
    print(f"[T1j][NO CONSTANT PULL] 守卫通过；非零奖励项 = {n_on}（T1a2/T1i 为 16，本臂 -2 ⇒ 14）")
    if n_on != 14:
        raise ValueError(f"[T1j] 非零奖励项数异常：{n_on}（应为 14）")


@configclass
class G1AmpWalkT1jEnvCfg(G1AmpWalkT1a2EnvCfg):
    """T1j = T1i 的配置 + 解除 `joint_deviation_torso/knee` 的常角拉力。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_t1j_no_constant_pull(self)
        self.run_name = "G1AmpWalkT1jMVPPlane_obs378_rew14_noconstantpull"


@configclass
class G1AmpWalkT1jEnvCfg_PLAY(G1AmpWalkT1a2EnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_t1j_no_constant_pull(self)
        self.run_name = "G1AmpWalkT1jMVPPlane_obs378_rew14_noconstantpull"
