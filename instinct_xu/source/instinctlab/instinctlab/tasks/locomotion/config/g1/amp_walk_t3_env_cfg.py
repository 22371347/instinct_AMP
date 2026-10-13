"""**T3 环境侧**：T2（全权照抄 mjlab + 真 RSI）之上**只改 `track_base_height` 的靶与权重**。

用户决策（2026-10-10）
----------------------
> "现在 AMP 风格已经初步具有了，尽量不做加法而是做减法简化思路。我计划把
>  `track_base_height` 的 target 调整为 0.8025，同时权重削减为 1.0。"

本臂**只动两个数**：

    track_base_height:
        weight        1.5  →  1.0
        params.std    0.3  →  0.3（不动）
        params.target_height  None  →  **0.8025**

为什么 target 是 **0.8025**（`instinct/tools/analyze_default_pose_height.py` 实测）
-------------------------------------------------------------------------------
用我们自己的 MJCF（`instinct/assets/g1_29dof_popsicle_play.xml`，回放工具同款，
free joint 挂在 `torso_link`）做正运动学，把 base 抬到生成高度 0.82 后看**脚底**（`left_foot`/
`right_foot`，实测正好在鞋底）：

| 姿势 | 脚底离地 | ⇒ 落地后**躯干**高度 |
|---|---|---|
| 关节全零（腿伸直） | −1.79 cm（穿地） | 0.838 |
| ⭐ **default 站姿**（= 动作原点 = `track_default_pose_on_stand` 的靶） | **+1.75 cm（悬空）** | **0.8025** |
| default + 腰前倾 0.201（参考中位） | +2.52 cm | 0.7948 |
| 参考中位姿态（腰 0.201 / 膝 0.383） | +0.13 cm | 0.8187 |

⇒ `default_root_state[:, 2]` = **0.82** 只是**手填的生成高度**，它比"站姿靶"的实际离地高度
**高 1.75 cm**。于是：

* 站姿项把腰拉向 0（= default）、`track_base_height` 却要求"比 default 再高 1.75 cm" ⇒ **两项互斥**，
  而"**立腰**（腰 +0.2 → 0 能拿回 0.75 cm）+ **伸膝**（膝 0.669 → 0.383 能拿回 1.6 cm）"是拿高度的两条路径；
* 参考走路要**腰前倾 0.2**（代价正是 −0.75 cm），而**没有任何项要求前倾**
  （`joint_deviation_torso` T2 已删、风格项只有 0.025）⇒ 策略自然选择"立腰 + 端着躯干"。
* 移动段 `track_default_pose_on_stand` 衰减到 ≈0 后，**唯一还在的"姿态力"就是那个偏高 1.75 cm 的高度项**。

改完之后该项**基本变成常数（防塌保险）**：T2 实测它 0.9943（e≈2.2 cm），新靶下 e≈0.5~2 cm、
权重 1.0 ⇒ 该项值 ≈ **0.998~1.0** ⇒ 对姿态不再产生有效梯度，只在"要塌"时才收钱。
这符合用户"做减法"的思路：**不新增项，只把一个有偏置的项中和掉**。

本臂**不改**的东西（`T3_KEEP` 守卫逐项断言）
---------------------------------------------
观测 1500 / 命令分布与范围 / 终止 3 项 / 事件与真 RSI / 参考集 / 其余 11 项奖励 /
动作语义（`scale=1.0`、`use_default_offset=True`、`clip=None`）。
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import instinctlab.tasks.locomotion.mdp as locomotion_mdp

from .amp_walk_t2_env_cfg import (
    T2_EXPECT_NONZERO,
    G1AmpWalkT2EnvCfg,
    G1AmpWalkT2EnvCfg_PLAY,
    G1AmpWalkT2TeacherEnv,
)

#: 本臂唯一改动的两个数
T3_HEIGHT_TARGET = 0.8025   # = default 站姿的实测离地高度（不是生成高度 0.82）
T3_HEIGHT_WEIGHT = 1.0
#: 未改动，但要显式写出来并回读（防止"顺手改了 std"）
T3_HEIGHT_STD = 0.3
#: 改动前的基线（fail-fast：不符说明继承链漂移了）
T3_HEIGHT_WEIGHT_BEFORE = 1.5


def _apply_t3(env_cfg) -> None:
    R = env_cfg.rewards
    term = getattr(R, "track_base_height", None)
    if term is None:
        raise ValueError("[T3] 找不到奖励项 track_base_height")

    # ---- 回读基线（必须是 T2 的值）-----------------------------------------
    if abs(float(term.weight) - T3_HEIGHT_WEIGHT_BEFORE) > 1e-12:
        raise ValueError(f"[T3] track_base_height 基线权重不符：{term.weight}（预期 {T3_HEIGHT_WEIGHT_BEFORE}）")
    p0 = dict(getattr(term, "params", {}) or {})
    if abs(float(p0.get("std", -1.0)) - T3_HEIGHT_STD) > 1e-12:
        raise ValueError(f"[T3] track_base_height.std 基线不符：{p0.get('std')}（预期 {T3_HEIGHT_STD}）")
    if p0.get("target_height") is not None:
        raise ValueError(
            f"[T3] 基线里已经有 target_height={p0.get('target_height')} —— 说明父链被改过，"
            "先查明再跑（本臂的语义是 None → 显式 0.8025）"
        )

    # ---- 只替换这一项（func/std 与父链逐字一致，只动 target 与 weight）------
    R.track_base_height = RewTerm(
        func=locomotion_mdp.track_base_height_exp,
        weight=T3_HEIGHT_WEIGHT,
        params={"std": T3_HEIGHT_STD, "target_height": T3_HEIGHT_TARGET},
    )

    # ---- 回读断言 ----------------------------------------------------------
    t = R.track_base_height
    if abs(float(t.weight) - T3_HEIGHT_WEIGHT) > 1e-12:
        raise ValueError(f"[T3] 权重未生效：{t.weight}")
    if abs(float(t.params["std"]) - T3_HEIGHT_STD) > 1e-12:
        raise ValueError(f"[T3] std 未生效：{t.params['std']}")
    if abs(float(t.params["target_height"]) - T3_HEIGHT_TARGET) > 1e-12:
        raise ValueError(f"[T3] target_height 未生效：{t.params['target_height']}")
    n_on = sum(1 for n in dir(R) if getattr(getattr(R, n, None), "weight", None) not in (None, 0))
    if n_on != T2_EXPECT_NONZERO:
        raise ValueError(f"[T3] 非零奖励项 = {n_on}（应与 T2 相同 = {T2_EXPECT_NONZERO}）")

    print("[T3][减一项偏置] track_base_height：")
    print(
        f"    weight {T3_HEIGHT_WEIGHT_BEFORE:.1f} → {float(t.weight):.1f}；"
        f"std {T3_HEIGHT_STD}（不动）；target None(=0.82 生成高度) → **{T3_HEIGHT_TARGET}**"
        f"（= default 站姿的实测离地高度）"
    )
    print(
        "    ⇒ 该项值应从 0.994（T2 实测）升到 ≈0.998~1.0：**基本变成常数**，"
        "只在'要塌'时才收钱（做减法，不加新项）"
    )
    print(f"    守卫生效；非零奖励项 = {n_on}（与 T2 相同）")


@configclass
class G1AmpWalkT3EnvCfg(G1AmpWalkT2EnvCfg):
    """T3 = T2 + `track_base_height` 的 target 0.82→0.8025、weight 1.5→1.0。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_t3(self)
        self.run_name = "G1AmpWalkT3MVPPlane_obs1500_rew12_h08025_w1p0"


@configclass
class G1AmpWalkT3EnvCfg_PLAY(G1AmpWalkT2EnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_t3(self)
        self.run_name = "G1AmpWalkT3MVPPlane_obs1500_rew12_h08025_w1p0"


class G1AmpWalkT3TeacherEnv(G1AmpWalkT2TeacherEnv):
    """T3 的契约与 T2 **完全相同**（obs 1500 / 命令范围 / 保持时长 / 动作语义）⇒ 直接继承。"""


__all__ = [
    "T3_HEIGHT_STD",
    "T3_HEIGHT_TARGET",
    "T3_HEIGHT_WEIGHT",
    "G1AmpWalkT3EnvCfg",
    "G1AmpWalkT3EnvCfg_PLAY",
    "G1AmpWalkT3TeacherEnv",
]
