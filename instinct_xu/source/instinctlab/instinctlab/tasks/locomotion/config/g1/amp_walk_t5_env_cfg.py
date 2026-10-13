"""**T5（2026-10-10，用户决策）**：把 mjlab 的 root 高度项从"**躯干高度**"改成"**骨盆高度**"。

用户决策（verbatim）
--------------------
> "明白，我也认为 `track_base_height` 改为 `track_pelvis_height` 比较好。具体的目标高度可以
>  参考 AMP_mjlab。过高会导致腿部伸直进入奇异点动作受限。"

本臂**只做一处语义改动**（不是新增项、不改维度、不动资产）
----------------------------------------------------------
    track_base_height        （量 root = `torso_link` 高度，靶 = 0.82 生成高度）
  → track_pelvis_height      （量 `pelvis` 高度，靶 = **0.7585** = FK 一致的站姿骨盆高度）

    weight 1.5 / std 0.3 **保持 mjlab 原值**（T2 就是照抄 mjlab 的 1.5/0.3）
    ⇒ 非零奖励项仍是 **12**，其余 11 项 func/params 逐字不动。

### 为什么这是"换一个体"而不是"改一个数"
原项量的是 **root（= torso_link）高度**，而**躯干高度可以用腰换**：
MJCF FK 实测"腰前倾 0.2 rad ⇒ 躯干矮 1.5 cm"；回放取证显示策略把躯干钉在竖直
（世界系 **+0.81°** vs 参考 **+10.10°**），同时骨盆后仰 −8.9° 去补偿。
**骨盆高度只由腿决定**（骨盆是两条腿的根）⇒ 换体之后这一项第一次变成
"**约束腿的伸展量**"，也正是 mjlab `track_root_height` 的原意（mjlab 的 root 就是 pelvis）。

### 目标高度怎么定（三个候选取舍）
| 候选 | 值 (m) | 取舍 |
|---|---|---|
| mjlab 原值（他们 root = pelvis） | **0.78** | `KNEES_BENT_KEYFRAME.pos_z`，关节位姿与我们**逐字相同** |
| 我们资产的"生成高度等价" | 0.776 | = 0.82 − 0.044（torso−pelvis 偏移）|
| ⭐ **本臂采用：FK 一致的站姿高度** | **0.7585** | default 站姿 + 脚掌着地时的骨盆离地高度（MJCF FK 实测）|

⚠️ 不取 0.78 的理由（正是用户提醒的那件事）：0.78 比"脚掌着地时的站姿"高 **2.15 cm**，
和我们踩过的躯干靶 0.82（比站姿高 1.75 cm）**同款偏高** ⇒ 会持续逼腿**伸展**，
即"腿伸直 → 进奇异点 → 动作受限"。取 0.7585 后该项是"**别蹲、也别过伸**"的对称约束，
**伸展方向零压力**。
⇒ 若日后要改口径，只需改 `T5_PELVIS_TARGET` 一处（守卫会拦住"高于 FK 站姿"的值）。

### 与蒸馏的关系（调研结论，见 `/memories/repo/amp-walk-t2.md`）
**零影响**：只动了奖励函数，**资产/关节序/动作语义/镜像映射/obs 维度全部不变**
⇒ 仍与越障（同样是 `G1_29DOF_TORSOBASE_POPSICLE_CFG`）同源，仍可作异构 DAgger 教师。
⇒ 也正因为 obs/动作契约没变，本臂**可以热启动续训**（不需要冷启动）。

本臂**不改**
------------
观测 1500（7 项 ×4 帧）/ 命令范围与保持 / 终止 3 项 / 事件与真 RSI / 资产与动作语义
（`scale=1.0`、`use_default_offset=True`、`clip=None`）/ 其余 11 项奖励。
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import instinctlab.tasks.locomotion.mdp as locomotion_mdp

from .amp_walk_t2_env_cfg import (
    T2_EXPECT_NONZERO,
    G1AmpWalkT2EnvCfg,
    G1AmpWalkT2EnvCfg_PLAY,
    G1AmpWalkT2RewardsCfg,
    G1AmpWalkT2TeacherEnv,
)

# --- 目标高度（三个候选，见模块 docstring 的表）--------------------------------
#: MJCF FK 实测：`default 站姿 + 脚掌着地` 时的 **pelvis 离地高度**
#: （`analyze_default_pose_height.py` 的 ②：0.7760 − 0.0175）
#: 这是"不许再高"的**上限**：再高就等于要求腿比自然站姿更直 ⇒ 奇异点风险（用户明确提醒）
T5_FK_STANDING = 0.7585
#: ⭐ 本臂采用的目标 = FK 一致的站姿高度（显式写下同一个数，便于守卫校验"没有偷偷调高"）
T5_PELVIS_TARGET = 0.7585
#: mjlab 原值（`KNEES_BENT_KEYFRAME.pos=(0,0,0.78)`，他们 root = pelvis）
T5_MJLAB_LITERAL = 0.78
#: 我们资产的"生成高度等价" = 0.82 − torso/pelvis 偏移
T5_SPAWN_EQUIV = 0.776
#: torso_link 与 pelvis 的高度差（MJCF FK 实测，用于一切换算）
T5_TORSO_MINUS_PELVIS = 0.044
#: 权重/std 保持 mjlab 原值（T2 同款）
T5_WEIGHT = 1.5
T5_STD = 0.3
#: 被移除的老项（字段保留 ⇒ 一行改回）
T5_REPLACED_TERM = "track_base_height"
#: RSI/观测等一切契约不变 ⇒ 非零项与 T2 相同
T5_EXPECT_NONZERO = T2_EXPECT_NONZERO


@configclass
class G1AmpWalkT5RewardsCfg(G1AmpWalkT2RewardsCfg):
    """T2 的 12 项奖励，**只把高度项换成"量骨盆"**（其余逐字不动）。

    ⚠️ 为什么用"子类 + 新字段"而不是 `setattr` 动态加字段：IsaacLab 的
    `RewardManager` 是按 **dataclass 字段**枚举奖励项的 ⇒ 运行时 `setattr` 出来的新字段
    会被**静默忽略**（维度/名字都不会报错）。建成真字段才安全。
    """

    #: ⭐ 新项：量骨盆高度。`target_height` 由 `__post_init__` 回读校验后写入。
    track_pelvis_height: RewTerm = RewTerm(
        func=locomotion_mdp.track_pelvis_height_exp,
        weight=T5_WEIGHT,
        params={
            "std": T5_STD,
            "target_height": T5_PELVIS_TARGET,
            "one_sided": False,          # = mjlab 的对称口径
            "body_name": "pelvis",
        },
    )

    def __post_init__(self):
        # 1) 先让 T2 把 12 项全部建好（删 legacy / 换 func / 设权重）
        super().__post_init__()
        # 2) 再把"量躯干"的老项关掉（字段保留 ⇒ 一行可改回）
        if not hasattr(self, T5_REPLACED_TERM):
            raise ValueError(f"[T5] 找不到待替换的项 {T5_REPLACED_TERM}")
        if float(getattr(self, T5_REPLACED_TERM).weight) != T5_WEIGHT:
            raise ValueError(
                f"[T5] {T5_REPLACED_TERM} 的基线权重不是 mjlab 的 {T5_WEIGHT}："
                f"{getattr(self, T5_REPLACED_TERM).weight}"
            )
        setattr(self, T5_REPLACED_TERM, None)
        # 3) 守卫：新字段必须是"量骨盆"且目标不得高于 FK 站姿高度
        t = self.track_pelvis_height
        if t is None or t.func is not locomotion_mdp.track_pelvis_height_exp:
            raise ValueError(f"[T5] track_pelvis_height 未生效：{t}")
        if abs(float(t.params.get("target_height", -1)) - T5_PELVIS_TARGET) > 1e-12:
            raise ValueError(f"[T5] 目标未生效：{t.params.get('target_height')}")
        if str(t.params.get("body_name", "")) != "pelvis":
            raise ValueError(f"[T5] 测量体必须是 pelvis：{t.params.get('body_name')}")
        # ⚠️ 用户明确要求避免"目标过高 ⇒ 逼腿伸直进奇异点" ⇒ 硬守卫：不得高于 FK 站姿高度
        if T5_PELVIS_TARGET > T5_FK_STANDING + 1e-9:
            raise ValueError(
                f"[T5] 目标 {T5_PELVIS_TARGET} 高于 FK 一致的站姿骨盆高度 {T5_FK_STANDING} "
                "⇒ 会逼腿伸展进奇异点（用户明确要求避免）"
            )


def _readback_t5(env_cfg) -> None:
    """回读断言 + 打印（fail-fast：任何一项没生效就 raise）。"""
    R = env_cfg.rewards
    if getattr(R, T5_REPLACED_TERM, "MISSING") is not None:
        raise ValueError(f"[T5] {T5_REPLACED_TERM} 未被移除：{getattr(R, T5_REPLACED_TERM)}")
    t = getattr(R, "track_pelvis_height", None)
    if t is None:
        raise ValueError("[T5] track_pelvis_height 缺失（必须是被 RewardManager 枚举到的真字段）")
    if abs(float(t.weight) - T5_WEIGHT) > 1e-12:
        raise ValueError(f"[T5] weight 未生效：{t.weight}（应 {T5_WEIGHT}）")
    p = dict(t.params or {})
    for k, want in (("std", T5_STD), ("target_height", T5_PELVIS_TARGET)):
        if abs(float(p.get(k, -1.0)) - want) > 1e-12:
            raise ValueError(f"[T5] {k} 未生效：{p.get(k)}（应 {want}）")
    if bool(p.get("one_sided", True)):
        raise ValueError("[T5] one_sided 应为 False（mjlab 对称口径）")
    n_on = sum(1 for n in dir(R)
               if getattr(getattr(R, n, None), "weight", None) not in (None, 0))
    if n_on != T5_EXPECT_NONZERO:
        raise ValueError(f"[T5] 非零奖励项 = {n_on}（应 {T5_EXPECT_NONZERO}，与 T2 同数）")
    kept = sorted(n for n in dir(R)
                  if getattr(getattr(R, n, None), "weight", None) not in (None, 0))

    print("[T5][换体] 高度项：root(torso_link) → **pelvis**（唯一环境侧改动）")
    print(f"    {T5_REPLACED_TERM}(量躯干, 靶 0.82 生成高度) → **None**；"
          f"track_pelvis_height(量 pelvis) weight={float(t.weight)} std={p['std']} "
          f"target=**{T5_PELVIS_TARGET}** one_sided={p['one_sided']}")
    print(f"    非零奖励项仍为 {n_on}；保留项：{' '.join(kept)}")
    print("    目标高度取舍：")
    print(f"      * mjlab 原值（他们 root=pelvis）= {T5_MJLAB_LITERAL}"
          f"（比 FK 站姿高 {T5_MJLAB_LITERAL - T5_PELVIS_TARGET:+.4f} m ⇒ 会逼腿伸展）")
    print(f"      * 我们资产的生成高度等价 = {T5_SPAWN_EQUIV}（= 0.82 − {T5_TORSO_MINUS_PELVIS}）")
    print(f"      * ⭐ 采用 = **{T5_PELVIS_TARGET}**（FK：default 站姿 + 脚掌着地）"
          "⇒ '别蹲也别过伸'的对称约束，伸展方向零压力")
    print("    ⚠️ 必看探针：`Step/pelvis_height`（实测骨盆高度）应与靶同量级；"
          "若它长期远离靶，说明靶或测量体又不自洽")


@configclass
class G1AmpWalkT5EnvCfg(G1AmpWalkT2EnvCfg):
    """T5 = T2 + 高度项换体（躯干 → 骨盆，靶 = FK 一致的站姿骨盆高度）。"""

    def __post_init__(self):
        super().__post_init__()                    # T2 的整套照抄 + 其自检（此刻仍是 12 项）
        self.rewards = G1AmpWalkT5RewardsCfg()     # ⚠️ 必须放在 super() 之后：
        #   `_apply_t2` 内部会 `env_cfg.rewards = G1AmpWalkT2RewardsCfg()`，
        #   若先换会被覆盖 ⇒ 只有在这之后换成 T5 的类才生效。
        _readback_t5(self)
        self.run_name = "G1AmpWalkT5MVPPlane_obs1500_rew12_pelvisheight"


@configclass
class G1AmpWalkT5EnvCfg_PLAY(G1AmpWalkT2EnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = G1AmpWalkT5RewardsCfg()
        _readback_t5(self)
        self.run_name = "G1AmpWalkT5MVPPlane_obs1500_rew12_pelvisheight"


class G1AmpWalkT5TeacherEnv(G1AmpWalkT2TeacherEnv):
    """契约与 T2 **完全相同**（obs 1500 / 命令 / 保持 / 动作语义 / 资产）⇒ 直接继承。

    ⇒ 与越障（同资产同动作语义）**仍同源**，蒸馏不受影响；本臂可热启动续训。
    """


__all__ = [
    "T5_EXPECT_NONZERO",
    "T5_MJLAB_LITERAL",
    "T5_PELVIS_TARGET",
    "T5_SPAWN_EQUIV",
    "T5_TORSO_MINUS_PELVIS",
    "G1AmpWalkT5EnvCfg",
    "G1AmpWalkT5EnvCfg_PLAY",
    "G1AmpWalkT5RewardsCfg",
    "G1AmpWalkT5TeacherEnv",
]
