"""**T1a**：AMP-Walk 接口/特性骨架 + **T0（OmniLoco 基座）等价奖励集**（风格项关闭；2026-10-07 决策）。

## ⚠️ 本层对**今天**仍有影响的部分

* ⭐⭐ **白名单机制今天仍在执行**（被 T1a2 → … → T1k/T1l 逐字继承）：把奖励表改写成"T0 等价集"
  = **16 项按 T0 权重赋值 + 5 项显式置 0**，并带 **fail-fast**：白名单 / 待关闭项必须存在，
  **未归类的奖励项直接报错**（将来新增奖励项忘了归类会被拦住，而不是悄悄进训练）。
* ⭐ **今天 `track_lin_vel_xy_exp = 1.0` / `track_ang_vel_z_exp = 1.0` 的来源就是本层白名单**
  （V5 原值本是 **1.5 / 1.2**）。⚠️ **T1l 已把 `track_lin_vel_xy_exp` 改回 1.5**（仅这一项）。
* `stand_still` 整项**已删除（None）**，由 `track_default_pose_on_stand` 等权替代 ⇒ 今天 −0.8。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | 白名单 16 项 = **逐项抄 T0 run 的 `params/env.yaml`**（原则：回归测试不顺手调参，否则无法归因）| 见下"16 项状态" |
| ② | 5 项**显式置 0**（字段保留 ⇒ 消融时一行改回）：`track_base_height` / `body_ang_vel_xy` / `feet_lateral_width` / `feet_crossing` / `feet_swing_time_penalty` | `body_ang_vel_xy` ✅ 仍 0；`feet_swing_time_penalty` ✅ 仍 0；`track_base_height` ❌ **T1k → +1.5**；`feet_lateral_width` ❌ **T1k → −1.0**；`feet_crossing` ❌ **T1k → −2.0** |
| ③ | `stand_still` 整项删除，改用 `track_default_pose_on_stand(−0.8)` **等权替代**（= 交付要求②：静止恢复初始姿态）| ✅ 仍 −0.8（`params` 已被 V23 重建 → 逐关节权重 + `vel_scale 0.8`）|
| ④ | 接口/特性骨架：`track_default_pose_on_stand(−0.8)`（静止恢复初始姿态，交付要求②）+ 策略层 `mirror_loss`（交付要求②）+ **风格项关闭**（`style_lerp=1.0` ⇒ 总奖励 = 纯任务项）| `track_default_pose_on_stand` ✅ 仍 −0.8；`mirror_loss` / `style_lerp` 属算法侧 → 见下表指针 |
| ⑤ | 观测 **378**（含 `base_lin_vel`）与 **全向方向分档指令**（`side_vel_min=0.15`）—— **V33-S 已具备，本层沿用**（非本层新设计）| ✅ 仍生效 |

**16 项白名单的今天状态**（T1k = T1l，除注明外）：`termination_penalty −200` ✅、
`track_lin_vel_xy_exp 1.0` ⚠️（**T1l → 1.5**）、`track_ang_vel_z_exp 1.0` ✅、`feet_air_time 1.0` ✅、
`feet_slide −0.1` ✅、`flat_orientation_l2 −1.0` ✅、`track_default_pose_on_stand −0.8` ✅、
`dof_pos_limits −1.0` ✅、`joint_deviation_hip −0.1` ✅、`joint_deviation_arms −0.1` ✅、
`lin_vel_z_l2 −0.1` ✅、`action_rate_l2 −0.05` ✅、`dof_acc_l2 −2e-7` ✅、`dof_torques_l2 −4e-6` ✅、
`joint_deviation_torso −0.1` ❌ **T1j → 0**、`joint_deviation_knee −0.05` ❌ **T1j → 0**。
⚠️ 算法侧（`mirror_loss_coeff` / `style_lerp` / `clip_min_std` …）→ 见 `EFFECTIVE_REWARDS.md`
   → `Instinct-Amp-Walk-T1a-G1-v0` / `-T1l-G1-v0` 的「算法侧」节。

## 推理与判读（已归档，勿在本文件重复）

* ⭐ **为什么做 T1a（回归测试，不是新研究）**：T0（`Instinct-Unified-OmniLoco-G1-v0`，2000 轮冷启动）实测
  `cmd=+0.8` 实际 **0.670 m/s（84%）**、6 s 净位移 **4.008 m**、P1 探针"维持"；而 AMP 版 S8（14500 轮）
  只有 **0.014 m/s** ⇒ **是 AMP 判别器（无指令条件 + 5 模态并集参考）把高速行走弄没了**，
  不是奖励形状、不是执行器、不是观测契约 → `instinct/docs/v33_tracking_gait_analysis.md` + 该版 run 日志
* T1a 补齐 5 项（观测 378 / 全向分档指令 / 静止恢复初始姿态 / `mirror_loss` / 风格项关闭）以回答
  "**T0 的能力还在不在**"；真正的单变量增量留给 **T1b**（只需把 `style_lerp` 翻回 0.75）→ 该版 run 日志
* ⚠️ `params` 里的 `SceneEntityCfg` **必须带上 `body_names`/`joint_names`** —— 作用域不同等于换了奖励
  （T1a2 审计踩过的坑）→ `docs/amp_walk/EXPERIMENTS.md` §P2
* ⚠️ 本层**只对齐了权重**，实现/作用域大面积不同 ⇒ 无法回答"T0 能力还在不在"；
  真正的实现对齐是 **T1a2** → `amp_walk_t1a2_env_cfg.py` + §E3

⚠️ 本文件**不改动、不删除**任何既有 cfg（用户要求：白名单化必须非破坏性）；
   备份快照见 `instinct/backups/config_g1_<TS>.tar.gz` + `MANIFEST_<TS>.md5`。
   ⚠️ 参考集/指令分布取自 V33-S 链（今天 = 5 clip + 全向分档指令）—— **grep 旁证、未运行期核对**。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-T1a-G1-v0`
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v33_env_cfg import G1AmpWalkV33SEnvCfg, G1AmpWalkV33SEnvCfg_PLAY

#: run_name（拼进 run 目录名；保持 obs/rew 标记便于工具识别）
AMP_WALK_T1A_RUN_NAME = "G1AmpWalkT1aMVPPlane_obs378_rew16_t0equiv_sym"

#: ⭐ T1a 生效的 16 项奖励权重 —— **逐项抄自 T0 run 的 `params/env.yaml`**（权威）。
#: 原则：**回归测试不做顺手调参**，权重必须与 T0 一致，否则无法归因。
T1A_REWARD_WEIGHTS: dict[str, float] = {
    "termination_penalty": -200.0,
    "track_lin_vel_xy_exp": 1.0,
    "track_ang_vel_z_exp": 1.0,
    "feet_air_time": 1.0,
    "feet_slide": -0.1,
    "flat_orientation_l2": -1.0,
    # ⭐ 等权替代 T0 的 `stand_still(−0.8)`：两者都是"静止时"门控，前者罚"动"、
    #    后者罚"偏离默认姿态"。后者正是交付要求②（静止恢复初始姿态）。
    "track_default_pose_on_stand": -0.8,
    "dof_pos_limits": -1.0,
    "joint_deviation_hip": -0.1,
    "joint_deviation_arms": -0.1,
    "joint_deviation_torso": -0.1,
    "joint_deviation_knee": -0.05,
    "lin_vel_z_l2": -0.1,
    "action_rate_l2": -0.05,
    "dof_acc_l2": -2.0e-7,
    "dof_torques_l2": -4.0e-6,
}

#: V33 独有、T1a **显式置 0**（字段保留 ⇒ 后续做消融时一行改回，不用删代码）
T1A_ZEROED: tuple[str, ...] = (
    "track_base_height",        # T0 没有
    "body_ang_vel_xy",          # T0 没有（且与风格项争夺"躯干姿态"的职责）
    "feet_lateral_width",       # T0 没有（−10，很重）
    "feet_crossing",            # T0 没有（−20，很重）
    "feet_swing_time_penalty",  # 本来就是 0.0
)


def _apply_t1a_reward_whitelist(cfg) -> None:
    """把奖励表变成"T0 等价集"：16 项按 T0 权重赋值 + 5 项显式置 0。

    ⚠️ **fail-fast**（用户要求：白名单必须可审计、不可静默漂移）：
      · 白名单/待关闭项都必须**存在**（改名或删除会立刻报错）
      · **未归类的奖励项直接报错** —— 将来有人新增奖励项而忘了归类，
        这里会拦住，而不是让它悄悄进训练。
    """
    seen: set[str] = set()
    for name, w in T1A_REWARD_WEIGHTS.items():
        if not hasattr(cfg.rewards, name):
            raise ValueError(f"[T1a] 白名单项不存在：rewards.{name}（被改名或删除了？）")
        getattr(cfg.rewards, name).weight = float(w)
        seen.add(name)
    for name in T1A_ZEROED:
        if not hasattr(cfg.rewards, name):
            raise ValueError(f"[T1a] 待关闭项不存在：rewards.{name}（被改名或删除了？）")
        getattr(cfg.rewards, name).weight = 0.0
        seen.add(name)

    # ---- 未归类检查 + 打印生效权重表（发车前过目、留档）----------------------
    all_names: set[str]
    try:
        import dataclasses

        all_names = {f.name for f in dataclasses.fields(cfg.rewards)}
    except Exception:  # pragma: no cover - configclass 不保证是 dataclass
        all_names = {k for k in vars(cfg.rewards) if not k.startswith("_")}

    # ⚠️ 祖先链用 `= None` 表示**删除**该项（`ManagerBase` 会跳过 `None` 的 term），
    #    这与"权重设成 0"是两回事：`None` 项**不可能产生任何奖励** ⇒ 无需归类。
    #    但**必须打印出来**，否则"悄悄少了一项"无法察觉（可审计性要求）。
    #    2026-10-07：这里曾把 `stand_still`（V2 已置 None）误报为"未归类"而拦下
    #    T1a 发车 —— 教训：闸门必须区分"已删除"与"未归类"。
    removed = sorted(n for n in all_names if getattr(cfg.rewards, n, None) is None)
    alive = all_names - set(removed)

    missing = sorted(alive - seen)
    if missing:
        raise ValueError(
            f"[T1a] 奖励白名单未覆盖：{missing}\n"
            "  新增奖励项必须**显式归类**（T1A_REWARD_WEIGHTS 或 T1A_ZEROED），"
            "否则本配置不可审计 —— 这是刻意设的闸门。\n"
            f"  （另有 {len(removed)} 项已被祖先链删除、不参与奖励：{removed}）"
        )
    n_on = sum(1 for n in alive if abs(getattr(cfg.rewards, n).weight) > 0.0)
    print(
        f"[T1a][REWARD WHITELIST] 字段共 {len(all_names)} 项："
        f"已删除(None) {len(removed)} 项 / 存活 {len(alive)} 项（非零 {n_on} 项）"
    )
    for name in removed:
        print(f"    -  {name:34s} <None 已删除，不参与奖励>")
    for name in sorted(alive):
        w = getattr(cfg.rewards, name).weight
        mark = "  " if abs(w) > 0.0 else "× "  # × = 已关闭
        print(f"    {mark}{name:34s} {w:+.6g}")


@configclass
class G1AmpWalkT1aEnvCfg(G1AmpWalkV33SEnvCfg):
    """T1a = V33-S 骨架（378 观测 / 396 AMP 状态 / 全向分档指令）+ T0 等价奖励集。

    风格项由**算法 cfg** 置 0（`style_lerp=1.0`），环境侧不碰 AMP 组 ⇒ T1b 只需要
    改算法 cfg 的一个数。
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_t1a_reward_whitelist(self)
        self.run_name = AMP_WALK_T1A_RUN_NAME


@configclass
class G1AmpWalkT1aEnvCfg_PLAY(G1AmpWalkV33SEnvCfg_PLAY):
    """回放用：与 T1a 训练同口径。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_t1a_reward_whitelist(self)
        self.run_name = AMP_WALK_T1A_RUN_NAME
