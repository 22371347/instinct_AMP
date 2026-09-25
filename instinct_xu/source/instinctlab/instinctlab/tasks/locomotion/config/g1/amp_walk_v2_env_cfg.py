"""AMP 行走教师 **v2** 配置：奖励集换成 AMP_mjlab 的 12 项配方。

为什么要有 v2
-------------
v1（`amp_walk_env_cfg.G1AmpWalkRewardsCfg`）的奖励集是
`G1FlatRewardsCfg`（学长做**越障**动作测试时攒下的 16 项）→ 覆盖 3 项权重 → 新增 5 项
= **21 项**。它和 AMP_mjlab `amp_loco` 的奖励表并不等价，实测后果：

* 训练 5000/10000 迭代后，机器人出现**单脚支撑 + 跳跃**的异常步态；
* 回合 464/1000 步，**66% 回合因触地终止**；
* 脚距均值 0.348（超过 0.30 上限，宽站姿）。

本文件**只做两件事**，且**不改任何奖励函数本体、不改任何现有文件**：

1. **删除** 9 个「AMP_mjlab 奖励表里根本没有、纯由越障基类继承而来」的项
   （`AMP_WALK_V2_REMOVED_TERMS`，用置 `None` 的方式，这是本仓库既定机制：
   `G1FlatEnvCfg.__post_init__` 的 run_name 生成器对其中 4 项都写了 `is not None` 保护）；
2. **对齐** 3 个与 AMP_mjlab 数值不一致的权重（`AMP_WALK_V2_WEIGHT_FIXES`）。

删除的理由（逐项）
------------------
======================== ======== ==========================================
被删项                    v1 权重  为什么删
======================== ======== ==========================================
``feet_air_time``        +1.0     函数内 `sum(in_contact) == 1` 才给分 →
                                   **构造上奖励单脚支撑**。AMP_mjlab 的
                                   amp_loco 奖励表无此项（只在其 velocity
                                   任务里有）。实测贡献仅 +0.018/s，但方向
                                   本身是错的。
``joint_deviation_knee`` −0.05    把膝关节往 default（直腿）拽。走路需要
``joint_deviation_hip``  −0.1     屈膝屈髋的腾空相，这 4 项与正常步态**正
``joint_deviation_arms`` −0.1     面对抗**。AMP_mjlab 无这 4 项。
``joint_deviation_torso``−0.1
``flat_orientation_l2``  −1.0     与新增的 `body_ang_vel_xy` 语义重叠；
                                   AMP_mjlab 只保留后者。
``stand_still``          −0.8     与新增的 `track_default_pose_on_stand`
                                   （−1.0，还带关节速度阻尼）语义重叠。
``lin_vel_z_l2``         −0.1     抑制竖直速度 → **直接压制跳跃的起跳/落地**；
                                   AMP_mjlab 靠 `track_root_height` + 3D
                                   锚点速度跟踪覆盖，无此单独项。
``dof_torques_l2``       −4e-6    数值上纯装饰（实测 −0.018/s），AMP_mjlab 无。
======================== ======== ==========================================

对齐的 3 个权重（AMP_mjlab → 本仓库）
-------------------------------------
=================== ========= ========== ================================
项                  v1 权重    v2 权重    AMP_mjlab 对应
=================== ========= ========== ================================
``action_rate_l2``  −0.05     **−0.01**  ``action_rate_l2`` = −0.01（**5×**）
``dof_pos_limits``  −1.0      **−10.0**  ``joint_pos_limits`` = −10.0
``dof_acc_l2``      −2.0e-7   **−2.5e-7**``joint_acc_l2`` = −2.5e-7
=================== ========= ========== ================================

``action_rate_l2`` 是这次最关键的一项：它实测是**最大单项成本**（−0.646/s，
而全部任务正向项合计才 ~+1.6/s）。跳跃恰好是「一次爆发 + 长时间不需要动作」的
**低动作频率**策略，走路则是持续换腿 —— 5× 的过强惩罚本身就在鼓励跳跃。

保留（已与 AMP_mjlab 对齐的 9 项）
---------------------------------
`track_lin_vel_xy_exp`(+1.5)、`track_ang_vel_z_exp`(+1.2)、`track_base_height`(+1.5)、
`body_ang_vel_xy`(+0.7)、`track_default_pose_on_stand`(−1.0)、`feet_lateral_width`(−10.0)、
`feet_crossing`(−20.0)、`termination_penalty`(−200.0)、`feet_slide`(−0.25)。
⇒ v2 共 **12 项**，与 AMP_mjlab `amp_loco` 的项数一致。

**刻意保留的偏离**（本次不动，避免一次改太多导致归因不清）
--------------------------------------------------------
1. `track_lin_vel_xy_exp` / `track_ang_vel_z_exp` 的 `std` 仍为 **0.5**
   （AMP_mjlab 是 1.0 / 3.14）—— 与学长同族 OmniLoco/Vault 教师保持形状一致，
   便于日后对照与蒸馏；且 `std` 变小是**更严**的要求，不是放宽。
2. `dof_pos_limits` 仍**只作用于踝关节**（AMP_mjlab 是全关节）。
3. `dof_acc_l2` 仍**只作用于髋/膝**（AMP_mjlab 是全关节）。
4. `feet_lateral_width` / `feet_crossing` 实测**几乎不激活**（−0.0014 / −0.0001，
   `max_cost=0.10` 封顶），因此宽站姿问题**不会**因为本文件而自动解决。

接口契约**完全不变**（本文件不碰契约）
-------------------------------------
动作 29D / 顺序 / `scale=1.0` / default offset / `clip=None`、观测 626 / 99 / 670、
时序 50 Hz、执行器 —— 全部继承 `G1AmpWalkEnvCfg`，**逐项与 `INTERFACE_CONTRACT.md` 相同**。
因此：

* v2 的 checkpoint 结构与 v1 **完全互换**，可以从 `model_5000.pt` 直接续训；
* `G1AmpWalkTeacherEnv` 契约校验器**直接复用**（期望值全是类属性）；
* **蒸馏 teacher 侧接口不受影响** —— 仍是「含 `model_*.pt` + `params/agent.yaml` 的
  run 目录」，`obs_format` / `num_actions=29` / normalizer 类型全都不变。

⚠️ 与 v1 唯一的不兼容点：**奖励语义变了**，所以 v1 的 value function 不再准确
（critic 需要重新适应），续训初期回报会先掉再涨，这是**预期**现象。
"""

from __future__ import annotations

from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .amp_walk_env_cfg import G1AmpWalkEnvCfg, G1AmpWalkEnvCfg_PLAY, G1AmpWalkRewardsCfg

# ============================================================================
# v2 与 v1 的差异（供日志/文档/run_name 引用，也是唯一的事实来源）
# ============================================================================

#: 从 `G1FlatRewardsCfg` 继承、但 AMP_mjlab `amp_loco` 奖励表里**没有**的 9 项。
#: v2 全部置 `None` 删除（`ManagerBase` 会跳过值为 `None` 的 term）。
AMP_WALK_V2_REMOVED_TERMS: tuple[str, ...] = (
    "feet_air_time",
    "joint_deviation_knee",
    "joint_deviation_hip",
    "joint_deviation_arms",
    "joint_deviation_torso",
    "flat_orientation_l2",
    "stand_still",
    "lin_vel_z_l2",
    "dof_torques_l2",
)

#: (项名, v1 权重, v2 权重)：3 个与 AMP_mjlab 不一致的权重 → 改成 AMP_mjlab 的值。
AMP_WALK_V2_WEIGHT_FIXES: tuple[tuple[str, float, float], ...] = (
    ("action_rate_l2", -0.05, -0.01),
    ("dof_pos_limits", -1.0, -10.0),
    ("dof_acc_l2", -2.0e-7, -2.5e-7),
)

#: v2 的 run_name（= 日志目录前缀，用于和 v1 区分）
AMP_WALK_V2_RUN_NAME = "G1AmpWalkV2Plane_obs626_rew12_alignAMP"


# ============================================================================
# 奖励集
# ============================================================================


@configclass
class G1AmpWalkV2RewardsCfg(G1AmpWalkRewardsCfg):
    """`G1AmpWalkRewardsCfg`(21 项) → AMP_mjlab 等价配方(12 项)。

    只做「删 9 项 + 改 3 个权重」，`func` / `params` 一律逐字沿用基类，
    **不动奖励函数本体，也不动 `G1FlatRewardsCfg`**（基类保持原样，
    OmniLoco / Vault / 越障任务全部不受影响）。
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 1. 删除 AMP_mjlab 配方里没有的 9 项 ---------------------------
        # 置 None 是本仓库既定机制：`G1FlatEnvCfg.__post_init__` 的 run_name
        # 生成器对 feet_air_time / stand_still / action_rate_l2 /
        # joint_deviation_knee 都做了 `is not None` 保护。
        for term_name in AMP_WALK_V2_REMOVED_TERMS:
            setattr(self, term_name, None)

        # --- 2. 对齐 3 个权重（只改 weight，func/params 与基类逐字一致）----
        # `action_rate_l2`：−0.05 → −0.01。实测它是最大单项成本，过强惩罚
        # 会奖励「低动作频率」的跳跃步态、压制持续换腿的正常行走。
        self.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
        # `dof_pos_limits`：−1.0 → −10.0（作用域仍限踝关节，见模块 docstring 偏离 #2）
        self.dof_pos_limits = RewTerm(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
                )
            },
        )
        # `dof_acc_l2`：−2.0e-7 → −2.5e-7（作用域仍限髋/膝，见偏离 #3）
        self.dof_acc_l2 = RewTerm(
            func=mdp.joint_acc_l2,
            weight=-2.5e-7,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
        )


# ============================================================================
# 环境配置
# ============================================================================


@configclass
class G1AmpWalkV2EnvCfg(G1AmpWalkEnvCfg):
    """训练用：场景 / 观测 / 事件 / 终止 / 时序 / 动作 / 命令 **全部继承 v1**，
    只替换 `rewards`。"""

    rewards: G1AmpWalkV2RewardsCfg = G1AmpWalkV2RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 父类把 run_name 设成 v1 的，这里覆盖成 v2（否则日志目录会和 v1 撞名）。
        self.run_name = AMP_WALK_V2_RUN_NAME


@configclass
class G1AmpWalkV2EnvCfg_PLAY(G1AmpWalkEnvCfg_PLAY):
    """Play 用：继承 v1 PLAY 的全部差异（关掉外力/推力事件、参考动作 debug_vis），
    只替换 `rewards`（play 阶段奖励不生效，但保持配置自洽）。"""

    rewards: G1AmpWalkV2RewardsCfg = G1AmpWalkV2RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V2_RUN_NAME
