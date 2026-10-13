"""**T2 环境侧：全权照抄 AMP_mjlab 的 walk 配方**（2026-10-09，用户 + 学长决策）。

背景
----
`AMP_mjlab` 已经证明"人形风格行走"可训；我们 V1~T1o 的观测/奖励/复位一直是**自己设计**的，
学长结论：**旧观测下的结论都不具备关键参考意义** ⇒ 本臂**照抄 mjlab**，只保留 3 处例外。

mjlab 源码位置（⚠️ 不在本 workspace 内，搜索时要给绝对路径）
-----------------------------------------------------------
* 环境：`/home/leo/rsl_project/AMP_mjlab/src/tasks/amp_loco/{amp_env_cfg.py,config/g1/env_cfgs.py}`
* 算法：`.../config/g1/rl_cfg.py`、`rsl_rl/{algorithms/amp_ppo.py,modules/discriminator.py,utils/utils.py}`
* `mjlab` 包本体：`/home/leo/miniconda3/envs/beyondmimic/lib/python3.11/site-packages/mjlab/`

照抄什么（逐行核对过）
----------------------
| 块 | mjlab walk 的值 |
|---|---|
| actor 观测 | 6 项 × `history_length=4`（`history_ordering="time"`）= 96×4 = **384**；`height_scan` 被 `del`（walk 把 terrain_scan 传感器也删了） |
| 奖励 | **13 项**（见 `G1AmpWalkT2RewardsCfg`） |
| 命令 | `DirectionalVelocityCommand`，resample **(6,15)**、`prob_low .20 / zero_frac_in_low .6`、`side .20`、`lin .40(fwd .7)`、`back .10`、`diag .10` |
| 终止 | `time_out` / `bad_orientation 70°` / `bad_base_height 0.5 m` |
| 复位 | ⭐ **RSI**：`reset_from_motion_data`，`idx = randint(0, total_frames)`（全池均匀抽帧），写 root pose/vel + joint pos/vel，**无偏移、无额外随机化**；**没有** `reset_base` / `reset_robot_joints` |
| 事件 | `push_robot`(**1~3 s**，v±1.0/±0.5/±0.4 + rot ±.52/±.52/±.78)、`foot_friction`、`base_com`、`encoder_bias` |
| 课程 | walk 把 `command_vel` 两段写成**相同值** ⇒ **等效无课程** |
| 时序 | `episode_length_s=20`、`decimation=4`、`dt=0.005` ⇒ 50 Hz（我们本来就一致） |

3 处例外（用户指定）
--------------------
* **A. `action_scale` 保持 1.0**（mjlab = per-joint `0.25·effort/stiffness`）。本文件不改它 ——
  契约校验器会断言 `scale == 1.0`。
* **B. 探索噪声按 action_scale 放大比缩小**（算法侧）：`init_noise_std 1.0 → 0.25`、
  `clip_min_std 0.05 → 0.0125`。见 `agents/amp_walk_t2_ppo_cfg.py`。
* **C. `height_scan` 保留并发 4 帧**（mjlab 把它删了；我们要求与其它专家策略对齐）
  ⇒ actor 观测 = `279×4 + 4×96` = **1500**。

⭐ 本臂同时修掉一个**此前的错误认知**
------------------------------------
我们**从来没有"机器人端 RSI"**：全仓 `reset_robot_state_by_reference` 只在 `shadowing` 任务里
被接上，locomotion 的 AMP-Walk 一直用 `reset_base`（±0.5 m / ±3.14 yaw / vel ±0.5）+
`reset_robot_joints`（scale 0.8~1.2）随机站姿复位。T1m 改的
`motion_start_from_middle_range` **只影响参考传感器播放的起点**（`motion_files/amass_motion.py:855-878`），
机器人自身的复位分布**没有变** ⇒ 那轮的"相位随机化"结论要重新看。
本臂把 RSI **真正接上**（`reset_from_reference` 事件），并把
`reset_base` / `reset_robot_joints` **整项去掉**（= mjlab 的复位状态就是参考帧状态）。

已知偏离（逐条记录，便于日后归因）
----------------------------------
| # | 项 | mjlab | 本臂 | 理由 |
|---|---|---|---|---|
| 1 | `self_collisions` | −0.1（自碰撞传感器） | ❌ **没有该项** | 我们的场景**没有自碰撞接触传感器**，PhysX 后端也不支持"多对多"过滤（见 `rewards_amp.py` 文件头）。兜底靠 `enabled_self_collisions=True` 的物理接触 |
| 2 | 判别器优化器 | **与策略共用** `Adam(lr=1e-3 adaptive)`；trunk wd 1e-3 / head wd 1e-2 | 参数对齐，但**仍是独立优化器**（`AdamW` lr 1e-3 / wd 1e-3 / 1e-2） | 改"共用优化器"要动 `wasabi.py`/`instinct_rl` 上游，风险大；本臂先只对齐**参数** |
| 3 | critic 观测 | actor 6 项 ×4 帧 + `base_lin_vel` + `height_scan` + 13 body 的 `body_pos_b`/`body_ori_b` | 沿用 `G1FlatCriticObsCfg`（**99 维单帧**） | 改它要新增 `body_pos_b/body_ori_b` 观测项并重定义特权信息契约，单独一臂做 |
| 4 | `feet_crossing` 的门控 | **无**方向门控（无条件罚 `sep<0`） | 保留我们的 `gate_lateral=True` | 我们 5-clip 参考里的**侧移 clip 本身就是交叉跨步**（实测 21% 帧 `sep<0`）；无条件罚会与 AMP 风格先验正面对撞（`amp_walk_v13_motion_selection.yaml` 原文） |
| 5 | 命令档 | `rel_standing_envs=0.05`、`rel_heading_envs=0.25`、`heading_command=True` | 全为 **0 / False** | 我们的 `DirectionalVelocityCommand` **fail-fast 要求**（它会重算/清零三轴，与分档采样语义打架，见 `mdp/commands.py`），契约校验器也断言 `heading_command=False` |
| 6 | 命令速度包络 | `lin_vel_x=(-0.7, 2.0)`（含慢跑） | **`(-0.5, 1.0)`** | 用户决策。实测我们 5 个 clip 的 `|v|` p95 ≤ 1.34、最快 1.09 ⇒ 2.0 档**完全在包络外**（会逼策略跑出判别器要抑制的姿态） |
| 7 | 摩擦随机化 | 只随脚部 geom，0.3~1.2，全脚共享 | 沿用我们的 `physics_material`（全身 body，0.25~0.8） | 需要脚部 geom 名清单（与 USD 绑定），且属 sim2real 鲁棒性项、非步态学习主因 |
| 8 | `encoder_bias` / `foot_friction` | 有 | ❌ 未移植 | 我们的 isaaclab 侧没有等价函数（`dr.encoder_bias` 是 mjlab 专有） |
| 9 | 判别器输入 | 390 = 13body×15×2 帧 | **396**（+`anchor_gravity_b` 6 维 = V33S 起的历史口径） | 改动要动 V20/V29/V33 的 obs 组与离线专家集（`amp_expert_states_396.pt`）；两侧一致 ⇒ 对比仍然有效 |
| 10 | 风格强度 | λ=0.75、`amp_reward_coef=0.1` ⇒ **0.025** | **本臂照抄 0.025**（T1l/T1o 是 4× 的 0.1） | 见算法侧 |
"""

from __future__ import annotations

import dataclasses
import math

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as Event
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.locomotion.mdp as locomotion_mdp

from .amp_walk_env_cfg import G1AmpWalkRewardsCfg
from .amp_walk_t1l_env_cfg import G1AmpWalkT1lEnvCfg, G1AmpWalkT1lEnvCfg_PLAY
from .amp_walk_v31_env_cfg import G1AmpWalkV31TeacherEnv

# ============================================================================
# ① actor 观测 = 1500（例外 C：height_scan 保留并发 4 帧）
# ============================================================================

#: 帧数 = mjlab 的 `history_length=4`（`history_ordering="time"`）
T2_HISTORY = 4

#: 契约：项顺序 = 声明顺序（`_validate_teacher_contract` 逐项比对）
T2_OBSERVATION_TERMS = (
    "height_scan",
    "velocity_command",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos_rel",
    "joint_vel",
    "last_action",
)
T2_SCAN_RAYS = 279
#: ⚠️ 校验器比较 `math.prod(dim)` ⇒ 必须是**标量**
T2_OBSERVATION_DIMS = (
    T2_SCAN_RAYS * T2_HISTORY,   # height_scan（例外 C：mjlab 没有这一项）
    3 * T2_HISTORY,              # velocity_command
    3 * T2_HISTORY,              # projected_gravity
    3 * T2_HISTORY,              # base_ang_vel
    29 * T2_HISTORY,             # joint_pos_rel
    29 * T2_HISTORY,             # joint_vel
    29 * T2_HISTORY,             # last_action
)
T2_OBSERVATION_SIZE = sum(T2_OBSERVATION_DIMS)      # == 1500

# ============================================================================
# ② 命令（mjlab walk 的分档结构；⚠️ 速度包络见偏离 #6）
# ============================================================================

#: mjlab walk：`lin_vel_x=(-0.7, 2.0)`、`lin_vel_y=(-1.0, 1.0)`、`ang_vel_z=(-1.57, 1.57)`。
#: ⚠️ 本臂按用户决策把 x 上限收到 **1.0**（偏离 #6：我们的参考集没有慢跑 clip）。
T2_COMMAND_RANGES = ((-0.5, 1.0), (-1.0, 1.0), (-1.57, 1.57))
#: mjlab walk：`resampling_time_range=(6.0, 15.0)`（"匀速直线"要被持续采样）
T2_COMMAND_HOLD = (6.0, 15.0)

#: mjlab walk 的方向分档概率（逐项核对 `env_cfgs.py` 的 `DirectionalVelocityCommandCfg`）。
#: `prob_low .20 + prob_side .20 + prob_lin .40 + prob_back .10 + 斜向 .10 = 1.0`
T2_CMD_PROBS = {
    "prob_low": 0.20,
    "low_vel": 0.10,
    "zero_frac_in_low": 0.60,        # mjlab 0.6（我们历史值 0.3）
    "prob_side": 0.20,
    "prob_lin": 0.40,
    "prob_lin_fwd": 0.70,
    "straight_lin_vel_range": (0.4, 1.0),   # mjlab (0.4, 2.0) → 按包络收到 1.0
    "straight_ang_vel_range": (-0.1, 0.1),
    "prob_back": 0.10,
    "back_lin_vel_range": (-0.7, -0.4),
    "back_ang_vel_range": (-0.1, 0.1),
    #: ⚠️ 我们特有的字段（mjlab 没有）：纯侧向档的 `|vy|` 下限。
    #: mjlab 的侧向档就是 `vy ~ U(lin_vel_y)`（可低于 `low_vel` ⇒ 静止门控会把它当静止）。
    #: 本臂取 **0.0 = 与 mjlab 逐字一致**（我们历史值 0.15，是修"监督自相矛盾"的自创补丁）。
    "side_vel_min": 0.0,
    "side_ang_vel_range": None,
}


@configclass
class G1AmpWalkT2PolicyObsCfg(ObsGroup):
    """T2 的 actor 契约：**1500 = 4×(279 + 96)**。

    与 T1o（663）的唯一差别：`height_scan` 也堆叠 **4 帧**（用户："height_scan 也是 4 帧"）。

    噪声**照抄 mjlab**（`amp_env_cfg.py:52-79` 的 `Unoise`）：
    `base_ang_vel ±0.2` / `projected_gravity ±0.05` / `joint_pos ±0.01` / `joint_vel ±0.5`
    / `height_scan ±0.1`；`command` 与 `actions` **不加噪**。
    ⚠️ `joint_vel` 我们历史口径是 ±1.5（OmniLoco 系）⇒ 本臂改为 mjlab 的 ±0.5。
    """

    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
        noise=Unoise(n_min=-0.1, n_max=0.1),
        history_length=T2_HISTORY,
        flatten_history_dim=True,
    )
    velocity_command = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
        history_length=T2_HISTORY,
        flatten_history_dim=True,
    )
    projected_gravity = ObsTerm(
        func=instinct_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
        history_length=T2_HISTORY,
        flatten_history_dim=True,
    )
    base_ang_vel = ObsTerm(
        func=instinct_mdp.base_ang_vel,
        noise=Unoise(n_min=-0.2, n_max=0.2),
        history_length=T2_HISTORY,
        flatten_history_dim=True,
    )
    joint_pos_rel = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        history_length=T2_HISTORY,
        flatten_history_dim=True,
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        noise=Unoise(n_min=-0.5, n_max=0.5),
        history_length=T2_HISTORY,
        flatten_history_dim=True,
    )
    last_action = ObsTerm(
        func=instinct_mdp.last_action,
        history_length=T2_HISTORY,
        flatten_history_dim=True,
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


# ============================================================================
# ③ 奖励 = mjlab 的 13 项
# ============================================================================

#: mjlab walk 的奖励表（权重 → 我们的实现）。`self_collisions` 见偏离 #1。
T2_REWARD_WEIGHTS = {
    "track_anchor_linear_velocity": 1.5,     # mjlab track_anchor_linear_velocity (std 1.0)
    "track_anchor_angular_velocity": 1.2,    # mjlab track_anchor_angular_velocity (std 3.14)
    "track_base_height": 1.5,                # mjlab track_root_height (std 0.3)
    "body_ang_vel_xy": 0.7,                  # mjlab body_ang_vel_xy_l2 (std 3.14)
    "track_default_pose_on_stand": -1.0,
    "feet_lateral_width": -10.0,
    "feet_crossing": -20.0,
    "feet_slide": -0.25,                     # mjlab foot_slip
    "action_rate_l2": -0.01,
    "dof_pos_limits": -10.0,                 # mjlab joint_pos_limits（**线性**软限位，全关节）
    "dof_acc_l2": -2.5e-7,                   # mjlab joint_acc_l2（全关节）
    "termination_penalty": -200.0,           # mjlab is_terminated
}
#: 上述 12 项 + `self_collisions`（我们无传感器，未实现）= mjlab 的 13 项
T2_EXPECT_NONZERO = len(T2_REWARD_WEIGHTS)

#: mjlab 的 `feet_lateral_width` 用 `sep_floor=0.15`（我们自己的标定值是 0.145，差 5 mm；
#: 本臂**照抄 mjlab**，便于归因）。
T2_SEP_FLOOR = 0.15
#: ⚠️ 偏离 #4：`feet_crossing` 保留我们的方向门控（参考集的侧移 clip 是交叉跨步）
T2_FEET_CROSSING_GATE_LATERAL = True

#: mjlab 没有、本臂**整项删除**的奖励（一律置 `None`，字段保留 ⇒ 消融时一行改回）
T2_DROPPED_TERMS = (
    "track_lin_vel_xy_exp",      # → 换成 track_anchor_lin_vel_exp（含 v_z）
    "track_ang_vel_z_exp",       # → 换成 track_anchor_ang_vel_exp（含 ω_xy）
    "feet_air_time",
    "flat_orientation_l2",
    "lin_vel_z_l2",              # mjlab 把"别上下弹"放在三维速度跟踪误差里（见 rewards_amp.py）
    "dof_torques_l2",
    "joint_deviation_hip",
    "joint_deviation_arms",
    "joint_deviation_torso",
    "joint_deviation_knee",
    "joint_pos_limits_margin",   # 我们 T1e 自创的软边距（mjlab 用的是线性软限位 ×−10）
    "feet_swing_time_penalty",
    "stand_still",
)


@configclass
class G1AmpWalkT2RewardsCfg(G1AmpWalkRewardsCfg):
    """mjlab walk 的奖励集（13 项，其中 12 项落地）。

    两个新的**速度/角速度跟踪项**用 mjlab 的**函数**（含竖直速度 / 含 ω_xy），
    而不是仓库既有的 xy-only / wz-only 版本 —— 见 `rewards_amp.py` 的
    `track_anchor_lin_vel_exp` / `track_anchor_ang_vel_exp` 的 docstring。

    其余项全部复用 `G1AmpWalkRewardsCfg` 的类体字面值（它们本来就是 mjlab 的值：
    `track_base_height 1.5/0.3`、`body_ang_vel_xy 0.7/3.14`、
    `track_default_pose_on_stand −1.0`（`gate_tau 0.25` / `pos_scale 4.0` / `vel_scale 0.5`
    / `pos_joint_weights={waist_yaw:3.0}`）、`feet_lateral_width −10`、`feet_crossing −20`、
    `feet_slide −0.25`、`termination_penalty −200`），只把**长度尺**与**缺失项**补齐。
    """

    track_anchor_linear_velocity = RewTerm(
        func=locomotion_mdp.track_anchor_lin_vel_exp,
        weight=T2_REWARD_WEIGHTS["track_anchor_linear_velocity"],
        params={"command_name": "base_velocity", "std": 1.0},
    )
    track_anchor_angular_velocity = RewTerm(
        func=locomotion_mdp.track_anchor_ang_vel_exp,
        weight=T2_REWARD_WEIGHTS["track_anchor_angular_velocity"],
        params={"command_name": "base_velocity", "std": 3.14},
    )

    def __post_init__(self):
        # ---- 1. 删掉 mjlab 没有的项（含我们自创的 margin 版限位）----
        for name in T2_DROPPED_TERMS:
            if hasattr(self, name):
                setattr(self, name, None)

        # ---- 2. 动作/加速度代价：全关节 + mjlab 权重 ----
        self.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
        # mjlab: `joint_pos_limits` = isaaclab 同款**线性**软限位（`Σ clamp`），**全关节**，
        # 权重 −10。⚠️ 不是"margin/power"版（我早前笔记写反了，已按源码更正）。
        self.dof_pos_limits = RewTerm(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        self.dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

        # ---- 3. 步宽地板：照抄 mjlab 的 0.15 ----
        self.feet_lateral_width = RewTerm(
            func=locomotion_mdp.feet_lateral_width_cost,
            weight=-10.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "sep_floor": T2_SEP_FLOOR,
                "gate_lateral": True,
                "max_cost": 0.10,
                "walk_max_lin_vel": 1.0,
                "walk_gate_width": 0.2,
            },
        )
        # ---- 4. 真交叉：权重照抄 −20；门控见偏离 #4 ----
        self.feet_crossing = RewTerm(
            func=locomotion_mdp.feet_crossing_cost,
            weight=-20.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "gate_lateral": T2_FEET_CROSSING_GATE_LATERAL,
                "max_cost": 0.10,
                "walk_max_lin_vel": 1.0,
                "walk_gate_width": 0.2,
            },
        )
        # ---- 5. 低速回 default 站姿：**清掉 V23 的全关节权重/死区表**，回到 mjlab 口径 ----
        self.track_default_pose_on_stand = RewTerm(
            func=locomotion_mdp.track_default_pose_on_stand,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "pos_scale": 4.0,
                "vel_scale": 0.5,
                "gate_tau": 0.25,
                "pos_joint_weights": {"waist_yaw_joint": 3.0},
            },
        )


# ============================================================================
# ④ 终止 = mjlab 的 3 项
# ============================================================================


@configclass
class G1AmpWalkT2TerminationsCfg:
    """`time_out` / `bad_orientation 70°` / `bad_base_height 0.5 m`。

    ⚠️ 与 mjlab 的差别：mjlab 靠**姿态与高度**终止；我们旧口径的 `base_contact`
    （`torso/肩/肘/腕/髋/膝` 接触 1 N 即 −200 终止）**被去掉**了 —— 它是"膝盖碰地"
    这类姿态的硬兜底，而 mjlab 认为那是可以靠奖励修的正常行为。
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    )
    bad_base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.5},
    )


# ============================================================================
# ⑤ 事件 = mjlab 的事件集 + ⭐ 真正接上 RSI
# ============================================================================


@configclass
class G1AmpWalkT2EventsCfg:
    """`physics_material`（我们的口径，见偏离 #7）+ `base_com` + `push_robot`(1~3 s) + **RSI**。

    ⚠️ 与旧口径的**三处结构差异**：
    1. **删掉 `reset_base`**（均匀 root pose ±0.5 m / yaw ±3.14 + 速度 ±0.5）；
    2. **删掉 `reset_robot_joints`**（关节 scale 0.8~1.2 + 关节速度 ±1.0）；
    3. **删掉 `add_base_mass`**（躯干 ±5 kg —— mjlab 没有这项质量随机化）。

    mjlab 的复位**只有** `reset_from_motion_data`：状态 = 参考帧状态（无偏移、无随机化）。
    我们对应的函数是 `instinct_mdp.reset_robot_state_by_reference`，
    比例参数全部设为 **1.0**、位置偏移 **[0,0,0]** ⇒ 与 mjlab 等价。
    """

    physics_material = Event(  # noqa: E501  # 偏离 #7：mjlab 只随脚部 geom（0.3~1.2）
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.25, 0.8),
            "dynamic_friction_range": (0.2, 0.6),
            "restitution_range": (0.0, 0.8),
            "num_buckets": 64,
        },
    )
    # mjlab `base_com`：torso 质心偏移 x/y ±0.025、z ±0.03（`operation="add"`）
    base_com = Event(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.03, 0.03)},
        },
    )
    # mjlab `push_robot`：**每 1~3 s 推一次**，力度比我们旧口径（10~15 s、±0.5）大得多
    push_robot = Event(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-0.5, 0.5),
                "z": (-0.4, 0.4),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )
    # ⭐ RSI：把机器人放到**参考帧**上（起点由 `motion_start_from_middle_range` 随机抽）
    reset_from_reference = Event(
        func=instinct_mdp.reset_robot_state_by_reference,
        mode="reset",
        params={
            "position_offset": [0.0, 0.0, 0.0],
            "dof_vel_ratio": 1.0,
            "base_lin_vel_ratio": 1.0,
            "base_ang_vel_ratio": 1.0,
        },
    )


# ============================================================================
# ⑥ 装配（含 fail-fast 回读）
# ============================================================================

#: RSI 起点在 clip 的 **[0%, 100%]** 上均匀采样（mjlab = 全池均匀抽帧）
T2_RSI_BEFORE = (0.0, 0.0)
T2_RSI_AFTER = (0.0, 1.0)


def _apply_t2(env_cfg) -> None:
    # ---- 0. 契约自检 --------------------------------------------------------
    if T2_OBSERVATION_SIZE != 1500:
        raise ValueError(f"[T2] 观测契约自检失败：{T2_OBSERVATION_DIMS} 合计 {T2_OBSERVATION_SIZE}")

    # ---- 1. actor 观测 → 1500 ----------------------------------------------
    env_cfg.observations.policy = G1AmpWalkT2PolicyObsCfg()
    pol = env_cfg.observations.policy
    got_hist = tuple(int(getattr(pol, n).history_length or 1) for n in T2_OBSERVATION_TERMS)
    if got_hist != (T2_HISTORY,) * len(T2_OBSERVATION_TERMS):
        raise ValueError(f"[T2] history_length 回读不符：{got_hist}")
    if hasattr(pol, "base_lin_vel"):
        raise ValueError("[T2] base_lin_vel 仍在 policy 观测里（应已删除，对齐 mjlab）")

    # ---- 2. 奖励集整体替换为 mjlab 的 13 项 --------------------------------
    env_cfg.rewards = G1AmpWalkT2RewardsCfg()
    R = env_cfg.rewards
    for name, want in T2_REWARD_WEIGHTS.items():
        term = getattr(R, name, None)
        if term is None:
            raise ValueError(f"[T2] 奖励项缺失：{name}")
        if abs(float(term.weight) - float(want)) > 1e-12:
            raise ValueError(f"[T2] {name} 权重 {term.weight} != {want}")
    # 反向检查：不能再有第 13 个非零项（`self_collisions` 是唯一允许缺席的）
    n_on = sum(
        1 for n in dir(R) if getattr(getattr(R, n, None), "weight", None) not in (None, 0)
    )
    if n_on != T2_EXPECT_NONZERO:
        raise ValueError(
            f"[T2] 非零奖励项 = {n_on}（应为 {T2_EXPECT_NONZERO}）；"
            f"多出来的项："
            f"{[n for n in dir(R) if getattr(getattr(R, n, None), 'weight', None) not in (None, 0)]}"
        )

    # ---- 3. 命令：mjlab 的分档结构（速度包络见偏离 #6）---------------------
    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range = T2_COMMAND_HOLD
    cmd.rel_standing_envs = 0.0      # 偏离 #5
    cmd.rel_heading_envs = 0.0       # 偏离 #5（契约校验器硬性要求）
    cmd.heading_command = False      # 偏离 #5
    cmd.ranges.lin_vel_x = T2_COMMAND_RANGES[0]
    cmd.ranges.lin_vel_y = T2_COMMAND_RANGES[1]
    cmd.ranges.ang_vel_z = T2_COMMAND_RANGES[2]
    cmd.ranges.heading = (-math.pi, math.pi)
    for k, v in T2_CMD_PROBS.items():
        setattr(cmd, k, v)
    for k, v in T2_CMD_PROBS.items():
        if isinstance(v, tuple):
            if tuple(getattr(cmd, k)) != v:
                raise ValueError(f"[T2] 命令参数 {k} 回读失败：{getattr(cmd, k)} != {v}")
        elif getattr(cmd, k) != v:
            raise ValueError(f"[T2] 命令参数 {k} 回读失败：{getattr(cmd, k)} != {v}")

    # ---- 4. 终止 → mjlab 的 3 项（去掉 base_contact）-----------------------
    env_cfg.terminations = G1AmpWalkT2TerminationsCfg()
    if tuple(env_cfg.terminations.__dict__) != ("time_out", "bad_orientation", "bad_base_height"):
        raise ValueError(f"[T2] 终止项意外：{tuple(env_cfg.terminations.__dict__)}")

    # ---- 5. 事件 → mjlab 的事件集 + RSI ------------------------------------
    env_cfg.events = G1AmpWalkT2EventsCfg()
    for gone in ("reset_base", "reset_robot_joints", "add_base_mass", "base_external_force_torque"):
        if hasattr(env_cfg.events, gone):
            raise ValueError(f"[T2] 事件 {gone} 未被移除（mjlab 没有它）")

    # ---- 6. ⭐ 真正打开 RSI 的随机起点 --------------------------------------
    mr = env_cfg.scene.motion_reference
    bufs = mr.motion_buffers
    if len(bufs) != 1:
        raise ValueError(f"[T2] 预期只有 1 个 motion buffer，实际 {list(bufs)}")
    (key, old_buf), = bufs.items()
    got = tuple(old_buf.motion_start_from_middle_range)
    if got != T2_RSI_BEFORE:
        raise ValueError(f"[T2] motion_start_from_middle_range 基线不符：{got}（预期 {T2_RSI_BEFORE}）")
    if old_buf.motion_bin_length_s is not None:
        raise ValueError(f"[T2] motion_bin_length_s 应为 None，实际 {old_buf.motion_bin_length_s}")
    # ⚠️ 必须用 `dataclasses.replace`：`motion_reference_cfg` 是**模块级共享对象**
    new_buf = dataclasses.replace(old_buf, motion_start_from_middle_range=T2_RSI_AFTER)
    env_cfg.scene.motion_reference = dataclasses.replace(mr, motion_buffers={**bufs, key: new_buf})
    back = tuple(env_cfg.scene.motion_reference.motion_buffers[key].motion_start_from_middle_range)
    if back != T2_RSI_AFTER:
        raise ValueError(f"[T2] RSI 回读失败：{back}（应为 {T2_RSI_AFTER}）")

    # ---- 7. 打印 ----------------------------------------------------------
    print("[T2][COPY MJLAB] actor 观测 378 → **1500**（4 帧 × 375）：")
    print(f"    项顺序 = {T2_OBSERVATION_TERMS}")
    print(f"    逐项维度 = {T2_OBSERVATION_DIMS}（sum={T2_OBSERVATION_SIZE}）")
    print(
        "    例外 C：height_scan 279 **发 4 帧**（mjlab 直接把这一项删了）⇒ "
        f"{T2_SCAN_RAYS}×4 + 96×4 = 1500；❌ base_lin_vel 已删除"
    )
    print(f"[T2][COPY MJLAB] 奖励 {T2_EXPECT_NONZERO} 项（mjlab 13 项中缺 self_collisions，见偏离 #1）：")
    for name, w in T2_REWARD_WEIGHTS.items():
        _p = getattr(getattr(R, name), "params", {}) or {}
        print(f"    {name:32s} {w:+.3g}   {dict(_p)}")
    print(f"[T2][COPY MJLAB] 命令：ranges={T2_COMMAND_RANGES} hold={T2_COMMAND_HOLD}")
    print(f"    {T2_CMD_PROBS}")
    print("[T2][COPY MJLAB] 终止：time_out / bad_orientation 70° / bad_base_height 0.5 m（去掉 base_contact）")
    print(
        "[T2][COPY MJLAB] 事件：physics_material + base_com + push_robot(1~3 s, 强推) "
        "+ **reset_from_reference（真 RSI）**"
    )
    print(
        f"    ❌ 已移除 reset_base / reset_robot_joints / add_base_mass"
        f"（T1m 的 RSI 只改了参考播放起点，机器人复位分布没变 —— 本臂修正）"
    )
    print(f"[T2][COPY MJLAB] RSI 起点 {got} → {back}（clip 内均匀采样，对齐 mjlab 的全池抽帧）")


@configclass
class G1AmpWalkT2EnvCfg(G1AmpWalkT1lEnvCfg):
    """T2：obs 1500 / mjlab 13 项奖励 / mjlab 命令 / mjlab 终止 / mjlab 事件 + 真 RSI。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_t2(self)
        self.run_name = "G1AmpWalkT2MVPPlane_obs1500_rew12_mjlabcopy_rsi_cold"


@configclass
class G1AmpWalkT2EnvCfg_PLAY(G1AmpWalkT1lEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_t2(self)
        # 回放：不推、不复位（episode 已无限长）
        self.events.push_robot = None
        self.events.reset_from_reference = None
        self.run_name = "G1AmpWalkT2MVPPlane_obs1500_rew12_mjlabcopy_rsi_cold_PLAY"


class G1AmpWalkT2TeacherEnv(G1AmpWalkV31TeacherEnv):
    """契约校验器：只覆盖 T2 改动的期望值（观测 / 命令范围 / 保持时长）。

    其余检查（动作维度 29、关节顺序、`scale == 1.0`、`dt=0.005/decimation=4`、
    `heading_command == False`、`rel_standing_envs == 0`、default-offset 语义探针、
    执行器配置、无课程……）全部继承 —— 那些本臂**一个字都没改**。
    """

    expected_policy_terms = T2_OBSERVATION_TERMS
    expected_policy_dims = T2_OBSERVATION_DIMS
    expected_scan_rays = T2_SCAN_RAYS
    expected_command_ranges = T2_COMMAND_RANGES
    expected_command_hold = T2_COMMAND_HOLD
    expected_standing_ratio = 0.0


__all__ = [
    "T2_CMD_PROBS",
    "T2_COMMAND_HOLD",
    "T2_COMMAND_RANGES",
    "T2_DROPPED_TERMS",
    "T2_EXPECT_NONZERO",
    "T2_HISTORY",
    "T2_OBSERVATION_DIMS",
    "T2_OBSERVATION_SIZE",
    "T2_OBSERVATION_TERMS",
    "T2_REWARD_WEIGHTS",
    "T2_RSI_AFTER",
    "T2_SCAN_RAYS",
    "G1AmpWalkT2EnvCfg",
    "G1AmpWalkT2EnvCfg_PLAY",
    "G1AmpWalkT2EventsCfg",
    "G1AmpWalkT2PolicyObsCfg",
    "G1AmpWalkT2RewardsCfg",
    "G1AmpWalkT2TeacherEnv",
    "G1AmpWalkT2TerminationsCfg",
]
