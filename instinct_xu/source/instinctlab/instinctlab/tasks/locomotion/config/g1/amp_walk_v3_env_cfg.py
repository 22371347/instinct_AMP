"""V3：修 v2 的「σ 膨胀」退步 —— `action_rate_l2` −0.01→−0.02 + 探索噪声起点 1.0→0.3。

## ⚠️ 本层对**今天**仍有影响的部分

* ⭐ **机制洞察（永久有效，已归档 §B5）**：`action_rate_l2 = mean((Δa)²)` 是二次型、
  **每一步每个关节都有梯度**的**密集**信号；而 exp 核跟踪奖励**在误差小时梯度趋近 0**。
  ⇒ **削掉它 = 抽走奖励集里绝大部分密集梯度** ⇒ PPO 的 σ 会**膨胀**
  （方差大→探索多→梯度估计更稳）。**今天把 `action_rate_l2` 往下调时，要预期 σ 会胀。**
* ⭐ **「跳跃是'大探索噪声'下才成立的吸引子」**（支撑腿自稳、对 ±57° 噪声鲁棒；平滑交替
  步态依赖精确摆动腿时序，会被噪声摧毁）⇒ 这是「局部收敛到单脚跳」的**机制解释**。
* 本层两个**具体数值**今天都已不生效（见下表）；只有上面两条**结论**仍在用。

## 本层改动一览

| # | 改动 | 现状（今日 T1l 实际生效）|
|---|---|---|
| ① | 本文件：`action_rate_l2` −0.01 → **−0.02**（补密集梯度；不回到 v1 的 −0.05）| **已被覆盖** → **−0.05** |
| ② | agent cfg（`agents/amp_walk_v3_ppo_cfg.py`）：`init_noise_std` 1.0 → **0.3**（±57°→±17°）| **已被覆盖** → `init_noise_std **1.0**`、`clip_min_std 1e-12`（今天 T1l 用 `amp_walk_t1i_ppo_cfg`）|
| ③ | **从零训练**（不续训已塌陷的 v2 ckpt）| 方法论结论仍在用 → §C4 |
| ④ | 参考数据集 17 → **7 clip**（`motions/amp_walk/amp_walk_v3_motion_selection.yaml`）| **已被 V18 覆盖** → **5 clip**（`amp_walk_v13_motion_selection.yaml`）|
| ⑤ | 指令分布换成 AMP_mjlab v4 的（原地/纵向/横向/斜向/后退五档；vx(−0.7,2.0)/vy(±1.0)/wz(±1.57)）| 分档**结构** ✅ 沿用；**范围已被改窄**（今天 ≈ vx(−0.5,1.0)/vy(−0.5,0.5)/wz(−0.8,0.8)）|

## 推理与判读（已归档，勿在本文件重复）

* `action_rate_l2` 的密集梯度机制 + σ 实测（`Policy/mean_noise_std` 0.766→1.580 = ±90°）
  + v2 全面退步（回合 604.7→560.0、触地终止 0.5498→0.6772、脚距 0.3518→0.4616）→ §B5
* 「从已塌陷的 ckpt 续训基本出不来自吸引子，从零重学 ~2500 轮可接受」→ §C4
* 「运动数据过多会摊平判别器风格分布」→ 关联 §D1（⚠️ §D1 的结论是**改命令分布解决不了、
  只能补参考数据**）

⚠️ 两项**刻意不照抄 AMP_mjlab**（照抄会报错/失效，不是保守）：`rel_standing_envs` 仍 **0.0**
   （我们的实现有 fail-fast）、`heading_command` 仍 **False**（开了 `wz` 会被航向环重算、
   方向分档直接失效）。
⚠️ **已知覆盖缺口**：精选 7 clip 实测 |wz| 上限 ≈1.13 而指令范围 ±1.57 ⇒ **wz 超出**
   （大角速度下步态劣化先怀疑这里）；⚠️ **未核实**今天的 5-clip 集是否仍有此缺口。
⚠️ 本次不动：**不加**「步宽过宽」惩罚（脚距 0.352→0.462 的**第二病灶本轮不解决**）、
   不改速度/角速度 `std`（仍 0.5）、不动 `G1FlatRewardsCfg` 与 v1/v2 的任何文件。
⚠️ 接口契约**完全不变**（动作/观测/时序/执行器与 `INTERFACE_CONTRACT.md` 逐项相同）
   ⇒ ckpt / ONNX / MuJoCo 回放 / **蒸馏 teacher 侧接口**全部不受影响。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V3-G1-v0`
"""

from __future__ import annotations

import copy
import os

from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .amp_walk_env_cfg import (
    G1AmpWalkCommandsCfg,
    G1AmpWalkTeacherEnv,
    MOTION_DIR,
    MOTION_NAME,
)
from .amp_walk_v2_env_cfg import (
    AMP_WALK_V2_RUN_NAME,
    G1AmpWalkV2EnvCfg,
    G1AmpWalkV2EnvCfg_PLAY,
    G1AmpWalkV2RewardsCfg,
)

#: v3 的 run_name（= 日志目录前缀）。`ar` = action_rate，`ns` = noise std，
#: `rew14` = 奖励集项数（v2 的 12 项 + 恢复的 flat_orientation_l2 / joint_deviation_torso）。
AMP_WALK_V3_RUN_NAME = "G1AmpWalkV3Plane_obs626_rew14_ar0.02_ns0.3"

#: v3 的 AMP 参考数据集 = AMP_mjlab 精选 7 clip（6 行走 + 1 慢跑）。
#: 动机与逐条对应见该 yaml 的注释；要恢复 17 clip 只需把本常量改回
#: `MOTION_SELECTION_FILE`。
AMP_WALK_V3_MOTION_SELECTION_FILE = os.path.join(
    MOTION_DIR, "amp_walk_v3_motion_selection.yaml"
)

#: ⭐ 恢复的躯干水平项权重。`flat_orientation_l2` 的 `func` 是
#: `sum(projected_gravity_b[:, :2]²)` = **`torso_link`（= 关节树根）相对重力方向倾斜的 sin²**。
#: **用户选定 −1.0 = 基类（v1）原值**（最保守）。
#: ⚠️ 实测提醒（保留供日后对比）：v1 用 −1.0 时该项贡献仅 **−0.016/s**
#: （反推平均倾斜 ~7.2°），对照任务正向项 ~+1.6/s ⇒ **−1.0 几乎不构成约束**
#: （歪 10° 才 −0.030、20° 才 −0.117 = 正向收益的 7%）。
#: 若本轮姿态歪斜没改善，优先把它提到 **−5.0**（歪 20° → −0.585，占 36%）。
AMP_WALK_V3_FLAT_ORI_WEIGHT = -1.0

#: v3 的指令范围 = AMP_mjlab v4 的 `env_cfgs.py:285-290`。
AMP_WALK_V3_COMMAND_RANGES = ((-0.7, 2.0), (-1.0, 1.0), (-1.57, 1.57))


def apply_amp_walk_v3_command_ranges(cmd_cfg) -> None:
    """按 AMP_mjlab v4 写回 v3 的指令**分档分布**与**范围**。

    对照 `~/rsl_project/AMP_mjlab/src/tasks/amp_loco/config/g1/env_cfgs.py:266-290`：

    | 字段 | v1 | **v3**（= AMP_mjlab v4）|
    |---|---|---|
    | `prob_low`（含 60% 精确 (0,0,0)）| 0.15 | **0.20** |
    | `prob_side`（纯侧向 $v_x\\equiv 0$）| 0.25 | **0.20** |
    | `prob_lin`（纯前后 $v_y\\equiv 0$）| 0.25 | **0.40** |
    | `prob_lin_fwd`（其中持续直线前向占比）| 0.6 | **0.7** |
    | `prob_back`（专属持续后退）| 0.10 | 0.10 |
    | （余下）均匀斜向 | 0.25 | **0.10** |
    | `straight_lin_vel_range` | (0.5, 1.2) | **(0.4, 2.0)** |
    | `back_lin_vel_range` | (−1.0, −0.5) | **(−0.7, −0.4)** |
    | `lin_vel_x` | (−1.0, 1.2) | **(−0.7, 2.0)** |
    | `lin_vel_y` | ±0.5 | **±1.0** |
    | `ang_vel_z` | ±0.8 | **±1.57** |
    | `resampling_time_range` | (10, 10) | **(6, 15)** |

    ⇒ 分档概率和 = 0.20+0.20+0.40+0.10 = **0.90**，余 **0.10** 为均匀斜向（与
    AMP_mjlab 注释「low 0.20 + side 0.20 + lin 0.40(fwd 70%) + back 0.10 + diag 0.10」**逐位一致**）。

    ⚠️ **两项刻意不照抄**（照抄会直接报错 or 失效，不是保守）：
    - `rel_standing_envs` 仍为 **0.0**（AMP_mjlab 是 0.05）：
      我们的 `DirectionalVelocityCommand.__init__` 有 **fail-fast 断言**，
      因为站立语义已交给 `prob_low` 分支，两套机制叠加会互相打架。
    - `heading_command` 仍为 **False**（AMP_mjlab 是 True，且 `rel_heading_envs=0.25`）：
      开了之后 `wz` 会被**航向环重算**，我们手工采样的方向分档会**直接失效**。
    """
    cmd_cfg.resampling_time_range = (6.0, 15.0)
    cmd_cfg.rel_standing_envs = 0.0      # ⚠️ 不能改（有 fail-fast）
    cmd_cfg.rel_heading_envs = 0.0       # ⚠️ 同上
    cmd_cfg.heading_command = False      # ⚠️ 开了方向分档就失效
    cmd_cfg.ranges.lin_vel_x = AMP_WALK_V3_COMMAND_RANGES[0]
    cmd_cfg.ranges.lin_vel_y = AMP_WALK_V3_COMMAND_RANGES[1]
    cmd_cfg.ranges.ang_vel_z = AMP_WALK_V3_COMMAND_RANGES[2]
    cmd_cfg.ranges.heading = (-1.57, 1.57)

    # 分档概率（和 = 0.90，余 0.10 为均匀斜向）
    cmd_cfg.prob_low = 0.20
    cmd_cfg.low_vel = 0.1
    cmd_cfg.zero_frac_in_low = 0.6      # 低速档里 60% 给精确 (0,0,0)
    cmd_cfg.prob_side = 0.20
    cmd_cfg.prob_lin = 0.40
    cmd_cfg.prob_lin_fwd = 0.7
    cmd_cfg.straight_lin_vel_range = (0.4, 2.0)
    cmd_cfg.straight_ang_vel_range = (-0.1, 0.1)
    cmd_cfg.prob_back = 0.10
    cmd_cfg.back_lin_vel_range = (-0.7, -0.4)
    cmd_cfg.back_ang_vel_range = (-0.1, 0.1)


def _apply_v3_motion_selection(env_cfg) -> None:
    """把参考动作的「选择清单」换成 v3 的精選 7 clip。

    ⚠️ **必须 deepcopy**：`motion_reference.motion_buffers` 是 dict，而
    `@configclass` 的默认值在多实例间可能共享 → 直接改会**连带改到 v1/v2 的配置**。
    """
    env_cfg.scene.motion_reference = copy.deepcopy(env_cfg.scene.motion_reference)
    env_cfg.scene.motion_reference.motion_buffers[
        MOTION_NAME
    ].filtered_motion_selection_filepath = AMP_WALK_V3_MOTION_SELECTION_FILE


@configclass
class G1AmpWalkV3CommandsCfg(G1AmpWalkCommandsCfg):
    """v3 的指令配置：继承 v1 的类，只改分档概率与范围（见 `apply_amp_walk_v3_command_ranges`）。"""

    def __post_init__(self):
        super().__post_init__()
        apply_amp_walk_v3_command_ranges(self.base_velocity)

#: v3 相对 v2 的改动（唯一的事实来源，供日志/文档引用）
AMP_WALK_V3_CHANGES: tuple[tuple[str, float, float], ...] = (
    ("rewards.action_rate_l2", -0.01, -0.02),
    ("rewards.flat_orientation_l2", float("nan"), -1.0),  # 从「删除」恢复
    ("rewards.joint_deviation_torso", float("nan"), -0.1),  # 从「删除」恢复
    ("policy.init_noise_std", 1.0, 0.3),
    ("algorithm.clip_min_std", 0.05, 0.015),
)


@configclass
class G1AmpWalkV3RewardsCfg(G1AmpWalkV2RewardsCfg):
    """v2 的 12 项配方 → **14 项**，共 4 处改动：

    1. `action_rate_l2`：−0.01 → **−0.02**（补回密集梯度，抑制 σ 膨大）；
    2. 从「删除」恢复 `flat_orientation_l2`：weight **−1.0**（基类原值）；
    3. 从「删除」恢复 `joint_deviation_torso`：weight **−0.1**（覆盖 `waist_.*`）；
    4. 其余 11 项（含 v2 的 `dof_pos_limits` −10.0 / `dof_acc_l2` −2.5e-7）原样保留。

    `func` 逐字沿用基类，只动 `weight`；新增的两项连 `params` 也逐字沿用基类。
    动机见模块 docstring。
    """

    def __post_init__(self):
        super().__post_init__()
        # 补回密集梯度：v1 用 −0.05 时 σ 稳在 0.756；v2 削到 −0.01 后 σ 涨到 1.580。
        # 取中间值 −0.02，兼顾「足够密集的信号」与「不过度惩罚动作变化」。
        self.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)

        # ⭐ 恢复 `flat_orientation_l2`（v2 删掉的 9 项之一）。
        #   用户实测：初次训练后模型**姿态歪斜** → 难以平衡、速度无法跟踪
        #   （略改变姿态就棓倒）。该项惩罚躯干/骨盆非水平，鼓励直立。
        #   ⚠️ 它读的是 **`torso_link`（= 关节树根）** 的 projected_gravity（⚠️ 2026-10-08 勘误：原文误写 root/pelvis）：
        #   如果歪斜来自腰部关节（waist_pitch/roll 行程只有 ±0.520 rad），
        #   本项可能盖不住，那种情况要靠 `joint_deviation_torso` 或
        #   躯干专用的 tilt 项（AMP_mjlab 有 `torso_posture`，但默认是注释掉的）。
        #   ⚠️ 必须在 super() 之后写：v2 的 `__post_init__` 会把它置成 None。
        self.flat_orientation_l2 = RewTerm(
            func=mdp.flat_orientation_l2, weight=AMP_WALK_V3_FLAT_ORI_WEIGHT
        )

        # ⭐ 同时恢复 `joint_deviation_torso`（−0.1，覆盖 `waist_.*`）。
        #   理由：`flat_orientation_l2` 读的是 **`torso_link`** 的 projected gravity（同上勘误），
        #   而 `waist_pitch` / `waist_roll` 的行程只有 **±0.520 rad**（除踝外最挤的两个）
        #   —— 腰部歪斜不会体现在骨盆倾角上，必须靠这一项。
        #   另一个作用：它也是「腰关节漂到限位」的唯一专项刹车。
        #   基类定义逐字沿用（func / weight / params 完全一致）。
        self.joint_deviation_torso = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*")},
        )


@configclass
class G1AmpWalkV3EnvCfg(G1AmpWalkV2EnvCfg):
    """训练用：场景 / 观测 / 事件 / 终止 / 时序 / 动作 **全部继承 v2**。

    本类替换三样：
    1. `rewards` —— v2 的 12 项配方 → **14 项**（`action_rate_l2` 提到 −0.02，
       并恢复 `flat_orientation_l2` −1.0 与 `joint_deviation_torso` −0.1）；
    2. `commands` —— 指令分布换成 AMP_mjlab v4 的（含精选 7-clip 数据集）；
    3. 参考动作的选择清单 —— 17 clip → 精选 7 clip（在 `__post_init__` 里 deepcopy 后改）。
    """

    rewards: G1AmpWalkV3RewardsCfg = G1AmpWalkV3RewardsCfg()
    commands: G1AmpWalkV3CommandsCfg = G1AmpWalkV3CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        # ⚠️ 父类 G1AmpWalkEnvCfg.__post_init__ 会把 ranges 写回 v1 的值，
        #    所以必须在它之后再写一次 v3 的。
        apply_amp_walk_v3_command_ranges(self.commands.base_velocity)
        _apply_v3_motion_selection(self)
        self.run_name = AMP_WALK_V3_RUN_NAME


@configclass
class G1AmpWalkV3EnvCfg_PLAY(G1AmpWalkV2EnvCfg_PLAY):
    """Play 用：继承 v2 PLAY 的全部差异，同样替换 rewards / commands / 数据集。"""

    rewards: G1AmpWalkV3RewardsCfg = G1AmpWalkV3RewardsCfg()
    commands: G1AmpWalkV3CommandsCfg = G1AmpWalkV3CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_amp_walk_v3_command_ranges(self.commands.base_velocity)
        _apply_v3_motion_selection(self)
        self.run_name = AMP_WALK_V3_RUN_NAME


class G1AmpWalkV3TeacherEnv(G1AmpWalkTeacherEnv):
    """v3 的契约校验器：只把「指令范围」与「指令保持时长」的**期望值**改成 v3 的。

    其余全部与 v1 逐字相同：动作空间 / joint 顺序 / `scale` / 时序 / 执行器 /
    default offset / 观测 626 / 高程图 527 射线 / 站立比 / 不允许课程。

    ⚠️ 为什么必须有这个子类：基类把许多期望值放在类属性里（所以子类只需覆盖元组），
    但「指令保持时长」原先**写死在方法体里**（`omni_loco_env_cfg.py` 的
    `_validate_teacher_contract`）。已把它改成类属性 `expected_command_hold`
    （默认值不变，对 OmniLoco/v1/v2 行为完全不变），这里就能正常声明 v3 的值。
    """

    expected_command_ranges = AMP_WALK_V3_COMMAND_RANGES
    expected_command_hold = (6.0, 15.0)


__all__ = [
    "AMP_WALK_V2_RUN_NAME",
    "AMP_WALK_V3_CHANGES",
    "AMP_WALK_V3_COMMAND_RANGES",
    "AMP_WALK_V3_FLAT_ORI_WEIGHT",
    "AMP_WALK_V3_MOTION_SELECTION_FILE",
    "AMP_WALK_V3_RUN_NAME",
    "G1AmpWalkV3CommandsCfg",
    "G1AmpWalkV3EnvCfg",
    "G1AmpWalkV3EnvCfg_PLAY",
    "G1AmpWalkV3RewardsCfg",
    "G1AmpWalkV3TeacherEnv",
    "apply_amp_walk_v3_command_ranges",
]
