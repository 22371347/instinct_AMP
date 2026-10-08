"""V10：**一次把 4 个概念变量全上**（放弃归因，目标是尽快拿到可参考的行走策略）。

| # | 变量 | V9j（参照） | **V10** | 依据 |
|---|---|---|---|---|
| ① | **探索量级** | σ 0.3 / 上限 0.4 / ent 0.0025 | **σ 0.6 / 上限 0.7 / ent 0.005**（`clip_min 0.03`） | 仓库里不含 AMP 的行走配方是 σ 1.0 / ent 0.008；用户实测 σ=1 那版"略有走路模态"。有界区间（0.3→0.7）避免 v8 那种 σ 失控到 1.04 |
| ② | **数据层镜像增广** | OFF | **ON** | 直接消除"镜像态从未被访问"：判别器的 expert 分布变成**左右对称**⇒ 单侧步态不再"像人"。现成代码路径 + 用**本仓库已验证的镜像算子**现算映射 |
| ③ | **指令扩展** | 前向 (0,0.7)/0/0 | **(0,0.7) × (−0.35,0.35) × (−0.6,0.6)** | 左前/右前对角、转向都要练 ⇒ 任务本身要求左右两侧都会用。取值在数据集包络内（vy∈[−0.48,0.55]、wz∈[−44°,57°]） |
| ④ | **交替支撑约束** | `feet_swing_time_penalty −0.5` | 同（21 项） | 堵住"单脚吊腿"这条近路（V9h@4770 观察到的退化解） |

⚠️ **本版不做单变量归因** —— 4 个变量同时动，好处是"明天就能有一个可参考的行走策略"，
代价是若失败无法直接定位。为此本版把**诊断键**留全了，失败时仍能分头看：

- `Mean step feet_swing_excess`（单脚超长腾空秒数）⇒ ②④ 是否把吊腿堵住
- `Mean action noise std` ⇒ ① 的 σ 是否停在 0.6 附近（而不是被 0.7 卡死）
- `Episode_Reward/dof_pos_limits` ⇒ ① 的限位代价（V9h 仅 −0.0207/s，有富余）
- `Mean step amp_d` / `amp_d_ref` ⇒ ② 是否让判别器更"拿策略当外人"（gap 变大是预期的）
- `vx_mid/high RMSE`（现在含侧向/转向指令的行进能力）

## ② 的实现细节（**标签正确性**是关键）

- 关节映射**现算**：`build_joint_mirror(OMNI_LOCO_JOINT_ORDER)` —— 与策略层镜像损失
  **同一套已验证代码**（本地 46/46 PASS），不使用 `unitree_g1.py` 里那两个现成常量
  （它们的索引口径未经本仓库核对，用错会静默产生错误标签）。
- link 映射：12 个左右 link 按名字配对 + pelvis/torso 自映射，可逐项核对（见常量表）。
- 启动时会打印 `[AMP-WALK INFO] 数据层镜像增广=ON；joint 顺序 vs OMNI_LOCO_JOINT_ORDER：一致`
  —— 这一行是**增广标签是否可信的唯一在线证据**，必须看。
- 每个 env 每段参考动作随机决定"是否镜像"（`amass_motion.reset` 里的 `randint`）。

## 判读（≥2000 轮看包络）

| 判据 | 期望 |
|---|---|
| `Episode_Reward/feet_air_time` | **> 0.1 /s**（真交替支撑；V9h 0.041） |
| `Mean step feet_swing_excess` | ≈ 0（脚能放下来） |
| `vx_mid` / `vx_high` RMSE | 继续降（V9h 0.155 / 0.2185） |
| `Episode_Termination/base_contact` | < 0.45（V9h 0.449） |
| `Mean action noise std` | 停在 0.5~0.7 之间（未被上限卡死） |
"""

from __future__ import annotations

from isaaclab.utils import configclass
from instinctlab.utils.mirror import build_joint_mirror

from .amp_walk_v9_env_cfg import G1AmpWalkV9TeacherEnv
from .amp_walk_v9j_env_cfg import AMP_WALK_V9J_RUN_NAME, G1AmpWalkV9jEnvCfg, G1AmpWalkV9jEnvCfg_PLAY
from .omni_loco_env_cfg import OMNI_LOCO_JOINT_ORDER

AMP_WALK_V10_RUN_NAME = "G1AmpWalkV10MVPPlane_obs378_rew21_explore_aug_cmdLT"

#: 速度指令范围：**前向 + 侧向(±) + 转向(±)** ⇒ 左前/右前对角 + 转向都能练。
#: 数据集实测包络 vx∈[−1.07,1.69]、vy∈[−0.48,0.55]、wz∈[−44.2°,57.0°] ⇒ 取值在包络内
#: （超出包络的档位缺乏参考密度，判别器会抑制对应姿态）。
AMP_WALK_V10_COMMAND_RANGES = ((0.0, 0.7), (-0.35, 0.35), (-0.6, 0.6))
#: 低速档里"精确 (0,0,0)"的比例（与 v9e 起一致）
AMP_WALK_V10_ZERO_FRAC_IN_LOW = 0.3
#: `motion_reference_cfg.link_of_interests` 的镜像映射（**顺序必须与那份列表逐项对应**）：
#: pelvis/torso 自映射，其余 6 对左右 link 互换。
AMP_WALK_V10_LINK_MIRROR = (
    0,
    1,   # pelvis, torso_link
    3,
    2,   # left/right_shoulder_roll_link
    5,
    4,   # left/right_elbow_link
    7,
    6,   # left/right_wrist_roll_link
    9,
    8,   # left/right_hip_roll_link
    11,
    10,  # left/right_knee_link
    13,
    12,  # left/right_ankle_roll_link
)


def _apply_v10(cfg) -> None:
    """把①②③三处改动写回配置（奖励集由父类 V9j 提供 ⇒ ④ 已在内）。"""
    # ③ 指令：前向 + 侧向(±) + 转向(±)
    cmd = cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = AMP_WALK_V10_COMMAND_RANGES[0]
    cmd.ranges.lin_vel_y = AMP_WALK_V10_COMMAND_RANGES[1]
    cmd.ranges.ang_vel_z = AMP_WALK_V10_COMMAND_RANGES[2]
    cmd.zero_frac_in_low = AMP_WALK_V10_ZERO_FRAC_IN_LOW

    # ② 数据层镜像增广：映射从**本仓库已验证的镜像算子**现算
    perm, sign = build_joint_mirror(list(OMNI_LOCO_JOINT_ORDER))
    ref = cfg.scene.motion_reference
    ref.symmetric_augmentation_joint_mapping = [int(i) for i in perm.tolist()]
    ref.symmetric_augmentation_joint_reverse_buf = [float(s) for s in sign.tolist()]
    ref.symmetric_augmentation_link_mapping = list(AMP_WALK_V10_LINK_MIRROR)


@configclass
class G1AmpWalkV10EnvCfg(G1AmpWalkV9jEnvCfg):
    """训练用：V9j（21 项奖励）+ 探索量级 + 数据层镜像增广 + 指令扩展。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_v10(self)
        self.run_name = AMP_WALK_V10_RUN_NAME


@configclass
class G1AmpWalkV10EnvCfg_PLAY(G1AmpWalkV9jEnvCfg_PLAY):
    """回放用：与训练同奖励/同指令分布/同增广设置。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_v10(self)
        self.run_name = AMP_WALK_V10_RUN_NAME


class G1AmpWalkV10TeacherEnv(G1AmpWalkV9TeacherEnv):
    """⚠️ **契期望值的唯一有效位置是 env 类**（不是 cfg 类）。

    `omni_loco_env_cfg.py::_validate_teacher_contract` 比较的是
    `self.expected_command_ranges` —— 那是 **env** 的类属性；写在 `@configclass` 的 cfg 上
    完全不会生效（V10 首次启动就是因此被契约拦住：
    `command ranges ((0.0,0.7),(-0.35,0.35),(-0.6,0.6)) != ((0.0,0.7),(0.0,0.0),(0.0,0.0))`）。
    """

    expected_command_ranges = AMP_WALK_V10_COMMAND_RANGES


__all__ = [
    "AMP_WALK_V10_COMMAND_RANGES",
    "AMP_WALK_V10_LINK_MIRROR",
    "AMP_WALK_V10_RUN_NAME",
    "AMP_WALK_V10_ZERO_FRAC_IN_LOW",
    "G1AmpWalkV10EnvCfg",
    "G1AmpWalkV10EnvCfg_PLAY",
    "G1AmpWalkV10TeacherEnv",
]
