"""AMP 行走任务（`Instinct-Amp-Walk-G1-v0`）专用的新奖励项 —— 自 `AMP_mjlab` 迁移。

作用域
------
本文件**只服务 AMP 行走这一个新任务**。仓库里既有的任何奖励函数与其权重
（`tasks/locomotion/mdp/rewards.py`、`G1FlatRewardsCfg` 等）**一行都不改**，
OmniLoco / Vault / flat 三条已训好的链路完全不受影响。

迁移对照（`AMP_mjlab` v5 → 本文件）
-----------------------------------
| AMP_mjlab 项 | 权重 | 本文件实现 | 说明 |
|---|---|---|---|
| `track_anchor_linear_velocity` | 1.5 | **不新建**，直接复用仓库既有 `track_lin_vel_xy_yaw_frame_exp` 并改权重 | 语义等价（误差范数在偏航旋转下不变） |
| `track_anchor_angular_velocity` | 1.2 | **不新建**，复用既有 `track_ang_vel_z_world_exp` | 同上 |
| `track_root_height` | 1.5 | `track_base_height_exp` | 仓库里 `base_height_l2` 被注释掉了，需新写 |
| `body_ang_vel_xy_l2` | 0.7 | `track_root_ang_vel_xy_exp` | ⚠️ AMP_mjlab 那个名字带 `l2`、**其实是 exp 正奖励** |
| `track_default_pose_on_stand` | −1.0 | `track_default_pose_on_stand` | L1 + 平滑门控 `gate_tau` |
| `feet_lateral_width` | −10.0 | `feet_lateral_width_cost` | `sep_floor` 需在本仓库重新标定 |
| `feet_crossing` | −20.0 | `feet_crossing_cost` | 同上；**额外加了方向门控**，见下 |
| `foot_slip` | −0.25 | **不新建**，复用既有 `instinct_mdp.contact_slide`（`feet_slide`）并改权重 | 实现不同但同族 |
| `self_collisions` | −0.1 | ❌ **不迁移** | 见下 |
| `torso_posture` / `feet_swing_clearance` / `soft_landing` | 已注释 | 不迁移 | AMP_mjlab 自己也没用 |

`self_collisions` 为什么不迁移
------------------------------
IsaacLab 的 `ContactSensorCfg.filter_prim_paths_expr` 在 PhysX 后端**只支持一对多**：

    .. attention::
        Filtered contact reporting only works when ``prim_path`` matches a single
        primitive per environment.

即 `prim_path="{ENV_REGEX_NS}/Robot/.*"` 配 `filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"]`
这种「多对多」自碰撞检测**在 PhysX 上不可用**。而且本任务的兜底已经比 AMP_mjlab 的
`-0.1` 更强：资产开了 `enabled_self_collisions=True`，`G1FlatTerminationsCfg.base_contact`
对 `torso_link` / 手臂 / 髋 / 膝的接触（阈值 1 N）直接 **−200 并终止**。

方向门控（对 AMP_mjlab 的**有意偏离**，有实测依据）
--------------------------------------------------
AMP_mjlab 的 `feet_crossing` **不乘**方向门控（他们的理由是「真交叉在任何方向都会绊脚」），
`feet_lateral_width` 则乘 `w_dir = |vx| / hypot(vx, vy)`（纯侧移时关掉）。

但本仓库参考动作的**侧移 clip 就是交叉式侧步（grapevine）**，实测（`tools/calibrate_foot_sep.py`）：

    直线行走 4 段：sep min = +0.1387，真交叉帧占比 = 0.000
    侧移 2 段    ：sep min = -0.1503，真交叉帧占比 = **0.210**   ← 参考动作本身的风格

所以对纯侧向指令无条件施加 −20 的交叉惩罚，会**直接和 AMP 风格先验对撞**（判别器说「像参考」，
奖励说「不许」），压制侧移能力 —— 正是 `PLAN_amp_walk.md` §3.4 记录的那个坑。
因此本文件让 `feet_crossing_cost` **与宽度项共用同一个方向门控**（默认开启）。

标定量（`tools/amp_walk_foot_sep_stats.json`）
--------------------------------------------
| 量 | 实测值 |
|---|---|
| 脚部参考点 | `LL_FOOT` / `LR_FOOT`（URDF 里挂在 `*_ankle_roll_link` 下、偏移 (0.04, 0, −0.037)） |
| 直线行走 sep | min 0.1387 / p1 0.1455 / 中位 0.2140 / max 0.4345（2186 帧） |
| **`sep_floor`** | **0.145**（= 直线行走 p1 向下取整；AMP_mjlab 在其模型上取 0.15） |
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

# ⭐ V23（2026-10-03）：进度门控（方向投影+EMA）与逐关节死区/内部权重 —— 纯 torch 小模块，
#   可在本地无 Isaac 单测（`instinct/tools/test_amp_reward_utils.py`）。
from . import amp_reward_utils as _amp_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

#: 脚部参考 body 候选（前者优先）。
#:
#: `LL_FOOT` / `LR_FOOT` 与 AMP_mjlab 的 `left_foot` / `right_foot` **site** 同源
#: （实测同一帧下两者世界坐标一致），且左右横向偏移为 0，所以退化成 ankle 时
#: 横向间距几乎不变 —— 兜底是安全的。
#: 左在 **+y**（URDF: `left_hip_pitch_joint` 的 origin y = +0.0645）。
_FOOT_BODY_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("LL_FOOT", "LR_FOOT"),
    ("left_ankle_roll_link", "right_ankle_roll_link"),
)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _foot_body_ids(asset) -> tuple[int, int]:
    """返回 (左脚 body 索引, 右脚 body 索引)。

    ⚠️ **按名字查，不依赖 `SceneEntityCfg.body_ids` 的顺序** —— IsaacLab 的
    `resolve_matching_names` 是否保序取决于 `SceneEntityCfg.preserve_order`
    （老版本没有这个字段），而左右搞反会让「交叉」判断完全反号。
    """
    names = list(asset.body_names)
    for left, right in _FOOT_BODY_CANDIDATES:
        if left in names and right in names:
            return names.index(left), names.index(right)
    raise ValueError(
        f"在 body 列表里找不到脚部参考 body，候选 {_FOOT_BODY_CANDIDATES} 均缺失；"
        f"实际 body_names={names}"
    )


def _resolved_joint_ids(asset_cfg: SceneEntityCfg, n_joints: int):
    """取（已解析的）关节索引；未解析时退化为「全部关节」。"""
    joint_ids = asset_cfg.joint_ids
    return slice(None) if joint_ids is None else joint_ids


def _selected_joint_indices(joint_ids, n_joints: int) -> list[int]:
    if isinstance(joint_ids, slice):
        return list(range(*joint_ids.indices(n_joints)))
    return [int(i) for i in torch.as_tensor(joint_ids).reshape(-1).tolist()]


def _selected_joint_names(asset, joint_ids) -> list[str]:
    """取已选关节的**名字**（顺序与 `asset.data.*[:, joint_ids]` 一致）。"""
    all_names = list(asset.joint_names)
    return [all_names[i] for i in _selected_joint_indices(joint_ids, len(all_names))]


def _is_leg_joint(n: str) -> bool:
    return any(k in n for k in ("hip_", "knee", "ankle"))


def _is_arm_joint(n: str) -> bool:
    return any(k in n for k in ("shoulder", "elbow", "wrist"))


def _is_waist_joint(n: str) -> bool:
    return "waist" in n


def _joint_weight_row(asset, joint_ids, weights: dict[str, float]) -> torch.Tensor:
    """按**关节名**构造 (n_sel,) 的逐关节权重，供 `track_default_pose_on_stand` 用。"""
    all_names = list(asset.joint_names)
    selected = _selected_joint_indices(joint_ids, len(all_names))
    row = torch.ones(len(selected), device=asset.device)
    col_of = {j: c for c, j in enumerate(selected)}
    for name, weight in weights.items():
        if name not in all_names:
            continue
        col = col_of.get(all_names.index(name))
        if col is not None:
            row[col] = float(weight)
    return row


def _walk_speed_gate(
    command: torch.Tensor,
    walk_max_lin_vel: float | None,
    gate_width: float = 0.2,
) -> torch.Tensor:
    """「行走速档」门控：只对**慢速前向**指令生效（慢跑/跑步豁免）。

    - `vx_cmd <= walk_max_lin_vel - gate_width` → 1（全量行走塑形）
    - `vx_cmd >= walk_max_lin_vel`              → 0（豁免）
    - 中间线性过渡（避免硬边界造成策略不连续）

    为什么需要它：实测慢跑 clip 摆动腿会合法内收（`jog_forward_loop_003__A021`
    的 sep 低到 0.040），按行走地板惩罚会压制跑步步态。
    """
    if walk_max_lin_vel is None or walk_max_lin_vel <= 0.0 or gate_width <= 0.0:
        return torch.ones_like(command[:, 0])
    return ((float(walk_max_lin_vel) - command[:, 0]) / float(gate_width)).clamp(0.0, 1.0)


def _lateral_dir_gate(command: torch.Tensor, lateral_eps: float = 1e-6) -> torch.Tensor:
    """方向门控 `w_dir = |vx| / hypot(vx, vy) ∈ [0, 1]`。

    纯前后 = 1、45° 斜向 ≈ 0.71、**纯侧移 = 0** → 保护侧向并步/交叉式侧步。
    （AMP_mjlab 原文用 `cos(vy/vx)`，本实现用更稳的 `|vx|/‖v‖`。）
    """
    vx, vy = command[:, 0], command[:, 1]
    return torch.abs(vx) / torch.sqrt(vx * vx + vy * vy + float(lateral_eps))


def _command_or_raise(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    if command is None:
        raise ValueError(f"找不到速度指令项 '{command_name}'（检查 cfg.commands 里的项名）")
    return command


def _log_metric(env: ManagerBasedRLEnv, key: str, value: torch.Tensor) -> None:
    """把诊断量写进 `extras["step"]`；runner 会以 `Step/<key>` 打到 TensorBoard，
    并在每轮迭代的控制台里打印 `Mean step <key>`。

    值必须是 0 维 tensor 或 python 标量（runner 会先 `unsqueeze(0)` 再 `cat`）。

    ⚠️ **为什么不用 `extras["log"]`（踩过的坑，务必别改回去）**

    IsaacLab 的 `ManagerBasedRLEnv._reset_idx`（`manager_based_rl_env.py`）里有
    `self.extras["log"] = dict()`，是**整体替换**；而 reward 项的 `compute()` 发生在
    `_reset_idx` **之前**。于是同一条日志通道出现两种 step：

    - 本 step 有 env 终止 → log 被换成只含 `Episode_Reward/*` 的新 dict，本 step 写的
      诊断量**被丢掉**；
    - 本 step 无 env 终止 → `extras` 是持久的（只在 `manager_based_env.py` 的
      `__init__` 里建过一次），上一 step 的 dict 还在，诊断量**侥幸留下**。

    两类 step 交替出现，`ep_infos` 里各条 dict 就**有的带 key、有的不带**；而 runner 的
    `OnPolicyRunner.log()` 是拿 `ep_infos[0]` 的 key 去索引**全部** dict：

        KeyError: 'Metrics/foot_sep_mean'

    改走 `extras["step"]`：`InstinctRlEnv.step()` 里只有
    `self.extras["step"] = self.extras.get("step", {}).update(...)` —— **始终是同一个
    dict 对象、永不替换**，因此 key 集合对所有 step 天然一致。
    """
    env.extras.setdefault("step", {})[key] = value


# ---------------------------------------------------------------------------
# 姿态 / 高度（对应 AMP_mjlab 的 track_root_height / body_ang_vel_xy_l2）
# ---------------------------------------------------------------------------


def track_base_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """**奖励** root（本仓库 = `torso_link`）高度贴近目标高度（负值取反 → 正权重）。

    `target_height=None` 时取 `default_root_state[:, 2]`（= `init_state.pos_z` = 0.82 m），
    与 AMP_mjlab 的 `track_root_height` 一致。

    Args:
        std: 高度误差的高斯尺度（m）。AMP_mjlab 用 0.3。
    """
    asset = env.scene[asset_cfg.name]
    root_height = asset.data.root_pos_w[:, 2]
    if target_height is None:
        target = asset.data.default_root_state[:, 2]
    else:
        target = torch.full_like(root_height, float(target_height))
    return torch.exp(-torch.square(target - root_height) / std**2)


#: T5（2026-10-10）默认的身体与目标（见 `track_pelvis_height_exp` 的 docstring）
T5_DEFAULT_BODY = "pelvis"


def track_pelvis_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: float | None = None,
    one_sided: bool = False,
    body_name: str = T5_DEFAULT_BODY,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """**奖励** 指定刚体（默认 `pelvis`）的高度贴近目标（`exp(-e²/std²)`，正权重）。

    与 `track_base_height_exp` 的**唯一**区别：被测量的体从 **root（= `torso_link`）**
    换成 **`pelvis`** —— 即把 mjlab 的 `track_root_height` 放回它本来的语义
    （mjlab 的 root 就是 `pelvis`；我们 root 是 `torso_link`，所以那一项量的是"躯干高度"）。

    为什么必须换（2026-10-10 回放取证，`analyze_gait_trace.py`）
    -------------------------------------------------------------
    在 T3@12500 的同一条真实轨迹（vx=+0.8）上实测：
      * 原项（**躯干**高度，靶 0.82）得分 **0.999**，姿态力只有 **0.0035 reward/s**
        （= 12 项总幅度 4.26/s 的 **0.08%**）；改靶到 0.8025 后 12600 轮姿态也**毫无变化**；
      * 更关键：**躯干高度可以用腰换**（MJCF FK：腰前倾 0.2 rad ⇒ 躯干矮 **1.5 cm**）
        ⇒ 它**不约束腿的伸展**，反而给"扭腰换高度"开了口子（实测策略：躯干世界系仅
        **+0.81°**（钉在竖直），而参考是 **+10.10°**；骨盆则后仰 −8.9° 去"补偿"）；
      * **骨盆高度只由腿决定**（骨盆是两条腿的根，它的 z 由髋/膝/踝的伸展量决定）
        ⇒ 换成骨盆后，这一项第一次变成"**约束腿的伸展量**"，正是 mjlab 要的"别塌"语义。

    目标取值（`target_height`，m）
    -----------------------------
    | 候选 | 值 | 说明 |
    |---|---|---|
    | mjlab 原值（root=pelvis） | **0.78** | `KNEES_BENT_KEYFRAME.pos_z`（关节位姿与我们逐字相同）|
    | 我们资产的"生成高度等价" | 0.776 | = 0.82 − 0.044（torso−pelvis 偏移，MJCF FK 实测）|
    | ⭐ T5 采用：**FK 一致的站姿高度** | **0.7585** | default 站姿 + 脚掌着地时的骨盆离地高度 |

    ⚠️ 为什么不用 0.78：它比"脚掌着地时的站姿"高 **2.15 cm** —— 和我们踩过的躯干靶
    0.82（比站姿高 1.75 cm）**同款"偏高"**，会持续逼腿**伸展**，正是 T5 要避免的
    "腿伸直 → 奇异点 → 动作受限"。取 0.7585 ⇒ 该项成为"**别蹲、也别过伸**"的**对称**约束，
    伸展方向**零压力**。

    Args:
        std: 高度误差的高斯尺度（m）。mjlab 用 0.3。
        target_height: 目标高度（m）。`None` ⇒ 退回 `default_root_state[:, 2]`
            （⚠️ 那是**躯干**生成高度，对骨盆**不正确**，只作兜底，T5 必须显式给值）。
        one_sided: `True` ⇒ 只罚"低于靶"（`relu(target − z)`），完全放开向上。
            T5 默认 `False`（= mjlab 的对称口径）；若实测出现蹲走，这是第一根杠杆。
        body_name: 默认 `pelvis`。**找不到就 raise**（不许静默退回 root —— 那会变成
            静默地继续量躯干，是最危险的失效模式）。

    探针：`Step/pelvis_height`（实测骨盆高度，用来验证靶是否自洽）。
    """
    asset = env.scene[asset_cfg.name]
    ids, names = asset.find_bodies([body_name], preserve_order=True)
    if len(ids) != 1:
        raise ValueError(
            f"[T5] 在 {asset_cfg.name!r} 上找不到唯一刚体 {body_name!r}（命中 {list(names)}）"
        )
    h = asset.data.body_pos_w[:, ids[0], 2]
    if target_height is None:
        target = asset.data.default_root_state[:, 2]
    else:
        target = torch.full_like(h, float(target_height))
    err = target - h
    if one_sided:
        err = torch.clamp(err, min=0.0)
    _log_metric(env, "pelvis_height", h.mean())
    return torch.exp(-torch.square(err) / std**2)


def track_root_ang_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """**奖励** root 的横滚/俯仰角速度尽量小（`exp(-‖ω_xy^b‖²/std²)`，正权重）。

    对应 AMP_mjlab 的 `body_ang_vel_xy_l2` —— 注意它名字带 `l2` 但**返回的是正奖励**，
    且当时的 body 就是它的 root（`pelvis`）。本仓库 root = `torso_link`，
    因此直接取 `root_ang_vel_b[:, :2]`（body frame），与「root body 的 ω_xy」完全等价，
    不需要查 body 索引。

    与 `flat_orientation_l2`（罚静态倾角）互补：这条管**动态**摆动。
    """
    asset = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.root_ang_vel_b[:, :2]
    return torch.exp(-torch.sum(torch.square(ang_vel_xy), dim=-1) / std**2)


# ---------------------------------------------------------------------------
# ⭐⭐ T2（2026-10-09）：照抄 AMP_mjlab 的速度/角速度跟踪**形式**
#     `AMP_mjlab/src/tasks/amp_loco/mdp/rewards.py:71-130`
# ---------------------------------------------------------------------------


def track_anchor_lin_vel_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """**奖励**：根速度与指令的**三维世界系**误差（`exp(-‖R_yaw·(vx,vy,0) − v_w‖²/std²)`）。

    与仓库既有的 `track_lin_vel_xy_yaw_frame_exp` 的**唯一**区别：本项把**竖直速度 `v_z`
    也算进误差**（指令的 z 分量恒为 0）。

    为什么 T2 要专门补这一项（逐行核对 AMP_mjlab 源码）
    ----------------------------------------------------
    `AMP_mjlab/src/tasks/amp_loco/mdp/rewards.py:87-95`::

        command_xyz_b = torch.cat((command[:, :2], zeros_like(command[:, :1])), dim=-1)
        command_xyz_w = quat_apply(yaw_quat(body_link_quat_w[:, anchor]), command_xyz_b)
        lin_vel_error = torch.sum(torch.square(
            command_xyz_w[:, :3] - body_link_lin_vel_w[:, anchor, :3]), dim=1)

    ⇒ 误差是**三维**的 ⇒ mjlab 的这一项本来就顺带承担了「别上下弹」的职责，
    **这正是 mjlab 奖励集里没有 `lin_vel_z_l2` 的原因**。若沿用 xy-only 版本、
    又删掉 `lin_vel_z_l2`，就等于**丢掉了全部竖直速度抑制**（“小跳模态”的直接燃料）。

    `std` 照抄 mjlab walk 的 **1.0**（我们历史值 0.5 更严格 ⇒ 等效灵敏度 4 倍）。
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    if command is None:
        raise ValueError(f"[T2] 找不到指令项 {command_name!r}")

    cmd_b = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=-1)
    cmd_w = quat_apply(yaw_quat(asset.data.root_quat_w), cmd_b)
    err = torch.sum(torch.square(cmd_w - asset.data.root_lin_vel_w[:, :3]), dim=-1)
    return torch.exp(-err / std**2)


def track_anchor_ang_vel_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """**奖励**：`exp(-[(ωz_cmd − ωz_w)² + ‖ω_xy^b‖²] / std²)`。

    照抄 `AMP_mjlab/src/tasks/amp_loco/mdp/rewards.py:98-128`：偏航误差用**世界系** `ωz`，
    横滚/俯仰用**机体系** `ω_xy`，两者相加后同一个 `exp`。
    ⚠️ 与既有的 `track_ang_vel_z_world_exp` 的区别：后者**不含** `‖ω_xy^b‖²` 那一项。
    `std` 照抄 mjlab walk 的 **3.14**（我们历史值 0.5 ⇒ 等效严格 ~40 倍）。
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    if command is None:
        raise ValueError(f"[T2] 找不到指令项 {command_name!r}")

    wz_err = torch.square(command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    wxy = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=-1)
    return torch.exp(-(wz_err + wxy) / std**2)


def track_default_pose_on_stand(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    pos_scale: float = 4.0,
    vel_scale: float = 0.5,
    gate_tau: float = 0.25,
    pos_joint_weights: dict[str, float] | None = None,
    # --- ⭐ V23（2026-10-03）：逐关节死区（未列出的关节用 `*_tol_default`）--------
    pos_joint_tol: dict[str, float] | None = None,
    pos_tol_default: float = 0.0,
    vel_joint_tol: dict[str, float] | None = None,
    vel_tol_default: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """**惩罚**（负权重）：指令速度越低越把关节拉回 default 站姿 + 抑制关节晃动。

        cost = w · (pos_scale · mean(|q − q_default|·k) + vel_scale · mean(|dq|))

    用 **L1** 而不是 L2/exp：梯度线性、不饱和，回位更彻底（AMP_mjlab v3.4 的结论）。

    门控（`gate_tau`）：
    - `gate_tau > 0`：平滑 `w = exp(−cmd_speed / gate_tau)`，指令 → 0 时 w → 1，
      指令变大指数衰减 → 策略学会「指令结束时**边减速边回位**」。
    - `gate_tau <= 0`：二元门控 `w = 1[cmd_speed ≤ command_threshold]`（回退行为）。

    `pos_joint_weights` 按关节名放大某些关节的位姿误差（如 `waist_yaw_joint: 3.0`
    防止腰部锁在 90° 的假吸引子）。

    ## ⭐ V23（2026-10-03）：逐关节死区 `tol` + 内部权重改为按 `Σw_j` 归一

    两处改动（都有实测依据，见 `RESULTS_v19cd_v20_stand_probes.md`）：

    1. **逐关节死区**：`relu(|e_j| − tol_j)` —— 站立时 ±3° 的自然微调不罚，
       15~30° 的"抬脚级"偏离照罚。把"微调"与"抬脚"分开。
       `default_joint_pos` **本身就是站姿**（膝 0.669 rad ≈ 38°）⇒ 0 指令下的
       正确姿势就是贴近 default，所以容差取"几度"级别（臂/腰 0.05 rad）。
    2. **归一化改为 `Σ_j w_j·relu(e_j−tol_j) / Σ_j w_j`**（旧版是 `mean(...)`）：
       ⇒ 权重退化为"优先级"，**不再缩放整项**（旧版把 29 关节均值乘以 w，等于整项
       放大 —— 历史上已验证"权重不是杠杆"）。
       ⚠️ 由此带来的副作用：同一 `pos_joint_weights` 下本项**数值会小 ~6%**（分母由
       29 变为 `Σw_j`）；旧 run 的数值不可直接与本项逐字对比。

    为什么要按关节分组给权重（实测，V19d，全步 RMSE vs default）：
    臂 14 关节贡献 5.66（**58%**）、腿 12 关节 3.37（35%）、腰 0.57（6%）——
    即这一项一直是"主要在管手臂"。内部权重及分组探针就是为了让腿/腰/臂的定价
    可分别观测、可分别调。

    本任务的速度指令里有 `prob_low` 档会给**精确 (0,0,0)** 指令，
    因此 `cmd_speed = 0` → `w = 1`，与 AMP_mjlab 的门控语义天然对齐。
    """
    asset = env.scene[asset_cfg.name]
    command = _command_or_raise(env, command_name)

    cmd_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    if gate_tau > 0.0:
        active = torch.exp(-cmd_speed / float(gate_tau))
    else:
        active = (cmd_speed <= command_threshold).float()

    joint_ids = _resolved_joint_ids(asset_cfg, len(asset.joint_names))
    sel_names = _selected_joint_names(asset, joint_ids)
    abs_err = torch.abs(asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids])
    abs_vel = torch.abs(asset.data.joint_vel[:, joint_ids])
    # ⭐ V24：高频诊断用（加速度，未乘权重）
    abs_acc = torch.abs(asset.data.joint_acc[:, joint_ids])

    # ⭐ V23：逐关节死区 + 内部权重（分母 Σw_j ⇒ 权重只表达"优先级"，不再缩放整项）
    w_row = _amp_utils.name_weight_row(sel_names, pos_joint_weights, 1.0).to(abs_err.device)
    pos_tol_row = _amp_utils.name_weight_row(sel_names, pos_joint_tol, pos_tol_default).to(abs_err.device)
    vel_tol_row = _amp_utils.name_weight_row(sel_names, vel_joint_tol, vel_tol_default).to(abs_err.device)
    pos_err = _amp_utils.weighted_deadband_mean(abs_err, w_row, pos_tol_row)
    vel_err = _amp_utils.weighted_deadband_mean(abs_vel, w_row, vel_tol_row)

    # 探针：保留 V19d 的旧口径（便于跨版本对比）+ V23 新口径与分组（回答"谁在贡献"）
    _log_metric(env, "stand_pose_pos_err", abs_err.mean())        # 旧口径：未加权、无死区
    _log_metric(env, "stand_pose_vel_err", abs_vel.mean())        # 旧口径
    _log_metric(env, "stand_pose_active", active.mean())
    _log_metric(env, "stand_pose_pos_cost", pos_err.mean())       # ⭐ 新口径：死区+加权
    _log_metric(env, "stand_pose_vel_cost", vel_err.mean())       # ⭐
    _log_metric(env, "stand_pose_pos_cost_active", (pos_err * active).mean())  # 各 2.5 节账目用
    _log_metric(env, "stand_pose_vel_cost_active", (vel_err * active).mean())
    for grp, pred in (("legs", _is_leg_joint), ("arms", _is_arm_joint), ("waist", _is_waist_joint)):
        gm = torch.tensor([pred(n) for n in sel_names], device=abs_err.device)
        if bool(gm.any()):
            _log_metric(env, f"stand_pose_pos_err_{grp}", abs_err[:, gm].mean())
            _log_metric(env, f"stand_pose_vel_err_{grp}", abs_vel[:, gm].mean())
            # ⭐ V24：**高频诊断** —— 未乘权重的 `mean|ddq|`（加速度对频率是 f2 ⇒
            #    高频小跳在加速度上会明显放大）。只写探针，不改任何罚项量级。
            _log_metric(env, f"hf_acc_{grp}", abs_acc[:, gm].mean())
    _log_metric(env, "hf_acc_all", abs_acc.mean())
    return (pos_scale * pos_err + vel_scale * vel_err) * active


# ---------------------------------------------------------------------------
# 步宽 / 交叉（对应 AMP_mjlab 的 feet_lateral_width / feet_crossing）
# ---------------------------------------------------------------------------


def feet_lateral_sep(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """两脚在 **root 偏航系** 下的横向间距：`sep = y_left − y_right`（m）。

    与 `instinct/tools/calibrate_foot_sep.py` 的标定量**完全一致**（同样的
    `LL_FOOT`/`LR_FOOT`、同样的 `yaw_quat(root_quat_w)`），因此配置里的
    `sep_floor` 可以直接照搬标定结果。

    物理含义：自然站姿 / 健康步态 > 0（直线行走实测 min 0.1387、中位 0.2140）；
    收得越拢越小；真交叉时 < 0。
    """
    asset = env.scene[asset_cfg.name]
    left_id, right_id = _foot_body_ids(asset)
    foot_pos_w = asset.data.body_pos_w[:, [left_id, right_id], :]      # (N, 2, 3)
    delta_w = foot_pos_w[:, 0] - foot_pos_w[:, 1]                      # world 下 left − right
    heading = yaw_quat(asset.data.root_quat_w)                         # 只保留偏航
    return quat_apply_inverse(heading, delta_w)[:, 1]


def feet_lateral_width_cost(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    sep_floor: float = 0.145,
    gate_lateral: bool = True,
    lateral_eps: float = 1e-6,
    max_cost: float | None = 0.10,
    walk_max_lin_vel: float | None = 1.0,
    walk_gate_width: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """**惩罚**（负权重）两脚横向脚距低于 `sep_floor` 的「窄步」：

        cost = clamp(sep_floor − sep, 0, max_cost) · w_act · w_dir · w_speed

    - 只作用于**横向分量**，不影响前后交替步态；
    - `sep_floor` 取在数据集健康步态的窄尾之下（本仓库实测 p1 = 0.1455）→
      健康步态几乎零惩罚，窄步获得强梯度；
    - `gate_lateral` → `w_dir = |vx|/‖v‖`：纯侧移 = 0，保护侧向并步（实测侧移
      clip 有 21% 帧 sep < 0，属参考风格）；
    - `walk_max_lin_vel` → 慢跑/跑步豁免（摆动腿合法内收）；
    - `w_act`：指令速度 > `command_threshold` 才激活（静止交给 `track_default_pose_on_stand`）。
    """
    command = _command_or_raise(env, command_name)
    w_act = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > command_threshold).float()
    w_dir = _lateral_dir_gate(command, lateral_eps) if gate_lateral else torch.ones_like(command[:, 0])

    sep = feet_lateral_sep(env, asset_cfg)
    raw = torch.clamp(sep_floor - sep, min=0.0)
    if max_cost is not None:
        raw = raw.clamp(max=float(max_cost))

    # 诊断量：真机值应落在数据集直线走区间（中位 0.2140、p1 0.1455）附近。
    # 这条曲线同时是「脚部参考 body 退路是否选对」的**在线自检**：若误用错 body
    # （例如把 root 系当脚），这个数会明显偏离 0.1~0.3。
    _log_metric(env, "foot_sep_mean", sep.mean())
    return raw * w_act * w_dir * _walk_speed_gate(command, walk_max_lin_vel, walk_gate_width)


def feet_crossing_cost(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    gate_lateral: bool = True,
    lateral_eps: float = 1e-6,
    max_cost: float | None = 0.10,
    walk_max_lin_vel: float | None = 1.0,
    walk_gate_width: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """**惩罚**（负权重）两脚**真交叉**（`sep < 0`，左脚越过中线到右侧）：

        cost = clamp(−sep, 0, max_cost) · w_act · w_dir · w_speed

    与 `feet_lateral_width_cost` 的差别：不做 `sep_floor` 死区，`sep` 一过 0 就给梯度，
    形成「越过中线」的额外陡峭惩罚。

    ⚠️ **本实现比 AMP_mjlab 多了 `gate_lateral`（默认开）**：AMP_mjlab 原文不乘方向
    门控，但本仓库参考动作的侧移 clip 是**交叉式侧步**（实测 21% 帧 sep < 0），
    无条件惩罚会与 AMP 风格先验对撞、压制侧移。理由与实测数据见模块 docstring。
    """
    command = _command_or_raise(env, command_name)
    w_act = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > command_threshold).float()
    w_dir = _lateral_dir_gate(command, lateral_eps) if gate_lateral else torch.ones_like(command[:, 0])

    sep = feet_lateral_sep(env, asset_cfg)
    raw = torch.clamp(-sep, min=0.0)
    if max_cost is not None:
        raw = raw.clamp(max=float(max_cost))

    # 直线走应 ≈ 0；侧移（参考是交叉式侧步）允许 > 0，用来验证方向门控确实在起作用。
    _log_metric(env, "foot_cross_frac", (sep < 0).float().mean())
    return raw * w_act * w_dir * _walk_speed_gate(command, walk_max_lin_vel, walk_gate_width)


#: `_motion_diag_once` 的"只打一次"标记
_MOTION_DIAG_PRINTED = False


def _motion_diag_once(env) -> None:
    """**首步**打一次"数据层镜像增广 / 参考 clip 数"的真值 —— 只打印一次。

    为什么需要：`_validate_teacher_contract` 在 `load_managers()` 里跑，**早于任何 reset**，
    而 `_env_symmetric_augmentation_mask` 是在 `motion_buffer.reset()` 里抽签填的
    ⇒ 那时它必然全 0（"运行时条件=False、被镜像 env 比例=0.0" 只是**启动瞬间**的快照，
    不能证明增广没生效）。这条在**训练步**里读，看到的才是稳态真值。
    """
    global _MOTION_DIAG_PRINTED
    if _MOTION_DIAG_PRINTED:
        return
    _MOTION_DIAG_PRINTED = True
    try:
        _ref = env.scene.sensors.get("motion_reference") if isinstance(env.scene.sensors, dict) else None
    except Exception as _exc:  # pragma: no cover
        print(f"[MOTION-DIAG] 拿不到 motion_reference：{_exc!r}")
        return
    if _ref is None:
        print("[MOTION-DIAG] scene.sensors 里没有 motion_reference")
        return
    _out = {}
    for _k in ("symmetric_augmentation_joint_mapping", "symmetric_augmentation_link_mapping"):
        _out[_k] = "已设" if getattr(_ref.cfg, _k, None) is not None else "未设"
    try:
        _out["条件"] = bool(_ref._symmetric_augmentation_conditions_met)
    except Exception as _exc:
        _out["条件"] = f"?({_exc!r})"
    try:
        _m = _ref._env_symmetric_augmentation_mask
        _out["被镜像env比例"] = f"{float(_m.float().mean()):.3f}({int(_m.sum())}/{_m.numel()})"
    except Exception as _exc:
        _out["被镜像env比例"] = f"?({_exc!r})"
    try:
        _sel = _ref._buffer_motion_selection
        _out["buffer内clip索引"] = str([int(i) for i in list(_sel)[:8]])
    except Exception as _exc:
        _out["buffer内clip索引"] = f"?({_exc!r})"
    print("[MOTION-DIAG] " + "，".join(f"{k}={v}" for k, v in _out.items()))


def _log_tilt_probes(env, asset) -> None:
    """记录**躯干倾角的分布**（不只是均值）——回答"是不是经常出现 45° 这种大倾角"。

    背景（重要）：`Episode_Reward/flat_orientation_l2` 是**所有 env、所有时刻的平均**。
    均值小（我们实测反推 ≈4°）**不等于**"任何时刻都竖直"：跳跃/腾空/落地瞬间可以有
    几十度倾角，只要它时间占比小、且被 `θ²` 压低，均值依然很小
    ⇒ 判断"视觉上有没有 45° 倾斜"必须看**分布/超阈比例**。

    记录：
        Step/tilt_deg_mean     平均倾角（度）
        Step/tilt_p95_deg      **95 分位**倾角（度）⭐ 看它才能看到偶发大倾角
        Step/tilt_gt{20,30,45}_frac  倾角超过 20/30/45 度的 env 比例
        Step/grav_bx_mean      躯干系重力 x 分量（**有符号**：判别前倾/后倾）
        Step/grav_bz_mean      躯干系重力 z 分量（竖直时为 −1）
    """
    g = asset.data.projected_gravity_b  # (N, 3)；root = torso_link ⇒ 躯干系下的重力
    tilt = torch.atan2(torch.linalg.vector_norm(g[:, :2], dim=1), g[:, 2].abs().clamp(min=1e-6))
    tilt_deg = tilt * (180.0 / torch.pi)

    _log_metric(env, "tilt_deg_mean", tilt_deg.mean())
    _log_metric(env, "tilt_p95_deg", torch.quantile(tilt_deg, 0.95))
    for thr in (20.0, 30.0, 45.0):
        _log_metric(env, f"tilt_gt{int(thr)}_frac", (tilt_deg > thr).float().mean())
    _log_metric(env, "grav_bx_mean", g[:, 0].mean())
    _log_metric(env, "grav_bz_mean", g[:, 2].mean())


#: `waist_pitch_joint` 在 `asset.joint_names` 里的下标缓存（按 `id(asset)` 键，避免每步搜字符串）
_WAIST_PITCH_IDX: dict[int, int | None] = {}


def _waist_pitch_index(asset) -> int | None:
    """找 `waist_pitch_joint` 的下标（找不到返回 None）。结果缓存，每步调用零开销。"""
    key = id(asset)
    if key not in _WAIST_PITCH_IDX:
        idx: int | None = None
        for i, name in enumerate(getattr(asset, "joint_names", ())):
            if name.endswith("waist_pitch_joint"):
                idx = i
                break
        _WAIST_PITCH_IDX[key] = idx
    return _WAIST_PITCH_IDX[key]


def joint_pos_limits_margin(
    env,
    asset_cfg: SceneEntityCfg,
    margin: float = 0.15,
    power: float = 2.0,
) -> torch.Tensor:
    """**软边距**关节限位惩罚：在硬限位**内侧** `margin` (rad) 就开始逐步收费。

    为什么需要它（实测踩坑）
    ------------------------
    `isaaclab.envs.mdp.joint_pos_limits` 的形式是
    `Σ [clamp(soft_lower − q, 0) + clamp(q − soft_upper, 0)]` —— **线性，且在限位内严格为 0**。
    ⇒ 策略只要停在限位**内侧一点**就完全不付钱，而"贴近限位"常常能换来省力/更稳的姿势
    ⇒ "`waist_pitch_joint` 长期顶在 ±0.52"就成了它的**最优点**，不是随机现象。
    （更糟的是：我们此前这一项 `joint_names` **只写了踝关节** ⇒ 腰/髋/膝**根本不罚**，
    所以在 TB 里看 `Episode_Reward/dof_pos_limits` 完全反映不出腰的饱和。）

    本函数的形状（逐关节、归一化到"硬限位处 = 1.0"）
    ------------------------------------------------
        u_j   = [clamp(lower_j + margin − q_j, 0) + clamp(q_j − (upper_j − margin), 0)] / margin
        cost  = Σ_j u_j ** power

    * `u_j = 0`：距限位仍有 `margin` 以上 ⇒ **不罚**（不干扰正常大幅动作）；
    * `u_j = 0.5`：进入边距一半 ⇒ `0.25`（power=2）⇒ 温和提醒；
    * `u_j = 1`：**到达硬限位** ⇒ `1.0`；越界继续按幂增长（不会像线性项那样"过界就跳"）。

    `power=2` 在边距入口处**梯度为 0**（进入边距不会"抽一下"），越靠近限位越陡 —— 典型软墙。
    想更"硬"可以 `power=1.0`（线性 = 进入边距立刻线性收费）或调大权重。

    权重语义
    --------
    `weight = −1.0` ⇒ **单个关节顶到硬限位时 −1.0/步**（与满额速度跟踪 ~1.5/步同量级）。
    `weight = −1.5, margin = 0.15` 在贴近限位处的斜率 ≈ `−1.5 × 2 × u/margin ≈ −10/rad`，
    与此前踝关节的 `−10 × 线性` **同量级** ⇒ 所以是"把完全没人管的关节（腰/髋/膝）
    纳入同等级别的约束"，而不是无脑加倍。
    """
    asset = env.scene[asset_cfg.name]
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]  # (N, J, 2)
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (N, J)

    lower = limits[..., 0] + margin
    upper = limits[..., 1] - margin
    # 行程比 2×margin 还小的关节：把边距夹到中点，避免 lower > upper 造成"恒罚"
    mid = 0.5 * (lower + upper)
    lower = torch.minimum(lower, mid)
    upper = torch.maximum(upper, mid)

    u = (torch.clamp(lower - q, min=0.0) + torch.clamp(q - upper, min=0.0)) / float(margin)

    # ── 诊断（只打日志、不参与奖励）：回答"到底哪个关节长期贴着限位" ─────────────
    # `limit_margin_u_max = 1` ⇒ 有某个关节已经顶到**硬限位**；
    # `limit_margin_u_frac`    ⇒ 有多少比例的关节已经进了边距（>0）。
    # `waist_pitch_*`          ⇒ 本轮的重点病号（腰部 pitch 是否长期饱和）。
    try:
        _log_metric(env, "limit_margin_u_max", u.max())
        _log_metric(env, "limit_margin_u_frac", (u > 0.0).float().mean())

        wp = _waist_pitch_index(asset)
        if wp is not None:
            sel = asset_cfg.joint_ids
            sel_ids = list(range(asset.num_joints))[sel] if isinstance(sel, slice) else list(sel)
            if wp in sel_ids:
                k = sel_ids.index(wp)
                _log_metric(env, "waist_pitch_abs_mean", q[:, k].abs().mean())
                _log_metric(env, "waist_pitch_enter_frac", (u[:, k] > 0.8).float().mean())

        _log_tilt_probes(env, asset)

        # 顶格的是哪几个关节（每 500 步打印一次，避免刷屏）——回答"谁在超限"
        step = int(getattr(env, "common_step_counter", 0))
        if step % 500 == 0:
            mean_u = u.mean(dim=0)                     # (J,)
            top = torch.topk(mean_u, k=min(5, mean_u.numel()))
            sel = asset_cfg.joint_ids
            sel_ids = list(range(asset.num_joints))[sel] if isinstance(sel, slice) else list(sel)
            names = [asset.joint_names[i] for i in sel_ids]
            desc = ", ".join(f"{names[j]}={mean_u[j].item():.3f}" for j in top.indices.tolist())
            print(f"[LIMIT-TOP] step={step} 平均边距占用最高的 5 个关节：{desc}", flush=True)
    except Exception:  # noqa: BLE001 —— 纯诊断，绝不能因为它影响训练
        pass

    return torch.sum(u.pow(float(power)), dim=1)


def feet_swing_time_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    max_air_time: float = 0.6,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """**惩罚**（负权重）"某只脚腾空太久" —— 机械上强制**交替支撑**。

        cost = Σ_f clamp(air_time_f − max_air_time, 0) · 1[‖cmd_xy‖ > command_threshold]

    ## 为什么需要它（V9h@4770 实测暴露的漏洞）

    `feet_air_time_positive_biped` 只要求"**恰好一只脚着地**"（`single_stance`），
    它**不区分**这只脚是刚落地还是已经踩了 10 秒 —— 而且它取的是
    `min(支撑脚 contact_time, 摆动脚 air_time)` 的**时长**，两者都随时间增长
    ⇒ 一个"**单脚滑行 / 单脚侧跳**"（另一条腿一直吊在空中）可以**永久**满足条件、
    持续拿满该项（最多 +1.0/步）。
    同时 `lin_vel_z_l2` 只罚**竖直**速度（V9h 实测该项只有 −0.0034/s ⇒ 机器人几乎
    没有竖直弹跳）⇒ 于是"一侧脚一直踩地、另一侧吊着、靠上身侧倾推进"成了
    既躲过抗跳项、又拿满步态项的最优解，即实测观察到的"**单脚侧身跳着走**"。

    ⚠️ **镜像损失拦不住这种现象**（V9h 实测镜像损失 0.0368 很小，步态却完全单侧）。
    ⚠️ 机理已勘误（2026-10-08）：原先写"单侧步态实际访问到的状态，其镜像态从未被访问过
    （off-distribution）⇒ 约束可以在'没去过的状态'上被满足" —— 那是**策略侧**的说法，
    **不是当时的实际根因**。根因在**命令分布**：V9h 当时**只给了前向速度**（纯 `(0, 0.7)`，
    vy = wz = 0；侧向/转向是 V10 的③指令扩展才引入的）⇒ 状态分布集中在"直行流形"上、
    **没有镜像场景的计算空间 ⇒ 镜像损失在"只有前向速度"的条件下无法正常生效**
    （见 `docs/amp_walk/EXPERIMENTS.md` §A1）。

    ## 语义

    - **走路**：摆动相 0.2~0.5 s ⇒ 恒为 0，**完全免费**（`max_air_time=0.6` 留了余量）；
    - **单脚吊腿**：罚值随时间**线性增长**。权重 −0.5 时，腾空 1 s = −0.5/步（−25/s），
      足以压过 `feet_air_time` 最多 +1.0/步的支付 ⇒ **"把脚放下来"变成唯一划算的走法**，
      而"放下来"就必然进入交替支撑。

    诊断量 `feet_swing_excess`（单脚平均超长腾空秒数）会进日志：
    走路应 ≈ 0，单脚吊腿会明显 > 0。
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    # 只统计"不在接触"的那只脚（着地脚的 air_time 无意义）——与 `feet_air_time_positive_biped` 同样的取法
    excess = torch.where(
        in_contact,
        torch.zeros_like(air_time),
        (air_time - float(max_air_time)).clamp(min=0.0),
    )
    cost = excess.sum(dim=1)
    command = _command_or_raise(env, command_name)
    gate = (torch.norm(command[:, :2], dim=1) > command_threshold).float()
    _log_metric(env, "feet_swing_excess", cost.mean())
    # ⭐ 指令分布**真值**探针（环境层，读 `command_manager`）——
    # 与桥接层那条"切片读出来的指令"（`Mean step cmd_slice_*`）对照，
    # 用来判定 `disable_style_on_stand` 的门控切片有没有错位。
    _cmd_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    _log_metric(env, "cmd_speed_mean", _cmd_speed.mean())
    _log_metric(env, "cmd_low_frac_true", (_cmd_speed <= 0.1).float().mean())
    _motion_diag_once(env)
    return cost * gate


def feet_air_time_target_biped(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    good_lo: float = 0.4,
    good_hi: float = 1.0,
    soft: float = 0.3,
    tau: float | None = None,
    command_threshold: float = 0.1,
    motion_threshold: float = 0.0,
    motion_ang_scale: float = 0.5,
    stand_penalty: float = 0.0,
    stand_min_d: float = 0.03,
    # --- ⭐ V23（2026-10-03）：进度判据（方向投影 + EMA 低通），`progress_kappa=0` = 旧行为 ---
    progress_kappa: float = 0.0,
    progress_tau: float = 0.3,
    progress_omega_kappa: float = 0.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """**目标时长带**版的步态奖励（用户 2026-10-01 指定）：单脚支撑相位时长落在
    `[good_lo, good_hi]` 内给满额，两侧**线性衰竭**到 0。

        d     = min(支撑脚 current_contact_time, 摆动脚 current_air_time)   # 仅"恰好单脚着地"时
        score = 1 − clamp((good_lo − d)/soft, 0, 1) − clamp((d − good_hi)/soft, 0, 1)   ∈ [0, 1]
        reward = score · 1[恰好一只脚着地] · 1[‖cmd_xy‖ > command_threshold]

    ## 为什么改（与 `feet_air_time_positive_biped` 的对照）

    旧版是**时长本身**、并 `clamp(max=threshold)`：

        reward_old = clamp(min(支撑接触时长, 摆动腾空时长), max=0.6)

    ⇒ 两个致命性质：

    1. **短到 5 ms 也给分**（只是"给得少"）⇒ 高频小碎步照样能刷（V17 实测该项核值
       **0.005~0.009 s**，即单支撑相位只有几毫秒 —— "蹭地快倒脚"）；
    2. **越长越给**（到 0.6 s 封顶）⇒ "单脚一直踩着、另一只脚吊着"这条捷径能长期拿满
       （V9h 观察到的"单脚侧身跳着走"；V14 从零更是直接收敛成单脚跳）。

    新版把"时长"换成"**落在目标带里就给满额**"：

    | 单支撑时长 d | 旧版得分 | **新版得分** | 物理含义 |
    |---|---|---|---|
    | 0.01 s（碎步/蹭地） | 0.01 | **0** | 几乎没抬脚 |
    | 0.2 s | 0.20 | 0.33 | 偏快，衰竭中 |
    | **0.4 ~ 1.0 s** | 0.4 / 0.6（且封顶） | **1.0** | ✅ 人类步行单支撑带 |
    | 1.3 s 以上 | 0.6（封顶） | **0** | 单脚长时间站立/跳 |

    ⇒ **同时反对碎步（<good_lo）与反对"单脚长时间支撑/跳"（>good_hi）**，
    而且它**不能靠"踩久一点"无限刷分**（上限就是 1.0）⇒ V14 那个"跳着刷分"的坑
    从**结构上**消失了（不再需要靠压低权重来防它，V18 因此保持 V17 的权重 4.0）。
    注意：同样的权重下，本项的每步收益会从旧版的 ~0.03（按秒计费）抬到 **~1.2/步**
    （单支撑占比 ~0.4 × 得分 1.0 × 权重 4 × λ 0.75）—— 这是**有意**的：步态项必须
    有和速度跟踪（混合后 1.06/步）同量级的话语权，否则压不住碎步。

    ## 参数怎么定

    - `good_lo = 0.4 s`：人类常速步行单支撑相位 ≈ 0.35~0.45 s；本仓库前向 clip
      `walk_forward_loop_002__A022` 巡航 0.6~1.0 m/s，落在同一量级。
    - `good_hi = 1.0 s`：明显慢于正常步频（1 s 单支撑 ≈ 0.5 步/s）；比这更慢就属于
      "单脚吊着/站立"，所以上界不设更高。
    - `soft = 0.3 s`：区间外线性衰竭的尺度 ⇒ 0.4−0.3 = 0.1 s 及以下得 0；
      1.0+0.3 = 1.3 s 及以上得 0；中间线性过渡（有梯度，不会"抽"）。
    - ⭐ `tau`（**V19b 新增**，默认 `None` = 保持线性）：> 0 时把**下侧**换成
      "近零端加密"的指数型短缺

          below(d) = clamp( (exp(−d/tau) − exp(−good_lo/tau)) / (1 − exp(−good_lo/tau)), 0, 1 )

      它在 `d = good_lo` 处恰为 **0**、在 `d = 0` 处恰为 **1**，中间**下凸**：

      | d（单支撑时长） | 线性 `(good_lo−d)/soft` | **指数 `tau=0.08`** |
      |---|---|---|
      | 0.005 s（碎步） | 0.988 | **0.939** |
      | 0.010 s | 0.975 | **0.881** |
      | 0.050 s | 0.875 | **0.532** |
      | 0.100 s（线性档的死区边界） | 0.750 | **0.282** |
      | 0.200 s | 0.500 | **0.076** |
      | 0.400 s | 0 | **0** |

      ⇒ 得分视角：`score = 1 − below − above`，在 `d = 0.01 s`（小跳的工作点）处
      **score 0.025 → 0.118（×4.7）**，且**梯度 1/soft=2.5/s → ≈12.5/s（×5）**；
      从 0.35 s 起两者都≈1 ⇒ 与 `[good_lo, good_hi]` 好区间**不打架**。
      
      **为什么需要它（V18/V19a 实测）**：两种形状都留了“小跳拿不到钱但也亏不了”的
      平台 —— V18 `soft=0.3 < good_lo=0.4` ⇒ `d < 0.1 s` 处 `below` 饱和到 1，
      **score 与梯度同时恒 0**（实测 `in_band_frac=0.0000`、`d=0.009~0.011 s`）；
      V19a 改成 `soft=0.4` 后消除了死区，但在 `d=0.01 s` 处 score 只有 0.025、
      梯度只有 2.5/s ⇒ 爬了 1150 轮 `d` 只从 0.0096 到 0.0111 s（**平台化**）。
      指数形状让“小跳”从**第一步**起就明显亏钱（拿不到那 ~2.4/步的上限）。

    ⚠️ **与其他项的配合**：`feet_swing_time_penalty`（罚腾空 > 0.6 s）与本项上界**功能重叠**
    ⇒ 启用本项时建议把它权重置 0（V18 就是这么做的），否则两项会互相打架
    （0.6~1.0 s 区间里一个给满分、另一个在罚）。

    ## 诊断量（进 TB）

    - `air_time_single_stance`：单支撑相位时长 d 的**均值**（旧版 TB 上那个
      `Episode_Reward/feet_air_time` 因为按秒计费而被压到 0.03 难判断，这个量直接可读）；
    - `air_time_in_band_frac`：`score > 0.5` 的步数占比（"步态合规率"）；
    - `air_time_hop_frac`：`d < 0.1 s` 的占比（**小跳/蹭地占比**，越低越好）；
    - `air_time_motion_gate_off_frac`：被"实际未移动"门控关掉的比例（仅 `motion_threshold>0` 时）。

    ## ⭐ 门控（`motion_threshold`，V19c 新增）

    原有的命令门控只阻止**指令接近 0** 的情形：

        gate_cmd = 1[ ‖v_cmd,xy‖ > command_threshold ]

    它**管不住**另一种刷分：指令非零（如 0.6 m/s 前进）但**实际不前进**，靠
    "原地左右脚交替抬放"去拿单支撑分红。实测（V19b）确实出现"静止/慢速也在
    原地交替站"。⇒ 追加一道**实测运动**门控：

        speed = ‖v_actual,yaw-frame‖_xy + motion_ang_scale · |ω_z,actual|
        gate_motion = 1[ speed > motion_threshold ]
        reward = score · 1[恰好单支撑] · gate_cmd · gate_motion

    `motion_ang_scale` 把**转向**也算作"在动"（原地转身本来就要迈步，不应被掐掉）。
    `motion_threshold = 0` ⇒ 完全关掉这道门（向后兼容 V18/V19a/V19b）。

    ## ⭐ 站立时给**负值**（`stand_penalty`，V19d 新增）

    上面两道门对"站立时原地踏步"是**零价**的：`cmd_speed ≤ threshold` ⇒ `gate_cmd = 0`
    ⇒ 该奖励贡献 **0**（既不奖也不罚）。加上 `disable_style_on_stand=True` ⇒ 站立 env
    的风格奖励也为 0 ⇒ **"站着左右脚交替抬放"几乎不需要付任何代价**（唯一成本是
    `track_default_pose_on_stand`，实测约 0.36/步）。实测（V19c@9700）该行为稳定存在。

    ⇒ 把站立档从"0 价"改成"负价"：

        stand = 1[ ‖v_cmd,xy‖ + |ω_z,cmd| ≤ command_threshold ]     # 真正的站立/低速
        pen   = −stand_penalty · min(d / good_lo, 1)                 # 抬得越久罚得越多
        reward = score·1[单支撑]·gate_cmd·gate_motion  +  pen·1[单支撑]·stand

    两个分支互斥（`gate_cmd` 与 `stand` 互补）。`min(d/good_lo, 1)` 使 `d → 0` 时罚也→ 0
    （"抬一下立刻放"几乎不罚），罚满对应 `d ≥ good_lo`（真的在站立时长抬脚）⇒ 方向明确、
    上限 = `stand_penalty`（在 score 空间，乘 weight 4.0 后最大 −4·stand_penalty/步）。
    `stand_penalty = 0` ⇒ 完全关闭（向后兼容）。

    ## ⭐ V23（2026-10-03）：进度判据（`progress_kappa > 0` 启用）

    把奖/罚两支的**判据**从"指令 + 实测瞬时幅值"换成"**实测进度**"：

        v̂ = v_cmd,xy / max(‖v_cmd,xy‖, eps)
        EMA ← (1−α)·EMA + α·⟨v_act,xy^yaw, v̂⟩           # ⭐ 带符号投影（不取 relu）
        progress = 1[ EMA > κ·‖v_cmd,xy‖ ] OR 1[ |ω_cmd|>eps 且 |ω_act| > κ_ω|ω_cmd| ]
        reward   = +score·1[单支撑]·gate_cmd·progress  −  stand_penalty·min(d/good_lo,1)·1[单支撑]·(1−progress)

    为什么：
    * `motion_threshold` 那道"实测根速度**瞬时幅值**"门在三代 run 里关闭率都 < 1%
      （踏步时前后摇摆的瞬时幅值本来就有 0.3~1 m/s），从未拦住踏步；且幅值对**符号
      不敏感** ⇒ "摇得越猛越容易过门"；
    * 踏步的**方向投影**有正有负、低通后均值≈0，而真走恒正 ⇒ 能区分；
    * 负价条件由 `stand`（指令≈0）推广为 `no_progress` ⇒ **同时**覆盖"0 指令站立"
      与 V19c 记录的"指令 0.6 m/s 但原地踏步"两档（后者在旧版里是完全零成本的）。

    `progress_kappa = 0` ⇒ 完全回退到旧行为（不建 EMA、不写 progress 判据）。
    新探针：`air_time_proj_ema` / `air_time_proj_ratio` / `air_time_progress_frac` /
    `air_time_progress_off_frac` / `air_time_lift_frac`。
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    # 与旧版同样的取法：取两脚中较小者（摆动脚的腾空时长 vs 支撑脚的接触时长），
    # 非单支撑时被 `where(..., 0)` 置零 ⇒ 后面得分自然为 0。
    d = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]

    if tau is not None and float(tau) > 0.0:
        # ⭐ 近零端加密（V19b）：下凸的指数型短缺，两端分别锚在 good_lo / 0 上。
        _tau = float(tau)
        e_lo = math.exp(-float(good_lo) / _tau)
        below = torch.clamp((torch.exp(-d / _tau) - e_lo) / (1.0 - e_lo), min=0.0, max=1.0)
    else:
        below = torch.clamp((float(good_lo) - d) / float(soft), min=0.0, max=1.0)
    above = torch.clamp((d - float(good_hi)) / float(soft), min=0.0, max=1.0)
    score = 1.0 - below - above

    command = _command_or_raise(env, command_name)
    gate_cmd = (torch.norm(command[:, :2], dim=1) > command_threshold).float()
    gate = gate_cmd

    if float(motion_threshold) > 0.0:
        asset = env.scene[asset_cfg.name]
        vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
        speed_actual = torch.norm(vel_yaw[:, :2], dim=1) + float(motion_ang_scale) * torch.abs(
            asset.data.root_ang_vel_w[:, 2]
        )
        gate_motion = (speed_actual > float(motion_threshold)).float()
        _log_metric(env, "air_time_motion_gate_off_frac", (1.0 - gate_motion).mean())
        gate = gate * gate_motion

    # ⭐ V23：进度判据（方向投影 + EMA 低通）
    #   替代"实测根速度瞬时幅值"门 —— 那个门在 V19c/V19d/V20 三代里的关闭率只有
    #   0.46% / 0.46% / 0.63~0.95%（几乎永远开着），且对符号不敏感（摇得越猛越易
    #   过门，是反向激励）。方向投影 + 低通把"原地抬脚"与"真的在走"分开。
    progress = None
    if float(progress_kappa) > 0.0:
        gate_prog = _amp_utils.get_progress_gate(env, progress_tau)
        # ⚠️ 刚 reset 的 env 必须清 EMA，否则新回合头几步带着上一段残留 ⇒ 门虚假开启
        gate_prog.reset_by_mask(env.episode_length_buf <= 1)
        asset = env.scene[asset_cfg.name]
        vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
        ema, progress, ratio = gate_prog(
            vel_yaw[:, :2],
            command,
            asset.data.root_ang_vel_w[:, 2],
            command[:, 2],
            kappa=float(progress_kappa),
            omega_kappa=float(progress_omega_kappa),
        )
        _log_metric(env, "air_time_proj_ema", ema.mean())
        _log_metric(env, "air_time_proj_ratio", ratio.mean())
        _log_metric(env, "air_time_progress_frac", progress.mean())
        _log_metric(env, "air_time_progress_off_frac", (1.0 - progress).mean())

    _log_metric(env, "air_time_single_stance", d.mean())
    _log_metric(env, "air_time_in_band_frac", (score > 0.5).float().mean())
    _log_metric(env, "air_time_hop_frac", (d < 0.1).float().mean())
    out = score * single_stance.float() * gate
    if progress is not None:
        # ⭐ V23：正分只在"实测在走"时给
        out = out * progress

    if float(stand_penalty) > 0.0:
        pen_score = float(stand_penalty) * torch.clamp(d / float(good_lo), max=1.0)
        if progress is not None:
            # ⭐ V23：负价条件 = **实测没在走**（同时覆盖"0 指令站立"与"有指令但不前进"两档）
            no_prog = 1.0 - progress
            out = out - pen_score * single_stance.float() * no_prog
            # 探针：全局"抬了脚"占比（不分档） + 实际付出的负价均值
            _log_metric(env, "air_time_lift_frac", ((d > float(stand_min_d)) & single_stance).float().mean())
            _log_metric(env, "air_time_stand_pen_mean", (pen_score * single_stance.float() * no_prog).mean())
        else:
            # 旧行为（V19d/V20/V21）：只在"指令≈0"档给负价
            cmd_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
            stand = (cmd_speed <= command_threshold).float()
            out = out - pen_score * single_stance.float() * stand
            _log_metric(
                env,
                "air_time_stand_step_frac",
                (stand * ((d > float(stand_min_d)) & single_stance).float()).mean(),
            )
            _log_metric(env, "air_time_stand_pen_mean", (pen_score * single_stance.float() * stand).mean())
    return out


# ============================================================================
# ⭐ V33 只读诊断：把 `track_lin_vel_xy` **无损拆成 x / y 两轴**并按指令档分桶
# ============================================================================
#
# 【为什么这是"无损"拆分】
# ------------------------
# 仓库既有实现（`rewards.py::track_lin_vel_xy_yaw_frame_exp`）是
#       R = exp( −(e_x² + e_y²) / σ² )
# 而 exp(−(a+b)) = exp(−a)·exp(−b) ⇒
#       R = **F_x · F_y**,  F_x = exp(−e_x²/σ²), F_y = exp(−e_y²/σ²)
# ⇒ 两个"轴向因子"的**乘积**逐样本等于原奖励项，**一分不差**。
#   所以**完全不用改奖励函数**：只要把 F_x / F_y 记进日志，就能直接回答
#   "跟踪差是 x 轴拖的还是 y 轴拖的"。
#
# 【为什么侧移问题值得单独看这一项（第 4 个机制）】
# ------------------------------------------------
# 乘法形式对"单轴掉链子"是**结构性温和**的：e_y = 0.3、σ = 0.5 时
#       F_y = exp(−0.36) = 0.70
# ⇒ **即使 y 方向完全不跟**，只要 x 跟得好，仍能拿到 **70%** 的奖励。
#   ⇒ 策略"半放弃侧移"的代价被乘法结构压低了 —— 与"方向门控 / 进度判据"
#     那两条并列，但更基础（它不依赖任何新加的门，是既有实现自带的）。
#
# 【替代方案（若诊断证实 y 轴确实是瓶颈）】
# ----------------------------------------
#   真·拆分奖励：`0.5·F_x + 0.5·F_y`（算术平均）或 `sqrt(F_x·F_y)`（几何平均）。
#   ⚠️ 注意：**几何平均 == 今天的形式**（√(F_x·F_y) = exp(−(e_x²+e_y²)/2σ²)，
#   只是等效地把 σ 放大 √2 倍）⇒ 要真正"抬高单轴权重"必须用**算术平均**。
#   本函数本轮**不改**训练奖励口径（返回值仍是无损的原项），只做测量。
#
# 【本函数的接法（见 `amp_walk_v33_env_cfg.py`）】
# -------------------------------------------------
#   它**顶替**既有项 `track_lin_vel_xy_exp` 的 `func`（权重/params 逐字不动）
#   ⇒ 训练奖励**一分不变**（有一次性恒等自检钉住），同时拿到全部诊断曲线。
#   为什么不新加一个 weight=0 的项：IsaacLab 是否跳过 weight==0 的 term 不确定，
#   而**后置新增**字段能否被 manager 识别也不确定 ⇒ 换 func 是零风险的做法。

#: 恒等自检只跑一次（见 `track_lin_vel_xy_axis_diag`）
_LIN_TRACK_IDENTITY_CHECKED = False


def track_lin_vel_xy_axis_diag(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """**无损诊断版**跟踪项：按 x/y 两轴 + 按指令档分桶记日志，返回值 = 原跟踪项本身。

    返回值 `F_x · F_y` **逐样本等于** `track_lin_vel_xy_yaw_frame_exp`（配置里就用它
    顶替那一项的 `func`，权重与 `params` 原样保留）⇒ **训练奖励完全不变**，
    只是白拿一堆 TensorBoard 曲线。

    ⚠️ **必须每一步都记（禁止做"降频/条件记日志"优化，踩过的坑）**

    `OnPolicyRunner.log()` 聚合 `step_infos` 的方式是：

        for key in locs["step_infos"][0]:        # ← 用**第一条** step_info 的 key
            for step_info in locs["step_infos"]: # ← 去索引**全部** step_info
                step_info[key]                   # ⇒ 缺 key 立即 KeyError

    即"本轮迭代里第一条 step 记了哪些 key"决定了后面每一步**必须**都有这些 key。
    所以像 `if step % 4 == 0: _log_metric(...)` 这种降频写法会让**训练直接崩**
    （`common_step_counter = 0` 满足条件、`1` 不满足 ⇒ 第一步有 key、第二步没有）。
    本函数里所有 `_log_metric` **无条件执行**。
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # 与 `track_lin_vel_xy_yaw_frame_exp` **逐字相同**的观测量
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    e = command[:, :2] - vel_yaw[:, :2]
    fx = torch.exp(-(e[:, 0] ** 2) / std**2)
    fy = torch.exp(-(e[:, 1] ** 2) / std**2)
    joint = fx * fy  # == 原奖励项（逐样本）

    # ⭐ 一次性恒等自检：本项要**顶替** `track_lin_vel_xy_yaw_frame_exp` 进训练
    #   ⇒ 必须是它的忠实替身（`exp(-(ex²+ey²)/σ²) ≡ exp(-ex²/σ²)·exp(-ey²/σ²)`）。
    #   只做一次（浮点误差 ~1e-16，阈值给足）；不通过就直接抛错，不让训练带着错奖励跑。
    global _LIN_TRACK_IDENTITY_CHECKED
    if not _LIN_TRACK_IDENTITY_CHECKED:
        _joint_ref = torch.exp(-torch.sum(torch.square(e), dim=1) / std**2)
        _err = float((joint - _joint_ref).abs().max())
        if _err > 1e-5:
            raise RuntimeError(
                f"track_lin_vel_xy_axis_diag 与原跟踪项不等价！max|Δ|={_err:.3e}"
                "（拆分恒等式被破坏 ⇒ 会悄悄改变训练奖励，拒绝继续）"
            )
        _LIN_TRACK_IDENTITY_CHECKED = True

    _log_metric(env, "lin_track_fx", fx.mean())
    _log_metric(env, "lin_track_fy", fy.mean())
    _log_metric(env, "lin_track_joint", joint.mean())
    _log_metric(env, "lin_track_err_x_abs", e[:, 0].abs().mean())
    _log_metric(env, "lin_track_err_y_abs", e[:, 1].abs().mean())

    # ---- 分桶：口径与指令采样**同源**（纯按"指令长什么样"分，不硬编码边界）----
    speed = torch.norm(command[:, :2], dim=1) + command[:, 2].abs()
    vx, vy = command[:, 0], command[:, 1]
    tiny = 1e-6
    low = speed <= float(command_threshold)          # 低速/立定（命中站立门控）
    side = (~low) & (vx.abs() <= tiny)               # 纯侧向：vx≡0, ωz≡0
    back = (~low) & (~side) & (vx < -tiny) & (vy.abs() <= tiny)   # 纯后退
    lin = (~low) & (~side) & (~back) & (vy.abs() <= tiny)         # 纯前后
    diag = ~(low | side | back | lin)                # 对角/带转向
    for name, mask in (("low", low), ("side", side), ("lin", lin), ("back", back), ("diag", diag)):
        m = mask.float()
        n = m.sum().clamp(min=1.0)  # 桶内为空时不会 NaN
        _log_metric(env, f"lin_track_{name}_frac", m.mean())
        _log_metric(env, f"lin_track_{name}_joint", (joint * m).sum() / n)
        _log_metric(env, f"lin_track_{name}_fy", (fy * m).sum() / n)
        _log_metric(env, f"lin_track_{name}_ey_abs", (e[:, 1].abs() * m).sum() / n)
    return joint
