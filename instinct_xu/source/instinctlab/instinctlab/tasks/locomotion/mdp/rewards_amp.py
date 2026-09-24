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

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

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


def track_default_pose_on_stand(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    pos_scale: float = 4.0,
    vel_scale: float = 0.5,
    gate_tau: float = 0.25,
    pos_joint_weights: dict[str, float] | None = None,
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
    防止腰部锁在 90° 的假吸引子），且不会把 29 关节的均值整体抬高。

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
    abs_err = torch.abs(asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids])
    if pos_joint_weights:
        abs_err = abs_err * _joint_weight_row(asset, joint_ids, pos_joint_weights)

    pos_err = torch.mean(abs_err, dim=1)
    vel_err = torch.mean(torch.abs(asset.data.joint_vel[:, joint_ids]), dim=1)
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
