"""AMP 判别器输入：**刚体位姿 + 刚体速度**（对齐 AMP_mjlab / mjlab 系主流做法）。

背景（2026-10-02 定案）
----------------------
本仓库 V1~V19d 的判别器输入一直是**关节空间 5 项**
（`projected_gravity` / `joint_pos_rel` / `joint_vel_rel` / `base_lin_vel` /
 `base_ang_vel` = **67 维/帧**，见 `amp_walk_env_cfg.py::AMP_STATE_TERMS`），
而 AMP_mjlab（及大多数 mjlab/muJoCo 系 AMP 工程）用的是**刚体空间 4 项**：

| 项 | 每体维度 | 定义 |
|---|---|---|
| `body_pos_b` | 3 | 刚体位置，**相对 anchor（`torso_link`）且在 anchor 系下** |
| `body_ori_b` | 6 | 刚体姿态（相对 anchor），`matrix_from_quat(ori_b)[..., :, :2]` = 旋转矩阵**前两列**（连续 6D 表示） |
| `body_lin_vel_b` | 3 | 刚体线速度，从世界系打到**该刚体自身系** |
| `body_ang_vel_b` | 3 | 刚体角速度，同上 |

⇒ 13 体 × 15 = **195 维/帧**。13 体名单与 anchor 见
`AMP_mjlab/src/tasks/amp_loco/config/g1/env_cfgs.py:52-64`（`pelvis` + 双髋 roll +
双膝 + 双踝 roll + 双肩 roll + 双肘 + 双腕 yaw）。

**为什么这套表征更"可读"**：刚体位姿是关节角的**非线性函数**（前向运动学），刚体速度
是 `J(q)·q̇` 的**线性函数**（雅可比）。喂关节角 ⇒ 判别器必须自己学会 FK/雅可比才能
判断"脚离地多高、摆得多快"；喂刚体量 ⇒ 这些量**线性可读**。这是 mjlab 侧 2 帧就够、
而我们在 67 维上要 8~15 帧的根本原因（见 `RESULTS_amp_frames_offline_study.md`）。

**代价（刻意保留的取舍）**：所有量都对全局平移/偏航不敏感（anchor 相对 + 刚体系），
**看不见静态倾角**。我们的 `projected_gravity` 正好补这一块 ⇒ 两者互补，不是替换关系。

⚠️ 两处已知细节
----------------
1. **6D 取"前两列"而不是"前两行"**：mjlab 的**专家侧**
   （`rsl_rl/utils/motion_loader.py:110-112`）用 `mat[..., :, :2]`（前两**列**），
   而它的**策略侧观测**（`src/tasks/amp_loco/mdp/observations.py:64-65`）写的是
   `mat[..., :2]`（前两**行**）⇒ 它自己两侧口径不一致。本模块**两侧统一取前两列**
   （= 其专家侧口径 = Zhou et al. 2019 的连续 6D 表示），否则判别器会拿到两套不同的
   编码。
2. **帧序是 term-major**：IsaacLab 的 `_flatten_obs_group()` 逐 term `flatten`，
   所以 `history=2` 的 390 维是 `[pos_b(老), pos_b(新), ori_b(老), ori_b(新), …]`，
   而 mjlab 是 `[s(195), s'(195)]`。两者只差一个**固定置换**（对 MLP 等价，
   可被首层权重吸收），且**策略侧/参考侧两侧一致** ⇒ 不影响判别器语义。

返回值一律**展平**成 `(num_envs, K * num_bodies)`，这样单个 `ObsTerm` 配
`history_length = h` 就得到 `h * K * num_bodies` 维（与既有 5 项的做法一致）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

    from instinctlab.motion_reference import MotionReferenceManager

#: 默认的**被观测刚体**：不指定时 = 全部刚体（必须显式传名单，否则维度不可控）
_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_REFERENCE_CFG = SceneEntityCfg("motion_reference")
#: 默认 anchor = 躯干（与 AMP_mjlab 一致）。相对位姿/姿态都以它为原点与参考系。
_DEFAULT_ANCHOR_CFG = SceneEntityCfg("robot", body_names=("torso_link",))
_DEFAULT_ANCHOR_REFERENCE_CFG = SceneEntityCfg("motion_reference", body_names=("torso_link",))


def _rotation_6d(quat: torch.Tensor) -> torch.Tensor:
    """四元数 → **旋转矩阵前两列**（连续 6D 表示）。`(…, 4)` → `(…, 6)`。

    `isaaclab.utils.math.matrix_from_quat` 在不同版本里返回 `(…, 3, 3)` 或
    `(…, 4, 4)`（齐次），这里统一取左上 3×3 再切片，两种版本都正确。
    """
    mat = math_utils.matrix_from_quat(quat)
    if mat.shape[-1] == 4:  # 兼容返回齐次 4x4 的 IsaacLab 版本
        mat = mat[..., :3, :3]
    return mat[..., :, :2].reshape(*quat.shape[:-1], 6)


def _anchor_relative_pose(
    anchor_pos_w: torch.Tensor,
    anchor_quat_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 `(num_envs, num_bodies, ·)` 的刚体位姿换算到 anchor 系（`T_anchor^-1 · T_body`）。

    与 mjlab 一致：anchor 广播到每个刚体，再调 `subtract_frame_transforms`。
    """
    num_bodies = body_pos_w.shape[1]
    return math_utils.subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )


def _vel_in_body_frame(body_quat_w: torch.Tensor, vel_w: torch.Tensor) -> torch.Tensor:
    """世界系速度 → **各刚体自身系**：`q_body^-1 ⊗ v_world`。`(N, B, ·)` → `(N, B, ·)`。"""
    return math_utils.quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        vel_w.reshape(-1, 3),
    ).reshape(body_quat_w.shape[0], body_quat_w.shape[1], 3)


# ===========================================================================
# 策略侧（`robot`）
# ===========================================================================


def body_pos_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    anchor_cfg: SceneEntityCfg = _DEFAULT_ANCHOR_CFG,
) -> torch.Tensor:
    """各刚体位置，相对 anchor 且在 anchor 系下。`(num_envs, 3 * num_bodies)`。"""
    asset: Articulation = env.scene[asset_cfg.name]
    pos_b, _ = _anchor_relative_pose(
        asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]],
        asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]],
        asset.data.body_link_pos_w[:, asset_cfg.body_ids],
        asset.data.body_link_quat_w[:, asset_cfg.body_ids],
    )
    return pos_b.reshape(env.num_envs, -1)


def body_ori_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    anchor_cfg: SceneEntityCfg = _DEFAULT_ANCHOR_CFG,
) -> torch.Tensor:
    """各刚体姿态（相对 anchor），6D（旋转矩阵前两列）。`(num_envs, 6 * num_bodies)`。"""
    asset: Articulation = env.scene[asset_cfg.name]
    _, ori_b = _anchor_relative_pose(
        asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]],
        asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]],
        asset.data.body_link_pos_w[:, asset_cfg.body_ids],
        asset.data.body_link_quat_w[:, asset_cfg.body_ids],
    )
    return _rotation_6d(ori_b).reshape(env.num_envs, -1)


def body_lin_vel_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """各刚体线速度（各自刚体系）。`(num_envs, 3 * num_bodies)`。

    ⚠️ 注意**没有** `anchor_cfg`：体速度不经过 anchor 变换，只做"世界系 → 自身刚体系"。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_b = _vel_in_body_frame(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids],
        asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids],
    )
    return vel_b.reshape(env.num_envs, -1)


def body_ang_vel_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """各刚体角速度（各自刚体系）。`(num_envs, 3 * num_bodies)`。"""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_b = _vel_in_body_frame(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids],
        asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids],
    )
    return vel_b.reshape(env.num_envs, -1)


# ===========================================================================
# 参考侧（`motion_reference`）—— 与策略侧逐项同构
# ===========================================================================
#
# 数据源：`MotionReferenceManager.reference_frame`（`MotionReferenceData`），
# 已有 `link_pos_w / link_quat_w / link_lin_vel_w / link_ang_vel_w`，形状
# `(num_envs, num_frames, num_links, ·)`，`num_links == len(cfg.link_of_interests)`。
# ⇒ **不需要离线 FK、也不需要 ghost 模型**。
#
# `SceneEntityCfg("motion_reference", body_names=(...))` 的 `body_ids` 由
# `MotionReferenceManager.find_bodies()` 解析，**索引空间是 `link_of_interests`**
# （该 manager 刻意实现了与 IsaacLab SceneEntity 相同的接口）。


def body_pos_b_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_REFERENCE_CFG,
    anchor_cfg: SceneEntityCfg = _DEFAULT_ANCHOR_REFERENCE_CFG,
) -> torch.Tensor:
    """参考运动的各刚体位置（相对 anchor，anchor 系下）。`(num_envs, 3 * num_bodies)`。"""
    motion_reference: MotionReferenceManager = env.scene[asset_cfg.name]
    frame = motion_reference.reference_frame
    pos_b, _ = _anchor_relative_pose(
        frame.link_pos_w[:, 0, anchor_cfg.body_ids[0]],
        frame.link_quat_w[:, 0, anchor_cfg.body_ids[0]],
        frame.link_pos_w[:, 0, asset_cfg.body_ids],
        frame.link_quat_w[:, 0, asset_cfg.body_ids],
    )
    return pos_b.reshape(env.num_envs, -1)


def body_ori_b_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_REFERENCE_CFG,
    anchor_cfg: SceneEntityCfg = _DEFAULT_ANCHOR_REFERENCE_CFG,
) -> torch.Tensor:
    """参考运动的各刚体姿态（相对 anchor），6D。`(num_envs, 6 * num_bodies)`。"""
    motion_reference: MotionReferenceManager = env.scene[asset_cfg.name]
    frame = motion_reference.reference_frame
    _, ori_b = _anchor_relative_pose(
        frame.link_pos_w[:, 0, anchor_cfg.body_ids[0]],
        frame.link_quat_w[:, 0, anchor_cfg.body_ids[0]],
        frame.link_pos_w[:, 0, asset_cfg.body_ids],
        frame.link_quat_w[:, 0, asset_cfg.body_ids],
    )
    return _rotation_6d(ori_b).reshape(env.num_envs, -1)


def body_lin_vel_b_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_REFERENCE_CFG,
) -> torch.Tensor:
    """参考运动的各刚体线速度（各自刚体系）。`(num_envs, 3 * num_bodies)`。"""
    motion_reference: MotionReferenceManager = env.scene[asset_cfg.name]
    frame = motion_reference.reference_frame
    vel_b = _vel_in_body_frame(
        frame.link_quat_w[:, 0, asset_cfg.body_ids],
        frame.link_lin_vel_w[:, 0, asset_cfg.body_ids],
    )
    return vel_b.reshape(env.num_envs, -1)


def body_ang_vel_b_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_REFERENCE_CFG,
) -> torch.Tensor:
    """参考运动的各刚体角速度（各自刚体系）。`(num_envs, 3 * num_bodies)`。"""
    motion_reference: MotionReferenceManager = env.scene[asset_cfg.name]
    frame = motion_reference.reference_frame
    vel_b = _vel_in_body_frame(
        frame.link_quat_w[:, 0, asset_cfg.body_ids],
        frame.link_ang_vel_w[:, 0, asset_cfg.body_ids],
    )
    return vel_b.reshape(env.num_envs, -1)


# ===========================================================================
# ⭐ V33：anchor 的「投影重力」—— 给判别器补上**绝对俯仰/横滚**
# ===========================================================================
#
# 【为什么必须补这 3 维】
# 现有 390 维状态**全部**是「anchor 相对位姿 + 各刚体自身系速度」：
#   * `body_pos_b/ori_b` 相对 anchor ⇒ **整体旋转不变**；
#   * `lin_vel_b/ang_vel_b` 在**各刚体自身系**下 ⇒ 也随整体旋转一起转。
# ⇒ 把整段运动绕横轴转 20° 再喂进去，判别器**看不出任何区别**。
# 后果（V30 实测）：判别器对「后仰 24°」完全无感 ⇒ 姿态只能靠手写奖励项定价
# （`flat_orientation_l2` 的边际价 < 容忍后仰的收益 ⇒ 后仰是"理性"的）。
#
# 补的这 3 维 = anchor 自身系下的**世界重力方向** `R_anchor^T · (0,0,-1)`：
#   * **对偏航不变** —— 重力平行于偏航轴，绕 z 转不动它（`R' = Rz(ψ)·R` ⇒
#     `R'^T ĝ = R^T Rz(ψ)^T ĝ = R^T ĝ`）⇒ 与"判别器只看 yaw 不变特征"的口径一致；
#   * 对**平移**不变（本来就有）；
#   * **对俯仰/横滚敏感** —— 直立时 `(0,0,-1)`，绕横轴后仰 θ 时 ≈ `(±sin θ, 0, −cos θ)`。
#
# ⚠️⚠️ **千万不要"先 `yaw_quat` 再投影"**（2026-10-06 犯过，被数值自检抓住）：
#   重力本来就沿偏航轴 ⇒ 去掉偏航后结果**恒为 `(0,0,-1)`**（常量、零信息）。
#   偏航不变性是**白送**的，不需要任何对齐操作。

def _anchor_gravity_b(quat_w: torch.Tensor) -> torch.Tensor:
    """世界系四元数 `(N, 4)` → **该刚体自身系**下的重力方向 `(N, 3)`。

    直立 ⇒ `(0, 0, -1)`；绕横轴后仰 θ ⇒ 第二/第一分量出现 `±sin θ`。
    """
    gravity_w = torch.zeros(quat_w.shape[0], 3, device=quat_w.device, dtype=quat_w.dtype)
    gravity_w[:, 2] = -1.0
    return math_utils.quat_apply_inverse(quat_w, gravity_w)


def anchor_gravity_b(
    env: ManagerBasedEnv,
    anchor_cfg: SceneEntityCfg = _DEFAULT_ANCHOR_CFG,
) -> torch.Tensor:
    """anchor 刚体（躯干）自身系下的投影重力。`(num_envs, 3)`。"""
    asset: Articulation = env.scene[anchor_cfg.name]
    quat_w = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]
    return _anchor_gravity_b(quat_w)


def anchor_gravity_b_reference_as_state(
    env: ManagerBasedEnv,
    anchor_cfg: SceneEntityCfg = _DEFAULT_ANCHOR_REFERENCE_CFG,
) -> torch.Tensor:
    """参考运动 anchor 刚体自身系下的投影重力。`(num_envs, 3)`。

    ⚠️ 必须用 `link_quat_w`（**世界系**，由 `MotionReferenceManager` 还原），
    **不能**用缓冲里的 `link_quat_b`（相对 base）—— 后者在第一层就把绝对姿态丢掉了。
    """
    motion_reference: MotionReferenceManager = env.scene[anchor_cfg.name]
    quat_w = motion_reference.reference_frame.link_quat_w[:, 0, anchor_cfg.body_ids[0]]
    return _anchor_gravity_b(quat_w)


__all__ = [
    "anchor_gravity_b",
    "anchor_gravity_b_reference_as_state",
    "body_ang_vel_b",
    "body_ang_vel_b_reference_as_state",
    "body_lin_vel_b",
    "body_lin_vel_b_reference_as_state",
    "body_ori_b",
    "body_ori_b_reference_as_state",
    "body_pos_b",
    "body_pos_b_reference_as_state",
]
