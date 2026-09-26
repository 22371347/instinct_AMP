"""G1 矢状面镜像算子（y → −y）+ mirror loss 所需工具。

仿 AMP_mjlab `src/tasks/amp_loco/mdp/symmetry.py`，但对齐**本仓库**的观测/动作契约。

机制（AMP_mjlab v5 用来替代手写落脚对称惩罚）::

    π(M_obs(o)) ≈ M_act(π(o))

⚠️ **不是**"同一时刻左腿 == 右腿" —— 正常行走左右相位不同，同刻左右关节本就不同。
它约束的是"**左摆/右支撑**"与"**右摆/左支撑**"两个状态之间的对应关系。

为什么对本仓库安全
------------------
它是**策略层**约束，只要求"镜像输入 → 镜像输出"。G1 在矢状面反射下是自同构，
所以该约束**对任何参考数据都成立** ⇒ 不受"symmetric_augmentation（数据层）会把
侧移/弧线 clip 镜像成错误标签"这一顾虑影响。

映射规则
--------
* 极向量 ``[x,y,z] → [x, −y, z]``；轴向量 ``[wx,wy,wz] → [−wx, wy, −wz]``
* 左右关节交换 ``left_* ↔ right_*``
* 关节符号：**pitch 保号；roll / yaw 反号**
* 腰部（无左右对应）自映射：``yaw −``、``roll −``、``pitch +``（同一条符号规则）

★ 规则已被本仓库默认姿态验证（天然不动点检验）：
``left_shoulder_roll = +0.2`` / ``right_shoulder_roll = −0.2``，交换+反号后仍是
``+0.2 / −0.2`` ✓；``hip_roll`` 左右行程 ``[−0.524, 2.967]`` / ``[−2.967, 0.524]``
互为镜像 ✓ 佐证"roll 反号"。

height_scan 的排布（⚠️ 来源：IsaacLab 源码，已确认，不是猜）
-----------------------------------------------------------
``GridPatternCfg(size=(3.0, 1.6), resolution=0.1)``，``ordering`` 默认 ``"xy"``::

    indexing = "xy"                                     # torch.meshgrid(x, y, indexing="xy")
    x = arange(-1.5, 1.5+eps, 0.1)    # 31 点 → 传感器 local x（前向）
    y = arange(-0.8, 0.8+eps, 0.1)    # 17 点 → local y（侧向）
    grid_x, grid_y = meshgrid(x, y, indexing="xy")      # shape (17, 31)
    ray_starts[:, 0] = grid_x.flatten()                 # 行主序 ⇒ idx = iy*31 + ix

⇒ 527 = 31(前向) × 17(侧向)，**内层 x、外层 y**；x/y 都关于 0 对称 ⇒
**翻转外层索引 == 侧向镜像**：``scan.reshape(17, 31)[::-1, :].reshape(-1)``。

⚠️ 若以后改了 ``SCAN_SIZE_XY`` / ``SCAN_RESOLUTION`` / ``ordering``，
``SCAN_NY/NX`` 必须同步（下方 ``assert`` 会在维度不符时立刻报错）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# 策略观测（626 维）分段布局 —— 与 env cfg 的 obs 顺序一一对应
# ---------------------------------------------------------------------------
SCAN_NY, SCAN_NX = 17, 31                 # 侧向 × 前向（见模块 docstring）
SCAN_DIM = SCAN_NY * SCAN_NX              # 527

SL_SCAN = slice(0, 527)
SL_CMD = slice(527, 530)                  # (vx, vy, wz)
SL_LIN_VEL = slice(530, 533)              # body frame
SL_GRAV = slice(533, 536)
SL_ANG_VEL = slice(536, 539)              # body frame
SL_JP = slice(539, 568)                   # 29
SL_JV = slice(568, 597)                   # 29
SL_LA = slice(597, 626)                   # 29

OBS_DIM = 626


#: **pitch 类**关节：矢状面镜像**保号**；其余（roll / yaw 类）**反号**。
#: ⚠️ **不能只判 `"pitch" in name`** —— **knee 与 elbow 也是 pitch 类**，
#:    但名字里没有 "pitch"（实测被单测抓到：naive 规则下 `max|Δ|=1.338=2×0.669`）。
#: 判据来自默认姿态的**不动点**性质：默认姿态里 `knee`(L=R=+0.669)、
#: `elbow`(L=R=+0.6)、`hip_pitch`/`ankle_pitch`/`shoulder_pitch` 都**同号**
#: ⇒ 镜像必须保号才能保持不变。
PITCH_LIKE_SUBSTR = (
    "hip_pitch", "knee", "ankle_pitch", "shoulder_pitch", "elbow",
    "wrist_pitch", "waist_pitch",
)


def _is_pitch_like(name: str) -> bool:
    return any(s in name for s in PITCH_LIKE_SUBSTR)


def build_joint_mirror(joint_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """由关节名构造镜像 ``(perm, sign)``，使 ``(M q)[i] = sign[i] * q[perm[i]]``。

    ⚠️ **必须用运行时的 joint_names 现算，不能硬编码索引** ——
    本仓库动作/观测用的是 articulation（肩优先交错）序，
    与 npz 的"左腿→右腿→腰→左臂→右臂"序**不同**。

    满足：
      * 对合 ``M(M(q)) == q``
      * ``M(q_default) == q_default``（默认姿态是不动点）
    """
    idx = {n: i for i, n in enumerate(joint_names)}
    n = len(joint_names)
    perm = torch.empty(n, dtype=torch.long)
    sign = torch.empty(n, dtype=torch.float32)
    for i, name in enumerate(joint_names):
        if name.startswith("left_"):
            counterpart = "right_" + name[len("left_") :]
        elif name.startswith("right_"):
            counterpart = "left_" + name[len("right_") :]
        else:
            counterpart = name                      # 腰部自映射
        if counterpart not in idx:
            raise KeyError(f"关节 {name!r} 找不到镜像对应项 {counterpart!r}；joint_names={joint_names}")
        perm[i] = idx[counterpart]
        sign[i] = 1.0 if _is_pitch_like(name) else -1.0
    return perm, sign


@dataclass(frozen=True)
class MirrorSpec:
    """一次镜像所需的全部静态信息（可以放在 env 上，只构造一次）。"""

    joint_names: tuple[str, ...]
    perm: torch.Tensor      # (29,) long
    sign: torch.Tensor      # (29,) float

    @staticmethod
    def from_joint_names(joint_names: list[str]) -> "MirrorSpec":
        perm, sign = build_joint_mirror(joint_names)
        return MirrorSpec(tuple(joint_names), perm, sign)

    # --- 单段 ---
    def mirror_joints(self, q: torch.Tensor) -> torch.Tensor:
        """``q`` 最后一维 = 29（articulation 序）。

        ⚠️ `perm`/`sign` 构造时是 **CPU** 张量（与设备无关），而训练时 `q` 在 CUDA
        ⇒ 直接相乘会 `RuntimeError: Expected all tensors to be on the same device...`。
        **实测踩到：v4 第一次启动就崩在这一行。**
        这里按需搬到 `q.device`（`Tensor.to()` 在设备相同时返回 self，无拷贝开销）。
        对应的回归测试在 `instinct/tools/test_mirror_ops.py` 的第 [7] 节（CUDA）。
        """
        return self.sign.to(q.device) * q[..., self.perm.to(q.device)]

    def mirror_scan(self, scan: torch.Tensor) -> torch.Tensor:
        """(..., 527) → 侧向翻转。见模块 docstring 的排布推导。"""
        shape = scan.shape
        if shape[-1] != SCAN_DIM:
            raise ValueError(f"height_scan 最后一维应为 {SCAN_DIM}，收到 {shape[-1]}")
        g = scan.reshape(*shape[:-1], SCAN_NY, SCAN_NX)
        return g.flip(dims=(-2,)).reshape(shape)

    # --- 整段观测 ---
    def mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """``obs`` 最后一维 = 626（policy obs 契约）。返回同形状镜像观测。"""
        if obs.shape[-1] != OBS_DIM:
            raise ValueError(f"policy obs 最后一维应为 {OBS_DIM}，收到 {obs.shape[-1]}")
        out = obs.clone()
        out[..., SL_SCAN] = self.mirror_scan(obs[..., SL_SCAN])
        # 极 / 轴向量：y 反号 / (wx, wz) 反号
        out[..., SL_CMD] = obs[..., SL_CMD] * obs.new_tensor([1.0, -1.0, -1.0])
        out[..., SL_LIN_VEL] = obs[..., SL_LIN_VEL] * obs.new_tensor([1.0, -1.0, 1.0])
        out[..., SL_GRAV] = obs[..., SL_GRAV] * obs.new_tensor([1.0, -1.0, 1.0])
        out[..., SL_ANG_VEL] = obs[..., SL_ANG_VEL] * obs.new_tensor([-1.0, 1.0, -1.0])
        out[..., SL_JP] = self.mirror_joints(obs[..., SL_JP])
        out[..., SL_JV] = self.mirror_joints(obs[..., SL_JV])
        out[..., SL_LA] = self.mirror_joints(obs[..., SL_LA])
        return out

    def mirror_action(self, act: torch.Tensor) -> torch.Tensor:
        if act.shape[-1] != len(self.joint_names):
            raise ValueError(f"action 最后一维应为 {len(self.joint_names)}，收到 {act.shape[-1]}")
        return self.mirror_joints(act)


def mirror_loss(
    policy_mean_fn,
    obs: torch.Tensor,
    spec: MirrorSpec,
    *,
    coeff: float = 0.1,
) -> torch.Tensor:
    """``coeff · || μ_π(M_obs(o)) − M_act(μ_π(o)) ||²``（对 minibatch 求均值）。

    用**策略均值**而非采样动作 —— 降噪。AMP_mjlab 的 docstring 写作 ``π``，
    实现层面同为均值；``coeff = 0.1`` 与其 ``mirror_loss_coeff`` 一致。
    """
    lhs = policy_mean_fn(spec.mirror_obs(obs))
    rhs = spec.mirror_action(policy_mean_fn(obs))
    return coeff * torch.mean(torch.sum((lhs - rhs) ** 2, dim=-1))
