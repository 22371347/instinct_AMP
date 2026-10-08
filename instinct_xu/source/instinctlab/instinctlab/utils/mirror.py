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
# 观测布局：**由观测契约推导**（v9 起的主路径）
# ---------------------------------------------------------------------------
# ⚠️ 历史教训（v9 第一次启动实测崩在这里）：原来把 626 契约**写死**在这几个常量里
#    （`OBS_DIM=626` / `SCAN_NY,NX=17,31` / `SL_*` 位置切片），v9 换成 378 后第一个
#    `PPO.update()` 就抛 `ValueError: policy obs 最后一维应为 626，收到 378`。
#    现在改成「**按项名**定规则 + 布局从 `obs_format` 累加推导」⇒ v1~v8(626) 与
#    v9(378) 共用同一份代码；以后增删观测项也不会再碰这里。
#
# 下面四个 `LEGACY_*` 常量**仅为向后兼容**保留（旧测试/旧调用码用它们），
# 训练路径不再读它们。
SCAN_NY, SCAN_NX = 17, 31                 # 旧 626 布局：侧向 × 前向
SCAN_DIM = SCAN_NY * SCAN_NX              # 527
OBS_DIM = 626

#: `height_scan` 的 (ny=侧向, nx=前向) 布局表，**键 = 射线数**。
#: ⚠️ 新增扫描尺寸时**必须在这里登记** —— 否则 `from_obs_format` 会 fail-fast，
#:    而不是静默用错误的 reshape 去镜像（那会变成一个极难发现的训练 bug）。
SCAN_LAYOUTS: dict[int, tuple[int, int]] = {
    31 * 17: (17, 31),   # v1~v8：size=(3.0, 1.6)、res=0.1 ⇒ 前向 31、侧向 17
    31 * 9: (9, 31),     # v9：   size=(3.0, 0.8)、res=0.1 ⇒ 前向 31、侧向 9
}

#: 向量类观测的镜像符号（极向量 `[x,y,z]→[x,−y,z]`；轴向量 `[wx,wy,wz]→[−wx,wy,−wz]`）。
#: 键 = **观测项名**（不是位置）。
VECTOR_TERM_SIGNS: dict[str, tuple[float, float, float]] = {
    "velocity_command": (1.0, -1.0, -1.0),   # (vx, vy, wz)
    "base_lin_vel": (1.0, -1.0, 1.0),        # body frame
    "projected_gravity": (1.0, -1.0, 1.0),
    "base_ang_vel": (-1.0, 1.0, -1.0),       # body frame
}
#: 关节类项名（最后一维 = 29，走 `mirror_joints`）
JOINT_TERMS: tuple[str, ...] = ("joint_pos_rel", "joint_vel", "last_action")
#: 高程图项名（走 `mirror_scan`）
SCAN_TERM: str = "height_scan"

#: 旧 626 布局的段表（仅 `from_joint_names` 兼容入口使用）
LEGACY_SEGMENTS: tuple[tuple[str, slice], ...] = (
    (SCAN_TERM, slice(0, 527)),
    ("velocity_command", slice(527, 530)),
    ("base_lin_vel", slice(530, 533)),
    ("projected_gravity", slice(533, 536)),
    ("base_ang_vel", slice(536, 539)),
    ("joint_pos_rel", slice(539, 568)),
    ("joint_vel", slice(568, 597)),
    ("last_action", slice(597, 626)),
)


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
    """一次镜像所需的全部静态信息（可以放在 env 上，只构造一次）。

    ⚠️ 布局字段（`obs_dim` / `scan_ny` / `scan_nx` / `segments`）**不再写死**：
    由观测契约构造（见 `from_obs_format`）⇒ **同一份代码同时支持 v1~v8 的 626
    （527 射线）与 v9 的 378（279 射线）**。`from_joint_names` 保留为
    「按旧 626 布局」的兼容入口。
    """

    joint_names: tuple[str, ...]
    perm: torch.Tensor      # (29,) long
    sign: torch.Tensor      # (29,) float
    obs_dim: int = OBS_DIM
    scan_ny: int = SCAN_NY
    scan_nx: int = SCAN_NX
    #: `((项名, slice), ...)`，顺序 = 观测拼装顺序
    segments: tuple[tuple[str, slice], ...] = LEGACY_SEGMENTS

    @staticmethod
    def from_joint_names(joint_names: list[str]) -> "MirrorSpec":
        perm, sign = build_joint_mirror(joint_names)
        return MirrorSpec(tuple(joint_names), perm, sign)

    @staticmethod
    def from_obs_format(joint_names: list[str], policy_format) -> "MirrorSpec":
        """按**观测契约**（`obs_format['policy']`，有序 dict：项名 → shape）构造。

        ⚠️ v9 起的主路径：布局完全由契约推导 ⇒ 以后改扫描尺寸/增删项，
        只要在 `SCAN_LAYOUTS` / 规则表里登记过，镜像就自动跟上。
        不认识的项名直接 **fail-fast**（宁可启动报错，也不要静默错镜像）。
        """
        perm, sign = build_joint_mirror(joint_names)
        segments: list[tuple[str, slice]] = []
        offset = 0
        scan_dim: int | None = None
        for name, shape in dict(policy_format).items():
            dim = 1
            for s in shape:
                dim *= int(s)
            if name == SCAN_TERM:
                if scan_dim is not None:
                    raise ValueError(f"观测里出现了多个 {SCAN_TERM!r} 项")
                scan_dim = dim
            elif name not in VECTOR_TERM_SIGNS and name not in JOINT_TERMS:
                raise KeyError(
                    f"镜像算子不认识观测项 {name!r}（dim={dim}）⇒ 请在 `mirror.py` 的 "
                    f"VECTOR_TERM_SIGNS / JOINT_TERMS / SCAN_TERM 里登记它的镜像规则。"
                    f"已知：scan={SCAN_TERM!r}、向量={sorted(VECTOR_TERM_SIGNS)}、"
                    f"关节={list(JOINT_TERMS)}"
                )
            segments.append((str(name), slice(offset, offset + dim)))
            offset += dim
        if scan_dim is None:
            raise KeyError(f"观测里找不到 {SCAN_TERM!r} 项")
        if scan_dim not in SCAN_LAYOUTS:
            raise KeyError(
                f"{SCAN_TERM} 有 {scan_dim} 条射线，但 `SCAN_LAYOUTS` 里没登记它的 (ny, nx)。"
                f"已知 {SCAN_LAYOUTS}；新增扫描尺寸时请在此登记（ny=侧向、nx=前向）"
            )
        ny, nx = SCAN_LAYOUTS[scan_dim]
        return MirrorSpec(tuple(joint_names), perm, sign, offset, ny, nx, tuple(segments))

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
        """(..., 527 或 279) → 侧向翻转（reshape 成 (ny, nx) 后翻转 **ny** 轴）。

        ⚠️ v9 的扫描框**中心前移了 1.5 m**，但前后方向（nx）不参与镜像、
        左右方向仍关于躯干对称 ⇒ **翻转规则完全不变**，只是 ny 从 17 变 9。
        """
        shape = scan.shape
        expect = self.scan_ny * self.scan_nx
        if shape[-1] != expect:
            raise ValueError(
                f"height_scan 最后一维应为 {expect}（{self.scan_ny}×{self.scan_nx}），"
                f"收到 {shape[-1]}"
            )
        g = scan.reshape(*shape[:-1], self.scan_ny, self.scan_nx)
        return g.flip(dims=(-2,)).reshape(shape)

    # --- 整段观测 ---
    def mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """`obs` 最后一维 = `self.obs_dim`（v1~v8: 626；v9: 378）。返回同形状镜像观测。"""
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"policy obs 最后一维应为 {self.obs_dim}，收到 {obs.shape[-1]}")
        out = obs.clone()
        for name, sl in self.segments:
            if name == SCAN_TERM:
                out[..., sl] = self.mirror_scan(obs[..., sl])
            elif name in JOINT_TERMS:
                out[..., sl] = self.mirror_joints(obs[..., sl])
            else:
                # 极 / 轴向量：y 反号 / (wx, wz) 反号
                out[..., sl] = obs[..., sl] * obs.new_tensor(VECTOR_TERM_SIGNS[name])
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
