"""方向分档速度指令：DirectionalVelocityCommand。

为什么需要它
------------
IsaacLab 的 `UniformVelocityCommand` 对 $(v_x, v_y, \\omega_z)$ 三轴**独立连续均匀**采样，
因此 $v_x$ 精确为 0 的概率为零 ⇒ 训练中几乎不出现「纯侧向」命令（$v_x=0, |v_y|$ 大），
而部署时「左右直移」恰恰就是 $v_x=0$。策略没见过纯侧向命令 → 侧向站在原地不动。

本类按概率强制生成几类**真实部署中会出现**的命令：

| 分支 | 概率 | 内容 |
|---|---|---|
| 低速 | `prob_low` | 线速度幅值 $\\le$ `low_vel`、$\\omega_z=0$；其中 `zero_frac_in_low` 比例给**精确 (0,0,0)** |
| 纯侧向 | `prob_side` | $v_x \\equiv 0$，$v_y$ 全范围 ⇒ 左右直移 |
| 纯前后 | `prob_lin` | $v_y \\equiv 0$；其中 `prob_lin_fwd` 比例给「持续直线前向」（$v_x$ 正且快、$\\omega_z\\approx0$） |
| 持续后退 | `prob_back` | $v_y=0$，$v_x$ 取 `back_lin_vel_range`（负、幅度较大），$\\omega_z\\approx0$ |
| 均匀斜向 | 剩余 | 即父类原逻辑 |

实现策略：**最小假设**
----------------------
`_resample_command` 先调用 `super()._resample_command(env_ids)`——由父类负责
默认均匀采样、`is_standing_env` / `is_heading_env` / `heading_target` 的维护；
然后本类只**覆盖** `self.vel_command_b` 中属于各模式的行。

这样本类只依赖父类的两个东西：
1. 内部缓冲 `self.vel_command_b`（IsaacLab `UniformVelocityCommand` 的公开实现细节）
2. `self.cfg.ranges.{lin_vel_x, lin_vel_y, ang_vel_z}`

**刻意不依赖** `is_standing_env` / `is_heading_env` / `heading_target` / `self.robot`
等其它内部字段——因此即使 IsaacLab 版本对它们做了改动，本类也不会失效。
`__init__` 里另有 fail-fast 契约校验，把「属性不存在」这类问题变成**明确的报错信息**，
而不是训练跑起来之后才出现的 `AttributeError`。

来源与差异
----------
设计参照 `AMP_mjlab/src/tasks/amp_loco/mdp/velocity_command.py`。**刻意不移植**的部分：
- `init_velocity_prob`（把命令速度直接写入 root state）：是 mjlab 专有字段与 API
  （`root_link_pos_w` / `write_root_state_to_sim`），且属于激进的训练技巧，非必需；
- `quat_apply` 世界系换算：同上，本类不需要。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils import configclass

# IsaacLab 的 mdp 包同时导出命令类与其配置类（`flat_env_cfg.py` 已用到 Cfg 那半）。
# 这里用 try/except 把「版本不满足假设」变成可读的错误信息。
try:
    from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
except ImportError as exc:  # pragma: no cover - 依赖 IsaacLab 版本
    raise ImportError(
        "DirectionalVelocityCommand 需要 isaaclab.envs.mdp 同时暴露 "
        "`UniformVelocityCommand` 与 `UniformVelocityCommandCfg`；当前 IsaacLab 版本不满足该假设。"
        f"原始错误：{exc}"
    ) from exc

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DirectionalVelocityCommand(UniformVelocityCommand):
    """带纯方向 / 低速分档的均匀速度指令生成器。"""

    cfg: "DirectionalVelocityCommandCfg"

    def __init__(self, cfg: "DirectionalVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._validate_directional_contract()

    # -- 契约校验 -----------------------------------------------------------------
    def _validate_directional_contract(self) -> None:
        """把「静默跑错」变成「启动即报错」。

        两类检查：
        1. 依赖的父类内部缓冲是否存在（版本兼容）；
        2. 两项前提假设——因为本类会**覆盖**父类采样结果，若父类的
           standing / heading 机制同时生效，两者语义会互相打架：
           - `rel_standing_envs > 0`：父类会把部分环境命令清零，本类又会写回非零值；
           - `heading_command=True`：`_update_command` 会用航向误差重算 $\\omega_z$，
             使本类采样的 $\\omega_z$ 失效。
        """
        errors: list[str] = []
        if not hasattr(self, "vel_command_b"):
            errors.append(
                "父类 UniformVelocityCommand 没有 `vel_command_b` 缓冲——"
                "本实现依赖该内部名称，请核对 IsaacLab 版本"
            )
        if self.cfg.heading_command or self.cfg.rel_heading_envs != 0.0:
            errors.append(
                f"要求 heading_command=False 且 rel_heading_envs=0，"
                f"实际 heading_command={self.cfg.heading_command}, "
                f"rel_heading_envs={self.cfg.rel_heading_envs}"
            )
        if self.cfg.rel_standing_envs != 0.0:
            errors.append(
                f"要求 rel_standing_envs=0（站立语义交给 prob_low 分支），"
                f"实际为 {self.cfg.rel_standing_envs}"
            )
        if errors:
            raise RuntimeError(
                "DirectionalVelocityCommand 契约校验失败：\n  - " + "\n  - ".join(errors)
            )

    # -- 采样 ---------------------------------------------------------------------
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """先让父类填好默认均匀命令，再按模式覆盖需要分档的环境。"""
        super()._resample_command(env_ids)

        n = int(env_ids.numel())
        if n == 0:
            return
        dev = env_ids.device
        r = self.cfg.ranges

        def uni(k: int, lo: float, hi: float) -> torch.Tensor:
            return torch.empty(k, device=dev).uniform_(lo, hi)

        p_low = float(self.cfg.prob_low)
        p_side = p_low + float(self.cfg.prob_side)
        p_lin = p_side + float(self.cfg.prob_lin)
        p_back = p_lin + float(self.cfg.prob_back)

        # 抽签：m < p_low -> 低速；[p_low, p_side) -> 纯侧向；
        #       [p_side, p_lin) -> 纯前后；[p_lin, p_back) -> 持续后退；
        #       m >= p_back -> 保留父类的均匀斜向
        m = torch.rand(n, device=dev)

        # --- 1) 低速：线速度 <= low_vel，wz = 0 ---------------------------------
        low_mask = m < p_low
        if low_mask.any():
            idx = env_ids[low_mask]
            k = int(idx.numel())
            self.vel_command_b[idx, 0] = 0.0
            self.vel_command_b[idx, 1] = 0.0
            self.vel_command_b[idx, 2] = 0.0
            # 其中一部分给精确 (0,0,0)：对齐「摇杆回中 = 立定」
            nonzero = idx[~(torch.rand(k, device=dev) < float(self.cfg.zero_frac_in_low))]
            if nonzero.numel() > 0:
                k2 = int(nonzero.numel())
                mag = uni(k2, 0.0, float(self.cfg.low_vel))
                theta = uni(k2, 0.0, 2.0 * torch.pi)
                self.vel_command_b[nonzero, 0] = mag * torch.cos(theta)
                self.vel_command_b[nonzero, 1] = mag * torch.sin(theta)

        # --- 2) 纯侧向：vx == 0 ---------------------------------------------------
        side_mask = (m >= p_low) & (m < p_side)
        if side_mask.any():
            idx = env_ids[side_mask]
            k = int(idx.numel())
            self.vel_command_b[idx, 0] = 0.0
            self.vel_command_b[idx, 1] = uni(k, *r.lin_vel_y)
            self.vel_command_b[idx, 2] = uni(k, *r.ang_vel_z)

        # --- 3) 纯前后：vy == 0 ---------------------------------------------------
        lin_mask = (m >= p_side) & (m < p_lin)
        if lin_mask.any():
            idx = env_ids[lin_mask]
            k = int(idx.numel())
            fwd = torch.rand(k, device=dev) < float(self.cfg.prob_lin_fwd)
            fwd_idx = idx[fwd]
            if fwd_idx.numel() > 0:
                kf = int(fwd_idx.numel())
                self.vel_command_b[fwd_idx, 0] = uni(kf, *self.cfg.straight_lin_vel_range)
                self.vel_command_b[fwd_idx, 1] = 0.0
                self.vel_command_b[fwd_idx, 2] = uni(kf, *self.cfg.straight_ang_vel_range)
            other_idx = idx[~fwd]
            if other_idx.numel() > 0:
                ko = int(other_idx.numel())
                self.vel_command_b[other_idx, 0] = uni(ko, *r.lin_vel_x)
                self.vel_command_b[other_idx, 1] = 0.0
                self.vel_command_b[other_idx, 2] = uni(ko, *r.ang_vel_z)

        # --- 4) 持续直线后退：vy = 0，vx < 0，wz ~ 0 ------------------------------
        # 均匀采样下后退会被前向范围稀释，策略容易用「双脚同时蹬地后跳」蹭跟踪奖励；
        # 给后退专属且持续的配额，逼出真正交替的后退步态。
        back_mask = (m >= p_lin) & (m < p_back)
        if back_mask.any():
            idx = env_ids[back_mask]
            k = int(idx.numel())
            self.vel_command_b[idx, 0] = uni(k, *self.cfg.back_lin_vel_range)
            self.vel_command_b[idx, 1] = 0.0
            self.vel_command_b[idx, 2] = uni(k, *self.cfg.back_ang_vel_range)


@configclass
class DirectionalVelocityCommandCfg(UniformVelocityCommandCfg):
    """`DirectionalVelocityCommand` 的配置。

    默认值取自 `PLAN_amp_walk.md` §3.4 实测的参考动作速度包络
    （$v_x\\in[-1.07,+1.69]$、$v_y\\in[-0.48,+0.55]$ m/s、$\\omega_z\\in[-44.2°,+57.0°]$/s），
    并留了余量。**命令范围不应超出参考数据集包络**，否则该档位缺乏参考密度，
    判别器会把对应姿态判为「不像参考」而抑制。
    """

    class_type: type = DirectionalVelocityCommand

    # -- 分档概率（四项之和须 <= 1，剩余部分为均匀斜向）------------------------
    prob_low: float = 0.15
    """低速命令概率（三轴均被压到 <= `low_vel`），覆盖「停稳 / 回 default_pos」。"""
    prob_side: float = 0.25
    """纯侧向命令概率（$v_x\\equiv0$），覆盖左右直移。"""
    prob_lin: float = 0.25
    """纯前后命令概率（$v_y\\equiv0$），覆盖前后直行与转弯行走。"""
    prob_back: float = 0.10
    """持续直线后退概率（$v_y=0$、$v_x<0$、$\\omega_z\\approx0$）。"""

    # -- 低速分支 ---------------------------------------------------------------
    low_vel: float = 0.1
    """低速命令的最大幅值（m/s 或 rad/s）。"""
    zero_frac_in_low: float = 0.6
    """低速分支内给**精确 (0,0,0)** 的比例，其余为 <= `low_vel` 的随机小速度。"""

    # -- 持续直线前向 -----------------------------------------------------------
    prob_lin_fwd: float = 0.6
    """`prob_lin` 分支内给「持续直线前向」的比例。"""
    straight_lin_vel_range: tuple[float, float] = (0.5, 1.2)
    """持续直线前向的 $v_x$ 范围 (m/s)。"""
    straight_ang_vel_range: tuple[float, float] = (-0.1, 0.1)
    """持续直线前向的 $\\omega_z$ 容差 (rad/s)，≈0 表示不转向。"""

    # -- 持续直线后退 -----------------------------------------------------------
    back_lin_vel_range: tuple[float, float] = (-1.0, -0.5)
    """持续直线后退的 $v_x$ 范围 (m/s)：取负、幅度偏大。"""
    back_ang_vel_range: tuple[float, float] = (-0.1, 0.1)
    """持续直线后退的 $\\omega_z$ 容差 (rad/s)。"""
