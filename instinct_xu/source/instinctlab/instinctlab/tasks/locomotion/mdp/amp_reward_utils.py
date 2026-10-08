"""V23 的两个新数学件 —— **纯 torch、零 Isaac 依赖**，因此可在本地（无 GPU）单测。

为什么要单独成模块：这两段是 V23 的核心新逻辑，抽出来就能用
`instinct/tools/test_amp_reward_utils.py` 直接验证；`rewards_amp.py` 只负责取量、
调用、写日志。

## ① `ProgressGate`：方向投影 + EMA 低通（替代"实测瞬时幅值门"）

背景（三代 run 实测，见 `RESULTS_v19cd_v20_stand_probes.md`）：

* `motion_threshold = 0.15` 的"实测根速度**瞬时幅值**"门，在 V19c / V19d / V20 上的
  关闭率都只有 **0.46% / 0.46% / 0.63~0.95%** ⇒ 它几乎永远开着，从来没拦住过踏步；
* 幅值对**符号不敏感** ⇒ "摇得越猛越容易过门"，是**反向激励**（踏步时前后摇摆的
  瞬时幅值本来就有 0.3~1 m/s）；
* 踏步的**方向投影**（沿指令轴）有正有负、均值≈0；真走则恒正 ⇒ 投影 + 低通即可把
  "原地抬脚"与"真的在走"分开。

    v̂   = v_cmd,xy / max(‖v_cmd,xy‖, eps)
    p   = ⟨v_act,xy^yaw, v̂⟩                         # ⭐ **带符号**，不取 relu
    EMA ← (1−α)·EMA + α·p ,    α = 1 − exp(−dt/τ)      # dt=0.02 s, τ≈0.3 s
    progress = 1[ EMA > κ·‖v_cmd,xy‖ ]
               OR 1[ |ω_cmd| > eps 且 |ω_act| > κ_ω·|ω_cmd| ]   # 纯转向档放行

⚠️ **为什么投影必须带符号（不取 relu）** —— 本地单测抓出来的真问题：
   踏步的根速度是**前后交替**（≈ ±0.5 m/s 摇摆），取 `relu` 只砍掉负半边、
   前冲量却全留 ⇒ `EMA ≈ +0.25` 会**超过** `κ·‖v_cmd‖ = 0.21` ⇒ 踏步照样拿分。
   只有带符号才会有"前冲量 − 后撤量 ≈ 0"的抵消。
   （反向走 ⇒ EMA<0 ⇒ 也不放行；不对称摇摆（边踏边前漂）⇒ EMA≈净漂移速度，仍低于门线。）

⚠️ **reset 必须清零**：否则新回合头几步会带着上一段的 EMA ⇒ 门虚假开启。
   本模块不依赖 manager 的 `func.reset` 钩子，改用 `episode_length_buf <= 1` 掩码，
   在任何调用点都能工作（也便于本地单测）。

## ② `weighted_deadband_mean`：逐关节死区 + 内部权重

    cost = Σ_j w_j · relu(|e_j| − tol_j) / Σ_j w_j

* 死区 `tol_j`：允许"该关节应有的自然偏离"，超出部分才罚 ⇒ 站立时 ±3° 的微调不罚，
  而 15~30° 的"抬脚级"偏离照罚；
* 内部权重 `w_j`：**优先级**（不再兼任量纲平衡，因为分母是 Σw_j）⇒ 腿/踝可给 >1、
  手臂可给 <1，而不会把整项缩放。
"""

from __future__ import annotations

import math

import torch


def weighted_deadband_mean(
    err_abs: torch.Tensor,
    weight_row: torch.Tensor,
    tol_row: torch.Tensor,
) -> torch.Tensor:
    """逐关节死区 + 内部权重，按 `Σw_j` 归一。

    Args:
        err_abs: `(N, J)` **非负**误差（例如 `|q − q_def|` 或 `|dq|`）。
        weight_row: `(J,)` 内部权重（优先级）。
        tol_row: `(J,)` 逐关节死区（与 `err_abs` 同单位）。

    Returns:
        `(N,)`：`Σ_j w_j·relu(e_j − tol_j) / Σ_j w_j`（与 `err_abs` 同单位）。
    """
    pen = torch.relu(err_abs - tol_row) * weight_row
    return pen.sum(dim=1) / weight_row.sum().clamp_min(1e-6)


def name_weight_row(names: list[str], weights: dict[str, float] | None, default: float = 1.0) -> torch.Tensor:
    """按关节名构造 `(J,)` 权重行；未列出的关节用 `default`。"""
    if not weights:
        return torch.full((len(names),), float(default))
    return torch.tensor([float(weights.get(n, default)) for n in names])


class ProgressGate:
    """逐 env 的 `EMA(relu(v_act·v̂_cmd))` + 二值进度判据（见模块 docstring）。"""

    def __init__(self, num_envs: int, dt: float, tau: float, eps: float = 1e-3):
        self.dt = float(dt)
        self.tau = float(tau)
        self.eps = float(eps)
        self.alpha = 1.0 - math.exp(-self.dt / max(self.tau, 1e-6))
        self.num_envs = int(num_envs)
        self.ema = torch.zeros(self.num_envs)

    def _ensure(self, n: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.ema.shape[0] != n or self.ema.device != device or self.ema.dtype != dtype:
            # 首次调用 / 换 env 数或 device：按实际形状重建（只在第一 step 发生）
            self.ema = torch.zeros(n, device=device, dtype=dtype)

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """清零（`env_ids=None` = 全部）。"""
        if env_ids is None:
            self.ema.zero_()
        else:
            self.ema[env_ids] = 0.0

    @torch.no_grad()
    def reset_by_mask(self, mask: torch.Tensor) -> None:
        """按掩码清零 —— 在 reward 函数里用 `episode_length_buf <= 1` 调用。"""
        self.ema[mask] = 0.0

    @torch.no_grad()
    def __call__(
        self,
        vel_yaw_xy: torch.Tensor,
        command: torch.Tensor,
        omega_act: torch.Tensor,
        omega_cmd: torch.Tensor,
        kappa: float,
        omega_kappa: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """更新一步 EMA 并给出进度判据。

        Args:
            vel_yaw_xy: `(N, 2)` 实测根速度（**yaw 系**，与指令同系）。
            command: `(N, ≥2)` 速度指令 `[vx, vy, ωz, ...]`（yaw 系）。
            omega_act: `(N,)` 实测偏航角速度。yaw 系与世界系同值（只差绕 z 的旋转）。
            omega_cmd: `(N,)` 指令偏航角速度。
            kappa: `progress` 阈值（相对指令速度）：「实际前进速度 ≥ κ·‖v_cmd‖」。
            omega_kappa: 纯转向档的放行阈值（`|ω_act| ≥ κ_ω·|ω_cmd|`）。

        Returns:
            `(ema, progress, ratio)`：EMA、0/1 进度判据、`EMA / ‖v_cmd,xy‖`（κ 标定用）。
        """
        self._ensure(vel_yaw_xy.shape[0], vel_yaw_xy.device, vel_yaw_xy.dtype)
        cmd_xy = command[:, :2]
        cmd_speed = torch.norm(cmd_xy, dim=1)
        # 方向单位向量（cmd≈0 时未定义 ⇒ 用零向量，投影也自然为 0）
        safe = cmd_speed.clamp_min(self.eps)
        v_hat = cmd_xy / safe.unsqueeze(1)
        # ⚠️ **带符号**投影（**不**取 relu）：踏步的前后冲量必须互相抵消，
        #    否则前冲量会被留下 ⇒ EMA ≈ +0.25 也能过门（本地单测抓出的真问题）。
        proj = (vel_yaw_xy * v_hat).sum(dim=1)
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * proj

        moving = self.ema > float(kappa) * cmd_speed
        # 纯转向档：**必须要求 ω 指令本身非零**，否则站立档（ω_cmd=0、ω_act 有噪声）
        # 会被误放行（progress=1 ⇒ 站立档免罚）—— 这是本地单测专门守住的一条。
        turning = (
            (cmd_speed < self.eps)
            & (omega_cmd.abs() > self.eps)
            & (omega_act.abs() > float(omega_kappa) * omega_cmd.abs())
        )
        progress = (moving | turning).to(vel_yaw_xy.dtype)
        ratio = self.ema / cmd_speed.clamp_min(self.eps)
        return self.ema, progress, ratio


def get_progress_gate(env, tau: float, attr: str = "_amp_progress_gate") -> ProgressGate:
    """惰性创建/复用挂在 `env` 上的 `ProgressGate`（τ 变了就重建）。"""
    gate = getattr(env, attr, None)
    dt = float(getattr(env, "step_dt", 0.02))
    n = int(getattr(env, "num_envs", 1))
    if gate is None or abs(gate.tau - float(tau)) > 1e-9 or gate.num_envs != n:
        gate = ProgressGate(n, dt=dt, tau=float(tau))
        setattr(env, attr, gate)
    return gate
