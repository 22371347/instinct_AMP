"""V33-S5 的算法配置：**AMP 口径与 S3/S4 逐字相同** ⇒ 判别器可**保留**，只改奖励权重。

## 与 S4 的唯一差异：风格占比回到 0.75

| 项 | S3 | S4 | **S5** | 说明 |
|---|---|---|---|---|
| 判别器白化 | 无 | 无 | **无** | 逐字相同 |
| GP 空间 | `raw` | `raw` | **`raw`** | 同上 |
| AMP 状态 | 396（含重力通道）| 396 | **396** | 同上 |
| 专家集 | `/root/amp_expert_states_396.pt` | 同上 | **同上** | 同上 |
| `ckpt_manipulator` | `drop_amp_discriminator` | `None`（保留）| **`None`（保留）** | 口径没变 ⇒ 不用重学 |
| `style_lerp` | 0.75 | 0.70 | **0.75** | ⭐ 撤销 S4 的风格增强 |

## 为什么把 `style_lerp` 改回 0.75（而不是继续 0.70）

实测跨臂（末点 `Step_Monitor/tracking/tracking_vxy_rmse`，越低越好）：

| 臂 | 环境侧要点 | `vxy_rmse` | `style_reward` | `tilt` | `ep_len` |
|---|---|---|---|---|---|
| S0@12490 | air4 / λ0.75 / 风格被打死 | **0.222** | 0.058 | 7.1° | 605 |
| S3@12490 | air2 / λ0.75 | 0.311 | 0.241 | 15.8° | 588 |
| S4@13490 | air0 / **λ0.70** | **0.404** | 0.296 | 19.2° | 509 |

⇒ 风格项是**跟踪项的直接竞争者**：风格占比 0.25→0.30（λ 0.75→0.70）与线速度跟踪恶化同现。
用户本轮的诉求是"速度和角度都跟踪更好" ⇒ 把 `style_lerp` 拉回 0.75。

⚠️ 只动 `style_lerp`（= 重新分配权重），**不动** `discriminator_reward_coef`
（V28 把它放大到 4.0 崩过）。

环境侧改动（air_time 2.0→1.0、保留 `track_lin_vel_xy_exp=3.0` / `track_ang_vel_z_exp=1.5`）
见 `amp_walk_v33_env_cfg.py::G1AmpWalkV33S5EnvCfg`。

## 起点

**续训**：从 **S3 的 `model_12500.pt`** 继续（用户指令：回退到 S4 退化之前）。
AMP 口径与网络结构不变 ⇒ actor / critic / **discriminator** 三者都完整加载。

## 判据（行为优先 —— 用 MuJoCo 回放裁决，遥测只做参考）

1. `tracking/vxy_rmse` 从 0.311（S3）继续下降；`vx_high_vxy_rmse`（S4 已到 0.63）明显回落；
2. `wz_rmse` **不退化**（S4 已把它从 0.83~0.90 压到 0.69~0.77）；
3. `air_time_*` 三条节奏遥测回来（weight=1.0 ⇒ term 被调用）⇒ 看单支撑/小跳/落带比例；
4. 红线：`ep_len ≥ 500`、`action_abs_mean ≤ 0.62`；`Mean reward` 因权重变化**不可跨版比较**。

## 早停点

- `model_13000.pt` / `model_13500.pt` 各存一份（`save_interval=500`）
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v31_ppo_cfg import G1AmpWalkV31PPORunnerCfg
from .amp_walk_v33s_ppo_cfg import AMP_V33S_EXPERT_BUFFER

#: 风格占比：任务 0.75 / 风格 0.25（撤销 S4 的 0.70/0.30）。
#: ⚠️ 只动 `style_lerp`，**不动** `discriminator_reward_coef`（V28 放大到 4.0 崩过）。
AMP_V33S5_STYLE_LERP = 0.75


@configclass
class G1AmpWalkV33S5PPORunnerCfg(G1AmpWalkV31PPORunnerCfg):
    """v33-S5：S3/S4 的判别器口径（无白化 / raw GP / 396 / 同专家集）+ 风格占比回 0.75 + 保留判别器续训。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- 判别器口径：与 S3/S4 逐字相同（防御性 pop，确保没有 normalizer 残留）----
        dk = dict(self.algorithm.discriminator_kwargs)
        dk.pop("normalizer_class_name", None)
        dk.pop("normalizer_kwargs", None)
        self.algorithm.discriminator_kwargs = dk
        self.algorithm.amp_grad_pen_space = "raw"
        self.algorithm.amp_normalizer_expert_only = True
        # ---- 396 维专家集（维度必须与环境侧的 anchor 重力通道一致）--------------
        self.algorithm.amp_expert_buffer_path = AMP_V33S_EXPERT_BUFFER
        # ---- 风格占比：撤销 S4 的增强，回到 S3 的 0.75 --------------------------
        self.algorithm.style_lerp = AMP_V33S5_STYLE_LERP
        # ---- ⭐ 判别器**保留**：口径没变，续训即可 ---------------------------------
        self.ckpt_manipulator = None

        self.run_name = "v33s5_air1_boosttrack_style75_keepdisc_from12500"


__all__ = ["AMP_V33S5_STYLE_LERP", "G1AmpWalkV33S5PPORunnerCfg"]
