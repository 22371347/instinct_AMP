"""V33-S4 的算法配置：**AMP 口径与 S3 逐字相同** ⇒ 判别器可**保留**，只改奖励权重。

## 与 S3 的唯一差异：AMP 风格占比

| 项 | S3 | **S4** | 说明 |
|---|---|---|---|
| 判别器白化 | 无 | **无** | 逐字相同（S3 已验证：风格信号回到 V31 水平）|
| GP 空间 | `raw` | **`raw`** | 同上 |
| AMP 状态 | 396（含重力通道）| **396** | 同上 |
| 专家集 | `/root/amp_expert_states_396.pt` | 同上 | 同上 |
| `ckpt_manipulator` | `drop_amp_discriminator` | **`None`（保留判别器）** | ⭐ 口径没变 ⇒ **不用重学**，没有重学扰动 |
| `style_lerp` | 0.75（风格权重 0.25）| **0.70（0.30）** | 用户要求"增强 AMP 风格" |

环境侧的四项参数改动见 `amp_walk_v33_env_cfg.py::G1AmpWalkV33S4EnvCfg`
（取消 `feet_air_time`、增强 `track_lin_vel_xy_exp` / `track_ang_vel_z_exp` / `flat_orientation_l2`）。

## 起点

**续训**：从 S3 的 `model_12500.pt` 继续（同 task 族的权重变了 ⇒ 新建 task id，
但 AMP 口径与网络结构不变 ⇒ actor/critic/**discriminator** 三者都完整加载）。

⇒ 这是本项目第一次"只改奖励权重"的单变量实验（前几臂都伴随判别器重学）。

## 判据（行为优先 —— 用 MuJoCo 回放裁决，遥测只做参考）

1. **抬脚高度/单脚时间是否变自然**（用户上一版已看到 `air_time` 降低后的改良 ⇒ 归零后应更明显）；
2. **向右迈步是否长出来**（若续训后仍不对称，则不是"轮数不足"，下一轮再动指令集/左右分界限）；
3. **后倾是否缓解**（用户机制解释：后倾是为单脚稳定服务 ⇒ 取消该项后应有改善）；
4. 不退化闸门：`ep_len ≥ 500`、`action_abs_mean ≤ 0.62`、`Mean reward` 因权重变化**不可跨版比较**（只看相对趋势）。

## 早停点

- `model_13000.pt` / `model_13500.pt` 各存一份（`save_interval=500`）
- **S0 的教训**：不是训得越久越好 ⇒ 13000 与 13500 都回放后再决定停在哪
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v31_ppo_cfg import G1AmpWalkV31PPORunnerCfg
from .amp_walk_v33s_ppo_cfg import AMP_V33S_EXPERT_BUFFER

#: 风格占比：任务 0.70 / 风格 0.30（S3 是 0.75/0.25）。
#: ⚠️ 只动 `style_lerp`（= 重新分配权重），**不动** `discriminator_reward_coef`（V28 放大到 4.0 崩过）。
AMP_V33S4_STYLE_LERP = 0.70


@configclass
class G1AmpWalkV33S4PPORunnerCfg(G1AmpWalkV31PPORunnerCfg):
    """v33-S4：S3 的判别器口径（无白化 / raw GP / 396 / 同专家集）+ 风格占比 0.70 + 保留判别器续训。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- 判别器口径：与 S3 逐字相同（防御性 pop，确保没有 normalizer 残留）----
        dk = dict(self.algorithm.discriminator_kwargs)
        dk.pop("normalizer_class_name", None)
        dk.pop("normalizer_kwargs", None)
        self.algorithm.discriminator_kwargs = dk
        self.algorithm.amp_grad_pen_space = "raw"
        self.algorithm.amp_normalizer_expert_only = True
        # ---- 396 维专家集（维度必须与环境侧的 anchor 重力通道一致）--------------
        self.algorithm.amp_expert_buffer_path = AMP_V33S_EXPERT_BUFFER
        # ---- ⑤ 风格占比增强 ------------------------------------------------------
        self.algorithm.style_lerp = AMP_V33S4_STYLE_LERP
        # ---- ⭐ 判别器**保留**：口径没变，续训即可（前几臂都必须重学的唯一原因就是口径变了）----
        self.ckpt_manipulator = None

        self.run_name = "v33s4_noairtime_boostangvel_style70_keepdisc_from12500"


__all__ = ["AMP_V33S4_STYLE_LERP", "G1AmpWalkV33S4PPORunnerCfg"]
