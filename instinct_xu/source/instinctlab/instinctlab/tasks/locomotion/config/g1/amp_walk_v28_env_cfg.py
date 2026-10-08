"""V28：**放大 AMP 风格项的权重** —— 让风格第一次有能力和任务项"掰手腕"。

## 诊断（为什么 V27 有活的判别器却没有类人风格）

我们的奖励混合是 lerp 形式（`algorithms.py` 的 `compute_auxiliary_reward`）：

    r_total = λ·r_task + (1−λ)·coef·style        （λ = style_lerp = 0.75, coef = discriminator_reward_coef）

当前 λ=0.75、**coef=1.0** ⇒ **风格项的有效权重只有 (1−λ)·coef = 0.25**。
而 V27 实测 `amp_style_std = 0.058`（风格奖励的**跨状态方差** = 真正能引导策略的"排序信号"）
⇒ 风格项对策略的作用量 ≈ 0.25 × 0.058 = **0.0145 / 步**，
比任务项（`track_lin_vel_xy_exp +1.2`、`feet_air_time +1.1`、罚项 O(0.1~1)）小 **10~70 倍**。

⇒ **判别器就算完全准确，风格项也没有能力改变步态。** 这与"V27 练到 10000 轮仍无类人风格"吻合。

## 本轮唯一变量

| 项 | V27 | **V28** |
|---|---|---|
| `discriminator_reward_coef` | 1.0 | **4.0** ⇒ 风格项权重 0.25 → **1.0（4×）** |
| `style_lerp` (λ) | 0.75 | **不动**（⇒ 任务项权重仍是 0.75，**不被削弱**）|
| 判别器口径（去白化）/ σ=0.20 / 奖励表 / AMP 观测（13body×8frame）| — | **全不动** |
| 起点 | V26@9000 | **V27 最新 ckpt**；判别器 `drop_amp_discriminator` 从零重学 |

**为什么只动 coef、不动 λ**：`r = λ·r_task + (1−λ)·coef·style`。
- 只提高 `coef` ⇒ 风格项 ×4，**任务项权重不变**（λ 仍 0.75）✓ 单变量、无副作用；
- 若改 λ（0.75→0.5）会同时把任务项权重砍到 0.5 ⇒ 两个变量纠缠，且会牺牲已学到的步态。

## 预登记判据

1. **风格项影响力**：`amp_style_reward` 的跨 env 标准差（`Step/amp_style_std`）不变（≈0.058），
   但**有效作用量**从 0.0145 → **0.058**（4×）；`discriminator_reward` 项的量级应 ×4。
2. **策略必须开始"为风格让路"**（这是本轮的核心可观测）：
   - `air_time_single_stance` 上升（单支撑相变长 = 更接近参考行走）、`air_time_lift_frac` 上升；
   - `hop_frac` 下降（< 0.72）；
   - `track_default_pose_on_stand`、`track_lin_vel_xy_exp` **不该崩**（守住 ≥0.9×）。
3. **判别器**（仍是 V27 的去白化口径）：`amp_d` 会再次饱和（AMP 里很正常），**这不是判据**；
   真正的判据是 **`amp_style_std` 是否保持 >0.03** 以及 `amp_d_ref − amp_d` 的 gap 是否 ≈2。
4. **风险判据**：若 500 轮内 `Mean reward` 从 ~+45 掉到 <+20，或 `hf_acc_all` 明显上升
   ⇒ 判定"风格项在带坏步态/重引入高频动作" ⇒ 回退 coef 到 2.0。
5. 若"风格项权重 ×4 后步态仍无变化" ⇒ 说明问题不在权重而在**信号**（判别器给不出与风格相关的
   排序）⇒ 下一轮（V29）转向结构对齐：①判别器输入改回单帧对 ②专家 batch 从运动数据**随机抽帧**
   （解耦相位）③恢复判别器正则（GP 1→5、weight_decay 1e-5→3e-4、修 `logit_weight_decay_coef` 键名 bug）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v27_env_cfg import (
    AMP_WALK_V27_RUN_NAME,
    G1AmpWalkV27EnvCfg,
    G1AmpWalkV27EnvCfg_PLAY,
    G1AmpWalkV27TeacherEnv,
)

AMP_WALK_V28_RUN_NAME = (
    "G1AmpWalkV28MVPPlane_obs378_rew21_amp13body8frame_nodrop_white_sigma0.20_ampcoef4"
)


@configclass
class G1AmpWalkV28EnvCfg(G1AmpWalkV27EnvCfg):
    """训练用：与 V27 逐字相同（奖励/AMP 观测/指令集/σ 都不动）；只有 run_name 不同。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V28_RUN_NAME


@configclass
class G1AmpWalkV28EnvCfg_PLAY(G1AmpWalkV27EnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V28_RUN_NAME


class G1AmpWalkV28TeacherEnv(G1AmpWalkV27TeacherEnv):
    """指令范围与前几代相同（全向四档）⇒ 期望值直接继承。"""


__all__ = [
    "AMP_WALK_V28_RUN_NAME",
    "G1AmpWalkV28EnvCfg",
    "G1AmpWalkV28EnvCfg_PLAY",
    "G1AmpWalkV28TeacherEnv",
]
