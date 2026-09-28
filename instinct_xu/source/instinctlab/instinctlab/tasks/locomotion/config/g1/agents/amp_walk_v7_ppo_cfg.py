"""AMP 行走教师 **v7** 的 PPO runner 配置 —— 从零重训的"收敛性修复"版。

v7 相对 v6 改 **2 个字段**（都是用户确认的路线 A）：

| 字段 | v6 | v7 | 理由 |
|---|---|---|---|
| `policy.init_noise_std` | 0.3 | **0.2** | 降低探索幅度：`scale=1.0` 下 σ 直接就是**弧度**（±0.3 rad ≈ ±17°），而关节设计尺度只有 0.4386（肩/肘）⇒ 有效扰动偏大。改 0.2 后 ±0.2 rad ≈ ±11° |
| `algorithm.discriminator_reward_coef` | 0.1 | **0.05** | ⭐ **AMP 奖励减半**。实测 AMP 占**真实训练回报 ~80%**（§11.12：AMP 3.76 / 任务 0.97 per-episode）⇒ 减半后约 **66%**，把主导权还给任务项 |

⚠️ **配套改动（必须一起）**：`clip_min_std` 按同一比例缩放 —— v3 定的是
「保持 AMP_mjlab 的**归一化** σ 下限 5%」的语义：`0.05 × (0.2 / 1.0) = 0.01`。
若不同步，σ 下限会从 5% 偷偷放宽到 7.5%。

⚠️ `discriminator_reward_coef` 同时出现在 lerp 修正公式里
（`aux = (1-λ)(style − r_task / c)`），改它是**自洽**的：
$R = \lambda r_{task} + (1-\lambda)c\,r_{amp}$ 仍然成立，只是风格项整体缩了一半。

⚠️ **本轮是「从零训练」，不带 `--resume`**：用户判断（并被数据支持）——
① v1 起就存在"整体缓慢向限位漂移"的趋势（每 500 轮约 +10~15%），
② 已经在坏吸引子里的策略靠续训拉不回来（v3 那次"跳跃 checkpoint"同一逻辑），
③ 改 σ 与 AMP 配比后，回报尺度也变了，续训的 critic 反而成了负担。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v6_ppo_cfg import G1AmpWalkV6PPORunnerCfg

#: v7 的探索噪声起点（绝对标准差，rad）
NOISE_STD_V7 = 0.2
#: = 0.05 × (0.2 / 1.0)：继续保持 AMP_mjlab「归一化下限 5%」的语义
CLIP_MIN_STD_V7 = 0.01
#: ⭐ AMP 奖励减半（v6 是 0.1）。实测减半后 AMP 占真实回报约 66%（原 ~80%）。
DISC_REWARD_COEF_V7 = 0.05


@configclass
class G1AmpWalkV7PPORunnerCfg(G1AmpWalkV6PPORunnerCfg):
    """v7：v6 全部配置（含 `joint_deviation_arms` / 镜像损失 / log 风格 / 站立关风格）
    + σ 0.3→0.2 + AMP 系数 0.1→0.05。"""

    def __post_init__(self):
        super().__post_init__()

        # --- 1. 探索幅度 ---
        self.policy.init_noise_std = NOISE_STD_V7
        self.algorithm.clip_min_std = CLIP_MIN_STD_V7

        # --- 2. AMP 奖励减半 ---
        # 实测（§11.12）：AMP per-episode 3.76、任务项 0.97 ⇒ AMP 占 ~80%。
        # 减半 ⇒ 0.0125·r_amp/new-step ⇒ 约 66%。
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V7

        self.run_name = f"v7_ns{NOISE_STD_V7}_ampc{DISC_REWARD_COEF_V7}_arms0.1"


__all__ = [
    "CLIP_MIN_STD_V7",
    "DISC_REWARD_COEF_V7",
    "NOISE_STD_V7",
    "G1AmpWalkV7PPORunnerCfg",
]
