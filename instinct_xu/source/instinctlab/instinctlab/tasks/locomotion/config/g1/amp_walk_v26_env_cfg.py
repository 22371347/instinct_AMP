"""V26：**σ（动作噪声）单变量** —— 从 V24@8000 热启动，把 σ 从 0.57 压到 0.20。

## 本轮唯一变量

环境侧、AMP 侧、奖励侧、`action_scale=1.0` **全部逐字沿用 V24**；只改 algo 侧两项：

| 项 | V24 | **V26** |
|---|---|---|
| `clip_max_std` | 0.6（**未 binding**，σ 实平衡在 0.572） | **0.20** |
| `clip_max_std_schedule` | —（不存在） | **300**（前 300 次 update 线性 0.57→0.20） |
| 起点 | 从零 | **V24 `model_8000.pt` 热启动**（`--resume`，AMP 输入维度相同 ⇒ 不丢判别器） |

## 依据（Step 0 的 σ 扫描，同一份 V24 权重、只缩放噪声，见 `/root/sigma_scan_v24.txt`）

| σ_eff | 噪声税/步 | 税/清洁任务 | `hf_acc_all` | `vel_err` | `hop_frac` | `in_band` | `single_stance` |
|---|---|---|---|---|---|---|---|
| 0.569（现状） | −5.261 | −94% | 133.9 | 4.809 | 0.762 | 0.385 | **0.063** |
| 0.300 | −1.466 | −26% | 64.5 | 2.617 | 0.639 | 0.463 | 0.118 |
| **0.200** | **−0.580** | **−10%** | 42.6 | 1.804 | **0.626** | 0.443 | **0.138** |
| 0.150 | −0.474 | −8% | 33.4 | 1.424 | 0.649 | 0.421 | 0.127 |
| 0.100 | −0.133 | −2% | 24.8 | 1.056 | 0.652 | 0.407 | 0.134 |
| 0（真值） | 0 | 0% | 13.1 | 0.556 | 0.680 | 0.375 | 0.121 |

⇒ σ=0.20 是**步态最优区**（`hop` 最低、`single_stance` 最高）且噪声税只剩 −10%；
而"完全去噪"反而不是最优（σ=0 时 `hop` 回升到 0.680）⇒ 选 **0.20**。

## 预登记判据（对照 V24 同期/终态）

1. `Episode_Reward/action_rate_l2` −1.24 → 期望 ~**−0.17**；`dof_acc_l2` −1.46 → ~**−0.16**；
   `track_default_pose_on_stand` −1.01 → ~**−0.35**；`dof_pos_limits` −0.42 → ~**−0.04**
   （斜率：前两项 ≈σ¹·⁷⁵，后两项 ≈σ¹·⁰）
2. `Mean reward` 由 **−0.48 → 期望 ≥ +3.5**（manager 口径）
3. `single_stance` **> 0.12**；`in_band` **> 0.44**；`hop_frac` 期望 < 0.65
4. `amp_style_std` 0.008 → 是否回到 **> 0.03**（若仍死 ⇒ **白化**而非 σ 才是元凶）
5. `progress_frac` 0.80 → 回落到 **~0.73**（去掉噪声虚警）
6. **风险判据**：若 500 轮内 `Mean reward` 不涨、步态不动 ⇒ 判定"σ 砍太多、探索不足"

## 为什么用 300 轮退火而不是一次性硬夹

σ 从 0.57 直接跳到 0.20 会让**奖励分布整体上移 ~4/步**（噪声税 −5.26 → −0.58）
⇒ value function 立刻错位一大截，PPO 会先把 V 重学一遍（浪费 100~300 轮且前期
advantage 全错）。线性退火把这个冲击摊到 300 轮。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v24_env_cfg import G1AmpWalkV24EnvCfg, G1AmpWalkV24EnvCfg_PLAY
from .amp_walk_v19d_env_cfg import G1AmpWalkV19DTeacherEnv

AMP_WALK_V26_RUN_NAME = (
    "G1AmpWalkV26MVPPlane_obs378_rew21_amp13body8frame_white_progress0.35"
    "_posdead0.8_accall_sigma0.20"
)


@configclass
class G1AmpWalkV26EnvCfg(G1AmpWalkV24EnvCfg):
    """训练用：与 V24 逐字相同（只有 run_name 不同）；σ 由 algo 侧的 cap 退火控制。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V26_RUN_NAME


@configclass
class G1AmpWalkV26EnvCfg_PLAY(G1AmpWalkV24EnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V26_RUN_NAME


class G1AmpWalkV26TeacherEnv(G1AmpWalkV19DTeacherEnv):
    """指令范围与 V19d/V20/V21/V23/V24 相同（全向四档）⇒ 期望值直接继承。"""


__all__ = [
    "AMP_WALK_V26_RUN_NAME",
    "G1AmpWalkV26EnvCfg",
    "G1AmpWalkV26EnvCfg_PLAY",
    "G1AmpWalkV26TeacherEnv",
]
