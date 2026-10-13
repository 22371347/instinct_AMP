"""V28：**放大 AMP 风格项的权重** —— 让风格第一次有能力和任务项"掰手腕"。

## ⚠️ 本层对**今天**仍有影响的部分

* ⭐⭐ 本层的唯一变量（`discriminator_reward_coef` **×4**）**今天仍生效**（与 T1k/T1l 的算法侧一致）
  —— 具体值见下表 ① 的「算法侧」指针。
* 本层的**环境侧零改动**（与 V27 逐字相同，只换 `run_name`）。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | ⭐ 唯一变量：`discriminator_reward_coef` 1.0 → **4.0**（风格项有效权重 `(1−λ)·coef` 0.25 → **1.0**，即 ×4）；`style_lerp`(λ) **不动**〔算法侧〕| 见 `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V28-G1-v0` 的**「算法侧」节**（今天是否仍生效，对照 `Instinct-Amp-Walk-T1l-G1-v0` 同节）|
| ② | 判别器口径（去白化）/ 策略 σ / 奖励表 / AMP 观测（13body×8frame）**全不动** | "8 帧" **已被 V29 覆盖** → **2 帧**（`AMP_V29_STATE_HISTORY=2`，今天）；其余见各自层 |
| ③ | 起点 = V27 最新 ckpt；判别器 `drop_amp_discriminator` **从零重学** | 已不适用（T1j 热启动自 T1a2 的 `model_7000.pt`）|

## 推理与判读（已归档，勿在本文件重复）

* **诊断（为什么 V27 有活的判别器却没有类人风格）**：风格项的有效权重只有 `(1−λ)·coef = 0.25`；
  V27 实测 `amp_style_std`（风格奖励的**跨状态方差** = 真正能引导策略的"排序信号"）只有 **0.058**
  ⇒ 风格项对策略的作用量 ≈ 0.25 × 0.058 = **0.0145 / 步**，比任务项小 **10~70 倍**
  ⇒ **判别器就算完全准确，风格项也没有能力改变步态**（与"V27 练到 10000 轮仍无类人风格"吻合）
  → §A3（"风格权重 ×4 能改善风格" —— 实测**无效**）+ `RESULTS_amp_v22_whitening.md`（白化后 `style_std` ×2.7）
* ⭐ **为什么只动 `coef`、不动 λ**：`r = λ·r_task + (1−λ)·coef·style`；只提 `coef` ⇒ 风格项 ×4 而
  **任务项权重不变**（λ 仍 0.75 ⇒ 单变量、无副作用）；改 λ 会同时把任务项权重砍到 0.5
  ⇒ 两个变量纠缠，且会牺牲已学到的步态 → §B3（"同一数值 ≠ 同一强度"：λ 的陷阱）
* 预登记判据 1~5：① 风格项**有效作用量** 0.0145 → **0.058**（4×），`discriminator_reward` 项量级应 ×4；
  ② 策略必须开始"为风格让路"（`air_time_single_stance` / `air_time_lift_frac` 升、`hop_frac` 降；
  `track_default_pose_on_stand` / `track_lin_vel_xy_exp` **不该崩**，守 ≥0.9×）；③ 判别器 `amp_d` 会再次饱和
  —— **这不是判据**，真判据是 `amp_style_std` 是否保持 >0.03 与 `amp_d_ref − amp_d` 的 gap 是否 ≈2；
  ④ 风险判据：500 轮内 `Mean reward` 从 ~+45 掉到 <+20、或 `hf_acc_all` 明显上升 ⇒ 回退 `coef` 到 2.0；
  ⑤ 若"×4 后步态仍无变化" ⇒ 问题不在**权重**而在**信号** ⇒ 下一轮（V29）转结构对齐：
  判别器输入改回单帧对 / 专家 batch 从运动数据**随机抽帧**（解耦相位）/ 恢复判别器正则
  → 该版 run 日志 + §A6（判别器饱和是常态）

⚠️ 其余逐字不动：奖励表 / 指令集 / AMP 观测 / σ（见上表 ②）。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V28-G1-v0`
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
