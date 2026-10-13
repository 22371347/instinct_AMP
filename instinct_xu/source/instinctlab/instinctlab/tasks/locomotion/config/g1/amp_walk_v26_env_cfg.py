"""V26：**σ（动作噪声）单变量** —— 从 V24@8000 热启动，把 σ 从 0.57 压到 0.20。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— 本文件是 env cfg，而 V26 的实质改动全在算法侧**（`agents/amp_walk_v26_ppo_cfg.py`）：
  本文件相对 V24 **env 侧零改动**（奖励 / AMP 观测 / 指令 / 数据集 / `action_scale` 全同），
  只换了 `run_name`（σ 的 cap 与退火都写在算法 cfg 里）。

## 本层改动一览（下表**全是算法侧**）

| # | 改动〔算法 cfg〕| 现状（今天实际生效）|
|---|---|---|
| ① | `clip_max_std` **0.6 → 0.20**（V24 的 0.6 其实**未 binding**，σ 实平衡在 0.572）| **未核实（超出 EFFECTIVE_REWARDS.md 覆盖范围）** |
| ② | 新增 `clip_max_std_schedule = 300`（前 300 次 update 线性 0.57 → 0.20）| **未核实（超出 EFFECTIVE_REWARDS.md 覆盖范围）** |
| ③ | 起点 = **V24 `model_8000.pt` 热启动**（`--resume`，AMP 输入维度相同 ⇒ 不丢判别器）| 已不适用 |
| ④ | ⭐ **env 侧 0 项改动**（只改 `run_name`）| —（奖励表与 V24 **逐字相同**，可对比 EFFECTIVE_REWARDS.md 两条）|

## 推理与判读（已归档，勿在本文件重复）

* Step 0 的 σ 扫描（同一份 V24 权重、只缩放噪声）：噪声税/步 −5.261(σ0.569) → **−0.580(σ0.20)**、
  `hf_acc_all` 133.9 → 42.6、`single_stance` 0.063 → **0.138** ⇒ **σ=0.20 是步态最优区**；
  而"完全去噪"（σ=0）反而更差（`hop_frac` 回升到 0.680）⇒ 选 0.20
  → 该版 run 日志 + `/root/sigma_scan_v24.txt`
* 为什么用 **300 轮退火**而不是一次性硬夹：σ 直跳会让奖励分布整体上移 ~4/步 ⇒ value function 立刻错位、
  前 100~300 轮 advantage 全错；线性退火把冲击摊平 → 该版 run 日志
* 预登记判据 1–6（四项 `Episode_Reward/*` 的 σ^1.75 / σ¹ 斜率、`single_stance` > 0.12、`in_band` > 0.44、
  `amp_style_std` 是否回到 > 0.03、`progress_frac`、以及"500 轮不涨 ⇒ σ 砍太多"的风险判据）→ 该版 run 日志
* ⚠️ 后续复盘：V26 是**热启动**，其判别器**从第 0 轮就已饱和**（第一个 log 块 `amp_d` 已 = −1.0004）
  ⇒ 本层结论**只能回答 σ 问题**，不能用来判 AMP → `amp_walk_v27_env_cfg.py` 的推理小节 + §A6

⚠️ 其余逐字不动：V24 的全部环境侧（21 项奖励 / 5-clip 数据集 / 全向指令 / 13body×8 帧 AMP 观测）。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V26-G1-v0`
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
