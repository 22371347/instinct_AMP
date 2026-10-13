"""V29：**回退 V28 的系数 + 把判别器向 AMP_mjlab 对齐**（起点 = V27 的 `model_10000.pt` 热启动）。

## ⚠️ 本层对**今天**仍有影响的部分

* ⭐⭐ **`AMP_V29_STATE_HISTORY = 2`（判别器输入帧数）今天仍生效 —— 当前值 = 2 帧**。
  今天训练臂的判别器帧数就取这个常量：V31 / V32 / V33 都
  `from .amp_walk_v29_env_cfg import AMP_V29_STATE_HISTORY`，V33 更直接用它算维度
  （`AMP_WALK_V33_STATE_SIZE = AMP_V29_STATE_HISTORY × 198 = 396`）。
* ❌ ②③④ 全在**算法侧**（`agents/…`）⇒ 见下表"未核实"。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | ⭐ 判别器输入帧数 **8 → 2**（`AMP_V29_STATE_HISTORY = 2`；13body×15×2 = 390）| ✅ **仍生效，当前值 = 2 帧**（V33 补 anchor 投影重力后判别器输入 → **396 维**）|
| ② | 专家批（参考批）**独立随机重采 (T,B)**（`AmpStorage(independent_reference_sampling=True)`）〔算法侧〕| **未核实（超出 EFFECTIVE_REWARDS.md 覆盖范围）** |
| ③ | 风格奖励形状 `log_quad` → **`bounded_quad`** = `clamp(1−0.25(d−1)², 0, 1)`〔算法侧〕| **未核实（超出 EFFECTIVE_REWARDS.md 覆盖范围）** |
| ④ | 判别器系数**逐字回退 V27**（GP 1.0 / wd 1e-5 / coef 1.0，把 V28 的 4.0/5/3e-4 改回）〔算法侧〕| **未核实（超出 EFFECTIVE_REWARDS.md 覆盖范围）** |
| ⑤ | 白化"已去掉（V27）"⇒ 本层**不动** | ⚠️ **存疑**：后续核查发现 V22 是**旁支**、V27~V31 的判别器里**没有** `normalizer.*` ⇒「白化」可能**本就不存在**（该条疑为 no-op）→ 与 V27 稿同一冲突，**待归档** |

## 推理与判读（已归档，勿在本文件重复）

* ⭐ **"放大 AMP 风格权重"已被证伪两次**（V15 无效 / V28 有害）→ `docs/amp_walk/EXPERIMENTS.md` §A3
* V28 死因复盘（iter 10470 一步摔死：奖励噪声 ×8.3 / 风格项占比 16%→~47% / `log_quad` 形状无上界
  ⇒ 奖励离群拖爆优势）→ §A3 + 该版 run 日志
* 有效风格权重仍是 `(1−0.75)×1.0 = 0.25`；**不要为了让风格项"更响"再抬 coef —— 那正是 V28 的错**
  → §B3（λ 的陷阱："同一数值 ≠ 同一强度"）
* 预登记判据 1–5（判别器不立刻贴边 / `amp_style_std` > 0.058 且 `zero_frac` 不长期 = 1.0 /
  不退化闸门 / 步态侧 `single_stance`↑ · `hop_frac`↓ / 崩溃守护）+ 守护脚本 `run_amp_walk_v29.sh`
  （`episode length < 50` 连续 20 轮自动停训并保留 ckpt）→ 该版 run 日志
* ⚠️ **后续实测（尚未进 `EXPERIMENTS.md`）**：单靠"2 帧 + 独立专家批"**没让判别器不再饱和**
  ⇒ 本臂预登记判据 1 未达成；根因后来被定位为"策略与参考分布相差过大"，**不是**相位俘获 → 该版 run 日志

⚠️ 其余逐字不动：V27 的 21 项奖励 / 指令分布 / σ=0.20 + 退火 / λ 0.75 / 专家集。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V29-G1-v0`
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v20_env_cfg import AmpV20PolicyStateObsCfg, AmpV20ReferenceStateObsCfg
from .amp_walk_v23_env_cfg import _group_with_history
from .amp_walk_v27_env_cfg import (
    G1AmpWalkV27EnvCfg,
    G1AmpWalkV27EnvCfg_PLAY,
    G1AmpWalkV27TeacherEnv,
)

#: 判别器输入的 AMP 历史帧数：8 → **2**（对齐 mjlab 的单帧状态对 `(s_t, s_{t+1})`，390 维）。
#: 旁证：V20 用 2 帧时 `amp_style_std = 0.094`（健康）；V23+ 换成 8 帧后风格反而更弱。
AMP_V29_STATE_HISTORY = 2

AMP_WALK_V29_RUN_NAME = (
    "G1AmpWalkV29MVPPlane_obs378_rew21_amp13body2frame_nodrop_white_sigma0.20_mjlabalign"
)


@configclass
class G1AmpWalkV29EnvCfg(G1AmpWalkV27EnvCfg):
    """训练用：V27 的奖励/指令/σ/去白化全部不动，只把 AMP 观测历史 8 帧 → 2 帧。"""

    def __post_init__(self):
        super().__post_init__()
        # ⚠️ 父链的 `_patch_amp_history()` 会覆盖这两组 ⇒ 必须在 super() 之后再赋一次
        #    （V21/V23 同款踩坑记录，见 `amp_walk_v20_env_cfg.py:44-48`）。
        self.observations.amp_policy = _group_with_history(
            AmpV20PolicyStateObsCfg, AMP_V29_STATE_HISTORY
        )
        self.observations.amp_reference = _group_with_history(
            AmpV20ReferenceStateObsCfg, AMP_V29_STATE_HISTORY
        )
        self.run_name = AMP_WALK_V29_RUN_NAME


@configclass
class G1AmpWalkV29EnvCfg_PLAY(G1AmpWalkV27EnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广/同 AMP 观测口径。"""

    def __post_init__(self):
        super().__post_init__()
        self.observations.amp_policy = _group_with_history(
            AmpV20PolicyStateObsCfg, AMP_V29_STATE_HISTORY
        )
        self.observations.amp_reference = _group_with_history(
            AmpV20ReferenceStateObsCfg, AMP_V29_STATE_HISTORY
        )
        self.run_name = AMP_WALK_V29_RUN_NAME


class G1AmpWalkV29TeacherEnv(G1AmpWalkV27TeacherEnv):
    """指令范围与前几代相同（全向四档）⇒ 期望值直接继承。"""


__all__ = [
    "AMP_V29_STATE_HISTORY",
    "AMP_WALK_V29_RUN_NAME",
    "G1AmpWalkV29EnvCfg",
    "G1AmpWalkV29EnvCfg_PLAY",
    "G1AmpWalkV29TeacherEnv",
]
