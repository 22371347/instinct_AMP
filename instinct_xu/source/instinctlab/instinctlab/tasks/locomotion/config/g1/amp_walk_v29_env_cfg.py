"""V29：**回退 V28 的系数 + 把判别器向 AMP_mjlab 对齐**（起点 = V27 的 `model_10000.pt` 热启动）。

## 为什么回退（V28 死因复盘）

V28（`coef` 1.0→4.0 + GP 1→5 + wd 1e-5→3e-4）在 **iter 10470 崩了**：
`Mean reward` 14.85→−7.28、`Mean episode length` 489→**2.7**、`action_abs_mean` 0.35→2.6、
`surrogate_loss` 单点 12× 尖峰。三层机制：

1. **奖励噪声 ×8.3**：风格项作用量 = 权重 × `amp_style_std`：
   V27 = 0.25×0.058 = **0.0145/步** → V28 = 1.0×0.12 = **0.12/步**。
   LR 固定 1e-3 且无 KL 自适应 ⇒ 一次坏批次即可毁掉策略。
2. **风格项从"任务的 16%"变成"~47%"**（`r = 0.75·r_task + 1.0·style`，实测 style≈0.70）
   ⇒ 崩前 470 轮已从 42 退化到 ~15（渐变），然后一步摔死。
3. **`log_quad` 的形状无上界**（`−log(1−D)`，d→1 时可达 13.8；×4 ⇒ 单步 **+12**）
   ⇒ 产生**奖励离群** ⇒ 优势被单点拖爆。

⇒ **"放大 AMP 风格权重"这条路已被证伪两次**（V15 无效、V28 有害）。本轮回退，改攻"**让风格信号有信息**"。

## 本轮改动（全部是"向 mjlab 对齐"，**不动任何权重**）

| # | 项 | V27 | **V29** | mjlab 对应 |
|---|---|---|---|---|
| ① | 判别器输入 | 13body×15×**8 帧** = 1560（展平）| **13body×15×2 帧 = 390**（单帧对 `(s_t, s_{t+1})` 语义）| `history_length=1` 的状态对 = 390 |
| ② | 专家批（参考批）| 与策略**同一时刻、同一 env**（`reference_states[T_select,B_select]`）| **独立随机重采 (T,B)** ⇒ 每个梯度步覆盖全相位 | `AMPLoader`：clip 轮换 + clip 内随机抽帧 |
| ③ | 风格奖励形状 | `−log(1−quad)`，∈[0,13.8]，`d≤−1` 时恒 0.693 | **`clamp(1−0.25(d−1)², 0, 1)`**，∈[0,1]，`d≤−1` 时**恰为 0** | `amp_reward_coef·clamp(1−0.25(d−1)², min=0)` |
| ④ | 判别器系数 | GP 1.0 / wd 1e-5 / coef 1.0 | **逐字回退（并显式写回）** | — |
| ⑤ | 白化 | 已去掉（V27）| 不动 | mjlab 也不白化 |

**有效风格权重仍是 (1−0.75)×1.0 = 0.25**（= V27 的值），只是形状换成有界版本 ⇒
风格项单步贡献从"典型 0.21 / 尾巴 0.75+"降到 **≤0.25、典型 ≈0.03**（`d≈−0.87` 时 `style=0.126`）⇒ **方向更安全**。

⚠️ **不要**为了让风格项"更响"再抬 coef —— 那正是 V28 的错。若 V29 显示风格项太弱
（`amp_style_reward` 长期 ≈0、`amp_style_zero_frac`≈1.0），正确的下一步是**修判别器饱和**（结构性），不是抬权重。

## 预登记判据（约 300~500 轮即可判）

1. **判别器不再"20 轮内贴边"**：`amp_d` 不迅速冲到 ±1（这正是 ② 的目标效果：参考批覆盖全相位 ⇒
   "相位/时序"这条捷径不再能把两类完全分开）。
2. `amp_style_std` > V27 的 0.058（②③ 生效的直接读数）；`amp_style_zero_frac` **不要**长期 = 1.0
   （=1.0 ⇒ `d ≤ −1` 死区吃满 ⇒ 风格项彻底死掉）。
3. **不可退化**：`Mean reward` ≥ 30（V27 同期 34~44）、`Mean episode length` 不跌破 300、
   `action_abs_mean` ≤ 0.6（V27 是 0.354）、`hf_acc_all` ≈44。
4. 行为侧：`air_time_single_stance` ↑ / `air_time_hop_frac` ↓ —— **这才是"风格项是否推动步态"的检验**。
5. **崩溃守护**（在 `run_amp_walk_v29.sh` 里）：`Mean episode length < 50` 连续 20 轮 ⇒ 自动停训并保留 ckpt。
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
