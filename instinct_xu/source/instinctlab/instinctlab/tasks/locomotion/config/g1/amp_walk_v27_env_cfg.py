"""V27：**AMP 判别器"重启"** —— 丢掉已饱和的判别器 + 去掉白化，保留 V26 的 σ=0.20。

## 为什么必须重置判别器（本轮的硬前提）

热启动时 `runner.load()` 走的是 `self.alg.load_state_dict(loaded_dict)`，而 V26 的
`ckpt_manipulator=None` ⇒ **判别器权重 + 判别器优化器 + normalizer buffer 全部继承**。
实测证据：V26 **第一个** log 块（iteration 8000）就已经是
`amp_d = −1.0004` / `amp_d_ref = 0.9999` / `amp_style_std = 0.0014`
—— 全新初始化的判别器输出应 ≈0、`style_std` 应 O(0.1~1)。
⇒ **V26 的风格项从第 0 轮就是死的**，它的结果只能回答 σ 的问题，
**不能**用来判定"白化是否杀死风格项"。

## 本轮变量

相对 V26，只动**判别器口径**（外加"必须重置判别器"这个机械动作）：

| 项 | V26（= V22~V24 口径） | **V27** |
|---|---|---|
| 判别器输入 | 1560（195×8）+ **白化**（`EmpiricalNormalization(eps=0.01)`） | **1560，不白化**（回到 V19b/V19d/V20 口径） |
| 白化统计量来源 | `amp_normalizer_expert_only=True`（**只由 reference 侧**累积） | —（无 normalizer） |
| `amp_grad_pen_space` | `"normalized"` | **`"raw"`**（无白化时必须配对回 raw） |
| `ckpt_manipulator` | `None`（判别器继承） | **`drop_amp_discriminator`**（判别器+其优化器从零重学） |
| `clip_max_std` / 退火 | 0.20 / 300 | **不变** |
| 奖励表 / `action_scale=1.0` / LR / `entropy_coef` / 策略权重 | — | **全不动**（策略从 V26 ckpt 热启动） |

## 依据（三代对照）

| 版本 | 判别器输入 | `style_std`（末 25%） | 判定 |
|---|---|---|---|
| V19b | 原始 1560 | **0.124** | ✅ 健康 |
| V20 | 原始（2 帧） | **0.094** | ✅ 健康 |
| V23 / V24 | **白化** expert-only | 0.003 / 0.008 | ❌ 死 |
| V26（继承 V24 判别器 + 白化） | 白化 expert-only | **0.0008** | ❌ 死 |

机制：白化把"策略特征 vs 参考特征"的**系统性偏移**放大 `1/eps = 100×`
⇒ logits 巨大 ⇒ sigmoid 全饱和（`d_ref=+1.0002, d=−0.998, disc_loss=1e-5`）
⇒ `−log(1−D)` 变成**跨 env、跨时间都基本恒定** ⇒ `style_std ≈ 0` ⇒ 风格项梯度为 0。
（讽刺的是 V22 引入白化的动机是离线 AUC 0.850→0.927——"更可分辨"，
而 AMP 需要的恰恰相反：**判别器不能过早分辨完**。）

## 另两个变体（若想换口径，只改 `amp_walk_v27_ppo_cfg.py` 里标注的那两行）

* **V27-a（只重置判别器，保留白化）**：把"去白化两行"注释掉。
  预期：新判别器会"活"一小段，但 ~200-500 轮内因同样的系统性偏移再次饱和。
  **信息量**：这是"白化必然杀死判别器"的确证实验。
* **V27-c（保留白化但两侧共同累积统计量）**：`amp_normalizer_expert_only=False`
  （rsl_rl/mjlab 的标准口径）。测"白化本身是否可行"，与 mjlab 严格对齐。

## 预登记判据（对照 V26@8750 与 V19b/V20）

1. **风格项复活**：`Step/amp_style_std` 0.0008 → **> 0.03**（目标 0.09~0.12，= V19b/V20 水平）；
   `Loss/discriminator_loss` 1e-5 → **> 1e-3**；`amp_d` 从 −0.998 抬到 **> −0.9**。
2. **判别器不立刻饱和**：`amp_d_ref` 不再恒等于 +1.000（期望 < 0.9）。
3. **步态**：在 σ=0.20 的基线上继续改善（`hop < 0.648`、`in_band > 0.446`、
   `single_stance > 0.119`）；`track_lin_vel_xy_exp > 1.412`（超过 V24）。
4. **不退化**：`hf_acc_all` 仍 ~45（不被风格项重新拉回高频抖动）、`Mean reward` 仍 > +35
   （回合口径）。
5. **风险判据**：若风格项复活但 `hop`/`hf_acc` 恶化 ⇒ 说明判别器恢复分辨力后
   在"奖励高频动作"（判别器偏好参考的高频细节），需要同时降 `amp` 权重/`quad_k`。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v26_env_cfg import (
    AMP_WALK_V26_RUN_NAME,
    G1AmpWalkV26EnvCfg,
    G1AmpWalkV26EnvCfg_PLAY,
    G1AmpWalkV26TeacherEnv,
)

AMP_WALK_V27_RUN_NAME = (
    "G1AmpWalkV27MVPPlane_obs378_rew21_amp13body8frame_nodrop_white_sigma0.20"
)


@configclass
class G1AmpWalkV27EnvCfg(G1AmpWalkV26EnvCfg):
    """训练用：与 V26 逐字相同（奖励/AMP 观测/指令集都不动）；只有 run_name 不同。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V27_RUN_NAME


@configclass
class G1AmpWalkV27EnvCfg_PLAY(G1AmpWalkV26EnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V27_RUN_NAME


class G1AmpWalkV27TeacherEnv(G1AmpWalkV26TeacherEnv):
    """指令范围与 V19d/V20/V21/V23/V24/V26 相同（全向四档）⇒ 期望值直接继承。"""


__all__ = [
    "AMP_WALK_V27_RUN_NAME",
    "G1AmpWalkV27EnvCfg",
    "G1AmpWalkV27EnvCfg_PLAY",
    "G1AmpWalkV27TeacherEnv",
]
