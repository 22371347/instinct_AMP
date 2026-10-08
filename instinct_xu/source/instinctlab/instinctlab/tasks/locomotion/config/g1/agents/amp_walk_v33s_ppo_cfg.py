"""V33-S 的算法配置：**判别器结构级对齐 mjlab**（白化 + 归一化空间 GP）+ 396 维专家数据集。

## 三件事（都在 V31 的判别器配方之上，其余逐字不动）

| # | 项 | V31（现状） | **V33-S** | 依据 |
|---|---|---|---|---|
| ① | 判别器输入白化 | **无**（`discriminator_kwargs` 里没有 normalizer）| `EmpiricalNormalization` | 见下"白化与 eps" |
| ② | GP 作用空间 | `raw` | **`normalized`** | 与 ① 必须成对：`∂d/∂x=(∂d/∂x̃)/(σ+eps)`，raw 空间下 σ→0 的近常数维把 ‖∇d‖ 放大到 10²~10³ ⇒ GP 支配损失 |
| ③ | 专家数据集 | `/root/amp_expert_states.pt`（390）| **`/root/amp_expert_states_396.pt`** | 环境侧加了"anchor 投影重力"通道（390→396）⇒ 专家批维度必须跟着变（`wasabi.py` 有 assert）|

`hidden_sizes` / `nonlinearity` / GP 系数（10.0，expert-only）/ 优化器分组 wd / `style_lerp` /
`discriminator_reward_coef` / `bounded_quad` —— **全部逐字不动**。
这两项（白化 + GP 空间）与 V22 引入的完全相同，只是当年挂在 **1560 维**输入上。

## ⚠️ 白化与 `eps`：本仓库有一个**三次实测的失效模式**，所以这里**有意偏离 mjlab**

V27 的对照表（`amp_walk_v27_env_cfg.py`，逐字引用）：

| 版本 | 判别器输入 | `style_std`（末 25%）| 判定 |
|---|---|---|---|
| V19b | 原始 1560 | 0.124 | ✅ 健康 |
| V20 | 原始（2 帧）| 0.094 | ✅ 健康 |
| V23 / V24 | **白化** expert-only | 0.003 / 0.008 | ❌ 死 |
| V26 | 白化 expert-only | **0.0008** | ❌ 死 |

机制（当时已定位）：`eps = 0.01` 把 `‖logit‖` 放大到 `1/eps = 100×` ⇒ sigmoid 全饱和
⇒ `d ≈ ±1` 跨 env 跨时间几乎恒定 ⇒ `style_std → 0` ⇒ 风格项**没有梯度**。

⚠️ 但那个结论是在**另一个根因**下取得的：当时的"专家批"其实是**单个点**
（4096 个影子机器人近同相位，`σ_ref` 中位数 0.003~0.04），判别器**本来就该**秒分完。
V30 已经用**离线数据集 i.i.d. 抽帧**修掉了这个根因（`disc_loss` 从 1e-5 抬到 **0.086**、
`amp_d` 从 −0.998 抬到 **−0.74**、`style_std` 从 0.0008 抬到 **0.175**）。
⇒ 白化**现在**是在一个"判别器本来就能活"的基础上重试，且这次还叠了 GP=10（expert-only）
—— 而 GP 平方项正是唯一直接压制"把 ±1 拟合到完美"的机制。

**仍然保留安全阀**：`eps` 取 **0.1（放大上限 10×）而不是 mjlab 的 0.01（100×）**。
* 理由是"放大上限"这一项**没有任何被修正过的证据**支持取 100×（失效模式正是它）；
* 白化的主要收益（条件数、统一尺度）在 10× 处已经拿到；
* 想严格对齐 mjlab 只改这一行：`AMP_V33S_NORMALIZER_EPS = 0.01`。

## 判据（预登记，看 S0 相对 A-cold 的**方向**，不看绝对值）

1. **白化是否把判别器弄死**：`Step/amp_style_std` 前 200 iters 内不得塌到 < 0.03；
   `Step/amp_style_zero_frac` 不得 > 0.5；`Loss/discriminator_loss` 不得掉到 < 1e-4。
   （这三条任一条长时间成立 ⇒ 白化在本口径下仍然有害，含 `eps=0.1` 也救不回来。）
2. **③ 是否让判别器"看见"姿态**：`Train/disc_grad/gravity` 应显著非零
   （V32 的归因探针，索引从运行时 `obs_format` 读）；`Step/grav_bx_mean` 应比 A-cold 更靠近
   参考的前倾 9°（V31@12300 是后倾 15.2°）。
3. **风格项是否更强**：`Step/amp_style_reward` / `style_std` 不低于 A-cold。
4. 不退化闸门：`Mean reward ≥ 30`、`episode length ≥ 300`、`action_abs_mean ≤ 0.6`。

起点：**V31 `model_11500.pt` 热启动 actor/critic**；判别器（含 normalizer 的 `_mean/_var/_std/count`
buffer）**从零重学** —— 输入维度变了（390→396）+ 输入定义变了（白化）⇒ 旧权重不可复用。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v27_ppo_cfg import DROP_DISC
from .amp_walk_v31_ppo_cfg import G1AmpWalkV31PPORunnerCfg

#: 白化统计量的 **σ 下限**（`EmpiricalNormalization.eps`）—— 同时是"放大倍数上限" `1/eps`。
#: ⚠️ 有意取 0.1（mjlab 是 0.01）：见模块 docstring「白化与 eps」。改成 0.01 即严格对齐 mjlab。
AMP_V33S_NORMALIZER_EPS = 0.1
#: 统计量累积上限（`None` = 永不冻结，与 mjlab 的 running normalizer 一致）。
AMP_V33S_NORMALIZER_UNTIL = None

#: 专家数据集 = 全运动数据集 i.i.d. 抽帧（mjlab `AMPLoader` 口径），**396 维**（含 anchor 投影重力）。
#: 由 `instinct/tools/cloud_export_amp_expert.py --with_anchor_gravity` 在云端导出。
AMP_V33S_EXPERT_BUFFER = "/root/amp_expert_states_396.pt"


@configclass
class G1AmpWalkV33SPPORunnerCfg(G1AmpWalkV31PPORunnerCfg):
    """v33-S：判别器白化 + 归一化空间 GP + 396 维专家数据集（S0 / S1 共用）。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- ① 判别器输入白化（`hidden_sizes` / `nonlinearity` 严格保持 V31 值）----
        self.algorithm.discriminator_kwargs = {
            **self.algorithm.discriminator_kwargs,
            "normalizer_class_name": "EmpiricalNormalization",
            "normalizer_kwargs": {
                "eps": AMP_V33S_NORMALIZER_EPS,
                "until": AMP_V33S_NORMALIZER_UNTIL,
            },
        }
        # ---- ② GP 必须在归一化空间（与 ① 成对，缺一个就崩）--------------------
        self.algorithm.amp_grad_pen_space = "normalized"
        # ---- 白化统计量只由 expert/reference 侧累积（沿用 V30/V31 的值）--------
        #   专家批 = 离线数据集的 i.i.d. 抽帧 ⇒ 白化是由**专家分布唯一确定**的固定仿射映射，
        #   actor/reference 走同一映射 ⇒ "同一把尺子"；策略分布不参与定尺（少一个污染自由度）。
        self.algorithm.amp_normalizer_expert_only = True
        # ---- ③ 396 维专家数据集 ----------------------------------------------
        self.algorithm.amp_expert_buffer_path = AMP_V33S_EXPERT_BUFFER
        # ---- 机械动作：判别器（含 normalizer buffer）从零重学 -------------------
        self.ckpt_manipulator = DROP_DISC

        self.run_name = "v33s_white010_normgp_grav396_side015_fork11500"


__all__ = [
    "AMP_V33S_EXPERT_BUFFER",
    "AMP_V33S_NORMALIZER_EPS",
    "AMP_V33S_NORMALIZER_UNTIL",
    "G1AmpWalkV33SPPORunnerCfg",
]
