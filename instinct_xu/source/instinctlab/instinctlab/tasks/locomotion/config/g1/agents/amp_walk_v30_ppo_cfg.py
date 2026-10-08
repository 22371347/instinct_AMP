"""V30 的算法配置：**判别器 = AMP_mjlab 的"原始配方"**（在 V29 的基础上只改判别器侧）。

## 背景（为什么不是继续调权重）

V29 已把"输入/形状"对齐 mjlab（判别器输入 `(s_t,s_{t+1})`=390、风格奖励形状
`clamp(1−0.25(d−1)²,0,1)`），但**判别器仍然 1 个迭代（20 梯度步）内把 MSE 压到 0.035**
（V27 稳态 `disc_loss` 甚至到 0.0014）⇒ 风格奖励变常数 ⇒ 风格项无梯度。

我们逐条排除了两个"猜测的缺陷"（都被数据否掉）：
  * 「速度维是分离主因」❌：稳态下 `lin_vel_b` 均值 d=0.21、`ang_vel_b` d=0.07（几乎重叠）；
  * 「专家批是单点/相位锁死」❌：V20 期 16 envs × 1995 步实测跨 env σ ≈ 跨时间 σ（比值 1.05~1.32）。

⇒ 只剩**从没单独试过**的 mjlab 判别器配方：

| # | 项 | V29（现状） | **V30** | mjlab |
|---|---|---|---|---|
| ① | `discriminator_gradient_penalty_coef` | 1.0 | **10.0** | λ = **10** |
| ② | GP 的作用对象 | **actor + expert 合并** | **只在 expert 侧**（`gp_expert_only=True`）| `compute_grad_pen(*sample_amp_expert, lambda_=10)` |
| ③ | 判别器 weight decay | loss 项 `wd=1e-5` | **优化器级分组**：trunk **1e-3** / head **1e-2**（loss 项设 0）| AdamW wd trunk 1e-3 / head 1e-2 |
| ④ | 专家批来源 | 活体影子机器人 | **离线运动数据集 i.i.d. 抽帧**（`amp_expert_states.pt`，2934×390）| `AMPLoader`：clip 轮换 + 帧内随机 |
| ⑤ | `discriminator_reward_coef` | 1.0（有效风格权重 0.25）| **保持 1.0** | 0.1（有效 0.025）|
| — | 输入 390 / 形状 `bounded_quad` / `λ=0.75` / 去白化 | 已对齐 | 不动 | — |
| — | 起点 | V27 `model_10000` | 同 | — |
| — | `ckpt_manipulator` | drop disc | drop disc（判别器口径变了 ⇒ 必须重学）| — |

**为什么 ⑤ 不跟着降到 0.1**：那是"风格强度"这个另立的变量；V28 已证明"我们把它放大有害"
≠ "取 mjlab 的值有益"。本轮只动**判别器本身的配方**，把风格强度固定在 V27/V29 的值（0.25），
这样"判别器是否活过来"才是唯一变量。

**为什么 ① 可能真的有用**：`disc_loss=0.0014` 意味着判别器把 ±1 **拟合到了完美**
（平均误差 0.04）。GP 平方项是**唯一直接压制"把两类拟合到完美"**的机制（它惩罚 ‖∂d/∂x‖²，
逼 `d` 变平滑 ⇒ 不能在整个 actor 云上恒为 −1 ⇒ 风格奖励处处保留梯度）。
而且 mjlab 的 GP **只在专家流形上**施加 ⇒ 不额外要求"对策略分布也不敏感"（我们 V29 把
actor 侧也拉进来，与"判别器要能区分两者"自相矛盾）。

## 预登记判据（**只看判别器是否活过来**，行为变化慢、不作本轮判据）

| 指标 | V27/V29 现状 | **期望（若配方是主因）** |
|---|---|---|
| `Loss/discriminator_loss` | 0.0014 ~ 0.035 | **> 0.05**（V11 是 0.121，是唯一未饱和的版本）|
| `Step/amp_d` / `amp_d_ref` | −0.92 / +0.87（贴边）| **−0.2 ~ −0.7** / 不恒 +1 |
| `Step/amp_style_std` | 0.058 | **> 0.10** |
| `Step/amp_style_zero_frac` | 0.36 → 1 | **< 0.3** |
| `Loss/discriminator_gradient_penalty` | 0.017 | 显著上升（λ=10 ⇒ 期望项量级 ~0.1~1）|
| 不退化闸门 | — | `Mean reward ≥ 30`、`episode length ≥ 300`、`action_abs_mean ≤ 0.6` |
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v29_ppo_cfg import DROP_DISC, G1AmpWalkV29PPORunnerCfg

# ---- ①② GP：λ=10 且只在专家侧（mjlab `compute_grad_pen(*sample_amp_expert, lambda_=10)`）----
GP_COEF_V30 = 10.0
GP_EXPERT_ONLY_V30 = True
# ---- ③ 优化器级分组 wd（mjlab：trunk 1e-3 / head 1e-2）------------------------------------
OPT_WD_TRUNK_V30 = 1.0e-3
OPT_WD_HEAD_V30 = 1.0e-2
# ---- ④ 专家批 = 离线运动数据集（云端路径；由导出脚本产出）--------------------------------
EXPERT_BUFFER_V30 = "/root/amp_expert_states.pt"


@configclass
class G1AmpWalkV30PPORunnerCfg(G1AmpWalkV29PPORunnerCfg):
    """v30：V29（输入 390 + 有界形状 + 去白化 + σ=0.20）+ **mjlab 判别器原始配方**。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- ① GP 强度：1.0 → 10.0（对齐 mjlab λ）--------------------------------
        self.algorithm.discriminator_gradient_penalty_coef = GP_COEF_V30
        # ---- ② GP 只在专家侧（不再把 actor 侧也拉进来）----------------------------
        self.algorithm.discriminator_gp_expert_only = GP_EXPERT_ONLY_V30
        # ---- ③ wd 从 loss 项搬到**优化器分组**（trunk / head）---------------------
        self.algorithm.discriminator_weight_decay_coef = 0.0
        self.algorithm.discriminator_optimizer_trunk_weight_decay = OPT_WD_TRUNK_V30
        self.algorithm.discriminator_optimizer_head_weight_decay = OPT_WD_HEAD_V30
        # ---- ④ 专家批来自离线数据集 ----------------------------------------------
        self.algorithm.amp_expert_buffer_path = EXPERT_BUFFER_V30
        # （`amp_expert_independent_sampling` 保持 V29 的 True：有 expert_buffer 时
        #   它会被优先接管，留着只是让两份配置的差异最小。）
        # ---- ⑤ 风格强度刻意不动（= V27/V29 的 1.0）--------------------------------
        #      ⚠️ 不要为了让风格"更响"在这里加权重 —— V28 就是这么做崩的。

        # ---- 机械动作：判别器（与其优化器）从零重学 ------------------------------
        self.ckpt_manipulator = DROP_DISC

        self.run_name = "v30_mjlab_disc_recipe_gp10_expertonly_optwd_expertbuf_4096"


__all__ = [
    "EXPERT_BUFFER_V30",
    "GP_COEF_V30",
    "GP_EXPERT_ONLY_V30",
    "OPT_WD_HEAD_V30",
    "OPT_WD_TRUNK_V30",
    "DROP_DISC",
    "G1AmpWalkV30PPORunnerCfg",
]
