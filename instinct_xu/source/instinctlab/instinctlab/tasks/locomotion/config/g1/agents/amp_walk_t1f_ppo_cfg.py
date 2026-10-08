"""**T1f 算法侧**：把 PPO/AMP 的探索与网络口径对齐 AMP_mjlab。

对照依据（`AMP_mjlab/src/tasks/amp_loco/config/g1/rl_cfg.py`）
-----------------------------------------------------------
    | 参数                | mjlab            | T1a2（我们）      | T1f（本臂）      |
    |---------------------|------------------|-------------------|------------------|
    | `amp_task_reward_lerp`（λ）| 0.75      | 0.75              | **0.90** ⚠️      |
    | 风格绝对量级        | 0.25×[0,0.1]=≤0.025 | 0.25×D ≈0.075–0.25 | 0.10×D          |
    | σ 下限              | `min_normalized_std=0.05` | `clip_min_std=1e-12` | **0.05** |
    | policy hidden       | (512,256,128)    | (256,128,128)     | **(512,256,128)** |
    | entropy_coef        | 0.005            | 0.008             | **0.005**        |
    | disc hidden         | (1024,512,256)   | 未改（后续消融）   | 未改             |

⚠️ **λ 为什么取 0.90 而不是照抄 mjlab 的 0.75**：
   mjlab 的风格项是**有界回归式** `style = amp_reward_coef × clamp(1 − (d−1)²/4, 0)`
   （`discriminator.py:135`），且 `amp_reward_coef = 0.1`、再与 task 做凸组合
   `r = λ·task + (1−λ)·style`（`tools/plot_rewards.py:172`）
   ⇒ mjlab 风格绝对贡献 **≤ 0.025/步**。
   我们的合并式（`instinct_rl/algorithms.py:572`）是 `r = task·(1−(1−λ)/c) + (1−λ)·style`，
   `style` 为判别器原始输出 D（典型 0.3–1.0）⇒ λ=0.75 时贡献 **0.075–0.25**，
   **约为 mjlab 的 3–10×**。取 λ=0.90 ⇒ 0.10×D ≈ 0.03–0.10，把风格强度拉回 mjlab
   量级，同时任务项系数升到 0.90（**更强的速度跟踪**，与目标一致）。
   本仓库若日后支持 `amp_reward_coef`/有界风格，应改回 λ=0.75 + 有界式。

⚠️ **σ 下限 0.05 是本次最可能的"踝冻结"解药**：T1a2 的 `clip_min_std = 1e-12`
   允许某些关节的策略标准差塌到 ~0 ⇒ 该自由度**再也无法被探索**（踝 `ankle_roll`
   误差从 ≤500 轮起恒定、全程不变，与此一致）。mjlab 用每关节 0.05 的下限避免这点。
   我们用标量 0.05 近似（本仓库 `clip_min_std` 为标量），后续可换每关节列表。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_t1a2_ppo_cfg import G1AmpWalkT1a2PPORunnerCfg

AMP_T1F_STYLE_LERP = 0.90
AMP_T1F_CLIP_MIN_STD = 0.05
AMP_T1F_HIDDEN_DIMS = (512, 256, 128)
AMP_T1F_ENTROPY_COEF = 0.005


@configclass
class G1AmpWalkT1fPPORunnerCfg(G1AmpWalkT1a2PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        a, p = self.algorithm, self.policy
        a.style_lerp = float(AMP_T1F_STYLE_LERP)
        a.clip_min_std = float(AMP_T1F_CLIP_MIN_STD)
        p.actor_hidden_dims = list(AMP_T1F_HIDDEN_DIMS)
        p.critic_hidden_dims = list(AMP_T1F_HIDDEN_DIMS)
        a.entropy_coef = float(AMP_T1F_ENTROPY_COEF)
        self.run_name = "t1f_mjlabalign"

        # 回读断言（fail-fast）
        if abs(float(a.style_lerp) - AMP_T1F_STYLE_LERP) > 1e-12:
            raise ValueError(f"[T1f] style_lerp 未生效：{a.style_lerp}")
        if abs(float(a.clip_min_std) - AMP_T1F_CLIP_MIN_STD) > 1e-12:
            raise ValueError(f"[T1f] clip_min_std 未生效：{a.clip_min_std}")
        if list(p.actor_hidden_dims) != list(AMP_T1F_HIDDEN_DIMS):
            raise ValueError(f"[T1f] hidden_dims 未生效：{p.actor_hidden_dims}")
        print("[T1f][ALGO ALIGN] 已对齐 AMP_mjlab 的探索/网络口径：")
        print(f"    style_lerp     = {a.style_lerp:.2f}（风格权重 {1 - a.style_lerp:.2f}；mjlab 风格绝对量级 ≤0.025）")
        print(f"    clip_min_std   = {a.clip_min_std:.3f}（mjlab min_normalized_std=0.05；T1a2 原为 1e-12）")
        print(f"    hidden_dims    = {list(p.actor_hidden_dims)}（mjlab；T1a2 原为 [256,128,128]）")
        print(f"    entropy_coef   = {a.entropy_coef:.4f}（mjlab；T1a2 原为 0.008）")
        print(f"    use_mirror_loss= {getattr(a, 'use_mirror_loss', None)}（交付要求②）")
