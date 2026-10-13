"""**T3 算法侧**：T2（全权照抄 mjlab）之上**只改一个开关** —— 续训时**保住判别器**。

    fresh_discriminator_on_load : True  →  **False**

为什么必须改
------------
T2 的 cfg 里 `fresh_discriminator_on_load = True`（T1n 为"救饱和判别器"引入）。
T2 是**冷启动**臂，该开关当时是 no-op；但如果直接 `--resume` 续训，它会**把刚刚活过来的
判别器连同其优化器一起丢掉**：

| 指标 | T1l/T1n（饱和） | **T2 @11.9k（刚活）** |
|---|---|---|
| `Step/amp_d` | −0.94 | **−0.83** |
| `Step/amp_d_ref` | +0.93（贴死） | **+0.52** |
| `Step/amp_style_zero_frac` | 0.35~0.38 | **0.26** |
| `Step/amp_style_reward` | ≈常数 | **0.168** |

判别器"有分辨力"是 T1 全程都没做到的事，是本臂唯一的增益来源之一。
⇒ T3 必须显式设为 **False**，让判别器、`EmpiricalNormalization` 的统计量、
优化器状态全部接着用（= 真正的"续训"）。

其余**逐字沿用 T2**（本臂不碰）
-------------------------------
λ=0.75 / c=0.1（风格 0.025）、判别器 `[1024,512,256]` + 白化 + GP10(expert)
+ trunk/head wd 1e-3/1e-2 + 离线专家集、`entropy_coef 0.005`、hidden (512,256,128)、
例外 A（`action_scale=1.0`）、例外 B（`init_noise_std 0.25` / `clip_min_std 0.0125`）、
`disable_style_on_stand=True`、镜像损失 0.1、`policy_obs_frames` 7 项 ×4 帧。

⚠️ 热启动：`t2_mjlabcopy_obs1500_rsi_cold` / `model_11500.pt`（obs 契约未变 ⇒ 可直接 resume）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_t2_ppo_cfg import (
    T2_CLIP_MIN_STD,
    T2_DISC_HIDDEN,
    T2_DISC_REWARD_COEF,
    T2_INIT_NOISE_STD,
    T2_OBS_FRAMES,
    T2_STYLE_LERP,
    G1AmpWalkT2PPORunnerCfg,
)


@configclass
class G1AmpWalkT3PPORunnerCfg(G1AmpWalkT2PPORunnerCfg):
    """T3 的算法侧：父链（T2）已做完全部回读断言，本臂只翻一个开关 + 回读。"""

    def __post_init__(self):
        super().__post_init__()
        a = self.algorithm

        # ---- 唯一改动：续训时**保留**判别器 ------------------------------------
        a.fresh_discriminator_on_load = False
        if bool(a.fresh_discriminator_on_load):
            raise ValueError("[T3] fresh_discriminator_on_load 未生效（必须为 False）")

        # ---- 守卫：其余必须仍是 T2 的口径（防止"顺手改了别的"）-------------------
        guards = [
            ("style_lerp", float(a.style_lerp), T2_STYLE_LERP),
            ("discriminator_reward_coef", float(a.discriminator_reward_coef), T2_DISC_REWARD_COEF),
            ("clip_min_std", float(a.clip_min_std), T2_CLIP_MIN_STD),
            ("init_noise_std", float(self.policy.init_noise_std), T2_INIT_NOISE_STD),
        ]
        for name, got, want in guards:
            if abs(got - want) > 1e-15:
                raise ValueError(f"[T3] {name} 被改动：{got}（应 {want}，与 T2 一致）")
        if list((a.discriminator_kwargs or {}).get("hidden_sizes", [])) != T2_DISC_HIDDEN:
            raise ValueError(f"[T3] 判别器容量被改动：{a.discriminator_kwargs}")
        if dict(a.policy_obs_frames) != T2_OBS_FRAMES:
            raise ValueError(f"[T3] policy_obs_frames 被改动：{a.policy_obs_frames}")
        if not bool(a.amp_normalizer_expert_only):
            raise ValueError("[T3] amp_normalizer_expert_only 必须保持 True（与 T2 一致）")

        self.run_name = "t3_h08025_w1p0_from11500"

        print("[T3][KEEP DISCRIMINATOR] 续训时保留判别器（唯一算法侧改动）：")
        print(
            f"    fresh_discriminator_on_load True → **{a.fresh_discriminator_on_load}**"
            "  ⇒ 判别器 / 白化统计量 / 优化器状态全部接着用"
        )
        print(
            f"    （T2 的判别器已'活过来'：amp_d −0.83、amp_d_ref **+0.52**（T1l 是 +0.93 贴死）、"
            f"zero_frac 0.26 ⇒ 不能丢）"
        )
        print(
            f"    其余沿用 T2：λ={a.style_lerp} c={a.discriminator_reward_coef}"
            f"（风格 {(1 - a.style_lerp) * a.discriminator_reward_coef:.4f}）"
            f" 判别器={list((a.discriminator_kwargs or {}).get('hidden_sizes', []))}"
            f" clip_min_std={a.clip_min_std} init_std={self.policy.init_noise_std}"
        )
        print("    ⚠️ 热启动源：t2_mjlabcopy_obs1500_rsi_cold / model_11500.pt（obs 契约未变）")


__all__ = ["G1AmpWalkT3PPORunnerCfg"]
