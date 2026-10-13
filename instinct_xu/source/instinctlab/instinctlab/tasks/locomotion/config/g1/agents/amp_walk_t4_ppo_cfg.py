"""**T4 算法侧**：与 T3 **逐字相同**（唯一算法侧开关仍然是"续训时保住判别器"）。

T4 是**环境侧消融臂**（移除 `track_base_height`），算法侧不需要任何新开关：

    fresh_discriminator_on_load = False   ← 与 T3 相同（保住 T2 刚活过来的判别器）

⇒ 直接继承 `G1AmpWalkT3PPORunnerCfg`，只回读 + 改 `run_name`。
   好处：T3/T4 的**唯一变量**就是那一项奖励，算法侧连守卫都不用重写。

⚠️ 热启动：`t2_mjlabcopy_obs1500_rsi_cold` / `model_11500.pt`
   （与 T3 **同一个起点** ⇒ T2 / T3 / T4 三臂可直接对比）
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_t3_ppo_cfg import G1AmpWalkT3PPORunnerCfg


@configclass
class G1AmpWalkT4PPORunnerCfg(G1AmpWalkT3PPORunnerCfg):
    """T4 算法侧 = T3（父链已断言全部开关），本臂只回读 + 改 run_name。"""

    def __post_init__(self):
        super().__post_init__()
        a = self.algorithm
        if bool(a.fresh_discriminator_on_load):
            raise ValueError("[T4] fresh_discriminator_on_load 必须为 False（与 T3 一致）")
        self.run_name = "t4_noh_from11500"
        print("[T4][ALGO] 算法侧逐字沿用 T3（唯一开关：fresh_discriminator_on_load=False）")
        print(f"    style_lerp={a.style_lerp}  c={a.discriminator_reward_coef}  "
              f"clip_min_std={a.clip_min_std}  init_std={self.policy.init_noise_std}")
        print("    ⚠️ 本臂与环境侧 T4 配对：T3 的 `track_base_height` 被**整项移除**（12→11 项）")


__all__ = ["G1AmpWalkT4PPORunnerCfg"]
