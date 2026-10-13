"""**T5 算法侧**：与 T3 逐字相同（唯一算法侧开关仍是"续训时保住判别器"）。

T5 是**环境侧换体臂**（高度项：量躯干 → 量骨盆），算法侧无新开关：

    fresh_discriminator_on_load = False    ← 与 T3 相同（保住 T2 刚活过来的判别器）

⇒ 直接继承 `G1AmpWalkT3PPORunnerCfg`，只回读 + 改 `run_name`。

⚠️ 热启动：`t2_mjlabcopy_obs1500_rsi_cold` / `model_11500.pt`（**与 T3/T4 同一起点**）
   ⇒ T2 / T3 / T4 / T5 四臂可直接对比。本臂**不需要冷启动**：obs/动作契约一字未改
   （只换了奖励里被测量的刚体）⇒ `--resume` 合法。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_t3_ppo_cfg import G1AmpWalkT3PPORunnerCfg


@configclass
class G1AmpWalkT5PPORunnerCfg(G1AmpWalkT3PPORunnerCfg):
    """T5 算法侧 = T3（父链已断言全部开关），本臂只回读 + 改 run_name。"""

    def __post_init__(self):
        super().__post_init__()
        a = self.algorithm
        if bool(a.fresh_discriminator_on_load):
            raise ValueError("[T5] fresh_discriminator_on_load 必须为 False（与 T3 一致）")
        self.run_name = "t5_pelvis_height_from11500"
        print("[T5][ALGO] 算法侧逐字沿用 T3（唯一开关：fresh_discriminator_on_load=False）")
        print(f"    style_lerp={a.style_lerp}  c={a.discriminator_reward_coef}  "
              f"clip_min_std={a.clip_min_std}  init_std={self.policy.init_noise_std}")
        print("    ⚠️ 与环境侧 T5 配对：高度项 `track_base_height`(躯干) → "
              "`track_pelvis_height`(骨盆, 靶 0.7585)")
        print("    ⚠️ 动作/观测/资产**一字未改** ⇒ 与越障仍同源、蒸馏不受影响、可热启动")


__all__ = ["G1AmpWalkT5PPORunnerCfg"]
