"""**T1n 环境侧**：与 T1l **逐字相同**（本臂改动全在算法侧）。

为什么要建这个文件
------------------
1. 与 T1* 系列惯例一致（每个臂一个 cfg ⇒ 会进 `instinct/tools/audit_reward_table.py`
   与 `docs/amp_walk/EFFECTIVE_REWARDS.md` 的枚举），便于日后审计。
2. 提供一个 `[T1n]` 前缀的**回读断言**：把"环境侧零改动"这件事本身变成可执行检查，
   而不是靠"我记得我没改"。

⚠️ 关于 `track_base_height`（2026-10-09 核对，别再做无效功）
-----------------------------------------------------------
云端 `params/env.yaml` 实测（不是源码推断）：

    T1k : std **0.2** / weight 1.5      ← V9H 的收紧值
    T1l : std **0.3** / weight 1.5      ← **T1l 已经改回 mjlab 值**（run 名 `hstd0p3`）
    T1m : std **0.3** / weight 1.5

⇒ "std 0.2 → 0.3" 这个动作 **在 T1l 就已经做完了**，本臂继承即得（所以本文件只做断言，
   不做修改）。如果"抑制抬脚"在 std=0.3 下仍然存在，剩余的旋钮只有
   **weight 1.5 → 更小** 或 **std 0.3 → 更大** —— 那需要**另开一臂**（单变量）。

RSI 相位随机化的处置
--------------------
T1m 引入的 `motion_start_from_middle_range (0.0, 0.0) → (0.0, 1.0)` **本臂不启用**
（继承 T1l = `(0.0, 0.0)`）。理由：T1m 实测 `amp_style_zero_frac` 0.377→0.35、
步态无任何改善 ⇒ 在判别器饱和的前提下它不可能有效（RSI 的价值是把判别器暴露到
更多相位上，前提是判别器还"有话说"）。等 T1n 的判别器修好后可以再单独试一次。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_t1l_env_cfg import G1AmpWalkT1lEnvCfg, G1AmpWalkT1lEnvCfg_PLAY

#: 期望值（= T1l = mjlab 值）。本臂**不修改**它们，只断言。
T1N_EXPECT_HEIGHT_STD = 0.3
T1N_EXPECT_HEIGHT_WEIGHT = 1.5


@configclass
class G1AmpWalkT1nEnvCfg(G1AmpWalkT1lEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        term = self.rewards.track_base_height
        _std = float(term.params["std"])
        _w = float(term.weight)
        if abs(_std - T1N_EXPECT_HEIGHT_STD) > 1e-12:
            raise ValueError(f"[T1n] track_base_height.std 意外：{_std}（应 {T1N_EXPECT_HEIGHT_STD}）")
        if abs(_w - T1N_EXPECT_HEIGHT_WEIGHT) > 1e-12:
            raise ValueError(
                f"[T1n] track_base_height.weight 意外：{_w}（应 {T1N_EXPECT_HEIGHT_WEIGHT}）"
            )

        self.run_name = "G1AmpWalkT1nMVPPlane_obs378_rew17_whiten"
        print("[T1n][ENV = T1L] 环境侧与 T1l 逐字相同（本臂改动**全在算法侧**）：")
        print(
            f"    track_base_height: std={_std} weight={_w}"
            f" ← mjlab 值（T1l 已从 V9H 的 0.2 改回 0.3；本臂未再做改动）"
        )
        print(
            "    RSI 相位随机化：**未启用**（继承 T1l 的 motion_start_from_middle_range (0.0, 0.0)；"
            "T1m 的 (0.0, 1.0) 已撤回）"
        )


@configclass
class G1AmpWalkT1nEnvCfg_PLAY(G1AmpWalkT1lEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "G1AmpWalkT1nMVPPlane_obs378_rew17_whiten_PLAY"
