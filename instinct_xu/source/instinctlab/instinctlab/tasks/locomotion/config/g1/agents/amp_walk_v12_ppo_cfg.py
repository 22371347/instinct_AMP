"""V12 的算法配置：**与 V11 逐字相同**（唯一变量都在环境侧，见 `amp_walk_v12_env_cfg.py`）。

| 字段 | V11 | **V12** | 说明 |
|---|---|---|---|
| `style_lerp`（λ） | 0.5 | 0.5 | 不动 |
| `discriminator_reward_coef`（c） | 0.5 | 0.5 | 不动（有效风格权重 (1−λ)c = 0.25） |
| `clip_max_std` | 0.6 | 0.6 | 不动 |
| `init_noise_std` | 0.35 | 0.35 | ⚠️ 热启动时**不生效**（σ 用 ckpt 里学到的值） |
| `clip_min_std` | 0.02 | 0.02 | 不动 |
| `entropy_coef` | 0.005 | 0.005 | 不动 |
| `schedule` | fixed | fixed | 不动 |
| `use_mirror_loss` | True | True | 不动 |

## ⚠️ 更正（实测踩坑）：**判别器是存在 checkpoint 里的**

早先我只看了 `ppo.py::state_dict()`（它确实只写 actor/critic + optimizer），就下了
"判别器不在 ckpt 里 ⇒ 换输入维度也能热启动"的结论 —— **这是错的**：
`instinct_rl/algorithms/wasabi.py` **override 了** `state_dict()` / `load_state_dict()`，
把 `"discriminator"` 与 `"discriminator_optimizer"` 也写进/读出 checkpoint。

第一次起 V12 就因此在启动 62 s 后崩了：

    RuntimeError: Error(s) in loading state_dict for Discriminator:
        size mismatch for model.model.0.weight: [1024, 670] vs [1024, 134]

✅ **正确做法**（本版已启用）：用框架自己的 `ckpt_manipulator` 钩子
（`instinctlab.utils.ckpt_manipulators:drop_amp_discriminator`）把这两个键删掉；
`Wasabi.load_state_dict` 对**键缺失是容忍的**（只打一行 `[Warning] ... not found`）
⇒ actor/critic + PPO 优化器照常热启动、判别器（含其 AdamW 状态）从零重学。

⚠️ 因此：**该钩子只在判别器口径真的变了时才开**（V13 继承本配置也带着它，同样适用）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v11_ppo_cfg import (
    CLIP_MAX_STD_V11,
    CLIP_MIN_STD_V11,
    DISC_REWARD_COEF_V11,
    NOISE_STD_V11,
    STYLE_LERP_V11,
    G1AmpWalkV11PPORunnerCfg,
)


@configclass
class G1AmpWalkV12PPORunnerCfg(G1AmpWalkV11PPORunnerCfg):
    """v12：算法侧与 v11 一致（改动全在环境侧）+ 换判别器口径所需的 ckpt 修补器。"""

    #: ⭐ 判别器输入维度 670 → 134 ⇒ 丢掉 ckpt 里的旧判别器（详见模块 docstring 的 "更正"）
    ckpt_manipulator: str = "instinctlab.utils.ckpt_manipulators:drop_amp_discriminator"

    def __post_init__(self):
        super().__post_init__()
        # 显式再赋一次：确保 `to_dict()` 带给 runner（它读的是 cfg 顶层的 `ckpt_manipulator`）
        self.ckpt_manipulator = "instinctlab.utils.ckpt_manipulators:drop_amp_discriminator"

        self.algorithm.style_lerp = STYLE_LERP_V11
        self.algorithm.discriminator_reward_coef = DISC_REWARD_COEF_V11
        self.algorithm.clip_max_std = CLIP_MAX_STD_V11
        self.algorithm.clip_min_std = CLIP_MIN_STD_V11
        self.policy.init_noise_std = NOISE_STD_V11

        self.run_name = (
            f"v12_amp2frame_standpose1.0_ns{NOISE_STD_V11}_maxstd{CLIP_MAX_STD_V11}"
            f"_ent0.005_fixedlr_lerp{STYLE_LERP_V11}_ampc{DISC_REWARD_COEF_V11}_fromV11"
        )


__all__ = ["G1AmpWalkV12PPORunnerCfg"]
