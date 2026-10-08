"""V27 的算法配置：**丢掉已饱和的判别器 + 去掉白化**（保留 V26 的 σ=0.20）。

唯一变量 = **判别器输入口径**（白化 expert-only → 不白化），
外加一个机械动作 = **判别器重置**（`drop_amp_discriminator`；这是必须的，
因为要换输入口径，且旧判别器已饱和，见 `amp_walk_v27_env_cfg.py` 的对照表）。

⚠️ 判别器重置**不脏**实验：V26 的判别器从第 0 轮就是饱和的（`amp_d=−1.0004`），
它提供不了任何风格信号；无论换不换口径，都必须重学。

想换成另外两个变体时，只改下面标注 `# ← 变体开关` 的两行：

* **V27-a（只重置判别器、保留白化）**：注释掉那两行（保持白化 + normalized GP）。
* **V27-c（白化但两侧共同累积统计量）**：保留白化，把
  `self.algorithm.amp_normalizer_expert_only = False`（对齐 rsl_rl/mjlab 标准）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v26_ppo_cfg import G1AmpWalkV26PPORunnerCfg

#: `drop_amp_discriminator` 会删掉 ckpt 里的 `discriminator` + `discriminator_optimizer`
#: 两个键（`instinctlab/utils/ckpt_manipulators.py`），其余（actor/critic + PPO 优化器 +
#: 策略观测 normalizer）原样热启动 ⇒ "保留已学会的步态、判别器从零重学"。
DROP_DISC = "instinctlab.utils.ckpt_manipulators:drop_amp_discriminator"


@configclass
class G1AmpWalkV27PPORunnerCfg(G1AmpWalkV26PPORunnerCfg):
    """v27：V26（σ=0.20 + 退火）+ 去白化 + 判别器重置。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- 变体开关（V27 主方案 = 去白化）-----------------------------------
        # ① 去掉判别器输入白化：把 normalizer 两个 key 从 discriminator_kwargs 里摘掉
        dk = dict(self.algorithm.discriminator_kwargs)
        dk.pop("normalizer_class_name", None)
        dk.pop("normalizer_kwargs", None)
        self.algorithm.discriminator_kwargs = dk
        # ② GP 回到原始空间（无白化时必须与 raw 配对；否则 GP 在"未归一化"的输入上
        #    按 normalized 分支求梯度会与实际前向不一致）
        self.algorithm.amp_grad_pen_space = "raw"
        # ---------------------------------------------------------------------
        # 无 normalizer ⇒ 该 flag 实际不再起作用（`forward` 已按 normalizer is None
        # 短路），保持继承值即可，不做额外改动，避免引入第二个变量。

        # ---- 机械动作：判别器（与其优化器）从零重学 ----
        self.ckpt_manipulator = DROP_DISC

        self.run_name = "v27_dropdisc_nodrop_white_sigma0.20"


__all__ = ["DROP_DISC", "G1AmpWalkV27PPORunnerCfg"]
