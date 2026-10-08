"""V24 的算法配置：**逐字沿用 V23/V22**（白化 + GP 归一化空间 + AMP 口径 = V19d）。

唯一区别在**环境侧**：`dof_acc_l2` 从"只罚髋/膝"改为**全关节**（权重不变 −1.0e-6）。
算法侧一个数都不动 ⇒ 归因干净。

⚠️ 但请注意 V23 的实测发现（详见 `amp_walk_v24_env_cfg.py` 与报告）：
白化让判别器**更早饱和**（`Loss/discriminator_loss 0.0001`、`Step/amp_style_std 0.0031`，
比前几代的 0.09~0.14 低 30~45×）⇒ 风格项实际上已经"死"。
本版**保留白化不动**（为了与 V23 保持单变量），但已把它列为**下一个待修项**：
候选是"降低判别器学习速度"（`AdamW lr 1e-4 → 1e-5`）或"每 k 轮更新一次判别器"
—— 判别器太容易分开时，`d → ±1` 饱和 ⇒ `log_quad` 输出常数 ⇒ 失去排序信号。

起点 **从零**（V23 已进入高频小跳模态；历史教训：从塌陷 ckpt 续训出不来自吸引子）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v23_ppo_cfg import G1AmpWalkV23PPORunnerCfg


@configclass
class G1AmpWalkV24PPORunnerCfg(G1AmpWalkV23PPORunnerCfg):
    """v24：算法侧与 V23 逐字相同（白化 + 归一化空间 GP），只有 env 侧改了 acc 罚的关节集。"""

    def __post_init__(self):
        super().__post_init__()
        # 起点：从零（与 V23 相同，判别器/策略都不复用）
        self.ckpt_manipulator = None
        self.run_name = "v24_scratch_amp13body8frame_white_progress_accall"


__all__ = ["G1AmpWalkV24PPORunnerCfg"]
