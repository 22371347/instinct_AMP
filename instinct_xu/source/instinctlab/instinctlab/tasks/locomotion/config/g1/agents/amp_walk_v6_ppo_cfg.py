"""AMP 行走教师 **v6** 的 PPO runner 配置 —— **算法侧一个字都没改**。

v6 相对 v5 的唯一改动在**环境侧**（`amp_walk_v6_env_cfg.py`）：
恢复基类的 `joint_deviation_arms`（−0.1）⇒ 奖励集 14 → 15 项。

本文件只负责：注册任务时指向正确的 runner cfg、并把日志名改掉。
镜像损失 / 站立关风格 / `log` 风格奖励 / 判别器全套超参**全部继承 v5**。

⚠️ 因为算法侧无改动，**续训是安全的**（`--resume` 会按 key 恢复 policy/critic/
   判别器及其优化器）；奖励集变更会立刻生效，**不需要重训**。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v5_ppo_cfg import G1AmpWalkV5PPORunnerCfg


@configclass
class G1AmpWalkV6PPORunnerCfg(G1AmpWalkV5PPORunnerCfg):
    """v6：v5 全部超参（`log` 风格奖励 + 镜像损失 + 站立关风格）原样继承。"""

    def __post_init__(self):
        super().__post_init__()

        # 算法侧没有任何改动 —— 唯一变化是环境侧的 `joint_deviation_arms`。
        # 这里只改日志名，便于与 v5 的曲线区分。
        self.run_name = "v6_arms0.1_logamp_mirror0.1_nostyleonstand"


__all__ = ["G1AmpWalkV6PPORunnerCfg"]
