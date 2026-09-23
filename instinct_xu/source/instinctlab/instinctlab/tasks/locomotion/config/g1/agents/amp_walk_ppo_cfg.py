"""AMP 行走教师（privileged）的 RL 配置。

**步骤②**：与 OmniLoco 完全一致，只是换了 `experiment_name` / `run_name`，
保证“任务骨架本身可训练”这一性质与学长基线严格可比（此时还没有 AMP）。

步骤⑥ 会在这里追加 WASABI/AMP 相关字段（判别器、奖励混合权重等）——
届时必须在此类**显式声明**这些字段（`@configclass` 是 dataclass，
父类没有的字段不声明就无法从 runner 传下去，见 PLAN_amp_walk.md §5.2 第 4 条）。
"""

from isaaclab.utils import configclass

from .omni_loco_ppo_cfg import G1OmniLocoPPORunnerCfg


@configclass
class G1AmpWalkPPORunnerCfg(G1OmniLocoPPORunnerCfg):
    experiment_name = "g1_amp_walk"

    def __post_init__(self):
        super().__post_init__()
        self.resume = False
        self.load_run = None
        # 步骤⑥ 接入 AMP 后，这里会变成类似 "wasabi_amp_lerp0.75"
        self.run_name = "step2_privileged_obs626_no_amp"
