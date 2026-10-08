"""V15 的环境：**逐字沿用 V14**（2 帧判别器 134 维 / 三档指令 / 2-clip 数据集 /
`track_default_pose_on_stand −0.3` / 限位软边距全关节 margin 0.15 power 2 weight −1.5）。

唯一改动在**算法侧**（`discriminator_reward_coef` 0.25 → 1.0，即风格分量 ×4），
见 `agents/amp_walk_v15_ppo_cfg.py`。

⚠️ 关于你看到的那张"躯干接近 45°"的截图：`Episode_Reward/flat_orientation_l2` 是
**所有 env、所有时刻的平均**，均值小不代表"任何时刻都竖直"。因此本轮起
`mdp/rewards_amp.py::joint_pos_limits_margin` 会额外记录**倾角分布**探针：

    Step/tilt_deg_mean      平均倾角（度）
    Step/tilt_p95_deg       **95 分位**倾角（度）⭐ 偶发大倾角看这个
    Step/tilt_gt{20,30,45}_frac   倾角超阈的 env 比例
    Step/grav_bx_mean       躯干系重力 x（有符号 ⇒ 判别前/后倾）
    [LIMIT-TOP]             每 500 步打印"边距占用最高的 5 个关节"（谁在超限）

⇒ 有了它们，"均值 4° 但偶发 45°"就能被直接证实/证伪。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v14_env_cfg import (
    G1AmpWalkV14EnvCfg,
    G1AmpWalkV14EnvCfg_PLAY,
    G1AmpWalkV14TeacherEnv,
)

AMP_WALK_V15_RUN_NAME = "G1AmpWalkV15MVPPlane_obs378_rew21_amp2frame_style_x4_c1.0_fromV11"


@configclass
class G1AmpWalkV15EnvCfg(G1AmpWalkV14EnvCfg):
    """训练用：与 V14 同环境/同奖励（差别只在 agent cfg 的 c）。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V15_RUN_NAME


@configclass
class G1AmpWalkV15EnvCfg_PLAY(G1AmpWalkV14EnvCfg_PLAY):
    """回放用：与训练同奖励/同分布/同增广。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V15_RUN_NAME


class G1AmpWalkV15TeacherEnv(G1AmpWalkV14TeacherEnv):
    """契约期望值只写在 env 类上（指令范围与 V14/V12/V11 相同）。"""


__all__ = [
    "AMP_WALK_V15_RUN_NAME",
    "G1AmpWalkV15EnvCfg",
    "G1AmpWalkV15EnvCfg_PLAY",
    "G1AmpWalkV15TeacherEnv",
]
