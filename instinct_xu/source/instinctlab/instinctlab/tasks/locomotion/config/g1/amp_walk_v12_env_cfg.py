"""V12：**把 AMP 判别器输入换成 LLP 论文的“两帧状态转移”** + 抬高静止姿态权重。

## ① 判别器输入 10 帧 → 2 帧（本次的**主变量**，来自 LLP 原论文）

论文里 AMP/LightLP 判别器吃的是**状态转移 (s_t, s_{t+1})** —— 两帧（当前 + 下一帧），
既不是单帧、也不是长历史帧堆叠。我们此前沿用 beyondmimic/parkour 的
`AMP_STATE_HISTORY = 10`（`amp_policy` / `amp_reference` 各 670 维）。

⚠️ 为什么 10 帧可能"让 AMP 说不出话"：10 帧（0.2 s）的**姿态轮廓/抖动统计**本身就足以
把 policy 与 reference 分开 ⇒ 判别器**没有任何压力**去学"单步转移像不像人"。
这与我们反复观测到的现象自洽：`amp_d` 越训越负（−0.56 → −0.78）、`r_amp` 0.50 → 0.24，
即风格奖励**理论上给得起**（到参考水平还有 +2.16/step 的空间）但**够不着**。

改成 2 帧后：输入 = 67×2 = **134 维**；项的构成与缩放逐字不变
（`joint_vel_rel ×0.05`、`base_ang_vel`、`base_lin_vel` **本来就在状态里** ⇒
单步转移的信息是完整的；10 帧堆叠的边际信息最少）。
我们给两帧的次序是 (s_{t−1}, s_t)（history 缓冲是"老→新"），对"判别平稳分布"的目标
等价于论文的 (s_t, s_{t+1})；且 `amp_policy` / `amp_reference` 用**同一机制** ⇒ 天然同构
（wasabi 的 `init_storage` 会断言两侧等长）。

实现：在 env cfg 的 `__post_init__` 里调 `_patch_amp_two_frame(self)` ——
**新建**两个 obs 群对象再挂上去（不去改基类的共享实例），5 个 term 的 `history_length`
统一改 2。**策略 378 维观测、价值观测一个字都没动。**

## ② `track_default_pose_on_stand`：−0.3 → −1.0（静止稳定性）

用户要求：模型静止时更稳。它带 `gate_tau=0.25` 的平滑门控（`exp(−‖cmd‖/0.25)`）⇒
走动时基本不生效，只在低速/立定段把关节往默认站姿拉。所以加回设计值 −1.0
**不会**拖累行走能力。
⚠️ 历史教训：V9c 记过它"−15.3/回合"，那是**门控读错数组**时的数（30% 站立 env 一直在
被风格项拉、门控近似恒开）⇒ 与现在不可比。现在门控走真值指令
（`Step/amp_stand_frac ≈ 0.30` 已实测）、且 `disable_style_on_stand` 已真正生效。

## ③ 其余逐字不动

21 项奖励结构（含 `feet_swing_time_penalty −0.5`、V11 的 `feet_air_time 8.0/thr0.6`、
`action_rate_l2 −0.06`、`dof_acc_l2 −1e−6`）、三档指令、2-clip 数据集、镜像增广/损失、
σ 上限 0.6、LR 固定、算法超参（λ=0.5、c=0.5 ⇒ 有效风格权重 0.25）。

## ⚠️ 本轮同时动了**两个**量，判读口径必须分开

| 关注点 | 看什么 | 期望 |
|---|---|---|
| 风格（①的效应） | `amp_d` / `r_amp` / `Loss/amp_*` | `amp_d` 从 −0.78 抬向 −0.4 以内，`r_amp` 回升 |
| 静止稳定（②的效应） | 站立档回合的 `track_default_pose_on_stand`、`base_ang_vel`、`base_contact` | 静止时漂移/晃动下降 |
| 行走**不**退化 | `feet_air_time` >0.15/s、`vx_*_rmse`、中位回合长度 | 不低于 V11 |

若要严格单变量归因，下一轮把 ② 撤回即可（= V12b，只留 2 帧判别器）。

## ⚠️ 热启动：判别器**在** checkpoint 里，必须用 `ckpt_manipulator` 丢掉它

`instinct_rl/algorithms/wasabi.py` **override 了** `state_dict()` / `load_state_dict()`，
把 `"discriminator"` 与 `"discriminator_optimizer"` 也写进/读出 checkpoint
（早先只看 `ppo.py` 得出的"判别器不在 ckpt 里"是**错的**，第一次起 V12 就因此崩：
`size mismatch for model.model.0.weight: [1024, 670] vs [1024, 134]`）。

✅ 已在 `agents/amp_walk_v12_ppo_cfg.py` 里设 `ckpt_manipulator =
"instinctlab.utils.ckpt_manipulators:drop_amp_discriminator"`：
删掉这两个键后 `Wasabi.load_state_dict` 只打一行 `[Warning] ... not found`，
actor/critic（观测仍 378/99，形状未变）+ PPO 优化器照常热启动，判别器从零重学。
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .amp_walk_env_cfg import (
    AMP_STATE_TERMS,
    AmpPolicyStateObsCfg,
    AmpReferenceStateObsCfg,
)
from .amp_walk_v10b_env_cfg import AMP_WALK_V10B_COMMAND_RANGES
from .amp_walk_v11_env_cfg import (
    G1AmpWalkV11EnvCfg,
    G1AmpWalkV11EnvCfg_PLAY,
    G1AmpWalkV11RewardsCfg,
    G1AmpWalkV11TeacherEnv,
)

#: ⭐ 主变量：判别器输入帧数。论文 (s_t, s_{t+1}) = **2 帧**；10 帧是 beyondmimic/parkour 风格。
AMP_STATE_HISTORY_V12 = 2
#: ① 时 5 项各占的维度（`3 + 29 + 29 + 3 + 3 = 67`/帧）
AMP_STATE_SIZE_V12 = AMP_STATE_HISTORY_V12 * 67          # == 134
#: ② 静止姿态权重（V9h/V10b/V11 都是 −0.3）
AMP_WALK_V12_STAND_POSE_WEIGHT = -1.0

AMP_WALK_V12_RUN_NAME = (
    "G1AmpWalkV12MVPPlane_obs378_rew21_amp2frame_standpose1.0_fromV11"
)


def _amp_group_with_history(group_cls, history: int):
    """**新建**一个 obs 群实例并把 5 个 term 的 `history_length` 全改成 `history`。

    ⚠️ 刻意不原地修改基类里的共享实例（`G1AmpWalkObservationsCfg` 的类属性），
    否则同进程内其它版本（V9~V11）的 cfg 会被一起改掉。
    """
    grp = group_cls()
    for name in AMP_STATE_TERMS:
        getattr(grp, name).history_length = history
    return grp


def _patch_amp_history(env_cfg, history: int) -> None:
    """把判别器两组 obs 的 history 统一改成 `history` 帧（**两组必须相同**）。

    ⚠️ 判别器输入维度 = Σ_terms (history × feat/帧)：
        history=2  ⇒ 2×67  = **134**（论文 (s_t, s_{t+1}) 的 2 帧状态转移，V12/V14/V15）
        history=10 ⇒ 10×67 = **670**（beyondmimic/parkour 的长历史，V1~V11、**V16**）
    两侧（`amp_policy` / `amp_reference`）用**同一个** history ⇒ 天然同构
    （wasabi 的 `init_storage` 会断言两侧等长）。

    帧序（实测，见 `instinct/tools/cloud_probe_amp_frames.py`）：
    `CircularBuffer.buffer` 是「最老一帧在 dim=1 开头、最新一帧在末尾」，且
    `_flatten_obs_group()` 逐 term `flatten(start_dim=1)` ⇒ 每个 term 内部帧连续
    （term-major），所以 `history=2` 的 134 维 = `[老帧 | 新帧]` 按 term 分块，
    **不是**"前半段/后半段"。
    """
    env_cfg.observations.amp_policy = _amp_group_with_history(AmpPolicyStateObsCfg, history)
    env_cfg.observations.amp_reference = _amp_group_with_history(AmpReferenceStateObsCfg, history)


def _patch_amp_two_frame(env_cfg) -> None:
    """把判别器两组 obs 的 history 改成 2 帧（= `AMP_STATE_HISTORY_V12`）。"""
    _patch_amp_history(env_cfg, AMP_STATE_HISTORY_V12)


@configclass
class G1AmpWalkV12RewardsCfg(G1AmpWalkV11RewardsCfg):
    """V11 的 21 项（逐字继承）+ ② 静止姿态权重 −0.3 → −1.0。

    只改 `weight`，`func` / `params`（`gate_tau`、`pos_joint_weights` 等）原样保留。
    """

    def __post_init__(self):
        super().__post_init__()
        base = self.track_default_pose_on_stand
        self.track_default_pose_on_stand = RewTerm(
            func=base.func,
            weight=AMP_WALK_V12_STAND_POSE_WEIGHT,
            params=base.params,
        )


@configclass
class G1AmpWalkV12EnvCfg(G1AmpWalkV11EnvCfg):
    """训练用：V11 的环境/奖励 + ① 2 帧判别器输入 + ② 站姿权重。"""

    rewards: G1AmpWalkV12RewardsCfg = G1AmpWalkV12RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _patch_amp_two_frame(self)  # ①
        self.run_name = AMP_WALK_V12_RUN_NAME


@configclass
class G1AmpWalkV12EnvCfg_PLAY(G1AmpWalkV11EnvCfg_PLAY):
    """回放用：与训练同观测（含 2 帧判别器输入）、同奖励、同分布、同增广。"""

    rewards: G1AmpWalkV12RewardsCfg = G1AmpWalkV12RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _patch_amp_two_frame(self)  # ①
        self.run_name = AMP_WALK_V12_RUN_NAME


class G1AmpWalkV12TeacherEnv(G1AmpWalkV11TeacherEnv):
    """契约期望值只能写在 env 类上（指令范围与 V10b/V11 相同）。

    ⚠️ 契约校验器检查的是**策略**观测（378）与指令范围；判别器输入长度不在契约里
    （`AMP_STATE_SIZE` 那组常量只是文档，运行时由 obs 群自动推断）。
    """

    expected_command_ranges = AMP_WALK_V10B_COMMAND_RANGES


__all__ = [
    "AMP_STATE_HISTORY_V12",
    "AMP_STATE_SIZE_V12",
    "AMP_WALK_V12_RUN_NAME",
    "AMP_WALK_V12_STAND_POSE_WEIGHT",
    "G1AmpWalkV12EnvCfg",
    "G1AmpWalkV12EnvCfg_PLAY",
    "G1AmpWalkV12RewardsCfg",
    "G1AmpWalkV12TeacherEnv",
    "_amp_group_with_history",
    "_patch_amp_history",
    "_patch_amp_two_frame",
]
