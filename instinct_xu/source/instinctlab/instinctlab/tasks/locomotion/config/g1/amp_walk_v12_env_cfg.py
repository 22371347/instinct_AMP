"""V12：把 AMP 判别器输入换成 LLP 论文的「两帧状态转移」(s_t, s_{t+1}) + 抬高静止姿态权重。

## ⚠️ 本层对**今天**仍有影响的部分

* ⭐ **「2 帧 = 论文口径」这条结论今天仍站得住**：今天 T1l 的判别器输入也是 **2 帧**
  （`AMP_V29_STATE_HISTORY = 2`）—— 但**状态构成与维度完全不同**：本层是旧的
  5 项模板 `67×2 = 134`，今天自 V20 起换成 **13body×15×2 = 390** → V33 再补 anchor 重力
  = **396 维**。⇒ 帧数"回归"，实现**不**是同一套。
* ⭐ **判别器**在**ckpt 里**（`wasabi.py` override 了 `state_dict()`）⇒ 换判别器形状时必须用
  `ckpt_manipulator` 丢掉，否则首次启动即崩 —— 今天仍是硬性约束。
* ② 的数值已被后层覆盖（见下表）。

## 本层改动一览（⚠️ 本轮同时动 **2 个量**，归因必须分开）

| # | 改动 | 现状（今日 T1l 实际生效）|
|---|---|---|
| ① | 判别器输入 10 → **2 帧**（`_patch_amp_two_frame`；`67×2 = 134` 维）| 帧数 ✅ 今天也是 **2**；但**构成/维度不同** → 13body×15×2 = 390 → +anchor 重力 = **396**（V20/V29/V33）|
| ② | `track_default_pose_on_stand` −0.3 → **−1.0**（用户要求：模型静止时更稳）| **已被 V23 覆盖** → **−0.8**（并新增全关节 `pos_joint_weights`/tol 表、`vel_scale 0.8`）|
| ③ | 判读口径：两变量分开看（风格看 `amp_d`/`r_amp`；静止稳定看 `track_default_pose_on_stand`/`base_ang_vel`）| 方法论仍有效（§E3 单变量）|

## 推理与判读（已归档，勿在本文件重复）

* ① 的推理「10 帧的姿态轮廓/抖动统计本身就足以把 policy 与 reference 分开 ⇒ 判别器没有压力去
  学'单步转移像不像人'」→ §A2（离线：h=1–2 对**时间结构类**退化完全瞎）
* ② 的代价曾被**严重低估**（V12 掉掉的 ~18.5/回合里一半以上可直接归到站姿权重；
  `Mean reward` 27.73→9.28、`feet_air_time` 0.160→0.099 更碎步）→ §B1
* ⚠️ 历史教训：V9c 记的它「−15.3/回合」是**门控读错数组**时的数（门控近似恒开），与现在不可比
  → 关联 §B1 / 该版 run 日志

⚠️ 算法侧热启动的关键（勿删）：`instinct_rl/algorithms/wasabi.py` **override 了**
   `state_dict()`/`load_state_dict()`，把 `"discriminator"` 与 `"discriminator_optimizer"`
   也写进 ckpt（早先只看 `ppo.py` 得出的「判别器不在 ckpt 里」是**错的**，第一次起 V12 就因此崩：
   `size mismatch for model.model.0.weight: [1024, 670] vs [1024, 134]`）
   ⇒ 必须设 `ckpt_manipulator = "instinctlab.utils.ckpt_manipulators:drop_amp_discriminator"`。
⚠️ 若要做严格单变量归因，下一轮把 ② 撤回即可（= V12b，只留 2 帧判别器）。
⚠️ 其余逐字不动：21 项奖励结构（含 `feet_swing_time_penalty −0.5`、`feet_air_time 8.0/thr0.6`、
   `action_rate_l2 −0.06`、`dof_acc_l2 −1e−6`）、三档指令、2-clip 数据集、镜像增广/损失、
   σ 上限 0.6、固定 LR、算法超参（λ=0.5、c=0.5 ⇒ 有效风格权重 0.25）。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V12-G1-v0`
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
