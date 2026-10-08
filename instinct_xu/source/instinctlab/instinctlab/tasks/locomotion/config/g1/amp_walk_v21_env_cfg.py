"""V21 的环境侧：**刚体空间 195 维 × 8 帧**（判别器输入 **1560**）。

## 这一代在做什么（相对 V20 只改帧数；相对 V19b/V19d 只改表征）

| | V19b/V19d（已验证能走出正常步态） | V20（失败） | **V21** |
|---|---|---|---|
| 判别器输入表征 | 关节空间 5 项 = 67/帧 | 刚体 4 项 = 195/帧 | **刚体 4 项 = 195/帧** |
| 帧数 h | 10 | 2 | **8** |
| 判别器输入维 | 670 | 390 | **1560** |
| 起点 | 从零 | 从零 | **从零** |
| 奖励/指令/数据集 | V19d | 逐字同 V19d | **逐字同 V19d（不改任何奖励）** |

## 为什么 h=8（不是 2、也不是 10）

来自离线判别性研究 v2（`RESULTS_amp_frames_body_vs_joint.md`，同协议，参考流形 + 受控退化）：

| h | body195 AUC（时间结构类） | body195 `style_std` | 判别器输入维 |
|---|---|---|---|
| 2（=V20） | 0.625 | 0.0070 | 390 |
| 4 | 0.740 | 0.0222 | 780 |
| 6 | 0.853 | 0.0472 | 1170 |
| **8** | **0.890** | **0.0791** | **1560** |
| 10（=V19b 的帧数，但换成刚体） | 0.910 | 0.0988 | 1950 |
| 15 | 0.925 | 0.1303 | 2925 |

* **h=2 不够**（0.625）⇒ 这解释了 V20 为什么退回"高频小跳/原地踏步"模态、MuJoCo 里不走；
* **h=8 是"进入健康区"的拐点**：AUC 0.890 已**超过** legacy67@h=10 的 0.838（= V19b 走出正常步态时的水平），
  而输入维只 1560（比 h=10 方案省 20% 算力）；
* **不要 h≤4**：0.740 离健康区差得远。

## 与 V20 完全相同的口径（只把 history 从 2 改成 8）

13 体名单、anchor=`torso_link`、6D 取旋转矩阵前两列、体速打到自己刚体系、term-major 帧序 —— 逐字同 V20。
详见 `amp_walk_v20_env_cfg.py` 与 `envs/mdp/observations/amp_body_state.py` 的说明。

## ⚠️ 继承来的坑（务必保留这两行覆盖）

祖先链 `amp_walk_v16_env_cfg.__post_init__` → `_patch_amp_history()` 会用**基类 5 项模板**重建
`observations.amp_policy/amp_reference`，把本类声明的组整个冲掉（V20 已踩过）。
⇒ V21 的 `__post_init__` 必须在 `super().__post_init__()` **之后**重新赋一次，并加 `isinstance` 断言。
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .amp_walk_env_cfg import G1AmpWalkObservationsCfg
from .amp_walk_v19d_env_cfg import (
    G1AmpWalkV19DEnvCfg,
    G1AmpWalkV19DEnvCfg_PLAY,
    G1AmpWalkV19DTeacherEnv,
)
from .amp_walk_v19d_env_cfg import G1AmpWalkV19DRewardsCfg
from .amp_walk_v20_env_cfg import (
    AMP_V20_BODY_NAMES,
    AMP_V20_FRAME_SIZE,
    AMP_V20_STATE_TERMS,
    AmpV20PolicyStateObsCfg,
    AmpV20ReferenceStateObsCfg,
)

#: ⭐ 用户 2026-10-02 决定：把**偏航角速度跟踪**的权重对齐 AMP_mjlab（0.8 → 1.0）。
#:
#: ⚠️ **只对齐权重，不抄它的 `std=3.14`**。两边口径对比：
#:
#: | | 本仓库 | AMP_mjlab |
#: |---|---|---|
#: | 权重 | 0.8 → **1.0** | 1.0 |
#: | `std` | **0.5** | 3.14（软 6 倍） |
#: | 取用 | `root_ang_vel_w[:,2]` | `torso_ang_vel_w[:,2]` |
#: | 形式 | `exp(-e²/std²)` | 同 |
#:
#: 若连 `std` 一起抄（0.5→3.14），该项会软 **~25 倍**（ωz 误差 0.5 rad/s 时每步惩罚
#: 0.506 → 0.025），与“用偏航阻尼抑制原地踏步”的意图**正好相反** ⇒ **std 保持 0.5**。
#: 若要更强的抗踏步，下一步应该动 **`std`**（0.5→0.3）而不是权重。
#: 本常量是可回退的总开关：改回 0.8 即“V21 = 纯单变量（只代表征/帧数）”。
AMP_WALK_V21_ANG_VEL_Z_WEIGHT = 1.0

#: ⭐ 本轮唯一变量：判别器输入帧数（V20 是 2）
AMP_V21_STATE_HISTORY = 8
#: 判别器输入维度：8 × 195 = **1560**
AMP_V21_STATE_SIZE = AMP_V21_STATE_HISTORY * AMP_V20_FRAME_SIZE

AMP_WALK_V21_RUN_NAME = (
    "G1AmpWalkV21MVPPlane_obs378_rew21_amp13body8frame_flat2.0_airband_exptau0.08"
    "_motiongate0.15_standpen0.5_ar0.06_standpose0.5_5clip_omnidir"
)


def _group_with_history(group_cls, history: int):
    """**新建**一个 obs 群实例，把刚体 4 项的 `history_length` 全改成 `history`。

    与 `amp_walk_v12_env_cfg._amp_group_with_history` 同一手法：
    刻意不原地修改 V20 里共享的类属性实例，否则同进程内 V20 的 cfg 会被一起改掉。
    """
    grp = group_cls()
    for name in AMP_V20_STATE_TERMS:
        getattr(grp, name).history_length = history
    return grp


@configclass
class G1AmpWalkV21RewardsCfg(G1AmpWalkV19DRewardsCfg):
    """V19d 的 21 项（逐字继承）+ **仅一项权重对齐**：`track_ang_vel_z_exp` 0.8 → 1.0。

    只改 `weight`；`func`（`locomotion_mdp.track_ang_vel_z_world_exp`，在
    `tasks/locomotion/mdp/rewards.py:60`）与 `std=0.5` 原样保留（见
    `AMP_WALK_V21_ANG_VEL_Z_WEIGHT` 的注释：**不抄 mjlab 的 std=3.14**）。
    """

    def __post_init__(self):
        super().__post_init__()
        base = self.track_ang_vel_z_exp
        self.track_ang_vel_z_exp = RewTerm(
            func=base.func,
            weight=AMP_WALK_V21_ANG_VEL_Z_WEIGHT,
            params=base.params,
        )


@configclass
class G1AmpWalkV21ObservationsCfg(G1AmpWalkObservationsCfg):
    """actor/critic 不动（378 / flat critic）；判别器两组换成"刚体 195 × 8 帧"。"""

    amp_policy: AmpV20PolicyStateObsCfg = _group_with_history(AmpV20PolicyStateObsCfg, AMP_V21_STATE_HISTORY)
    amp_reference: AmpV20ReferenceStateObsCfg = _group_with_history(
        AmpV20ReferenceStateObsCfg, AMP_V21_STATE_HISTORY
    )


@configclass
class G1AmpWalkV21EnvCfg(G1AmpWalkV19DEnvCfg):
    """训练用：奖励 = V19d 逐字 + ① 权重对齐，判别器输入 = 刚体 195 × 8 帧。"""

    rewards: G1AmpWalkV21RewardsCfg = G1AmpWalkV21RewardsCfg()
    observations: G1AmpWalkV21ObservationsCfg = G1AmpWalkV21ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # ⚠️ 必须在 super() 之后再赋一次（父链的 _patch_amp_history 会用基类模板覆盖这两组）
        self.observations.amp_policy = _group_with_history(AmpV20PolicyStateObsCfg, AMP_V21_STATE_HISTORY)
        self.observations.amp_reference = _group_with_history(
            AmpV20ReferenceStateObsCfg, AMP_V21_STATE_HISTORY
        )
        assert isinstance(self.observations.amp_policy, AmpV20PolicyStateObsCfg), (
            "V21 的 amp_policy 被父类的 _patch_amp_history 覆盖了！"
            f" 实际类型 = {type(self.observations.amp_policy).__name__}"
        )
        assert isinstance(self.observations.amp_reference, AmpV20ReferenceStateObsCfg), (
            "V21 的 amp_reference 被父类的 _patch_amp_history 覆盖了！"
            f" 实际类型 = {type(self.observations.amp_reference).__name__}"
        )
        self.run_name = AMP_WALK_V21_RUN_NAME


@configclass
class G1AmpWalkV21EnvCfg_PLAY(G1AmpWalkV19DEnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广。"""

    rewards: G1AmpWalkV21RewardsCfg = G1AmpWalkV21RewardsCfg()
    observations: G1AmpWalkV21ObservationsCfg = G1AmpWalkV21ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.amp_policy = _group_with_history(AmpV20PolicyStateObsCfg, AMP_V21_STATE_HISTORY)
        self.observations.amp_reference = _group_with_history(
            AmpV20ReferenceStateObsCfg, AMP_V21_STATE_HISTORY
        )
        self.run_name = AMP_WALK_V21_RUN_NAME


class G1AmpWalkV21TeacherEnv(G1AmpWalkV19DTeacherEnv):
    """指令范围与 V19d/V20 相同（全向四档）⇒ 期望值直接继承。"""


__all__ = [
    "AMP_V21_STATE_HISTORY",
    "AMP_V21_STATE_SIZE",
    "AMP_WALK_V21_ANG_VEL_Z_WEIGHT",
    "AMP_WALK_V21_RUN_NAME",
    "G1AmpWalkV21EnvCfg",
    "G1AmpWalkV21EnvCfg_PLAY",
    "G1AmpWalkV21ObservationsCfg",
    "G1AmpWalkV21RewardsCfg",
    "G1AmpWalkV21TeacherEnv",
]
