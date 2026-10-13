"""V24：**V23 + `dof_acc_l2` 的作用关节从「髋/膝」扩到「全关节」**（用户 2026-10-03 指定；权重不变）。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— 本层唯一的改动已被 T1a2 覆盖**（见下表）⇒ 今天（T1l）`dof_acc_l2` 回到 `-2e-07`
  且**只作用 `.*_hip_.*` / `.*_knee_joint`**。
* 仍"活着"的只有**方法论**：本层是本项目的**标准单变量**样板（只扩覆盖面、`weight`/`func` 一字不动），
  外加一组**只写日志、不改罚额**的探针（`Step/hf_acc_all/legs/arms/waist`）——
  后续版本（V26/V29…）都沿用这套"高频"读数。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | ⭐ 唯一变量：`dof_acc_l2` 的 `asset_cfg` 由 `[.*_hip_.*, .*_knee_joint]` 改为 **`.*`（全 29 关节）**；`weight` 仍 `-1.0e-6`、`func` 仍是内建 `joint_acc_l2`（= AMP_mjlab 的 `joint_acc_l2`：无 `asset_cfg`）| ❌ **已被 T1a2 覆盖** → `-2e-07` + 只作用 `.*_hip_.*` / `.*_knee_joint` |
| ② | 在 `track_default_pose_on_stand` 内新增**只读探针** `Step/hf_acc_all/legs/arms/waist`（未乘权重的 `mean|ddq|`）| 代码侧读数，不改变任何罚额；判读见该版 run 日志 |

## 推理与判读（已归档，勿在本文件重复）

* ⭐ 历史教训（v2→v3）：**从已塌陷的 ckpt 续训基本出不来自吸引子** ⇒ 塌陷后别指望"续训救回来"，
  直接改配置重学 → `docs/amp_walk/EXPERIMENTS.md` §C4
  （本层起点因此 = **从零**；V23 的 ckpt 保留备查）
* 本层"为什么"的物理依据：加速度 L2 对频率是 **4 次方**（a ∝ A(2πf)² ⇒ 罚 a² ∝ A²f⁴），
  对"高频小跳"的选择性比 `action_rate_l2`（一阶差分，f²）高**两个倍频程** ⇒ 赌它能"专打高频而不干扰正常步态"
* V23 的实测（`air_time_hop_frac` 0.974 / `in_band` 0.092 / `single_stance` 0.016 s / `amp_style_std` 0.0031
  —— 全谱系最差）+"三条改动各自在工作、却都碰不到高频"的归因
  → `RESULTS_v22_whitening.md` + §A6（判别器饱和是常态）+ 该版 run 日志

⚠️ 其余逐字不动：V23 的 21 项奖励结构、白化/进度门/死区口径、5-clip 数据集、全向指令、判别器（13body×8 帧）。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V24-G1-v0`
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .amp_walk_v23_env_cfg import (
    G1AmpWalkV23EnvCfg,
    G1AmpWalkV23EnvCfg_PLAY,
    G1AmpWalkV23RewardsCfg,
)
from .amp_walk_v19d_env_cfg import G1AmpWalkV19DTeacherEnv
from .amp_walk_v11_env_cfg import AMP_WALK_V11_DOF_ACC_WEIGHT

AMP_WALK_V24_RUN_NAME = (
    "G1AmpWalkV24MVPPlane_obs378_rew21_amp13body8frame_white_progress0.35"
    "_posdead0.8_accall5clip_omnidir"
)


@configclass
class G1AmpWalkV24RewardsCfg(G1AmpWalkV23RewardsCfg):
    """V23 的 21 项 + **只重写 `dof_acc_l2` 的作用关节**（权重、func 逐字不动）。"""

    def __post_init__(self):
        super().__post_init__()
        # ⚠️ 只删 `asset_cfg`（⇒ 全关节，与 AMP_mjlab 的 `joint_acc_l2` 一致）；
        #    `func`（isaaclab 内建 `joint_acc_l2`）与 `weight` 都不动
        #    ⇒ 这是一个纯"扩大覆盖面"的单变量。
        self.dof_acc_l2 = RewTerm(
            func=self.dof_acc_l2.func,
            weight=AMP_WALK_V11_DOF_ACC_WEIGHT,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        )


@configclass
class G1AmpWalkV24EnvCfg(G1AmpWalkV23EnvCfg):
    """训练用：V23 + `dof_acc_l2` 全关节。"""

    rewards: G1AmpWalkV24RewardsCfg = G1AmpWalkV24RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards = G1AmpWalkV24RewardsCfg()
        self.run_name = AMP_WALK_V24_RUN_NAME


@configclass
class G1AmpWalkV24EnvCfg_PLAY(G1AmpWalkV23EnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广。"""

    rewards: G1AmpWalkV24RewardsCfg = G1AmpWalkV24RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards = G1AmpWalkV24RewardsCfg()
        self.run_name = AMP_WALK_V24_RUN_NAME


class G1AmpWalkV24TeacherEnv(G1AmpWalkV19DTeacherEnv):
    """指令范围与 V19d/V20/V21/V23 相同（全向四档）⇒ 期望值直接继承。"""


__all__ = [
    "AMP_WALK_V24_RUN_NAME",
    "G1AmpWalkV24EnvCfg",
    "G1AmpWalkV24EnvCfg_PLAY",
    "G1AmpWalkV24RewardsCfg",
    "G1AmpWalkV24TeacherEnv",
]
