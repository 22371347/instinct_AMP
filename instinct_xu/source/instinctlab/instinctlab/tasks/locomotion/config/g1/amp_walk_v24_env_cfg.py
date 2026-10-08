"""V24：**V23 + `dof_acc_l2` 改为全关节**（用户 2026-10-03 指定；权重不变）。

## 本轮唯一变量

| 项 | V23 | **V24** |
|---|---|---|
| `dof_acc_l2` 权重 | −1.0e-6（= mjlab 的 4×） | **不变** −1.0e-6 |
| `dof_acc_l2` 作用关节 | ⚠️ **只有 `.*_hip_.*` + `.*_knee_joint`**（V2 的刻意偏离） | **全关节**（= `AMP_mjlab` 的 `joint_acc_l2`：无 `asset_cfg`） |

## 为什么（两条依据）

1. **物理**：加速度 L2 对频率是 **4 次方**（正弦 $a\\propto A(2\\pi f)^2$ ⇒ 罚 $a^2\\propto A^2 f^4$）——
   对"高频小跳"的选择性是 `action_rate_l2`（动作一阶差分，$f^2$）的**两个倍频程**，
   也是位置类项（$f^0$）的**四个倍频程**。所以极小权重就能"专打高频而不干扰正常步态"。
2. **我们的配置错在关节集**：mjlab 的 `joint_acc_l2 = −2.5e-7` **无 `asset_cfg` ⇒ 全关节**；
   我方从 V2 起就**只罚髋/膝**，而"高频小跳"恰恰主要是**踝（足底屈/离地）+ 腰（俯仰）**的运动
   ⇒ 罚不到病灶。历史证据：V11 已把权重 ×4（−2.5e-7 → −1.0e-6），
   V12→V20 仍反复出现小跳（V14 对照表：`dof_acc_l2` −0.10 → −0.16 反而恶化）
   ⇒ "只加权重不扩关节"这条路已试过且不够。

## V23 的实测（iters 0–2780）—— 为什么必须换杠杆

| 指标 | V19c（基线） | V20 | **V23（白化+进度门+死区）** | 判读 |
|---|---|---|---|---|
| `air_time_hop_frac`（小跳占比） | 0.68 | 0.89 | **0.974** ❌ | 全谱系**最差** |
| `air_time_in_band_frac` | 0.535 | 0.26 | **0.092** ❌ | 10× 差距 |
| `air_time_single_stance` | 0.079 s | 0.037 s | **0.016 s** ❌ | 高频碎步 |
| `Step/amp_style_std` | 0.130 | 0.094 | **0.0031** ❌ | 判别器过早死透 |
| `Loss/discriminator_loss` | 0.0046 | 0.0027 | **0.0001** ❌ | ~460 轮就饱和 |
| `air_time_progress_frac` | — | — | 0.675 ✅ | 进度判据"以为 67% 的步在走" |
| `air_time_proj_ratio` | — | — | 0.697 ✅ | 超过 κ=0.35 |
| `stand_pose_pos_err` | 0.291 | — | **0.207** ✅ | 死区让姿态偏差降 29% |
| `stand_pose_vel_err` | 5.17 | — | 4.94 ❌ | **速度（抖动）没降** |

⇒ **三条改动各自"在工作"，但都碰不到"高频"本身**：
* 进度判据只能识别"原地踏步"，而高顰小跳**确实产生净位移**（`track_lin_vel 1.084` 与历史持平）⇒ 它把"小跳着往前走"判成了"在走"；
* 死区/权重只降了**位置**分量，**速度没降** ⇒ 抖动依旧；
* 白化提升了可分性，却把判别器**推向更早饱和**（详见报告）⇒ 风格项实际已死。

所以本轮直接上"高频专用"的杠杆：**加速度罚（f⁴）+ 全关节覆盖**。

## 新增探针（用于验证"究竟是谁在高频抖"）

在 `track_default_pose_on_stand` 内新增（**只写日志，不改任何罚项量级**）：
`Step/hf_acc_all` / `hf_acc_legs` / `hf_acc_arms` / `hf_acc_waist`（未乘权重的 `mean|ddq|`）。

## 起点

**从零**。V23 在 2500 轮已确定进入"高频小跳"模态（与 V20 同型）——
历史教训（v2→v3）：**从已塌陷的 ckpt 续训基本出不来自吸引子**，而改动本身只是扩了一个罚项的覆盖面，
从零重学 ~2500 轮 ≈ 3 h 是可接受的代价。V23 的 ckpt 保留，必要时仍可续跑对照。
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
