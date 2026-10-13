"""V17：**从零开始**训练人类步行风格（用户 2026-10-01 指定）。基准 = V16（10 帧 AMP）。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— 两项数值改动均已不生效**（见下表）。
* 仍"活着"的是**方法论/判读口径**：V15↔V16 的 10 帧消融结论；以及
  「本层与 V16 起点不同（从零 vs 热启动）⇒ **不能**直接比'风格好坏'」这条口径。

## 本层改动一览（⚠️ 本轮共 **3 处**差异，非单变量）

| # | 改动 | 现状（今日 T1l 实际生效）|
|---|---|---|
| ① | `flat_orientation_l2` −1 → **−2**（依据：倾角是唯一仍在恶化的量；θ=28° 时 −1 只收 0.127/步，而 `track_lin_vel_xy` +1.45/步 ⇒ 定价过低）| **已被覆盖** → **−1** |
| ② | `feet_air_time` 8.0 → **4.0（减半）**（压低「单脚跳刷分」收益）| **已被覆盖** → **+1**；且 V18 起换成 `feet_air_time_target_biped`，T1a2 又换回 `feet_air_time_positive_biped`（`threshold=0.5`）|
| ③ | **从零开始**（V16 是热启动 V11 `model_9000.pt`）| 起点不同 ⇒ 只能看「能否从零学会走路 + 姿态是否更直 + 步态是否自然」，**不能**与 V16 比风格好坏 |
| ④ | 只改 **`weight`**，`func`/`params`（`threshold 0.6`、`sensor_cfg`、`command_name`）逐字不变 | 已随后层整体重建 `RewTerm` 而失效 |

## 推理与判读（已归档，勿在本文件重复）

* 旧 `feet_air_time_positive_biped` 的**两个结构性漏洞**（短到几毫秒也给分 / 越长越给
  ⇒「单脚一直踩着、另一只吊着」拿满）→ §B2；并记「跳着刷分**不该靠压低权重去防**」
* 止损信号与会话（`Episode_Reward/feet_air_time ≥ 0.3` 且倾角不降 ⇒ 权重减半**没有**治好
  结构陷阱，下一版需把 air-time 改成「落地触发、每次迈步只付一次」）→ 该版 run 日志 +
  `RESULTS_amp_walk_v16_frame_ablation.md`
* 「从已塌陷的 ckpt 续训基本出不来自吸引子」→ §C4

⚠️ 「从零」必须面对的点：`motion_start_from_middle_range = (0.0, 0.0)`（**关闭 RSI**）
   ⇒ 参考 clip 从第 0 帧起、**不与策略相位对齐**，判别器只给「姿态是否像人」、不拉步频。
   若从零能站直但步态碎/不前进，下一刀应加**步频/相位**类奖励或打开 RSI（属 V18 变量）
   —— 关联 §附待办 **#4**（至今**没有 motion reset**，未闭环）。
⚠️ 其余逐字不动：环境（含 `joint_pos_limits_margin`、`track_default_pose_on_stand −0.3`）、
   其余 19 项奖励、10 帧判别器（670 维）、三档指令、2-clip 数据集、镜像增广/损失、
   σ 上限 0.6、固定 LR、λ 0.75 / c 1.0（有效风格权重 0.25）。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V17-G1-v0`
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .amp_walk_v14_env_cfg import G1AmpWalkV14RewardsCfg
from .amp_walk_v16_env_cfg import (
    G1AmpWalkV16EnvCfg,
    G1AmpWalkV16EnvCfg_PLAY,
    G1AmpWalkV16TeacherEnv,
)

#: ⭐ 改动①：直立惩罚 −1 → **−2**
AMP_WALK_V17_FLAT_ORIENTATION_WEIGHT = -2.0
#: ⭐ 改动②：反小碎步/反单脚的 `feet_air_time` 权重 8.0 → **4.0（减半）**
AMP_WALK_V17_AIR_TIME_WEIGHT = 4.0

AMP_WALK_V17_RUN_NAME = (
    "G1AmpWalkV17MVPPlane_obs378_rew21_amp10frame_flat2.0_airtime4.0_scratch"
)


@configclass
class G1AmpWalkV17RewardsCfg(G1AmpWalkV14RewardsCfg):
    """V14/V16 的 21 项（逐字继承）+ ① 直立惩罚 ×2 + ② air-time 权重减半。

    两项都**只改 `weight`**，`func` / `params`（含 air-time 的 `threshold 0.6`、
    `sensor_cfg=ankle_roll_link`、`command_name`）原样保留。
    """

    def __post_init__(self):
        super().__post_init__()

        base = self.flat_orientation_l2
        self.flat_orientation_l2 = RewTerm(
            func=base.func,
            weight=AMP_WALK_V17_FLAT_ORIENTATION_WEIGHT,
            params=base.params,
        )

        air = self.feet_air_time
        self.feet_air_time = RewTerm(
            func=air.func,
            weight=AMP_WALK_V17_AIR_TIME_WEIGHT,
            params=air.params,
        )


@configclass
class G1AmpWalkV17EnvCfg(G1AmpWalkV16EnvCfg):
    """训练用：V16（10 帧判别器 / 三档指令 / 2-clip / 限位软边距）+ 直立惩罚 ×2。"""

    rewards: G1AmpWalkV17RewardsCfg = G1AmpWalkV17RewardsCfg()

    def __post_init__(self):
        super().__post_init__()          # 其中会打 10 帧补丁（V16 的 __post_init__）
        self.run_name = AMP_WALK_V17_RUN_NAME


@configclass
class G1AmpWalkV17EnvCfg_PLAY(G1AmpWalkV16EnvCfg_PLAY):
    """回放用：与训练同观测（10 帧判别器）/同奖励/同分布/同增广。"""

    rewards: G1AmpWalkV17RewardsCfg = G1AmpWalkV17RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V17_RUN_NAME


class G1AmpWalkV17TeacherEnv(G1AmpWalkV16TeacherEnv):
    """契约期望值只写在 env 类上（指令范围与 V16/V15/V14/V12/V11/V10b 相同）。"""


__all__ = [
    "AMP_WALK_V17_AIR_TIME_WEIGHT",
    "AMP_WALK_V17_FLAT_ORIENTATION_WEIGHT",
    "AMP_WALK_V17_RUN_NAME",
    "G1AmpWalkV17EnvCfg",
    "G1AmpWalkV17EnvCfg_PLAY",
    "G1AmpWalkV17RewardsCfg",
    "G1AmpWalkV17TeacherEnv",
]
