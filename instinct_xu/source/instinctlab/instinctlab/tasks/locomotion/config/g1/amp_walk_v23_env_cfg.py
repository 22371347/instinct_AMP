"""V23：**奖励结构 + 判别器输入** 三处改动合并（用户 2026-10-03 指定；基准 = V21）。

## ⚠️ 本层对**今天**仍有影响的部分

* ⭐⭐ **② 重建的 `track_default_pose_on_stand` 今天仍在整层生效** —— 本层用
  `RewTerm(func=..., weight=-AMP_V23_POSE_WEIGHT, params=params)` **整体重建**该项
  （不是只改 weight），后层只沿用、未再重建 ⇒ 今天（T1k/T1l）实际参数 =
  `weight −0.8` / `pos_scale 4.0` / **`vel_scale 0.8`** / `gate_tau 0.25` /
  `pos_joint_weights {waist_yaw 3.0, hip 1.5, **knee 2.0**, **ankle_pitch 2.0**, **ankle_roll 2.0**}` /
  全关节 `pos_joint_tol` + `vel_joint_tol` / `pos_tol_default 0.08` / `vel_tol_default 0.5`。
* ⭐ 这是今天的一处**隐藏覆盖**：V5 原值**只有一个 `waist_yaw_joint: 3.0`**（其余关节等权、无死区）
  ⇒ 与 **2026-10-08 的 `track_base_height` `std 0.2` 事故**属**同一类**问题
  （只看本层源码定义处会读错实际生效值）→ `docs/amp_walk/EXPERIMENTS.md` §P2。
* 其余：① 已被 T1a2 覆盖、③ 属算法侧，均见表。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | `feet_air_time` 门控/负价判据：`motion_threshold 0.15→0.0`（删掉"实测瞬时幅值"那道门）+ 改用**进度判据** `progress`（方向投影 + EMA 低通）+ 负价条件 `stand → no_progress` | ❌ **已被 T1a2 覆盖** → 换回 `feet_air_time_positive_biped`（`threshold=0.5`）、权重 1.0（整条"目标带"实现一并失效）|
| ② | ⭐ `track_default_pose_on_stand`：29 关节**等权均值 / 无死区**、weight −0.5 → **逐关节死区** `tol_j` + **内部权重** `w_j`（按 `Σw_j` 归一）+ weight **−0.8** | ✅ **本层是当前生效层**（T1k/T1l 逐字沿用，见上）|
| ③ | AMP 判别器输入**白化**（`EmpiricalNormalization(eps=0.01)`，expert-only 统计）+ GP 移到**归一化空间**〔算法侧〕| 见 `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V23-G1-v0` 的「算法侧」节（算法侧值不在此抄录；今天是否仍生效见同表后层各臂）|

## 推理与判读（已归档，勿在本文件重复）

* ① 的数学（**必须带符号**：踏步根速度 ±0.5 交替，取 `relu` 只砍负半边 ⇒ `EMA≈+0.25` 会假过门；
  `κ=0.35` 对四档指令自动标定；`τ=0.3 s` ⇒ 起步 ~0.14 s 延迟；纯转向档必须要求 ω 指令本身非零）
  —— 实现在 `mdp/amp_reward_utils.py`（本地单测 19/19 通过）→ 该版 run 日志
* ① 的动机：`motion_threshold=0.15` 那道门在三代 run 里关闭率仅 0.46%/0.46%/0.63~0.95%，
  从未拦住踏步、且对符号不敏感 ⇒ 删掉 → `RESULTS_v19cd_v20_stand_probes.md`
* ② 的动机：per-joint 偏差里**臂 14 关节贡献 58%、腿只 35%** ⇒ 该项一直"主要在管手臂"；
  死区把"站立微调（±3°）"与"抬脚级偏离（15~30°）"分开 → `RESULTS_v19cd_v20_stand_probes.md`；
  ⚠️ 其**价格平衡未实测校准** → §附待办第 3 条
* ③ 的动机：三代判别器**全线饱和** ⇒ 必须让判别器"敢说话"；白化后 GP 留在原始空间会被 `1/eps` 放大
  ⇒ **成对改**（`GP coef` 沿用 1.0）→ §A4（raw GP 学不了区分）/ §A6（饱和是常态）
* ⚠️ **本版是合并改动**（三项一起上）⇒ 事后归因只能靠各自探针（① `air_time_progress_frac`/`air_time_proj_ema`/
  `air_time_stand_pen_mean`/`air_time_lift_frac`；② `stand_pose_pos_cost`/`stand_pose_vel_cost`/`stand_pose_{pos,vel}_err_{legs,arms,waist}`；
  ③ `amp_d`/`amp_d_ref`/`amp_style_std`/`Loss/discriminator_loss`）；若某项无效可单独回退
  （① `progress_kappa=0`；② 恢复 weight 0.5 且 `pos_tol_default=0`；③ 去掉 normalizer + GP 回 `raw`）→ §E3

⚠️ 其余逐字不动：数据集/指令分布/其余 19 项奖励**逐字沿用 V19d**（V21 只是把 `track_ang_vel_z_exp` 权重对齐到 1.0）；
   起点 **从零**（判别器多了 normalizer buffer 且白化改变了输入分布的定义）。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V23-G1-v0`
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import instinctlab.tasks.locomotion.mdp as locomotion_mdp

from .amp_walk_v21_env_cfg import (
    G1AmpWalkV21EnvCfg,
    G1AmpWalkV21EnvCfg_PLAY,
    G1AmpWalkV21RewardsCfg,
)
from .amp_walk_v20_env_cfg import (
    AmpV20PolicyStateObsCfg,
    AmpV20ReferenceStateObsCfg,
    AMP_V20_STATE_TERMS,
)
from .amp_walk_v19d_env_cfg import G1AmpWalkV19DTeacherEnv

_SIDES = ("left", "right")
_ARM_PARTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
_LEG_PARTS_W = (("hip_pitch", 1.5), ("hip_roll", 1.5), ("hip_yaw", 1.5), ("knee", 2.0), ("ankle_pitch", 2.0), ("ankle_roll", 2.0))

#: ⭐ ① `feet_air_time` 的进度判据
AMP_V23_PROGRESS_KAPPA = 0.35       # 实际沿指令方向的速度必须 ≥ κ·‖v_cmd‖
AMP_V23_PROGRESS_TAU = 0.3          # EMA 时间常数（s）
AMP_V23_PROGRESS_OMEGA_KAPPA = 0.3  # 纯转向档放行阈值
AMP_V23_MOTION_THRESHOLD = 0.0      # ⭐ 删掉"实测瞬时幅值"门（关闭率 <1%，见文件头）
AMP_V23_STAND_PENALTY = 0.5         # 负价幅度（沿用 V19d），但条件改为 no_progress

#: ⭐ ② `track_default_pose_on_stand`
AMP_V23_POSE_WEIGHT = 0.8                                 # 0.5 → 0.8（带死区后 cost 变小）
AMP_V23_POSE_VEL_SCALE = 0.8                              # 0.5 → 0.8（速度项要配死区）
AMP_V23_POSE_TOL_DEFAULT = 0.08                           # 未列出关节的死区（rad）
AMP_V23_POSE_VEL_TOL_DEFAULT = 0.5                        # 未列出关节的速度死区（rad/s）
#: 内部权重（优先级；未列出的关节 = 1.0）。腿/踝抬脚主力 ⇒ >1；腰偏航沿用 3.0 防"锁 90°"。
AMP_V23_POSE_JOINT_WEIGHTS = {
    "waist_yaw_joint": 3.0,
    **{f"{s}_{p}_joint": w for s in _SIDES for p, w in _LEG_PARTS_W},
}
#: 逐关节死区：臂/腰 0.05（站姿应贴近 default）、髋/膝 0.08、踝 0.12（承重微调多）
AMP_V23_POSE_JOINT_TOL = {
    **{f"{s}_{p}_joint": 0.05 for s in _SIDES for p in _ARM_PARTS},
    "waist_yaw_joint": 0.05,
    "waist_roll_joint": 0.05,
    "waist_pitch_joint": 0.05,
    **{f"{s}_{p}_joint": (0.12 if p.startswith("ankle") else 0.08) for s in _SIDES for p in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")},
}
#: 速度死区（rad/s）：腿在行走时本来就快，站姿噪声也大 ⇒ 腿给大死区，臂/腰小
AMP_V23_POSE_VEL_TOL = {
    **{f"{s}_{p}_joint": 1.0 for s in _SIDES for p in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")},
    **{f"{s}_{p}_joint": 0.5 for s in _SIDES for p in _ARM_PARTS},
    "waist_yaw_joint": 0.5,
    "waist_roll_joint": 0.5,
    "waist_pitch_joint": 0.5,
}

AMP_WALK_V23_RUN_NAME = (
    "G1AmpWalkV23MVPPlane_obs378_rew21_amp13body8frame_white_progress0.35"
    "_posdead0.8_5clip_omnidir"
)


def _group_with_history(group_cls, history: int):
    grp = group_cls()
    for name in AMP_V20_STATE_TERMS:
        getattr(grp, name).history_length = history
    return grp


@configclass
class G1AmpWalkV23RewardsCfg(G1AmpWalkV21RewardsCfg):
    """V21 的 21 项 + 只重写 2 项（`feet_air_time` / `track_default_pose_on_stand`）。

    其余 19 项（含 V21 对齐过的 `track_ang_vel_z_exp = 1.0`）**逐字不动**。
    """

    def __post_init__(self):
        super().__post_init__()

        # ---- ① feet_air_time：进度判据（方向投影 + EMA 低通）+ no_progress 负价 ----
        base = self.feet_air_time
        params = dict(base.params)
        params.update(
            {
                # ⭐ 删掉那道"实测瞬时幅值"门（关闭率 <1%，且反向激励）
                "motion_threshold": AMP_V23_MOTION_THRESHOLD,
                # ⭐ 进度判据（`progress_kappa > 0` 才启用）
                "progress_kappa": AMP_V23_PROGRESS_KAPPA,
                "progress_tau": AMP_V23_PROGRESS_TAU,
                "progress_omega_kappa": AMP_V23_PROGRESS_OMEGA_KAPPA,
                # 负价幅度不变，但条件由 `stand`（指令≈0）推广为 `no_progress`
                "stand_penalty": AMP_V23_STAND_PENALTY,
            }
        )
        self.feet_air_time = RewTerm(func=base.func, weight=base.weight, params=params)

        # ---- ② track_default_pose_on_stand：逐关节死区 + 内部权重 + 权重 0.8 ----
        base = self.track_default_pose_on_stand
        params = dict(base.params)
        params.update(
            {
                "pos_joint_weights": AMP_V23_POSE_JOINT_WEIGHTS,
                "pos_joint_tol": AMP_V23_POSE_JOINT_TOL,
                "pos_tol_default": AMP_V23_POSE_TOL_DEFAULT,
                "vel_joint_tol": AMP_V23_POSE_VEL_TOL,
                "vel_tol_default": AMP_V23_POSE_VEL_TOL_DEFAULT,
                "vel_scale": AMP_V23_POSE_VEL_SCALE,
            }
        )
        self.track_default_pose_on_stand = RewTerm(
            func=base.func, weight=-AMP_V23_POSE_WEIGHT, params=params
        )


@configclass
class G1AmpWalkV23EnvCfg(G1AmpWalkV21EnvCfg):
    """训练用：V21（刚体 195×8）+ 奖励结构两项（见文件头表格）。"""

    rewards: G1AmpWalkV23RewardsCfg = G1AmpWalkV23RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # ⚠️ 父链的 `_patch_amp_history` 会覆盖这两组 ⇒ 必须在 super() 之后再赋一次（V21 同款）
        self.observations.amp_policy = _group_with_history(AmpV20PolicyStateObsCfg, 8)
        self.observations.amp_reference = _group_with_history(AmpV20ReferenceStateObsCfg, 8)
        # 让奖励子类生效（configclass 的赋值顺序有时序问题，这里再兜一次）
        self.rewards = G1AmpWalkV23RewardsCfg()
        self.rewards.feet_air_time.params["progress_kappa"] = AMP_V23_PROGRESS_KAPPA
        self.rewards.track_default_pose_on_stand.weight = -AMP_V23_POSE_WEIGHT
        self.run_name = AMP_WALK_V23_RUN_NAME


@configclass
class G1AmpWalkV23EnvCfg_PLAY(G1AmpWalkV21EnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广。"""

    rewards: G1AmpWalkV23RewardsCfg = G1AmpWalkV23RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.amp_policy = _group_with_history(AmpV20PolicyStateObsCfg, 8)
        self.observations.amp_reference = _group_with_history(AmpV20ReferenceStateObsCfg, 8)
        self.rewards = G1AmpWalkV23RewardsCfg()
        self.run_name = AMP_WALK_V23_RUN_NAME


class G1AmpWalkV23TeacherEnv(G1AmpWalkV19DTeacherEnv):
    """指令范围与 V19d/V20/V21 相同（全向四档）⇒ 期望值直接继承。"""


__all__ = [
    "AMP_V23_PROGRESS_KAPPA",
    "AMP_V23_PROGRESS_TAU",
    "AMP_V23_PROGRESS_OMEGA_KAPPA",
    "AMP_V23_MOTION_THRESHOLD",
    "AMP_V23_STAND_PENALTY",
    "AMP_V23_POSE_WEIGHT",
    "AMP_V23_POSE_VEL_SCALE",
    "AMP_V23_POSE_TOL_DEFAULT",
    "AMP_V23_POSE_VEL_TOL_DEFAULT",
    "AMP_V23_POSE_JOINT_WEIGHTS",
    "AMP_V23_POSE_JOINT_TOL",
    "AMP_V23_POSE_VEL_TOL",
    "AMP_WALK_V23_RUN_NAME",
    "G1AmpWalkV23EnvCfg",
    "G1AmpWalkV23EnvCfg_PLAY",
    "G1AmpWalkV23RewardsCfg",
    "G1AmpWalkV23TeacherEnv",
]
