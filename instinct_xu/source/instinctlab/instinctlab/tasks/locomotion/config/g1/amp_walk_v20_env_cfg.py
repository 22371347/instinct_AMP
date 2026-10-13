"""V20：把判别器输入从"关节空间 67 维"换成"刚体空间 195 维"（照抄 AMP_mjlab），帧数用 2（论文式 `(s_t, s_{t+1})`）。

## ⚠️ 本层对**今天**仍有影响的部分

* ⭐⭐ **本层的刚体空间 AMP 状态（13 body × 15 = 195/帧）今天仍在使用** —— 本层新增的
  `AmpV20PolicyStateObsCfg` / `AmpV20ReferenceStateObsCfg` 被 **V29 复用**、被 **V33 直接继承**
  （V33 追加第 5 项 `anchor_gravity_b`）⇒ 今天（T1k/T1l）判别器输入 = **13body×15×2 帧 + anchor 重力 = 396 维**
  （grep 旁证 `amp_walk_v29_env_cfg.py:54,70`、`amp_walk_v33_env_cfg.py:139-172`；**未运行期核对**）。
* ⭐ **本层的踩坑今天仍是硬性约束**：祖先链 `amp_walk_v16_env_cfg.__post_init__` 会调
  `amp_walk_v12_env_cfg._patch_amp_history()`，后者用**基类的 5 项模板**重建 `amp_policy` / `amp_reference`
  （并把 history 拉回 10）⇒ 把子类声明的组**整个冲掉**（症状：判别器输入悄无声息退回旧维度）。
  **修法**：子类在 `super().__post_init__()` **之后**重赋一次 + 加 `isinstance`/维度断言
  （V21/V23/V29/V30/V31/V32/V33 全部照抄这一修法）。
* **本层口径（今天仍沿用，见 `amp_body_state.py`）**：anchor = `torso_link`；13 体 = `pelvis` + 双髋 roll
  + 双膝 + 双踝 roll + 双肩 roll + 双肘 + 双腕；**6D 取旋转矩阵前两列**（mjlab 专家侧口径）；帧序
  **term-major**（与 mjlab 的 `[s, s']` 只差一个固定置换 ⇒ 对 MLP 等价，两侧一致）。
  * ⚠️ 腕用 `wrist_roll_link`（**不是** mjlab 的 `wrist_yaw_link`）—— 本任务的
    `motion_reference.link_of_interests`（及 npz 里存的 link 集）只有 `wrist_roll_link`，
    换名要重跑动作转换 ⇒ 取结构等价替代；维度仍严格 **13×15 = 195**。
  * 本层 4 项对**全局平移/偏航不敏感** ⇒ **看不见静态倾角**；原文据此**刻意保留**
    `track_base_height` / `flat_orientation_l2` 管姿态（用户 2026-10-02：整体倾斜已由奖励函数解决）。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | ⭐ AMP 状态项：`projected_gravity` + `joint_pos_rel` + `joint_vel_rel` + `base_lin_vel` + `base_ang_vel`（**67/帧**）→ **`body_pos_b` + `body_ori_b` + `body_lin_vel_b` + `body_ang_vel_b`**（13 体 ×(3+6+3+3) = **195/帧**）| ✅ **今天仍在用**：V33 直接继承本层观测组 + 追加 `anchor_gravity_b` ⇒ 今天 = **396 维** |
| ② | 帧数 **10 → 2**；判别器输入 670 → **390** | **已被覆盖** → **8 帧**（V21~V28，`amp13body8frame`）→ **V29 又回 2 帧**（`AMP_V29_STATE_HISTORY=2`，今天仍是 2 帧，但属 V29 的独立设定、**非本层直接继承**）|
| ③ | 起点 **从零**（`ckpt_manipulator = None`）| 该臂设定；今天不适用（T1j 热启动自 T1a2 的 `model_7000.pt`）|
| ④ | 奖励 21 项 + 站立档负罚 / 指令四档 / 5 clip / 高程图 / actor 626 维 —— **逐字沿用 V19d** | **已被后层整体覆盖**：今天（T1k/T1l）非零 17 项 → 见 `Instinct-Amp-Walk-T1l-G1-v0` |
| ⑤ | AMP 奖励 / 判别器口径（quad k、GP、weight_decay、logit_wd、eps、λ、c）**逐字沿用 V19d**〔算法侧〕| 见 `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V20-G1-v0` 的**「算法侧」节** |

## 推理与判读（已归档，勿在本文件重复）

* 「V20 的失败是**表征错**（body vs joint）」—— ❌ **不是**，是 **2 帧错**（判别器 2000 轮就饱和
  `loss 0.0030` / `d = −0.996`，`style_std` 只有 V19b 的 0.59×，步态退回"高频小跳/原地踏步"，MuJoCo 里不走）→ §A5
* 换 body 表征的**动机**：`body_pos_b` / `body_ori_b` 是关节角的**非线性函数**（前向运动学），
  `body_lin_vel_b` / `body_ang_vel_b` 是 `J(q)·q̇` 的**线性函数**（雅可比）；喂关节角/关节速度等于逼判别器
  **自学 FK 与雅可比**。离线实测：67 维状态在 `h=1~2` 对"时间结构"类退化 **AUC≈0.5（完全不可分）**，
  要到 `h=8~15` 才有效 ⇒ mjlab 侧 2 帧就够，正因为它把这批量**线性可读**地喂了进去
  → `RESULTS_amp_frames_offline_study.md`
* 「全新消融维度」（全谱系此前**从未**用过刚体位姿输入）；且静态族上 body195 反而**更低**（0.75 vs 0.83）
  ⇒ 其判别力来自**时间结构**而非静态幅度 → `RESULTS_amp_frames_body_vs_joint.md`
* 监测清单（启动横幅 `Linear(in_features=390)` / `Loss/discriminator_loss` / `Step/amp_d` / `amp_d_ref`
  应尽快张到 ±1 / `Step/amp_style_*` + `Episode_Reward/feet_air_time` + `Step/air_time_{single_stance,in_band_frac,hop_frac}`
  是步态质量主指标，基线 = V19d）→ 该版 run 日志

⚠️ 其余逐字不动：见上表 ④⑤。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V20-G1-v0`
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import instinctlab.envs.mdp as instinct_mdp

from .amp_walk_env_cfg import G1AmpWalkObservationsCfg
from .amp_walk_v19d_env_cfg import (
    G1AmpWalkV19DEnvCfg,
    G1AmpWalkV19DEnvCfg_PLAY,
    G1AmpWalkV19DTeacherEnv,
)

#: 判别器输入帧数：论文式 `(s_t, s_{t+1})`
AMP_V20_STATE_HISTORY = 2

#: anchor（相对位姿/姿态的参考刚体）
AMP_V20_ANCHOR_NAME = "torso_link"

#: ⭐ 13 个被观测刚体，**顺序必须与参考侧完全一致**（`preserve_order=True`）。
#: 顺序照抄 `AMP_mjlab/src/tasks/amp_loco/config/g1/env_cfgs.py:52-64`，
#: 仅把不可得的 `*_wrist_yaw_link` 换成 `*_wrist_roll_link`（见模块 docstring 第 2 点）。
AMP_V20_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_link",
)

#: 每项的单帧维度（不含帧数）
AMP_V20_TERM_DIMS = {
    "body_pos_b": 3,
    "body_ori_b": 6,
    "body_lin_vel_b": 3,
    "body_ang_vel_b": 3,
}
AMP_V20_STATE_TERMS = tuple(AMP_V20_TERM_DIMS)
#: 单帧总维度：13 × 15 = **195**
AMP_V20_FRAME_SIZE = sum(AMP_V20_TERM_DIMS.values()) * len(AMP_V20_BODY_NAMES)
#: 判别器输入维度：2 × 195 = **390**
AMP_V20_STATE_SIZE = AMP_V20_STATE_HISTORY * AMP_V20_FRAME_SIZE

AMP_WALK_V20_RUN_NAME = (
    "G1AmpWalkV20MVPPlane_obs378_rew21_amp13body2frame_flat2.0_airband_exptau0.08"
    "_motiongate0.15_standpen0.5_ar0.06_standpose0.5_5clip_omnidir"
)


def _robot_body_cfg() -> SceneEntityCfg:
    """策略侧被观测刚体（顺序 = `AMP_V20_BODY_NAMES`）。"""
    return SceneEntityCfg("robot", body_names=AMP_V20_BODY_NAMES, preserve_order=True)


def _robot_anchor_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=(AMP_V20_ANCHOR_NAME,))


def _reference_body_cfg() -> SceneEntityCfg:
    """参考侧被观测刚体（`body_ids` 索引空间 = `link_of_interests`）。"""
    return SceneEntityCfg("motion_reference", body_names=AMP_V20_BODY_NAMES, preserve_order=True)


def _reference_anchor_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("motion_reference", body_names=(AMP_V20_ANCHOR_NAME,))


@configclass
class AmpV20PolicyStateObsCfg(ObsGroup):
    """判别器输入（**策略侧**）：真实机器人的**刚体位姿 + 刚体速度** = 195 维/帧。

    与 `AmpV20ReferenceStateObsCfg` **逐项同构**（项名 / 形状 / 顺序都一致），
    否则 WASABI 在 `init_storage` 会因两侧 `get_subobs_size` 不等而断言失败。

    ⚠️ 数据源是 `instinctlab.envs.mdp`（本仓库自研，见
    `envs/mdp/observations/amp_body_state.py`），**不在** `isaaclab.envs.mdp`。
    ⚠️ **保持默认 `enable_corruption=False`** —— 判别器输入不应加噪声。
    ⚠️ 不加 `scale`：AMP_mjlab 对这 4 项**不做缩放**（与既有的 `joint_vel_rel` 0.05
       不同），此处必须"逐字照抄"，否则 195 维的含义就变了。
    """

    concatenate_terms = False

    body_pos_b = ObsTerm(
        func=instinct_mdp.body_pos_b,
        history_length=AMP_V20_STATE_HISTORY,
        flatten_history_dim=True,
        params={"asset_cfg": _robot_body_cfg(), "anchor_cfg": _robot_anchor_cfg()},
    )
    body_ori_b = ObsTerm(
        func=instinct_mdp.body_ori_b,
        history_length=AMP_V20_STATE_HISTORY,
        flatten_history_dim=True,
        params={"asset_cfg": _robot_body_cfg(), "anchor_cfg": _robot_anchor_cfg()},
    )
    body_lin_vel_b = ObsTerm(
        func=instinct_mdp.body_lin_vel_b,
        history_length=AMP_V20_STATE_HISTORY,
        flatten_history_dim=True,
        params={"asset_cfg": _robot_body_cfg()},
    )
    body_ang_vel_b = ObsTerm(
        func=instinct_mdp.body_ang_vel_b,
        history_length=AMP_V20_STATE_HISTORY,
        flatten_history_dim=True,
        params={"asset_cfg": _robot_body_cfg()},
    )


@configclass
class AmpV20ReferenceStateObsCfg(ObsGroup):
    """判别器输入（**参考侧**）：来自 `motion_reference` 的参考运动刚体状态。

    与 `AmpV20PolicyStateObsCfg` 项名、形状、顺序逐项一致，只把数据源从 `robot`
    换成 `motion_reference`（`MotionReferenceData` 里本来就有
    `link_pos_w/link_quat_w/link_lin_vel_w/link_ang_vel_w` ⇒ **无需离线 FK**）。
    """

    concatenate_terms = False

    body_pos_b = ObsTerm(
        func=instinct_mdp.body_pos_b_reference_as_state,
        history_length=AMP_V20_STATE_HISTORY,
        flatten_history_dim=True,
        params={"asset_cfg": _reference_body_cfg(), "anchor_cfg": _reference_anchor_cfg()},
    )
    body_ori_b = ObsTerm(
        func=instinct_mdp.body_ori_b_reference_as_state,
        history_length=AMP_V20_STATE_HISTORY,
        flatten_history_dim=True,
        params={"asset_cfg": _reference_body_cfg(), "anchor_cfg": _reference_anchor_cfg()},
    )
    body_lin_vel_b = ObsTerm(
        func=instinct_mdp.body_lin_vel_b_reference_as_state,
        history_length=AMP_V20_STATE_HISTORY,
        flatten_history_dim=True,
        params={"asset_cfg": _reference_body_cfg()},
    )
    body_ang_vel_b = ObsTerm(
        func=instinct_mdp.body_ang_vel_b_reference_as_state,
        history_length=AMP_V20_STATE_HISTORY,
        flatten_history_dim=True,
        params={"asset_cfg": _reference_body_cfg()},
    )


@configclass
class G1AmpWalkV20ObservationsCfg(G1AmpWalkObservationsCfg):
    """actor/critic **不动**（626 / flat critic）；只换判别器那两组。

    ⇒ 换表征**只影响判别器**，起点从零即可，actor/critic 的观测契约不受影响。
    """

    amp_policy: AmpV20PolicyStateObsCfg = AmpV20PolicyStateObsCfg()
    amp_reference: AmpV20ReferenceStateObsCfg = AmpV20ReferenceStateObsCfg()


@configclass
class G1AmpWalkV20EnvCfg(G1AmpWalkV19DEnvCfg):
    """训练用：V19d 的奖励/指令/数据**逐字不动**，只换判别器输入。"""

    observations: G1AmpWalkV20ObservationsCfg = G1AmpWalkV20ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # ⚠️⚠️ **必须在 `super()` 之后再赋一次**：祖先链里 V16/V15 的 `__post_init__`
        #      会调 `amp_walk_v12_env_cfg._patch_amp_history()`，那函数用**基类的 5 项模板**
        #      重建 `observations.amp_policy/amp_reference`（还会把 history 拉回 10）
        #      ⇒ 会把上面声明的 V20 组**整个冲掉**，判别器输入悄悄退化成旧的 **670 维**。
        #      2026-10-02 实测（`cloud_probe_v20_cfg.py`）：不加这两行时
        #      `env.observations.amp_policy` 拿到的是 `AmpPolicyStateObsCfg`（V1 版），
        #      连 `body_pos_b` 属性都没有。
        self.observations.amp_policy = AmpV20PolicyStateObsCfg()
        self.observations.amp_reference = AmpV20ReferenceStateObsCfg()
        # 自检：万一又被覆盖，在**建 cfg 时**（秒级）就炸，而不是等训练 2 分钟才出错
        assert isinstance(self.observations.amp_policy, AmpV20PolicyStateObsCfg), (
            "V20 的 amp_policy 被父类的 _patch_amp_history 覆盖了！"
            f" 实际类型 = {type(self.observations.amp_policy).__name__}"
        )
        assert isinstance(self.observations.amp_reference, AmpV20ReferenceStateObsCfg), (
            "V20 的 amp_reference 被父类的 _patch_amp_history 覆盖了！"
            f" 实际类型 = {type(self.observations.amp_reference).__name__}"
        )
        self.run_name = AMP_WALK_V20_RUN_NAME


@configclass
class G1AmpWalkV20EnvCfg_PLAY(G1AmpWalkV19DEnvCfg_PLAY):
    """回放用：与训练同奖励/同数据集/同指令分布/同增广。"""

    observations: G1AmpWalkV20ObservationsCfg = G1AmpWalkV20ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 同训练版：父链的 `_patch_amp_history` 会用基类模板重建这两组 ⇒ 必须再赋一次
        self.observations.amp_policy = AmpV20PolicyStateObsCfg()
        self.observations.amp_reference = AmpV20ReferenceStateObsCfg()
        self.run_name = AMP_WALK_V20_RUN_NAME


class G1AmpWalkV20TeacherEnv(G1AmpWalkV19DTeacherEnv):
    """指令范围与 V19d 相同（全向四档）⇒ 期望值直接继承。"""


__all__ = [
    "AMP_V20_ANCHOR_NAME",
    "AMP_V20_BODY_NAMES",
    "AMP_V20_FRAME_SIZE",
    "AMP_V20_STATE_HISTORY",
    "AMP_V20_STATE_SIZE",
    "AMP_V20_STATE_TERMS",
    "AMP_V20_TERM_DIMS",
    "AMP_WALK_V20_RUN_NAME",
    "AmpV20PolicyStateObsCfg",
    "AmpV20ReferenceStateObsCfg",
    "G1AmpWalkV20EnvCfg",
    "G1AmpWalkV20EnvCfg_PLAY",
    "G1AmpWalkV20ObservationsCfg",
    "G1AmpWalkV20TeacherEnv",
]
