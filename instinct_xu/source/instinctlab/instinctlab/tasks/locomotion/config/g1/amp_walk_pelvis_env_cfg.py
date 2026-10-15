"""**P1（2026-10-13）**：换根 —— 用**官方 Unitree G1 rev1.0（pelvis 为根）**重做 AMP 行走。

## 为什么换根

我们一直用的 `G1_29DOF_TORSOBASE_POPSICLE_CFG` 把腰链父子顺序整体反转（torso→…→pelvis）
但**没有取反腰关节轴** ⇒ `θ_ours = −θ_std` ⇒ **所有参考动作被镜像解读**。判据（2026-10-10）：

| 判据 | 原样解读 | 取反后 |
|---|---|---|
| 骨盆世界系倾角 | **+4…+30°（强前倾）** | −11…+5°（自然） |
| 躯干相对骨盆 | **−14.7°（后伸）** | +14.6°（前屈，自然） |
| 支撑脚离地间隙 | 散布 −4.6…+12 cm | 紧致 ≈−0.2 cm |

⇒ 从 2026-10-13 起**改用官方 rev1.0 模型**（pelvis 为根、腰链顺序与厂商一致）
⇒ 参考数据**不需要任何取反**，直接自洽。

## 本臂改了什么（只动"与根/资产有关"的部分）

| # | 项 | 旧（torso 根） | **本臂（官方 pelvis 根）** |
|---|---|---|---|
| 1 | 机器人资产 | `G1_29DOF_TORSOBASE_POPSICLE_CFG`（URDF：torsobase_popsicle） | **`G1_29DOF_REV10_PELVIS_CFG`**（`official_rev10/g1_29dof_rev_1_0.urdf`） |
| 2 | 浮动基/init 高度 | root = `torso_link`，`init_state.pos.z = 0.82` | root = **`pelvis`**，`init_state.pos.z = **0.7585**`（FK：default 站姿 + 脚掌着地） |
| 3 | 高程图挂点 | `{ENV_REGEX_NS}/Robot/torso_link`（279 射线，前移 1.5 m） | **`{ENV_REGEX_NS}/Robot/pelvis`**（几何逐字不变） |
| 4 | 运动参考 | `prim_path`/`reference_prim_path` = `…/torso_link`；数据 = `motions/amp_walk`（base = **躯干**位姿） | 两者都指向 `…/pelvis`；数据 = **`motions/amp_walk_pelvis`**（base = **骨盆**位姿，由 `tools/reroot_motion_to_pelvis.py` 用官方 MJCF 换根生成） |
| 5 | 判别器 anchor | `AMP_V20_ANCHOR_NAME = "torso_link"`（两侧） | **`pelvis`**（两侧；含 V33 的 `anchor_gravity_b`） |
| 6 | 高度奖励 | `track_base_height` 量 **torso**，靶 0.82 | **同名同函数**，但 root = pelvis ⇒ 量**骨盆**、靶 **0.7585** ⇒ **无需改奖励**（T5 那套"换体"在本臂自然成立） |
| 7 | 关节序 / 镜像损失 | `OMNI_LOCO_JOINT_ORDER`（肩优先/左右交错） | **`PELVIS_JOINT_ORDER`**（官方序：腿→腰→臂）+ fail-fast 断言 |

## ⚠️ 上线前必须知道的三件事

1. **新资产的 articulation 关节序只能实测**：IsaacLab 把 URDF 转 USD 之后的 dof 顺序
   **既不等于** `<joint>` 声明序、**也不等于** link 序/DFS 序（这是旧资产给我们的教训）。
   ⇒ 本文件用 `PELVIS_JOINT_ORDER` 做**fail-fast 断言**，失败时错误信息会**打印实际顺序**，
   回填即可（预期 = 官方 URDF 的自然序，即 `instinct/tools/joint_order.py::NPZ_ORDER`）。
2. **离线专家集必须重导**：判别器 anchor 从 torso 变 pelvis ⇒ `amp_expert_states_396.pt`
   的内容变了（维度仍 396）。本臂指向 **`/root/amp_expert_states_396_pelvis.pt`**，
   由 `instinct/tools/cloud_export_amp_expert.py --with_anchor_gravity` 在云端重导。
3. **必须冷启动**：动作/观测语义随关节序一起变了 ⇒ 旧 checkpoint 一律不可续训。

## 与越障/蒸馏的关系

* 越障侧仍用旧 `G1_29DOF_TORSOBASE_POPSICLE_CFG`（由小组其他成员迁移）⇒ 本文件**不做任何全局改动**，
  只在子类里替换对象，老任务行为逐字不变。
* 蒸馏契约：`action_adapter` 只支持 `identity` ⇒ 两侧必须同资产同序；越障侧迁移完成前，
  本臂**不能**作为蒸馏教师（记录在案）。
"""

from __future__ import annotations

import copy
import dataclasses
import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

from instinctlab.assets.unitree_g1 import G1_29DOF_REV10_PELVIS_CFG

from .amp_walk_env_cfg import MOTION_NAME
from .amp_walk_t2_env_cfg import (
    T2_SCAN_RAYS,
    G1AmpWalkT2EnvCfg,
    G1AmpWalkT2EnvCfg_PLAY,
    G1AmpWalkT2TeacherEnv,
)
from .omni_loco_env_cfg import OMNI_LOCO_JOINT_ORDER

# ============================================================================
# 常量
# ============================================================================

#: 换根后的参考数据目录（base = **骨盆**位姿；由 `tools/reroot_motion_to_pelvis.py` 生成）
PELVIS_MOTION_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "motions", "amp_walk_pelvis"
)

#: 官方 rev1.0 URDF 的**实测** articulation 关节序（2026-10-13 云端 smoke 读数）
#:
#: 规律（两个资产都验证过）：顺序 = **按树的深度分层**，同层内 L/R 成对排列，
#: 层与层之间"父刚体先被访问"者在前 ⇒ **换根会改变顺序**（与关节名字母序无关：
#: 实测与字母序 0/29 位相同）。旧 torso 根资产 = "臂→腰→腿"的层次序，
#: 本 pelvis 根资产 = "腿→腰→臂"的层次序。
#:
#: ⚠️ 这一处常量**同时**驱动 ① 契约断言 ② 镜像损失 `mirror_joint_names`
#:   （见 `agents/amp_walk_pelvis_ppo_cfg.py`）⇒ 只要这里对，镜像映射就自动对。
PELVIS_JOINT_ORDER = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

#: FK 实测：default 站姿 + 脚掌着地时的**骨盆**离地高度（= 新资产的 `init_state.pos.z`）
PELVIS_INIT_Z = 0.7585
#: 高度项的目标（= `init_state.pos.z`，`track_base_height_exp` 的 `target_height=None` 就是这么取的）
PELVIS_HEIGHT_TARGET = PELVIS_INIT_Z

#: 上线前必须重导的离线专家集（anchor/pelvis 口径）
PELVIS_EXPERT_BUFFER = "/root/amp_expert_states_396_pelvis.pt"

# 自检：两份常量必须是同一批 29 个关节名（只允许顺序不同）
assert set(PELVIS_JOINT_ORDER) == set(OMNI_LOCO_JOINT_ORDER), "PELVIS_JOINT_ORDER 与 OMNI_LOCO_JOINT_ORDER 的关节名集合不一致"
assert len(PELVIS_JOINT_ORDER) == 29


def _pelvis_height_scanner_cfg() -> MultiMeshRayCasterCfg:
    """高程图扫描器：**几何与 V9/T2 逐字相同**（279 射线、前移 1.5 m），只把挂点换到 `pelvis`。

    挂点是根 ⇒ 与 mjlab（root = pelvis）一致；且不再随腰关节转动而倾斜。
    """
    from .amp_walk_v9_env_cfg import _v9_height_scanner_cfg

    cfg = _v9_height_scanner_cfg()
    cfg.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
    return cfg


def _apply_pelvis(env_cfg) -> None:
    """把与"根/资产/参考数据/判别器 anchor"有关的内容全部换成 pelvis 口径。

    ⚠️ 必须**在 `super().__post_init__()` 之后**调用：祖先链会写回资产、场景、
       观测组（含 V33 重建 AMP 组）与 run_name。
    """
    # ---- 1. 机器人资产（机器人本体 + 参考可视化机器人）--------------------------
    env_cfg.scene.robot = G1_29DOF_REV10_PELVIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.robot_reference = G1_29DOF_REV10_PELVIS_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotReference"
    )

    # ---- 2. 高程图挂点 → pelvis（几何不变）-------------------------------------
    env_cfg.scene.height_scanner = _pelvis_height_scanner_cfg()

    # ---- 3. 运动参考：挂点 + 资产 + 数据（换根后的 npz）------------------------
    mr = copy.deepcopy(env_cfg.scene.motion_reference)
    bufs = dict(mr.motion_buffers)
    buf = bufs[MOTION_NAME]
    # 片单文件按**原文件名**在新目录里找（新目录由 reroot 脚本连 yaml 一起复制）
    sel_src = str(buf.filtered_motion_selection_filepath)
    sel_dst = os.path.join(PELVIS_MOTION_DIR, os.path.basename(sel_src))
    if not os.path.isfile(sel_dst):
        raise FileNotFoundError(
            f"[P1] 片单文件不存在：{sel_dst}\n"
            f"    （源：{sel_src}；请先跑 tools/reroot_motion_to_pelvis.py 生成新目录）"
        )
    bufs[MOTION_NAME] = dataclasses.replace(
        buf,
        path=PELVIS_MOTION_DIR,
        filtered_motion_selection_filepath=sel_dst,
    )
    env_cfg.scene.motion_reference = dataclasses.replace(
        mr,
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        reference_prim_path="/World/envs/env_.*/RobotReference/pelvis",
        robot_model_path=G1_29DOF_REV10_PELVIS_CFG.spawn.asset_path,
        motion_buffers=bufs,
    )

    # ---- 4. 判别器的 anchor：torso_link → pelvis（策略侧 + 参考侧，含 V33 第 5 项）----
    env_cfg.observations = copy.deepcopy(env_cfg.observations)
    _patch_amp_anchors(env_cfg)


def _patch_amp_anchors(env_cfg) -> None:
    """把 `amp_policy` / `amp_reference` 两组里所有 `anchor_cfg` 换成 `pelvis`。

    做法是**按 dataclass 字段遍历 + `dataclasses.replace`**（而不是原地改 `params`）：
    1. configclass 的默认值在实例间可能共享 ⇒ 原地改会连带影响其它臂；
    2. 遍历字段可以自动覆盖 V20 的 4 项与 V33 追加的 `anchor_gravity_b`（未来加项也一样）。
    """
    n_patch = 0
    for grp_name, asset_name in (("amp_policy", "robot"), ("amp_reference", "motion_reference")):
        grp = getattr(env_cfg.observations, grp_name)
        n_grp = 0
        for f in dataclasses.fields(grp):
            term = getattr(grp, f.name)
            params = getattr(term, "params", None)
            if isinstance(params, dict) and "anchor_cfg" in params:
                setattr(
                    grp,
                    f.name,
                    dataclasses.replace(
                        term,
                        params={**params, "anchor_cfg": SceneEntityCfg(asset_name, body_names=("pelvis",))},
                    ),
                )
                n_grp += 1
        if n_grp == 0:
            raise ValueError(
                f"[P1] {grp_name} 里没有任何 anchor_cfg 可替换 —— 观测组结构变了？"
                "（换根后判别器会静默地继续用 torso 作 anchor）"
            )
        n_patch += n_grp
        print(f"    [pelvis] {grp_name}: {n_grp} 个 anchor_cfg → pelvis")
    print(f"    [pelvis] 共替换 {n_patch} 处 anchor")


def _readback_pelvis(env_cfg) -> None:
    """回读断言 + 打印（fail-fast：任何一项没生效就 raise）。"""
    robot = env_cfg.scene.robot
    if not str(robot.spawn.asset_path).endswith("official_rev10/g1_29dof_rev_1_0.urdf"):
        raise ValueError(f"[P1] 机器人资产未换成官方 rev1.0：{robot.spawn.asset_path}")
    z = float(robot.init_state.pos[2])
    if abs(z - PELVIS_INIT_Z) > 1e-9:
        raise ValueError(f"[P1] init_state.pos.z = {z}（应 {PELVIS_INIT_Z}）")

    sc = env_cfg.scene.height_scanner
    if not str(sc.prim_path).endswith("/pelvis"):
        raise ValueError(f"[P1] 高程图挂点未换成 pelvis：{sc.prim_path}")
    rays = int(sc.pattern_cfg.size[0] / sc.pattern_cfg.resolution + 1) * int(
        sc.pattern_cfg.size[1] / sc.pattern_cfg.resolution + 1
    )
    if rays != T2_SCAN_RAYS:
        raise ValueError(f"[P1] 高程图射线数变了：{rays}（应 {T2_SCAN_RAYS}）")

    mr = env_cfg.scene.motion_reference
    for nm, v in (("prim_path", mr.prim_path), ("reference_prim_path", mr.reference_prim_path)):
        if not str(v).endswith("/pelvis"):
            raise ValueError(f"[P1] motion_reference.{nm} 未换成 pelvis：{v}")
    if not str(mr.robot_model_path).endswith("official_rev10/g1_29dof_rev_1_0.urdf"):
        raise ValueError(f"[P1] motion_reference.robot_model_path 未换：{mr.robot_model_path}")
    buf = mr.motion_buffers[MOTION_NAME]
    if str(buf.path) != PELVIS_MOTION_DIR:
        raise ValueError(f"[P1] 参考数据目录未换：{buf.path}")

    # 高度项：换根后 root 就是骨盆 ⇒ 量 root 即量骨盆，且 target = default_root_state.z = init z
    h = getattr(env_cfg.rewards, "track_base_height", None)
    if h is None or float(getattr(h, "weight", 0.0)) == 0.0:
        raise ValueError("[P1] track_base_height 缺失或权重为 0（本臂靠它约束 pelvis 高度）")
    if getattr(h, "params", {}).get("target_height") is not None:
        raise ValueError(f"[P1] track_base_height 被写死 target={h.params.get('target_height')}（应 None ⇒ 取 init z）")

    print("[P1][换根] 官方 G1 rev1.0（pelvis 为根）")
    print(f"    资产      : {os.path.basename(str(robot.spawn.asset_path))}  init_z=**{z}**")
    print(f"    高程图    : {sc.prim_path}（{rays} 射线，几何与 T2 逐字相同）")
    print(f"    运动参考  : prim={mr.prim_path}")
    print(f"              数据={buf.path}")
    print(f"              片单={os.path.basename(str(buf.filtered_motion_selection_filepath))}")
    print(f"    高度项    : track_base_height weight={float(h.weight)} std={h.params.get('std')}"
          f" target=**{PELVIS_HEIGHT_TARGET}**（= init z，量的是骨盆）")
    print(f"    关节序    : PELVIS_JOINT_ORDER（官方序；契约会 fail-fast 核对实测顺序）")
    print("    ⚠️ 上线前两件事：① 实测关节序若与此不符，契约会打印实际顺序，回填 `PELVIS_JOINT_ORDER` 即可；")
    print(f"                    ② 离线专家集需重导到 {PELVIS_EXPERT_BUFFER}（anchor 已变）")


# ============================================================================
# 环境配置
# ============================================================================


@configclass
class G1AmpWalkPelvisEnvCfg(G1AmpWalkT2EnvCfg):
    """P1：T2 的整套配方（obs1500 / 13 项奖励 / mjlab 命令与终止 / 真 RSI）**逐字不动**，
    只做上面那 7 项"与根/资产有关"的替换。"""

    def __post_init__(self):
        super().__post_init__()          # T2 的整套 + 自检（此刻仍是 torso 口径）
        _apply_pelvis(self)              # 覆盖根相关的一切
        _readback_pelvis(self)
        self.run_name = "p1_pelvis_official_rev10_cold"


@configclass
class G1AmpWalkPelvisEnvCfg_PLAY(G1AmpWalkT2EnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_pelvis(self)
        _readback_pelvis(self)
        self.run_name = "p1_pelvis_official_rev10_cold"


class G1AmpWalkPelvisTeacherEnv(G1AmpWalkT2TeacherEnv):
    """契约校验器：只把**关节顺序**的期望值换成 `PELVIS_JOINT_ORDER`。

    其余（obs 1500 项/维、279 射线、动作 29/scale 1.0/`use_default_offset`/`clip=None`、
    时序、执行器、指令范围、站立比、不许课程）逐字继承 T2。
    """

    expected_joint_order = PELVIS_JOINT_ORDER


__all__ = [
    "PELVIS_EXPERT_BUFFER",
    "PELVIS_HEIGHT_TARGET",
    "PELVIS_INIT_Z",
    "PELVIS_JOINT_ORDER",
    "PELVIS_MOTION_DIR",
    "G1AmpWalkPelvisEnvCfg",
    "G1AmpWalkPelvisEnvCfg_PLAY",
    "G1AmpWalkPelvisTeacherEnv",
]
