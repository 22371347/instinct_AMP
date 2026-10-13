"""AMP 行走 **v9（MVP）**：v8 续训发散后把变量收敛到四项的最小实现（2026-09-28）。

## ⚠️ 本层对**今天**仍有影响的部分

* ⭐ **`height_scan` = 3.0×0.8 m（279 射线）、中心前移 1.5 m** 今天仍生效
  ⇒ 策略观测 **626 → 378**（`obs378` 沿用至 T1l；观测项名字/顺序未动）。
* ⭐ **"越障侧与我们是同一个 `G1_29DOF_TORSOBASE_POPSICLE_CFG`（`default_joint_pos` 逐字相同）"
  今天仍有效** ⇒ 动作语义 `q_target = default + 1.0·a`、`clip=None` 两边本来就一致。
* ❌ MVP 指令集（仅前向 0~0.7）已被完全覆盖（V10 "③ 指令扩展" → V18 方案 B → V33 `side_vel_min`）。
* ❌ `noise_std` 缩小（在算法 cfg）已被 T1a2 覆盖（`init_noise_std=1.0`）。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | `height_scan` 3.0×0.8（**279**，`resolution=0.1`）+ **中心前移 1.5 m**（`AMP_WALK_V9_SCAN_FORWARD`，`ray_alignment="yaw"`）⇒ 观测 **626 → 378** = 279+3+3+3+3+29+29+29 | ✅ 仍生效（观测 378）|
| ② | `default_joint_pos` / `offset` 对齐越障 | ⚪ **核查后本来就一致**，未改（见下）|
| ③ | `noise_std` → AMP_mjlab 等效值（+ std 上限）〔算法 cfg〕| ❌ 已被 T1a2 覆盖 |
| ④ | MVP 指令集（原地 + 前向 0~0.7）+ 仅前向数据集 | ❌ 已被 V10 / V18 / V33 覆盖 |

**② 的核查（今天仍适用）**：两边同一个 POPSICLE cfg（`default_joint_pos` 逐字相同）；
`JointPositionActionCfg.use_default_offset` **默认 True**（`actions_cfg.py:52`）⇒ 越障侧没写也等于 True；
蒸馏教师 `perceptive_vault` 的 `scale` 也是 **1.0**（只有 `beyondmimic_vault` 用逐关节 `0.25·e/kp`）。

**④ 当时口径**：低速/原地 **0.30**（幅值 ≤0.1、`wz=0`；其中 60% 给精确 (0,0,0) ⇒ 占全体 18%）；
纯侧向 / 纯前后 / 持续后退 **0 / 0 / 0**；均匀 0.70（`vx∈(0,0.7) / vy=0 / wz=0`，`vy`/`wz` 范围钉成 (0,0) 以免留死配置）。
⚠️ `rel_standing_envs` / `rel_heading_envs` 仍 **0.0**、`heading_command` 仍 False（否则 `DirectionalVelocityCommand` fail-fast）；
`resampling_time_range` 仍 **(6, 15) s**。

## 推理与判读（已归档，勿在本文件重复）

* ⚠️ **`height_scan` 出处对账**：手上越障源码 `perceptive_vault_cfg.py:327-334` 写的是 `pos=(0,0,20)`（**居中**）；
  学长口头确认他的更新版**中心在机器人正前方** ⇒ 本文件按**前移 1.5 m** 实现（用户 2026-09-28 确认）。
  若日后拿到他的 `params/env.yaml`，只需核对/修改 `AMP_WALK_V9_SCAN_FORWARD` 一行 → 见该版 run 日志
* "仅前向数据集 / 指令 ⇒ AMP 压制侧移后退（**改命令分布无法解决**，只能补参考运动数据）"
  → `docs/amp_walk/EXPERIMENTS.md` §D1
* "镜像损失在**只有前向速度**时无法正常生效"（本层 MVP 指令即纯前向，正是 §A1 的前提）→ §A1
* "95% 摔倒"只是断崖快照（v9g 振荡）→ §C1；"单脚侧身跳着走"退化模态 → §C3

⚠️ 其余逐字不动：除 `height_scan` 宽度外，v8 的环境（动作 / 观测项 / 奖励表）逐字沿用。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V9-G1-v0`
"""

from __future__ import annotations

import copy
import os

from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass

from .amp_walk_env_cfg import (
    AMP_WALK_OBSERVATION_TERMS,
    MOTION_DIR,
    MOTION_NAME,
    G1AmpWalkSceneCfg,
)
from .amp_walk_v3_env_cfg import G1AmpWalkV3TeacherEnv
from .amp_walk_v8_env_cfg import G1AmpWalkV8EnvCfg, G1AmpWalkV8EnvCfg_PLAY

# ---------------------------------------------------------------------------
# v9(MVP) 的契约常量 —— **唯一事实来源**，日志/文档/工具都引用这里
# ---------------------------------------------------------------------------
#: 扫描区域尺寸（前后, 左右）m
AMP_WALK_V9_SCAN_SIZE = (3.0, 0.8)
#: 扫描分辨率 m
AMP_WALK_V9_SCAN_RESOLUTION = 0.1
#: ⭐ 扫描中心相对躯干的前向偏移 m（0.0 = 居中，1.5 = 落点 x∈[0,3]）
AMP_WALK_V9_SCAN_FORWARD = 1.5
#: 射线条数 = 31 × 9（由上面两项推导，不是硬写常数）
AMP_WALK_V9_SCAN_RAYS = (int(AMP_WALK_V9_SCAN_SIZE[0] / AMP_WALK_V9_SCAN_RESOLUTION) + 1) * (
    int(AMP_WALK_V9_SCAN_SIZE[1] / AMP_WALK_V9_SCAN_RESOLUTION) + 1
)

#: 观测项名/顺序**与 v1~v8 逐字相同**（只改了 height_scan 的宽度）
AMP_WALK_V9_OBSERVATION_TERMS = AMP_WALK_OBSERVATION_TERMS
#: 各段维数：279 + 3 + 3 + 3 + 3 + 29 + 29 + 29 = **378**
AMP_WALK_V9_OBSERVATION_DIMS = (
    AMP_WALK_V9_SCAN_RAYS,
    3,  # velocity_command
    3,  # base_lin_vel
    3,  # projected_gravity
    3,  # base_ang_vel
    29,  # joint_pos_rel
    29,  # joint_vel
    29,  # last_action
)
AMP_WALK_V9_OBSERVATION_SIZE = sum(AMP_WALK_V9_OBSERVATION_DIMS)

#: MVP 指令范围：**原地 + 前向**（vy / wz 恒为 0）
AMP_WALK_V9_COMMAND_RANGES = ((0.0, 0.7), (0.0, 0.0), (0.0, 0.0))
#: 低速/原地档概率（其余 0.70 为父类均匀分支，范围即上面的 (0,0.7)）
AMP_WALK_V9_PROB_LOW = 0.30
AMP_WALK_V9_LOW_VEL = 0.1
AMP_WALK_V9_ZERO_FRAC_IN_LOW = 0.6  # 低速档里 60% 给精确 (0,0,0)
AMP_WALK_V9_COMMAND_HOLD = (6.0, 15.0)

#: MVP 数据集：**只留 1 个纯前向「走」clip**（见 yaml 里的逐条理由）
#: ⚠️ `jog_forward` 也已删除（用户 2026-09-28）：同为前向但**步态不同**（跑 vs 走），
#:    两种前向形态共存会让风格奖励在「走」与「跑」之间拉扯，污染最小验证。
AMP_WALK_V9_MVP_MOTION_SELECTION_FILE = os.path.join(
    MOTION_DIR, "amp_walk_v9_mvp_motion_selection.yaml"
)

#: run_name（日志目录前缀）。`obs378` / `fwdscan279` / `ns0.25` / `mvpcmd`
AMP_WALK_V9_RUN_NAME = "G1AmpWalkV9MVPPlane_obs378_rew15_ar0.02_ns0.25_ampc0.05_fwdscan279_mvpcmd"


def _v9_height_scanner_cfg() -> MultiMeshRayCasterCfg:
    """v9 的高程图扫描器：与越障同尺寸、**中心前移 1.5 m**。

    ⚠️ 用**新建对象**（而不是原地改 `self.scene.height_scanner` 的字段）：
    `@configclass` 的默认值在实例间可能共享，原地改会连带影响 v1~v8
    —— 与越障那边 `_apply_perceptive_vault_settings` 的做法一致。
    """
    return MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        update_period=0.02,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(AMP_WALK_V9_SCAN_FORWARD, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=AMP_WALK_V9_SCAN_RESOLUTION,
            size=list(AMP_WALK_V9_SCAN_SIZE),
        ),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


def apply_amp_walk_v9_mvp_command_ranges(cmd_cfg) -> None:
    """把指令收敛成「原地 + 前向 (0, 0.7)」。

    做法：范围只留前向，并把侧移/纯前后/后退三个分档概率**清零** ⇒
    剩下的 0.70 走父类均匀分支，而均匀分支的范围已经是 (0,0.7)×0×0。
    """
    cmd_cfg.resampling_time_range = AMP_WALK_V9_COMMAND_HOLD
    cmd_cfg.rel_standing_envs = 0.0  # ⚠️ 不能改（DirectionalVelocityCommand 有 fail-fast）
    cmd_cfg.rel_heading_envs = 0.0  # ⚠️ 同上
    cmd_cfg.heading_command = False  # ⚠️ 开了方向分档会失效
    cmd_cfg.ranges.lin_vel_x = AMP_WALK_V9_COMMAND_RANGES[0]
    cmd_cfg.ranges.lin_vel_y = AMP_WALK_V9_COMMAND_RANGES[1]
    cmd_cfg.ranges.ang_vel_z = AMP_WALK_V9_COMMAND_RANGES[2]
    cmd_cfg.ranges.heading = (-0.0, 0.0)
    # 分档概率：只留低速/原地档，其余清零
    cmd_cfg.prob_low = AMP_WALK_V9_PROB_LOW
    cmd_cfg.low_vel = AMP_WALK_V9_LOW_VEL
    cmd_cfg.zero_frac_in_low = AMP_WALK_V9_ZERO_FRAC_IN_LOW
    cmd_cfg.prob_side = 0.0
    cmd_cfg.prob_lin = 0.0
    cmd_cfg.prob_back = 0.0
    # 与上面三个「关闭的档位」配套的范围也收进 MVP 区间，避免留下冲突的死配置
    cmd_cfg.straight_lin_vel_range = AMP_WALK_V9_COMMAND_RANGES[0]
    cmd_cfg.straight_ang_vel_range = (0.0, 0.0)
    cmd_cfg.back_lin_vel_range = (0.0, 0.0)
    cmd_cfg.back_ang_vel_range = (0.0, 0.0)


def _apply_v9_motion_selection(env_cfg) -> None:
    """把参考动作清单换成 MVP 的 **1 个纯前向「走」clip**。

    ⚠️ **必须 deepcopy**：`motion_buffers` 是 dict，`@configclass` 的默认值可能
    在多实例间共享 ⇒ 直接改会连带改到 v3~v8（与 v3 的 `_apply_v3_motion_selection`
    同样的理由）。
    ⚠️ 调用顺序：必须**在 super().__post_init__() 之后**（v3 会先把清单设成 7 clip）。
    """
    env_cfg.scene.motion_reference = copy.deepcopy(env_cfg.scene.motion_reference)
    env_cfg.scene.motion_reference.motion_buffers[
        MOTION_NAME
    ].filtered_motion_selection_filepath = AMP_WALK_V9_MVP_MOTION_SELECTION_FILE


# ---------------------------------------------------------------------------
# 环境配置
# ---------------------------------------------------------------------------


@configclass
class G1AmpWalkV9EnvCfg(G1AmpWalkV8EnvCfg):
    """MVP 训练用：v8（15 项奖励 + log_quad 风格形式）之上做上面四项改动。"""

    def __post_init__(self):
        super().__post_init__()
        # ⚠️ 顺序：父类链（v1→v8）会写回 run_name / 指令范围 / 7-clip 清单，
        #    所以四项覆盖都必须放在 super() **之后**。
        self.scene.height_scanner = _v9_height_scanner_cfg()  # (1) 279 + 前移 1.5 m
        apply_amp_walk_v9_mvp_command_ranges(self.commands.base_velocity)  # (4) 指令
        _apply_v9_motion_selection(self)  # (4) 数据集
        self.run_name = AMP_WALK_V9_RUN_NAME


@configclass
class G1AmpWalkV9EnvCfg_PLAY(G1AmpWalkV8EnvCfg_PLAY):
    """回放用：同样的四项改动（`motion_reference` 的 `debug_vis` 继承 v1 PLAY）。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner = _v9_height_scanner_cfg()
        apply_amp_walk_v9_mvp_command_ranges(self.commands.base_velocity)
        _apply_v9_motion_selection(self)
        self.run_name = AMP_WALK_V9_RUN_NAME


class G1AmpWalkV9TeacherEnv(G1AmpWalkV3TeacherEnv):
    """v9 的契约校验器：把「观测量/射线条数/指令范围/保持时长」的期望值改成 v9 的。

    其余校验（动作维 29 / joint 顺序 / `scale=1.0` / `use_default_offset` /
    `clip=None` / 时序 / 执行器 / 站立比 0 / 不允许课程）**逐字继承**，即
    仍然按学长那套越障契约卡。
    ⚠️ 校验器**只断言射线条数**（`scanner.data.ray_hits_w.shape[1]`），不查 offset/size
    ⇒ 前移 1.5 m 不会触发校验失败，但**条数必须是 279**。
    """

    expected_policy_terms = AMP_WALK_V9_OBSERVATION_TERMS
    expected_policy_dims = AMP_WALK_V9_OBSERVATION_DIMS
    expected_scan_rays = AMP_WALK_V9_SCAN_RAYS
    expected_command_ranges = AMP_WALK_V9_COMMAND_RANGES
    expected_command_hold = AMP_WALK_V9_COMMAND_HOLD


__all__ = [
    "AMP_WALK_V9_COMMAND_HOLD",
    "AMP_WALK_V9_COMMAND_RANGES",
    "AMP_WALK_V9_MVP_MOTION_SELECTION_FILE",
    "AMP_WALK_V9_OBSERVATION_DIMS",
    "AMP_WALK_V9_OBSERVATION_SIZE",
    "AMP_WALK_V9_OBSERVATION_TERMS",
    "AMP_WALK_V9_PROB_LOW",
    "AMP_WALK_V9_RUN_NAME",
    "AMP_WALK_V9_SCAN_FORWARD",
    "AMP_WALK_V9_SCAN_RAYS",
    "AMP_WALK_V9_SCAN_RESOLUTION",
    "AMP_WALK_V9_SCAN_SIZE",
    "G1AmpWalkV9EnvCfg",
    "G1AmpWalkV9EnvCfg_PLAY",
    "G1AmpWalkV9TeacherEnv",
    "G1AmpWalkSceneCfg",
    "apply_amp_walk_v9_mvp_command_ranges",
]
