"""**P1 契约 + 运行时校验器**（手写；与自动生成的 `amp_walk_p1_env_cfg.py` 分开）。

为什么单独一个文件
------------------
`amp_walk_p1_env_cfg.py` 是 **`instinct/tools/flatten_cfg.py` 自动生成**的
（"只允许加注释，不要手改结构"，否则下次重生成会被覆盖）。而契约校验器是要**长期维护**
的断言代码 ⇒ 放在这里。

P1 是什么
---------
P1 = "官方 Unitree G1 **rev1.0（pelvis 为根）** 的 AMP 行走策略"，是 2026-10-13 换根之后的
唯一在训臂。它原先的配置继承链有 **46 层 / 43 个文件 / 7852 行**：

    amp_walk_pelvis → t2 → t1l → t1k → t1a2 → v33s → v33 → v32b → v31 → v30 → v29
    → v27 → v26 → v24 → v23 → v21 → v19d → … → v2 → amp_walk → omni_loco → flat
    → ManagerBasedRLEnvCfg

现已**展平成一份自包含配置**（不再 import 任何 `config/g1/*` 旧版本），
等价性由"解析结果快照 diff"机检证明（见下方命令），而不是靠人读代码。

生成 / 校验（**在有 IsaacLab 的机器上跑**）
-----------------------------------------
    # 生成（env + PLAY 同文件；算法侧另一个文件）
    $PY instinct/tools/flatten_cfg.py generate \\
        --cls instinctlab.tasks.locomotion.config.g1:G1AmpWalkPelvisEnvCfg,\\
instinctlab.tasks.locomotion.config.g1:G1AmpWalkPelvisEnvCfg_PLAY \\
        --root-name G1AmpWalkP1EnvCfg,G1AmpWalkP1EnvCfg_PLAY \\
        --out <本目录>/amp_walk_p1_env_cfg.py

    # 校验（必须 diff 为空；两个类各验一次）
    $PY instinct/tools/flatten_cfg.py verify \\
        --chain instinctlab.tasks.locomotion.config.g1:G1AmpWalkPelvisEnvCfg \\
        --flat-file <本目录>/amp_walk_p1_env_cfg.py --flat-attr G1AmpWalkP1EnvCfg
    $PY instinct/tools/flatten_cfg.py verify \\
        --chain instinctlab.tasks.locomotion.config.g1:G1AmpWalkPelvisEnvCfg_PLAY \\
        --flat-file <本目录>/amp_walk_p1_env_cfg.py --flat-attr G1AmpWalkP1EnvCfg_PLAY

无 IsaacLab 时也可在本地跑（用宽容 stub，`--no-app`）：
    PYTHONPATH=<repo>/instinct_xu/source/instinctlab:<repo>/instinct_xu/source/instinct_rl \\
    python3 instinct/tools/offline_stub_isaaclab.py instinct/tools/flatten_cfg.py verify ...

⚠️ 本地 stub 的局限
------------------
本地那套只保证"**展平逻辑**与链式配置等价"（两边都在 stub 下，字段集合对称）；
上游类的**字段形状**以真 IsaacLab 为准 ⇒ 结论必须由云端那一次 `verify` 背书。
（两边都已跑过：本地 ✅ / 云端 ✅。）

本文件里的常量从哪来
--------------------
逐项取自链式配置的**解析结果**（不是从某一份旧文件抄的），并由下面的校验器在
**每个 env 实例启动时** fail-fast 核对 ⇒ 一旦生成物与契约不一致，训练**起不来**，
不会静默跑偏。
"""

from __future__ import annotations

import math

import torch

from instinctlab.envs import InstinctRlEnv

# ============================================================================
# ① 关节序（换根的**唯一**真相源）
# ============================================================================

#: 官方 rev1.0 URDF 的**实测** articulation 关节序（2026-10-13 云端 smoke 读数）。
#:
#: ⚠️ IsaacLab 把 URDF 转 USD 之后的 dof 顺序，**既不等于** `<joint>` 声明序、
#: 也不等于 link 序/DFS 序，**只能实测**。规律：按树的深度分层，同层内 L/R 成对，
#: 层与层之间"父刚体先被访问"者在前 ⇒ **换根会改变顺序**。
#: 本常量同时驱动 ① 契约断言 ② 镜像损失 `mirror_joint_names`（见 ppo 配置）。
P1_JOINT_ORDER = (
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

# ============================================================================
# ② actor 观测 = 1500（4 帧 × 375）
# ============================================================================

P1_HISTORY = 4
P1_OBSERVATION_TERMS = (
    "height_scan",
    "velocity_command",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos_rel",
    "joint_vel",
    "last_action",
)
P1_SCAN_RAYS = 279
#: ⚠️ 校验器比较 `math.prod(dim)` ⇒ 必须是**标量**
P1_OBSERVATION_DIMS = (
    P1_SCAN_RAYS * P1_HISTORY,   # height_scan（mjlab 无此项，我们保留并堆 4 帧）
    3 * P1_HISTORY,              # velocity_command
    3 * P1_HISTORY,              # projected_gravity
    3 * P1_HISTORY,              # base_ang_vel
    29 * P1_HISTORY,             # joint_pos_rel
    29 * P1_HISTORY,             # joint_vel
    29 * P1_HISTORY,             # last_action
)
P1_OBSERVATION_SIZE = sum(P1_OBSERVATION_DIMS)      # == 1500

# ============================================================================
# ③ 命令 / 站立比 / 时序
# ============================================================================

#: (lin_vel_x, lin_vel_y, ang_vel_z)。x 上限 1.0 是**刻意收窄**：参考集里没有慢跑 clip。
P1_COMMAND_RANGES = ((-0.5, 1.0), (-1.0, 1.0), (-1.57, 1.57))
#: 指令保持 6~15 s（mjlab walk 的值；"匀速直线"要被持续采样）
P1_COMMAND_HOLD = (6.0, 15.0)
#: `DirectionalVelocityCommandCfg` 的 fail-fast 要求：这两项必须为 0 / False
P1_STANDING_RATIO = 0.0

# ============================================================================
# ④ 换根相关的关键数
# ============================================================================

#: FK 实测：default 站姿 + 脚掌着地时的**骨盆**离地高度（= 新资产的 `init_state.pos.z`）
P1_INIT_Z = 0.7585
#: 高度项 `track_base_height` 的目标（= `init_state.pos.z`；量的是骨盆）
P1_HEIGHT_TARGET = P1_INIT_Z
#: 上线前必须重导的离线专家集（判别器 anchor 已从 torso 改成 pelvis）
P1_EXPERT_BUFFER = "/root/amp_expert_states_396_pelvis.pt"


# ============================================================================
# ⑤ 运行时 fail-fast 校验器
# ============================================================================


class G1AmpWalkP1TeacherEnv(InstinctRlEnv):
    """P1 的**契约校验器**：env 每次 `load_managers()` 时核对整套接口。

    设计原则：**不继承**任何旧 AMP 版本的校验器（那些链本身很脆），而是把 P1 的期望
    值写在这里、直接从 `InstinctRlEnv` 起。校验失败 ⇒ 抛错 ⇒ 训练/回放起不来
    （fail-fast 优于"静默跑偏"）。
    """

    expected_standing_ratio = P1_STANDING_RATIO
    expected_command_ranges = P1_COMMAND_RANGES
    expected_command_hold = P1_COMMAND_HOLD
    expected_policy_terms = P1_OBSERVATION_TERMS
    expected_policy_dims = P1_OBSERVATION_DIMS
    expected_scan_rays = P1_SCAN_RAYS
    expected_joint_order = P1_JOINT_ORDER
    allow_curriculum = False

    def load_managers(self):
        super().load_managers()
        self._validate_p1_contract()

    def _validate_p1_contract(self):  # noqa: C901 - 逐项断言，长一点更清楚
        errors: list[str] = []

        # ---- 观测 ----------------------------------------------------------
        policy_terms = tuple(self.observation_manager.active_terms["policy"])
        policy_dims = tuple(math.prod(dim) for dim in self.observation_manager.group_obs_term_dim["policy"])
        if policy_terms != self.expected_policy_terms:
            errors.append(f"policy terms {policy_terms} != {self.expected_policy_terms}")
        if policy_dims != self.expected_policy_dims or sum(policy_dims) != P1_OBSERVATION_SIZE:
            errors.append(
                f"policy dims {policy_dims} (sum={sum(policy_dims)}) != {self.expected_policy_dims}"
                f" (sum={P1_OBSERVATION_SIZE})"
            )

        # ---- 高程图 --------------------------------------------------------
        scanner = self.scene.sensors["height_scanner"]
        ray_count = int(scanner.data.ray_hits_w.shape[1])
        if ray_count != P1_SCAN_RAYS:
            errors.append(f"height scan has {ray_count} rays, expected {P1_SCAN_RAYS}")

        # ---- 动作：29 维 / scale 1.0 / 默认偏移 / 不裁剪 ---------------------
        action_term = self.action_manager.get_term("joint_pos")
        if self.action_manager.total_action_dim != 29:
            errors.append(f"action dimension is {self.action_manager.total_action_dim}, expected 29")
        if tuple(action_term._joint_names) != tuple(P1_JOINT_ORDER):
            errors.append(
                "resolved joint order differs:\n"
                f"      实际 = {tuple(action_term._joint_names)}\n"
                f"      期望 = {tuple(P1_JOINT_ORDER)}"
            )
        if not action_term.cfg.use_default_offset:
            errors.append("JointPositionAction use_default_offset is false")
        if action_term.cfg.clip is not None:
            errors.append(f"JointPositionAction clip differs from the unclipped contract: {action_term.cfg.clip}")
        scale = action_term._scale
        scale_ok = (
            bool(torch.allclose(scale, torch.ones_like(scale)))
            if isinstance(scale, torch.Tensor)
            else float(scale) == 1.0
        )
        if not scale_ok:
            errors.append(f"action scale is {scale}, expected 1.0")

        # ---- 时序（50 Hz）--------------------------------------------------
        if self.cfg.sim.dt != 0.005 or self.cfg.decimation != 4 or not math.isclose(self.step_dt, 0.02):
            errors.append(
                f"timing is dt={self.cfg.sim.dt}, decimation={self.cfg.decimation}, step_dt={self.step_dt};"
                " expected 0.005, 4, 0.02"
            )

        # ---- 命令 ----------------------------------------------------------
        command_cfg = self.cfg.commands.base_velocity
        resolved_ranges = (
            tuple(command_cfg.ranges.lin_vel_x),
            tuple(command_cfg.ranges.lin_vel_y),
            tuple(command_cfg.ranges.ang_vel_z),
        )
        if resolved_ranges != tuple(self.expected_command_ranges):
            errors.append(f"command ranges {resolved_ranges} != {self.expected_command_ranges}")
        if command_cfg.heading_command or command_cfg.rel_heading_envs != 0.0:
            errors.append("heading command is enabled")
        if not math.isclose(command_cfg.rel_standing_envs, self.expected_standing_ratio):
            errors.append(
                f"standing-command ratio is {command_cfg.rel_standing_envs},"
                f" expected {self.expected_standing_ratio}"
            )
        if tuple(command_cfg.resampling_time_range) != tuple(self.expected_command_hold):
            errors.append(
                f"command hold is {command_cfg.resampling_time_range},"
                f" expected {self.expected_command_hold}"
            )

        # ---- 课程必须关掉（P1 不用课程）-------------------------------------
        if self.curriculum_manager.active_terms and not self.allow_curriculum:
            errors.append(f"curriculum terms are active: {self.curriculum_manager.active_terms}")

        # ---- 动作语义探针：0 → default q，±0.1 → default q ± 0.1 ------------
        default_joint_pos = self.scene["robot"].data.default_joint_pos[:, action_term._joint_ids]
        expected_offset = default_joint_pos
        if isinstance(action_term._offset, torch.Tensor):
            if not torch.allclose(action_term._offset, expected_offset):
                errors.append("action offset differs from the default pose")
        else:
            errors.append("action offset is not the per-joint default pose")

        zero = torch.zeros_like(action_term.raw_actions)
        action_term.process_actions(zero)
        if not torch.allclose(action_term.processed_actions, expected_offset, atol=1.0e-6, rtol=0.0):
            errors.append("zero action does not map to the default pose")
        if self.num_envs >= 29:
            probe = torch.zeros_like(zero)
            probe[:29, :] = 0.1 * torch.eye(29, device=self.device)
            action_term.process_actions(probe)
            if not torch.allclose(
                action_term.processed_actions[:29], expected_offset[:29] + probe[:29], atol=1.0e-6, rtol=0.0
            ):
                errors.append("+0.1 action probe does not map to default pose +0.1")
            action_term.process_actions(-probe)
            if not torch.allclose(
                action_term.processed_actions[:29], expected_offset[:29] - probe[:29], atol=1.0e-6, rtol=0.0
            ):
                errors.append("-0.1 action probe does not map to default pose -0.1")
        action_term.process_actions(zero)

        if errors:
            raise RuntimeError("P1 (pelvis-rooted official G1 rev1.0) teacher contract failed:\n  - " + "\n  - ".join(errors))
        print(
            f"[P1 PASS] 契约核对通过：policy={sum(P1_OBSERVATION_DIMS)}（height_scan={P1_SCAN_RAYS}×4）、"
            f"action=29、scale=1.0、dt=0.005/decimation=4（50 Hz）、"
            f"command_hold={P1_COMMAND_HOLD}、stand={P1_STANDING_RATIO:.2f}、关节序=P1_JOINT_ORDER"
        )


__all__ = [
    "P1_COMMAND_HOLD",
    "P1_COMMAND_RANGES",
    "P1_EXPERT_BUFFER",
    "P1_HEIGHT_TARGET",
    "P1_HISTORY",
    "P1_INIT_Z",
    "P1_JOINT_ORDER",
    "P1_OBSERVATION_DIMS",
    "P1_OBSERVATION_SIZE",
    "P1_OBSERVATION_TERMS",
    "P1_SCAN_RAYS",
    "P1_STANDING_RATIO",
    "G1AmpWalkP1TeacherEnv",
]
