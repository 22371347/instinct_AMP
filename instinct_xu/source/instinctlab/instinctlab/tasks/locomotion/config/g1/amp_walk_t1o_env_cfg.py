"""**T1o 环境侧（方案 C）**：actor 观测从 **378（单帧）** 改成 **663（4 帧本体感受 + 单帧 height_scan）**，
并**删掉 `base_lin_vel`** —— 对齐 AMP_mjlab 的 actor 契约。

动机（2026-10-09，逐行核对 mjlab 源码）
--------------------------------------
`AMP_mjlab/src/tasks/amp_loco/amp_env_cfg.py:148-169`::

    "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True,
                                 enable_corruption=True,
                                 history_length=4, history_ordering="time")

⇒ mjlab 的 actor = **96 维 × 4 帧 = 384**：`base_ang_vel 3 + projected_gravity 3 + command 3
   + joint_pos 29 + joint_vel 29 + actions 29`，**无 height_scan、无 base_lin_vel**。
我们（V9 起）的 actor = **378 单帧**，其中 **279（74%）是平地上近乎常数的 height_scan**。

两条推论
--------
1. ⭐ **步态相位只能从历史推断** —— 单帧观测在数学上分不出"左腿摆动/右腿支撑"与
   "右腿摆动/左腿支撑"这两种相位（同一瞬时姿态可以属于两个不同相位），而"交替步"
   正是相位依赖的动作。这可能是"单脚跳 / 不交替 / 不抬脚"的**结构性**原因，
   比"AMP 权重调多大"根本得多。
2. mjlab 的 actor **不需要** `base_lin_vel`，因为 4 帧历史足以估计本体速度；我们给
   特权 `base_lin_vel` 实际上是在**补偿缺失的历史** —— 但补的是速度，**不是相位**。

本臂的 663 契约（= mjlab 的 384 + 你要求保留的 279 `height_scan`）
----------------------------------------------------------------
| 项 | 单帧 | 帧数 | 小计 |
|---|---|---|---|
| `height_scan` | 279 | **1**（保留，与其它专家策略对齐）| 279 |
| `velocity_command` | 3 | **4** | 12 |
| `projected_gravity` | 3 | **4** | 12 |
| `base_ang_vel` | 3 | **4** | 12 |
| `joint_pos_rel` | 29 | **4** | 116 |
| `joint_vel` | 29 | **4** | 116 |
| `last_action` | 29 | **4** | 116 |
| **合计** | | | **663** = 279 + 384 |

❌ 删掉 `base_lin_vel`：对齐 mjlab（它只在 critic），且真机上也没有这个量。

⚠️ 与 T1j~T1n 的可比性 / 纪律
-----------------------------
* obs 维度变了 ⇒ **必须从头训**（不带 `--resume`）。本臂是**新基线族**，不能与
  T1j~T1n（378 热启动）逐项比数值，只能与"378 冷启动"比（`Instinct-Amp-Walk-V33ACold-G1-v0`）。
* 环境侧**只有 obs 变了**：奖励集 / 命令采样 / 终止 / 场景全部继承 T1l
  （`track_base_height` std=0.3 weight=1.5、`track_lin_vel_xy` 1.5、非零奖励项 17）。
* 算法侧沿用 T1n（判别器白化 + GP 在归一化空间）。
* ⭐ **镜像损失**：`instinctlab/utils/mirror.py` 本次已改成"帧感知"（原实现遇到 4 帧
  堆叠会直接抛形状错误）。回归测试：`python3 instinct/tools/test_mirror_frames.py`
  （已覆盖 663 布局 + 逐帧等价 + 两个 fail-fast）。

遗留（不影响训练）
------------------
MuJoCo 回放 / ONNX 导出链路按 ONNX 输入维选观测布局（`g1_mujoco_play.py --obs-mode auto`），
需要为 663 增加一套布局才能回放 —— **训练不受影响**，但本臂出 ckpt 前要补。
"""

from __future__ import annotations

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import instinctlab.envs.mdp as instinct_mdp

from .amp_walk_t1l_env_cfg import G1AmpWalkT1lEnvCfg, G1AmpWalkT1lEnvCfg_PLAY
from .amp_walk_v31_env_cfg import G1AmpWalkV31TeacherEnv

#: 帧数 = mjlab 的 `history_length=4`（`history_ordering="time"`）
T1O_HISTORY = 4

#: 契约（顺序 = 下面的**声明顺序**，`_validate_teacher_contract` 会逐项比对）。
T1O_OBSERVATION_TERMS = (
    "height_scan",
    "velocity_command",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos_rel",
    "joint_vel",
    "last_action",
)
T1O_SCAN_RAYS = 279
#: ⚠️ 校验器比较的是 `math.prod(dim)` ⇒ 这里必须是**标量**（不是 `(4, 29)`）。
T1O_OBSERVATION_DIMS = (
    T1O_SCAN_RAYS,
    3 * T1O_HISTORY,    # velocity_command
    3 * T1O_HISTORY,    # projected_gravity
    3 * T1O_HISTORY,    # base_ang_vel
    29 * T1O_HISTORY,   # joint_pos_rel
    29 * T1O_HISTORY,   # joint_vel
    29 * T1O_HISTORY,   # last_action
)
T1O_OBSERVATION_SIZE = sum(T1O_OBSERVATION_DIMS)      # == 663


@configclass
class G1AmpWalkT1oPolicyObsCfg(ObsGroup):
    """T1o 的 actor 契约：**663 = 279（height_scan，单帧）+ 4×96（本体感受，4 帧）**。

    与 `G1AmpWalkPolicyObsCfg`（378）的差异只有两条：
      ① 删掉 `base_lin_vel`（对齐 mjlab：它只在 critic）
      ② 其余 6 项加 `history_length=4` + `flatten_history_dim=True`
    `height_scan` **保持单帧**（保留你的对齐要求）。

    噪声幅度**逐项沿用** 378 的口径，便于归因（只动"有没有历史"这一维）。
    """

    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )
    velocity_command = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
        history_length=T1O_HISTORY,
        flatten_history_dim=True,
    )
    projected_gravity = ObsTerm(
        func=instinct_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
        history_length=T1O_HISTORY,
        flatten_history_dim=True,
    )
    base_ang_vel = ObsTerm(
        func=instinct_mdp.base_ang_vel,
        noise=Unoise(n_min=-0.2, n_max=0.2),
        history_length=T1O_HISTORY,
        flatten_history_dim=True,
    )
    joint_pos_rel = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        history_length=T1O_HISTORY,
        flatten_history_dim=True,
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        history_length=T1O_HISTORY,
        flatten_history_dim=True,
    )
    last_action = ObsTerm(
        func=instinct_mdp.last_action,
        history_length=T1O_HISTORY,
        flatten_history_dim=True,
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


def _apply_t1o(env_cfg) -> None:
    """把 actor 观测换成 663 契约 + 回读断言（fail-fast）。"""
    assert sum(T1O_OBSERVATION_DIMS) == 663, f"T1o 契约自检失败：{T1O_OBSERVATION_DIMS}"
    env_cfg.observations.policy = G1AmpWalkT1oPolicyObsCfg()
    pol = env_cfg.observations.policy
    got = tuple(
        int(getattr(pol, n).history_length or 1)
        for n in T1O_OBSERVATION_TERMS
    )
    if got != (1, T1O_HISTORY, T1O_HISTORY, T1O_HISTORY, T1O_HISTORY, T1O_HISTORY, T1O_HISTORY):
        raise ValueError(f"[T1o] history_length 回读不符：{got}")
    if hasattr(pol, "base_lin_vel"):
        raise ValueError("[T1o] base_lin_vel 仍在 policy 观测里（应已删除）")
    print("[T1o][HISTORY = 4] actor 观测 378 → **663**：")
    print(
        f"    项顺序 = {T1O_OBSERVATION_TERMS}\n"
        f"    逐项维度 = {T1O_OBSERVATION_DIMS}（sum={T1O_OBSERVATION_SIZE}）"
    )
    print(
        "    height_scan 279 保持**单帧**（与其它专家策略对齐）；"
        "本体感受 6 项 ×4 帧 = 384（= mjlab 的 actor 输入）"
    )
    print("    ❌ base_lin_vel 已删除（mjlab 里它只在 critic；真机也没有这个量）")


@configclass
class G1AmpWalkT1oEnvCfg(G1AmpWalkT1lEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_t1o(self)
        self.run_name = "G1AmpWalkT1oMVPPlane_obs663_rew17_h4_cold"


@configclass
class G1AmpWalkT1oEnvCfg_PLAY(G1AmpWalkT1lEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_t1o(self)
        self.run_name = "G1AmpWalkT1oMVPPlane_obs663_rew17_h4_cold_PLAY"


class G1AmpWalkT1oTeacherEnv(G1AmpWalkV31TeacherEnv):
    """契约校验器：只覆盖期望值（663 / 无 `base_lin_vel` / 本体感受 4 帧）。

    其余检查（动作维度 29、joint 顺序、`use_default_offset`、命令范围/保持时长、
    default-offset 语义探针……）全部继承 —— 那些契约本臂**一个字都没改**。
    """

    expected_policy_terms = T1O_OBSERVATION_TERMS
    expected_policy_dims = T1O_OBSERVATION_DIMS
    expected_scan_rays = T1O_SCAN_RAYS
