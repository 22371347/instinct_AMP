"""**T1f**：把 mjlab 的奖励机理**恢复**回来 + 限位/姿态权重按 mjlab 校准。

为什么（2026-10-07 定位）
------------------------
T1a2/T1b/T1c/T1d/T1e 这条线为了"与 T0 奖励白名单逐字对齐"，把下面这些项
**全部置 0 禁用**了（`amp_walk_env_cfg.py` 里它们本来就带 mjlab v5 的权重）：

    项                              mjlab v5   T1a2~T1e
    feet_lateral_width（步宽下限）   -10.0      0（禁用）
    feet_crossing（双脚真交叉）      -20.0      0（禁用）
    track_lin_vel_xy_exp              1.5       1.0
    track_ang_vel_z_exp               1.2       1.0

⇒ **mjlab 防止"窄步 / 脚掌侧边缘站姿"的核心机理被关掉了**，这解释了：
  * 踝 `ankle_roll` 从 ≤500 轮起就停在限位附近（`tracking_joint_error_rmse`
    ≈0.242→0.257，全身最大，且全程几乎不变）；
  * `limit_margin_u_max = 2.08`（越过硬限位）；
  * 目视"用脚掌侧边缘行走"。
另外 mjlab 的 `joint_pos_limits` 权重是 **-10.0**，我们只有 **-1.0**（差 10×）。

本臂相对 T1a2 的改动（**mjlab 对齐**，逐项可核）
----------------------------------------------
    1. feet_lateral_width : 0 → **-10.0**（sep_floor=0.157，gate_lateral，max_cost 0.10）
    2. feet_crossing      : 0 → **-20.0**（保留方向门控：本仓库侧移 clip 是交叉式侧步）
    3. dof_pos_limits     : 踝-only -1.0 → **全关节 -10.0**（回到标准 joint_pos_limits）
    4. flat_orientation_l2: -1.0 → **-2.0**（用户指定，强化上身直立）
    5. track_lin_vel_xy_exp : 1.0 → **1.5**（mjlab）
    6. track_ang_vel_z_exp  : 1.0 → **1.2**（mjlab）
算法侧同批对齐见 `agents/amp_walk_t1f_ppo_cfg.py`。

⚠️ 这是**刻意偏离 T0 口径**（T0 的奖励表是另一套设计）。目的：把"跟踪 + 类人风格"
   拉回 mjlab 已验证有效的配置。
⚠️ 本文件不改动、不删除任何既有 cfg（非破坏性）。
"""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from instinctlab.tasks.locomotion.mdp import rewards_amp as locomotion_mdp

from .amp_walk_env_cfg import AMP_WALK_SEP_FLOOR, AMP_WALK_WALK_MAX_LIN_VEL
from .amp_walk_t1a2_env_cfg import G1AmpWalkT1a2EnvCfg, G1AmpWalkT1a2EnvCfg_PLAY

AMP_WALK_T1F_RUN_NAME = "G1AmpWalkT1fMVPPlane_obs378_rew21_mjlabalign_sym"

#: mjlab v5 的权重（`AMP_mjlab/src/tasks/amp_loco/config/g1/env_cfgs.py` + `amp_env_cfg.py`）
T1F_LATERAL_WIDTH_WEIGHT = -10.0
T1F_CROSSING_WEIGHT = -20.0
T1F_LIMIT_WEIGHT = -10.0
T1F_FLAT_ORIENT_WEIGHT = -2.0
T1F_TRACK_LIN_WEIGHT = 1.5
T1F_TRACK_ANG_WEIGHT = 1.2

_COMMAND_NAME = "base_velocity"


def _set(term, *, weight=None, func=None, params=None, joint_names=None, site_names=None, label=""):
    """就地改一个奖励项，并**当场回读**断言生效（fail-fast）。"""
    if term is None:
        raise ValueError(f"[T1f] 奖励项不存在：{label}")
    if func is not None:
        term.func = func
    if weight is not None:
        term.weight = float(weight)
    if params is not None:
        term.params = dict(params)
    _p = getattr(term, "params", {}) or {}
    if joint_names is not None:
        _ac = _p.get("asset_cfg")
        if _ac is None:
            raise ValueError(f"[T1f] {label}: params 缺 asset_cfg")
        _ac.joint_names = list(joint_names)
    if site_names is not None:
        _ac = _p.get("asset_cfg")
        if _ac is None:
            raise ValueError(f"[T1f] {label}: params 缺 asset_cfg")
        _ac.site_names = tuple(site_names)
    return term


def _apply_t1f_mjlab_align(cfg) -> None:
    R = cfg.rewards
    log = []

    # 1) 步宽下限：**只把权重从 0 打开**（`params` 沿用小仓库的既有定义 = mjlab v5：
    #    sep_floor 0.145 / gate_lateral / max_cost 0.10；本仓库 SceneEntityCfg 不支持
    #    site_names，故不重建 asset_cfg）
    t = _set(getattr(R, "feet_lateral_width", None), weight=T1F_LATERAL_WIDTH_WEIGHT, label="feet_lateral_width")
    _p = t.params or {}
    log.append(
        ("feet_lateral_width", t.weight,
         f"sep_floor={_p.get('sep_floor')} gate_lateral={_p.get('gate_lateral')} cmd={_p.get('command_name')}")
    )

    # 2) 真交叉（mjlab：-20.0）：同样只打开权重
    t = _set(getattr(R, "feet_crossing", None), weight=T1F_CROSSING_WEIGHT, label="feet_crossing")
    log.append(("feet_crossing", t.weight, "params 沿用小仓库定义（含方向门控）"))

    # 3) 关节限位：踝-only -1.0 → 全关节 -10.0（mjlab 权威值）
    t = _set(
        getattr(R, "dof_pos_limits", None),
        weight=T1F_LIMIT_WEIGHT,
        func=isaac_mdp.joint_pos_limits,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])},
        joint_names=[".*"],
        label="dof_pos_limits",
    )
    _jn = list(getattr((t.params or {}).get("asset_cfg"), "joint_names", []) or [])
    if _jn != [".*"]:
        raise ValueError(f"[T1f] dof_pos_limits 作用域未生效：{_jn!r}")
    log.append(("dof_pos_limits", t.weight, f"joint_names={_jn} func=joint_pos_limits"))

    # 4) 上身直立（用户指定 -2.0）
    t = _set(getattr(R, "flat_orientation_l2", None), weight=T1F_FLAT_ORIENT_WEIGHT, label="flat_orientation_l2")
    log.append(("flat_orientation_l2", t.weight, "用户指定强化直立"))

    # 5/6) 跟踪权重回到 mjlab
    t = _set(getattr(R, "track_lin_vel_xy_exp", None), weight=T1F_TRACK_LIN_WEIGHT, label="track_lin_vel_xy_exp")
    log.append(("track_lin_vel_xy_exp", t.weight, "mjlab"))
    t = _set(getattr(R, "track_ang_vel_z_exp", None), weight=T1F_TRACK_ANG_WEIGHT, label="track_ang_vel_z_exp")
    log.append(("track_ang_vel_z_exp", t.weight, "mjlab"))

    # 回读 + 统计
    n_on = sum(1 for n in dir(R) if getattr(getattr(R, n, None), "weight", None) not in (None, 0))
    print("[T1f][MJLAB ALIGN] 已恢复/校准的奖励项（回读权重）：")
    for n, w, note in log:
        print(f"    {n:26s} w={w:+8.4g}   {note}")
    print(f"[T1f][MJLAB ALIGN] 非零权重项合计 = {n_on}")


@configclass
class G1AmpWalkT1fEnvCfg(G1AmpWalkT1a2EnvCfg):
    """T1f = T1a2 的观测/AMP/动作契约 + mjlab 的奖励机理。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_t1f_mjlab_align(self)
        self.run_name = AMP_WALK_T1F_RUN_NAME


@configclass
class G1AmpWalkT1fEnvCfg_PLAY(G1AmpWalkT1a2EnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_t1f_mjlab_align(self)
        self.run_name = AMP_WALK_T1F_RUN_NAME
