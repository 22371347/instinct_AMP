"""**T1f**：一次性把 mjlab 的 6 项奖励机理**恢复/校准**回来（2026-10-07）。

## ⚠️ 本层对**今天**仍有影响的部分

* ⭐ **唯一今天仍生效的改动 = `track_lin_vel_xy_exp` 1.0 → 1.5**（经 T1k 保持、T1l 沿用）。
* ⭐ **这是"必须单变量"的最典型案例**：6 项齐改 ⇒ 训练失败（站立盆地，见下）。
* 其余 5 项**今天全部不再生效**（被 T1a2 白名单/实现回退覆盖，见下表）。

## 本层改动一览（⚠️ 本轮是「6 组变量」，非单变量）

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | `feet_lateral_width` 0 → **-10**（`gate_lateral`、`max_cost` 0.10）| **已被 T1a2 覆盖** → **-1** |
| ② | `feet_crossing` 0 → **-20**（保留方向门控）| **已被覆盖** → **-2** |
| ③ | `dof_pos_limits` 踝-only -1 → **全关节 -10**（回标准 `joint_pos_limits`）| **已被 T1a2 覆盖** → **-1**（踝-only，`isaaclab...joint_pos_limits`）|
| ④ | `flat_orientation_l2` -1 → **-2** | **已被覆盖** → **-1** |
| ⑤ | `track_lin_vel_xy_exp` 1.0 → **1.5** | ✅ **1.5**（经 T1k 保持、T1l 沿用）|
| ⑥ | `track_ang_vel_z_exp` 1.0 → **1.2** | **已被覆盖** → **1.0** |

## 推理与判读（已归档，勿在本文件重复）

* ⭐ **失败结果**：6 项齐改 ⇒ `vx_high 0.02@6010`（站立盆地；T1a2 在 1010 轮就有 0.56）⇒ 无法归因 → `EXPERIMENTS.md` §B4
* 动机「踝 `ankle_roll` 贴限位 / `limit_margin_u_max=2.08` / 脚掌侧缘行走」及 mjlab `joint_pos_limits` -10 vs 我们 -1 → §附待办第 2 条
* 算法侧同批对齐见 `agents/amp_walk_t1f_ppo_cfg.py`（不在本文件）

⚠️ 本层是**刻意偏离 T0 口径**的合并改动（目的：把"跟踪 + 类人风格"拉回 mjlab 已验证配置）。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-T1f-G1-v0`（算法侧见同节）
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
