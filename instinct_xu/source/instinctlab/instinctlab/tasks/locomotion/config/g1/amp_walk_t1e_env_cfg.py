"""**T1e**：T1d + 把 `dof_pos_limits` 换回「全关节软边距版」（2026-10-07）。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— 已被 T1a2 覆盖**：今天 = T0 版 `isaaclab.envs.mdp.joint_pos_limits`（**线性、仅踝**，weight -1），非本层全关节软边距版。

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | `dof_pos_limits` 踝-only → **全关节 `joint_pos_limits_margin`**（margin 0.15 rad、power 2.0）+ 倾角/限位诊断 | **已被 T1a2 覆盖** → `joint_pos_limits`（线性、仅踝，-1）|

## 推理与判读（已归档，勿在本文件重复）

* 动机「T1d 暴露全关节限位盲区（T0 版只罚踝 ⇒ 腰/髋/膝在奖励与指标里都成盲区）」→ `EXPERIMENTS.md` §附待办第 2 条
* 软边距形状 + 随之消失的诊断探针（`limit_margin_u_max` 等）→ 见该版 run 日志；⚠️ 相对 T1d 的改动 = "作用域 + 函数形状"（**有意偏离 T0 口径**）

⚠️ 非破坏性；`RewTerm(...)` 字面值不代表生效值（后层整体覆盖 `params`）→ 实际值见
   `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-T1e-G1-v0`（算法侧见同节）
"""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from instinctlab.tasks.locomotion.mdp import rewards_amp as _amp_rewards

from .amp_walk_t1a2_env_cfg import G1AmpWalkT1a2EnvCfg, G1AmpWalkT1a2EnvCfg_PLAY

#: run_name（拼进 run 目录名；`fullbodylimit` 标记"全关节限位惩罚"）
AMP_WALK_T1E_RUN_NAME = "G1AmpWalkT1eMVPPlane_obs378_rew16_fullbodylimit_sym"

#: 恢复 V33 的全关节软边距实现（`margin` / `power` 用函数默认 0.15 / 2.0）
T1E_LIMIT_FUNC = _amp_rewards.joint_pos_limits_margin
T1E_LIMIT_PARAMS = {"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])}
T1E_LIMIT_WEIGHT = -1.0


def _apply_t1e_full_body_limit(cfg) -> None:
    """把 `dof_pos_limits` 换成全关节软边距版，并**当场回读断言**生效。"""
    term = getattr(cfg.rewards, "dof_pos_limits", None)
    if term is None:
        raise ValueError("[T1e] rewards.dof_pos_limits 不存在（被改名或删除了？）")
    term.func = T1E_LIMIT_FUNC
    term.params = dict(T1E_LIMIT_PARAMS)
    term.weight = float(T1E_LIMIT_WEIGHT)

    got_q = getattr(getattr(term, "func", None), "__qualname__", None)
    if got_q != getattr(T1E_LIMIT_FUNC, "__qualname__", None):
        raise ValueError(f"[T1e] dof_pos_limits.func 未生效：{got_q!r}")
    _p = getattr(term, "params", {}) or {}
    _ac = _p.get("asset_cfg")
    _jn = list(getattr(_ac, "joint_names", []) or []) if _ac is not None else None
    if _jn != [".*"]:
        raise ValueError(f"[T1e] 作用域未生效：joint_names={_jn!r}（应为 ['.*'] 全关节）")

    print("[T1e][LIMIT REWARD] dof_pos_limits 已换回**全关节**软边距版：")
    print(f"    func        = {T1E_LIMIT_FUNC.__module__}.{T1E_LIMIT_FUNC.__qualname__}")
    print(f"    joint_names = {_jn}   weight = {term.weight:+.3g}   margin/power = 0.15 / 2.0")


@configclass
class G1AmpWalkT1eEnvCfg(G1AmpWalkT1a2EnvCfg):
    """T1e = T1a2/T1d 的奖励表 + 全关节限位惩罚（其余逐字不动）。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_t1e_full_body_limit(self)
        self.run_name = AMP_WALK_T1E_RUN_NAME


@configclass
class G1AmpWalkT1eEnvCfg_PLAY(G1AmpWalkT1a2EnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_t1e_full_body_limit(self)
        self.run_name = AMP_WALK_T1E_RUN_NAME
