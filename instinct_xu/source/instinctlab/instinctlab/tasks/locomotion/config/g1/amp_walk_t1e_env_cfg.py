"""**T1e**：T1d 的配置 + **把 `dof_pos_limits` 换回「全关节软边距版」**。

为什么（2026-10-07，T1d 实测暴露）
----------------------------------
T1a2 为了"与 T0 实现逐字对齐"，把 V33 的 `joint_pos_limits_margin` 换成了 T0 的
`isaaclab.envs.mdp.joint_pos_limits`，而 T0 那项的作用域是
    joint_names = ['.*_ankle_pitch_joint', '.*_ankle_roll_joint']
⇒ **只罚踝**。于是腰/髋/膝的限位在**奖励**与**指标**里都成了盲区：
  * `Episode_Reward/dof_pos_limits` 只反映踝关节；
  * margin 版自带的诊断（`limit_margin_u_max` / `limit_margin_u_frac` /
    `waist_pitch_abs_mean` / `waist_pitch_enter_frac`）随实现一起消失。
T1d 因此出现关节限位问题（用户 2026-10-07 观察）⇒ 本臂把全关节软边距版**恢复**。

形状（逐关节归一化到"硬限位处 = 1.0"，`margin=0.15 rad`、`power=2`）
    u_j  = [clamp(lower+margin − q, 0) + clamp(q − (upper−margin), 0)] / margin
    cost = Σ_j u_j ** power
⇒ 距限位仍有 margin 以上**不罚**；进入边距温和提醒；到硬限位 = 1.0；越界按幂增长。

⚠️ 这是**有意的偏离**：T0 的踝-only 口径是它自己的取舍，本臂不再与之逐字等价。
   ⇒ 相对 T1d 的改动是"限位奖励的作用域 + 函数形状"这一项（登记在案）。
⚠️ 本文件不改动、不删除任何既有 cfg（保持非破坏性）。
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
