"""**T1a2**：T1a + **把奖励"实现/作用域"逐字对齐 T0**（T1a 只对齐了权重）。

============================================================================
为什么要 T1a2（2026-10-07 控制变量审计的结论）
============================================================================
`cloud_diff_t1a_t0.py` 把两套 cfg 现场建出来对比后发现：T1a 的 17 项权重虽然与 T0
逐项相等，但**同名奖励的实现/作用域大面积不同** ⇒ 无法用 T1a 回答"T0 的能力是否还在"。
且实测 T1a 在 1500 轮收敛到"站着不动"（九档 `vel_mean` 全 ≈0，`vx_high` 0.006，
T0 同期 0.643）—— 与 S8 的失败模式一致。

本文件把**生效项**中实现不同的三处换回 T0：

| 项 | T1a（错） | **T1a2（= T0）** |
|---|---|---|
| `feet_air_time` (w=+1) | `rewards_amp.feet_air_time_target_biped`（目标带 0.4~1.0 s） | `rewards.feet_air_time_positive_biped`，`threshold=0.5` |
| `dof_pos_limits` (w=−1) | `rewards_amp.joint_pos_limits_margin`（margin 0.15、二次、**全 29 关节**） | `isaaclab.envs.mdp.joint_pos_limits`（线性、**仅踝 pitch/roll**） |
| `dof_acc_l2` (w=−2e-7) | `joint_acc_l2`，`asset_cfg=.*`（**全 29 关节**） | `joint_acc_l2`，`asset_cfg=hip/knee` |

============================================================================
**刻意保留**（用户点名的核心部分，不算"未对齐"）
============================================================================
1. `track_default_pose_on_stand`(−0.8) —— 交付要求②（静止恢复初始姿态）。
2. `mirror_loss`（策略层镜像损失，coef 0.1）—— 交付要求②（对称能力）。
3. AMP 判别器全套 + `style_lerp=1.0`（风格项贡献恰为 0；判别器仍训练 ⇒ T1b 可热启动）。
4. V33S 骨架：378 观测（含 `base_lin_vel` 教师特权）/ 396 AMP 状态 / 全向分档指令。

============================================================================
**已知且接受的残余差异**（写在这里，避免以后误读为"完全等价"）
============================================================================
* `track_lin_vel_xy_exp` 用 `rewards_amp.track_lin_vel_xy_axis_diag`：它是
  `track_lin_vel_xy_yaw_frame_exp` 的**逐样本等价**无损诊断版，**运行期恒等自检**
  （`max|Δ| > 1e-5` 直接抛错）⇒ 训练奖励完全不变，白拿 `lin_track_*` 曲线。
* T0 的 `stand_still`(−0.8) 换成 `track_default_pose_on_stand`(−0.8)：
  T0 那项本质是"静止时关节偏离 default 的 L1"，语义相近（见审计记录）。
* 观测 378 vs 623、指令分布（`ang_vel_z ±0.8`、6~15 s 重采样）保持 V33S —— 用户确认。
"""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from instinctlab.tasks.locomotion.mdp import rewards as locomotion_rewards

from .amp_walk_t1a_env_cfg import (
    T1A_REWARD_WEIGHTS,
    T1A_ZEROED,
    _apply_t1a_reward_whitelist,
)
from .amp_walk_v33_env_cfg import G1AmpWalkV33SEnvCfg, G1AmpWalkV33SEnvCfg_PLAY

#: run_name（拼进 run 目录名；`t0impl` 标记"实现已对齐 T0"）
AMP_WALK_T1A2_RUN_NAME = "G1AmpWalkT1a2MVPPlane_obs378_rew16_t0impl_sym"

#: ⭐ 与 T0 **逐字相同**的 `(func, params)`。`params` 里的 `SceneEntityCfg` 必须带上
#: `body_names` / `joint_names` —— 作用域不同等于换了奖励（审计踩过的坑）。
T1A2_T0_IMPL: dict[str, tuple] = {
    "feet_air_time": (
        locomotion_rewards.feet_air_time_positive_biped,
        {
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.5,
        },
    ),
    "dof_pos_limits": (
        isaac_mdp.joint_pos_limits,
        {
            "asset_cfg": SceneEntityCfg(
                name="robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
            )
        },
    ),
    "dof_acc_l2": (
        isaac_mdp.joint_acc_l2,
        {
            "asset_cfg": SceneEntityCfg(
                name="robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
            )
        },
    ),
}


def _apply_t1a2_t0_impls(cfg) -> None:
    """把上面三项的 `func` 与 `params` **整体替换**为 T0 的版本，并当场断言生效。

    ⚠️ 只写权重不写实现 = 假对齐（T1a 的教训）。这里改完立刻回读断言：
      · `func.__qualname__` 必须等于登记的实现
      · 作用域参数必须存在（`sensor_cfg` / `asset_cfg`）
    不通过就抛错，不让训练带着"假对齐"的奖励跑。
    """
    for name, (func, params) in T1A2_T0_IMPL.items():
        term = getattr(cfg.rewards, name, None)
        if term is None:
            raise ValueError(f"[T1a2] 待对齐项不存在：rewards.{name}（被改名或删除了？）")
        term.func = func
        term.params = dict(params)

    # ---- 回读断言（fail-fast，防止"改了但没生效"）------------------------------
    bad: list[str] = []
    for name, (func, params) in T1A2_T0_IMPL.items():
        term = getattr(cfg.rewards, name)
        got_q = getattr(getattr(term, "func", None), "__qualname__", None)
        want_q = getattr(func, "__qualname__", None)
        if got_q != want_q:
            bad.append(f"{name}.func: {got_q} != {want_q}")
        for key in params:
            if key not in (getattr(term, "params", {}) or {}):
                bad.append(f"{name}.params 缺 {key}")
    if bad:
        raise ValueError("[T1a2] 奖励实现对齐断言失败：\n  " + "\n  ".join(bad))

    print("[T1a2][REWARD IMPL] 已对齐为 T0 的实现（func + 作用域）：")
    for name, (func, params) in T1A2_T0_IMPL.items():
        term = getattr(cfg.rewards, name)
        scope = ""
        p = getattr(term, "params", {}) or {}
        if "asset_cfg" in p:
            scope = f"joint_names={getattr(p['asset_cfg'], 'joint_names', None)!r}"
        elif "sensor_cfg" in p:
            scope = f"body_names={getattr(p['sensor_cfg'], 'body_names', None)!r}"
        print(f"    {name:22s} {getattr(func, '__qualname__', func):32s} {scope}")


@configclass
class G1AmpWalkT1a2EnvCfg(G1AmpWalkV33SEnvCfg):
    """T1a2 = V33S 骨架 + T0 权重 + **T0 的实现/作用域** + default_pose + mirror。"""

    def __post_init__(self):
        super().__post_init__()
        # ① 权重白名单：16 项按 T0 赋值 + 5 项显式置 0 + 未归类 fail-fast
        _apply_t1a_reward_whitelist(self)
        # ② ⭐ 实现对齐：把生效项里"同名不同实现"的三处换成 T0 的（T1a 漏掉的一步）
        _apply_t1a2_t0_impls(self)
        self.run_name = AMP_WALK_T1A2_RUN_NAME


@configclass
class G1AmpWalkT1a2EnvCfg_PLAY(G1AmpWalkV33SEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_t1a_reward_whitelist(self)
        _apply_t1a2_t0_impls(self)
        self.run_name = AMP_WALK_T1A2_RUN_NAME
