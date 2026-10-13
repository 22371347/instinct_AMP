"""V9g：**奖励重平衡 + 让风格项成为主导**（LR/σ 两条假设已被数据排除）。

## ⚠️ 本层对**今天**仍有影响的部分

* **本层的奖励权重改动几乎全部已被后层覆盖**（逐项见表）；**唯一活到今天的是 `feet_slide = −0.1`**
  （本层从 −0.25 抬上来的，今天仍是 −0.1）。
* 本层"压住站着白拿 / 抬动起来才有 / 让风格项主导"这条**方法论**今天仍在用（后来 V28 又把风格项 ×4，
  方向一致）→ §C2。

## 本层改动一览

| # | 改动（**只改权重，奖励项数仍 16、契约不变**）| 现状（今天实际生效）|
|---|---|---|
| ① | 压"站着白拿"三项：`track_base_height` 1.5→**0.5**、`track_ang_vel_z_exp` 1.2→**0.2**、`body_ang_vel_xy` 0.7→**0.2**；`track_default_pose_on_stand` −1.0→**0.0** | **全被覆盖** → 今天：`track_base_height` **1.5**（V9h 起；T1k/T1l 同值）、`track_ang_vel_z_exp` **1**、`body_ang_vel_xy` **0**（T1i 起）、`track_default_pose_on_stand` **−0.8**（V23 起）|
| ② | 抬"动起来才有"：`track_lin_vel_xy_exp` 1.5→**3.0**、`feet_air_time` 1.0→**2.0**、`feet_slide` −0.25→**−0.1** | `track_lin_vel_xy_exp` **已被覆盖** → **1**（T1k）/ **1.5**（T1l）；`feet_air_time` **已被 T1a2 覆盖** → **1**（换回 `feet_air_time_positive_biped`，`threshold=0.5`）；⭐ `feet_slide` **−0.1 今天仍是**（本层定）|
| ③ | 让风格项主导：`style_lerp` 0.75→**0.5**、`discriminator_reward_coef` 0.25→**0.5**〔算法侧〕| 见 `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V9g-G1-v0` 的**「算法侧」节** |

## 推理与判读（已归档，勿在本文件重复）

* **排除了两条"学不动"假设**（v9f @4360 实测）：学习率偏低 ❌（`Mean step effective_lr = 0.0010`，
  fixed 下正常；σ 从 0.3 **自己降**到 0.23 ⇒ 优化器工作正常）；动作幅度太小 ❌
  （`Mean step action_abs_mean = 0.716 rad ≈ 41°`、`action_abs_max ≈ 5` ⇒ 策略**在大幅发力**，
  只是没产生位移）→ 该版 run 日志 + §C2
* **奖励账目精确闭合**（各 /s 项 ×20 = 44.6 ≈ `Mean reward` 45.0）：`track_base_height` +23.7(53%)、
  `track_lin_vel_xy` +17.4(39%)、`track_ang_vel_z` +11.1(25%)、`body_ang_vel_xy` +10.8(24%)、
  `track_default_pose_on_stand` −13.4、其余负项 −4.6；⇒ **"站住"白拿的三项（23.7+11.1+10.8 = 45.6）
  比整个回报还大**，而唯一要求位移的项只有 17.4 ⇒ 策略不是"学不会"，而是**把奖励最优解找得很准，
  而那个最优解就是站着不动**；`Episode_Reward/feet_air_time = 0.0026/s`（≈0）说明**从未出现过
  "恰好单脚支撑"**（没有迈步）→ §C2
* 「跳跃性价比是自己调出来的」：本层把三个平衡刹车砍过头（`track_ang_vel_z` 1.2→0.2、
  `body_ang_vel_xy` 0.7→0.2、`track_base_height` 1.5→0.5），同时把 `track_lin_vel` 抬到 3.0 → §C2
* 「V9g 的'95% 摔倒'只是**断崖窗口的快照**，不是稳态」：日志反复出现 `rew 37 → 0.31（len=79）`
  这种"崩一下再恢复"的振荡 ⇒ **读结论要看日志包络，不要看单个快照** → §C1
* 判读：`feet_air_time` 从 0.0026/s 起来（>0.1/s ⇒ 真在交替支撑）+ `vx_*_rmse` 下降 + `amp_d` 上升
  ⇒ 重平衡起效；若仍 ≈0 ⇒ 下一步该动"**探索**"（σ/命令课程）而不是奖励（这直接引出 V9i）→ 该版 run 日志

⚠️ 其余逐字不动：镜像损失 ON、σ 与 `entropy_coef`、fixed LR（初值不变）、`joint_deviation_arms 0`、
   原地指令 9%、观测 / 指令范围 / 数据集。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V9g-G1-v0`
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v9e_env_cfg import G1AmpWalkV9eEnvCfg, G1AmpWalkV9eEnvCfg_PLAY

AMP_WALK_V9G_RUN_NAME = (
    "G1AmpWalkV9gMVPPlane_obs378_rew16_rebalance_linvel3_airtime2"
    "_nosstill_nomirror0_mirror_ampc0.5"
)

#: 只改权重（不改结构）：{奖励项名: 新权重}
AMP_WALK_V9G_WEIGHTS = {
    "track_base_height": 0.5,
    "track_ang_vel_z_exp": 0.2,
    "body_ang_vel_xy": 0.2,
    "track_default_pose_on_stand": 0.0,
    "track_lin_vel_xy_exp": 3.0,
    "feet_air_time": 2.0,
    "feet_slide": -0.1,
}


def _apply_weights(rewards) -> None:
    for name, w in AMP_WALK_V9G_WEIGHTS.items():
        term = getattr(rewards, name, None)
        if term is None:
            raise AttributeError(f"V9g 想改权重，但奖励集里没有 {name!r}（项名拼错了？）")
        term.weight = w


@configclass
class G1AmpWalkV9gEnvCfg(G1AmpWalkV9eEnvCfg):
    """训练用：v9e 的奖励集（16 项）**只改权重** + run_name。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_weights(self.rewards)
        self.run_name = AMP_WALK_V9G_RUN_NAME


@configclass
class G1AmpWalkV9gEnvCfg_PLAY(G1AmpWalkV9eEnvCfg_PLAY):
    """回放用：与训练同奖励/同指令分布。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_weights(self.rewards)
        self.run_name = AMP_WALK_V9G_RUN_NAME


__all__ = [
    "AMP_WALK_V9G_RUN_NAME",
    "AMP_WALK_V9G_WEIGHTS",
    "G1AmpWalkV9gEnvCfg",
    "G1AmpWalkV9gEnvCfg_PLAY",
]
