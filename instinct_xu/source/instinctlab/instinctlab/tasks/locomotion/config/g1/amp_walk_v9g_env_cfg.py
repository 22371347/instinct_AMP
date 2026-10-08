"""V9g：**奖励重平衡 + 让风格项成为主导**（LR/σ 两条假设已被数据排除）。

## 排除了什么（v9f @4360 轮实测）

| 假设 | 判定 | 证据 |
|---|---|---|
| 学习率偏低 | ❌ 排除 | `Mean step effective_lr = 0.0010`（已 fixed，日志可见）；σ 从 0.3 **自己降**到 0.23 ⇒ 优化器工作正常 |
| 动作幅度太小 | ❌ 排除 | `Mean step action_abs_mean = 0.716 rad`（≈41°！）、`action_abs_max ≈ 5` ⇒ 策略**在大幅发力**，只是没产生位移 |
| 奖励形状 | ✅ **成立** | 见下 |

**奖励账目精确闭合**（各 /s 项 ×20 = 44.6 ≈ `Mean reward` 45.0），且：

| 项 | 回合贡献 | 占 45.0 |
|---|---|---|
| `track_base_height` | +23.7 | **53%** ← 站着不动就能拿 |
| `track_lin_vel_xy` | +17.4 | 39%（唯一真正要求前进的项） |
| `track_ang_vel_z`（MVP 里 wz≡0） | +11.1 | 25% ← 站着不动就满分 |
| `body_ang_vel_xy` | +10.8 | 24% ← 站着不动就接近满分 |
| `track_default_pose_on_stand` | −13.4 | 最大负项 |
| 其余负项（feet_slide/action_rate/flat_orient/termination/…） | −4.6 | |
| 合计 | **+45.0** | ✓ 与日志一致 |

⇒ **"站住"能拿到的三项（23.7+11.1+10.8 = 45.6）比整个回报还大**，而唯一要求位移的项只有 17.4。
另外 `Episode_Reward/feet_air_time = 0.0026/s`（≈0）⇒ **从来没有出现过"恰好单脚支撑"**，
即完全没有迈步。⇒ 策略不是"学不会"，而是**它把奖励最优解找得很准，而这个最优解就是站着不动**。

## 本版改动

**① 压"站着白拿"的三项**（都只改权重，**奖励项数仍 16，契约不变**）：

| 项 | v9e/v9f | **v9g** |
|---|---|---|
| `track_base_height` | 1.5 | **0.5** |
| `track_ang_vel_z_exp` | 1.2 | **0.2** |
| `body_ang_vel_xy` | 0.7 | **0.2** |
| `track_default_pose_on_stand` | −1.0 | **0.0** |

**② 抬"动起来才有"的项**：

| 项 | v9e/v9f | **v9g** |
|---|---|---|
| `track_lin_vel_xy_exp` | 1.5 | **3.0** |
| `feet_air_time`（单脚支撑时长） | 1.0 | **2.0** |
| `feet_slide`（只罚"走"不罚"跳"） | −0.25 | **−0.1** |

**③ 让风格项成为主导（标准 AMP 的权重结构）**：`style_lerp 0.75 → 0.5`、
`discriminator_reward_coef 0.25 → 0.5` ⇒ 有效风格权重 `(1−λ)·c` 从 **0.0625 → 0.25（4×）**。
依据：AMP 的"模仿参考动作"信号在 v9e 里只有 `(1−λ)c·std(r_amp) ≈ 0.009/步`，而任务奖励的
env 间涨落同量级 ⇒ 风格教不动。放宽 λ 后，任务项仍占 50%，但风格项的**反差**翻了 4 倍。

保持不变：镜像 ON、σ 初值 0.25 / 上限 0.4、`entropy_coef 0.0025`、**fixed LR 1e-3**、
`joint_deviation_arms 0`、原地指令 9%、观测/指令范围/数据集全不动。

## 判读

`feet_air_time` 从 0.0026/s 起来（>0.1/s 说明真在交替支撑）+ `vx_*_rmse` 下降 + `amp_d` 上升
⇒ 奖励重平衡起效。若 `feet_air_time` 仍 ≈0，则说明**单脚支撑从当前策略出发探索不到**，
下一步就该动"探索"（σ/命令课程）而不是奖励。
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
