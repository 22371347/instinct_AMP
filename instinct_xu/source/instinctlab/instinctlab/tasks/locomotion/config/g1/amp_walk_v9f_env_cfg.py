"""V9f：**V9e + 两处针对"学不动"的改动**（用户假设：学习率偏低 / 探索幅度不足）。

## 用户的观察与假设

> 3k 轮仍在原地站立，σ 在 0.2 以下；我推测是**学习率偏低**导致学习速率降低无法进行动作。
> 之前只会跳跃可能也是该原因 —— 跳跃只需小幅转动脚踝，行走却需要较大的动作幅度。

## 代码核对结果：这个假设**站得住**，而且有一条具体的耦合链

1. 我们用的是 `schedule="adaptive"` + `desired_kl=0.01`（`instinct_rl_ppo_cfg.py:28-32`），
   而 `instinct_rl/algorithms/ppo.py:236-263` 是：
   ```python
   if kl_mean > desired_kl * 2.0:      self.learning_rate = max(1e-5, lr / 1.5)
   elif kl_mean < desired_kl / 2.0:    self.learning_rate = min(1e-2, lr * 1.5)
   param_group["lr"] = self.learning_rate
   ```
   ⇒ 配置里的 `learning_rate = 1e-3` **只是初值**，真实 LR 每步 ×/÷1.5、范围 **[1e-5, 1e-2]**，
   而且**日志里根本没有记录** ⇒ "学不动"到今天为止无从判定。
2. ⭐ **σ 与 LR 的耦合**：KL 大致按 `(Δμ/σ)²·dims/2` 增长 ⇒ **σ 越小，同样的策略均值变化
   产生的 KL 越大** ⇒ 越容易触发 `kl > 2·desired_kl` ⇒ LR 被 ÷1.5。v9c/v9e 把
   σ 从 0.39 压到 0.20（≈ 2× 小）⇒ 同样的动作改进会让 KL 大约 **4×** 大 ⇒ LR 更容易被压下去。
   这恰好解释了观察到的模式：**高 σ 的 v9b「会动」（跳着动），低 σ 的 v9c/v9e 完全不动**。

## 本版改动（两处，都是"让策略能动"的解放，不是加约束）

| 项 | v9e | **v9f** | 理由 |
|---|---|---|---|
| `algorithm.schedule` | `"adaptive"`（LR 每步 ±1.5×） | **`"fixed"`** | 直接去掉"LR 被自适应压低"这条可能；LR 稳定在 1e-3 |
| `policy.init_noise_std` | 0.2 | **0.3** | 探索**幅度**：走路需要髋/膝 0.4~0.6 rad 的往复，0.2 rad 的 σ 下这种幅度在 2~3σ 之外，基本探索不到；而跳只需踝关节小角度（≈1σ） |
| `algorithm.clip_max_std` | 0.3 | **0.4** | 给上面的 σ 留上升空间（仍封顶 ±23°） |

其余全部沿用 v9e：镜像损失 ON、`entropy_coef 0.0025`、`discriminator_reward_coef 0.25`、
`joint_deviation_arms 0`、16 项奖励（含 `feet_air_time`）、原地指令 9%、观测/数据集不变。

## 配套诊断（`algorithms.py` 桥接层，本版起生效）

- `Mean step effective_lr` —— **真实学习率**（自适应规则的输出）。若它远小于 1e-3 ⇒ 假设确认。
- `Mean step action_abs_mean` / `action_abs_max` —— `|上一帧动作|` 的均值/最大值。
  若长期 ≲0.1 ⇒ 策略确实"不敢做大动作"，与"只会小幅踝关节"一致。

## 怎么判读

| 现象 | 结论 |
|---|---|
| `effective_lr` ≈ 1e-3（fixed 下必然）且 `action_abs_mean` 上升到 0.2~0.4、`feet_air_time` > 0 | 假设成立 ⇒ 后面可以继续调 LR/σ |
| `action_abs_mean` 仍 ≲0.1、`feet_air_time` ≈ 0 | 不是 LR/σ 的问题 ⇒ 回到"奖励形状"（压 `track_base_height`/`body_ang_vel_xy`/`track_ang_vel_z` 三个"站着白拿"项） |
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v9e_env_cfg import G1AmpWalkV9eEnvCfg, G1AmpWalkV9eEnvCfg_PLAY

#: run_name = 日志目录前缀（把关键改动写进目录名，便于事后对账）
AMP_WALK_V9F_RUN_NAME = (
    "G1AmpWalkV9fMVPPlane_obs378_rew16_ar0.02_ns0.3_maxstd0.4_fixedlr_ent0.0025"
    "_ampc0.25_noarms_mirror_airtime_halfstand"
)


@configclass
class G1AmpWalkV9fEnvCfg(G1AmpWalkV9eEnvCfg):
    """训练用：与 v9e 的环境**逐字相同**（16 项奖励含 `feet_air_time` + 原地指令 9%），只改 run_name。

    本次改动全在算法侧（`schedule` / σ）⇒ 环境侧不动是**有意为之**（保持与 v9e 可比）。
    """

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V9F_RUN_NAME


@configclass
class G1AmpWalkV9fEnvCfg_PLAY(G1AmpWalkV9eEnvCfg_PLAY):
    """回放用：与训练同奖励/同指令分布。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V9F_RUN_NAME


__all__ = [
    "AMP_WALK_V9F_RUN_NAME",
    "G1AmpWalkV9fEnvCfg",
    "G1AmpWalkV9fEnvCfg_PLAY",
]
