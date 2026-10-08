"""V9c：在 **V9b（无镜像损失）** 之上修三处「我认为在阻止它学会走」的东西。

## 相对 V9b 的三处改动

| 项 | V9/V9b | V9c | 为什么 |
|---|---|---|---|
| `rewards.joint_deviation_arms.weight` | −0.1 | **0.0（归零）** | 实测该项吃 **−2.76/回合（占净回报 11.6%）**，是"抑制手臂探索"的主项；同时 `dof_pos_limits` 只有 −0.0127、`joint_deviation_torso` 只有 −0.0174 ⇒ **限位已经不再被顶**（V8 时代的"腰顶在 ±0.52"症状消失）⇒ 安全网可以松开 |
| `policy.init_noise_std` | 0.25 | **0.2** | 配合下面的 σ 上限下移 |
| `algorithm.entropy_coef` | 0.008 | **0.0025** | ⭐ 这才是把 σ 顶到天花板的推手（∂H/∂σ = 1/σ > 0）；σ=0.39 rad ≈ ±22°/关节/步在 50 Hz 下已经是"巨大的探索噪声"，行走需要的精确双腿时序会被它打碎，而"站/跳"对噪声更鲁棒 |
| `algorithm.clip_max_std` | 0.4 | **0.3** | 保险丝下移（仍允许自适应上升），让 σ 有机会落到 0.25 附近 |
| `algorithm.discriminator_reward_coef` | 0.05 | **0.25** | ⭐ 实测 AMP 项现在只占训练回报 **~11%**（`(1−λ)·c·r_amp = 0.25×0.05×0.2466 = 0.0031` vs `λ·r_task = 0.0254`）⇒ 风格几乎不影响步态。抬到 0.25 后 AMP ≈ **55%**，才是真正的"像参考动作"约束 |
| `use_mirror_loss` | V9b: False | False | 与 V9b 一致（继续隔离镜像损失这个变量） |

## 保留不动的

- `track_default_pose_on_stand`（−1.0）：它其实是**最大的单项负奖励**（−15.3/回合，
  占净回报 64%）——"站着就罚"，这很可能正是"原地跳"这种**既不走也不站**的廉价解
  的成因。但它与"不许驼背站立"的设计意图绑定，**单独消融**更干净，留给下一轮。
- `joint_deviation_torso`（−0.1）：腰的发散守卫，代价只有 −0.35/回合，留着。
- 观测/指令/数据集：与 V9 逐字相同（同一 teacher 契约，故可直接复用 V9 的校验器类）。

## ⚠️ 注意

这是**多变量**改动（不是干净的单变量消融）。它与 V9b 并行跑（各 2048/4096 env），
对比基线：V9（带镜像、7990 轮前）与 V9b（无镜像，其余同 V9）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v9_env_cfg import G1AmpWalkV9EnvCfg, G1AmpWalkV9EnvCfg_PLAY

#: run_name = 日志目录前缀（用户习惯：把关键超参写进目录名，便于事后对账）
AMP_WALK_V9C_RUN_NAME = (
    "G1AmpWalkV9cMVPPlane_obs378_rew15_ar0.02_ns0.2_maxstd0.3_ent0.0025_ampc0.25_noarms_nomirror"
)

#: 归零的手臂偏差权重（原基类值 −0.1）。
AMP_WALK_V9C_ARMS_WEIGHT = 0.0


@configclass
class G1AmpWalkV9cEnvCfg(G1AmpWalkV9EnvCfg):
    """训练用：V9 的环境（279 射线前移 1.5 m / MVP 指令 / 单 clip）+ 手臂偏差归零。"""

    def __post_init__(self):
        super().__post_init__()
        # ⭐ 唯一的环境侧改动：手臂偏差项归零（**保留该项本身**，权重设 0 ⇒
        #    奖励项数仍是 15，teacher 契约（项数/顺序）无需改动，日志里该行会变成 0.0000）。
        self.rewards.joint_deviation_arms.weight = AMP_WALK_V9C_ARMS_WEIGHT
        # 父类会写回 v9 的 RUN_NAME，所以必须在它之后再写一次。
        self.run_name = AMP_WALK_V9C_RUN_NAME


@configclass
class G1AmpWalkV9cEnvCfg_PLAY(G1AmpWalkV9EnvCfg_PLAY):
    """回放用：与训练同奖励/同观测（Play 变体本来就关掉随机化与噪声）。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_deviation_arms.weight = AMP_WALK_V9C_ARMS_WEIGHT
        self.run_name = AMP_WALK_V9C_RUN_NAME


__all__ = [
    "AMP_WALK_V9C_ARMS_WEIGHT",
    "AMP_WALK_V9C_RUN_NAME",
    "G1AmpWalkV9cEnvCfg",
    "G1AmpWalkV9cEnvCfg_PLAY",
]
