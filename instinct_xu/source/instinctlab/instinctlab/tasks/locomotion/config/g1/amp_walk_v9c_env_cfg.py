"""V9c：在 **V9b（无镜像损失）** 之上修三处"我认为在阻止它学会走"的东西（**多变量**；2026-09-29）。

## ⚠️ 本层对**今天**仍有影响的部分

* **无 —— 全部已被后层覆盖**：环境侧唯一改动（`joint_deviation_arms` 归零）已被 **T1a 白名单**
  设回 **−0.1**；5 项算法侧改动也都被后层改写（见下表）。
* 仍"活着"的是**结论**：① 膝/髋/腰限位当年已不再被顶（安全网可松）；② σ 是被 `entropy_coef`
  顶上去的，而 `action_rate_l2` 是奖励集里主要的**密集梯度来源**（信号变弱 ⇒ σ 膨胀）
  → `docs/amp_walk/EXPERIMENTS.md` §B5。

## 本层改动一览

| # | 项 | V9/V9b → **V9c** | 现状（今天实际生效）|
|---|---|---|---|
| ① | `rewards.joint_deviation_arms.weight` | −0.1 → **0.0（归零；保留该项 ⇒ 奖励项数仍 15**、日志该行变 0）| ❌ **已被 T1a 白名单覆盖** → **−0.1** |
| ② | `policy.init_noise_std`〔算法侧〕 | 0.25 → **0.2** | 见 `EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V9c-G1-v0` 的「算法侧」节 |
| ③ | `algorithm.entropy_coef`〔算法侧〕 | 0.008 → **0.0025** | 同上 |
| ④ | `algorithm.clip_max_std`〔算法侧〕 | 0.4 → **0.3**（保险丝下移，仍允许自适应上升）| 同上 |
| ⑤ | `algorithm.discriminator_reward_coef`〔算法侧〕 | 0.05 → **0.25** | 同上 |
| ⑥ | `use_mirror_loss`〔算法侧〕 | V9b=False → **False**（继续隔离镜像损失这个变量）| 同上 |

## 推理与判读（已归档，勿在本文件重复）

* ① 的依据：实测该项吃 **−2.76/回合（占净回报 11.6%）**，是"抑制手臂探索"的主项；同时
  `dof_pos_limits` 只有 −0.0127、`joint_deviation_torso` 只有 −0.0174 ⇒ **限位已经不再被顶**
  （V8 时代"腰顶在 ±0.52"的症状消失）⇒ 安全网可以松开 → 该版 run 日志
* ⭐ **σ 的机制（本层最重要的洞察）**：`∂H/∂σ = 1/σ > 0` ⇒ `entropy_coef` 才是把 σ 顶到天花板的推手；
  σ=0.39 rad ≈ **±22°/关节/步**（50 Hz）已经"巨大"，行走需要的精确双腿时序会被它打碎，而"站/跳"更鲁棒
  → `docs/amp_walk/EXPERIMENTS.md` §B5（`action_rate_l2` 被抽掉 ⇒ σ 膨胀的实测：0.766 → 1.580）
* ⑤ 的依据：实测 AMP 项只占训练回报 **~11%**（`(1−λ)·c·r_amp = 0.25×0.05×0.2466 = 0.0031` vs
  `λ·r_task = 0.0254`）；抬到 0.25 后 AMP ≈ **55%**，才算真正的"像参考动作"约束 → §B3（λ 的陷阱）+ §A3
* ⚠️ **本版是多变量改动**（不是干净的单变量消融），与 V9b 并行跑作对比基线
  （V9 = 带镜像、7990 轮前；V9b = 无镜像、其余同 V9）→ §E3

⚠️ 其余（**今天多已变动**）：保留不动的 `track_default_pose_on_stand(−1.0)` —— 它其实是**最大的单项负奖励**
   （−15.3/回合，占净回报 64%），"站着就罚"很可能正是"原地跳"这种**既不走也不站**的廉价解的成因；
   但它与"不许驼背站立"的设计意图绑定，**单独消融更干净，留给下一轮** → §B1。
   `joint_deviation_torso(−0.1)`（腰的发散守卫，代价只 −0.35/回合）**今天已被 T1j 置 0**。
   观测/指令/数据集与 V9 逐字相同（同一 teacher 契约 ⇒ 可直接复用 V9 的校验器类）；
   ⚠️ 该 MVP 数据集/指令此后已被 V10 → V18 → V33 覆盖（今天 = 5 clip + 全向分档指令）
   —— **grep 旁证、未运行期核对**。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V9c-G1-v0`
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
