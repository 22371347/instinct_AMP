"""AMP 行走教师（privileged）的 RL 配置。

- **步骤②** 先建立与 OmniLoco 完全一致的纯 PPO 骨架（保证「任务骨架本身可训练」，
  与学长基线严格可比）；
- **步骤⑥**（本文件现状）把它换成 **WASABI / AMP**：
  - 算法 `class_name` → `"instinctlab.utils.wrappers.instinct_rl:WasabiPPOAmpMix"`
    （`module:class` 形式，依赖对 `instinct_rl/runners/on_policy_runner.py` 的 1 行修复）；
  - `reward_mix_mode="lerp"` + `style_lerp=0.75` 精确复刻 AMP_mjlab 的
    $R=(1-\\lambda)\\,c\\,r_{\\text{amp}}+\\lambda\\,r_{\\text{task}}$；
  - 判别器参数以上游 parkour 官方 `AmpAlgoCfg` 为准，奖励形式取 AMP_mjlab 的 `"quad"`。

⚠️ `@configclass` 是 dataclass，**父类没有的字段必须在这里显式声明**，
否则 `agent_cfg.to_dict()` 不会带上它们，runner 也就传不到算法构造函数里去
（PLAN §5.2 第 4 条）。

⚠️ 为什么不用 `"WasabiPPO"` 而是自建子类：我们的奖励是 **AMP_mjlab 的凸组合**，
而 `WasabiPPO` 走的是**加法**（$R = r_{\\text{task}} + c\\,r_{\\text{amp}}$）。
自建子类只覆盖 `compute_auxiliary_reward` 一个方法就把两者严格等价起来，
且不碰任何奖励函数（推导见 `instinctlab/utils/wrappers/instinct_rl/algorithms.py`）。
"""

from isaaclab.utils import configclass

from .instinct_rl_ppo_cfg import AlgorithmCfg, NormalizersCfg, PolicyCfg
from .omni_loco_ppo_cfg import G1OmniLocoPPORunnerCfg

#: 奖励凸组合里的风格占比 $\lambda$（= AMP_mjlab 的 `amp_task_reward_lerp`）。
AMP_WALK_STYLE_LERP = 0.75
#: 风格项整体缩放 $c$（= AMP_mjlab 的 `amp_reward_coef`，也是 `WasabiPPO` 的
#: `discriminator_reward_coef`）。
#:
#: ⚠️ 从 **0.1** 起是**刻意取弱**：AMP_mjlab 的实测教训是「风格先验必须先弱」——
#: 他们一度把 `amp_reward_coef` 从 0.1 降到 0.05，因为强风格先验会压住任务奖励
#: （速度跟踪学不出来）。上游 parkour 用 0.25，那是它自己的任务标定值，不照抄。
#: 跑起来后看 TensorBoard 的 `Step/amp_style_reward` 与任务奖励的**量级对比**再决定是否下调。
AMP_WALK_DISCRIMINATOR_REWARD_COEF = 0.1
#: 判别器隐藏层。取上游 parkour 官方 `AmpAlgoCfg` 的值。
#:
#: ⚠️ 这里**不能省**：`Discriminator.build_model` 是
#: `MlpModel(input_size=..., output_size=..., **kwargs)`，而 `MlpModel.__init__` 的
#: `hidden_sizes` 是**必填参数**，漏了会在 `init_storage` 里 TypeError。
AMP_WALK_DISCRIMINATOR_HIDDEN_SIZES = [1024, 512]


@configclass
class AmpWalkPolicyCfg(PolicyCfg):
    """512/256/128（PLAN §5.2 第 7 条：AMP 行走建议与 AMP_mjlab 一致）。

    比 `PolicyCfg` 的 256/128/128 更大：判别器要区分风格，策略容量太小的话
    「任务奖励压不过风格项」时会先崩掉。

    继承 `PolicyCfg`（而不是平级的 `InstinctRlActorCriticCfg`）是**有意的**：
    这样 `G1AmpWalkPPORunnerCfg.policy` 的类型是父类 `policy` 类型的**子类**，
    重新声明字段在 IsaacLab 的 `@configclass` 下不会有任何类型兼容性歧义。
    """

    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]


@configclass
class AmpWalkAlgorithmCfg(AlgorithmCfg):
    """继承 `AlgorithmCfg`：PPO 超参（`value_loss_coef`/`clip_param`/`entropy_coef`/…）
    全部照旧，不重复声明；本类只改/加 AMP 相关字段。

    同样继承了 `InstinctRlPpoAlgorithmCfg`，可以直接覆盖 `class_name`。
    """

    # --- 算法类：`module:Class` 形式，指向 instinctlab 侧的自定义子类 ---
    class_name = "instinctlab.utils.wrappers.instinct_rl:WasabiPPOAmpMix"

    # --- 奖励混合（复刻 AMP_mjlab）-----------------------------------------
    reward_mix_mode = "lerp"
    style_lerp = AMP_WALK_STYLE_LERP
    # AMP_mjlab 的 `amp_disable_style_on_stand` / `stand_speed_threshold`。
    # 本任务默认**关**：`prob_low` 档已经用 `track_default_pose_on_stand` 的平滑门控
    # 处理了低速回位，两套机制叠加容易互相打架；需要时可一键打开。
    disable_style_on_stand = False
    stand_speed_threshold = 0.1

    # --- 镜像损失（PPO 左右等变性，仿 AMP_mjlab `symmetry_cfg`）--------------
    # ⚠️ 与 `style_lerp` 同理：**必须在算法 `@configclass` 里显式声明**，
    #    否则 `to_dict()` 不会带上它们（算法侧 `__init__` 的形参收不到）。
    use_mirror_loss = False
    mirror_loss_coeff = 0.1
    #: articulation 序的 29 个关节名（不是 npz 序！）。
    #: 算子按**关节名**求左右置换，所以顺序无所谓，但必须与 obs/动作一致。
    mirror_joint_names = None

    # --- 风格奖励的函数形式（v8 起：LLP 论文的 r = -log(1 - quad(d))）-----------
    # ⚠️ 同样**必须在算法 @configclass 里显式声明**，否则 `to_dict()` 不会带上它们。
    #: "raw"（默认，与父类一致）| "log_quad"（r = -log(1-D)，D = clamp(1-(d-1)^2/4, 0)）
    #: ⚠️ 用 "log_quad" 时**必须**把 `discriminator_reward_type` 设为 "wasserstein"
    #:    （那是父类里唯一**原样返回 d** 的分支）。二者不匹配会 fail-fast。
    amp_reward_form = "raw"
    #: `-log(1-D)` 在 d=1 处会 +inf（D 恰好=1，而 MSELoss 的参考侧目标就是 +1）
    #: ⇒ 把 D 夹到 1-eps，奖励上界 = -log(eps) ≈ 13.8。
    amp_reward_eps = 1.0e-6
    # ⚠️ 必须显式设 `[1.0]`：`PPO.__init__` 收到**空列表**时会把系数退化成标量 1.0，
    # 而 `process_env_step` 里是
    # `coef * v * self.auxiliary_reward_per_env_reward_coefs` —— 标量只能靠广播侥幸对上，
    # 显式 `[1.0]`（→ shape (1,1)）才与 `transition.rewards` 的
    # (num_envs, num_rewards) 严格广播。
    auxiliary_reward_per_env_reward_coefs = [1.0]

    # --- 判别器（上游 parkour 官方 `AmpAlgoCfg` + AMP_mjlab 的损失/奖励形式）---
    discriminator_class_name = "Discriminator"
    discriminator_kwargs = {
        "hidden_sizes": AMP_WALK_DISCRIMINATOR_HIDDEN_SIZES,
        "nonlinearity": "ReLU",
    }
    discriminator_reward_coef = AMP_WALK_DISCRIMINATOR_REWARD_COEF
    # ⚠️ `"quad"` = `clamp(1 - 0.25*(d-1)^2, min=0)`，正是 **AMP_mjlab 用的形式**
    #   （其 `rsl_rl/modules/discriminator.py`：
    #    `amp_reward_coef * torch.clamp(1 - (1/4)*torch.square(d - 1), min=0)`）。
    #   上游 parkour 也用 "quad"，两边一致。
    discriminator_reward_type = "quad"
    # `MSELoss` → `actor_d → -1`、`reference_d → +1`，AMP_mjlab 的用法。
    # （WASABI 默认的 `BCEWithLogitsLoss` 才是传统 AMP。）
    discriminator_loss_func = "MSELoss"
    discriminator_gradient_penalty_coef = 5.0
    discriminator_optimizer_class_name = "AdamW"
    discriminator_optimizer_kwargs = {"lr": 1.0e-4, "betas": [0.9, 0.999]}
    discriminator_weight_decay_coef = 3.0e-4
    discriminator_logit_weight_decay_coef = 0.04

    # --- AMP_mjlab 的 `min_normalized_std = [0.05]*29` 的**等价物** -------------
    # instinct_rl 的 `ActorCritic.std` 是绝对标准差（`init_noise_std * ones`），
    # `PPO.update` 末尾会 `clip_std(min=clip_min_std)`；本配置 `init_noise_std=1.0`，
    # 所以「归一化下限 0.05」== 「绝对下限 0.05」。
    # 防止策略过早塌成确定性策略、让判别器单方面主导。
    clip_min_std = 0.05


@configclass
class G1AmpWalkPPORunnerCfg(G1OmniLocoPPORunnerCfg):
    policy: AmpWalkPolicyCfg = AmpWalkPolicyCfg()
    algorithm: AmpWalkAlgorithmCfg = AmpWalkAlgorithmCfg()
    normalizers: NormalizersCfg = NormalizersCfg()

    experiment_name = "g1_amp_walk"

    def __post_init__(self):
        super().__post_init__()
        self.resume = False
        self.load_run = None
        self.run_name = "step6_wasabi_amp_quad_c0.1_lerp0.75"
