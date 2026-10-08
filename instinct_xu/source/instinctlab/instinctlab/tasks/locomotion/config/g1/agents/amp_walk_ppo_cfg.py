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
    #: | "bounded_quad"（⭐ V29：**对齐 AMP_mjlab** 的 `clamp(1 - k(d-1)^2, 0, 1)`，
    #:   有界、且 d<=-1 时恰为 0 ⇒ 从**形状**上消除“奖励离群”这个失效模式）
    #: ⚠️ 用 "log_quad" / "bounded_quad" 时**必须**把 `discriminator_reward_type` 设为 "wasserstein"
    #:    （那是父类里唯一**原样返回 d** 的分支）。二者不匹配会 fail-fast。
    amp_reward_form = "raw"
    #: ⭐ V29（对齐 AMP_mjlab 的数据管线）：`True` ⇒ 判别器的**参考批独立随机重采 (T,B)**，
    #: 不再使用“与策略同相位的那一步”的参考状态（mjlab 的 `AMPLoader` 是 clip 轮换 + 帧内随机）。
    #: ⚠️ 必须在本 @configclass 里显式声明，否则 `to_dict()` 会静默丢掉它。
    amp_expert_independent_sampling = False

    # ---- ⭐⭐ V30：判别器“AMP_mjlab 原始配方”（2026-10-05）------------------------
    #: `True` ⇒ GP **只在 expert/reference 侧**求梯度（mjlab：
    #: `compute_grad_pen(*sample_amp_expert, lambda_=10)`）。旧口径（False）= actor+expert 合并。
    discriminator_gp_expert_only = False
    #: 判别器优化器的 **trunk** weight decay（mjlab = 1e-3）。⚠️ 优化器级，不是 loss 项。
    discriminator_optimizer_trunk_weight_decay = 0.0
    #: 判别器优化器的 **head（最后一层）** weight decay（mjlab = 1e-2）。
    discriminator_optimizer_head_weight_decay = 0.0
    #: 专家批来自**离线运动数据集**（mjlab `AMPLoader` 口径）的文件路径；
    #: 形如 `/root/amp_expert_states.pt`（键 `states_pair`，形状 `(N, 390)`）。
    #: `None` = 用活体影子机器人（旧行为）。
    amp_expert_buffer_path = None
    #: `-log(1-D)` 在 d=1 处会 +inf（D 恰好=1，而 MSELoss 的参考侧目标就是 +1）
    #: ⇒ 把 D 夹到 1-eps，奖励上界 = -log(eps) ≈ 13.8。
    amp_reward_eps = 1.0e-6
    #: ⭐ `quad` 曲率 `D = clamp(1 − k·(d−1)², 0, 1−eps)`（默认 0.25 = 历史值）。
    #: ⚠️ 下端 clamp 位置 = `1 − sqrt(1/k)`：k=0.25 ⇒ `d ≤ −1.0` 起进死区；
    #:    k=0.125 ⇒ `d ≤ −1.83`。V19b 用 0.125（V19a 实测 d 展宽后有 5% 越过 −1.0）。
    amp_quad_k = 0.25

    # --- 探索噪声的**上限**（v9 起）：防 σ 无界膨胀 -------------------------------
    #: `None` = 不钳上限（v1~v8 的既有行为）。给数值时，桥接层会在 `PPO.update`
    #: 之后额外调一次 `actor_critic.clip_std(min=clip_min_std, max=clip_max_std)`。
    #: ⚠️ 动机：`instinct_rl` 只钳 σ **下限**（`ppo.py:216` 仅 `min=`），而
    #:    `entropy_coef` 会把 σ 一路推高 —— v8 实测 σ 涨到 **1.04 rad**（±60°），
    #:    在 `action_scale=1.0` 下就是直接加在关节目标上的噪声 ⇒ 训练发散。
    #: ⚠️ 同样必须显式声明，否则 `to_dict()` 不会带上它。
    clip_max_std = None
    #: ⭐ σ 上限的**退火周期**（V26 新增；单位 = 本 run 的 `update()` 次数）。
    #: `None` = 不退火（σ 上限恒为 `clip_max_std`，即 v11~v24 的行为）。
    #: 给 N 时：上限从「**本 run 第一次 update 时的 σ**」线性降到 `clip_max_std`，
    #: 用 N 次 update 走完，之后维持 `clip_max_std`。
    #: 动机（V26）：从 σ≈0.57 的 ckpt 热启动时，一次性硬夹到 0.20 会让奖励分布
    #: **整体上移 ~4/步**（噪声税 −5.26 → −0.58）⇒ value function 突跳 ⇒ 用退火摊平。
    #: ⚠️ 同 `clip_max_std`：必须显式声明，否则 `to_dict()` 不会带上它（静默失效）。
    clip_max_std_schedule = None
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

    # --- ⭐ 判别器输入白化（V22，2026-10-03）：AMP_mjlab `amp_normalizer` 的等价物 ----
    #: 真正开启白化还需在 `discriminator_kwargs` 里接上 normalizer，例如：
    #:     {"hidden_sizes": [...], "nonlinearity": "ReLU",
    #:      "normalizer_class_name": "EmpiricalNormalization",
    #:      "normalizer_kwargs": {"eps": 0.01, "until": None}}
    #: 这里默认 **不开**（= V1~V21 的既有行为，零回退风险）。
    #: `amp_normalizer_expert_only=True` = 统计量（mean/var）**只由 expert/reference 侧**累积。
    #: ⚠️ 必须为 `True`：`Discriminator.forward` 默认会顺手更新统计量，而 rollout 期每个
    #:    env step 都会拿**策略数据**前向一次算风格奖励（`wasabi.py: compute_auxiliary_reward`）
    #:    ⇒ 不关掉的话统计量会被策略分布带偏（实测漂移 1.51），白化等于没做。
    amp_normalizer_expert_only = True
    #: ⭐ GP（Lipschitz 惩罚）在哪个空间求 `‖∂d/∂x‖`：`"raw"`（历史行为）| `"normalized"`。
    #: ⚠️ **白化开启时必须用 `"normalized"`**：原始空间里 `∂d/∂x = (∂d/∂x̃)/(σ+eps)`，
    #:    单维放大上限 = `1/eps`；实测 GP 项被放大 311×（未训）~1.2e6×（饱和）
    #:    ⇒ 判别器损失被完全支配。归一化空间下 GP 项与"无白化旧配置"同量级
    #:    （1.0×~3.2×）⇒ `discriminator_gradient_penalty_coef` 可沿用。
    amp_grad_pen_space = "raw"

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
