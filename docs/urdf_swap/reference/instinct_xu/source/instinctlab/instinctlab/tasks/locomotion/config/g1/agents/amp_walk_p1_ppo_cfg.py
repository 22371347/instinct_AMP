"""⭐ **自动生成**，不要手改结构 —— 只允许添加注释/文档。

来源（继承链的**解析结果**）：
  * `instinctlab.tasks.locomotion.config.g1.agents.amp_walk_pelvis_ppo_cfg:G1AmpWalkPelvisPPORunnerCfg` → `G1AmpWalkP1PPORunnerCfg`
上游基类：`instinctlab.utils.wrappers.instinct_rl.rl_cfg.InstinctRlOnPolicyRunnerCfg`；另有 5 个自动生成的子 cfg 类。

重新生成（在有 IsaacLab 的机器上）：
    python3 instinct/tools/flatten_cfg.py generate \
        --cls instinctlab.tasks.locomotion.config.g1.agents.amp_walk_pelvis_ppo_cfg:G1AmpWalkPelvisPPORunnerCfg \
        --out <本文件> --root-name <根类名>
等价性校验（必须 diff 为空）：
    python3 instinct/tools/flatten_cfg.py verify \
        --chain instinctlab.tasks.locomotion.config.g1.agents.amp_walk_pelvis_ppo_cfg:G1AmpWalkPelvisPPORunnerCfg --flat-file <本文件> --flat-attr G1AmpWalkP1PPORunnerCfg
"""

from __future__ import annotations

from typing import Any
import os

#: 本文件所在目录（路径字面量都以它为基准，保证生成物换机可用）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import instinctlab.utils.wrappers.instinct_rl as _m0
from isaaclab.utils import configclass


@configclass
class PolicyCfg:

    """`instinctlab.tasks.locomotion.config.g1.agents.amp_walk_ppo_cfg.AmpWalkPolicyCfg` 的展平副本（链上位置 `policy`；字段值 = 链式配置的最终生效值）。"""
    class_name: str = None
    init_noise_std: float = None
    actor_hidden_dims: Any = None
    critic_hidden_dims: Any = None
    activation: str = None

    def __post_init__(self):
        self.class_name = 'ActorCritic'
        self.init_noise_std = 0.25
        self.actor_hidden_dims = [512, 256, 128]
        self.critic_hidden_dims = [512, 256, 128]
        self.activation = 'elu'


@configclass
class AlgorithmCfg:

    """`instinctlab.tasks.locomotion.config.g1.agents.amp_walk_ppo_cfg.AmpWalkAlgorithmCfg` 的展平副本（链上位置 `algorithm`；字段值 = 链式配置的最终生效值）。"""
    class_name: str = None
    value_loss_coef: float = None
    use_clipped_value_loss: bool = None
    clip_param: float = None
    entropy_coef: float = None
    num_learning_epochs: int = None
    num_mini_batches: int = None
    learning_rate: float = None
    optimizer_class_name: str = None
    schedule: str = None
    gamma: float = None
    lam: float = None
    advantage_mixing_weights: float = None
    desired_kl: float = None
    max_grad_norm: float = None
    clip_min_std: float = None
    reward_mix_mode: str = None
    style_lerp: float = None
    disable_style_on_stand: bool = None
    stand_speed_threshold: float = None
    use_mirror_loss: bool = None
    mirror_loss_coeff: float = None
    mirror_joint_names: Any = None
    amp_reward_form: str = None
    amp_expert_independent_sampling: bool = None
    discriminator_gp_expert_only: bool = None
    discriminator_optimizer_trunk_weight_decay: float = None
    discriminator_optimizer_head_weight_decay: float = None
    amp_expert_buffer_path: str = None
    amp_reward_eps: float = None
    amp_quad_k: float = None
    clip_max_std: Any = None
    clip_max_std_schedule: Any = None
    auxiliary_reward_per_env_reward_coefs: Any = None
    discriminator_class_name: str = None
    discriminator_kwargs: Any = None
    discriminator_reward_coef: float = None
    discriminator_reward_type: str = None
    discriminator_loss_func: str = None
    discriminator_gradient_penalty_coef: float = None
    discriminator_optimizer_class_name: str = None
    discriminator_optimizer_kwargs: Any = None
    discriminator_weight_decay_coef: float = None
    discriminator_logit_weight_decay_coef: float = None
    amp_normalizer_expert_only: bool = None
    amp_grad_pen_space: str = None
    fresh_discriminator_on_load: bool = None
    policy_obs_frames: Any = None

    def __post_init__(self):
        self.class_name = 'instinctlab.utils.wrappers.instinct_rl:WasabiPPOAmpMix'
        self.value_loss_coef = 1.0
        self.use_clipped_value_loss = True
        self.clip_param = 0.2
        self.entropy_coef = 0.005
        self.num_learning_epochs = 5
        self.num_mini_batches = 4
        self.learning_rate = 0.001
        self.optimizer_class_name = 'AdamW'
        self.schedule = 'adaptive'
        self.gamma = 0.99
        self.lam = 0.95
        self.advantage_mixing_weights = 1.0
        self.desired_kl = 0.01
        self.max_grad_norm = 1.0
        self.clip_min_std = 0.0125
        self.reward_mix_mode = 'lerp'
        self.style_lerp = 0.75
        self.disable_style_on_stand = True
        self.stand_speed_threshold = 0.1
        self.use_mirror_loss = True
        self.mirror_loss_coeff = 0.1
        self.mirror_joint_names = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']
        self.amp_reward_form = 'bounded_quad'
        self.amp_expert_independent_sampling = True
        self.discriminator_gp_expert_only = True
        self.discriminator_optimizer_trunk_weight_decay = 0.001
        self.discriminator_optimizer_head_weight_decay = 0.01
        self.amp_expert_buffer_path = os.path.normpath(os.path.join(_THIS_DIR, '../../../../../../../../../../amp_expert_states_396_pelvis.pt'))
        self.amp_reward_eps = 0.05
        self.amp_quad_k = 0.25
        self.clip_max_std = None
        self.clip_max_std_schedule = None
        self.auxiliary_reward_per_env_reward_coefs = [1.0,]
        self.discriminator_class_name = 'Discriminator'
        self.discriminator_kwargs = {'hidden_sizes': [1024, 512, 256], 'nonlinearity': 'ReLU', 'normalizer_class_name': 'EmpiricalNormalization', 'normalizer_kwargs': {'eps': 0.01, 'until': None}}
        self.discriminator_reward_coef = 0.1
        self.discriminator_reward_type = 'wasserstein'
        self.discriminator_loss_func = 'MSELoss'
        self.discriminator_gradient_penalty_coef = 10.0
        self.discriminator_optimizer_class_name = 'Adam'
        self.discriminator_optimizer_kwargs = {'lr': 0.001, 'betas': [0.9, 0.999]}
        self.discriminator_weight_decay_coef = 0.0
        self.discriminator_logit_weight_decay_coef = 0.0
        self.amp_normalizer_expert_only = True
        self.amp_grad_pen_space = 'normalized'
        self.fresh_discriminator_on_load = True
        self.policy_obs_frames = {'height_scan': 4, 'velocity_command': 4, 'projected_gravity': 4, 'base_ang_vel': 4, 'joint_pos_rel': 4, 'joint_vel': 4, 'last_action': 4}


@configclass
class InstinctRlNormalizerCfg:

    """`instinctlab.utils.wrappers.instinct_rl.rl_cfg.InstinctRlNormalizerCfg` 的展平副本（链上位置 `normalizers.policy`；字段值 = 链式配置的最终生效值）。"""
    class_name: str = None

    def __post_init__(self):
        self.class_name = 'EmpiricalNormalization'


@configclass
class InstinctRlNormalizerCfg_1:

    """`instinctlab.utils.wrappers.instinct_rl.rl_cfg.InstinctRlNormalizerCfg` 的展平副本（链上位置 `normalizers.critic`；字段值 = 链式配置的最终生效值）。"""
    class_name: str = None

    def __post_init__(self):
        self.class_name = 'EmpiricalNormalization'


@configclass
class NormalizersCfg:

    """`instinctlab.tasks.locomotion.config.g1.agents.instinct_rl_ppo_cfg.NormalizersCfg` 的展平副本（链上位置 `normalizers`；字段值 = 链式配置的最终生效值）。"""
    policy: Any = None  # → InstinctRlNormalizerCfg
    critic: Any = None  # → InstinctRlNormalizerCfg

    def __post_init__(self):
        self.policy = InstinctRlNormalizerCfg()
        self.critic = InstinctRlNormalizerCfg_1()


@configclass
class G1AmpWalkP1PPORunnerCfg(_m0.InstinctRlOnPolicyRunnerCfg):
    """`instinctlab.tasks.locomotion.config.g1.agents.amp_walk_pelvis_ppo_cfg:G1AmpWalkPelvisPPORunnerCfg` 的**展平**版本：不继承本项目任何其它配置类。

    由 `instinct/tools/flatten_cfg.py` 从链式配置的**解析结果**生成，上游基类 = `instinctlab.utils.wrappers.instinct_rl.rl_cfg.InstinctRlOnPolicyRunnerCfg` (链长 48 层)。
    """

    policy: Any = None  # → PolicyCfg
    algorithm: Any = None  # → AlgorithmCfg
    normalizers: Any = None  # → NormalizersCfg
    seed: int = None
    device: str = None
    num_steps_per_env: int = None
    max_iterations: int = None
    ckpt_manipulator: Any = None
    save_interval: int = None
    log_interval: int = None
    experiment_name: str = None
    run_name: str = None
    resume: bool = None
    load_run: Any = None
    load_checkpoint: str = None

    def __post_init__(self):
        super().__post_init__()  # 只跑上游（InstinctRlOnPolicyRunnerCfg）的钩子
        """全部字段按链式配置的**最终生效值**写死。"""
        self.policy = PolicyCfg()
        self.algorithm = AlgorithmCfg()
        self.normalizers = NormalizersCfg()
        self.seed = 42
        self.device = 'cuda:0'
        self.num_steps_per_env = 24
        self.max_iterations = 1000000000
        self.ckpt_manipulator = None
        self.save_interval = 500
        self.log_interval = 10
        self.experiment_name = 'g1_amp_walk'
        self.run_name = 'p1_pelvis_official_rev10_cold'
        self.resume = False
        self.load_run = None
        self.load_checkpoint = 'model_.*.pt'

        # ---- 展平自检锚点：run_name（链式 vs 展平必须一致，diff 会抓）----
        self.run_name = 'p1_pelvis_official_rev10_cold'
