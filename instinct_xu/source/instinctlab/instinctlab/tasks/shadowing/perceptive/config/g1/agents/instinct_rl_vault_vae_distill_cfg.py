import os

from isaaclab.utils import configclass

from instinctlab.tasks.shadowing.perceptive.config.g1 import perceptive_vault_cfg as vault_env_cfg
from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlActorCriticCfg,
    InstinctRlConv2dHeadCfg,
    InstinctRlEncoderActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)

PERCEPTIVE_VAULT_TEACHER_LOGDIR = os.environ.get("PERCEPTIVE_VAULT_TEACHER_LOGDIR")
# PolicyObsCfg puts depth_image first as (C,H,W); PHP VisionMLP skips empirical norm on depth.
_VAULT_DEPTH_FLAT_DIM = 1 * vault_env_cfg.VAULT_CAMERA_HEIGHT * vault_env_cfg.VAULT_CAMERA_WIDTH


@configclass
class PhpVisionConv2dEncoderCfg:
    """PHP VisionMLP depth stem: Conv 1→16→32→32 (s=2) + GAP → 32-D."""

    @configclass
    class DepthImageEncoderCfg(InstinctRlConv2dHeadCfg):
        channels = [16, 32, 32]
        kernel_sizes = [3, 3, 3]
        strides = [2, 2, 2]
        paddings = [1, 1, 1]
        hidden_sizes = []
        nonlinearity = "ELU"
        use_maxpool = False
        use_global_avg_pool = True
        output_size = 32
        component_names = ["depth_image"]
        takeout_input_components = True

    depth_image = DepthImageEncoderCfg()


@configclass
class VaultVisionPolicyCfg(InstinctRlEncoderActorCriticCfg):
    """PHP student: CNN+GAP depth encoder + MLP [1024, 512, 256]; critic MLP [512, 256, 128]."""

    init_noise_std = 0.01
    actor_hidden_dims = [1024, 512, 256]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"
    encoder_configs = PhpVisionConv2dEncoderCfg()
    critic_encoder_configs = None


@configclass
class AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "VaePpoDistill"
    kl_loss_func = "kl_divergence"
    # No VAE latent; keep coef at 0 so the zeroed KL term does not affect the loss.
    kl_loss_coef = 0.0
    using_ppo = True
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    # PHP Table VI (Distillation).
    entropy_coef = 0.001
    num_learning_epochs = 2
    num_mini_batches = 96
    learning_rate = 3e-4
    # Adaptive is gated in VaePpoDistill until λ_PPO > 0.1 (PHP §III-C / Table VI).
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0
    distill_target = "mse_sum"

    teacher_act_prob = 0.0
    teacher_obs_group = "teacher"
    distillation_mask_obs_group = "distill_mask"
    # PHP: L = λ_PPO L_PPO + λ_D L_D, with base DAgger scale 10 and λ_D floored at 0.1.
    distillation_loss_coef = 10.0
    distillation_loss_min_coef = 0.1
    # Curriculum over first half of 20k iterations (PHP: K/2 = 10,000).
    loss_curriculum_iterations = 10000
    adaptive_schedule_min_ppo_coef = 0.1
    surrogate_loss_coef = 1.0
    # PHP privileged teacher is a plain MLP ActorCritic (no depth encoder).
    teacher_policy_class_name = InstinctRlActorCriticCfg().class_name
    teacher_policy: dict = {
        "init_noise_std": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_format": {
            "policy": {
                "joint_pos_ref": (1, 29),
                "joint_vel_ref": (1, 29),
                "position_ref": (1, 3),
                "rotation_ref": (1, 6),
                "height_scan": (vault_env_cfg.VAULT_HEIGHT_SCAN_DIM,),
                "global_pelvis_pos": (3,),
                "global_pelvis_vel": (3,),
                "base_lin_vel": (3,),
                "projected_gravity": (3,),
                "base_ang_vel": (3,),
                "joint_pos": (29,),
                "joint_vel": (29,),
                "last_action": (29,),
            },
            "critic": {
                "joint_pos_ref": (1, 29),
                "joint_vel_ref": (1, 29),
                "position_ref": (1, 3),
                "rotation_ref": (1, 6),
                "height_scan": (vault_env_cfg.VAULT_HEIGHT_SCAN_DIM,),
                "global_pelvis_pos": (3,),
                "global_pelvis_vel": (3,),
                "base_lin_vel": (3,),
                "projected_gravity": (3,),
                "base_ang_vel": (3,),
                "joint_pos": (29,),
                "joint_vel": (29,),
                "last_action": (29,),
                # Must match beyondmimic vault link_of_interests (18 bodies, incl. wrist roll/pitch).
                "link_pos": (18, 3),
                "link_rot": (18, 6),
            },
        },
        "num_actions": 29,
        "num_rewards": 1,
    }
    teacher_logdir = os.path.expanduser(PERCEPTIVE_VAULT_TEACHER_LOGDIR) if PERCEPTIVE_VAULT_TEACHER_LOGDIR else None


@configclass
class PolicyNormalizerCfg(InstinctRlNormalizerCfg):
    """PHP-style: empirical-normalize proprio only; pass through depth pixels."""

    class_name: str = "EmpiricalNormalizationWithPassthrough"
    passthrough_slices: list = [(0, _VAULT_DEPTH_FLAT_DIM)]


@configclass
class NormalizersCfg:
    policy: PolicyNormalizerCfg = PolicyNormalizerCfg()
    critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()


@configclass
class G1PerceptiveVaultVaeDistillPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: VaultVisionPolicyCfg = VaultVisionPolicyCfg()
    algorithm: AlgorithmCfg = AlgorithmCfg()
    normalizers: NormalizersCfg = NormalizersCfg()

    num_steps_per_env = 24
    max_iterations = 20000
    #: ⚠️ 2026-10-16：`1000` → `200`（与 teacher 侧 `instinct_rl_vault_cfg.py` 保持一致），
    #: 便于蒸馏阶段挑选中间 ckpt 做回放/对比。
    save_interval = 200
    log_interval = 10
    experiment_name = "g1_perceptive_vault_vae_distill"

    load_run = None

    def __post_init__(self):
        self.resume = self.load_run is not None
        self.run_name = "".join(
            [
                f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
            ]
        )
