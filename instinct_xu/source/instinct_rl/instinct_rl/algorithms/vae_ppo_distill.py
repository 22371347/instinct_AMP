"""VAE distillation with PPO task loss enabled."""

import importlib
from collections import namedtuple

import torch
import torch.distributions as dist
import torch.nn.functional as F

from instinct_rl.algorithms.ppo import PPO
from instinct_rl.algorithms.tppo import TPPO
from instinct_rl.storage.rollout_storage import ActionLabelRollout


class ActionLabelMaskRollout(ActionLabelRollout):
    class Transition(ActionLabelRollout.Transition):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.distillation_loss_mask = None

    MiniBatch = namedtuple(
        "MiniBatch",
        [
            *ActionLabelRollout.MiniBatch._fields,
            "distillation_loss_mask",
        ],
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation_loss_mask = torch.ones(
            self.num_transitions_per_env,
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
        )

    def expand_buffer_once(self):
        expand_size = super().expand_buffer_once()
        self.distillation_loss_mask = torch.cat(
            [
                self.distillation_loss_mask,
                torch.ones(expand_size, self.num_envs, 1, dtype=torch.float, device=self.device),
            ],
            dim=0,
        ).contiguous()
        return expand_size

    def add_transitions(self, transition: Transition):
        if transition.distillation_loss_mask is None:
            self.distillation_loss_mask[self.step].fill_(1.0)
        else:
            self.distillation_loss_mask[self.step] = transition.distillation_loss_mask
        return super().add_transitions(transition)

    def untie_buffer_loop(self):
        self.distillation_loss_mask = self.swap_from_cursor(self.distillation_loss_mask)
        return super().untie_buffer_loop()

    def get_minibatch_from_selection(self, T_select, B_select, padded_B_slice=None, prev_done_mask=None):
        minibatch = super().get_minibatch_from_selection(T_select, B_select, padded_B_slice, prev_done_mask)
        distillation_loss_mask_batch = self.distillation_loss_mask[T_select, B_select]
        return ActionLabelMaskRollout.MiniBatch(*minibatch, distillation_loss_mask_batch)


class VaePpoAlgoMixin:
    def __init__(
        self,
        *args,
        kl_loss_func="kl_divergence",
        kl_loss_coef=1.0,
        using_ppo=True,
        loss_curriculum_iterations=10000,
        distillation_loss_min_coef=0.1,
        ppo_loss_initial_coef=0.0,
        ppo_loss_final_coef=1.0,
        surrogate_loss_coef=1.0,
        # PHP: enable adaptive LR / KL control only after λ_PPO exceeds this value.
        adaptive_schedule_min_ppo_coef=0.1,
        teacher_obs_group="teacher",
        distillation_mask_obs_group="distill_mask",
        **kwargs,
    ):
        base_distillation_loss_coef = kwargs.get("distillation_loss_coef", 1.0)
        super().__init__(*args, using_ppo=using_ppo, **kwargs)
        self.using_ppo = using_ppo
        if ":" in kl_loss_func:
            kl_loss_func_module, kl_loss_func_name = kl_loss_func.split(":")
            self.kl_loss_func = getattr(importlib.import_module(kl_loss_func_module), kl_loss_func_name)
        else:
            self.kl_loss_func = getattr(dist.kl, kl_loss_func) if "kl" in kl_loss_func else getattr(F, kl_loss_func)
        self.kl_loss_coef = kl_loss_coef
        self.normal_dist = dist.Normal(0, 1)

        self.loss_curriculum_iterations = int(loss_curriculum_iterations)
        self.distillation_loss_min_coef = float(distillation_loss_min_coef)
        self.ppo_loss_initial_coef = float(ppo_loss_initial_coef)
        self.ppo_loss_final_coef = float(ppo_loss_final_coef)
        self.adaptive_schedule_min_ppo_coef = (
            None if adaptive_schedule_min_ppo_coef is None else float(adaptive_schedule_min_ppo_coef)
        )
        self._base_surrogate_loss_coef = float(surrogate_loss_coef)
        self._base_value_loss_coef = float(self.value_loss_coef)
        self._base_entropy_coef = float(self.entropy_coef)
        self._base_distillation_loss_coef = float(base_distillation_loss_coef)
        self._configured_schedule = self.schedule
        self._current_ppo_loss_coef = 0.0
        self.teacher_obs_group = teacher_obs_group
        self.distillation_mask_obs_group = distillation_mask_obs_group
        # Initialize curriculum coefficients before the first update.
        self._update_loss_curriculum()

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        self.transition = ActionLabelMaskRollout.Transition()
        obs_size = sum(int(torch.tensor(shape).prod().item()) for shape in obs_format["policy"].values())
        critic_obs_size = (
            sum(int(torch.tensor(shape).prod().item()) for shape in obs_format["critic"].values())
            if "critic" in obs_format
            else None
        )
        self.storage = ActionLabelMaskRollout(
            num_envs,
            num_transitions_per_env,
            [obs_size],
            [critic_obs_size],
            [num_actions],
            num_rewards=num_rewards,
            buffer_dilation_ratio=self.buffer_dilation_ratio,
            device=self.device,
        )

    def act_with_obs_pack(self, obs, critic_obs, obs_pack):
        return_ = PPO.act(self, obs, critic_obs)
        if obs_pack is None or self.teacher_obs_group not in obs_pack:
            raise KeyError(f"Teacher observation group '{self.teacher_obs_group}' is required for VaePpoDistill.")
        teacher_obs = obs_pack[self.teacher_obs_group].to(self.device)
        self.transition.action_labels = self.get_teacher_actions_from_obs_pack(teacher_obs, obs_pack)
        if self.distillation_mask_obs_group not in obs_pack:
            raise KeyError(
                f"Distillation mask observation group '{self.distillation_mask_obs_group}' is required for VaePpoDistill."
            )
        self.transition.distillation_loss_mask = obs_pack[self.distillation_mask_obs_group].to(self.device).reshape(
            obs.shape[0],
            1,
        )

        if not hasattr(self, "use_teacher_act_mask"):
            self.use_teacher_act_mask = (
                torch.rand(obs.shape[0], device=self.device) < self.teacher_act_prob(self.current_learning_iteration)
            )
        return_[self.use_teacher_act_mask] = self.transition.action_labels[self.use_teacher_act_mask]
        return return_

    def get_teacher_actions_from_obs_pack(self, teacher_obs, obs_pack):
        del obs_pack
        return self.get_teacher_actions(teacher_obs, None)

    def _update_loss_curriculum(self) -> dict[str, torch.Tensor]:
        """PHP Eq.(3): L = λ_PPO L_PPO + λ_D L_D with λ_PPO + λ_D = 1.

        λ_D(k) = max(λ_D_min, 1 - k / K_half), λ_PPO = 1 - λ_D.
        """
        iteration = float(getattr(self, "current_learning_iteration", 0))
        if self.loss_curriculum_iterations <= 0:
            progress = 1.0
        else:
            progress = min(1.0, max(0.0, iteration / self.loss_curriculum_iterations))

        # Curriculum weights (sum to 1). Base distill/PPO scales are applied separately.
        distill_weight = max(self.distillation_loss_min_coef, 1.0 - progress)
        ppo_weight = 1.0 - distill_weight

        self.surrogate_loss_coef = self._base_surrogate_loss_coef * ppo_weight
        self.value_loss_coef = self._base_value_loss_coef * ppo_weight
        self.entropy_coef = self._base_entropy_coef * ppo_weight
        self.distillation_loss_coef = self._base_distillation_loss_coef * distill_weight
        self._current_ppo_loss_coef = ppo_weight

        # PHP: keep LR fixed while DAgger dominates; enable adaptive KL LR once λ_PPO > threshold.
        if self.adaptive_schedule_min_ppo_coef is None:
            self.schedule = self._configured_schedule
        elif ppo_weight >= self.adaptive_schedule_min_ppo_coef:
            self.schedule = self._configured_schedule
        else:
            self.schedule = "fixed"

        return {
            "loss_curriculum_progress": torch.tensor(progress, device=self.device),
            "ppo_loss_coef": torch.tensor(ppo_weight, device=self.device),
            "distillation_loss_coef": torch.tensor(self.distillation_loss_coef, device=self.device),
        }

    def compute_losses(self, minibatch):
        # Refresh curriculum (and LR schedule gate) before PPO adaptive-KL logic runs.
        self._update_loss_curriculum()
        losses, inter_vars, stats = super().compute_losses(minibatch)
        if hasattr(minibatch, "distillation_loss_mask"):
            distillation_loss_mask = minibatch.distillation_loss_mask.reshape(-1).to(
                dtype=self.actor_critic.action_mean.dtype,
                device=self.device,
            )
            dist_loss = self.compute_distill_loss(self.actor_critic.action_mean, minibatch.action_labels)
            losses["distillation_loss"] = (dist_loss * distillation_loss_mask).sum() / distillation_loss_mask.sum().clamp_min(1.0)
            stats["distillation_loss_mask_ratio"] = distillation_loss_mask.detach().mean()
        # TPPO may overwrite distillation_loss_coef; restore PHP-scaled coefficient for the backward pass.
        stats.update(self._update_loss_curriculum())

        kl_loss = torch.zeros_like(minibatch.obs[..., 0])
        if hasattr(self.actor_critic, "latent_distribution") and self.actor_critic.latent_distribution is not None:
            kl_loss += self.kl_loss_func(
                self.actor_critic.latent_distribution,
                self.normal_dist,
            ).sum(dim=-1)
        losses["kl_loss"] = kl_loss.mean()

        return losses, inter_vars, stats


class VaePpoDistill(VaePpoAlgoMixin, TPPO):
    pass


class MultiTeacherVaePpoDistill(VaePpoAlgoMixin, TPPO):
    """VAE/PPO distillation with per-environment routing between two teachers."""

    def __init__(
        self,
        *args,
        teacher_logdirs,
        teacher_routing_obs_group="teacher_route",
        allow_missing_teacher_checkpoints=False,
        **kwargs,
    ):
        if len(teacher_logdirs) != 2:
            raise ValueError(f"MultiTeacherVaePpoDistill requires exactly two teacher logdirs, got {len(teacher_logdirs)}")
        if any(logdir is None for logdir in teacher_logdirs) and not allow_missing_teacher_checkpoints:
            raise ValueError("Both teacher logdirs are required for MultiTeacherVaePpoDistill.")

        self.teacher_routing_obs_group = teacher_routing_obs_group
        self.teacher_logdirs = list(teacher_logdirs)
        kwargs.pop("teacher_logdir", None)
        super().__init__(*args, teacher_logdir=self.teacher_logdirs[0], **kwargs)

        second_teacher = self.build_teacher_policy()
        if self.teacher_logdirs[1] is None:
            second_teacher.to(self.device)
            second_teacher.eval()
            second_normalizer = None
            print(
                "MultiTeacherVaePpoDistill warning: No checkpoint loaded for teacher 1. "
                "This is only valid for student inference."
            )
        else:
            second_normalizer = self.load_teacher_policy_from_logdir(second_teacher, self.teacher_logdirs[1])
        self.teacher_actor_critics = [self.teacher_actor_critic, second_teacher]
        self.teacher_policy_normalizers = [self.teacher_policy_normalizer, second_normalizer]

    def _get_routed_teacher_actions(self, teacher_obs, route):
        route = route.to(device=self.device, dtype=torch.long).reshape(-1)
        if route.shape[0] != teacher_obs.shape[0]:
            raise ValueError(
                f"Teacher route batch ({route.shape[0]}) does not match teacher observations ({teacher_obs.shape[0]})."
            )
        if ((route < 0) | (route >= len(self.teacher_actor_critics))).any():
            raise ValueError("Teacher route values must be 0 or 1.")

        action_labels = torch.empty(
            teacher_obs.shape[0],
            self.teacher_policy_cfg_dict["num_actions"],
            dtype=teacher_obs.dtype,
            device=self.device,
        )
        for teacher_idx, (teacher, normalizer) in enumerate(
            zip(self.teacher_actor_critics, self.teacher_policy_normalizers)
        ):
            mask = route == teacher_idx
            if not mask.any():
                continue
            selected_obs = teacher_obs[mask]
            if normalizer is not None:
                selected_obs = normalizer(selected_obs)
            if self.action_labels_from_sample:
                selected_actions = teacher.act(selected_obs)
            else:
                selected_actions = teacher.act_inference(selected_obs)
            action_labels[mask] = selected_actions.detach()
        return action_labels

    def get_teacher_actions_from_obs_pack(self, teacher_obs, obs_pack):
        if any(logdir is None for logdir in self.teacher_logdirs):
            raise RuntimeError(
                "Both teacher checkpoints are required when requesting multi-teacher actions."
            )
        if self.teacher_routing_obs_group not in obs_pack:
            raise KeyError(
                f"Teacher routing observation group '{self.teacher_routing_obs_group}' is required "
                "for MultiTeacherVaePpoDistill."
            )
        return self._get_routed_teacher_actions(teacher_obs, obs_pack[self.teacher_routing_obs_group])

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        result = super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)
        self.teacher_actor_critics[1].reset(dones)
        return result
