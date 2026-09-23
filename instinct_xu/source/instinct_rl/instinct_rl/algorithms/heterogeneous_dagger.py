"""Route-balanced action distillation from teachers with independent observations."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from instinct_rl import modules
from instinct_rl.algorithms.ppo import PPO
from instinct_rl.storage.rollout_storage import RoutedActionLabelRollout
from instinct_rl.utils.utils import get_subobs_size


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    route_id: int
    checkpoint: str
    obs_group: str
    expected_obs_dim: int
    expected_action_dim: int
    actor_cfg: dict[str, Any]
    loss_weight: float = 1.0
    action_adapter: str = "identity"

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | TeacherSpec) -> TeacherSpec:
        spec = config if isinstance(config, cls) else cls(**config)
        if not spec.name or not spec.obs_group:
            raise ValueError("Teacher name and observation group must be nonempty.")
        if (
            spec.route_id < 0
            or spec.expected_obs_dim <= 0
            or spec.expected_action_dim <= 0
        ):
            raise ValueError(f"Invalid route or dimensions for teacher '{spec.name}'.")
        if spec.loss_weight <= 0:
            raise ValueError(f"Teacher '{spec.name}' must have a positive loss weight.")
        if spec.action_adapter != "identity":
            raise ValueError(
                f"Unknown action adapter '{spec.action_adapter}' for teacher '{spec.name}'."
            )
        if not spec.checkpoint:
            raise ValueError(
                f"Teacher '{spec.name}' needs an exact checkpoint file path."
            )
        return spec


class TeacherAdapter:
    """Load one frozen teacher and its own policy normalizer from an exact file."""

    def __init__(self, spec: TeacherSpec, device: str):
        self.spec = spec
        self.device = device
        checkpoint = Path(spec.checkpoint).expanduser()
        if (
            not checkpoint.is_file()
            or not checkpoint.name.startswith("model_")
            or checkpoint.suffix != ".pt"
        ):
            raise FileNotFoundError(
                f"Teacher '{spec.name}' requires an existing model_*.pt file: {checkpoint}"
            )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "model_state_dict" not in state:
            raise KeyError(f"Teacher checkpoint has no model_state_dict: {checkpoint}")

        actor_cfg = dict(spec.actor_cfg)
        class_name = actor_cfg.pop("class_name", "ActorCritic")
        obs_format = actor_cfg.pop("obs_format", None)
        if obs_format is None:
            # The current Loco and Vault experts use ActorCritic. Their critic is
            # unused for labeling, but its input width is needed for strict load.
            critic_weight = state["model_state_dict"].get("critic.0.weight")
            critic_dim = (
                int(critic_weight.shape[1])
                if critic_weight is not None
                else spec.expected_obs_dim
            )
            obs_format = {
                "policy": {"teacher_observation": (spec.expected_obs_dim,)},
                "critic": {"teacher_critic_observation": (critic_dim,)},
            }
        if get_subobs_size(obs_format["policy"]) != spec.expected_obs_dim:
            raise ValueError(
                f"Configured observation width differs for teacher '{spec.name}'."
            )
        actor_cfg.pop("num_actions", None)
        num_rewards = int(actor_cfg.pop("num_rewards", 1))
        self.actor = modules.build_actor_critic(
            class_name, actor_cfg, obs_format, spec.expected_action_dim, num_rewards
        )
        self.actor.load_state_dict(state["model_state_dict"], strict=True)
        self.actor.to(device).eval().requires_grad_(False)

        self.normalizer = None
        normalizer_state = state.get("policy_normalizer_state_dict")
        if normalizer_state is None:
            warnings.warn(
                f"Teacher '{spec.name}' checkpoint has no policy normalizer; using identity.",
                stacklevel=2,
            )
        else:
            agent_yaml = checkpoint.parent / "params" / "agent.yaml"
            if not agent_yaml.is_file():
                raise FileNotFoundError(
                    f"Cannot build teacher normalizer without {agent_yaml}"
                )
            with agent_yaml.open(encoding="utf-8") as stream:
                agent_cfg = yaml.safe_load(stream)
            normalizer_cfg = dict(agent_cfg["normalizers"]["policy"])
            self.normalizer = modules.build_normalizer(
                input_shape=(spec.expected_obs_dim,),
                normalizer_class_name=normalizer_cfg.pop("class_name"),
                normalizer_kwargs=normalizer_cfg,
            )
            if self.normalizer is None:
                raise ValueError(
                    f"Teacher '{spec.name}' has normalizer weights but no normalizer class."
                )
            self.normalizer.load_state_dict(normalizer_state, strict=True)
            self.normalizer.to(device).eval().requires_grad_(False)

        print(
            f"[teacher] name={spec.name} checkpoint={checkpoint} obs={spec.expected_obs_dim} "
            f"action={spec.expected_action_dim} actor={class_name} "
            f"hidden={actor_cfg.get('actor_hidden_dims')} normalizer={'loaded' if self.normalizer else 'identity'}"
        )

    @torch.no_grad()
    def act(self, raw_obs: torch.Tensor) -> torch.Tensor:
        if raw_obs.ndim != 2 or raw_obs.shape[1] != self.spec.expected_obs_dim:
            raise ValueError(
                f"Teacher '{self.spec.name}' expected [B,{self.spec.expected_obs_dim}], got {tuple(raw_obs.shape)}"
            )
        obs = raw_obs.to(self.device)
        if self.normalizer is not None:
            obs = self.normalizer(obs)
        actions = self.actor.act_inference(obs)
        if actions.shape != (raw_obs.shape[0], self.spec.expected_action_dim):
            raise ValueError(
                f"Teacher '{self.spec.name}' returned action shape {tuple(actions.shape)}"
            )
        return (
            actions.detach()
        )  # identity action adapter for the verified 29-D contract


class HeterogeneousMultiExpertDagger(PPO):
    """Pure action DAgger using one student and any number of routed teachers."""

    def __init__(
        self,
        actor_critic,
        *,
        teachers: list[Mapping[str, Any] | TeacherSpec],
        student_obs_dim: int = 623,
        student_obs_schema: list[tuple[str, int]] | None = None,
        route_obs_group: str = "teacher_route",
        valid_mask_obs_group: str = "teacher_valid_mask",
        rollout_mode: str = "student",
        teacher_rollin_start_prob: float = 1.0,
        teacher_rollin_end_prob: float = 0.0,
        teacher_rollin_decay_iterations: int = 1000,
        student_init_checkpoint: str | None = None,
        policy_normalizer_mode: str = "fresh_trainable",
        buffer_dilation_ratio: float = 1.0,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(actor_critic, device=device, **kwargs)
        self.student_obs_dim = int(student_obs_dim)
        if not student_obs_schema:
            raise ValueError("V0 requires an explicit, complete student_obs_schema.")
        self.student_obs_schema = student_obs_schema
        self.route_obs_group = route_obs_group
        self.valid_mask_obs_group = valid_mask_obs_group
        self.specs = [TeacherSpec.from_config(config) for config in teachers]
        if not self.specs or len({s.name for s in self.specs}) != len(self.specs):
            raise ValueError(
                "Teacher registry must contain at least one teacher with unique names."
            )
        if len({s.route_id for s in self.specs}) != len(self.specs):
            raise ValueError("Teacher route ids must be unique.")
        if len({s.obs_group for s in self.specs}) != len(self.specs):
            raise ValueError("Teacher observation groups must be unique.")
        forbidden_policy_terms = {
            route_obs_group,
            valid_mask_obs_group,
            *(spec.obs_group for spec in self.specs),
            "skill_id",
            "reference",
            "vault_phase",
            "global_position",
            "distance_to_box",
        }
        leaked_terms = {name for name, _ in student_obs_schema} & forbidden_policy_terms
        if leaked_terms:
            raise ValueError(f"Forbidden student policy observation terms: {sorted(leaked_terms)}")
        if rollout_mode not in {"student", "teacher_rollin"}:
            raise ValueError("rollout_mode must be 'student' or 'teacher_rollin'.")
        if policy_normalizer_mode not in {
            "fresh_trainable",
            "loco_frozen",
            "loco_trainable",
        }:
            raise ValueError("Unknown policy_normalizer_mode.")
        if policy_normalizer_mode.startswith("loco_") and not student_init_checkpoint:
            raise ValueError("Loco normalizer modes require student_init_checkpoint.")
        if (
            not 0.0 <= teacher_rollin_start_prob <= 1.0
            or not 0.0 <= teacher_rollin_end_prob <= 1.0
        ):
            raise ValueError("Teacher roll-in probabilities must be in [0, 1].")
        if teacher_rollin_decay_iterations < 0 or buffer_dilation_ratio < 1.0:
            raise ValueError("Invalid roll-in duration or buffer dilation ratio.")

        self.rollout_mode = rollout_mode
        self.teacher_rollin_start_prob = float(teacher_rollin_start_prob)
        self.teacher_rollin_end_prob = float(teacher_rollin_end_prob)
        self.teacher_rollin_decay_iterations = int(teacher_rollin_decay_iterations)
        self.student_init_checkpoint = student_init_checkpoint
        self.policy_normalizer_mode = policy_normalizer_mode
        self.buffer_dilation_ratio = float(buffer_dilation_ratio)
        self.teacher_adapters = [TeacherAdapter(spec, device) for spec in self.specs]
        self._student_initial_state = None
        if student_init_checkpoint:
            checkpoint = Path(student_init_checkpoint).expanduser()
            if not checkpoint.is_file() or checkpoint.suffix != ".pt":
                raise FileNotFoundError(
                    f"Student initialization checkpoint not found: {checkpoint}"
                )
            self._student_initial_state = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            actor_state = {
                key.removeprefix("actor."): value
                for key, value in self._student_initial_state[
                    "model_state_dict"
                ].items()
                if key.startswith("actor.")
            }
            if not actor_state or not hasattr(self.actor_critic, "actor"):
                raise ValueError(
                    "Student initialization requires a compatible ActorCritic actor."
                )
            self.actor_critic.actor.load_state_dict(actor_state, strict=True)
            print(f"[student] initialized actor only from {checkpoint}")

    def configure_policy_normalizer(self, normalizer, obs_format: dict):
        """Called by the runner once, after it has built its policy normalizer."""
        self._validate_student_obs_format(obs_format)
        if self.policy_normalizer_mode == "fresh_trainable":
            return
        if normalizer is None:
            raise ValueError(
                "Loco normalizer mode requires a runner policy normalizer."
            )
        state = self._student_initial_state.get("policy_normalizer_state_dict")
        if state is None:
            raise KeyError(
                "Student initialization checkpoint has no policy normalizer state."
            )
        normalizer.load_state_dict(state, strict=True)
        if self.policy_normalizer_mode == "loco_frozen":
            if not hasattr(normalizer, "until"):
                raise TypeError(
                    "Frozen mode requires a normalizer with an 'until' update limit."
                )
            normalizer.until = (
                0  # Runner may call train(); updates still remain disabled.
            )
        print(f"[student] policy normalizer mode={self.policy_normalizer_mode}")

    def _validate_student_obs_format(self, obs_format: dict):
        policy_format = obs_format["policy"]
        if get_subobs_size(policy_format) != self.student_obs_dim:
            raise ValueError(
                f"Student policy must have {self.student_obs_dim} observations."
            )
        actual = [
            (name, int(torch.tensor(shape).prod().item()))
            for name, shape in policy_format.items()
        ]
        expected = [(str(name), int(dim)) for name, dim in self.student_obs_schema]
        if actual != expected:
            raise ValueError(
                f"Student observation order/dimensions differ: {actual} != {expected}"
            )

    def init_storage(
        self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1
    ):
        self._validate_student_obs_format(obs_format)
        if (
            self.route_obs_group not in obs_format
            or self.valid_mask_obs_group not in obs_format
        ):
            raise KeyError(
                "Teacher route and valid-mask observation groups are required."
            )
        if get_subobs_size(obs_format[self.route_obs_group]) != 1:
            raise ValueError("Teacher route observation must have width 1.")
        if get_subobs_size(obs_format[self.valid_mask_obs_group]) != 1:
            raise ValueError("Teacher valid-mask observation must have width 1.")
        for spec in self.specs:
            if spec.obs_group not in obs_format:
                raise KeyError(f"Missing teacher observation group '{spec.obs_group}'.")
            if get_subobs_size(obs_format[spec.obs_group]) != spec.expected_obs_dim:
                raise ValueError(
                    f"Observation dimension mismatch for teacher '{spec.name}'."
                )
            if spec.expected_action_dim != num_actions:
                raise ValueError(
                    f"Action dimension mismatch for teacher '{spec.name}'."
                )
        if getattr(self.actor_critic, "is_recurrent", False):
            raise NotImplementedError(
                "V0 route-balanced DAgger currently requires a feed-forward student."
            )
        self.transition = RoutedActionLabelRollout.Transition()
        critic_dim = (
            get_subobs_size(obs_format["critic"]) if "critic" in obs_format else None
        )
        self.storage = RoutedActionLabelRollout(
            num_envs,
            num_transitions_per_env,
            [self.student_obs_dim],
            [critic_dim],
            [num_actions],
            num_rewards=num_rewards,
            buffer_dilation_ratio=self.buffer_dilation_ratio,
            device=self.device,
        )

    def _teacher_rollin_prob(self) -> float:
        if self.rollout_mode == "student":
            return 0.0
        if self.teacher_rollin_decay_iterations == 0:
            return self.teacher_rollin_end_prob
        progress = min(
            1.0, self.current_learning_iteration / self.teacher_rollin_decay_iterations
        )
        return self.teacher_rollin_start_prob + progress * (
            self.teacher_rollin_end_prob - self.teacher_rollin_start_prob
        )

    def _route_and_valid(
        self, obs_pack: dict, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        route = obs_pack[self.route_obs_group].to(self.device).reshape(-1)
        valid = obs_pack[self.valid_mask_obs_group].to(self.device).reshape(-1)
        if route.numel() != batch_size or valid.numel() != batch_size:
            raise ValueError(
                "Teacher route and validity mask must have one value per environment."
            )
        if not torch.isfinite(route).all() or not torch.isfinite(valid).all():
            raise ValueError("Teacher route and validity mask must be finite.")
        if not torch.equal(route, route.round()):
            raise ValueError("Teacher route ids must be integers.")
        route = route.long()
        if not torch.isin(
            route, torch.tensor([s.route_id for s in self.specs], device=self.device)
        ).all():
            raise ValueError("Teacher route contains an unregistered id.")
        if not ((valid == 0) | (valid == 1)).all():
            raise ValueError("Teacher valid mask must contain only 0 or 1.")
        return route, valid.bool()

    def act_with_obs_pack(self, obs, critic_obs, obs_pack):
        if obs_pack is None:
            raise KeyError(
                "HeterogeneousMultiExpertDagger requires the complete observation pack."
            )
        self.actor_critic.update_distribution(obs)
        student_action = self.actor_critic.action_mean.detach()
        if student_action.shape != (obs.shape[0], self.storage.actions_shape[0]):
            raise ValueError(f"Student returned action shape {tuple(student_action.shape)}")
        # PPO storage requires these fields, although imitation uses only policy
        # observations, action labels, route, and validity.
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.values = obs.new_zeros(obs.shape[0], self.storage.num_rewards)
        self.transition.action_mean = student_action
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.actions_log_prob = obs.new_zeros(obs.shape[0])
        route, valid = self._route_and_valid(obs_pack, obs.shape[0])
        labels = torch.zeros_like(student_action)
        for spec, teacher in zip(self.specs, self.teacher_adapters):
            selected = (route == spec.route_id) & valid
            if not selected.any():
                continue
            teacher_obs = obs_pack[spec.obs_group].to(self.device)
            if teacher_obs.shape != (obs.shape[0], spec.expected_obs_dim):
                raise ValueError(
                    f"Bad shape for teacher group '{spec.obs_group}': {tuple(teacher_obs.shape)}"
                )
            labels[selected] = teacher.act(teacher_obs[selected])

        rollin_prob = self._teacher_rollin_prob()
        if rollin_prob == 0.0:
            rollin = torch.zeros_like(valid)
        elif rollin_prob == 1.0:
            rollin = valid
        else:
            rollin = (torch.rand(obs.shape[0], device=self.device) < rollin_prob) & valid
        executed = torch.where(rollin.unsqueeze(-1), labels, student_action)
        self.transition.actions = executed.detach()
        self.transition.action_labels = labels.detach()
        self.transition.teacher_route = route.unsqueeze(-1).detach()
        self.transition.teacher_valid_mask = valid.float().unsqueeze(-1).detach()
        return executed

    def compute_returns(self, last_critic_obs):
        del last_critic_obs  # V0 optimizes action imitation only.

    def compute_losses(self, minibatch):
        prediction = self.actor_critic.act_inference(
            minibatch.obs.reshape(-1, self.student_obs_dim)
        )
        labels = minibatch.action_labels.reshape(-1, prediction.shape[-1])
        per_sample = F.mse_loss(prediction, labels, reduction="none").mean(dim=-1)
        route = minibatch.teacher_route.reshape(-1)
        valid = minibatch.teacher_valid_mask.reshape(-1).bool()
        weighted_loss = prediction.sum() * 0.0
        active_weight = 0.0
        stats = {}
        for spec in self.specs:
            selected = (route == spec.route_id) & valid
            if selected.any():
                loss_k = per_sample[selected].mean()
                weighted_loss = weighted_loss + spec.loss_weight * loss_k
                active_weight += spec.loss_weight
                stats[f"Distill/{spec.name}_loss"] = loss_k.detach()
                stats[f"Distill/{spec.name}_action_mse"] = loss_k.detach()
                stats[f"Distill/{spec.name}_action_l1"] = (
                    (prediction[selected] - labels[selected]).abs().mean().detach()
                )
            else:
                stats[f"Distill/{spec.name}_loss"] = per_sample.new_zeros(())
                stats[f"Distill/{spec.name}_action_mse"] = per_sample.new_zeros(())
                stats[f"Distill/{spec.name}_action_l1"] = per_sample.new_zeros(())
        total_loss = weighted_loss / active_weight if active_weight else weighted_loss
        stats["Distill/total_loss"] = total_loss.detach()
        return {"distillation_loss": total_loss}, {}, stats

    def update(self, current_learning_iteration):
        counts = {}
        route = self.storage.teacher_route.reshape(-1)
        valid = self.storage.teacher_valid_mask.reshape(-1).bool()
        for spec in self.specs:
            selected = route == spec.route_id
            counts[f"Distill/{spec.name}_valid_samples"] = (
                (selected & valid).sum().float()
            )
            counts[f"Distill/{spec.name}_valid_ratio"] = (
                selected & valid
            ).sum().float() / selected.sum().clamp_min(1)
        losses, stats = super().update(current_learning_iteration)
        stats.update(counts)
        self.current_learning_iteration = current_learning_iteration + 1
        return losses, stats
