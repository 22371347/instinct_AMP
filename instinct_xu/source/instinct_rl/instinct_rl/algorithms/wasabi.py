import importlib
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

import instinct_rl.modules as instinct_modules
from instinct_rl.algorithms.ppo import PPO
from instinct_rl.storage.amp_storage import AmpStorage
from instinct_rl.utils.buffer import buffer_func
from instinct_rl.utils.utils import get_obs_slice, get_subobs_size


class WasabiAlgoMixin:
    """A plugin algorithm for [WASABI](https://github.com/martius-lab/wasabi), which also includes traditional AMP algorithm
    (Set `discriminator_loss_func` to "BCEWithLogitsLoss" for AMP).
    """

    def __init__(
        self,
        *args,
        actor_state_key="amp_policy",  # the key of getting policy's state sequence
        reference_state_key="amp_reference",  # the key of getting expert's reference sequence
        discriminator_class_name="Discriminator",
        discriminator_kwargs={},
        discriminator_optimizer_class_name="AdamW",
        discriminator_optimizer_kwargs={},
        discriminator_reward_coef=1.0,
        discriminator_reward_type: Literal[
            "log", "quad", "wasserstein"
        ] = "log",  # check more on `compute_discriminator_reward`
        discriminator_loss_func: Literal[
            "WassersteinLoss", "BCEWithLogitsLoss", "MSELoss"
        ] = "BCEWithLogitsLoss",  # by default, lead to AMP
        discriminator_loss_coef=1.0,
        discriminator_gradient_penalty_coef=10.0,
        discriminator_weight_decay_coef=0.0,
        discriminator_logit_weight_decay_coef=0.0,  # loss for last layer weight
        discriminator_gradient_torlerance=0.0,  # If the computed gradient is smaller than this value, the gradient will not be penalized.
        discriminator_backbone_gradient_only=False,  # If True, the discriminator must support encoders and backbone_run function.
        #: ⭐⭐ V30（对齐 AMP_mjlab `compute_grad_pen(*sample_amp_expert, lambda_=10)`）：
        #: `True` ⇒ GP **只在 expert/reference 侧**求梯度（旧口径 = actor+expert 合并，见 V29）。
        discriminator_gp_expert_only=False,
        #: ⭐⭐ V30（对齐 AMP_mjlab：wd 在**优化器**上，trunk 1e-3 / head 1e-2）：
        #: 判别器 **head（最后一层）** 的 AdamW `weight_decay`；两个都为 0 时零行为变化。
        discriminator_optimizer_head_weight_decay=0.0,
        #: 判别器 **trunk（其余层）** 的 AdamW `weight_decay`。
        discriminator_optimizer_trunk_weight_decay=0.0,
        #: ⭐⭐ V30：专家批来自**离线运动数据集**（mjlab `AMPLoader` 口径）的文件路径。
        #: 形如 `/root/amp_expert_states.pt`（键 `states_pair`，形状 `(N, 390)`）。
        #: `None` = 用活体影子机器人（旧行为）。
        amp_expert_buffer_path=None,
        amp_normalizer_expert_only: bool = True,
        amp_grad_pen_space: str = "raw",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.actor_state_key = actor_state_key
        self.reference_state_key = reference_state_key
        self.discriminator_class_name = discriminator_class_name
        self.discriminator_kwargs = discriminator_kwargs
        self.discriminator_optimizer_class_name = discriminator_optimizer_class_name
        self.discriminator_optimizer_kwargs = discriminator_optimizer_kwargs
        self.discriminator_reward_coef = discriminator_reward_coef
        self.discriminator_reward_type = discriminator_reward_type
        self.discriminator_loss_func = discriminator_loss_func
        self.discriminator_loss_coef = discriminator_loss_coef
        self.discriminator_gradient_penalty_coef = discriminator_gradient_penalty_coef
        self.discriminator_weight_decay_coef = discriminator_weight_decay_coef
        self.discriminator_logit_weight_decay_coef = discriminator_logit_weight_decay_coef
        self.discriminator_gradient_torlerance = discriminator_gradient_torlerance
        self.discriminator_backbone_gradient_only = discriminator_backbone_gradient_only
        # ⭐⭐ V30（mjlab 判别器配方）
        self.discriminator_gp_expert_only = bool(discriminator_gp_expert_only)
        self.discriminator_optimizer_head_weight_decay = float(discriminator_optimizer_head_weight_decay)
        self.discriminator_optimizer_trunk_weight_decay = float(discriminator_optimizer_trunk_weight_decay)
        self.amp_expert_buffer_path = amp_expert_buffer_path
        #: ⭐ 判别器输入白化（AMP_mjlab 的 `amp_normalizer` 等价物，2026-10-03）
        #: `True` = 统计量**只用 reference/expert 侧**累积（actor 侧前向不更新）。
        #: 否则 actor 与 reference 都会流经 normalizer ⇒ 统计量被策略分布污染
        #: （策略越跑偏、统计量越被带偏 ⇒ 反而更难分辨）。
        self.amp_normalizer_expert_only = bool(amp_normalizer_expert_only)
        #: ⭐ GP 的作用空间：`"raw"`（历史行为）| `"normalized"`（白化方案）。
        #: **白化开启时必须用 `"normalized"`**：在原始空间求 `‖∂d/∂x‖` 时
        #: `∂d/∂x = (∂d/∂x̃)/σ`，而 σ 在近常数维上→0 ⇒ `‖∇d‖` 从 ~5e-3 爆到 10²~10³，
        #: GP 项（平方）直接支配损失。实测见 `RESULTS_amp_vs_mjlab_params.md` §F。
        if amp_grad_pen_space not in ("raw", "normalized"):
            raise ValueError(f"amp_grad_pen_space 只支持 'raw' / 'normalized'，收到 {amp_grad_pen_space!r}")
        self.amp_grad_pen_space = str(amp_grad_pen_space)

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        super().init_storage(num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards)
        # ⭐ V33：把判别器输入的**分项布局**记下来（`{项名: 形状}`，顺序 = 拼接顺序）
        #   ⇒ `_disc_grad_index` 直接读它，不再硬编码“帧优先/项优先”或 195/198 这堆常量。
        self._amp_obs_segments = obs_format.get(self.actor_state_key)
        # build discriminator
        if ":" in self.discriminator_class_name:
            DiscriminatorClass = getattr(
                importlib.import_module(self.discriminator_class_name.split(":")[0]),
                self.discriminator_class_name.split(":")[1],
            )
        else:
            DiscriminatorClass = getattr(instinct_modules, self.discriminator_class_name)
        self.discriminator = DiscriminatorClass(
            input_segment=obs_format[self.actor_state_key],
            **self.discriminator_kwargs,
        ).to(self.device)
        if not "lr" in self.discriminator_optimizer_kwargs:
            self.discriminator_optimizer_kwargs = dict(lr=self.learning_rate, **self.discriminator_optimizer_kwargs)
        # ⭐⭐ V30：判别器优化器的**分组 weight decay**（对齐 AMP_mjlab：trunk 1e-3 / head 1e-2）。
        #    mjlab 的 wd 是**优化器级**的（AdamW `weight_decay`），不是 loss 项；
        #    所以我们把 loss 形式的 `discriminator_weight_decay_coef` 设为 0（V30 cfg）。
        #    两个系数都为 0 时**完全退化为旧行为**（单一 param group，零行为变化）。
        _wd_head = self.discriminator_optimizer_head_weight_decay
        _wd_trunk = self.discriminator_optimizer_trunk_weight_decay
        if _wd_head > 0.0 or _wd_trunk > 0.0:
            # head = 最后一层（判别器的 logit 层）。镜像 `Discriminator.logit_layer_weights()`
            # 的定位逻辑（含 MlpModel / nn.Sequential / DDP 三重包装）。
            _m = self.discriminator.model
            if isinstance(_m, torch.nn.parallel.DistributedDataParallel):
                _m = _m.module
            if isinstance(_m, torch.nn.Sequential):
                _m = _m[0]
            if hasattr(_m, "model"):
                _m = _m.model
            head_params = list(_m[-1].parameters())
            head_ids = {id(p) for p in head_params}
            trunk_params = [p for p in self.discriminator.parameters() if id(p) not in head_ids]
            self.discriminator_optimizer = getattr(optim, self.discriminator_optimizer_class_name)(
                [
                    {"params": trunk_params, "weight_decay": _wd_trunk},
                    {"params": head_params, "weight_decay": _wd_head},
                ],
                **self.discriminator_optimizer_kwargs,
            )
        else:
            self.discriminator_optimizer = getattr(optim, self.discriminator_optimizer_class_name)(
                self.discriminator.parameters(),
                **self.discriminator_optimizer_kwargs,
            )

        self.amp_transition = AmpStorage.Transition()
        reference_state_size = get_subobs_size(obs_format[self.actor_state_key])
        actor_state_size = get_subobs_size(obs_format[self.reference_state_key])
        assert actor_state_size == reference_state_size, "The shape of robot state and reference state must be the same"
        self.amp_storage = AmpStorage(
            num_envs, num_transitions_per_env, [actor_state_size], [reference_state_size], device=self.device
        )
        # ⭐⭐ V30：接上**离线专家数据集**（mjlab 口径：专家批 = 从整个运动数据集 i.i.d. 抽帧）
        if self.amp_expert_buffer_path:
            _buf = torch.load(self.amp_expert_buffer_path, map_location=self.device)
            _buf = _buf["states_pair"] if isinstance(_buf, dict) else _buf
            _buf = _buf.to(device=self.device, dtype=torch.float32)
            assert _buf.shape[1] == self.amp_storage.reference_states.shape[-1], (
                f"专家 buffer 维度 {_buf.shape[1]} 与 AMP 状态维度 "
                f"{self.amp_storage.reference_states.shape[-1]} 不符"
            )
            self.amp_storage.expert_buffer = _buf
            print(
                f"[wasabi] 专家批来自离线数据集：{self.amp_expert_buffer_path} "
                f"shape={tuple(_buf.shape)} ⇒ 参考批从全数据集 i.i.d. 抽样（mjlab 口径）",
                flush=True,
            )

    def distributed_data_parallel(self):
        super().distributed_data_parallel()
        for param in self.discriminator.parameters():
            dist.broadcast(param.data, src=0)
        if self.discriminator.normalizer is not None:
            self.discriminator.normalizer.init_broadcast()

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        if not (self.actor_state_key in infos["observations"] and self.reference_state_key in infos["observations"]):
            raise ValueError(
                "The key of trajectory observations ({}) or reference observations ({}) is not found in the observation"
                " dictionary".format(self.actor_state_key, self.reference_state_key)
            )

        actor_state = infos["observations"][self.actor_state_key]
        reference_state = infos["observations"][self.reference_state_key]
        self.amp_transition.actor_states = actor_state
        self.amp_transition.reference_states = reference_state
        if self.discriminator.is_recurrent:
            self.amp_transition.hidden_states = self.discriminator.get_hidden_states()
        self.amp_transition.dones = dones
        self.amp_storage.add_transitions(self.amp_transition)
        self.amp_transition.clear()

        # do not call compute_auxilary_reward here, because it is called in the baseclass function
        super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)

    @torch.no_grad()
    def compute_auxiliary_reward(self, obs_pack: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        super_auxilary_reward = super().compute_auxiliary_reward(obs_pack)
        actor_state = obs_pack[self.actor_state_key]
        # ⭐⭐ V32 修正（2026-10-06）：rollout 期的 actor 前向**永远不更新**白化统计量。
        #   * 这里是**策略数据**，且 rollout 期每个 env step 都跑（24×4096 样本/迭代，
        #     与判别器训练 minibatch 的量级相当）⇒ 让它进统计量会把 μ/σ 向策略分布带偏；
        #   * AMP_mjlab 的口径是：**只在判别器训练路径**更新统计量
        #     （`amp_ppo.py:498-499`：`update(policy_state)` + `update(expert_state)` ≈ 两侧 50/50），
        #     奖励预测路径用 `normalize_torch`（**只看不改**）⇒ 这里也只看不改。
        #   * 旧写法 `update_normalizer=not self.amp_normalizer_expert_only` 在
        #     `expert_only=True`（A 臂 / V31 及之前全部版本）下取值就是 False
        #     ⇒ 本修正是**完全 no-op**，只影响"两侧白化"（B/C 臂），使其与 mjlab 逐字一致。
        disc = self.discriminator(
            actor_state,
            update_normalizer=False,
        ).detach()
        if self.discriminator_reward_type == "log":
            # Typically discimination is the output of a direct linear layer. This is the default AMP implementation.
            reward = -torch.log(1 - torch.clamp(torch.sigmoid(disc), 1e-6, 1 - 1e-6))
        elif self.discriminator_reward_type == "quad":
            # Copied from WASABI, not sure if this is correct. This assumes disc is the output of a direct linear layer.
            reward = torch.clamp(1 - (1 / 4) * torch.square(disc - 1), min=0)
        elif self.discriminator_reward_type == "wasserstein":
            # Copied from WASABI, not sure if this is correct. This assumes disc is the output of a direct linear layer.
            reward = disc
        super_auxilary_reward["discriminator_reward"] = reward
        return super_auxilary_reward

    def update(self, *args, **kwargs):
        mean_losses, average_stats = super().update(*args, **kwargs)

        # iterate over the discriminator optimization steps
        if self.discriminator.is_recurrent:
            # Leaving possibility for recurrent discriminator, but this is not well designed yet.
            amp_generator = self.amp_storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            amp_generator = self.amp_storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        num_updates = self.num_learning_epochs * self.num_mini_batches
        for amp_minibatch in amp_generator:
            losses, _, stats = self.compute_amp_losses(amp_minibatch)

            loss = 0.0
            for k, v in losses.items():
                loss += getattr(self, k + "_coef", 1.0) * v
                mean_losses[k] = mean_losses[k] + v.detach() / num_updates
            mean_losses["amp_total_loss"] = mean_losses["amp_total_loss"] + loss.detach() / num_updates
            for k, v in stats.items():
                average_stats[k] = average_stats[k] + v.detach() / num_updates

            self.wasabi_gradient_step(loss, average_stats)

        if self.discriminator.normalizer is not None:
            self.discriminator.normalizer.sync_across_processes()
        self.amp_storage.clear()

        return mean_losses, average_stats

    def compute_amp_losses(self, amp_minibatch: AmpStorage.MiniBatch):
        losses, stats, inter_vars = dict(), dict(), dict()

        # discriminator must compute the discriminator during act()
        # TODO: run recurrent discriminator.
        # ⭐ 判别器输入白化（AMP_mjlab `amp_normalizer` 等价物，2026-10-03）：
        #   统计量**只用 reference（expert）侧**累积 ⇒ actor 侧前向必须禁止更新。
        if self.discriminator.normalizer is not None and self.amp_normalizer_expert_only:
            with torch.no_grad():
                self.discriminator.normalizer.update(amp_minibatch.reference_states)
            update_norm = False
        else:
            update_norm = True
        actor_d = self.discriminator(
            amp_minibatch.actor_states,
            masks=amp_minibatch.masks,
            update_normalizer=update_norm,
        )
        reference_d = self.discriminator(
            amp_minibatch.reference_states,
            masks=amp_minibatch.masks,
            update_normalizer=update_norm,
        )

        # compute losses of the distriminator
        if self.discriminator_loss_func == "WassersteinLoss":
            actor_d_loss = actor_d.mean()
            reference_d_loss = -reference_d.mean()
        elif self.discriminator_loss_func == "BCEWithLogitsLoss":
            actor_d_loss = torch.nn.BCEWithLogitsLoss()(actor_d, torch.zeros_like(actor_d))
            reference_d_loss = torch.nn.BCEWithLogitsLoss()(reference_d, torch.ones_like(reference_d))
        elif self.discriminator_loss_func == "MSELoss":
            actor_d_loss = torch.nn.MSELoss()(actor_d, -torch.ones_like(actor_d))
            reference_d_loss = torch.nn.MSELoss()(reference_d, torch.ones_like(reference_d))
        else:
            raise NotImplementedError(f"discriminator loss function {self.discriminator_loss_func} not implemented")
        discriminator_loss = (actor_d_loss + reference_d_loss) * 0.5
        if self.discriminator_gradient_penalty_coef > 0:
            if self.discriminator_backbone_gradient_only:
                discriminator_gradient_penalty = self.compute_discriminator_backbone_gradient(amp_minibatch)
            else:
                # ⭐⭐ V30：`expert_only` 对齐 mjlab（GP 只在专家流形上限制 Lipschitz 常数）
                discriminator_gradient_penalty = self.compute_discriminator_gradient(
                    amp_minibatch, expert_only=self.discriminator_gp_expert_only
                )
        else:
            discriminator_gradient_penalty = torch.zeros(1, device=self.device)[0]

        # compute weight decay loss
        weight_decay_loss = torch.zeros(1, device=self.device)[0]
        for param in self.discriminator.parameters():
            weight_decay_loss += torch.sum(param**2)

        # Following ProtoMotions, compute the weight decay loss for the last layer of the discriminator.
        logit_weight_decay_loss = torch.zeros(1, device=self.device)[0]
        if self.discriminator_logit_weight_decay_coef > 0:
            logit_weight = self.discriminator.logit_layer_weights()
            logit_weight_decay_loss = torch.sum(logit_weight**2)

        losses["discriminator_loss"] = discriminator_loss
        losses["discriminator_gradient_penalty"] = discriminator_gradient_penalty
        losses["discriminator_weight_decay"] = weight_decay_loss
        losses["logit_weight_decay"] = logit_weight_decay_loss
        stats["discriminator_actor"] = actor_d.mean()
        stats["discriminator_reference"] = reference_d.mean()
        # ⭐ V32：判别器梯度归因（只读；`Train/disc_grad/...`）。`average_stats` 是 defaultdict
        #   ⇒ 新增键是安全的（见 `ppo.py::update` 的 `defaultdict(float)`）。
        for _k, _v in getattr(self, "_disc_grad_attr", {}).items():
            stats[_k] = _v.detach() if isinstance(_v, torch.Tensor) else _v

        return losses, inter_vars, stats

    def compute_discriminator_backbone_gradient(self, amp_minibatch: AmpStorage.MiniBatch):
        """Compute the gradient w.r.t discriminator input.
        In WASABI algorithm, it is used as a penalty to satisfy the condition of Lipschitz continuity
        """
        reference_states = amp_minibatch.reference_states
        reference_states = buffer_func(reference_states, torch.clone)

        actor_states = amp_minibatch.actor_states
        actor_states = buffer_func(actor_states, torch.clone)

        combined_states = torch.cat([actor_states, reference_states], dim=0)

        # NOTE: assumeing discriminator has encoders and backbone_run function
        latent = self.discriminator.encoders(combined_states).detach()
        latent.requires_grad = True

        disc = self.discriminator.backbone_run(latent)

        ones = torch.ones_like(disc)
        grad = torch.autograd.grad(
            outputs=disc,
            inputs=latent,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return torch.clamp(grad.norm(2, dim=1) - self.discriminator_gradient_torlerance, 0).pow(2).mean()

    # ------------------------------------------------------------------
    # ⭐ V32 只读诊断：判别器“靠什么在判断”（`|∂d/∂x|` 的分组归因）
    # ⭐ V33 修正：布局**从运行时 `obs_format` 读**，不再硬编码——见 `_disc_grad_index`
    # ------------------------------------------------------------------
    #: 观测项名 → 日志用的短名（判别器输入里出现的项名）
    _AMP_TERM_SHORT = {
        "body_pos_b": "pos",
        "body_ori_b": "ori",
        "body_lin_vel_b": "linvel",
        "body_ang_vel_b": "angvel",
        "anchor_gravity_b": "gravity",
    }
    #: 归因索引缓存（key = (观测段签名, 维度, 设备)）—— 段签名进 key，避免不同口径互相污染
    _AMP_DISC_GRAD_INDEX_CACHE: dict = {}
    #: 逐刚体项的"每刚体维度"（其余项无逐刚体结构）
    _AMP_TERM_BODY_DIMS = {
        "body_pos_b": 3,
        "body_ori_b": 6,
        "body_lin_vel_b": 3,
        "body_ang_vel_b": 3,
    }
    #: 判别器观测里的刚体数（`AMP_V20_BODY_NAMES` = 13）
    _AMP_NUM_BODIES = 13

    def _disc_grad_index(self, dim: int, device):
        """→ (按项分组, 按部位分组) 的列索引张量；结果按（维度, 设备）缓存。

        ⚠️ **修正 V32 的错误假设（2026-10-06）**：判别器输入的真实布局是**"项优先、帧在项内"**：

            观测管理器按项（ObsTerm）顺序拼接，`flatten_history_dim=True` 只在**项内**展平历史
            ⇒ [body_pos_b(2×39) | body_ori_b(2×78) | body_lin_vel_b(2×39) | body_ang_vel_b(2×39) |
                anchor_gravity_b(2×3)]

        而 V32 那版按"帧优先"硬编码（每帧 195 维 = pos|ori|lin|ang）⇒ 四类量的分组是**错的**
        （纯诊断量，不影响训练，但会把人带偏 ⇒ 必须修）。
        现在**直接读 `get_obs_slice(self._amp_obs_segments, 项名)`** ⇒ 项顺序 / 是否带 anchor 重力
        通道 / 历史帧数全都自动正确，不依赖任何硬编码假设。
        """
        segs = getattr(self, "_amp_obs_segments", None)
        # 段签名进 key：不同口径（390 / 396 / 老版本）不互相污染缓存
        seg_key = tuple((k, tuple(v)) for k, v in segs.items()) if segs else None
        key = (seg_key, int(dim), str(device))
        hit = self._AMP_DISC_GRAD_INDEX_CACHE.get(key)
        if hit is not None:
            return hit
        if not segs:
            # 拿不到布局（老接口）⇒ 放弃归因，但不影响训练
            self._AMP_DISC_GRAD_INDEX_CACHE[key] = ({}, {})
            return {}, {}
        parts = {"torso": (0,), "legs": (1, 2, 3, 4, 5, 6), "arms": (7, 8, 9, 10, 11, 12)}
        by_term: dict = {}
        by_part: dict = {k: [] for k in parts}
        for name in segs:
            sl, shape = get_obs_slice(segs, name)
            short = self._AMP_TERM_SHORT.get(name, name)
            by_term[short] = torch.arange(sl.start, sl.stop, dtype=torch.long, device=device)
            body_dims = self._AMP_TERM_BODY_DIMS.get(name)
            if body_dims is None:
                continue  # 非"逐刚体"项（如 anchor_gravity_b）⇒ 只进 by_term
            size = int(sl.stop) - int(sl.start)
            n_frames = size // (self._AMP_NUM_BODIES * body_dims)
            for f in range(n_frames):
                base_f = int(sl.start) + f * self._AMP_NUM_BODIES * body_dims
                for b in range(self._AMP_NUM_BODIES):
                    s = base_f + b * body_dims
                    for pname, bodies in parts.items():
                        if b in bodies:
                            by_part[pname].extend(range(s, s + body_dims))
        out = (
            by_term,
            {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in by_part.items() if v},
        )
        self._AMP_DISC_GRAD_INDEX_CACHE[key] = out
        return out

    def _disc_grad_attribution(self, grad) -> dict:
        """`|∂d/∂x|` 的分组均值（**只进日志**，不参与任何 loss）。grad: (N, D)。

        回答的问题：判别器把敏感度花在哪 ——
          * `disc_grad/pos|ori|linvel|angvel`：四类量哪一类是“主判据”；
          * `disc_grad/torso|legs|arms`：哪个部位是“主判据”；
          * `disc_grad/all`：整体 Lipschitz 规模（与 `Loss/discriminator_gradient_penalty` 呼应）。
        ⚠️ 若 GP 在**归一化空间**求梯度（`amp_grad_pen_space="normalized"`），归因也在该空间
        （尺度已统一，其含义更干净）。
        """
        try:
            by_term, by_part = self._disc_grad_index(grad.shape[-1], grad.device)
            if not by_term:
                return {}
            with torch.no_grad():
                ag = grad.detach().abs().mean(dim=0)  # (D,)
                out = {f"disc_grad/{k}": ag[idx].mean() for k, idx in by_term.items()}
                out.update({f"disc_grad/{k}": ag[idx].mean() for k, idx in by_part.items()})
                out["disc_grad/all"] = ag.mean()
            return out
        except Exception:  # noqa: BLE001 —— 纯诊断，绝不能影响训练
            return {}

    def compute_discriminator_gradient(self, amp_minibatch: AmpStorage.MiniBatch, expert_only: bool = False):
        """Compute the gradient w.r.t expert input.
        In WASABI algorithm, it is used as a penalty to satisfy the condition of Lipschitz continuity

        ⭐⭐ V30：`expert_only=True` 时只对 **expert/reference** 求梯度（对齐 AMP_mjlab 的
        `compute_grad_pen(*sample_amp_expert, lambda_=10)`）。旧口径（`False`）把 actor 侧
        也拉进来，等于额外要求“对策略分布也不要敏感”，与“判别器要能区分两者”相矛盾。
        """
        reference_states = amp_minibatch.reference_states
        reference_states = buffer_func(reference_states, torch.clone)
        buffer_func(reference_states, setattr, "requires_grad", True)

        if not expert_only:
            actor_states = amp_minibatch.actor_states
            actor_states = buffer_func(actor_states, torch.clone)
            buffer_func(actor_states, setattr, "requires_grad", True)
            combined_states = torch.cat([actor_states, reference_states], dim=0)
        else:
            combined_states = reference_states

        # ⭐ GP 的作用空间（2026-10-03）：白化开启后**必须**在归一化空间求 `‖∂d/∂x̃‖`。
        #    在原始空间求会得到 `(∂d/∂x̃)/σ` ⇒ σ 最小处爆量级 ⇒ GP 支配损失。
        if self.amp_grad_pen_space == "normalized" and self.discriminator.normalizer is not None:
            with torch.no_grad():
                combined_states = self.discriminator.normalize_input(combined_states)
            combined_states = combined_states.detach().requires_grad_(True)
            disc = self.discriminator.forward_normalized(
                combined_states,
                masks=amp_minibatch.masks,
            )
        else:
            disc = self.discriminator(
                combined_states,
                masks=amp_minibatch.masks,  # The mask is designed as if the discriminator is recurrent. But it is typically not.
                # ⭐ 统计量只由 `compute_amp_losses` 的 expert 路径控制（这里混了 actor 数据）
                # ⭐⭐ V30：expert_only 时不能在这里再更新一次统计量
                update_normalizer=(not self.amp_normalizer_expert_only) and (not expert_only),
            )

        ones = torch.ones_like(disc)
        grad = torch.autograd.grad(
            outputs=disc,
            inputs=combined_states,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # ⭐ V32 只读诊断：把同一个 `∂d/∂x` 顺手做分组归因（不额外求一次梯度，零额外前向）。
        self._disc_grad_attr = self._disc_grad_attribution(grad)

        return torch.clamp(grad.norm(2, dim=1) - self.discriminator_gradient_torlerance, 0).pow(2).mean()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["discriminator"] = self.discriminator.state_dict()
        state_dict["discriminator_optimizer"] = self.discriminator_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "discriminator" in state_dict:
            self.discriminator.load_state_dict(state_dict["discriminator"])
        else:
            print("[Warning] The discriminator state_dict is not found in the checkpoint")
        if "discriminator_optimizer" in state_dict:
            self.discriminator_optimizer.load_state_dict(state_dict["discriminator_optimizer"])
        else:
            print("[Warning] The discriminator_optimizer state_dict is not found in the checkpoint")

    def wasabi_gradient_step(self, loss: torch.Tensor, average_stats: dict):
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        if dist.is_initialized():
            world_size = dist.get_world_size()
            for param in self.discriminator.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
        # TODO: add clip grad norm
        self.discriminator_optimizer.step()


class WasabiPPO(WasabiAlgoMixin, PPO):
    pass
