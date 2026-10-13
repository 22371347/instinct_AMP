"""**T2 算法侧：全权照抄 AMP_mjlab walk 的 RL 配方**（2026-10-09）。

本臂的变量在**环境侧**（obs 1500 / mjlab 13 项奖励 / mjlab 命令 / mjlab 终止 / 事件 + 真 RSI，
见 `amp_walk_t2_env_cfg.py`）。算法侧**逐项照抄 mjlab**：

mjlab 原文（`AMP_mjlab/src/tasks/amp_loco/config/g1/rl_cfg.py::g1_amp_walk_ppo_runner_cfg`）
------------------------------------------------------------------------------------
| 项 | mjlab walk | 本文件 |
|---|---|---|
| actor/critic hidden | `(512, 256, 128)` elu + `obs_normalization` | ✅ 同 |
| `init_std` | 1.0（scalar） | **0.25**（例外 B） |
| `min_normalized_std` | `[0.05]*29` ⇒ 策略 σ 下限 `clamp(min=0.05)` | **0.0125**（例外 B） |
| `entropy_coef` | 0.005 | ✅ 0.005 |
| epochs / minibatches / lr / schedule | 5 / 4 / 1e-3 / adaptive(desired_kl 0.01) | ✅ 同 |
| γ / λ / clip / max_grad_norm | 0.99 / 0.95 / 0.2 / 1.0 | ✅ 同 |
| `num_steps_per_env` | 24 | ✅ 同 |
| **风格强度** | `amp_task_reward_lerp=0.75`、`amp_reward_coef=0.1` ⇒ **(1−λ)·c = 0.025** | ✅ **照抄 0.025**（T1i~T1o 是 4× 的 0.1） |
| 静止去风格 | `amp_disable_style_on_stand=True`（`cmd_speed ≤ 0.1` ⇒ `reward = λ·task`） | ✅ True |
| 判别器容量 | `[1024, 512, 256]` ReLU | ✅ 同（我们历史值 `[1024,512]`） |
| 判别器输入白化 | `Normalizer`（`clamp((x−mean)/√(var+1e-4), ±10)`），**两端**批量更新 | ✅ 开（⚠️ 只 expert 侧更新，见偏离 #11） |
| 判别器损失 | `0.5(MSE(expert→+1) + MSE(policy→−1))` + GP(λ=10, **仅 expert**) | ✅ 同 |
| 判别器优化器 | **与策略共用** `Adam(lr=1e-3 自适应)`；trunk wd 1e-3 / head wd 1e-2 / logit wd 0 | ⚠️ **只对齐参数**（偏离 #2） |
| 专家批 | 离线运动数据集（`AMPLoader`，预载 200k 转移） | ✅ `/root/amp_expert_states_396.pt` |
| 镜像损失 | `mirror_loss_coeff = 0.1`（`use_data_augmentation=False`） | ✅ 同 |

3 处例外（用户指定，与环境侧一致）
----------------------------------
* **A. `action_scale` 保持 1.0**（契约层断言）。⚠️ mjlab 的 scale 是 **per-joint
  `0.25·effort/stiffness`**，但它取到的是**它自己那套执行器增益**：
  hip/knee 0.351、hip_yaw 0.548、arms/waist/ankle 0.439（我按 armature·(2π·10)² 复算过）。
  我们那套增益的 `effort/stiffness ≈ 1`（hip 88/90、knee 139/140、ankle 20/20、arm 25/25）
  ⇒ **同一条公式在我们这里给出每关节 ≈0.25**。⇒ 我们的 scale=1.0 = 各自的 4× / 2.4×。
  推论（与踝部限位有关）：我们对踝的"位置权限"是 mjlab 的 2.3×（而踝刚度还更高：20 vs 14.25）
  ⇒ **踝更容易被怼到限位**。T2 用 mjlab 的 `dof_pos_limits = −10（全关节）` 去对抗它。
* **B. 探索噪声按放大比缩小**：mjlab 的归一化 σ 作用在"已乘 scale"的动作上，我们是 scale=1.0
  ⇒ 同等的**关节空间**噪声 = `σ_norm × 0.25`（我们增益下的等价 scale）
  ⇒ **初始 `1.0 × 0.25 = 0.25`**、**下限 `0.05 × 0.25 = 0.0125`**。
  ⭐ 这不是"拍一个保守值"：0.25 正是 mjlab 那套 σ 在我们关节空间里的**等价点**
  （若按 mjlab 自己的增益算，等价点会是 0.29~0.55，与我们 scale=1.0 无法同时满足）。
  ⚠️ 两边 σ 都是**标量**（mjlab `std_type="scalar"`）⇒ 关节间的差异全部由 action scale 承担；
  我们 scale 恒为 1.0 ⇒ 我们的关节空间噪声是**均匀** 0.25 rad（这是例外 A 的必然结果）。
  mjlab 的 σ 相关值：`init 1.0` / 下限 `0.05`（每步 `optimizer.step()` 后 `clamp(min=log(0.05))`）/
  **无上限**（我们的 `clip_max_std=None` 已对齐）/ `entropy_coef 0.005`（从 0.008 改过来）。
* **C. `height_scan` 发 4 帧**（环境侧）。

⚠️ 已知偏离 #11：**白化统计量只由 expert 侧累积**（`amp_normalizer_expert_only=True`）
----------------------------------------------------------------------------------------
mjlab 在**每个 minibatch** 里用 `vstack(policy_batch, expert_batch)` **批量**更新统计量；
而我们的 `EmpiricalNormalization.forward` 是"前向时顺手更新"，rollout 期**每个 env step**
都会拿策略数据前向一次算风格奖励 ⇒ 打开双端会让统计量被**策略分布**带偏
（`amp_walk_ppo_cfg.py` 里记录的实测漂移 **1.51**）。
⇒ 本臂保持 `expert_only=True`（统计量来自参考，干净）。这是"照抄"与"我们的实现细节"之间
唯一必须让步的地方。

⚠️ 必须**从头训**：obs 维度 378/663 ≠ 1500，`--resume` 会直接崩。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_t1o_ppo_cfg import G1AmpWalkT1oPPORunnerCfg

# ============================================================================
# mjlab walk 的 RL 口径（照抄值；改任何一项都要在此留痕）
# ============================================================================

#: `amp_task_reward_lerp`
T2_STYLE_LERP = 0.75
#: `amp_reward_coef` ⇒ 有效风格权重 `(1−λ)·c = 0.025`
T2_DISC_REWARD_COEF = 0.1
#: `amp_discr_hidden_dims`
T2_DISC_HIDDEN = [1024, 512, 256]
T2_ENTROPY_COEF = 0.005
T2_HIDDEN_DIMS = [512, 256, 128]
#: 例外 B：探索噪声（mjlab 归一化值 × 我们/他们的 action_scale 比 0.25）
T2_INIT_NOISE_STD = 0.25
#: 例外 B：σ 下限（= mjlab `min_normalized_std[0] = 0.05` × 0.25）
T2_CLIP_MIN_STD = 0.0125
#: `compute_grad_pen(..., lambda_=10)`
T2_GP_COEF = 10.0
#: 优化器分组 wd（mjlab：trunk 1e-3 / head 1e-2 / logit 无）
T2_OPT_WD_TRUNK = 1.0e-3
T2_OPT_WD_HEAD = 1.0e-2
#: mjlab 判别器与策略**共用**优化器、lr = `learning_rate` = 1e-3（KL 自适应）。
#: ⚠️ 我们仍是独立优化器、固定 lr（偏离 #2）⇒ 取 mjlab 的**名义** lr。
T2_DISC_LR = 1.0e-3
#: 离线专家集（396 = 13body×15×2 帧 + anchor 重力 3×2，与我们的 AMP 状态维度一致）
T2_EXPERT_BUFFER = "/root/amp_expert_states_396.pt"
#: 白化：`EmpiricalNormalization(eps=0.01)`。⚠️ mjlab 是 `eps=1e-4` **且带 ±10 夹紧**；
#: 我们的实现**没有夹紧**，故用 `eps=0.01` 防止"σ≈0 的维把差距放大"（T1n 起的口径）。
T2_NORMALIZER = {
    "normalizer_class_name": "EmpiricalNormalization",
    "normalizer_kwargs": {"eps": 0.01, "until": None},
}

#: ⭐ 时序堆叠项的帧数（项名 → 帧数），供镜像算子用。
#: 例外 C：`height_scan` 这一次**也**堆叠 4 帧 ⇒ 7 项全部要声明。
#: ⚠️ 不声明就会 fail-fast：`KeyError: 'joint_pos_rel' 每帧应为 29 维，收到 per_frame=116`
#: （isaaclab 的 `group_obs_term_dim` 报的是**已展平的标量**，帧结构在 `obs_format` 里不存在）。
T2_OBS_FRAMES = {
    "height_scan": 4,
    "velocity_command": 4,
    "projected_gravity": 4,
    "base_ang_vel": 4,
    "joint_pos_rel": 4,
    "joint_vel": 4,
    "last_action": 4,
}


@configclass
class G1AmpWalkT2PPORunnerCfg(G1AmpWalkT1oPPORunnerCfg):
    """T2 的算法侧（父链 = T1o → T1n → T1i → … → V33S，已含白化/GP 归一化/镜像损失）。"""

    def __post_init__(self):
        super().__post_init__()
        a, p = self.algorithm, self.policy

        # ---- 1. 风格强度：照抄 mjlab（本臂唯一的"强度"改动）-------------------
        a.style_lerp = T2_STYLE_LERP
        a.discriminator_reward_coef = T2_DISC_REWARD_COEF
        #: mjlab walk 有它（静止时风格为 0，避免与 `track_default_pose_on_stand` 对拉）
        a.disable_style_on_stand = True
        a.stand_speed_threshold = 0.1

        # ---- 2. 探索与网络：照抄 + 例外 B --------------------------------------
        p.init_noise_std = T2_INIT_NOISE_STD
        a.clip_min_std = T2_CLIP_MIN_STD
        a.clip_max_std = None            # mjlab 没有 σ 上限
        a.clip_max_std_schedule = None
        a.entropy_coef = T2_ENTROPY_COEF
        p.actor_hidden_dims = list(T2_HIDDEN_DIMS)
        p.critic_hidden_dims = list(T2_HIDDEN_DIMS)

        # ---- 3. 判别器：容量 / 白化 / GP / wd / 优化器 / 专家批 -----------------
        dk = dict(a.discriminator_kwargs)
        dk["hidden_sizes"] = list(T2_DISC_HIDDEN)
        dk["nonlinearity"] = "ReLU"
        dk.update(T2_NORMALIZER)
        a.discriminator_kwargs = dk
        a.amp_grad_pen_space = "normalized"
        a.discriminator_gradient_penalty_coef = T2_GP_COEF
        a.discriminator_gp_expert_only = True
        a.discriminator_weight_decay_coef = 0.0
        a.discriminator_logit_weight_decay_coef = 0.0
        a.discriminator_optimizer_trunk_weight_decay = T2_OPT_WD_TRUNK
        a.discriminator_optimizer_head_weight_decay = T2_OPT_WD_HEAD
        a.discriminator_optimizer_class_name = "Adam"
        a.discriminator_optimizer_kwargs = {"lr": T2_DISC_LR, "betas": [0.9, 0.999]}
        a.discriminator_loss_func = "MSELoss"
        a.amp_expert_independent_sampling = True
        a.amp_expert_buffer_path = T2_EXPERT_BUFFER
        a.fresh_discriminator_on_load = True
        #: ⚠️ 偏离 #11（见文件头）
        a.amp_normalizer_expert_only = True

        # ---- 4. 镜像算子：7 项全部 4 帧（含 height_scan）-----------------------
        a.policy_obs_frames = dict(T2_OBS_FRAMES)
        a.use_mirror_loss = True
        a.mirror_loss_coeff = 0.1

        self.run_name = "t2_mjlab_copy_obs1500_style0p025_cold"

        # ==================================================================
        # fail-fast 回读（烘进配置 ⇒ "不校验直接启动"也安全）
        # ==================================================================
        checks = [
            ("style_lerp", float(a.style_lerp), T2_STYLE_LERP),
            ("discriminator_reward_coef", float(a.discriminator_reward_coef), T2_DISC_REWARD_COEF),
            ("entropy_coef", float(a.entropy_coef), T2_ENTROPY_COEF),
            ("clip_min_std", float(a.clip_min_std), T2_CLIP_MIN_STD),
            ("init_noise_std", float(p.init_noise_std), T2_INIT_NOISE_STD),
            ("discriminator_gradient_penalty_coef", float(a.discriminator_gradient_penalty_coef), T2_GP_COEF),
            ("discriminator_optimizer_trunk_weight_decay", float(a.discriminator_optimizer_trunk_weight_decay), T2_OPT_WD_TRUNK),
            ("discriminator_optimizer_head_weight_decay", float(a.discriminator_optimizer_head_weight_decay), T2_OPT_WD_HEAD),
            ("discriminator_weight_decay_coef", float(a.discriminator_weight_decay_coef), 0.0),
            ("discriminator_logit_weight_decay_coef", float(a.discriminator_logit_weight_decay_coef), 0.0),
        ]
        for name, got, want in checks:
            if abs(got - want) > 1e-15:
                raise ValueError(f"[T2] {name} 回读不符：{got}（应 {want}）")
        if list(p.actor_hidden_dims) != T2_HIDDEN_DIMS:
            raise ValueError(f"[T2] actor_hidden_dims = {list(p.actor_hidden_dims)}（应 {T2_HIDDEN_DIMS}）")
        if list(p.critic_hidden_dims) != T2_HIDDEN_DIMS:
            raise ValueError(f"[T2] critic_hidden_dims = {list(p.critic_hidden_dims)}（应 {T2_HIDDEN_DIMS}）")
        if list(dk["hidden_sizes"]) != T2_DISC_HIDDEN:
            raise ValueError(f"[T2] 判别器容量 = {list(dk['hidden_sizes'])}（应 {T2_DISC_HIDDEN}）")
        if str(dk.get("normalizer_class_name")) != "EmpiricalNormalization":
            raise ValueError(f"[T2] 白化未生效：{dk.get('normalizer_class_name')!r}")
        if str(a.amp_grad_pen_space) != "normalized":
            raise ValueError(f"[T2] amp_grad_pen_space = {a.amp_grad_pen_space!r}（应 'normalized'）")
        if not bool(a.discriminator_gp_expert_only):
            raise ValueError("[T2] discriminator_gp_expert_only 必须为 True（mjlab 只在专家侧求 GP）")
        if not bool(a.disable_style_on_stand):
            raise ValueError("[T2] disable_style_on_stand 必须为 True（mjlab walk）")
        if not bool(a.amp_expert_independent_sampling):
            raise ValueError("[T2] amp_expert_independent_sampling 必须为 True（mjlab 专家批独立重采）")
        if not bool(a.amp_normalizer_expert_only):
            raise ValueError(
                "[T2] amp_normalizer_expert_only 必须保持 True —— 我们的 forward 会**每 env step**"
                " 更新统计量，打开双端会让它被策略分布带偏（mjlab 是每 minibatch 批量更新）"
            )
        if str(a.amp_reward_form) != "bounded_quad" or str(a.discriminator_reward_type) != "wasserstein":
            raise ValueError(
                f"[T2] 风格奖励形状意外：{a.amp_reward_form!r} / {a.discriminator_reward_type!r}"
                "（应 bounded_quad / wasserstein = mjlab 的 clamp(1−0.25(d−1)², 0)）"
            )
        if abs(float(a.amp_quad_k) - 0.25) > 1e-15:
            raise ValueError(f"[T2] amp_quad_k = {a.amp_quad_k}（应 0.25）")
        if dict(a.policy_obs_frames) != T2_OBS_FRAMES:
            raise ValueError(f"[T2] policy_obs_frames = {a.policy_obs_frames}（应 {T2_OBS_FRAMES}）")

        print("[T2][COPY MJLAB · ALGO] 算法侧照抄 AMP_mjlab walk：")
        print(
            f"    风格强度 (1−λ)·c = {(1 - a.style_lerp) * a.discriminator_reward_coef:.4f}"
            f"   (λ={a.style_lerp}, c={a.discriminator_reward_coef})"
            f"  ← T1i~T1o 是 0.1000（4×）"
        )
        print(
            f"    判别器：容量={list(dk['hidden_sizes'])} 白化=EmpiricalNormalization"
            f"(expert_only={a.amp_normalizer_expert_only}) GP={a.discriminator_gradient_penalty_coef}"
            f"(expert_only={a.discriminator_gp_expert_only}, space={a.amp_grad_pen_space})"
            f" 优化器={a.discriminator_optimizer_class_name}{a.discriminator_optimizer_kwargs}"
            f" wd(trunk/head)={a.discriminator_optimizer_trunk_weight_decay}/{a.discriminator_optimizer_head_weight_decay}"
        )
        print(f"    专家批 = {a.amp_expert_buffer_path}（独立重采 = {a.amp_expert_independent_sampling}）")
        print(
            f"    策略：hidden={T2_HIDDEN_DIMS} entropy={a.entropy_coef}"
            f" 例外 B：init_std={p.init_noise_std}（mjlab 1.0×0.25）、clip_min_std={a.clip_min_std}"
            f"（mjlab 0.05×0.25）；σ 上限 = {a.clip_max_std}（mjlab 无）"
        )
        print(
            f"    静止去风格 disable_style_on_stand={a.disable_style_on_stand}"
            f"（阈值 {a.stand_speed_threshold}）+ mirror_loss={a.use_mirror_loss}/{a.mirror_loss_coeff}"
        )
        print(f"    ⚠️ 必须**从头训**（obs 1500，--resume 会崩）")


__all__ = [
    "T2_CLIP_MIN_STD",
    "T2_DISC_HIDDEN",
    "T2_DISC_LR",
    "T2_DISC_REWARD_COEF",
    "T2_ENTROPY_COEF",
    "T2_EXPERT_BUFFER",
    "T2_GP_COEF",
    "T2_HIDDEN_DIMS",
    "T2_INIT_NOISE_STD",
    "T2_NORMALIZER",
    "T2_OBS_FRAMES",
    "T2_OPT_WD_HEAD",
    "T2_OPT_WD_TRUNK",
    "T2_STYLE_LERP",
    "G1AmpWalkT2PPORunnerCfg",
]
