"""项目自定义的 `instinct_rl` 算法类（放在 instinctlab 侧）。

为什么放在这里
--------------
`instinct_rl` 是学长交付的 RL 库。我们对它**只做 1 处必要修复**
（`runners/on_policy_runner.py`：`importlib.import_module()` 调用时漏传参数，
导致算法 `class_name` 不能用 `"module:Class"` 形式，只有算法这一处坏掉，
policy / discriminator 的同名分支都是好的）。

本文件承载**我们自己的算法子类**，用
`class_name = "instinctlab.utils.wrappers.instinct_rl:WasabiPPOAmpMix"`
引用，于是**不需要往 `instinct_rl` 里加任何文件** —— 这让「对交付包的侵入」保持为
「1 行修复 + 0 个新文件」。（若选方案 B，就得在 `instinct_rl/algorithms/` 新增文件并
改它的 `__init__.py`。）

本包已是「instinctlab ↔ instinct_rl 桥接层」：`rl_cfg.py` 是配置类，
`vecenv_wrapper.py` 是运行时类（`InstinctRlVecEnvWrapper`），所以这里再放一个运行时类
并不引入新概念。
"""

from __future__ import annotations

import torch

from instinct_rl.algorithms.wasabi import WasabiPPO
from instinct_rl.utils.utils import get_obs_slice

from instinctlab.utils.mirror import MirrorSpec

#: 速度指令项在 policy 观测组里的候选名（本任务用 `velocity_command`，
#: 与 `AMP_WALK_OBSERVATION_TERMS` 一致）。
_COMMAND_TERM_CANDIDATES = (
    "velocity_command",
    "velocity_commands",
    "commands",
    "base_velocity",
)


def find_command_slice(policy_segments) -> slice | None:
    """在 policy 观测组的 segment 表里定位速度指令项，返回它在展平观测里的切片。

    两级匹配：**精确候选名** → **唯一一个名字里含 `command` 的项**。
    两个级别都失败就返回 `None`（由调用方报错），
    **不做更模糊的猜测**：猜错会把别的观测当指令用，是典型的静默错误。
    """
    if not policy_segments:
        return None
    names = list(policy_segments.keys())
    for candidate in _COMMAND_TERM_CANDIDATES:
        if candidate in names:
            return get_obs_slice(policy_segments, candidate)[0]
    containing = [n for n in names if "command" in n]
    if len(containing) == 1:
        return get_obs_slice(policy_segments, containing[0])[0]
    return None


class WasabiPPOAmpMix(WasabiPPO):
    """`WasabiPPO` + AMP_mjlab 风格的**奖励凸组合**（零侵入）。

    动机
    ----
    | 实现 | 公式 |
    |---|---|
    | instinct_xu（`PPO.process_env_step`） | $R = r_{\\text{task}} + c \\cdot r_{\\text{amp}}$ |
    | AMP_mjlab（实测效果好） | $R = (1-\\lambda)\\,c \\cdot r_{\\text{amp}} + \\lambda\\, r_{\\text{task}}$ |

    本类**只覆盖 `compute_auxiliary_reward`**（外加构造/生命周期钩子），把 aux 从
    「风格奖励」换成「lerp 修正量」：

    $$\\text{aux} = (1-\\lambda)\\Big(r_{\\text{amp}} - \\frac{r_{\\text{task}}}{c}\\Big)
      \\;\\Longrightarrow\\;
      R = r_{\\text{task}} + c\\cdot\\text{aux}
      = (1-\\lambda)\\,c\\,r_{\\text{amp}} + \\lambda\\, r_{\\text{task}}$$

    于是 `wasabi.py` / `ppo.py` / `discriminator.py` / `amp_storage.py`
    **一行都不用改**，也不需要改任何奖励函数。

    `reward_mix_mode="additive"`（默认）时行为与 `WasabiPPO` **完全一致**
    （原样返回父类的 aux），因此可随时切回去做消融对照。

    Args:
        reward_mix_mode: `"additive"`（父类原样）或 `"lerp"`（AMP_mjlab 公式）。
        style_lerp: $\\lambda\\in[0,1]$，仅 lerp 模式生效。AMP_mjlab 用 0.75。
        disable_style_on_stand: 指令速度 ≤ `stand_speed_threshold` 时把风格项丢掉，
            奖励变成 $\\lambda\\cdot r_{\\text{task}}$（与 AMP_mjlab 的
            `amp_disable_style_on_stand` 一致）。避免 AMP 的「自然站姿」参考与
            `track_default_pose_on_stand` 惩罚互相拉扯。
        stand_speed_threshold: 上一条的阈值（默认 0.1，与 AMP_mjlab 一致）。

    Note:
        `style_lerp` / `disable_style_on_stand` / `stand_speed_threshold`
        必须同时在算法 `@configclass` 里显式声明，否则 `to_dict()` 不会带上它们。
    """

    def __init__(
        self,
        *args,
        reward_mix_mode: str = "additive",
        style_lerp: float = 0.75,
        disable_style_on_stand: bool = False,
        stand_speed_threshold: float = 0.1,
        use_mirror_loss: bool = False,
        mirror_loss_coeff: float = 0.1,
        mirror_joint_names: list[str] | None = None,
        amp_reward_form: str = "raw",
        amp_reward_eps: float = 1.0e-6,
        amp_quad_k: float = 0.25,
        amp_normalizer_expert_only: bool = True,
        amp_grad_pen_space: str = "raw",
        clip_max_std: float | None = None,
        clip_max_std_schedule: int | None = None,
        **kwargs,
    ):
        if reward_mix_mode not in ("additive", "lerp"):
            raise ValueError(f"reward_mix_mode 只支持 'additive' / 'lerp'，收到 {reward_mix_mode!r}")
        if not 0.0 <= float(style_lerp) <= 1.0:
            raise ValueError(f"style_lerp 必须落在 [0, 1]，收到 {style_lerp!r}")
        self.reward_mix_mode = reward_mix_mode
        self.style_lerp = float(style_lerp)
        self.disable_style_on_stand = bool(disable_style_on_stand)
        self.stand_speed_threshold = float(stand_speed_threshold)
        #: 镜像损失（PPO 左右等变性）：`L = coeff·||μ_π(M_obs o) − M_act(μ_π(o))||²`
        #: ⚠️ 系数必须叫 `mirror_coef`：`PPO.update` 用
        #:    `getattr(self, k + "_coef", 1.0)` 给 `losses` 里每个键取系数，
        #:    所以只要 `compute_losses` 里放一个 `losses["mirror"]` 就自动乘上它。
        self.use_mirror_loss = bool(use_mirror_loss)
        self.mirror_coef = float(mirror_loss_coeff)
        self._mirror_spec: MirrorSpec | None = None
        self._mirror_joint_names = tuple(mirror_joint_names) if mirror_joint_names else None
        #: policy 观测契约（有序 dict：项名 → shape），由 `init_storage` 填入。
        #: ⚠️ v9 起镜像布局**从它推导** ⇒ 626 / 378 共用同一份镜像代码。
        self._policy_obs_format: dict | None = None
        #: 当前 step 的环境奖励（`PPO.process_env_step` 期间有效），供 lerp 公式用。
        self._task_rewards: torch.Tensor | None = None
        #: policy 观测组里速度指令的切片；只有 `disable_style_on_stand` 才会去解析。
        self._command_slice: slice | None = None
        #: policy 观测组里 `last_action`（= 上一帧的**原始动作**）的切片，供动作幅度诊断用。
        self._last_action_slice: slice | None = None
        #: `PPO.process_env_step` 用 `getattr(self, f"{k}_coef", 1.0)` 决定每个 aux 项的
        #: 系数。我们额外暴露一个仅供日志用的 `amp_style_reward`（见
        #: `compute_auxiliary_reward`），把它钉成 0.0 → 该键**只进日志、不进奖励**。
        self.amp_style_reward_coef = 0.0
        #: 下三项同理：**全是诊断量，系数 0 ⇒ 绝不进奖励**。
        #: ⚠️ 新增 log 用的 aux 键**必须**在 `__init__` 里定义 `*_coef = 0.0`：
        #:    `process_env_step` 是 `getattr(self, k + "_coef", 1.0)`，**缺省是 1.0**
        #:    ⇒ 忘了定义就会被当成真奖励加进去，静默改变训练目标。
        self.amp_style_zero_frac_coef = 0.0
        self.amp_style_std_coef = 0.0
        self.amp_stand_frac_coef = 0.0
        #: 原始判别器输出 `d`（仅 `log_quad` 形式下有效）——**只进日志**。
        #: 它让我们第一次能直接看到 `d_actor` 的均值/方差，而不用从奖励反解。
        self.amp_d_coef = 0.0
        #: ⭐ **参考侧**判别器输出 `d_ref`（只进日志）。
        #: 判别器到底还有没有“分辨力”，不能只看策略侧的 `amp_d`，要看**两侧的差**：
        #:   gap = `d_ref − d_actor`。gap ≈ 0 ⇒ 判别器不区分（失效/饱和）；
        #:   gap 明显 ⇒ 判别器有分辨力，那风格项没用就是**权重/梯度路径**的问题。
        #: 对应 AMP_mjlab 的 `Loss/amp_policy_pred` / `Loss/amp_expert_pred`。
        self.amp_d_ref_coef = 0.0
        #: ⭐ 指令**切片**探针（只进日志）：从 policy obs 里按 `_command_slice` 取出的 vx/vy/wz。
        #: 与**环境层**的 `Step/cmd_speed_mean`（读 `command_manager` 的真值）对照，
        #: 就能判定 `disable_style_on_stand` 的门控切片有没有错位。
        self.cmd_slice_vx_coef = 0.0
        self.cmd_slice_vy_coef = 0.0
        self.cmd_slice_wz_coef = 0.0
        self.cmd_slice_speed_coef = 0.0
        #: ⭐ **真值**指令/动作（由 `vecenv_wrapper.step` 注入 `infos["amp_debug"]`）——
        #: 门控已改用它们（观测切片实测错位），下面三个键只进日志。
        self.cmd_true_vx_coef = 0.0
        self.cmd_true_speed_coef = 0.0
        self.action_true_abs_mean_coef = 0.0
        #: `infos["amp_debug"]` 的暂存（`process_env_step` 期间有效）。
        self._amp_debug: dict | None = None
        #: ⭐ 诊断（同样只进日志 ⇒ 必须钉成 0.0）：
        #:   `effective_lr`    — **真实的学习率**。我们用 `schedule="adaptive"`，
        #:       `ppo.py:236-263` 每步把 LR ×/÷1.5 并夹到 **[1e-5, 1e-2]** ⇒
        #:       配置里的 `learning_rate=1e-3` **只是初值**，"学不动"时必须看这个数。
        #:   `action_abs_mean` — `mean_j |last_action_j|`（逐 env）：动作幅度。
        #:       「只会小幅动作（跳只需小角度踝）而学不会走路（需大幅度）」这个假设，
        #:       一旦有这个数就能当场判定。
        #:   `action_abs_max`  — `max_j |last_action_j|`，看有没有哪个关节在动。
        self.effective_lr_coef = 0.0
        self.action_abs_mean_coef = 0.0
        self.action_abs_max_coef = 0.0
        #: 风格奖励的**函数形式**（在桥接层做，**避免改上游 `instinct_rl`**）：
        #:   `"raw"`      — 原样用父类给的 `discriminator_reward`（默认，与父类行为一致）；
        #:   `"log_quad"` — ⭐ LLP 论文形式 $r=-\log(1-D)$，$D=\mathrm{quad}(d)$。
        #: ⚠️ 用 `log_quad` 时 **必须**把 `discriminator_reward_type` 设为 `"wasserstein"`：
        #:    那是父类里**唯一原样返回 `d`** 的分支（`reward = disc`）。
        #:    否则拿到的是已经压过的值，反解 `d` 会有符号歧义。
        self.amp_reward_form = str(amp_reward_form)
        self.amp_reward_eps = float(amp_reward_eps)
        #: ⭐ `quad` 的**曲率系数 k**（V19b 新增）：`D = clamp(1 − k·(d−1)², 0, 1−eps)`。
        #: ⚠️ 它同时决定**下端 clamp 的位置**：`D = 0` 当 `d ≤ 1 − sqrt(1/k)`
        #:    ⇒ k=0.25（旧默认）时死区从 `d ≤ −1.0` 开始；k=0.125 ⇒ `d ≤ −1.83`。
        #:    实测（V19a）：`d` 展宽到 −0.81 后有 **5%** 的样本越过 −1.0 ⇒
        #:    `style ≡ 0` **且 `dr/dd = 0`**（纯死区）⇒ 放宽 k 把它推远。
        #: ⚠️ `k` **不改**局部斜率 `dr/dd = 2/(1−d)`（那是 log 项的贡献），只改
        #:    **奖励水平**与**死区位置**：k 越小 ⇒ 同一 d 下 D 越接近 1 ⇒ style 越大。
        self.amp_quad_k = float(amp_quad_k)
        if self.amp_quad_k <= 0.0:
            raise ValueError(f"amp_quad_k 必须 > 0，收到 {amp_quad_k!r}")
        #: ⭐ 判别器输入白化（AMP_mjlab 的 `amp_normalizer` 等价物，2026-10-03）。
        #: 真正生效需在 `discriminator_kwargs` 里接上 normalizer，例如：
        #:     discriminator_kwargs={"normalizer_class_name": "EmpiricalNormalization",
        #:                          "normalizer_kwargs": {"eps": 0.01, "until": None}}
        #: `amp_normalizer_expert_only=True` ⇒ 统计量**只用 reference/expert 侧**累积。
        self.amp_normalizer_expert_only = bool(amp_normalizer_expert_only)
        #: ⭐ GP 的作用空间：`"raw"`（上游行为）| `"normalized"`。
        #: **白化开启时必须用 `"normalized"`**（原始空间 `1/σ` 放大 ⇒ GP 爆量级支配损失）。
        if amp_grad_pen_space not in ("raw", "normalized"):
            raise ValueError(f"amp_grad_pen_space 只支持 'raw' / 'normalized'，收到 {amp_grad_pen_space!r}")
        self.amp_grad_pen_space = str(amp_grad_pen_space)
        # 这两个必须继续往 `WasabiAlgoMixin` 传（它才是损失侧的执行者）
        kwargs["amp_normalizer_expert_only"] = self.amp_normalizer_expert_only
        kwargs["amp_grad_pen_space"] = self.amp_grad_pen_space
        #: σ（动作噪声标准差）的**上限**，单位与 `std` 一致（`scale=1.0` 下即弧度）。
        #: `None` = 不钳上限（v1~v8 行为，与上游一致）。
        #: ⚠️ `instinct_rl` 的 `PPO.update` 末尾只调 `clip_std(min=...)`（`ppo.py:216`），
        #:    而 `entropy_coef` 会持续把 σ 推高 ⇒ 上限必须我们自己补（见 `update`）。
        self.clip_max_std = None if clip_max_std is None else float(clip_max_std)
        #: ⭐ σ 上限的**退火周期**（单位 = 本 run 的 `update()` 次数；`None` = 不退火）。
        #: `clip_max_std_schedule=N` ⇒ 上限从「本 run 第一次 update 时的 σ」**线性**降到
        #: `clip_max_std`，用 N 次 update 走完，之后维持 `clip_max_std`。
        #: 用途（V26）：从 σ≈0.57 的 ckpt 热启动时**渐进**压到 0.20 —— 一次性硬夹会让
        #: 奖励分布整体上移 ~4/步（噪声税 −5.3 → −0.6）⇒ value function 突跳。
        #: ⚠️ 进度**不随 ckpt 保存**：从退火中途的 ckpt 再续训，会以当时的 σ 为新起点
        #:    重算 N 次（幂等，不会重复压低）。
        self.clip_max_std_schedule = (
            None if clip_max_std_schedule is None else max(1, int(clip_max_std_schedule))
        )
        self._std_cap_step = 0
        self._std_cap_from: float | None = None
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def update(self, *args, **kwargs):
        """在 `PPO.update` 之后再补一次 **σ 上限**钳制（`clip_max_std=None` 时为 no-op）。

        为什么不在 `update` 内部钳：`PPO.update` 末尾只调
        `clip_std(min=self.clip_min_std)`（上游 `ppo.py:216`）而没有 max。
        `std.clip(min, max)` 是幂等的，所以在 `super().update()` **之后**再钳一次，
        与「在内部同时钳 min/max」完全等价，而**不必复制上游那一大段实现**。

        ⚠️ 血的教训（v9b 实测）：**不要在 `super().update()` 返回后再往 `losses` 里塞日志键**——
        上游 `ppo.py` 的时序是「`for k, v in losses.items(): loss += getattr(self, k+'_coef', 1.0) * v`
        → 累加 `mean_losses` → 最后才 `return`」⇒ 返回后才加的键：
        (1) 日志里**一个都看不到**（已经打完了）；(2) 万一挤进去了还会按**默认 coef = 1.0**
        污染 `total_loss`。要加逐关节 σ 诊断，只能从 `compute_losses` 的 dict 进去，
        并且必须显式声明 `<key>_coef = 0.0`。
        """
        losses = super().update(*args, **kwargs)
        if self.clip_max_std is not None and hasattr(self.actor_critic, "clip_std"):
            cap = float(self.clip_max_std)
            if self.clip_max_std_schedule is not None:
                # 起点 = 本 run 第一次 update 后的 σ（即 ckpt 里学到的 σ）
                if self._std_cap_from is None:
                    self._std_cap_from = float(self.actor_critic.std.max().item())
                self._std_cap_step += 1
                frac = min(1.0, self._std_cap_step / float(self.clip_max_std_schedule))
                cap = self._std_cap_from + (cap - self._std_cap_from) * frac
            self.actor_critic.clip_std(min=self.clip_min_std, max=cap)
            # 把**实际上限**与**实际 σ** 打进 TB（`Train/sigma_cap`、`Train/sigma_actual`），
            # 用来核对退火有没有按预期走。⚠️ 值必须是 tensor（日志侧会调 .isnan()/.item()）。
            try:
                stats = losses[1] if isinstance(losses, tuple) and len(losses) > 1 else None
                if isinstance(stats, dict):
                    stats["sigma_cap"] = torch.as_tensor(cap)
                    stats["sigma_actual"] = self.actor_critic.std.mean().detach().cpu()
            except Exception:  # noqa: BLE001  （日志失败绝不能影响训练）
                pass
        return losses

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        # ---- amp_reward_form 的 fail-fast（放在最前面，早于建判别器）----
        if self.amp_reward_form not in ("raw", "log_quad", "bounded_quad"):
            raise ValueError(
                f"amp_reward_form 只支持 'raw' / 'log_quad' / 'bounded_quad'，收到 {self.amp_reward_form!r}"
            )
        if self.amp_reward_form != "raw" and self.discriminator_reward_type != "wasserstein":
            raise ValueError(
                "amp_reward_form='log_quad' 需要**原始判别器输出 d** ⇒ 必须把 "
                "`discriminator_reward_type` 设为 'wasserstein'（父类里唯一原样返回 d 的分支）；"
                f"当前是 {self.discriminator_reward_type!r}"
            )
        if self.reward_mix_mode == "lerp":
            # lerp 公式里 `r_task` 与 `r_amp` 逐元素配对，多奖励组（多 critic）时
            # `r_task` 是 (num_envs, num_rewards)，语义不明确 → 直接 fail-fast。
            if num_rewards != 1:
                raise ValueError(
                    f"reward_mix_mode='lerp' 目前只支持单奖励组，收到 num_rewards={num_rewards}。"
                    "（公式里 r_amp 与 r_task 逐元素配对，多奖励组时含义不唯一）"
                )
            if float(self.discriminator_reward_coef) == 0.0:
                raise ValueError(
                    "reward_mix_mode='lerp' 需要 discriminator_reward_coef != 0："
                    "aux 里要算 r_task / c"
                )
            if self.disable_style_on_stand:
                self._command_slice = find_command_slice(obs_format.get("policy"))
                if self._command_slice is None:
                    raise ValueError(
                        "disable_style_on_stand=True 但没能在 policy 观测组里定位速度指令项；"
                        f"候选名 {list(_COMMAND_TERM_CANDIDATES)}，"
                        f"实际项 {list((obs_format.get('policy') or {}).keys())}"
                    )
        #: 镜像算子要用的观测布局（**必须在 super() 之前存**：镜像规格是懒构造的，
        #: 第一次 `compute_losses` 才用到，但不能依赖那时还能拿到 obs_format）。
        self._policy_obs_format = obs_format.get("policy")
        #: 动作幅度诊断用：`last_action` 在展平 policy 观测里的切片（缺失则诊断为 no-op）
        _pol_fmt = obs_format.get("policy") or {}
        if "last_action" in _pol_fmt:
            self._last_action_slice = get_obs_slice(_pol_fmt, "last_action")[0]
        super().init_storage(num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards)
        #: ⭐ 逐维探针用：把“每个迭代多少个 env step”记下来，探针据此按**迭代里程碑**采样
        #: （`compute_auxiliary_reward` 是每 env step 调一次 ⇒ 不这么换会把瞬态当稳态）。
        self._amp_probe_steps_per_iter = int(num_transitions_per_env)
        #: ⭐ V29：把“专家批解耦”开关接到 amp storage 上（`False` ⇒ 零行为变化）。
        try:
            self.amp_storage.independent_reference_sampling = bool(
                getattr(self, "amp_expert_independent_sampling", False)
            )
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # 镜像损失（左右等变性）
    # ------------------------------------------------------------------

    def _mirror_spec_or_raise(self) -> MirrorSpec:
        if self._mirror_spec is None:
            if not self._mirror_joint_names:
                raise ValueError(
                    "use_mirror_loss=True 但没给 mirror_joint_names；"
                    "应传 articulation 序的 29 个关节名（即 OMNI_LOCO_JOINT_ORDER）"
                )
            if self._policy_obs_format:
                # ⭐ 主路径：布局由观测契约推导 ⇒ v1~v8(626) / v9(378) 通用。
                #   项名不认识、或扫描尺寸没在 `SCAN_LAYOUTS` 登记 ⇒ 启动即报错。
                self._mirror_spec = MirrorSpec.from_obs_format(
                    list(self._mirror_joint_names), self._policy_obs_format
                )
            else:
                # 兼容路径：拿不到契约时退回旧 626 布局（等价于 v1~v8 的行为）
                self._mirror_spec = MirrorSpec.from_joint_names(list(self._mirror_joint_names))
        return self._mirror_spec

    def mirror_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """`mean_j ||μ_π(M_obs o)_j − M_act(μ_π(o))_j||²`（**不含**系数）。

        ⚠️ 已知近似：此时 `obs` 是**归一化之后**的观测（storage 存的就是它），
           所以约束的是「策略作为它真实输入的函数」的等变性，而不是物理量的等变性。
           这是通行做法（归一化本身不是镜像等变的），但值得记一笔。
        """
        spec = self._mirror_spec_or_raise()
        ac = self.actor_critic
        mu_obs = ac.action_mean.clone()          # = μ_π(o)（`compute_losses` 刚算过）
        ac.act(spec.mirror_obs(obs), masks=None, hidden_states=None)
        mu_mir = ac.action_mean                  # = μ_π(M_obs o)
        # ⚠️ `act()` 内部会 `update_distribution()`，**覆盖 `self.distribution`**。
        #    目前 `compute_losses` 之后没人再读 `action_mean`（σ 又是与输入无关的参数，
        #    回读也一致），但仍显式还原原分布 —— 已经因为一个"应该没问题"的假设崩过一次，
        #    这里用一次多余的前向换掉一类静默错误。
        ac.update_distribution(obs)
        return torch.mean(torch.sum((mu_mir - spec.mirror_action(mu_obs)) ** 2, dim=-1))

    def compute_losses(self, minibatch):
        """父类损失 + 镜像损失。

        只覆写这一个方法，**不复制 `PPO.update`**（它 90 行且会随上游演进）。
        返回元数保持不变（父类现在是三元组，这里用 `out[1:]` 兼容任意元数）。
        """
        out = super().compute_losses(minibatch)
        if self.use_mirror_loss:
            losses = dict(out[0])
            losses["mirror"] = self.mirror_loss(minibatch.obs)
            out = (losses,) + tuple(out[1:])
        return out

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        """把本次的**任务奖励**暂存起来，供 `compute_auxiliary_reward` 使用。

        `PPO.process_env_step` 正是在内部调用 `compute_auxiliary_reward` 的，
        所以这里用「进入前后设置/清空」的方式传递，不需要给父类签名打补丁。
        """
        if self.reward_mix_mode != "lerp":
            return super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)
        # ⭐ 真值旁路（`vecenv_wrapper.step` 注入）：暂存给 `compute_auxiliary_reward` 用
        self._amp_debug = infos.get("amp_debug") if isinstance(infos, dict) else None
        self._task_rewards = rewards
        try:
            return super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)
        finally:
            self._task_rewards = None

    # ------------------------------------------------------------------
    # 奖励混合
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_auxiliary_reward(self, obs_pack: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        aux = super().compute_auxiliary_reward(obs_pack)
        raw = aux.get("discriminator_reward")
        if raw is None:
            raise RuntimeError("父类没给出 'discriminator_reward'，WasabiAlgoMixin 未生效？")

        # ---- 风格奖励的函数形式（在桥接层做，不改上游）----
        if self.amp_reward_form == "log_quad":
            # ⭐ LLP 论文形式： r = -log(1 - D)，  D = quad(d) = clamp(1 - (d-1)^2/4, 0)
            #   闭式： r = -log(1/4) - 2*ln|d-1| = 1.3863 - 2*ln|d-1|
            #   ⇒ d=-1 → r=0；d=0 → 1.386；d=±0.9 → 0.103；梯度 dr/dd = -2/(d-1)（在 d=-0.9 处 = +2.0，
            #     比 log 形式的 sigmoid(d)=0.29 **强 7 倍且不消失**）
            # ⚠️⚠️ **奇点**：D 在 d=1 处**恰好等于 1** ⇒ -log(0) = +inf。
            #     MSELoss 的**参考侧目标就是 +1** ⇒ 不是罕见事件 ⇒ 必须夹住。
            #     夹到 1-eps 后奖励上界 = -log(eps) = 13.8（eps=1e-6），有限且不报 NaN。
            d = raw
            d_quad = torch.clamp(
                1.0 - self.amp_quad_k * torch.square(d - 1.0),
                min=0.0,
                max=1.0 - self.amp_reward_eps,
            )
            r_amp = -torch.log(1.0 - d_quad)
            # 只进日志（amp_d_coef = 0.0）：终于能直接看判别器原始输出，不用反解
            aux["amp_d"] = d
        elif self.amp_reward_form == "bounded_quad":
            # ⭐⭐ V29：**对齐 AMP_mjlab** 的形状（有界，且 d<=-1 时恰为 0）
            #    mjlab: `disc_r = amp_reward_coef * clamp(1 - 0.25*(d-1)^2, min=0)`
            #    这里取 k = amp_quad_k = 0.25（即 mjlab 的曲率）⇒
            #        style = clamp(1 - 0.25*(d-1)^2, 0, 1) ∈ [0, 1]
            #        零点：d <= -1；最大值：style=1 当且仅当 d=+1
            #    与 `log_quad` 的两个关键区别（都重要）：
            #      ① **有界**（[0,1] vs [0,13.8] 且 d→1 处凸增长）⇒ 不会产生
            #         “离群高分样本” ⇒ V28 的崩溃机制（单步 +12 的奖励离群，
            #         优势估计被单点拖爆）从**形状**上被消除。
            #      ② `d <= -1` 时 style 恰为 0 且梯度恰为 0（与 `log_quad` 在 d<=-1
            #         的“常数平台”等价：都不给梯度，但没有常数偏置）。
            #    ⚠️ 该形状**不解决饱和**（饱和时两侧都无梯度）——它只保证“安全”。
            d = raw
            r_amp = torch.clamp(1.0 - self.amp_quad_k * torch.square(d - 1.0), min=0.0, max=1.0)
            # 只进日志（amp_d_coef = 0.0）
            aux["amp_d"] = d
        else:
            r_amp = raw

        # 仅供 TensorBoard 的原始风格奖励。`amp_style_reward_coef = 0.0` 保证它
        # **不会**被 `PPO.process_env_step` 加进奖励（那里是 `if coef != 0.0: 才加`），
        # 但 `infos["step"][k] = v` 在 if 之外，所以照样会进日志。
        aux["amp_style_reward"] = r_amp

        # ---- ⭐ 诊断：判别器在**参考动作**上的输出（`amp_d_ref_coef = 0.0`，只进日志）----
        # 同一个判别器、same step，对 `amp_reference` 组前向一次 ⇒ `gap = d_ref - d`。
        #   `gap` 长期很小（比如 |gap| < 0.2）⇒ 判别器根本区分不开参考与策略
        #   ⇒ AMP 已经失效 —— 这才是"风格项调多大都没用"的直接证据。
        # 键名直接取 `reference_state_key`（父类用它读 `infos["observations"]` ⇒ 一定对得上）。
        _ref_key = getattr(self, "reference_state_key", "amp_reference")
        _ref = obs_pack.get(_ref_key)
        if _ref is not None:
            # `update_normalizer=False`：统计量只由 `compute_amp_losses` 的 expert 路径控制
            aux["amp_d_ref"] = (
                self.discriminator(_ref, update_normalizer=False).detach().reshape(-1).to(r_amp.dtype)
            )

        # ---- ⭐ 逐维探针（V29/V30 诊断用，只在两个时刻写档，之后完全不执行）---------------
        # 目的：把 actor / reference 的 AMP 状态**逐维 μ/σ** 落盘，离线算标准化分离度
        # `|μ_a − μ_r| / σ_r`，回答“**哪些维让判别器平凡可分**”。
        # ⭐ 采两次：`#5`（热启动瞬态，episode length ~8~137）与 `#50`（**稳态**，episode length ~500）
        #    —— 只有稳态那份才能当量化依据（瞬态会把 sep 放大）。
        # 文件格式：4 行（μ_actor / μ_ref / σ_actor / σ_ref），每行 D 个数（D = 195×history）。
        _n = getattr(self, "_amp_dim_probe_n", 0) + 1
        self._amp_dim_probe_n = _n
        # ⚠️⚠️ `compute_auxiliary_reward` 是**每个 env step** 调用一次（24 次/迭代）⇒
        #    直接按调用计数会在“训练第 1 秒”就采完（V29 踩过这个坑：把瞬态当稳态）。
        #    改成按**迭代里程碑**采：#1 / #100 / #200 / #400 / #600。
        _per_it = getattr(self, "_amp_probe_steps_per_iter", 0)
        _it = (_n // _per_it) if _per_it else 0
        _dump = bool(_per_it) and (_n % _per_it == 0) and (_it in (1, 100, 200, 400, 600))
        if _dump:
            try:
                import numpy as _np

                _pol = obs_pack.get(getattr(self, "policy_state_key", "amp_policy"))
                if _pol is not None and _ref is not None:
                    _pf = _pol.detach().float().reshape(_pol.shape[0], -1)
                    _rf = _ref.detach().float().reshape(_ref.shape[0], -1)
                    _np.savetxt(
                        f"/root/amp_dim_probe_it{_it:04d}.txt",
                        _np.stack(
                            [
                                _pf.mean(0).cpu().numpy(),
                                _rf.mean(0).cpu().numpy(),
                                _pf.std(0).cpu().numpy(),
                                _rf.std(0).cpu().numpy(),
                            ]
                        ),
                        fmt="%.6g",
                    )
            except Exception:  # noqa: BLE001  （探针绝不影响训练）
                pass
        # ----------------------------------------------------------------------------

        #: 诊断（**门控前**的 r_amp）：
        #: - `amp_style_zero_frac`：r_amp 恰好为 0 的样本比例。`quad` 形式在
        #:   `d <= -1` 时有死区（r ≡ 0 且 dr/dd = 0 ⇒ 该样本既不进奖励也不给梯度），
        #:   这个量就是死区大小；换成 `log` 后应降到 ≈0。
        #: - `amp_style_std`：⭐⭐ **r_amp 的跨 env 标准差 —— “排序信号”的直接度量**。
        #:   均值再大、若 std ≈ 0，风格对策略也只是个常数没有引导力。
        #: 用门控前的 r_amp 是为了**与 `disable_style_on_stand` 解耦**。
        aux["amp_style_zero_frac"] = (r_amp <= 0.0).to(r_amp.dtype).mean()
        aux["amp_style_std"] = r_amp.std() if r_amp.numel() > 1 else torch.zeros((), device=r_amp.device)

        # ---- ⭐ 诊断：真实（自适应）学习率 + 动作幅度 --------------------------------
        # 放在 `lerp` 的提前 return **之前**，保证所有 reward_mix_mode 都能看到这两个数。
        _pol = obs_pack["policy"]
        if self._last_action_slice is not None:
            _la = _pol[..., self._last_action_slice].abs().float()
            aux["action_abs_mean"] = _la.mean(dim=-1)
            aux["action_abs_max"] = _la.amax(dim=-1)
        #: ⭐ **真值**动作幅度（切片实测可疑，见 `cmd_slice_*` 的同类问题）
        _pa = (self._amp_debug or {}).get("prev_action")
        if _pa is not None:
            _pa = _pa.abs().float()
            aux["action_true_abs_mean"] = _pa.mean(dim=-1)
        aux["effective_lr"] = torch.full(
            (_pol.shape[0],), float(self.learning_rate), device=_pol.device, dtype=torch.float32
        )

        if self.reward_mix_mode != "lerp":
            return aux

        if self._task_rewards is None:
            raise RuntimeError(
                "'lerp' 模式下 compute_auxiliary_reward 必须在 process_env_step 内部被调用"
                "（否则拿不到本次的任务奖励）"
            )

        style = r_amp
        if self.disable_style_on_stand:
            # ⭐ 优先用**真值**：env 侧 `command_manager` 的指令由 `vecenv_wrapper.step`
            # 通过 `infos["amp_debug"]` 注入 ⇒ 不再依赖观测切片的偏移量。
            # ⚠️ 实测（V10b @iter10）：切片推出来的偏移**指错了位置** ——
            #    `cmd_slice_speed ≈ 1.72`、vx/vy/wz ≈ 0，而按分档设计速度应 ≈0.26
            #    ⇒ 门控一直没按设计工作（30% 的站立 env 从未被豁免风格奖励）。
            #    切片探针保留作为对照，待定位偏移量错在哪后再决定要不要删掉。
            _true_cmd = (self._amp_debug or {}).get("command")
            if _true_cmd is not None:
                command = _true_cmd.to(style.dtype)
            else:
                command = obs_pack["policy"][..., self._command_slice]
            cmd_speed = torch.norm(command[..., :2], dim=-1, keepdim=True) + torch.abs(command[..., 2:3])
            gate = (cmd_speed > self.stand_speed_threshold).to(style.dtype)
            style = style * gate
            #: 被门控判为“站立”的 env 比例（= 门控质量）。
            #: 之前只能用 `exp(-cmd_speed/0.25)` 粗估成 0.037，现在直接实测。
            aux["amp_stand_frac"] = (1.0 - gate).mean()
            aux["cmd_true_vx"] = command[..., 0].to(style.dtype)
            aux["cmd_true_speed"] = cmd_speed.reshape(-1).to(style.dtype)
            #: 切片侧对照（`*_coef = 0.0` ⇒ 只进日志）：与 `cmd_true_*` 一比就知道错多少。
            if _true_cmd is not None and self._command_slice is not None:
                _sl = obs_pack["policy"][..., self._command_slice]
                aux["cmd_slice_vx"] = _sl[..., 0].to(style.dtype)
                aux["cmd_slice_vy"] = _sl[..., 1].to(style.dtype)
                aux["cmd_slice_wz"] = _sl[..., 2].to(style.dtype)
                aux["cmd_slice_speed"] = (
                    torch.norm(_sl[..., :2], dim=-1) + torch.abs(_sl[..., 2])
                ).to(style.dtype)

        coef = float(self.discriminator_reward_coef)
        aux["discriminator_reward"] = (1.0 - self.style_lerp) * (style - self._task_rewards / coef)
        return aux
