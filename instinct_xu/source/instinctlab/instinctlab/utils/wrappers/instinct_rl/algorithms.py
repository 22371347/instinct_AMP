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
        #: 当前 step 的环境奖励（`PPO.process_env_step` 期间有效），供 lerp 公式用。
        self._task_rewards: torch.Tensor | None = None
        #: policy 观测组里速度指令的切片；只有 `disable_style_on_stand` 才会去解析。
        self._command_slice: slice | None = None
        #: `PPO.process_env_step` 用 `getattr(self, f"{k}_coef", 1.0)` 决定每个 aux 项的
        #: 系数。我们额外暴露一个仅供日志用的 `amp_style_reward`（见
        #: `compute_auxiliary_reward`），把它钉成 0.0 → 该键**只进日志、不进奖励**。
        self.amp_style_reward_coef = 0.0
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
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
        super().init_storage(num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards)

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        """把本次的**任务奖励**暂存起来，供 `compute_auxiliary_reward` 使用。

        `PPO.process_env_step` 正是在内部调用 `compute_auxiliary_reward` 的，
        所以这里用「进入前后设置/清空」的方式传递，不需要给父类签名打补丁。
        """
        if self.reward_mix_mode != "lerp":
            return super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)
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
        r_amp = aux.get("discriminator_reward")
        if r_amp is None:
            raise RuntimeError("父类没给出 'discriminator_reward'，WasabiAlgoMixin 未生效？")
        # 仅供 TensorBoard 的原始风格奖励。`amp_style_reward_coef = 0.0` 保证它
        # **不会**被 `PPO.process_env_step` 加进奖励（那里是 `if coef != 0.0: 才加`），
        # 但 `infos["step"][k] = v` 在 if 之外，所以照样会进日志。
        aux["amp_style_reward"] = r_amp

        if self.reward_mix_mode != "lerp":
            return aux

        if self._task_rewards is None:
            raise RuntimeError(
                "'lerp' 模式下 compute_auxiliary_reward 必须在 process_env_step 内部被调用"
                "（否则拿不到本次的任务奖励）"
            )

        style = r_amp
        if self.disable_style_on_stand:
            command = obs_pack["policy"][..., self._command_slice]
            cmd_speed = torch.norm(command[..., :2], dim=-1, keepdim=True) + torch.abs(command[..., 2:3])
            style = style * (cmd_speed > self.stand_speed_threshold).to(style.dtype)

        coef = float(self.discriminator_reward_coef)
        aux["discriminator_reward"] = (1.0 - self.style_lerp) * (style - self._task_rewards / coef)
        return aux
