"""AMP 行走 **v9（MVP）** 的 PPO runner 配置 —— 只改「探索噪声」这一项。

v9 = v8（15 项奖励 + `log_quad` 风格形式 + `ampc=0.05`）之上：

1. 环境侧四项改动（279 前向高程图 / MVP 指令 / 2-clip 数据集）—— 见 `amp_walk_v9_env_cfg.py`；
2. **本文件**：`init_noise_std` 0.2 → **0.25**，并新增 **`clip_max_std = 0.4`**。

### ⭐ noise_std 与 AMP_mjlab 的对账（用户 2026-09-28 要求核对）

| | 动作尺度 `action_scale` | `init_noise_std` | **关节空间 σ** |
|---|---|---|---|
| AMP_mjlab | **0.25**（`src/tasks/amp_loco/amp_env_cfg.py:192`）| **1.0**（`rsl_rl/config/dummy_config.yaml:69`，runner 用 `dist_cfg.get("init_std", 1.0)`）| **0.25 rad** |
| 我们（`scale=1.0`） | 1.0 | 若沿用 1.0 ⇒ 1.0 rad | ✗ 4 倍过大 |
| **我们（本版）** | 1.0 | **0.25** | **0.25 rad** ✅ |

⇒ 在 `scale = 1.0` 下，`ActorCritic.std` **就是弧度**（`init_noise_std * ones(29)`），
所以「AMP_mjlab 的 σ × (0.25/1.0) = **0.25**」才是等价的探索幅度。
（v3 曾定 0.3、v7 定 0.2，都属于「拍出来的值」；本版终于有了可对账的依据。）

### ⚠️ 为什么光改 `init_noise_std` 不够 —— 这才是发散的机理

`instinct_rl` 的 PPO **只钳下限、没有上限**：

```python
# instinct_rl/algorithms/ppo.py:215-216
if hasattr(self.actor_critic, "clip_std"):
    self.actor_critic.clip_std(min=self.clip_min_std)      # ← 只有 min
```

而 `entropy_coef = 0.008` 会**持续把 σ 往上推**，于是：
**v8 实测 `Mean action noise std = 1.04 rad ≈ ±60°`**（在 `scale = 1.0` 下这就是
直接加在关节目标上的噪声）⇒ 采集到的 rollout 严重偏离策略均值 ⇒ 训练发散。

✅ `ActorCritic.clip_std(min=None, max=None)` **本来就支持 `max`**
（`instinct_rl/modules/actor_critic.py:190-191`）⇒ 我们在**桥接层**
（`instinctlab/utils/wrappers/instinct_rl/algorithms.py` 的 `WasabiPPOAmpMix.update`）
补一次上限钳制即可，**不动上游 `instinct_rl`**。

参数取值：
- `CLIP_MAX_STD_V9 = 0.4`（= 2 × init）：允许 σ 自适应上升，但**封顶在 ±23°**；
  ⚠️ 这是本版**唯一需要后续微调的旋钮**（若训练仍发散就先收到 0.3）。
- `CLIP_MIN_STD_V9 = 0.0125`（= 5% × init，沿用 v3/v7 定的 5% 比例，
  避免 σ 一路塌到 0 失去探索）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v8_ppo_cfg import G1AmpWalkV8PPORunnerCfg

#: 探索噪声初值 = AMP_mjlab 等效值（其 `init_noise_std=1.0` × 其 `action_scale=0.25`）
NOISE_STD_V9 = 0.25
#: σ 下限（5% × init）
CLIP_MIN_STD_V9 = 0.0125
#: ⭐ σ **上限**（新增）：2 × init ⇒ ±23°，防止 σ 无界膨胀导致发散
CLIP_MAX_STD_V9 = 0.4


@configclass
class G1AmpWalkV9PPORunnerCfg(G1AmpWalkV8PPORunnerCfg):
    """v9：继承 v8 的全部算法/奖励设置，只把探索噪声改成 AMP_mjlab 等效并封顶。"""

    def __post_init__(self):
        super().__post_init__()

        # --- 唯一的算法侧改动：探索噪声 ---------------------------------
        self.policy.init_noise_std = NOISE_STD_V9
        self.algorithm.clip_min_std = CLIP_MIN_STD_V9
        #: ⚠️ 新字段：必须在 `AmpWalkAlgorithmCfg` 里声明才能传到算法侧
        #:（与 `amp_reward_form` / `mirror_loss_coeff` 同样的机制）
        self.algorithm.clip_max_std = CLIP_MAX_STD_V9

        # --- ✅ 镜像损失：**保持开启**（与 v4~v8 一致）----------------------
        # 历史：第一次启动时 `mirror.py` 把布局写死成 626 ⇒ V9 的 378 在第一个
        #   `PPO.update()` 就抛 `ValueError: policy obs 最后一维应为 626，收到 378`。
        # ✅ 已修复：`mirror.py` 改成**按项名定规则 + 布局从 `obs_format` 累加推导**
        #   ⇒ 626(527 射线) 与 378(279 射线) 共用同一份代码；
        #   新增扫描尺寸只需在 `mirror.py` 的 `SCAN_LAYOUTS` 登记 (ny, nx)。
        # ⇒ V9 与 V8 的损失构成**一致**（无这一项差异，对比干净）。
        self.algorithm.use_mirror_loss = True

        # 风格奖励形式等其余设置全部继承 v8（`wasserstein` + `log_quad` + `ampc=0.05`）
        self.run_name = f"v9_mvp_ns{NOISE_STD_V9}_maxstd{CLIP_MAX_STD_V9}_mirror"


__all__ = [
    "CLIP_MAX_STD_V9",
    "CLIP_MIN_STD_V9",
    "NOISE_STD_V9",
    "G1AmpWalkV9PPORunnerCfg",
]
