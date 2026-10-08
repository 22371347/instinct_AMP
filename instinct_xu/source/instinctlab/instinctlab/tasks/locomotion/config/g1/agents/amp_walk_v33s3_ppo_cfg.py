"""V33-S3 的算法配置：**去掉白化**（保留重力通道）+ 396 维专家集 + 判别器冷启动。

## 与相邻两臂的关系

| 项 | S0（白化 0.1）| **S3** | 含义 |
|---|---|---|---|
| ① 判别器白化 | `EmpiricalNormalization(eps=0.1)` | **无** | S0 的活体侧信号被打塌（活体参考 −0.05 vs V31 +0.42、风格奖励 0.056 vs 0.226、zero_frac 0.42 vs 0.13）⇒ 回退到 V31 的"无白化"口径 |
| ② GP 空间 | `normalized` | **`raw`** | 与 ① 成对（无白化时必须 raw，否则 GP 与实际前向不一致）|
| ③ AMP 状态 | 396（含 anchor 投影重力）| **396（保留）** | 唯一实测生效的结构改动：倾角 17.2°→6.5°、`disc_grad/gravity` 全场最高 |
| ④ 专家集 | `/root/amp_expert_states_396.pt` | 同上 | 维度必须与 ③ 一致（`wasabi.py` 有 assert）|
| ⑤ 奖励 | 未动 | **定向削弱三项**（见 `amp_walk_v33_env_cfg.py::G1AmpWalkV33S3EnvCfg`）| `feet_air_time 4→2`、`joint_deviation_arms −0.1→0`、`track_lin_vel_xy_exp 3→2` |

**对比读法**：
* **S3 − S0 = 去掉①②（白化 + 归一化空间 GP）**，其余逐字相同 ⇒ 干净回答"白化到底帮了还是害了"。
* **S3 − V31 = ③重力通道 + 三项奖励削弱**（V31 无重力通道、奖励原样）⇒ 回答"关白化之后，
  重力通道 + 定向削弱能不能既保住姿态、又拿回风格信号"。

## 起点与机械动作

* fork = V31 `model_11500.pt`（**重新 fork**，不从 S0@12500 续 —— 用户确认 S0 到 12500 步态有衰退）。
  起点与 S0/A-cold 完全一致 ⇒ 可直接对照。
* **判别器从零重学**：白化去掉 ⇒ 判别器输入口径变了（归一化 → 原始）⇒ 旧权重不可复用。
  actor/critic/PPO 优化器照常热启动。

## 预登记判据（不看单点，看趋势）

1. **风格信号是否回来**：`Step/amp_style_reward` 应从 S0 的 ~0.056 回到 **≥0.15**；
   `Step/amp_style_zero_frac` 应从 0.42 降到 **≤0.2**；`Step/amp_style_std` **≥0.13**；
   `Step/amp_d_ref`（活体参考）应回到 **≥+0.2**（不再是 bias ≈0）。
2. **姿态不能回退**：`Step/tilt_deg_mean` 应保持在 **≤10°**（S0 6.5° / V31 17.2° / 参考 9°）；
   `Step/grav_bx_mean` **> −0.15**。
3. **不退化闸门**：`Mean reward ≥ 45`、`episode length ≥ 550`、`action_abs_mean ≤ 0.62`、
   `Step/air_time_hop_frac` 不得比 S0 的 0.445 明显变差（削弱 `feet_air_time` 的主要风险就在这里）。
4. **早停点**：用户已观察 S0 到 12500 有衰退 ⇒ 本轮把 **12000（=fork+500）** 也当作可导出/可评估点，
   若 12000 明显优于 12500，就用 12000（`save_interval=500` ⇒ 两份 ckpt 都有）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v27_ppo_cfg import DROP_DISC
from .amp_walk_v31_ppo_cfg import G1AmpWalkV31PPORunnerCfg
from .amp_walk_v33s_ppo_cfg import AMP_V33S_EXPERT_BUFFER


@configclass
class G1AmpWalkV33S3PPORunnerCfg(G1AmpWalkV31PPORunnerCfg):
    """v33-S3：无白化 + raw GP + 396 重力通道 + 396 专家集 + 判别器冷启动。"""

    def __post_init__(self):
        super().__post_init__()

        # ---- ① 明确"没有白化"（防御性 pop：父链若带了 normalizer 键也退回 V31 口径）----
        dk = dict(self.algorithm.discriminator_kwargs)
        dk.pop("normalizer_class_name", None)
        dk.pop("normalizer_kwargs", None)
        self.algorithm.discriminator_kwargs = dk
        # ---- ② GP 回到原始空间（无白化时必须与 raw 配对）-----------------------
        self.algorithm.amp_grad_pen_space = "raw"
        # ---- 无 normalizer ⇒ 该 flag 实际不起作用（`forward` 按 `normalizer is None` 短路）----
        self.algorithm.amp_normalizer_expert_only = True
        # ---- ③ 396 维专家数据集（与环境侧的 anchor 重力通道维度一致）------------
        #   ⚠️ V31 的父链把 `amp_expert_buffer_path` 指向 V30 的 **390** 那份
        #      ⇒ 这里必须改成 396，否则 `wasabi.py::init_storage` 的
        #        `assert 专家集维度 == AMP 状态维度` 会在建环境时就炸（fail-fast，是好事）。
        self.algorithm.amp_expert_buffer_path = AMP_V33S_EXPERT_BUFFER
        # ---- 机械动作：判别器（含可能的 normalizer buffer）从零重学 --------------
        self.ckpt_manipulator = DROP_DISC

        self.run_name = "v33s3_nowhite_rawgp_grav396_side015_weakened_fork11500"


__all__ = ["G1AmpWalkV33S3PPORunnerCfg"]
