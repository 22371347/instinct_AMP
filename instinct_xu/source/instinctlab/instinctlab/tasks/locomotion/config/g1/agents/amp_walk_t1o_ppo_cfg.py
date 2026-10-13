"""**T1o 算法侧**：从 **0 轮冷启动**跑 663 观测契约。

本臂的变量**在环境侧**（actor 观测 378 → 663：本体感受加 4 帧历史 + 删 `base_lin_vel`，
见 `amp_walk_t1o_env_cfg.py`）。算法侧**沿用 T1n 已落地的 mjlab 口径**，不做新改动：

    `discriminator_kwargs` 含 `normalizer_class_name="EmpiricalNormalization"`
    `amp_grad_pen_space = "normalized"`
    `amp_normalizer_expert_only = True`（我们 `forward` 会逐步 update 统计量，这点与 mjlab 不同）
    `style_lerp = 0.975` / `discriminator_reward_coef = 4.0`（未对齐 mjlab 的 0.025，见 ARCH_DIFF）
    `amp_reward_form = "bounded_quad"` / `amp_quad_k = 0.25`

⚠️ `fresh_discriminator_on_load=True`（继承自 T1n）在**冷启动下是 no-op**
   （根本没有 checkpoint 可加载），保留它只是为了"从 663 的 ckpt 再热启动"时行为一致。

⚠️ **必须从头训**：`--resume` 会因 policy 观测维度 378 ≠ 663 而崩。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_t1n_ppo_cfg import G1AmpWalkT1nPPORunnerCfg


@configclass
class G1AmpWalkT1oPPORunnerCfg(G1AmpWalkT1nPPORunnerCfg):
    def __post_init__(self):
        #: 父类（T1n）已经做完白化的全部回读断言，这里只加冷启动的说明与标签。
        super().__post_init__()
        #: ⭐ **本臂的关键补充**：把时序堆叠的帧数显式告诉镜像算子。
        #: 原因：isaaclab 的 `group_obs_term_dim` 报的是**已展平的标量**
        #:（`velocity_command`（4 帧×3）→ `(12,)`）⇒ 帧结构在 `obs_format` 里不存在，
        #: 镜像算子**无法自行推导**。不声明就会 fail-fast：
        #:   `KeyError: 'velocity_command' 每帧应为 3 维，收到 per_frame=12`
        #: （T1o 首发实测就是这条 —— 兵底生效，没有静默错镜像。）
        self.algorithm.policy_obs_frames = {
            "velocity_command": 4,
            "projected_gravity": 4,
            "base_ang_vel": 4,
            "joint_pos_rel": 4,
            "joint_vel": 4,
            "last_action": 4,
        }
        self.run_name = "t1o_obs663_h4_cold"

        #: 冷启动的**显式**声明：由发车脚本决定不带 `--resume`，这里只做记录 + 提醒。
        print("[T1o][COLD START] 算法沿用 T1n（判别器白化 + GP 在归一化空间）；"
              "本臂变量在环境侧：")
        print("    actor 观测 378（单帧）→ **663** = height_scan 279（单帧）"
              " + 本体感受 4 帧 × 96（= mjlab 的 384）")
        print("    ❌ 删掉 base_lin_vel（对齐 mjlab：它只在 critic；真机上也没有这个量）")
        print("    ⚠️ **必须从头训**（--resume 会因 obs 维度 378 ≠ 663 而崩）")
