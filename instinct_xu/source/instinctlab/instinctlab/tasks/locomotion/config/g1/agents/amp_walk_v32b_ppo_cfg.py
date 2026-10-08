"""V32-B 臂（机制对齐）：**只改白化口径**，把 `amp_normalizer_expert_only` 由 True 改成 False。

## 这一臂是什么（控制变量实验的 B 臂）

结构包 A 的两个候选机制里，我核实后发现：

* **静止档禁用风格项**（mjlab `amp_disable_style_on_stand`）：**我们早就开着**
  —— V27/V29/V30/V31 的运行期 `params/agent.yaml` 实测都是
  `disable_style_on_stand: true / stand_speed_threshold: 0.1`。⇒ 本项**无需改动**
  （我上一轮说"27% 的步里风格项在与 default_pose 对扣"是**错的**，以运行期落盘的 yaml 为准）。
* **判别器输入白化口径**：mjlab 是 `amp_normalizer.update(policy_state)` **和**
  `update(expert_state)` **两侧都更新**（`rsl_rl/algorithms/amp_ppo.py:498-499`），
  而我们是 `amp_normalizer_expert_only: true`（V27/V29/V30/V31 实测）。

⇒ **B 臂的唯一变量 = 白化统计量从"只用专家侧"改成"两侧都用"**（对齐 mjlab）。

## ⚠️ 必须同时重置白化统计量（否则这一臂等于没改）

`EmpiricalNormalization.update()` 是**累计平均**（`count` 从上一轮起就是千万级）⇒
若沿用 checkpoint 里的旧统计量，新样本对 μ/σ 的影响可忽略，改动**几乎不生效**（假阴性）。
⇒ 因此本臂同时挂 `ckpt_manipulator = "instinctlab.utils.ckpt_manipulators:reset_amp_normalizer"`
（**只重置白化的 4 个 buffer，保留判别器权重与优化器**）。

启动后**必须**从运行期 `params/agent.yaml` 核对两行：
    `amp_normalizer_expert_only: false`
并在日志里看到 `[ckpt-manipulator] reset_amp_normalizer: 重置 4 个白化 buffer …`。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_v31_ppo_cfg import G1AmpWalkV31PPORunnerCfg

#: ⚠️ 2026-10-06 复核结论：**判别器根本没接 normalizer** ⇒ 这个开关是「幽灵开关」。
#:   证据：运行期 `discriminator_kwargs` 里没有 `normalizer_class_name`（V22 是**旁支**，
#:   不在 V27~V31 的继承链上）；ckpt 里 `discriminator` 只有 6 个键（3×Linear 的 W/b），
#:   没有任何 `normalizer.*`。⇒ 本臂**不再改这个开关**（保持有效默认值 `True`），
#:   让「C 对 A」在算法侧**零差异**。真正接上白化留到阶段 2：那时要**同时**设
#:   `amp_grad_pen_space="normalized"`，并且因为判别器输入口径变了，必须**配对冷启动**。
NORMALIZER_EXPERT_ONLY_V32B = True
#: 判别器没接 normalizer ⇒ **没有任何白化 buffer 可重置** ⇒ 必须回退成 `None`（与 A 臂/V31 一致）。
#: （保留这个常量只为记录这段历史；`ckpt_manipulators.py::reset_amp_normalizer` 本身是对的，
#:  等阶段 2 真把白化接上后它就能用了。）
RESET_NORMALIZER_V32B = None


@configclass
class G1AmpWalkV32BPPORunnerCfg(G1AmpWalkV31PPORunnerCfg):
    """v32-B：**复核后已确认算法侧与 V31 完全一致**（两项都回退成 no-op）。

    历史：本文件当初是为「白化统计量改为两侧累积 + 重置白化 buffer」而建；
    2026-10-06 复核发现**判别器根本没接 normalizer**（V22 是旁支）⇒ 两项都无效
    ⇒ 已回退，使 **C 对 A 只差 `amp_walk_v32_env_cfg.py` 里那两个奖励权重**。
    保留本文件是为了留下这段经过（以及阶段 2 真接上白化时可以直接复用）。
    """

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.amp_normalizer_expert_only = NORMALIZER_EXPERT_ONLY_V32B
        # ⚠️ 复核修正：判别器没接 normalizer ⇒ 没有白化 buffer 可重置
        #    （旧写法会打印一行误导信息，并打断 sweep 的自检 ②）
        #    ⇒ 与 A 臂/V31 保持完全一致的 `None`。
        self.ckpt_manipulator = RESET_NORMALIZER_V32B


__all__ = [
    "NORMALIZER_EXPERT_ONLY_V32B",
    "RESET_NORMALIZER_V32B",
    "G1AmpWalkV32BPPORunnerCfg",
]
