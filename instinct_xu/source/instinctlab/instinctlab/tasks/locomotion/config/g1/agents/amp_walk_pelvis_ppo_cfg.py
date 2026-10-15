"""**P1 算法侧**：官方 G1 rev1.0（pelvis 为根）臂的 PPO 配置。

与 T2 的算法侧**逐字相同**（style λ=0.75 / c=0.1、判别器 1024-512-256 + 白化 + GP(expert)、
trunk/head wd 1e-3/1e-2、隐藏层 (512,256,128)、`init_noise_std` 0.25 / `clip_min_std` 0.0125、
镜像损失 0.1、7 项 ×4 帧、`amp_normalizer_expert_only=True`），
只覆盖**三项与"换根"强相关的设置**：

| # | 项 | 说明 |
|---|---|---|
| 1 | `mirror_joint_names` | 必须 = **新的 articulation 关节序**（`PELVIS_JOINT_ORDER`）。镜像损失是按"名字→索引"做置换的，顺序不对会**静默地学错镜像**（不报错）。 |
| 2 | `amp_expert_buffer_path` | 判别器 anchor 从 `torso_link` 变 `pelvis` ⇒ 离线专家集的**内容**变了（维度仍 396）⇒ 必须重导到 `/root/amp_expert_states_396_pelvis.pt`。 |
| 3 | `fresh_discriminator_on_load` | 保持 `True`：动作/观测语义随关节序一起变了 ⇒ **禁止**从旧 checkpoint 续训。 |

⚠️ 本臂**必须冷启动**：`--resume` 旧 run 会加载维度相同但语义不同的权重（不会报错）。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ..amp_walk_pelvis_env_cfg import PELVIS_EXPERT_BUFFER, PELVIS_JOINT_ORDER
from .amp_walk_t2_ppo_cfg import G1AmpWalkT2PPORunnerCfg


@configclass
class G1AmpWalkPelvisPPORunnerCfg(G1AmpWalkT2PPORunnerCfg):
    """T2 的算法侧 + 三项换根覆盖（见模块 docstring）。"""

    def __post_init__(self):
        super().__post_init__()
        a = self.algorithm

        # ---- 1. 镜像损失：按新关节序 ------------------------------------------
        a.mirror_joint_names = list(PELVIS_JOINT_ORDER)

        # ---- 2. 离线专家集：pelvis anchor 口径（需云端重导）--------------------
        a.amp_expert_buffer_path = PELVIS_EXPERT_BUFFER

        # ---- 3. 冷启动（不得续训旧 checkpoint）--------------------------------
        a.fresh_discriminator_on_load = True

        self.run_name = "p1_pelvis_official_rev10_cold"

        # ==================================================================
        # fail-fast 回读
        # ==================================================================
        if list(a.mirror_joint_names) != list(PELVIS_JOINT_ORDER):
            raise ValueError("[P1] mirror_joint_names 未生效")
        if set(a.mirror_joint_names) != set(PELVIS_JOINT_ORDER):
            raise ValueError("[P1] mirror_joint_names 的关节名集合不对")
        if a.amp_expert_buffer_path != PELVIS_EXPERT_BUFFER:
            raise ValueError(f"[P1] 专家集路径未生效：{a.amp_expert_buffer_path}")
        if not bool(a.fresh_discriminator_on_load):
            raise ValueError("[P1] fresh_discriminator_on_load 必须为 True（禁止续训旧权重）")
        if not bool(a.use_mirror_loss):
            raise ValueError("[P1] use_mirror_loss 应为 True（与 T2 一致）")

        print("[P1][ALGO] 沿 T2（λ=0.75/c=0.1/白化/GP/镜像 0.1）+ 三项换根覆盖：")
        print(f"    mirror_joint_names  = PELVIS_JOINT_ORDER（{len(PELVIS_JOINT_ORDER)} 名，官方序）")
        print(f"    专家集              = {a.amp_expert_buffer_path}  ← ⚠️ 需云端重导")
        print("    fresh_discriminator_on_load = True（冷启动）")
        print("    ⚠️ 本臂不可 resume 任何旧 run（关节序变了、权重语义不同）")


__all__ = ["G1AmpWalkPelvisPPORunnerCfg"]
