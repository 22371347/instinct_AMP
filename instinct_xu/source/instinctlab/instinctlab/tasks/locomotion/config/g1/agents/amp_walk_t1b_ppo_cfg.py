"""**T1b 的 PPO runner**：T1a2 之上**只改一个数** —— `style_lerp` 1.0 → 0.75。

============================================================================
这就是"单变量增量"的字面实现
============================================================================
T1a2 的 `style_lerp = 1.0` ⇒ `aux = (1−λ)(style − r_task/coef) ≡ 0` ⇒ **风格项贡献恰为零**，
实测能力恢复到 T0 量级（`vx_high` 0.073@1000 → 0.542@2000 → 0.62@7000），
但"自主探索到的模态"有三处代价：

| 项 | T0@2000 | T1a2@7260 | 病症 |
|---|---|---|---|
| `track_ang_vel_z_exp` | +0.538 | **+0.148** | **偏航角速度只赚到 27%**（转向差） |
| `action_rate_l2` | −0.305 | **−0.433** | 动作抖、不协调 |
| `termination_penalty` | −0.0075 | **−0.092** | 早终止多 12×（ep_len 621 vs 970）|
| `vx_mid_vel_mean` | 0.20 | **0.02~0.04** | 中速档不会走（"站 or 快走"）|

本文件把风格项打开（λ=0.75，AMP_mjlab 口径），**其余一切不动** ——
判别器已在 T1a2 里连续训练 7000+ 轮，风格项一开就能立即给出有效梯度。

============================================================================
热启动（由启动器传 `--resume --load_run <T1a2 目录> --checkpoint model_7000.pt`）
============================================================================
⚠️ `cloud_train_amp_walk_v3.sh` 的已知行为：**续训会重置 AMP 参考动作的播放游标**
   （它不在 checkpoint 里）⇒ 所有环境从参考片段起点重新播放。对本实验无碍（专家集
   仍是全数据集 i.i.d. 抽样），但记录在案。
⚠️ 别忘 `disable_style_on_stand=True`（T1a2 已开）——否则站立档也会被风格项拉扯，
   与 `track_default_pose_on_stand` 抢职责。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_t1a2_ppo_cfg import G1AmpWalkT1a2PPORunnerCfg

#: T1b 的风格项权重（AMP_mjlab 口径；T1b 相对 T1a2 的**唯一**数值改动）
AMP_T1B_STYLE_LERP = 0.75


@configclass
class G1AmpWalkT1bPPORunnerCfg(G1AmpWalkT1a2PPORunnerCfg):
    """T1b：T1a2 + `style_lerp` 1.0 → 0.75（打开风格项），其余逐字不动。"""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.style_lerp = AMP_T1B_STYLE_LERP
        self.run_name = "t1b_style0p75_from_t1a2"
