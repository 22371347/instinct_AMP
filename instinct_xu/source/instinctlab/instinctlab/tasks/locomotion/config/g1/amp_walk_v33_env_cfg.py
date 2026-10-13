"""V33：**结构级对齐 mjlab 的三臂**（S0 / S1 / A-cold）+ 派生臂 S2~S8（2026-10-06 起，共用同一 fork 点）。

## ⚠️ 本层对**今天**仍有影响的部分

* ⭐ **AMP 状态 = 396 维**（body 系 195/帧 × 2 帧 + ③ 补的"anchor 投影重力" 3 维/帧）
  ⇒ 判别器输入布局今天仍沿用（`...amp13body2frame396gra...`）。
* ⭐ **指令修正 `side_vel_min = 0.15`**（纯侧向档永不落入静止门）今天仍生效（单测 `test_side_vel_min.py` 14/14）。
* ⭐ `track_lin_vel_xy_exp` 的 `func` 换成**无损诊断版** `track_lin_vel_xy_axis_diag`（返回值与原项逐样本
  相等 ⇒ 训练奖励一分不变），今天仍是该项的 func。
* **判别器口径**（S7 正则 + S8 的 `coef=4.0` / `adaptive` LR；**无白化 / raw GP**）**今天仍在用**
  —— 经 V33-**S8** → T1a → … → T1l（见 `agents/amp_walk_t1a_ppo_cfg.py`）。
* ❌ 奖励权重（`flat_orientation −4` / `air_time` / `track_lin 3` / `track_ang 1.5` / `arms 0`）与 λ 0.75/0.70 已被 T1a/T1a2/T1l 覆盖。

## 三臂 + 派生臂一览（每条臂一句话差异 —— 查"V33Sx 做了什么"从这里进）

| 臂 | 一句话差异（相对基准）| 环境 cfg |
|---|---|---|
| **S0 结构包** | 加 ③ anchor 投影重力（390→396）；①② 白化 + GP 归一化在算法 cfg | `G1AmpWalkV33SEnvCfg` |
| **S1** | S0 + `feet_air_time 4→2`、`joint_deviation_arms −0.1→0`（拆脚手架）| `G1AmpWalkV33S1EnvCfg` |
| **S2** | S0 环境侧不动，只把白化 `eps 0.1→0.01`（mjlab 原值，差异在算法 cfg）| `G1AmpWalkV33S2EnvCfg` |
| **S3** | S0 + **去白化** + 定向削弱三项（air 2 / arms 0 / track_lin 2）| `G1AmpWalkV33S3EnvCfg` |
| **S4** | S3 + `air_time 0`、`track_lin 3`、`track_ang 1.5`、λ 0.70 | `G1AmpWalkV33S4EnvCfg` |
| **S5** | S3 + `air_time 1.0`（回退一半）+ `track_lin 3` / `track_ang 1.5`（跟踪优先）| `G1AmpWalkV33S5EnvCfg` |
| **S6** | 环境侧 = S4 逐字，只把算法侧 `discriminator_reward_coef 1→4` | `G1AmpWalkV33S6EnvCfg` |
| **S7** | 环境侧 = S5 逐字，只把判别器正则回归 mjlab（GP 5 / wd 3e-4 / logit_wd 0.04），判别器重学 | `G1AmpWalkV33S7EnvCfg` |
| **S8** | 环境侧 = S5 逐字，算法侧 `coef 4.0` + `schedule adaptive`（配对）| `G1AmpWalkV33S8EnvCfg` |
| **A-cold** | 旧口径冷启动：只换指令分布 + 丢判别器（390 / 无白化 / raw GP）| `G1AmpWalkV33BaseEnvCfg` |

⚠️ **起点**：S0/S1/A-cold = V31 `model_11500.pt`；S3 起 V31 重新 fork（S3 与 S4 同起点）；S5/S7/S8 = S3/S5 `13500.pt`。

## 本层改动一览（环境侧）

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | **判别器输入白化**（`EmpiricalNormalization`，expert-only 累积）〔算法 cfg〕| ❌ S3/S4 起**已去白化**（今天 = 无白化）|
| ② | **GP 在归一化空间求**（`amp_grad_pen_space="normalized"`）〔算法 cfg〕| ❌ 已回 **`raw`**（今天 = 无白化 + raw GP）|
| ③ | **AMP 状态补 3 维 anchor 投影重力**（390 → 396）| ✅ 今天仍生效（判别器输入 396）|
| ④ | 三臂共有：`side_vel_min = 0.15` + `track_lin_vel_xy_exp` → 无损诊断版 | ✅ 都仍生效 |
| ⑤ | S1 拆脚手架（`feet_air_time 4→2`、`joint_deviation_arms →0`）| ❌ 已被覆盖 → T1l：`air_time +1` / `arms −0.1` |
| ⑥ | S3~S8 的奖励 / λ / coef 调整 | ❌ 已被 T1a/T1a2/T1l 覆盖（仅 `coef 4.0` / adaptive LR / S7 正则沿用至今）|

## 推理与判读（已归档，勿在本文件重复）

* **① 白化的动机与后果**（expert-only 统计量 ⇒ 判别器看"同一把尺子"；后实测 S0 的白化把风格信号打塌 ⇒ S3 去白化）
  · **② 必须与 ① 同时开**（归一化空间求 GP；原始空间 `∂d/∂x = (∂d/∂x̃)/σ` 在近常数维会爆量级）
  · **S2/S3 去白化、S4↔S5 的 `air_time` 权衡、S6 崩溃、S7 负结果、S8 配对**
  → `instinct/docs/v33_tracking_gait_analysis.md` §5.5/§5.7/§5.8 + 该版 run 日志
* **③ 为什么补 anchor 投影重力**（原 390 维**全部**是"anchor 相对位姿 + 各刚体自身系速度" ⇒ 对整体旋转不变
  ⇒ 判别器对"绕横轴转 20°"无感；V30 后仰 24°（参考前倾 9°）正是这个盲区）→ 见该版 run 日志；
  该改动改变了判别器输入维度 ⇒ 专家数据集须按 396 维重导（`cloud_export_amp_expert.py --with_anchor_gravity`）
* **④ 指令修正的监督矛盾**（纯侧向档 `vy ~ U(−0.5,0.5)` 会采到 `|vy| ≤ 0.1`，而所有"静止门控"共用
  `‖(vx,vy)‖ + |ωz| ≤ low_vel` ⇒ 同时被要求"侧移"和"立定回位"）→ 见该版 run 日志 + 单测

⚠️ 其余逐字不动：观测布局（字段声明顺序 = 判别器输入拼接顺序）、`flat_orientation_l2 = −4`（V31 值）、
   `style_lerp 0.75`、专家数据集；⚠️ **不扩充参考集**（用户实测：扩充反而导致风格混乱）⇒ 参考集本次不动。

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-V33S-G1-v0`
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.locomotion.mdp as locomotion_mdp

from .amp_walk_v20_env_cfg import (
    AMP_V20_FRAME_SIZE,
    AmpV20PolicyStateObsCfg,
    AmpV20ReferenceStateObsCfg,
    _reference_anchor_cfg,
    _reference_body_cfg,
    _robot_anchor_cfg,
    _robot_body_cfg,
)
from .amp_walk_v29_env_cfg import AMP_V29_STATE_HISTORY
from .amp_walk_v32_env_cfg import (
    AMP_WALK_V32C_AIR_TIME_WEIGHT,
    AMP_WALK_V32C_ARMS_DEVIATION_WEIGHT,
    G1AmpWalkV32BEnvCfg,
    G1AmpWalkV32BEnvCfg_PLAY,
    _reweight,
)

# ============================================================================
# 常量
# ============================================================================

#: 纯侧向档的 `|vy|` 下限（必须 > `low_vel = 0.1`，否则与静止门控冲突）
AMP_WALK_V33_SIDE_VEL_MIN = 0.15

#: 新增的 "anchor 投影重力" 通道维度（anchor 刚体自身系下的世界重力方向）
AMP_WALK_V33_ANCHOR_GRAVITY_DIM = 3
#: S 臂单帧维度：195 + 3 = **198**
AMP_WALK_V33_FRAME_SIZE = AMP_V20_FRAME_SIZE + AMP_WALK_V33_ANCHOR_GRAVITY_DIM
#: S 臂判别器输入维度：2 × 198 = **396**
AMP_WALK_V33_STATE_SIZE = AMP_V29_STATE_HISTORY * AMP_WALK_V33_FRAME_SIZE

AMP_WALK_V33_S_RUN_NAME = (
    "G1AmpWalkV33SMVPPlane_obs378_rew21_amp13body2frame396gra_white_normgp_side015_fork11500"
)
AMP_WALK_V33_S1_RUN_NAME = (
    "G1AmpWalkV33S1MVPPlane_obs378_rew19_amp13body2frame396gra_white_normgp_side015_noscaffold_fork11500"
)
AMP_WALK_V33_ACOLD_RUN_NAME = (
    "G1AmpWalkV33AColdMVPPlane_obs378_rew21_amp13body2frame390_normnone_rawgp_side015_nodisc_fork11500"
)
AMP_WALK_V33_S2_RUN_NAME = (
    "G1AmpWalkV33S2MVPPlane_obs378_rew21_amp13body2frame396gra_white001_normgp_side015_fork11500"
)
AMP_WALK_V33_S3_RUN_NAME = (
    "G1AmpWalkV33S3MVPPlane_obs378_rew21_amp13body2frame396gra_nowhite_rawgp_side015_weakened_fork11500"
)
AMP_WALK_V33_S4_RUN_NAME = (
    "G1AmpWalkV33S4MVPPlane_obs378_rew20_amp13body2frame396gra_nowhite_rawgp_noairtime_boostangvel_fork12500"
)
AMP_WALK_V33_S5_RUN_NAME = (
    "G1AmpWalkV33S5MVPPlane_obs378_rew21_amp13body2frame396gra_nowhite_rawgp_side015_air1_boosttrack_fork12500"
)
AMP_WALK_V33_S6_RUN_NAME = (
    "G1AmpWalkV33S6MVPPlane_obs378_rew20_amp13body2frame396gra_nowhite_rawgp_noairtime_ampcoef4_fork12500"
)
AMP_WALK_V33_S7_RUN_NAME = (
    "G1AmpWalkV33S7MVPPlane_obs378_rew21_amp13body2frame396gra_nowhite_rawgp_side015_air1_mjlabdisc_fork13500"
)
AMP_WALK_V33_S8_RUN_NAME = (
    "G1AmpWalkV33S8MVPPlane_obs378_rew21_amp13body2frame396gra_nowhite_rawgp_side015_air1_ampcoef4_adaptive_fork13500"
)


# ============================================================================
# 观测组：V20 的 4 项 + `anchor_gravity_b`
# ============================================================================
#
# ⚠️ **字段声明顺序 = 判别器输入向量的拼接顺序**（manager 按声明顺序 `torch.cat`）。
#    子类新增的 `anchor_gravity_b` 天然排在父类 4 项**之后** ⇒ 布局为
#        [body_pos_b(78) | body_ori_b(156) | body_lin_vel_b(78) | body_ang_vel_b(78) | anchor_gravity_b(6)]
#    其中每项内部是 [t−1 帧, t 帧]（`flatten_history_dim=True`，历史在**项内**展平）。
#    → 见 `wasabi.py::_disc_grad_index`：归因索引**从运行时 `obs_format` 读**，不猜布局。


@configclass
class AmpV33PolicyStateObsCfg(AmpV20PolicyStateObsCfg):
    """判别器输入（策略侧）：V20 的 195 维/帧 + anchor 投影重力 3 维/帧 = **198 维/帧**。"""

    anchor_gravity_b = ObsTerm(
        func=instinct_mdp.anchor_gravity_b,
        history_length=AMP_V29_STATE_HISTORY,
        flatten_history_dim=True,
        params={"anchor_cfg": _robot_anchor_cfg()},
    )


@configclass
class AmpV33ReferenceStateObsCfg(AmpV20ReferenceStateObsCfg):
    """判别器输入（参考侧）：与策略侧**逐项同构**，第 5 项也同名 `anchor_gravity_b`。"""

    anchor_gravity_b = ObsTerm(
        func=instinct_mdp.anchor_gravity_b_reference_as_state,
        history_length=AMP_V29_STATE_HISTORY,
        flatten_history_dim=True,
        params={"anchor_cfg": _reference_anchor_cfg()},
    )


def _v33_reassert_s_amp_obs(cfg) -> None:
    """V33-S 的 AMP 观测组 + fail-fast 维度自检。

    ⚠️ 必须在 `super().__post_init__()` **之后**调用：祖先链的 `_patch_amp_history()` 会用
    基类模板重建 `amp_policy/amp_reference`（V20 的注释里记着这个坑），把 V33 的组整个冲掉。
    """
    cfg.observations.amp_policy = AmpV33PolicyStateObsCfg()
    cfg.observations.amp_reference = AmpV33ReferenceStateObsCfg()
    for grp_name in ("amp_policy", "amp_reference"):
        grp = getattr(cfg.observations, grp_name)
        assert hasattr(grp, "anchor_gravity_b"), (
            f"V33 的 observations.{grp_name} 被父链覆盖了（缺少 anchor_gravity_b）——"
            f" 实际类型 = {type(grp).__name__}"
        )
    assert AMP_WALK_V33_STATE_SIZE == 396, f"V33 S 臂判别器输入应为 396，实际 {AMP_WALK_V33_STATE_SIZE}"


def _v33_apply_common(cfg, run_name: str) -> None:
    """三臂共有的改动：① 侧向指令下限 ② 跟踪项换成无损诊断版 ③ run_name。"""
    cmd = cfg.commands.base_velocity
    # ---- ① 纯侧向档 `|vy| >= side_vel_min` ------------------------------------
    cmd.side_vel_min = AMP_WALK_V33_SIDE_VEL_MIN
    assert AMP_WALK_V33_SIDE_VEL_MIN > float(cmd.low_vel) > 0.0, (
        f"side_vel_min({AMP_WALK_V33_SIDE_VEL_MIN}) 必须严格大于 low_vel({cmd.low_vel})，"
        "否则纯侧向档仍会被静止门控吞掉（本次修改的全部意义就在这条不等式）"
    )

    # ---- ② 跟踪项：`func` 换成**无损诊断版**（权重 / params 逐字不动）---------
    # 返回值 `exp(-ex²/σ²)·exp(-ey²/σ²)` 与原项 `exp(-(ex²+ey²)/σ²)` 逐样本相等
    # （函数内有一次性恒等自检，不等价会直接抛错）⇒ 训练奖励**一分不变**，白拿
    # `Step/lin_track_*` 曲线：x/y 两轴因子 + 按指令档分桶。
    _base = cfg.rewards.track_lin_vel_xy_exp
    _params = dict(_base.params)
    assert set(_params) <= {"command_name", "std"}, (
        f"`track_lin_vel_xy_exp` 的 params 出现新键 {sorted(_params)}：诊断版只接受 command_name/std，"
        "请同步 `track_lin_vel_xy_axis_diag` 的签名（否则运行期才 TypeError）"
    )
    cfg.rewards.track_lin_vel_xy_exp = RewTerm(
        func=locomotion_mdp.track_lin_vel_xy_axis_diag,
        weight=_base.weight,
        params=_params,
    )

    cfg.run_name = run_name


# ============================================================================
# 环境 cfg
# ============================================================================


@configclass
class G1AmpWalkV33BaseEnvCfg(G1AmpWalkV32BEnvCfg):
    """三臂共用底座：V31（`flat_orientation_l2 = −4`）逐字 + 上述公共改动。

    ⚠️ 这一层**不加** anchor 重力通道（保持 390）⇒ A-cold 臂直接用它。
    """

    def __post_init__(self):
        super().__post_init__()
        _v33_apply_common(self, AMP_WALK_V33_ACOLD_RUN_NAME)


@configclass
class G1AmpWalkV33BaseEnvCfg_PLAY(G1AmpWalkV32BEnvCfg_PLAY):
    """回放用：与底座训练同口径。"""

    def __post_init__(self):
        super().__post_init__()
        _v33_apply_common(self, AMP_WALK_V33_ACOLD_RUN_NAME)


@configclass
class G1AmpWalkV33SEnvCfg(G1AmpWalkV33BaseEnvCfg):
    """**S0 结构包**：底座 + ③ anchor 投影重力通道（390 → 396）。

    ①②（白化 / GP 归一化空间）在算法 cfg 里；本类的环境侧改动只有 ③。
    """

    def __post_init__(self):
        super().__post_init__()
        _v33_reassert_s_amp_obs(self)
        self.run_name = AMP_WALK_V33_S_RUN_NAME


@configclass
class G1AmpWalkV33SEnvCfg_PLAY(G1AmpWalkV33BaseEnvCfg_PLAY):
    """回放用：与 S0 训练同口径（396 维 AMP 状态）。"""

    def __post_init__(self):
        super().__post_init__()
        _v33_reassert_s_amp_obs(self)
        self.run_name = AMP_WALK_V33_S_RUN_NAME


@configclass
class G1AmpWalkV33S1EnvCfg(G1AmpWalkV33SEnvCfg):
    """**S1**：S0 + 拆脚手架（`feet_air_time` 4→2、`joint_deviation_arms` −0.1→0）。"""

    def __post_init__(self):
        super().__post_init__()
        _reweight(self, "feet_air_time", AMP_WALK_V32C_AIR_TIME_WEIGHT)
        _reweight(self, "joint_deviation_arms", AMP_WALK_V32C_ARMS_DEVIATION_WEIGHT)
        self.run_name = AMP_WALK_V33_S1_RUN_NAME


@configclass
class G1AmpWalkV33S1EnvCfg_PLAY(G1AmpWalkV33SEnvCfg_PLAY):
    """回放用：与 S1 训练同奖励。"""

    def __post_init__(self):
        super().__post_init__()
        _reweight(self, "feet_air_time", AMP_WALK_V32C_AIR_TIME_WEIGHT)
        _reweight(self, "joint_deviation_arms", AMP_WALK_V32C_ARMS_DEVIATION_WEIGHT)
        self.run_name = AMP_WALK_V33_S1_RUN_NAME


# ---------------------------------------------------------------------------
# S2：**与 S0 环境侧逐字相同**，只把算法侧的"白化强度"换回 mjlab 原值（eps 0.1 → 0.01）
# ---------------------------------------------------------------------------
# 用户 2026-10-06 决定：S0 用 eps=0.1（避开 V23/V24/V26 的 style_std→0 失效模式），
# 另加一臂 S2 用 mjlab 原值 0.01 ⇒ "白化强度"本身也变成一个**有对照的变量**。
# S2 排在**最后**跑，方便把训练过程中新冒出来的调整思路一并并进这一臂。
# ⚠️ 环境侧一行不改（396 通道 / 指令分布 / 奖励表全部继承 S0）⇒ S2 − S0 = **只差一个 eps**。


@configclass
class G1AmpWalkV33S2EnvCfg(G1AmpWalkV33SEnvCfg):
    """S2：S0 环境侧不动，只换 run_name（差异全在算法 cfg 的 `normalizer_kwargs.eps`）。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V33_S2_RUN_NAME


@configclass
class G1AmpWalkV33S2EnvCfg_PLAY(G1AmpWalkV33SEnvCfg_PLAY):
    """回放用：与 S2 训练同口径。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V33_S2_RUN_NAME


# ---------------------------------------------------------------------------
# S3：**保留③重力通道，去掉①②白化** + **定向削弱三项奖励**（用户 2026-10-06 决定）
# ---------------------------------------------------------------------------
# 依据（都是实测，不是猜）：
#
# 为什么去白化：S0（白化 eps=0.1）虽没把判别器弄死，但**风格信号被打塌了** ——
#   判别器对**活体**批的输出塔回 bias（活体参考 −0.05 vs 无白化的 V31 +0.42；
#   活体策略 −0.954、0=42%），风格奖励 0.056（V31 是 0.226）、zero_frac 0.42（V31 0.13），
#   且 GP 项 0.0034（V31 0.0156）⇒ 活体侧几乎没梯度。
#   机制：白化统计量只由**离线专家集**累积（`amp_normalizer_expert_only=True`），
#   而两侧的逐维 σ 中位差 1.7×（专家集 0.108 / 活体 0.063）⇒ 活体输入尺度不对 ⇒
#   网络输出塔回 bias ⇒ 判别器实际学的是"离线数据集 vs 活体"这个**来路特征**，不是步态。
#
# 为什么保留重力通道：它是本轮**唯一实测生效**的结构改动 ——
#   倾角 17.2° → **6.5°**（参考 9°）、`grav_bx` −0.276 → −0.034、
#   且 `disc_grad/gravity` 是所有分组里最高的 ⇒ 判别器真的在用绝对姿态。
#
# 三项定向削弱（**不是整体削**：S0 的 task 侧是四代里最好的，reward 67.2 / ep_len 692）：
#   * `joint_deviation_arms` −0.1 → **0**：手臂是 default↔reference 差异最大部位
#     （肘 0.41~0.51 rad，且 default 落在参考范围**之外**）⇒ 与 AMP 先验正面对撞。V9c 有先例。
#   * `feet_air_time` 4.0 → **2.0**：我们**自加**的手写节奏器（mjlab 行走奖励集无此项）；
#     V30 逐项会计显示它是"买后仰"的钱袋（+0.33/步）；且实测 `in_band` 已 0.60、
#     `hop_frac` 仍 0.445 ⇒ 它**没在管跳**。保留 2.0（不直接归零）是为了守 hop 红线。
#   * `track_lin_vel_xy_exp` 3.0 → **2.0**：削弱唯一"让策略跟得上"的手写项，
#     把奖励预算相对让给风格项（`style_lerp=0.75` ⇒ 任务侧占比越大越压风格）。
#     ⚠️ 保留 `std=0.5` 与"无损诊断版 func"（见 `_v33_apply_common`），只改权重 ⇒ 单变量。
#
# ⚠️ **不扩充参考集**（用户实测：扩充反而导致风格混乱）⇒ 参考集 / 指令集本次不动。
# ⚠️ 起点：V31 `model_11500.pt` **重新 fork**（不是从 S0@12500 续 —— 用户确认 12500 轮步态有衰退），
#    判别器从零重学（白化去掉 ⇒ 口径变）。

#: S3 定向削弱的三项
AMP_WALK_V33S3_AIR_TIME_WEIGHT = 2.0
AMP_WALK_V33S3_ARMS_DEVIATION_WEIGHT = 0.0
AMP_WALK_V33S3_TRACK_LIN_VEL_WEIGHT = 2.0


@configclass
class G1AmpWalkV33S3EnvCfg(G1AmpWalkV33SEnvCfg):
    """S3：S0 的环境侧（含 396 重力通道）+ 定向削弱三项奖励。"""

    def __post_init__(self):
        super().__post_init__()
        _reweight(self, "feet_air_time", AMP_WALK_V33S3_AIR_TIME_WEIGHT)
        _reweight(self, "joint_deviation_arms", AMP_WALK_V33S3_ARMS_DEVIATION_WEIGHT)
        _reweight(self, "track_lin_vel_xy_exp", AMP_WALK_V33S3_TRACK_LIN_VEL_WEIGHT)
        self.run_name = AMP_WALK_V33_S3_RUN_NAME


@configclass
class G1AmpWalkV33S3EnvCfg_PLAY(G1AmpWalkV33SEnvCfg_PLAY):
    """回放用：与 S3 训练同奖励。"""

    def __post_init__(self):
        super().__post_init__()
        _reweight(self, "feet_air_time", AMP_WALK_V33S3_AIR_TIME_WEIGHT)
        _reweight(self, "joint_deviation_arms", AMP_WALK_V33S3_ARMS_DEVIATION_WEIGHT)
        _reweight(self, "track_lin_vel_xy_exp", AMP_WALK_V33S3_TRACK_LIN_VEL_WEIGHT)
        self.run_name = AMP_WALK_V33_S3_RUN_NAME


# ---------------------------------------------------------------------------
# S4：**从 S3@12500 续训**的"调参臂"（用户 2026-10-06：转向调参找最好效果）
# ---------------------------------------------------------------------------
# 用户回放结论（**行为判据，优先于我的遥测推断**）：
#   * S3 学会了向左迈步，向右还没有（可能轮数不足）；
#   * `air_time` 反而导致**抬脚过高、步态不自然** ⇒ 削弱它后有效改良；
#   * 明显延长的单脚时间与**后倾**是同一个病灶（延长时间换取单脚稳定）；
#   * **`AMP_mjlab` 完全没有这一项奖励** ⇒ 我们这项是自加的补丁。
#
# ⇒ 本轮五件事：
#   ① `feet_air_time` 2.0 → **0.0**（**取消**：自加的最后一个"节奏类"补丁）
#   ② 速度跟踪 2.0 → **3.0**（增强"速度"）
#   ③ 角速度跟踪 1.0 → **1.5**（增强"角度"—— 用户 2026-10-06 确认“角度”= **转向跟踪**）
#   ④ 躯干倾角 `flat_orientation_l2` **不动（保持 −4.0）**
#      —— 用户明确选了"只增强转向跟踪"，不想再用另一个手写项硬拉姿态
#   ⑤ AMP 风格占比（算法侧）：`style_lerp` 0.75 → **0.70**（风格 0.25→0.30）
#      ⚠️ **不动** `discriminator_reward_coef` —— V28 把它放大到 4.0 崩过
#
# 为什么可以"续训"而不是重 fork：本臂 AMP 口径**逐字不变**
#   （396 / 无白化 / raw GP / 同专家集）⇒ **判别器保留**（`ckpt_manipulator=None`），
#   actor/critic 从 S3@12500 继续 ⇒ 第一次真正做到"只改奖励权重"的单变量实验。
#
# ⚠️ 已知副作用：`feet_air_time` 取消后，`air_time_hop_frac / single_stance / in_band`
#   三条遥测会消失（weight=0 时该 term 可能不再被调用）。它们本来也不能当步态质量
#   判据（本轮教训）⇒ 若要单支撑时长的读数，事后加一个**只读**探针即可。

#: S4 的三项环境侧参数（`joint_deviation_arms` 沿用 S3 的 0.0；
#:  `flat_orientation_l2` **不动**，保持父链的 −4.0）
AMP_WALK_V33S4_AIR_TIME_WEIGHT = 0.0
AMP_WALK_V33S4_TRACK_LIN_VEL_WEIGHT = 3.0
AMP_WALK_V33S4_TRACK_ANG_VEL_WEIGHT = 1.5


def _apply_s4_weights(cfg) -> None:
    """S4 的三项权重（训练/回放两侧共用，避免两处写不一致）。"""
    _reweight(cfg, "feet_air_time", AMP_WALK_V33S4_AIR_TIME_WEIGHT)
    _reweight(cfg, "joint_deviation_arms", AMP_WALK_V33S3_ARMS_DEVIATION_WEIGHT)
    _reweight(cfg, "track_lin_vel_xy_exp", AMP_WALK_V33S4_TRACK_LIN_VEL_WEIGHT)
    _reweight(cfg, "track_ang_vel_z_exp", AMP_WALK_V33S4_TRACK_ANG_VEL_WEIGHT)


@configclass
class G1AmpWalkV33S4EnvCfg(G1AmpWalkV33SEnvCfg):
    """S4：S3 口径 + 取消 air_time + 增强角/速/姿态。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_s4_weights(self)
        self.run_name = AMP_WALK_V33_S4_RUN_NAME


@configclass
class G1AmpWalkV33S4EnvCfg_PLAY(G1AmpWalkV33SEnvCfg_PLAY):
    """回放用：与 S4 训练同奖励。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_s4_weights(self)
        self.run_name = AMP_WALK_V33_S4_RUN_NAME


# ---------------------------------------------------------------------------
# S5：**从 S3@12500 续训**的"跟踪优先"臂（用户 2026-10-06 决定）
# ---------------------------------------------------------------------------
# 起因（用户原话）："策略改进不是很明显，而且对速度指令跟踪的比较差 ⇒ 希望速度和角度都跟踪更好"。
#
# 实测依据（`Step_Monitor/tracking/tracking_vxy_rmse` 末点；线速度跟踪与"风格压力"几乎单调反相关）：
#   | 臂        | 环境侧要点                    | vxy_rmse | style_reward | tilt  | ep_len |
#   | V31@11610 | air4 / λ.75 / track2→3        | 0.268    | 0.226        | —     | 623    |
#   | S0@12490  | air4 / λ.75 / **风格被打死**  | **0.222**| 0.058        | 7.1°  | 605    |
#   | S3@12490  | **air2** / λ.75 / track2      | 0.311    | 0.241        | 15.8° | 588    |
#   | S4@13490  | **air0** / **λ.70** / track3  | **0.404**| 0.296        | 19.2° | 509    |
#   ⇒ ① 取消 `feet_air_time` 的代价是**线速度跟踪变差**（air 4→2→0 与 0.222→0.311→0.404 **单调对应**）；
#     ② 风格占比升到 0.30（λ 0.70）是第二个原因 —— 风格项是跟踪项的**直接竞争者**；
#     ③ 唯一**验证生效**的增强是**角速度跟踪**：`wz_rmse` 0.83~0.90 → **0.69~0.77**（−15%），
#        奖励项归一化后 +24% ⇒ **不能回退**，否则等于白丢收益。
#   ⚠️ 副作用提醒：S4 把 `feet_air_time` 归零后该项整体被跳过（run 名 `rew20`），
#      `air_time_hop_frac / single_stance / in_band` 三条节奏遥测一起消失。
#      S5 取 1.0 ⇒ weight≠0 ⇒ **这三条探针回得来**。
#
# 本臂四项（相对 S3）：
#   ① `feet_air_time` 2.0 → **1.0**（用户指令"回退一半"；比 S4 的 0 温和）
#   ② `track_lin_vel_xy_exp` 2.0 → **3.0**（保留 S4 的"增强速度"；跨臂看 3.0 与更好的 vxy_rmse 同现，
#      S4 的退化应归因于 ①/λ，不是它）
#   ③ `track_ang_vel_z_exp` 1.0 → **1.5**（保留 S4 已证有效的角度增强）
#   ④ `flat_orientation_l2` **不动**（−4.0）
# 算法侧（`amp_walk_v33s5_ppo_cfg.py`）：`style_lerp` **回 0.75**（撤掉 S4 的 0.70），
# 无白化 / raw GP / 396 / 同专家集 ⇒ **判别器保留续训**（`ckpt_manipulator=None`，不再冷启动）。
#
# 起点：**S3 的 `model_12500.pt`**（用户指令：回退到 S4 退化之前）。
# 判据：`vxy_rmse`（主）↓、`wz_rmse` 不掉、`vx_high_vxy_rmse`（最差档 0.63）↓、
#      红线 `ep_len ≥ 500`、`action_abs_mean ≤ 0.62`；行为以用户 MuJoCo 回放为准。

#: S5 的三项环境侧参数（`joint_deviation_arms` 沿用 S3 的 0.0）
AMP_WALK_V33S5_AIR_TIME_WEIGHT = 1.0
AMP_WALK_V33S5_TRACK_LIN_VEL_WEIGHT = 3.0
AMP_WALK_V33S5_TRACK_ANG_VEL_WEIGHT = 1.5


def _apply_s5_weights(cfg) -> None:
    """S5 的四项权重（训练/回放两侧共用，避免两处写不一致）。"""
    _reweight(cfg, "feet_air_time", AMP_WALK_V33S5_AIR_TIME_WEIGHT)
    _reweight(cfg, "joint_deviation_arms", AMP_WALK_V33S3_ARMS_DEVIATION_WEIGHT)
    _reweight(cfg, "track_lin_vel_xy_exp", AMP_WALK_V33S5_TRACK_LIN_VEL_WEIGHT)
    _reweight(cfg, "track_ang_vel_z_exp", AMP_WALK_V33S5_TRACK_ANG_VEL_WEIGHT)


@configclass
class G1AmpWalkV33S5EnvCfg(G1AmpWalkV33SEnvCfg):
    """S5：S3 口径 + air_time 折半(1.0) + 保留速度/角速度跟踪增强。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_s5_weights(self)
        self.run_name = AMP_WALK_V33_S5_RUN_NAME


@configclass
class G1AmpWalkV33S5EnvCfg_PLAY(G1AmpWalkV33SEnvCfg_PLAY):
    """回放用：与 S5 训练同奖励。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_s5_weights(self)
        self.run_name = AMP_WALK_V33_S5_RUN_NAME


# ---------------------------------------------------------------------------
# S6：**保持 S4 其余不变，只把 `discriminator_reward_coef` 1.0 → 4.0**（用户 2026-10-06 决定）
# ---------------------------------------------------------------------------
# 依据 → `instinct/docs/v33_tracking_gait_analysis.md` §5.5（本轮最重要的结构发现）：
#   实测**风格项只占奖励的 ~2.9%**：
#     r_task ≈ 2.8/步（`Episode_Reward/*` 加权和）
#     coef(1−λ)style = 1.0×0.25×0.2517 = **0.063/步**
#   ⇒ 风格项根本没能力改写步态；步态实际全由手写项塑造
#   ⇒ S4 取消 `feet_air_time`（唯一的手写节律项）后“风格接不住” ⇒ 步态崩（倾角 19.2°、hop 0.71）
#
# ⇒ 本臂 = **S4 的全部环境侧配置**（`air_time=0` / `track_lin=3.0` / `track_ang=1.5` / `flat=−4` / `arms=0`）
#   **+ 算法侧 `discriminator_reward_coef` = 4.0**（风格占比 2.9% → 10.6%）
#   fork = **S3 的 `model_12500.pt`**（与 S4 **完全同起点**）⇒ **唯一变量 = coef**。
#
# ⚠️ V28 曾在 coef=4.0 崩（10470 轮）。当时三层崩因中**两层今天已不存在**：
#   ① 风格奖励形状**无上界**（`−log(1−D)` 单步可达 +12）⇒ 奖励离群 —— V29 起已换 `bounded_quad` ∈[0,1]
#   ② 风格占比冲到 **47%**（当时 `style≈0.70`）—— 今天 coef=4 只有 **10.6%**
#   ③ 判别器**饱和**、信号无信息（`style_std` 0.058）—— 今天 **0.173（×3）**、`zero_frac` 0.104
#   ⇒ 本臂是“AMP 到底能不能塑造步态”的**决定性实验**（历史上从未有过干净数据点）。
#
# 预登记判据（步态风格为主；用户已明确“速度可以放低，步态风格优先”）：
#   ↑ `air_time_single_stance`(0.084) / `air_time_in_band_frac`(0.361)
#   ↓ `air_time_hop_frac`(0.710) / `tilt_deg_mean`(15.0°) / `foot_sep_mean`(0.322)
#   `amp_style_reward` 升、`amp_style_std` **保持 > 0.05**、`amp_style_zero_frac` **不长期 = 1.0**
#   红线（**500 轮处必查**）：`ep_len ≥ 500`、`action_abs_mean ≤ 0.62`、
#   `Mean reward` 不跌破 **20**（V28 是 42→15 渐变 470 轮后摔死）、`vxy_rmse ≤ 0.35`
# ⚠️ `bounded_quad` 的占比会随策略变好自增（d→1 时 style→1.0 ⇒ coef=4 下可达 ~40%）⇒ 正反馈风险。


@configclass
class G1AmpWalkV33S6EnvCfg(G1AmpWalkV33S4EnvCfg):
    """S6：**环境侧与 S4 逐字相同**（继承 S4），只有 run 名不同；变量在算法侧（coef=4.0）。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V33_S6_RUN_NAME


@configclass
class G1AmpWalkV33S6EnvCfg_PLAY(G1AmpWalkV33S4EnvCfg_PLAY):
    """回放用：与 S6 训练同奖励（= S4 的奖励表）。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V33_S6_RUN_NAME


# ---------------------------------------------------------------------------
# S7：**只修复判别器口径**（回归 AMP_mjlab 的正则），权重一律不动（用户 2026-10-06 决定）
# ---------------------------------------------------------------------------
# 依据 → `instinct/docs/v33_tracking_gait_analysis.md` §5.7（与 InstinctMJ/parkour `AmpAlgoCfg` 逐项对照）：
#   实测**判别器从未到达设计工作点**（MSELoss 的目标是 expert→+1 / actor→−1）：
#     `amp_d_ref` = **+0.216**（整段 0.0~0.44 漂移）、`amp_d` = **−0.724**、
#     `Loss/discriminator_loss` = **0.103 且 1000 轮不降**、`amp_style_reward` 长期挤在 **0.25**、
#     `disc_grad/angvel ÷ pos` = **17×**（判别器几乎只看角速度 ⇒ 疑似在判“运动速率/平滑度”这类来路特征）
#   ⇒ 风格信号被压在 `bounded_quad` 斜坡中段、信噪比低 ⇒ (a) AMP 无法塑造步态；(b) 抬权重即放大噪声 ⇒ 崩。
#
# 本臂唯一变量（判别器侧 5 个超参，**全部向 mjlab 对齐**）：
#   | 项 | 我们（旧） | **S7** | mjlab |
#   |---|---|---|---|
#   | `discriminator_gradient_penalty_coef` | 10.0 | **5.0** | 5.0 |
#   | `discriminator_weight_decay_coef`（loss 级）| 0.0（V19a 关）| **3.0e-4** | 3e-4 |
#   | `discriminator_logit_weight_decay_coef` | 0.0（V19a 关）| **0.04** | 0.04 |
#   | 优化器 trunk wd | 1e-3 | **3e-4** | 3e-4（无分组） |
#   | 优化器 head wd | **1e-2（极强）** | **3e-4** | 3e-4 |
#   （`schedule` 仍 `fixed`；adaptive 作为后续单独变量）
#
# 权重**全部不动**：`coef=1.0`、`λ=0.75`、奖励表 = S5（`air_time=1.0` / `track 3.0+1.5`）。
# AMP 口径（396 / 无白化 / raw GP / 同专家集）不变 ⇒ **actor 可从 S5@13500 热启**（actor obs 378 不变）；
# 但判别器正则变了 ⇒ **判别器必须重学**（`drop_amp_discriminator`）。
#
# 预登记判据（**先看判别器是否学会分开，这是“风格权重是否有意义”的前置条件**）：
#   * `amp_d_ref` → **≥ +1**（现 0.22）、`amp_d` → **≤ −1**（现 −0.72）
#   * `Loss/discriminator_loss` 明显下降（现 0.103 不降）
#   * `amp_style_reward` 两端分化（不再长期挤 0.25）、`amp_style_std` 稳定
#   * 红线：`ep_len ≥ 500`、`action_abs_mean ≤ 0.62`、`tracking/vxy_rmse ≤ 0.35`


@configclass
class G1AmpWalkV33S7EnvCfg(G1AmpWalkV33S5EnvCfg):
    """S7：**环境侧与 S5 逐字相同**（air1 / track3.0+1.5 / flat−4 / arms0），只有 run 名不同。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V33_S7_RUN_NAME


@configclass
class G1AmpWalkV33S7EnvCfg_PLAY(G1AmpWalkV33S5EnvCfg_PLAY):
    """回放用：与 S7 训练同奖励（= S5 的奖励表）。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V33_S7_RUN_NAME


# ---------------------------------------------------------------------------
# S8：**抬风格分量（coef 1.0→4.0）+ 自适应 LR（schedule fixed→adaptive）** 配对实验
# ---------------------------------------------------------------------------
# 为何必须配对（→ `instinct/docs/v33_tracking_gait_analysis.md` §5.7/§5.8）：
#   * 实测风格项只占奖励 **2.2%**（风格 0.063/步 vs 任务 2.8/步）⇒ 它**没有能力**参与步态塑形；
#   * 但单独抬分量必崩：S6（coef=4）在 **150 轮悬崖崩溃**、V28（coef=4，当时还带无界形状）渐变崩溃，
#     机制一致 = 风格奖励是**非平稳 + 对抗性**信号，占大份额后优势函数被其噪声主导，
#     而 `schedule: fixed` 无法自动降 LR ⇒ **一个坏批次即毁**；
#   * mjlab 用的是 `schedule="adaptive"` + `desired_kl=0.01`（`instinct_rl/ppo.py:236-253` 确实支持，
#     只是我们从未启用）⇒ 本臂把“抬分量”与“自适应 LR”**配对**。
#   * S7（只把判别器正则回归 mjlab）实测：均衡点**没变**（`d` −0.75~−0.79、gap 稳定 ≈1.05）
#     ⇒ “判别器工作点”不是根因；且用户回放判定 S7 的类人风格**恶化**。
#
# 本臂两个变量（其余全部沿用 S5/S7）：
#   | # | 项 | S7 | **S8** |
#   |---|---|---|---|
#   | ① | `discriminator_reward_coef` | 1.0 | **4.0**（风格占比 2.2% → **10.6%**）|
#   | ② | `algorithm.schedule` | `fixed` | **`adaptive`**（`desired_kl=0.01`）|
#  保留 S7 的判别器正则（GP 5 / wd 3e-4 / logit_wd 0.04 / trunk=head=3e-4）；
#  权重表 = S5（air1 / track3.0+1.5 / flat−4 / arms0）；`λ=0.75`。
#
# 起点：**S5 的 `model_13500.pt`**（用户判定 S7 风格恶化 ⇒ 不从 S7 续）；
#  AMP 口径没变（396/无白化/raw GP/同专家集）⇒ **判别器保留续训**（`ckpt_manipulator=None`）。
#
# 预登记判据：
#   1. **稳定性（本臂的首要目标）**：不出现 S6 那种悬崖 —— `ep_len` 不跌破 300、
#      `action_abs_mean ≤ 0.7`、`Mean reward` 不跌跲 20；`sigma`/`KL` 应被自适应机制压住。
#   2. **风格是否真的开始塑形**（行为侧以用户 MuJoCo 回放为准）：`amp_style_reward` 分布拉开、
#      `amp_style_std` 上升且稳定、`disc_grad` 通道分布变化；`air_time_*` 节奏量、`tilt`、`foot_sep` 趋势。
#   3. **不退化闸门**：`tracking/vxy_rmse ≤ 0.40`（S5 0.324）、`Mean episode length ≥ 500`。


@configclass
class G1AmpWalkV33S8EnvCfg(G1AmpWalkV33S5EnvCfg):
    """S8：**环境侧与 S5 逐字相同**（air1 / track3.0+1.5 / flat−4 / arms0）；变量全在算法侧。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V33_S8_RUN_NAME


@configclass
class G1AmpWalkV33S8EnvCfg_PLAY(G1AmpWalkV33S5EnvCfg_PLAY):
    """回放用：与 S8 训练同奖励（= S5 的奖励表）。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V33_S8_RUN_NAME


@configclass
class G1AmpWalkV33AColdEnvCfg(G1AmpWalkV33BaseEnvCfg):
    """**A-cold 对照**：旧 AMP 口径（390 维 / 无白化 / raw GP）+ 丢判别器冷启动。

    与 S0 的差异 = ①②③ **全部**（故不能用来判定单一项，只作"旧口径在新指令分布下
    重学判别器能到哪"的参照系 —— 因为我们把 fork 点的判别器永久丢掉了）。
    """

    def __post_init__(self):
        super().__post_init__()
        # 明确断言：A-cold 必须**没有**重力通道（否则这臂就不是"旧口径"了）
        # ⚠️ 用**实例级** `hasattr`（`hasattr(type(x), 字段名)` 对 `@configclass` 会误报 False）
        amp_policy = self.observations.amp_policy
        assert not hasattr(amp_policy, "anchor_gravity_b"), (
            "A-cold 臂不应带 anchor_gravity_b 通道（那是 S 臂的变量）"
            f"——实际类型 = {type(amp_policy).__name__}"
        )
        self.run_name = AMP_WALK_V33_ACOLD_RUN_NAME


@configclass
class G1AmpWalkV33AColdEnvCfg_PLAY(G1AmpWalkV33BaseEnvCfg_PLAY):
    """回放用：与 A-cold 训练同口径（390 维 AMP 状态）。"""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = AMP_WALK_V33_ACOLD_RUN_NAME


__all__ = [
    "AMP_WALK_V33_ACOLD_RUN_NAME",
    "AMP_WALK_V33_ANCHOR_GRAVITY_DIM",
    "AMP_WALK_V33_FRAME_SIZE",
    "AMP_WALK_V33_S_RUN_NAME",
    "AMP_WALK_V33_S1_RUN_NAME",
    "AMP_WALK_V33_S2_RUN_NAME",
    "AMP_WALK_V33_S3_RUN_NAME",
    "AMP_WALK_V33_S4_RUN_NAME",
    "AMP_WALK_V33_S5_RUN_NAME",
    "AMP_WALK_V33_S6_RUN_NAME",
    "AMP_WALK_V33_S7_RUN_NAME",
    "AMP_WALK_V33_S8_RUN_NAME",
    "AMP_WALK_V33_SIDE_VEL_MIN",
    "AMP_WALK_V33S3_AIR_TIME_WEIGHT",
    "AMP_WALK_V33S3_ARMS_DEVIATION_WEIGHT",
    "AMP_WALK_V33S3_TRACK_LIN_VEL_WEIGHT",
    "AMP_WALK_V33S4_AIR_TIME_WEIGHT",
    "AMP_WALK_V33S4_TRACK_ANG_VEL_WEIGHT",
    "AMP_WALK_V33S4_TRACK_LIN_VEL_WEIGHT",
    "AMP_WALK_V33S5_AIR_TIME_WEIGHT",
    "AMP_WALK_V33S5_TRACK_ANG_VEL_WEIGHT",
    "AMP_WALK_V33S5_TRACK_LIN_VEL_WEIGHT",
    "AMP_WALK_V33_STATE_SIZE",
    "AmpV33PolicyStateObsCfg",
    "AmpV33ReferenceStateObsCfg",
    "G1AmpWalkV33AColdEnvCfg",
    "G1AmpWalkV33AColdEnvCfg_PLAY",
    "G1AmpWalkV33BaseEnvCfg",
    "G1AmpWalkV33BaseEnvCfg_PLAY",
    "G1AmpWalkV33S1EnvCfg",
    "G1AmpWalkV33S1EnvCfg_PLAY",
    "G1AmpWalkV33S2EnvCfg",
    "G1AmpWalkV33S2EnvCfg_PLAY",
    "G1AmpWalkV33S3EnvCfg",
    "G1AmpWalkV33S3EnvCfg_PLAY",
    "G1AmpWalkV33S4EnvCfg",
    "G1AmpWalkV33S4EnvCfg_PLAY",
    "G1AmpWalkV33S5EnvCfg",
    "G1AmpWalkV33S5EnvCfg_PLAY",
    "G1AmpWalkV33S6EnvCfg",
    "G1AmpWalkV33S6EnvCfg_PLAY",
    "G1AmpWalkV33S7EnvCfg",
    "G1AmpWalkV33S7EnvCfg_PLAY",
    "G1AmpWalkV33S8EnvCfg",
    "G1AmpWalkV33S8EnvCfg_PLAY",
    "G1AmpWalkV33SEnvCfg",
    "G1AmpWalkV33SEnvCfg_PLAY",
]
