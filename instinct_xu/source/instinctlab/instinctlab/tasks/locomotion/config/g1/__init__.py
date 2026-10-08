"""Register the G1 OmniLoco experiment, the AMP-walk teacher, and their play configs."""

import gymnasium as gym

from . import agents
from .amp_walk_env_cfg import G1AmpWalkEnvCfg, G1AmpWalkEnvCfg_PLAY, G1AmpWalkTeacherEnv
from .amp_walk_v2_env_cfg import G1AmpWalkV2EnvCfg, G1AmpWalkV2EnvCfg_PLAY
from .amp_walk_v3_env_cfg import G1AmpWalkV3EnvCfg, G1AmpWalkV3EnvCfg_PLAY, G1AmpWalkV3TeacherEnv
from .amp_walk_v4_env_cfg import G1AmpWalkV4EnvCfg, G1AmpWalkV4EnvCfg_PLAY
from .amp_walk_v5_env_cfg import G1AmpWalkV5EnvCfg, G1AmpWalkV5EnvCfg_PLAY
from .amp_walk_v6_env_cfg import G1AmpWalkV6EnvCfg, G1AmpWalkV6EnvCfg_PLAY
from .amp_walk_v7_env_cfg import G1AmpWalkV7EnvCfg, G1AmpWalkV7EnvCfg_PLAY
from .amp_walk_v8_env_cfg import G1AmpWalkV8EnvCfg, G1AmpWalkV8EnvCfg_PLAY
from .amp_walk_v9_env_cfg import G1AmpWalkV9EnvCfg, G1AmpWalkV9EnvCfg_PLAY, G1AmpWalkV9TeacherEnv
from .amp_walk_v9c_env_cfg import G1AmpWalkV9cEnvCfg, G1AmpWalkV9cEnvCfg_PLAY
from .amp_walk_v9d_env_cfg import G1AmpWalkV9dEnvCfg, G1AmpWalkV9dEnvCfg_PLAY
from .amp_walk_v9e_env_cfg import G1AmpWalkV9eEnvCfg, G1AmpWalkV9eEnvCfg_PLAY
from .amp_walk_v9f_env_cfg import G1AmpWalkV9fEnvCfg, G1AmpWalkV9fEnvCfg_PLAY
from .amp_walk_v9g_env_cfg import G1AmpWalkV9gEnvCfg, G1AmpWalkV9gEnvCfg_PLAY
from .amp_walk_v9h_env_cfg import G1AmpWalkV9hEnvCfg, G1AmpWalkV9hEnvCfg_PLAY
from .amp_walk_v9i_env_cfg import G1AmpWalkV9iEnvCfg, G1AmpWalkV9iEnvCfg_PLAY
from .amp_walk_v9j_env_cfg import G1AmpWalkV9jEnvCfg, G1AmpWalkV9jEnvCfg_PLAY
from .amp_walk_v10_env_cfg import G1AmpWalkV10EnvCfg, G1AmpWalkV10EnvCfg_PLAY, G1AmpWalkV10TeacherEnv
from .amp_walk_v10b_env_cfg import G1AmpWalkV10bEnvCfg, G1AmpWalkV10bEnvCfg_PLAY, G1AmpWalkV10bTeacherEnv
from .amp_walk_v11_env_cfg import (
    G1AmpWalkV11EnvCfg,
    G1AmpWalkV11EnvCfg_PLAY,
    G1AmpWalkV11TeacherEnv,
)
from .amp_walk_v12_env_cfg import (
    G1AmpWalkV12EnvCfg,
    G1AmpWalkV12EnvCfg_PLAY,
    G1AmpWalkV12TeacherEnv,
)
from .amp_walk_v13_env_cfg import (
    G1AmpWalkV13EnvCfg,
    G1AmpWalkV13EnvCfg_PLAY,
    G1AmpWalkV13TeacherEnv,
)
from .amp_walk_v14_env_cfg import (
    G1AmpWalkV14EnvCfg,
    G1AmpWalkV14EnvCfg_PLAY,
    G1AmpWalkV14TeacherEnv,
)
from .amp_walk_v15_env_cfg import (
    G1AmpWalkV15EnvCfg,
    G1AmpWalkV15EnvCfg_PLAY,
    G1AmpWalkV15TeacherEnv,
)
from .amp_walk_v16_env_cfg import (
    G1AmpWalkV16EnvCfg,
    G1AmpWalkV16EnvCfg_PLAY,
    G1AmpWalkV16TeacherEnv,
)
from .amp_walk_v17_env_cfg import (
    G1AmpWalkV17EnvCfg,
    G1AmpWalkV17EnvCfg_PLAY,
    G1AmpWalkV17TeacherEnv,
)
from .amp_walk_v18_env_cfg import (
    G1AmpWalkV18EnvCfg,
    G1AmpWalkV18EnvCfg_PLAY,
    G1AmpWalkV18TeacherEnv,
)
from .amp_walk_v19a_env_cfg import (
    G1AmpWalkV19AEnvCfg,
    G1AmpWalkV19AEnvCfg_PLAY,
    G1AmpWalkV19ATeacherEnv,
)
from .amp_walk_v19b_env_cfg import (
    G1AmpWalkV19BEnvCfg,
    G1AmpWalkV19BEnvCfg_PLAY,
    G1AmpWalkV19BTeacherEnv,
)
from .amp_walk_v19c_env_cfg import (
    G1AmpWalkV19CEnvCfg,
    G1AmpWalkV19CEnvCfg_PLAY,
    G1AmpWalkV19CTeacherEnv,
)
from .amp_walk_v19d_env_cfg import (
    G1AmpWalkV19DEnvCfg,
    G1AmpWalkV19DEnvCfg_PLAY,
    G1AmpWalkV19DTeacherEnv,
)
from .amp_walk_v20_env_cfg import (
    G1AmpWalkV20EnvCfg,
    G1AmpWalkV20EnvCfg_PLAY,
    G1AmpWalkV20TeacherEnv,
)
from .amp_walk_v21_env_cfg import (
    G1AmpWalkV21EnvCfg,
    G1AmpWalkV21EnvCfg_PLAY,
    G1AmpWalkV21TeacherEnv,
)
from .amp_walk_v23_env_cfg import (
    G1AmpWalkV23EnvCfg,
    G1AmpWalkV23EnvCfg_PLAY,
    G1AmpWalkV23TeacherEnv,
)
from .amp_walk_v24_env_cfg import (
    G1AmpWalkV24EnvCfg,
    G1AmpWalkV24EnvCfg_PLAY,
    G1AmpWalkV24TeacherEnv,
)
from .amp_walk_v26_env_cfg import (
    G1AmpWalkV26EnvCfg,
    G1AmpWalkV26EnvCfg_PLAY,
    G1AmpWalkV26TeacherEnv,
)
from .amp_walk_v27_env_cfg import (
    G1AmpWalkV27EnvCfg,
    G1AmpWalkV27EnvCfg_PLAY,
    G1AmpWalkV27TeacherEnv,
)
from .amp_walk_v28_env_cfg import (
    G1AmpWalkV28EnvCfg,
    G1AmpWalkV28EnvCfg_PLAY,
    G1AmpWalkV28TeacherEnv,
)
from .amp_walk_v29_env_cfg import (
    G1AmpWalkV29EnvCfg,
    G1AmpWalkV29EnvCfg_PLAY,
    G1AmpWalkV29TeacherEnv,
)
from .amp_walk_v30_env_cfg import (
    G1AmpWalkV30EnvCfg,
    G1AmpWalkV30EnvCfg_PLAY,
    G1AmpWalkV30TeacherEnv,
)
from .amp_walk_v31_env_cfg import (
    G1AmpWalkV31EnvCfg,
    G1AmpWalkV31EnvCfg_PLAY,
    G1AmpWalkV31TeacherEnv,
)
from .amp_walk_v32_env_cfg import (
    G1AmpWalkV32BEnvCfg,
    G1AmpWalkV32BEnvCfg_PLAY,
    G1AmpWalkV32CEnvCfg,
    G1AmpWalkV32CEnvCfg_PLAY,
)
from .amp_walk_v33_env_cfg import (
    G1AmpWalkV33AColdEnvCfg,
    G1AmpWalkV33AColdEnvCfg_PLAY,
    G1AmpWalkV33S1EnvCfg,
    G1AmpWalkV33S1EnvCfg_PLAY,
    G1AmpWalkV33S2EnvCfg,
    G1AmpWalkV33S2EnvCfg_PLAY,
    G1AmpWalkV33S3EnvCfg,
    G1AmpWalkV33S3EnvCfg_PLAY,
    G1AmpWalkV33S4EnvCfg,
    G1AmpWalkV33S4EnvCfg_PLAY,
    G1AmpWalkV33S5EnvCfg,
    G1AmpWalkV33S5EnvCfg_PLAY,
    G1AmpWalkV33S6EnvCfg,
    G1AmpWalkV33S6EnvCfg_PLAY,
    G1AmpWalkV33S7EnvCfg,
    G1AmpWalkV33S7EnvCfg_PLAY,
    G1AmpWalkV33S8EnvCfg,
    G1AmpWalkV33S8EnvCfg_PLAY,
    G1AmpWalkV33SEnvCfg,
    G1AmpWalkV33SEnvCfg_PLAY,
)
# ==== T1a（2026-10-07）：AMP-Walk 接口/特性骨架 + T0 等价奖励集（风格关闭）====
# 与 S0~S8 的关系：同底座（V33S），但奖励表换成 T0(OmniLoco 基座) 等价集，
# 并由算法 cfg 把 style_lerp 置 1.0（风格项归零）+ 打开策略层镜像损失。
# 详细依据见 `amp_walk_t1a_env_cfg.py` 的文件头注释。
from .amp_walk_t1a_env_cfg import (
    G1AmpWalkT1aEnvCfg,
    G1AmpWalkT1aEnvCfg_PLAY,
)
# ==== T1a2（2026-10-07）：T1a + 奖励**实现/作用域**逐字对齐 T0 + 探索/网络调回 T0 ====
# T1a 只对齐了权重、没对齐实现，实测收敛到"站着不动"（审计记录见
# `instinct/tools/cloud_diff_t1a_t0.py` 与 `amp_walk_t1a2_env_cfg.py` 文件头）。
from .amp_walk_t1a2_env_cfg import (
    G1AmpWalkT1a2EnvCfg,
    G1AmpWalkT1a2EnvCfg_PLAY,
)

# ==== T1e（2026-10-07）：T1a2/T1d 的奖励表 + **把 `dof_pos_limits` 换回全关节软边距版** ====
# 动机：T1a2 "对齐 T0 实现" 时把 V33 的 `joint_pos_limits_margin` 换成了 T0 的
#       `joint_pos_limits`（作用域**仅踝关节**）⇒ 腰/髋/膝限位在奖励与指标里全是盲区，
#       T1d 实测出现关节限位问题。本臂恢复全关节软边距版（margin 0.15 / power 2.0）
#       + 恢复它自带的诊断（limit_margin_u_max/u_frac、waist_pitch_*）。
# ⚠️ 有意的偏离 T0 口径。算法仍复用 T1b 的 λ=0.75（与 T1d 逐字同一支）。
from .amp_walk_t1e_env_cfg import (
    G1AmpWalkT1eEnvCfg,
    G1AmpWalkT1eEnvCfg_PLAY,
)

# ==== T1f（2026-10-07）：**把 mjlab 的奖励机理恢复回来** ====
# T1a2~T1e 为了"与 T0 奖励白名单逐字对齐"，把 mjlab v5 的 `feet_lateral_width(-10)`
# `feet_crossing(-20)` 置 0 禁用，并把 `joint_pos_limits` 从 -10 降到 -1（仅踝）
# ⇒ mjlab 防止"窄步 / 脚掌侧边缘站姿"的核心机理被关掉（踝 roll 从 ≤500 轮起冻结在
#   限位附近、`limit_margin_u_max=2.08`）。本臂恢复并校准，同时对齐 PPO 探索口径
#   （σ 下限 0.05、λ=0.90、hidden (512,256,128)、entropy 0.005）。
from .amp_walk_t1f_env_cfg import (
    G1AmpWalkT1fEnvCfg,
    G1AmpWalkT1fEnvCfg_PLAY,
)

# ==== T1j（2026-10-08）：解除"相对夹角的常数拉力" ====
# 用户观察：AMP 风格行走时"大腿弯曲但大腿−骨盆夹角保持常数 ⇒ 骨盆与上身一同倾斜"，
# 且**只在前进时**出现。核实：`joint_deviation_torso` 作用域是 `waist_.*`（**含 waist_pitch**）、
# `joint_deviation_knee` 把膝角拉向常数；而 `waist_pitch` 行程只有 ±0.52 rad（除踝外最挤）。
# 本臂 = **T1i + 只置零这两项** ⇒ 相对 T1i 严格单变量。
from .amp_walk_t1j_env_cfg import (
    G1AmpWalkT1jEnvCfg,
    G1AmpWalkT1jEnvCfg_PLAY,
)

# ==== T1k（2026-10-08）：在 T1j 基础上恢复 mjlab 三项机理 ====
# `track_base_height +1.5`（治蹲姿）＋ `feet_lateral_width −1.0` / `feet_crossing −2.0`
# （**mjlab 的 1/10**，用户要求以小值加入、便于后续消融对照曲线）。
# 起点 = T1a2 的 model_7000.pt（**与 T1j 同起点**）；算法仍用 T1i 的 λ=0.975。
from .amp_walk_t1k_env_cfg import (
    G1AmpWalkT1kEnvCfg,
    G1AmpWalkT1kEnvCfg_PLAY,
)

# ==== T1l（2026-10-08）：把两项"历史偏离"改回 V33 / AMP_mjlab 原值 ====
# 用户决策：① "std 放宽比较好。可以给到 0.3"；② "应该优先提升到1.5。因为我们本质上
#   调整姿态不就是在追求速度跟踪性能的优化吗？"。
# ① `track_lin_vel_xy_exp` weight **1.0 → 1.5**
#    由来：T1a 白名单抄 T0 值时从 V33 的 1.5 降到 1.0（`track_ang_vel_z_exp` 同步 1.2→1.0），
#    而 1.5 正是 AMP_mjlab walk 的权威值。
# ② `track_base_height` **std 0.2 → 0.3**（weight 1.5 不动）
#    由来：**V9H** 把核宽从 0.3 收紧到 0.2（`AMP_WALK_V9H_BASE_HEIGHT_STD`），
#    经 V32B→V33Base→V33S→T1a→T1a2→T1j→T1k 一路继承从未恢复。本臂首跑被 fail-fast 断言
#    当场抓出（`std = 0.2`）⇒ 用户"放宽 std"的判断成立。
#    量化：屈膝站姿（高度降 ~5cm）在 std0.2 下要付 **0.091/步**（= AMP 风格上限 0.025 的
#    3.6 倍 ⇒ 几乎不可能赢）；std0.3 降到 **0.041/步** ⇒ 这正是"坚持不曲膝"的价格来源。
# ⚠️ `track_lin_vel_xy_exp` 的 **std 0.5 不动**（改它 = 换尺子，曲线不可比）。
# 后置（待结果后逐项上）：`track_ang_vel_z_exp` 1.0→1.2、`action_rate_l2 −0.05→−0.01`、
#   `body_ang_vel_xy 0→+0.7`、`track_base_height` std→0.4。
# 起点 = T1a2 的 model_7000.pt（与 T1j/T1i/T1k 同起点）；算法仍用 T1i 的 λ=0.975。
from .amp_walk_t1l_env_cfg import (
    G1AmpWalkT1lEnvCfg,
    G1AmpWalkT1lEnvCfg_PLAY,
)
from .omni_loco_env_cfg import G1OmniLocoEnvCfg, G1OmniLocoEnvCfg_PLAY, G1OmniLocoTeacherEnv

for task_id, env_cfg in (
    ("Instinct-Unified-OmniLoco-G1-v0", G1OmniLocoEnvCfg),
    ("Instinct-Unified-OmniLoco-G1-Play-v0", G1OmniLocoEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1OmniLocoTeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.omni_loco_ppo_cfg:G1OmniLocoPPORunnerCfg",
        },
    )

# --- AMP 行走特权教师（PLAN_amp_walk.md）-------------------------------------
# 与 OmniLoco 的唯一接口差异：actor 观测 623 -> 626（新增 3 维 base_lin_vel）。
# 沿用同一套 fail-fast 契约校验器（G1AmpWalkTeacherEnv 只覆盖期望值）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-G1-v0", G1AmpWalkEnvCfg),
    ("Instinct-Amp-Walk-G1-Play-v0", G1AmpWalkEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkTeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_ppo_cfg:G1AmpWalkPPORunnerCfg",
        },
    )

# --- AMP 行走特权教师 v2：奖励集换成 AMP_mjlab 的 12 项配方 -------------------
# 动机与逐项对照见 `amp_walk_v2_env_cfg.py` 的模块 docstring。
# ⚠️ 环境类 **仍是 `G1AmpWalkTeacherEnv`**：动作/观测/时序/执行器契约与 v1 逐字相同，
#    因此 checkpoint、ONNX 导出、MuJoCo 回放、**蒸馏 teacher 侧接口**全部保持兼容；
#    agent cfg 也复用同一份（v2 只改奖励，不改策略/算法结构）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V2-G1-v0", G1AmpWalkV2EnvCfg),
    ("Instinct-Amp-Walk-V2-G1-Play-v0", G1AmpWalkV2EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkTeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_ppo_cfg:G1AmpWalkPPORunnerCfg",
        },
    )

# --- AMP 行走特权教师 v3：v2 奖励配方 + action_rate_l2 −0.02 + init_noise_std 0.3 ---
# 动机、实测证据（σ 0.766→1.580）与「从零训练」的理由见
# `amp_walk_v3_env_cfg.py` 与 `agents/amp_walk_v3_ppo_cfg.py` 的模块 docstring。
# ⚠️ 环境类仍是 `G1AmpWalkTeacherEnv`（契约与 v1/v2 逐字相同）；
#    唯一区别是 **agent cfg 换成 v3 那一份**（它只改 init_noise_std/clip_min_std）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V3-G1-v0", G1AmpWalkV3EnvCfg),
    ("Instinct-Amp-Walk-V3-G1-Play-v0", G1AmpWalkV3EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        # ⚠️ 用 v3 自己的校验器：指令范围与保持时长与 v1 不同，其余期望值逐字相同
        entry_point=G1AmpWalkV3TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v3_ppo_cfg:G1AmpWalkV3PPORunnerCfg",
        },
    )

# --- AMP 行走特权教师 v4：AMP 侧两项修正（镜像损失 + 站立关风格）---------------
# 环境 / 观测 / 动作 / 奖励集（14 项）/ σ 起点 **全部继承 v3**；
# 只换 agent cfg。动机与两边差异见 `agents/amp_walk_v4_ppo_cfg.py` 的模块 docstring。
# ⚠️ teacher 侧仍用 `G1AmpWalkV3TeacherEnv`（环境完全相同，契验期望值一致）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V4-G1-v0", G1AmpWalkV4EnvCfg),
    ("Instinct-Amp-Walk-V4-G1-Play-v0", G1AmpWalkV4EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV3TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v4_ppo_cfg:G1AmpWalkV4PPORunnerCfg",
        },
    )

# --- AMP 行走特权教师 v5：风格奖励形式 quad -> log ---------------------------
# ⚠️ 判别器训练逻辑一个字没动，只改「判别器输出 → 奖励」那一步的函数形式，
#    所以这属于**奖励构成**改动，可以 `--resume` 续训。动机与实测证据见
#    `agents/amp_walk_v5_ppo_cfg.py` 的模块 docstring。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V5-G1-v0", G1AmpWalkV5EnvCfg),
    ("Instinct-Amp-Walk-V5-G1-Play-v0", G1AmpWalkV5EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV3TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v5_ppo_cfg:G1AmpWalkV5PPORunnerCfg",
        },
    )

# --- AMP 行走特权教师 v6：v5 + 恢复基类 `joint_deviation_arms`（-0.1）----------
# ⭐ 用户实测：无论静止还是移动，**肩/肘高举且接近限位**
#   （`right_shoulder_roll` 偏差 2.03 rad ≈ 116°、双侧 elbow ≈ 85°）。
#   原因是 v2 迁移 AMP_mjlab 12 项配方时把这一项整项删掉了 ⇒ 全任务奖励里
#   **没有任何一项约束手臂**。环境侧改动（奖励集 14 → 15 项），算法侧未动。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V6-G1-v0", G1AmpWalkV6EnvCfg),
    ("Instinct-Amp-Walk-V6-G1-Play-v0", G1AmpWalkV6EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV3TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v6_ppo_cfg:G1AmpWalkV6PPORunnerCfg",
        },
    )

# --- AMP 行走特权教师 v7：收敛性修复（σ 0.3→0.2 + AMP 系数 0.1→0.05）------------
# ⭐ **从零训练**（不带 `--resume`）。实测依据（逐关节监控 `joint_error_rmse`，@V6-13840）：
#   `shoulder_roll` 偏差 2.038 rad ⇒ 实际角 2.238（硬限位 2.252）⚠️ 卡死
#   `elbow`         偏差 1.490 rad ⇒ 实际角 2.090（硬限位 2.094）⚠️ 卡死
#   且 v1 起就存在 ⇒ 已在吸引子里，续训拉不回来。
# AMP 减半：实测 AMP 占真实回报 ~80%，减半后约 66%。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V7-G1-v0", G1AmpWalkV7EnvCfg),
    ("Instinct-Amp-Walk-V7-G1-Play-v0", G1AmpWalkV7EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV3TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v7_ppo_cfg:G1AmpWalkV7PPORunnerCfg",
        },
    )

# --- AMP 行走特权教师 v8：风格奖励换成 LLP 论文形式 r = -log(1 - quad(d)) ------
# ⚠️ 父类用 `discriminator_reward_type="wasserstein"`（唯一原样返回 d 的分支），
#    塑形在我们自己的桥接层做 ⇒ **不改上游 `instinct_rl`**；判别器自身损失未动
#    ⇒ 属于**奖励构成**改动 ⇒ 可续训。
# ⚠️ `-log(1-D)` 在 d=1 处会 +inf（D 恰好=1，而 MSELoss 参考侧目标就是 +1）
#    ⇒ 实现里把 D 夹到 1-1e-6，奖励上界 ≈ 13.8。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V8-G1-v0", G1AmpWalkV8EnvCfg),
    ("Instinct-Amp-Walk-V8-G1-Play-v0", G1AmpWalkV8EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV3TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v8_ppo_cfg:G1AmpWalkV8PPORunnerCfg",
        },
    )

# --- AMP 行走特权教师 v9（MVP）：279(前移 1.5m) 高程图 + MVP 指令 + σ 上限 ----------
# 相对 v8 的四项改动见 `amp_walk_v9_env_cfg.py` / `agents/amp_walk_v9_ppo_cfg.py`：
#   1. height_scan 3.0×0.8（279）+ 中心前移 1.5 m ⇒ 落点 x∈[0,3]，actor obs 626→378
#   2. default pose / offset —— 核查后本来就与越障一致，不动
#   3. init_noise_std 0.2→0.25（AMP_mjlab 等效）+ **clip_max_std=0.4**（防 σ 膨胀）
#   4. 指令收敛为原地+前向 (0,0.7)；数据集只留 2 个纯前向 clip
# ⚠️ 契约期望值不同（378/279 条射线/(0,0.7)）⇒ 必须用 v9 自己的校验器类做 entry_point。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V9-G1-v0", G1AmpWalkV9EnvCfg),
    ("Instinct-Amp-Walk-V9-G1-Play-v0", G1AmpWalkV9EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV9TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v9_ppo_cfg:G1AmpWalkV9PPORunnerCfg",
        },
    )

# --- AMP 行走 v9b（消融）：**v9 − 镜像损失**，环境/契约/超参全部相同 ------------------
# 目的：验证"镜像损失把策略推向左右对称的跳跃（平凡解），而对行走收税更重"这一假设。
# 单变量 ⇒ 环境侧必须逐字相同（同一 `env_cfg`、同一 teacher 校验器）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V9b-G1-v0", G1AmpWalkV9EnvCfg),
    ("Instinct-Amp-Walk-V9b-G1-Play-v0", G1AmpWalkV9EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV9TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v9b_ppo_cfg:G1AmpWalkV9bPPORunnerCfg",
        },
    )

# --- AMP 行走 v9c：v9b（无镜像）+ 降 σ（entropy 0.008→0.0025 / 上限 0.4→0.3）
#     + 抬 AMP 权重（c 0.05→0.25）+ 手臂偏差归零（−0.1→0）-------------------------
# 依据（v9 实测）：① `model_7500.pt` 里 **23/29 个关节的 σ 恰好 = 0.4** ⇒ 上限硬卡着，
#   σ 膨胀是"没学到东西"的症状而不是可调旋钮；② AMP 项只占训练回报 ~11%（0.0031/步
#   vs 任务 0.0254/步）⇒ 风格几乎不影响步态；③ `joint_deviation_arms` −2.76/回合（11.6%）
#   而 `dof_pos_limits` 只有 −0.0127 ⇒ 限位不再被顶，安全网可以松开。
# 观测/指令/数据集与 v9 逐字相同 ⇒ 直接复用 v9 的 teacher 校验器。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V9c-G1-v0", G1AmpWalkV9cEnvCfg),
    ("Instinct-Amp-Walk-V9c-G1-Play-v0", G1AmpWalkV9cEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV9TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v9c_ppo_cfg:G1AmpWalkV9cPPORunnerCfg",
        },
    )

# --- AMP 行走 v9d：**v9c + 恢复镜像损失**（单变量） ---------------------------
# 依据（v9b/v9c 实测，~430 轮）：关掉镜像损失后策略掉进不对称局部最优 ——
#   v9b 单脚跳 + 姿态不平衡且不会移动（σ 同样顶在 0.39）；v9c 单脚跳会移动（σ 收敛在 0.21）。
# ⇒ 镜像损失（左右等变性）不能移除；而 v9c 的 σ 参数（entropy 0.0025 / 上限 0.3 / 初值 0.2）
#   与 AMP 权重（c=0.25，实测 amp_d −0.79→−0.45、r_amp 0.23→0.65）确实有帮助，全部保留。
# 观测/指令/数据集与 v9 逐字相同 ⇒ 直接复用 v9 的 teacher 校验器。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V9d-G1-v0", G1AmpWalkV9dEnvCfg),
    ("Instinct-Amp-Walk-V9d-G1-Play-v0", G1AmpWalkV9dEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV9TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v9d_ppo_cfg:G1AmpWalkV9dPPORunnerCfg",
        },
    )

# --- AMP 行走 v9e：v9d（镜像 ON + v9c 的 σ/AMP）+ 迈步奖励 + 减少原地指令 ----------
# ① 复原 `feet_air_time`（奖励集 **15 → 16 项**）：奖励"恰好单脚支撑"的持续时间，
#    双脚同时着地/离地的对称跳跃拿 **0 分** ⇒ 修补"没有任何一项奖励迈步"的缺口。
#    （该项当初在 v2 采用 AMP_mjlab 12 项配方时被整项删掉，这里是**复原**。）
# ② 原地（精确 0）指令占比减半：`zero_frac_in_low` 0.6 → 0.3（18% → 9%）。
#    这些 env 里 `track_lin_vel_xy_exp` 变成"别动才给分"、而 `track_default_pose_on_stand`
#    的门控 ≈ 1（完全生效）⇒ 等于每 5 个 env 就有 1 个在教"站住不动"。
# 契约校验只断言 `rel_standing_envs == 0.0`（不动），`zero_frac_in_low` **不参与断言** ⇒ 安全。
# 观测/动作/指令范围/数据集与 v9 逐字相同 ⇒ 直接复用 v9 的 teacher 校验器。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V9e-G1-v0", G1AmpWalkV9eEnvCfg),
    ("Instinct-Amp-Walk-V9e-G1-Play-v0", G1AmpWalkV9eEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV9TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v9e_ppo_cfg:G1AmpWalkV9ePPORunnerCfg",
        },
    )

# --- AMP 行走 v9f：v9e + **去掉自适应学习率** + 放大探索幅度 ----------------------
# 依据（v9e 实测 @3k 轮）：σ=0.20、`feet_air_time`≈0.0008（基本没有单脚支撑）、
# `vx_*_rmse` 与 v9c 几乎一致 ⇒ 仍然"站着不动"。用户假设：学习率偏低 / 探索幅度不足。
# 代码核对支持该假设：`schedule="adaptive"` 会让 LR 每步 ×/÷1.5（夹 [1e-5, 1e-2]，
# 且**日志不记录**）；而 KL ≈ (Δμ/σ)²·dims/2 ⇒ **σ 越小越容易被压 LR** —— 正好解释
# "高 σ 的 v9b 会动、低 σ 的 v9c/v9e 不动"。
#  ① `algorithm.schedule = "fixed"`（LR 稳定 1e-3）
#  ② `init_noise_std 0.2 → 0.3` + `clip_max_std 0.3 → 0.4`（走路需大幅度、跳只需小角度）
# 诊断：本版起 `Mean step effective_lr` / `action_abs_mean` / `action_abs_max` 可用。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V9f-G1-v0", G1AmpWalkV9fEnvCfg),
    ("Instinct-Amp-Walk-V9f-G1-Play-v0", G1AmpWalkV9fEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV9TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v9f_ppo_cfg:G1AmpWalkV9fPPORunnerCfg",
        },
    )

# --- AMP 行走 v9g：**奖励重平衡 + 风格项主导**（LR/σ 两条假设已被 v9f 数据排除）----
# v9f @4360 轮：`effective_lr=0.0010`（fixed，正常）、`action_abs_mean=0.72 rad`（在大幅发力，
# 不是“不敢动”）、σ 自己从 0.3 降到 0.23 ⇒ 学习率/探索幅度两条假设都不成立。
# 而奖励账目精确闭合（各 /s ×20 = 44.6 ≈ Mean reward 45.0），其中：
#   track_base_height +23.7/回合（53%）、track_ang_vel_z +11.1、body_ang_vel_xy +10.8
#   —— 这三项**站着不动就能拿**，合计 45.6 > 整个回报；
#   而唯一要求位移的 track_lin_vel_xy 只有 +17.4；`feet_air_time` = 0.0026/s（从未单脚支撑）。
# ⇒ 策略没“学不会”，它只是把奖励最优解（站着不动）找得很准。
# ① 压白拿项：base_height 1.5→0.5 / ang_vel_z 1.2→0.2 / body_ang_vel_xy 0.7→0.2 /
#    default_pose_on_stand −1.0→0
# ② 抬功绩项：track_lin_vel 1.5→3.0 / feet_air_time 1.0→2.0 / feet_slide −0.25→−0.1
# ③ 风格权重 ×4：style_lerp 0.75→0.5、discriminator_reward_coef 0.25→0.5
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V9g-G1-v0", G1AmpWalkV9gEnvCfg),
    ("Instinct-Amp-Walk-V9g-G1-Play-v0", G1AmpWalkV9gEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV9TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v9g_ppo_cfg:G1AmpWalkV9gPPORunnerCfg",
        },
    )

# --- AMP 行走 v9h：**抗跳 + 恢复平衡刹车 + 补回基类被删项**（16 → 20 项）----------
# V9g 复盘：① “95% 摔倒”是**断崖窗口的快照**，V9g 反复出现 `rew 37 → 0.31(len=79)` 的
# 崩溃-恢复振荡（崩溃时 `amp_style_std` 0.10→0.19/0.24、`d` −0.39→−0.51、`|a|` 0.48→0.58），
# 好轮次其实能到 rew 45/len 795；② 我把三个平衡刹车砍过头（ang_vel_z 1.2→0.2、
# body_ang_vel_xy 0.7→0.2、base_height 1.5→0.5）而把 lin_vel 抬到 3.0 ⇒ 跳跃性价比是调出来的；
# ③ 风格权重 ×4 实测**无效**（amp_d 只动 0.026）。
# ⇒ V9h：恢复 ang_vel_z 0.8 / body_ang_vel_xy 0.5 / base_height 1.0(std 0.2)、
#    default_pose_on_stand −0.3、保留 lin_vel 3.0 / feet_air_time 2.0、风格侧回退 0.75/0.25，
#    并补回 4 项基类项：lin_vel_z_l2 −0.2（⭐ 躯干竖直速度，直接给跳定价）、
#    dof_torques_l2 −4e-6、joint_deviation_hip −0.1、joint_deviation_knee −0.05。
#    刻意不恢复 `stand_still`（与 default_pose_on_stand 重复惩罚）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V9h-G1-v0", G1AmpWalkV9hEnvCfg),
    ("Instinct-Amp-Walk-V9h-G1-Play-v0", G1AmpWalkV9hEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV9TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v9h_ppo_cfg:G1AmpWalkV9hPPORunnerCfg",
        },
    )

# --- AMP 行走 v9i：**唯一变量 = 探索量级**（奖励集与 v9h 逐字相同）------------------
# 参照：仓库里**不含 AMP 的行走配方** `G1OmniLocoPPORunnerCfg` —— 同机器人、同 `action_scale=1.0`、
# 同时序与指令契约 —— 用的是 `init_noise_std=1.0`（±57°）+ `entropy_coef=0.008`，能从零学会走。
# 而我们 V9h 只有 0.3（±17°）+ 0.0025 ⇒ **比“能学会走路的配方”少 1.7~3.3 倍探索幅度**。
# V9h@2050 的实测也支持这个方向：σ 稳在 0.31（不再自降）、`feet_air_time` 单调爬到 0.0384，
# 而 `gap = d_ref − d_actor = 1.18` 说明判别器健康、瓶颈在“策略没探索过去”。
# ⇒ V9i：`init_noise_std 0.5` / `clip_max_std 0.8` / `entropy_coef 0.005`（+`clip_min_std 0.025`），
#    其余（20 项奖励、`lerp 0.75` / `ampc 0.25`、fixed LR、镜像 ON）与 V9h 逐字相同。
# ⚠️ 不要与 V9h 并行跑（CPU 瓶颈，会两边减半）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V9i-G1-v0", G1AmpWalkV9iEnvCfg),
    ("Instinct-Amp-Walk-V9i-G1-Play-v0", G1AmpWalkV9iEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV9TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v9i_ppo_cfg:G1AmpWalkV9iPPORunnerCfg",
        },
    )

# --- AMP 行走 v9j：**堵住“单脚吊腿”漏洞**（20 → 21 项）-------------------------------
# V9h@4770 实测：`lin_vel_z_l2` 只罚**竖直**速度（该项仅 −0.0034/s ⇒ 几乎无竖直弹跳），
# 而 `feet_air_time` 只要“恰好一只脚着地”就持续给满（不区分踩了 0.3 s 还是 10 s）
# ⇒ 策略退化成“**单脚侧身跳着走**”：一侧脚一直踩地、另一侧吊着、靠上身侧倾推进。
# ⚠️ **镜像损失拦不住它**：它约束 `π(M(o)) ≈ M(π(o))`（策略**函数**对称性），而单侧步态
# 访问到的状态其镜像态**从未被访问**（off-distribution）⇒ 约束可在“没去过的状态”上被满足
# （实测镜像损失仅 0.0368，步态却完全单侧）。
# ⇒ V9j 新增 `feet_swing_time_penalty`（`rewards_amp.py`）：
#     `cost = Σ_f clamp(air_time_f − 0.6 s, 0) × 1[‖cmd_xy‖>0.1]`，权重 −0.5。
#     走路摆动相 0.2~0.5 s ⇒ **恒为 0**；单脚吊腿腾空 1 s ⇒ −0.5/步（−25/s），
#     **压过** `feet_air_time` 最多 +1.0/步的支付 ⇒ “把脚放下来”变成唯一划算的走法。
#     其余 20 项与算法旋钮（σ0.3/上限0.4/ent0.0025/fixedLR/lerp0.75/ampc0.25/镜像）全部不动。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V9j-G1-v0", G1AmpWalkV9jEnvCfg),
    ("Instinct-Amp-Walk-V9j-G1-Play-v0", G1AmpWalkV9jEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV9TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v9j_ppo_cfg:G1AmpWalkV9jPPORunnerCfg",
        },
    )

# --- AMP 行走 v10：**一次把 4 个概念变量全上**（探索 / 数据层镜像增广 / 指令扩展 / 交替约束）
# ① 探索量级：`init_noise_std 0.6` / `clip_max_std 0.7` / `entropy_coef 0.005` / `clip_min_std 0.03`
#    （参照：仓库里不含 AMP 的行走配方 = 1.0 / 无上限 / 0.008；用户实测 σ=1 那版略有走路模态，
#      但无上限时 σ 会失控到 1.04 rad ⇒ 本版给硬天花板 0.7）
# ② 数据层镜像增广 ON（`symmetric_augmentation_*`）——直接治“镜像态从未被访问”：
#    判别器的 expert 分布变左右对称 ⇒ 单侧步态不再“像人”；映射用
#    `build_joint_mirror(OMNI_LOCO_JOINT_ORDER)` 现算（与镜像损失同一套已验证代码）
# ③ 指令扩展：(0,0.7) × (−0.35,0.35) × (−0.6,0.6) ⇒ 左前/右前对角 + 转向都要会（在数据集包络内）
# ④ 交替支撑约束：沿用 V9j 的 `feet_swing_time_penalty −0.5`（共 21 项）
# ⚠️ 放弃单变量归因；启动后必看 `[AMP-WALK INFO] 数据层镜像增广=ON；joint 顺序 …：一致`。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V10-G1-v0", G1AmpWalkV10EnvCfg),
    ("Instinct-Amp-Walk-V10-G1-Play-v0", G1AmpWalkV10EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV10TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v10_ppo_cfg:G1AmpWalkV10PPORunnerCfg",
        },
    )

# --- AMP 行走 v10b：**数据集收敛到 2 clip + 三档指令**（用户 2026-09-29 指定）---------
# ⚠️ 重要发现：实际生效的一直是 `amp_walk_motion_selection.yaml` 的 **17 个 clip**
# （没有任何版本 cfg 覆盖过它），而 v9 MVP 文档写着“只留 1 个 clip” ⇒ **文档与实际不符**。
# 17 种形态（后退×2、侧移×2、原地转身×2、慢跑×3、弧线×2、步态切换×1）把“走”这一模态的
# 密度摊薄 ⇒ 风格奖励对“像不像走”几乎没分辨力。
# ⇒ V10b：数据集 = **2 clip**（`walk_forward_loop_002__A022` 直行 + `arc_walk_left_loop_001__A029` 弧线）；
#    指令 = 三档：`prob_low 0.30`（静止/低速，‖v‖≤0.1、wz=0）+ `prob_lin 0.35`（纯前进，vx∈(0.4,0.7)、vy=wz=0）
#    + 余 0.35（前向斜向/弧线，vx∈(0,0.7)、vy∈±0.35、wz∈±0.4）；**删掉** prob_side / prob_back。
#    其余（探索 σ0.6/上限0.7/ent0.005、镜像增广、21 项奖励含交替支撑约束）与 V10 逐字相同。
#    同时装上“门控口径”探针：桥接层 `cmd_slice_vx/vy/wz/speed` vs 环境层 `Step/cmd_speed_mean`。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V10b-G1-v0", G1AmpWalkV10bEnvCfg),
    ("Instinct-Amp-Walk-V10b-G1-Play-v0", G1AmpWalkV10bEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV10bTeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v10b_ppo_cfg:G1AmpWalkV10bPPORunnerCfg",
        },
    )

# --- AMP 行走 v11：**从 V10b 的走路 ckpt 热启动**，主打“保走姿/去碎步/让 AMP 说得上话”-----
# V10b@7930 复盘：✅ 走路成功（中位回合跑满 1000 步、base_contact 0.375、vx_high RMSE 0.212），
#   ❌ 但是“高频小碎步”：action_rate_l2 恶化 4×（步频代理量）、dof_acc_l2 3×、feet_slide 2.5×、
#      feet_air_time 0.041→0.019（单支撑更短）、amp_d −0.56→−0.78（更不像参考）。
# ⇒ V11 三处步态形状改动：feet_air_time 2.0→8.0（threshold 0.5→0.6）、
#   action_rate_l2 −0.02→−0.06、dof_acc_l2 −2.5e-7→−1.0e-6；
#   并把有效风格权重 (1−λ)c 0.0625→0.25（λ0.5/c0.5，重测上次在“17clip+坏门控”下无效的结论）；
#   σ 上限 0.7→0.6（用户指定）。
# ⚠️ 续训：init_noise_std 不生效（σ 用 ckpt 学到的 0.58）；回报先掉再涨是预期。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V11-G1-v0", G1AmpWalkV11EnvCfg),
    ("Instinct-Amp-Walk-V11-G1-Play-v0", G1AmpWalkV11EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV11TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v11_ppo_cfg:G1AmpWalkV11PPORunnerCfg",
        },
    )

# --- AMP 行走 v12：**判别器输入换成论文的两帧 (s_t, s_{t+1})** + 站姿权重 ------------------
# ① `AMP_STATE_HISTORY` 10 → **2**（`amp_policy`/`amp_reference` 670 → 134 维）：
#    LLP 原论文的判别器输入是**状态转移 (st, st+1)**。10 帧（0.2 s）下判别器靠"姿态轮廓/
#    抖动统计"就能分辨 policy/reference ⇒ 没压力学"单步转移像不像人"；与我们观测到的
#    `amp_d` 越训越负（−0.56→−0.78）、`r_amp` 0.50→0.24（够不着而非给不起）自洽。
# ② `track_default_pose_on_stand` −0.3 → −1.0（用户：静止要更稳；门控在低速段才生效）。
# 其余（21 项奖励 / 三档指令 / 2-clip / 镜像增益+损失 / σ 上限 0.6 / λ0.5 c0.5）逐字同 V11。
# ⚠️ 判别器**在** checkpoint 里（`wasabi.py` override 了 `state_dict`/`load_state_dict`）⇒
#    换输入维度直接 `--resume` 会 `size mismatch ... [1024,670] vs [1024,134]` 启动即崩（已验）。
#    V12/V13 用 `ckpt_manipulator = instinctlab.utils.ckpt_manipulators:drop_amp_discriminator`
#    删掉那两个键（Wasabi 对键缺失只打 Warning）⇒ actor/critic 照常热启动、判别器从零重学。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V12-G1-v0", G1AmpWalkV12EnvCfg),
    ("Instinct-Amp-Walk-V12-G1-Play-v0", G1AmpWalkV12EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV12TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v12_ppo_cfg:G1AmpWalkV12PPORunnerCfg",
        },
    )

# --- AMP 行走 v13：**全量测试**（指令集向侧向/后向拓展 + 补进对应数据集）----------------
# 指令四档（用户 2026-09-30 指定）：
#   静止/低速 0.20（‖v‖≤0.1，其中 30% 精确 (0,0,0)）
#   前后向     0.25+0.10（vy=0、ωz=0；前向 0.5~0.9 / 均匀 vx∈(−0.5,1.0) / 后退 (−0.5,−0.3)）
#   纯侧向     0.25（vx=0、vy∈(−0.5,0.5)、ωz=0 —— 新增 `side_ang_vel_range` 钉零）
#   斜向/转向  余量 0.20（(vx,vy,ωz) 三轴全随机）
# 数据集 2 → **5 clip**（直行 + 弧线 + 侧移×2 + 后退；`walk_backward_loop_003__A143` 不存在，
#   按用户指示改用 `walk_backward_loop_001__A022`）。
# 奖励/算法**逐字沿用 V12**（2 帧判别器、track_default_pose_on_stand=−1）⇒ 单变量：指令+数据集。
# 侧移“交叉跨步”与 feet_crossing / feet_lateral_width 不冲突（两者都有 gate_lateral=|vx|/‖v‖）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V13-G1-v0", G1AmpWalkV13EnvCfg),
    ("Instinct-Amp-Walk-V13-G1-Play-v0", G1AmpWalkV13EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV13TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v13_ppo_cfg:G1AmpWalkV13PPORunnerCfg",
        },
    )

# --- AMP 行走 v14：**V12 退化修正版** + 腰部限位软边距惩罚 --------------------------
# V11→V12 同口径对照（2026-09-30）：任务回报 27.73→9.28/回合（其中 −10.2/回合直接来自
#   `track_default_pose_on_stand −0.3→−1.0`）、feet_air_time 0.160→0.099、
#   vx_mid RMSE 0.243→0.323、action_rate_l2 −0.45→−0.75；但 `amp_d` −0.675→−0.407、
#   `r_amp` 0.362→0.716 ⇒ **2 帧判别器有效，代价是任务被牺牲**。
# ⇒ V14：① 站姿权重回退到 −0.3（保留 2 帧判别器）；
#        ② `dof_pos_limits` 换软边距版（踝 → 全关节，margin 0.15 rad、power 2、weight −1.5）
#           —— 治 "`waist_pitch_joint` 长期顶在 ±0.52"（此前腰/髋/膝根本没被罚）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V14-G1-v0", G1AmpWalkV14EnvCfg),
    ("Instinct-Amp-Walk-V14-G1-Play-v0", G1AmpWalkV14EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV14TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v14_ppo_cfg:G1AmpWalkV14PPORunnerCfg",
        },
    )

# --- AMP 行走 v15：**风格分量 ×4 实验**（环境/奖励逐字沿用 V14）------------------------
# 动机：V14 实测 `Step/amp_style_reward` 5400 轮只动 1.7%（`amp_d` 从 450 轮起钉在 −0.47）
#   ⇒ 风格项**饱和/平坦**，不是“占比小”（它 0.0375/步 > 任务净额 0.0314/步）。
# ⇒ 唯一变量：`discriminator_reward_coef` 0.25 → **1.0**（风格入账 (1−λ)c：0.0625→0.25，×4；
#   λ=0.75 不动 ⇒ 任务权重不变）。看 `amp_style_reward`/`amp_d` 的变化率是否恢复。
# 起点：热启动自 V11 的 model_9000.pt（会走路的策略；V14 已收敛到单脚跳，不能从它起）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V15-G1-v0", G1AmpWalkV15EnvCfg),
    ("Instinct-Amp-Walk-V15-G1-Play-v0", G1AmpWalkV15EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV15TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v15_ppo_cfg:G1AmpWalkV15PPORunnerCfg",
        },
    )

# --- AMP 行走 v16：**帧数消融 —— 只把判别器输入从 2 帧回退到 10 帧** --------------------
# V11（10 帧，会走）↔ V15（2 帧，10k 轮退化）的 AMP 侧差异只有两项：**帧数** 与
#   **判别器是否预训练**。V16 = V15 逐字沿用（同环境/同奖励/同 λ=0.75 c=1.0、同起点
#   V11 model_9000.pt、同"丢弃判别器从零重学"）⇒ **唯一差异 = 2 帧(134) → 10 帧(670)**。
# ⚠️ V16 的输入 670 维其实与 V11 ckpt 里的判别器同形，但仍保留 ckpt_manipulator 丢弃它，
#    否则会引入第二个变量（判别器预训练与否）。
# 判读：250 轮窗口并排 V15/V16 的 `amp_d`/`amp_style_reward` 斜率、`Step/tilt_deg_*`、
#   `action_rate_l2`、`feet_air_time`、`Loss/discriminator_*`（工具 `cloud_amp_compare.py`）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V16-G1-v0", G1AmpWalkV16EnvCfg),
    ("Instinct-Amp-Walk-V16-G1-Play-v0", G1AmpWalkV16EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV16TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v16_ppo_cfg:G1AmpWalkV16PPORunnerCfg",
        },
    )

# --- AMP 行走 v17：**从零学人类步态**（基准 V16；改两项：直立惩罚 ×2 + air-time 减半）--------------
# V15(2 帧) ↔ V16(10 帧) 消融结论：10 帧让判别器分辨力 ×2.9、行为全面更像人（倾角 30.3°→28.0°、
#   步幅 0.104→0.241、抖动 −0.568→−0.356、跟踪 1.302→1.447），**但倾角仍在恶化**
#   （V16 从 22.7° 爬到 28.0°，`tilt_gt30_frac` 仍 34%）⇒ 本版：
#   ① `flat_orientation_l2` −1 → **−2**（28° 时每步收费 0.127 → 0.254，给“站直”加价）；
#   ② `feet_air_time` 8.0 → **4.0（减半）**（降“单脚跳刷分”收益；V14 从零曾因此收敛成跳）；
#   ③ 其余（10 帧判别器 / 其余 19 项奖励 / 三档指令 / 2-clip / 镜像 / σ0.6 / λ0.75 c1.0）逐字不动。
# ⚠️ 与 V16 的另一处差异是**起点**（V16 热启动 V11 model_9000，本版**从零**）⇒ 不能直接比“风格好坏”。
# ⚠️ 早期止损：权重减半后 `Episode_Reward/feet_air_time` 的健康区间也减半（**0.05~0.15**）；
#   **≥0.3** 即疑似单脚跳（陷阱来自核函数形状而非权重）⇒ 则应停掉，V18 改“落地触发、每次迈步只付一次”。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V17-G1-v0", G1AmpWalkV17EnvCfg),
    ("Instinct-Amp-Walk-V17-G1-Play-v0", G1AmpWalkV17EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV17TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v17_ppo_cfg:G1AmpWalkV17PPORunnerCfg",
        },
    )

# --- AMP 行走 v18：**步态目标带 + 抖动加大 + 方案 B（5 clip / 全向指令）**----------------
# 基准 V17（从零 / 10 帧判别器 / flat −2 / area_time 4.0）。四组改动（用户 2026-10-01 指定）：
#   ① `feet_air_time` → **目标时长带**（新函数 `feet_air_time_target_biped`）：单支撑相位
#      ∈ [0.4, 1.0] s 给满额、两侧线性衰竭 ⇒ **同时反碎步与反单脚拨腿/跳**；
#      （V17 实测旧版核值仅 0.005~0.009 s ⇒ “小幅高频跳走”的数据铁证）
#   ② `action_rate_l2` −0.06 → **−0.10**（V17 抖动一直在变大）；
#   ③ `feet_swing_time_penalty` 权重 → **0**（与①的上界重叠、会打架）；
#   ④ 数据集/指令 = **方案 B**（逐字取自 V13）：参考集 **5 clip**（直行/弧线/侧移×2/后退），
#      指令 **四档**（低 0.20 / 前后 0.25+后退 0.10 / 纯侧 0.25 / 斜向余 0.20），
#      `ranges=((-0.5,1.0),(-0.5,0.5),(-0.8,0.8))`。
#   ⭐ **三个跟踪项（lin_vel 3.0 / ang_vel 0.8 / base_height 1.0）一律不动**：方案 B 里
#      `ωz` 有真实指令 ⇒ `track_ang_vel_z_exp` 是有效激励（不像方案 A 那样退化成恒定白拿分）。
#   ⚠️ 合约：指令范围扩到全向 ⇒ `expected_command_ranges` 已同步（V13 当年漏了这一步）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V18-G1-v0", G1AmpWalkV18EnvCfg),
    ("Instinct-Amp-Walk-V18-G1-Play-v0", G1AmpWalkV18EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV18TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v18_ppo_cfg:G1AmpWalkV18PPORunnerCfg",
        },
    )

# --- AMP 行走 v19a：**让判别器“敢说话”**（AMP 三件套 + 风格奖励限幅）+ 步态带消死区 --------
# 基准 V18（10 帧 670 维 / log_quad + wasserstein / λ0.75 / c1.0 / 5 clip 全向指令），
# 数据集、指令、三个跟踪项、`flat −2`、`action_rate −0.10` **一律不动**。
#
# ① `discriminator_gradient_penalty_coef` 5.0 → **1.0**
#    （V18 实测：该分项占判别器损失 **51%**，且 `tolerance=0` ⇒ 惩罚任何输入梯度；
#     `mean‖∂d/∂x‖² = 0.042` ⇒ 判别器被练成一个极平的函数 ⇒ 风格项没有梯度）
# ② `discriminator_weight_decay_coef` 3.0e-4 → **1.0e-5**（占损失 8% → 0.3%）
# ③ `discriminator_logit_weight_decay_coef` 0.04 → **0.0**
#    ⚠️ 该项真实系数取的是 `logit_weight_decay_coef`（**不存在**）⇒ 落回默认 **1.0**，
#       配置里的 0.04 从未生效；它专罚输出层（实测 `‖w‖₂=0.125`，d 只能到 ±0.62）。
# ④ `amp_reward_eps` 1e-6 → **0.05**（`−log(1−D)` 上界 13.8 → 3.0，防判别器变强后风格项淹没任务项）
# ⑤ env 侧 `feet_air_time.soft` 0.3 → **0.4**（`d < 0.1 s` 的“值 0 且梯度 0”死区消失）
# 启动方式：**热启动 V18 的 checkpoint，但丢掉判别器**重学（`drop_amp_discriminator`）。
# ⚠️ 任务 id 的字母后缀**一律小写**（与 `V9b` / `V10b` 一致）：
#    `instinct/tools/pull_latest_onnx.sh` 是从 run 名里的 `G1AmpWalkV19a...`
#    反推 id 的（正则只认小写后缀），写成大写 `V19A` 会直接被它的预检拦下。
# ⚠️ 监测：`Step/amp_d` 是否**展宽**（从 ±0.62 走向 ±1）、`Loss/discriminator_gradient_penalty`、
#   `Step/amp_style_std`、`Episode_Reward/feet_air_time` 与 `Step/air_time_in_band_frac`
#   是否 >0，以及 `track_lin_vel_xy_exp` 是否守住。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V19a-G1-v0", G1AmpWalkV19AEnvCfg),
    ("Instinct-Amp-Walk-V19a-G1-Play-v0", G1AmpWalkV19AEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV19ATeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v19a_ppo_cfg:G1AmpWalkV19APPORunnerCfg",
        },
    )

# --- AMP 行走 v19b：**从零 + 步态带近零端加密（exp, τ=0.08）+ quad 下端放宽**-----------
# 基准 = V19a（10 帧 670 维 / log_quad + wasserstein / λ0.75 / c1.0 / GP1.0 / wd1e-5 /
#               logit_wd0 / eps0.05 / 5 clip 全向指令 / `flat −2` / `action_rate −0.10`）。
#
# ① **步态带下侧改为指数型**（`tau=0.08`）：在 `d=0.01 s`（小跳工作点）处 score 由
#    0.025 → **0.118**、梯度 2.5/s → **≈12.5/s**；0.35 s 后已≈1（不与 [0.4,1.0] 打架）。
#    理由：V18（`soft=0.3`）在 `d<0.1 s` 处**值=0 且梯度=0**；V19a 改成线性后虽有梯度
#    但太小 ⇒ `air_time_single_stance` 从 0.0096 爬到 0.0111 s 后**平台化**，而
#    `Episode_Reward/feet_air_time` 只拿到上限（2.4/步）的 **1.6%**（0.038/步）。
# ② `amp_quad_k` 0.25 → **0.125**：V19a 实测 `d` 展宽到 −0.81 后有 **5.0%** 样本越过
#    `d ≤ −1` ⇒ `style ≡ 0` **且 `dr/dd = 0`**（纯死区）；k=0.125 把下端推到 `d ≤ −1.83`，
#    同时把 `style_actor` 从 0.199 抬到 0.892（⇒ 风格项/步 0.049 → 0.223）。
#    ⚠️ k **不**改局部斜率 `dr/dd = 2/(1−d)`，只改水平与死区位置。
# ③ **从零训练**（起点也是变量）：V18（从零）与 V19a（热启 V18）**都**落在小跳模态
#    ⇒ “热启动”不是唯一原因，所以起点与定价形状一起换。
# 新探针：`Step/air_time_hop_frac`（`d < 0.1 s` 占比）—— 本轮主指标之一。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V19b-G1-v0", G1AmpWalkV19BEnvCfg),
    ("Instinct-Amp-Walk-V19b-G1-Play-v0", G1AmpWalkV19BEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV19BTeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v19b_ppo_cfg:G1AmpWalkV19BPPORunnerCfg",
        },
    )

# --- AMP 行走 v19c：**堵步态分红漏洞 + 抖动降权 + 站立姿态加强**（热启 V19b）-------------
# 基准 = V19b（从零 / 10 帧 / `log_quad`+`wasserstein` / `quad k=0.125` / GP1.0 / wd1e-5 /
#               logit_wd0 / eps0.05 / 步态带 `exp τ=0.08`）。**AMP 侧一个参数都不动。**
#
# ① `feet_air_time` 加**实测运动门控**（`motion_threshold=0.15`、`motion_ang_scale=0.5`）：
#    原命令门控只挡"指令≈0"，挡不住"指令 0.6 m/s 却不前进、原地左右脚交替抬放刷分红"
#    （用户实测观察到静止/慢速时也在原地交替站）。新探针 `Step/air_time_motion_gate_off_frac`。
# ② `action_rate_l2` −0.10 → **−0.06**（回 V17 值）：用户看 MuJoCo 判断动作幅度/频率正常
#    ⇒ 该项在过度收费（raw 已从 V18 的 5.6 升到 ~15 ⇒ 罚 −1.5/步，占正奖励 Σ的 60%）。
#    ⚠️ 副作用：会削弱防抖，必须盯 raw `action_rate`（= 值/0.06）是否失控。
# ③ `track_default_pose_on_stand` −0.3 → **−0.5**：该项只在指令≈ 0 时生效（`exp(−cmd/0.25)`），
#    内容 = `4·mean|q−q_def| + 0.5·mean|dq|` ⇒ 加大它能直接抑制"站立时抬脚/晃腿"。
# 起点：**热启动 V19b 最新 ckpt 并保留判别器**（V19b 的 `discriminator_loss ≈ 0.0055`、
#    `d≈−0.986/d_ref≈+0.985` 已几乎完全分开，本轮没改任何影响判别器的东西，丢掉等于白扔）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V19c-G1-v0", G1AmpWalkV19CEnvCfg),
    ("Instinct-Amp-Walk-V19c-G1-Play-v0", G1AmpWalkV19CEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV19CTeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v19c_ppo_cfg:G1AmpWalkV19CPPORunnerCfg",
        },
    )

# --- AMP 行走 v19d：**站立档改成负价**（把"原地踏步"从零价变负价）+ 新增分解探针 -------
# 基准 = V19c（运动门控 / ar −0.06 / 站立姿态 −0.5 / AMP 侧与 V19b 相同）。
# V19c@9700 逐项定位：站立 env（`cmd_speed ≤ 0.1`）下 "原地抬脚" 几乎无价 ——
#   `feet_air_time` **0**（`gate_cmd=0`）、AMP 风格 **0**（`disable_style_on_stand=True`，已实测）、
#   `track_lin_vel 3.0`（不动就满分）、唯一成本是 `track_default_pose_on_stand` ≈ −0.36/步
#   （实测：权重 −0.3→−0.5 后原始 cost 0.846→0.722 ⇒ **该项在工作**，但不足以改掉踏步）。
# ⇒ `feet_air_time` 新增 `stand_penalty = 0.5`：`stand = 1[‖v_cmd,xy‖+|ωz| ≤ 0.1]` 时
#    `reward −= 0.5 · min(d/good_lo, 1) · 1[单支撑]` ⇒ 罚额（×weight 4.0）：
#    0.08 s → −0.40/步、0.15 s → −0.75、0.30 s → −1.50、≥0.40 s → **−2.00**（站立正奖励 ≈3.5/步）。
# 新探针：`Step/air_time_stand_step_frac`（站立+真抬脚占比，主指标→ 0）、
#          `Step/air_time_stand_pen_mean`、`Step/stand_pose_{pos_err,vel_err,active}`。
# 起点：**热启动 V19c 最新 ckpt 并保留判别器**。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V19d-G1-v0", G1AmpWalkV19DEnvCfg),
    ("Instinct-Amp-Walk-V19d-G1-Play-v0", G1AmpWalkV19DEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV19DTeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v19d_ppo_cfg:G1AmpWalkV19DPPORunnerCfg",
        },
    )

# --- AMP 行走 v20：**判别器输入换成"刚体空间 195 维"（照抄 AMP_mjlab），2 帧，从零** -----
# 背景（2026-10-02 核查）：本仓库 V1~V19d 的判别器输入**恒为关节空间 5 项（67 维/帧）**，
#   `body_pos_b` 等字符串在 `.py` 里 **0 次命中**（只在 `PLAN_amp_walk.md` §3.2 记为"对方设计"）
#   ⇒ 当年"迁概念没迁内容"。而 mjlab 系（AMP_mjlab、绝大多数 muJoCo 系工程）用的是
#   13 刚体 × (pos3 + ori6 + lin3 + ang3) = **195 维/帧**。
# 为什么它更可读：刚体位姿 = 关节角的**非线性**函数（FK）、刚体速度 = `J(q)·q̇` 的**线性**
#   函数（雅可比）⇒ 判别器不必自己学 FK；这解释了"他们 2 帧够、我们 67 维要 8~15 帧"
#   （离线实测 h=1~2 时 AUC≈0.5，完全不可分 ⇒ `RESULTS_amp_frames_offline_study.md`）。
# 本轮**只换输入**：13 体（pelvis + 双髋 roll + 双膝 + 双踝 roll + 双肩 roll + 双肘 + 双腕）、
#   anchor = `torso_link`、6D 取旋转矩阵前两列、帧数 2 ⇒ 判别器输入 **390**；
#   奖励/指令/数据集/AMP 奖励口径（quad k=0.125、GP 1.0、wd 1e-5、eps 0.05、λ0.75、c1.0）
#   与 V19d **逐字相同**；起点 **从零**（维度变化，判别器权重不可复用）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V20-G1-v0", G1AmpWalkV20EnvCfg),
    ("Instinct-Amp-Walk-V20-G1-Play-v0", G1AmpWalkV20EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV20TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v20_ppo_cfg:G1AmpWalkV20PPORunnerCfg",
        },
    )

# --- AMP 行走 v21：**刚体空间 195 维 × 8 帧 = 1560**（在 V20 基础上只改帧数）---------------
# 离线研究 v2（`RESULTS_amp_frames_body_vs_joint.md`，同协议扫 h）结论：
#   * **h=2 不够**：body195@h2 的"时间结构类退化" AUC 仅 **0.625** ⇒ 解释 V20 为何退回
#     "高频小跳/原地踏步"模态（判别器 2000 轮就饱和、`style_std` 只有 V19b 的 0.59×）、MuJoCo 里不走；
#   * **h=8 是拐点**：AUC **0.890**（> legacy67@h10 的 0.838 = V19b 走出正常步态时的判别水平），
#     `style_std` 0.0791（是 legacy67@h10 的 5.3×）⇒ 风格项"敢说话"；
#   * h=10 稍好（0.910）但输入维 1950，比 h=8 多花 25% 算力 ⇒ 取 **h=8**。
# 相对 **V20** 只改 h（2→8）；相对 **V19b/V19d** 只改表征（关节 67×10 → 刚体 195×8）⇒
# 两个方向都是**单变量**。奖励/指令/数据集与 V19d **逐字不动**。起点：**从零**。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V21-G1-v0", G1AmpWalkV21EnvCfg),
    ("Instinct-Amp-Walk-V21-G1-Play-v0", G1AmpWalkV21EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV21TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v21_ppo_cfg:G1AmpWalkV21PPORunnerCfg",
        },
    )

# --- AMP 行走 v22：**V21 + 判别器输入白化**（单变量，2026-10-03）-------------------------
# 环境侧（奖励/指令/数据集/观测群 195×8）与 **V21 完全相同** —— 直接复用 V21 的 env cfg
# 类，不另建文件（避免抄错造成意外第二变量）；差异**全在算法侧**：
#   * `discriminator_kwargs` 接上 `EmpiricalNormalization(eps=0.01)`
#   * `amp_grad_pen_space = "normalized"`（GP 在归一化空间求，必须）
#   * `amp_normalizer_expert_only = True`（统计量只由 expert 侧累积）
# 依据（本地 `instinct/tools/test_amp_normalizer.py`，27/27 通过）：
#   * 白化让判别器的"可分性"显著提升（离线：AUC 0.850→0.927 @ h=8，`style_std` ×2.7）；
#   * 但**不改 GP 空间就会崩**：raw 空间 GP 项被 σ 下限放大 311×，判别器饱和后达 MSE 的 1.2e6 倍；
#   * 改到 normalized 空间后 GP 项回到与旧配置同量级 ⇒ `GP coef=1.0` 原样沿用。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V22-G1-v0", G1AmpWalkV21EnvCfg),
    ("Instinct-Amp-Walk-V22-G1-Play-v0", G1AmpWalkV21EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV21TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v22_ppo_cfg:G1AmpWalkV22PPORunnerCfg",
        },
    )

# --- AMP 行走 v23：**白化（AMP 风格）+ 奖励结构两项**（2026-10-03，用户指定合并）-------
# ① `feet_air_time`：删掉关闭率 <1% 的"实测瞬时幅值"门，改用**进度判据**
#    （方向投影 + EMA 低通，`progress = 1[EMA > κ‖v_cmd‖]`）；负价条件由 `stand`（指令≈0）
#    推广为 `no_progress`，同时覆盖"0 指令站立"与"有指令但不前进"两档。
# ② `track_default_pose_on_stand`：逐关节死区 + 内部权重（按 Σw 归一）+ weight 0.5→0.8。
#    依据：实测该项一直是"主要在管手臂"（臂 14 关节贡献 58%、腿 35%）。
# ③ 判别器输入白化 + GP 移到归一化空间（= V22），因三代判别器全线饱和（gap≈1.97/2）。
# ⚠️ 三项合并 ⇒ 归因靠各自探针（见 `amp_walk_v23_env_cfg.py` 文件头）。
# 起点：**从零**（判别器多了 normalizer buffer + 输入分布定义变了）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V23-G1-v0", G1AmpWalkV23EnvCfg),
    ("Instinct-Amp-Walk-V23-G1-Play-v0", G1AmpWalkV23EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV23TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v23_ppo_cfg:G1AmpWalkV23PPORunnerCfg",
        },
    )

# --- AMP 行走 v24：**V23 + `dof_acc_l2` 全关节**（2026-10-03，用户指定；权重不变）---------
# 加速度 L2 对频率是 **4 次方**（a∝A(2πf)² ⇒ 罚 a²∝A²f⁴）⇒ 对"高频小跳"的选择性是
# `action_rate_l2`（f²）的两个倍频程。AMP_mjlab 的 `joint_acc_l2 = −2.5e-7` **无 asset_cfg
# ⇒ 全关节**；我方从 V2 起只罚髋/膝，而小跳主要是**踝+腰**的运动 ⇒ 罚不到病灶。
# 历史：V11 已把权重 ×4（→ −1.0e-6），V12~V20 仍反复小跳 ⇒ "只加权重不扩关节"不够。
# 本轮唯一变量 = 关节集；权重 −1.0e-6 与 func 逐字不动。起点：**从零**（V23 已进入小跳模态）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V24-G1-v0", G1AmpWalkV24EnvCfg),
    ("Instinct-Amp-Walk-V24-G1-Play-v0", G1AmpWalkV24EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV24TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v24_ppo_cfg:G1AmpWalkV24PPORunnerCfg",
        },
    )

# --- AMP 行走 v26：**σ（动作噪声）单变量**（2026-10-04）-----------------------------
# V24 全部设置逐字不动（奖励/AMP/白化/`action_scale=1.0`），只在 algo 侧：
#   * `clip_max_std 0.6 → 0.20`（V24 的 σ 自然平衡在 0.572，cap 并未 binding）
#   * `clip_max_std_schedule = 300`（前 300 次 update 线性 0.57 → 0.20）
# 起点：**V24 model_8000.pt 热启动**（AMP 输入同形 ⇒ 判别器/优化器不丢）。
# 依据（Step 0 σ 扫描，同一份权重只缩放噪声）：
#   噪声税 −5.26/步（σ=0.569）→ −1.47（0.30）→ **−0.58（0.20）**→ −0.13（0.10）
#   步态在 σ=0.20 最优：hop 0.762→**0.626**、in_band 0.385→**0.443**、
#   `single_stance` 0.063→**0.138**（2.2×）；而 σ=0 反而变差（hop 0.680）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V26-G1-v0", G1AmpWalkV26EnvCfg),
    ("Instinct-Amp-Walk-V26-G1-Play-v0", G1AmpWalkV26EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV26TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v26_ppo_cfg:G1AmpWalkV26PPORunnerCfg",
        },
    )

# --- AMP 行走 v27：**AMP 判别器"重启"**（2026-10-04，用户指定）-----------------------
# 背景：V26 的判别器是从 V24 **继承**的（`ckpt_manipulator=None`），而 V24 的判别器已饱和
#   实测：V26 第一个 log 块（iter 8000）已 `amp_d=−1.0004` / `d_ref=0.9999` / `style_std=0.0014`
#   ⇒ V26 的风格项从第 0 轮就是死的，其数据只能回答 σ 的问题。
# 本轮 = V26（σ=0.20 + 300 轮退火）+ **去掉白化**（回 V19b/V19d/V20 口径，`style_std` 0.09~0.12
#   是已验证的健康值）+ **`drop_amp_discriminator`**（判别器与其优化器从零重学，策略保留）。
# 另两个变体（只重置判别器 / 白化但两侧共同累积）见 `amp_walk_v27_ppo_cfg.py` 的标注处。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V27-G1-v0", G1AmpWalkV27EnvCfg),
    ("Instinct-Amp-Walk-V27-G1-Play-v0", G1AmpWalkV27EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV27TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v27_ppo_cfg:G1AmpWalkV27PPORunnerCfg",
        },
    )

# --- AMP 行走 v28：**放大 AMP 风格项权重**（2026-10-04）------------------------------
# 诊断：V27 判别器是活的（`style_std` 0.058）但**权重不足以影响策略**——
#   `r = λ·r_task + (1−λ)·coef·style`，λ=0.75、coef=1.0 ⇒ 风格项权重仅 0.25，
#   而风格奖励的跨状态方差只有 0.058 ⇒ 作用量 0.0145/步，比任务项小 10~70 倍。
# 本轮 = V27 逐字不动（去白化 / σ=0.20 / 奖励表 / AMP 观测全部不变），**只把
#   `discriminator_reward_coef` 1.0 → 4.0**（风格项权重 0.25 → 1.0，4×；λ 不动 ⇒ 任务项权重不变），
#   判别器同样 `drop_amp_discriminator` 从零重学，策略热启动 V27 最新 ckpt。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V28-G1-v0", G1AmpWalkV28EnvCfg),
    ("Instinct-Amp-Walk-V28-G1-Play-v0", G1AmpWalkV28EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV28TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v28_ppo_cfg:G1AmpWalkV28PPORunnerCfg",
        },
    )

# --- AMP 行走 v29：**系数回退到 V27 + 判别器向 AMP_mjlab 对齐**（2026-10-05）----------------
# 背景：V28（coef 1.0→4.0 + GP 1→5 + wd 1e-5→3e-4）在 iter 10470 **训崩**：
#   `Mean reward` 14.85→−7.28、`Mean episode length` 489→**2.7**、`action_abs_mean` 0.35→2.6、
#   `surrogate_loss` 单点 12× 尖峰。机制：风格项噪声 ×8.3（0.0145→0.12/步）+ 从“任务的 16%”
#   变“~47%” + `log_quad` 形状无上界（离群到 13.8/单步）⇒ 优势被单点拖爆。
# ⇒ “放大 AMP 风格权重”已被证伪两次（V15 无效 / V28 有害）⇒ 本轮回退，改攻“**让风格信号有信息**”。
# 本轮 = V27 逐字不动（去白化 / σ=0.20 / 奖励表 / 系数全回退），只把**判别器向 mjlab 对齐**：
#   ① 输入 8 帧 1560 → **2 帧 390**（`(s_t, s_{t+1})` 语义）
#   ② 专家批从“与策略同相位” → **独立随机重采 (T,B)**（全相位覆盖）
#   ③ 风格奖励形状 `−log(1−quad)` → **`clamp(1−0.25(d−1)²,0,1)`**（有界；`d≤−1` 恰为 0）
#   ④ 系数回退到 GP 1.0 / wd 1e-5 / coef 1.0（有效风格权重仍 = 0.25，**刻意不动权重**）
#   起点 = V27 的 `model_10000.pt` 热启动 + `drop_amp_discriminator`（输入维度变了，必须重学）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V29-G1-v0", G1AmpWalkV29EnvCfg),
    ("Instinct-Amp-Walk-V29-G1-Play-v0", G1AmpWalkV29EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV29TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v29_ppo_cfg:G1AmpWalkV29PPORunnerCfg",
        },
    )

# --- AMP 行走 v30：**判别器 = AMP_mjlab 原始配方**（2026-10-05）----------------------------
# 背景：V29 已把判别器的“输入/形状”对齐 mjlab（390 + `bounded_quad`），但**仍然 1 个迭代内**
#   把 MSE 压到 0.035（V27 稳态甚至 0.0014）⇒ 风格奖励变常数 ⇒ 风格项无梯度。
# 两个猜测的缺陷已被数据否掉：①“速度维是主因”❌（稳态 d: lin 0.21 / ang 0.07）
#   ②“专家批是单点/相位锁死”❌（V20 期 16envs×1995 步：跟env σ ≈ 跟时间 σ，比值 1.05~1.32）。
# 只剩**从未单独试过**的 mjlab 判别器配方：
#   ① GP `λ=1.0→10.0`   ② GP 从“actor+expert 合并”→“**只在 expert 侧**”
#   ③ wd 从 loss 项（1e-5）→**优化器分组**（trunk 1e-3 / head 1e-2）
#   ④ 专家批从“活体影子机器人”→“**离线运动数据集 i.i.d. 抽帧**”（`amp_expert_states.pt`）
#   ⑤ 风格强度**刻意不动**（coef 保持 1.0；V28 已证明“放大”有害，本轮不引入权重这个变量）
# 起点 = V27 `model_10000` 热启动 + `drop_amp_discriminator`（判别器口径变了，必须重学）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V30-G1-v0", G1AmpWalkV30EnvCfg),
    ("Instinct-Amp-Walk-V30-G1-Play-v0", G1AmpWalkV30EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV30TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v30_ppo_cfg:G1AmpWalkV30PPORunnerCfg",
        },
    )

# --- AMP 行走 v31：**只改一个数** —— 躯干倾角惩罚 `flat_orientation_l2` −2 → −4（2026-10-06）----
# 姿态诊断（V30@11290 vs V27，实测）：`Step/tilt_deg_mean` 12.56° → **23.86°**、
#   `tilt_gt30_frac` 1.2% → **36%**、`Step/grav_bx_mean` −0.037 → **−0.366**（负 = 后仰 ≈21.5°）；
#   参考动作基线（17 clip/7417 帧）只有 **9.0° 且略前倾**（`g_x=+0.147`、`>30°` 为 0%）
#   ⇒ 策略与参考姿态相差 ~30°。而 **AMP 判别器对绝对倾角是盲的**（V20 口径不含 anchor 自身姿态）
#   ⇒ 姿态只由 `flat_orientation_l2`（唯一）定价：w=−2 时 24° 处的边际价格 2.97/rad <
#   收益梯度 ≈4.3/rad ⇒ 策略理性地继续后仰。w=−4 ⇒ 5.94/rad，净梯度转负。
# 其余（判别器 mjlab 配方 / coef=1.0 / λ=0.75 / σ=0.20）**逐字不动**；
# 起点 = **V30 最新 ckpt 热启动且保留判别器**（AMP 口径没变，只改了奖励权重）。
for task_id, env_cfg in (
    ("Instinct-Amp-Walk-V31-G1-v0", G1AmpWalkV31EnvCfg),
    ("Instinct-Amp-Walk-V31-G1-Play-v0", G1AmpWalkV31EnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV31TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v31_ppo_cfg:G1AmpWalkV31PPORunnerCfg",
        },
    )

# --- AMP 行走 v32：**控制变量实验（A/B/C 三臂共用同一 fork ckpt）**（2026-10-06）----------------
# fork 点 = V31 `model_11500.pt`；三臂同起点 / 同预算（800 iters）/ 同种子。
#   A 臂 = 旧 task id `Instinct-Amp-Walk-V31-G1-v0`（什么都不改，纯续训对照）
#   B 臂（本块）= V31 env 逐字 + **唯一变量：白化统计量从"只用专家侧"改成"两侧都用"**（对齐 mjlab）
#               + `reset_amp_normalizer`（累计平均 + 千万级 count ⇒ 不清零这一臂等于没改）
#   C 臂（本块）= B + **拆脚手架**：`feet_air_time` 4→2（我们自加的手写节奏器，mjlab 无此项）、
#               `joint_deviation_arms` −0.1→0（实测手臂是 default↔reference 差异最大部位，
#               该项与 AMP 先验在手臂上正面对撞）
# 两臂共用同一算法 cfg（`amp_walk_v32b_ppo_cfg`）⇒ C 对 B 只差"两行奖励权重"。
for task_id, env_cfg, agent_cls in (
    ("Instinct-Amp-Walk-V32B-G1-v0", G1AmpWalkV32BEnvCfg, "G1AmpWalkV32BPPORunnerCfg"),
    ("Instinct-Amp-Walk-V32B-G1-Play-v0", G1AmpWalkV32BEnvCfg_PLAY, "G1AmpWalkV32BPPORunnerCfg"),
    ("Instinct-Amp-Walk-V32C-G1-v0", G1AmpWalkV32CEnvCfg, "G1AmpWalkV32BPPORunnerCfg"),
    ("Instinct-Amp-Walk-V32C-G1-Play-v0", G1AmpWalkV32CEnvCfg_PLAY, "G1AmpWalkV32BPPORunnerCfg"),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV31TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.amp_walk_v32b_ppo_cfg:{agent_cls}",
        },
    )

# --- AMP 行走 v33：**结构级对齐 mjlab 的三臂（S0 / S1 / A-cold）**（2026-10-06）------------------
# fork 点 = V31 `model_11500.pt`（actor 热启动 + **判别器从零重学**），三臂同起点 / 同指令分布 /
# 同预算（11500 → **12500**，500 的倍数 ⇒ 可导出）/ 同种子。
#   S0  = 结构包：① 判别器输入白化 ② GP 移入归一化空间 ③ AMP 状态补 "anchor 投影重力"（390→396）
#         （①② 在 `amp_walk_v33s_ppo_cfg`，③ 在 `amp_walk_v33_env_cfg`；配套 396 维专家数据集）
#   S1  = S0 + 拆脚手架（`feet_air_time` 4→2、`joint_deviation_arms` −0.1→0）
#   A-cold = 旧口径（390 / 无白化 / raw GP）+ 丢判别器 ⇒ 给 S0/S1 一个同起点的参照系
# 三臂共有：`side_vel_min = 0.15`（纯侧向档不再落入"静止门控"）+ `track_lin_vel_xy` 换成
# 无损诊断版（奖励**一分不变**，白拿 `Step/lin_track_*` x/y 拆分曲线）。
for task_id, env_cfg, agent_module, agent_cls in (
    ("Instinct-Amp-Walk-V33S-G1-v0", G1AmpWalkV33SEnvCfg, "amp_walk_v33s_ppo_cfg", "G1AmpWalkV33SPPORunnerCfg"),
    (
        "Instinct-Amp-Walk-V33S-G1-Play-v0",
        G1AmpWalkV33SEnvCfg_PLAY,
        "amp_walk_v33s_ppo_cfg",
        "G1AmpWalkV33SPPORunnerCfg",
    ),
    ("Instinct-Amp-Walk-V33S1-G1-v0", G1AmpWalkV33S1EnvCfg, "amp_walk_v33s_ppo_cfg", "G1AmpWalkV33SPPORunnerCfg"),
    (
        "Instinct-Amp-Walk-V33S1-G1-Play-v0",
        G1AmpWalkV33S1EnvCfg_PLAY,
        "amp_walk_v33s_ppo_cfg",
        "G1AmpWalkV33SPPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-V33S2-G1-v0",
        G1AmpWalkV33S2EnvCfg,
        "amp_walk_v33s2_ppo_cfg",
        "G1AmpWalkV33S2PPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-V33S2-G1-Play-v0",
        G1AmpWalkV33S2EnvCfg_PLAY,
        "amp_walk_v33s2_ppo_cfg",
        "G1AmpWalkV33S2PPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-V33S3-G1-v0",
        G1AmpWalkV33S3EnvCfg,
        "amp_walk_v33s3_ppo_cfg",
        "G1AmpWalkV33S3PPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-V33S3-G1-Play-v0",
        G1AmpWalkV33S3EnvCfg_PLAY,
        "amp_walk_v33s3_ppo_cfg",
        "G1AmpWalkV33S3PPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-V33S4-G1-v0",
        G1AmpWalkV33S4EnvCfg,
        "amp_walk_v33s4_ppo_cfg",
        "G1AmpWalkV33S4PPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-V33S4-G1-Play-v0",
        G1AmpWalkV33S4EnvCfg_PLAY,
        "amp_walk_v33s4_ppo_cfg",
        "G1AmpWalkV33S4PPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-V33S5-G1-v0",
        G1AmpWalkV33S5EnvCfg,
        "amp_walk_v33s5_ppo_cfg",
        "G1AmpWalkV33S5PPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-V33S5-G1-Play-v0",
        G1AmpWalkV33S5EnvCfg_PLAY,
        "amp_walk_v33s5_ppo_cfg",
        "G1AmpWalkV33S5PPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-V33S6-G1-v0",
        G1AmpWalkV33S6EnvCfg,
        "amp_walk_v33s6_ppo_cfg",
        "G1AmpWalkV33S6PPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-V33S6-G1-Play-v0",
        G1AmpWalkV33S6EnvCfg_PLAY,
        "amp_walk_v33s6_ppo_cfg",
        "G1AmpWalkV33S6PPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-V33S7-G1-v0",
        G1AmpWalkV33S7EnvCfg,
        "amp_walk_v33s7_ppo_cfg",
        "G1AmpWalkV33S7PPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-V33S7-G1-Play-v0",
        G1AmpWalkV33S7EnvCfg_PLAY,
        "amp_walk_v33s7_ppo_cfg",
        "G1AmpWalkV33S7PPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-V33S8-G1-v0",
        G1AmpWalkV33S8EnvCfg,
        "amp_walk_v33s8_ppo_cfg",
        "G1AmpWalkV33S8PPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-V33S8-G1-Play-v0",
        G1AmpWalkV33S8EnvCfg_PLAY,
        "amp_walk_v33s8_ppo_cfg",
        "G1AmpWalkV33S8PPORunnerCfg",
    ),
    (
        # ⭐ T1a：AMP-Walk 接口/特性 + T0(基座) 等价奖励集 + 风格关闭 + 镜像损失
        "Instinct-Amp-Walk-T1a-G1-v0",
        G1AmpWalkT1aEnvCfg,
        "amp_walk_t1a_ppo_cfg",
        "G1AmpWalkT1aPPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-T1a-G1-Play-v0",
        G1AmpWalkT1aEnvCfg_PLAY,
        "amp_walk_t1a_ppo_cfg",
        "G1AmpWalkT1aPPORunnerCfg",
    ),
    (
        # ⭐ T1a2：T1a + 奖励实现/作用域对齐 T0（feet_air_time / dof_pos_limits /
        #          dof_acc_l2）+ 探索与网络调回 T0（σ=1.0、无 σ 上限、256/128/128）
        "Instinct-Amp-Walk-T1a2-G1-v0",
        G1AmpWalkT1a2EnvCfg,
        "amp_walk_t1a2_ppo_cfg",
        "G1AmpWalkT1a2PPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-T1a2-G1-Play-v0",
        G1AmpWalkT1a2EnvCfg_PLAY,
        "amp_walk_t1a2_ppo_cfg",
        "G1AmpWalkT1a2PPORunnerCfg",
    ),
    (
        # ⭐ T1b：**只改一个数** `style_lerp` 1.0 → 0.75（打开风格项）。
        #         环境侧**直接复用 T1a2 的 env cfg 类**（不复制）⇒ 环境零改动，
        #         “单变量”在代码里自证。热启动自 T1a2 的 model_7000.pt。
        "Instinct-Amp-Walk-T1b-G1-v0",
        G1AmpWalkT1a2EnvCfg,
        "amp_walk_t1b_ppo_cfg",
        "G1AmpWalkT1bPPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-T1b-G1-Play-v0",
        G1AmpWalkT1a2EnvCfg_PLAY,
        "amp_walk_t1b_ppo_cfg",
        "G1AmpWalkT1bPPORunnerCfg",
    ),
    (
        # ⭐ T1e：T1d 的配置 + **限位奖励换回全关节软边距版**（唯一改动）。
        #        env 用新类 `G1AmpWalkT1eEnvCfg`（继承 T1a2 后只覆写 dof_pos_limits）；
        #        agent 复用 T1b 的 λ=0.75（与 T1d 逐字相同）⇒ 单变量。
        #        热启动自 **T1d 的 model_7500.pt**（保留风格训练成果）。
        "Instinct-Amp-Walk-T1e-G1-v0",
        G1AmpWalkT1eEnvCfg,
        "amp_walk_t1b_ppo_cfg",
        "G1AmpWalkT1bPPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-T1e-G1-Play-v0",
        G1AmpWalkT1eEnvCfg_PLAY,
        "amp_walk_t1b_ppo_cfg",
        "G1AmpWalkT1bPPORunnerCfg",
    ),
    (
        # ⭐ T1f：T1a2 的观测/AMP/动作契约 + **mjlab 的奖励机理**（步宽 −10 / 交叉 −20 /
        #        全关节限位 −10 / flat_orientation −2 / 跟踪 1.5,1.2）+ σ 下限 0.05。
        #        **从零训练**（不 resume）⇒ 检验"早期就有正确机理"能否避免踝冻结。
        "Instinct-Amp-Walk-T1f-G1-v0",
        G1AmpWalkT1fEnvCfg,
        "amp_walk_t1f_ppo_cfg",
        "G1AmpWalkT1fPPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-T1f-G1-Play-v0",
        G1AmpWalkT1fEnvCfg_PLAY,
        "amp_walk_t1f_ppo_cfg",
        "G1AmpWalkT1fPPORunnerCfg",
    ),
    (
        # ⭐ T1i：**环境与 T1a2 逐字相同**，只改风格奖励口径对齐 mjlab：
        #       `amp_reward_form="bounded_quad"` + `discriminator_reward_type="wasserstein"`
        #       + `disable_style_on_stand=True`。热启动自 T1a2 的 model_7000.pt
        #       ⇒ 与 T1b/T1c/T1d/T1e **同起点可比**。
        "Instinct-Amp-Walk-T1i-G1-v0",
        G1AmpWalkT1a2EnvCfg,
        "amp_walk_t1i_ppo_cfg",
        "G1AmpWalkT1iPPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-T1i-G1-Play-v0",
        G1AmpWalkT1a2EnvCfg_PLAY,
        "amp_walk_t1i_ppo_cfg",
        "G1AmpWalkT1iPPORunnerCfg",
    ),
    (
        # ⭐ T1j：T1i 的配置 + 解除 `joint_deviation_torso/knee` 的常角拉力。
        #        起点 = T1a2 的 model_7000.pt（与 T1i 同起点可比）。
        "Instinct-Amp-Walk-T1j-G1-v0",
        G1AmpWalkT1jEnvCfg,
        "amp_walk_t1i_ppo_cfg",
        "G1AmpWalkT1iPPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-T1j-G1-Play-v0",
        G1AmpWalkT1jEnvCfg_PLAY,
        "amp_walk_t1i_ppo_cfg",
        "G1AmpWalkT1iPPORunnerCfg",
    ),
    (
        # ⭐ T1k：T1j + `track_base_height +1.5`（治蹲姿）+ 足部两项小权重（−1/−2）。
        "Instinct-Amp-Walk-T1k-G1-v0",
        G1AmpWalkT1kEnvCfg,
        "amp_walk_t1i_ppo_cfg",
        "G1AmpWalkT1iPPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-T1k-G1-Play-v0",
        G1AmpWalkT1kEnvCfg_PLAY,
        "amp_walk_t1i_ppo_cfg",
        "G1AmpWalkT1iPPORunnerCfg",
    ),
    (
        # ⭐ T1l：T1k + `track_lin_vel_xy_exp` **1.0 → 1.5**（V33/mjlab 值）
        #       + `track_base_height` **std 0.2 → 0.3**（V9H 收紧值 → V33/mjlab 值）。
        #       起点 = T1a2 的 model_7000.pt（与 T1j/T1i/T1k 同起点可比）。
        "Instinct-Amp-Walk-T1l-G1-v0",
        G1AmpWalkT1lEnvCfg,
        "amp_walk_t1i_ppo_cfg",
        "G1AmpWalkT1iPPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-T1l-G1-Play-v0",
        G1AmpWalkT1lEnvCfg_PLAY,
        "amp_walk_t1i_ppo_cfg",
        "G1AmpWalkT1iPPORunnerCfg",
    ),
    (
        # ⭐ T1c：`style_lerp` 1.0 → **0.9**（风格权重 0.10 = T1b 的 40%）。
        #        T1b（λ=0.75 ⇒ 权重 0.25）实测把 `vx_high` 从 0.59 压到 0.014。
        #        环境仍复用 T1a2 的 env cfg；热启动自 **T1a2** 的 model_7000.pt。
        "Instinct-Amp-Walk-T1c-G1-v0",
        G1AmpWalkT1a2EnvCfg,
        "amp_walk_t1c_ppo_cfg",
        "G1AmpWalkT1cPPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-T1c-G1-Play-v0",
        G1AmpWalkT1a2EnvCfg_PLAY,
        "amp_walk_t1c_ppo_cfg",
        "G1AmpWalkT1cPPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-V33ACold-G1-v0",
        G1AmpWalkV33AColdEnvCfg,
        "amp_walk_v33acold_ppo_cfg",
        "G1AmpWalkV33AColdPPORunnerCfg",
    ),
    (
        "Instinct-Amp-Walk-V33ACold-G1-Play-v0",
        G1AmpWalkV33AColdEnvCfg_PLAY,
        "amp_walk_v33acold_ppo_cfg",
        "G1AmpWalkV33AColdPPORunnerCfg",
    ),
):
    gym.register(
        id=task_id,
        entry_point=G1AmpWalkV31TeacherEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.{agent_module}:{agent_cls}",
        },
    )
