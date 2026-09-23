# AMP 行走特权策略 —— 代码梳理与修改计划

> 目标：在学长交付的 `instinct_xu`（instinct 训练环境，已训出越障策略）基础上，训练 **AMP 风格行走的特权策略**
> 本次范围：**只梳理代码 + 出修改计划**，不动环境、不开训

---

## 0. instinct_xu 现状盘点

### 0.1 已注册的三个任务

| 任务 ID | 内容 | 算法 | 说明 |
|---|---|---|---|
| `Instinct-Unified-OmniLoco-G1-v0` | 平地行走教师 | 纯 PPO | **623 维 obs 冻结契约**，29 维动作 |
| `Instinct-Perceptive-Vault-G1-v0` | 感知越障教师 | 纯 PPO | 参考动作 npz + 障碍箱 |
| `Instinct-Perceptive-Vault-VaeDistill-G1-v0` | 越障视觉学生蒸馏 | VaePpoDistill | 依赖越障教师 |

### 0.2 可直接复用的基建（关键）

| 能力 | 位置 | 状态 |
|---|---|---|
| **AMP / WASABI 算法** | `instinct_rl/algorithms/wasabi.py` | ✅ **已实现，但三个任务都没用** |
| AMP 判别器 | `modules/discriminator.py` · `Discriminator(input_segment=…)` | ✅ |
| AMP 存储 | `storage/amp_storage.py` | ✅ |
| 参考动作管理器 | `motion_reference/`（npz + metadata，逐 env 逐步采样） | ✅ |
| 参考动作当观测的函数 | `envs/mdp/observations/reference_as_state.py`（`base_lin_vel/base_ang_vel/projected_gravity/joint_pos_rel/joint_vel_rel/link_pos…_reference_as_state`） | ✅ |
| 局部高程图 | `MultiMeshRayCasterCfg` 3.0×1.6 m / 0.1 m → **31×17 = 527 射线** | ✅ 已有 |
| 多奖励分组 + 多 critic | `MultiRewardManager` · `advantage_mixing_weights` | ✅（本次可不用） |

### 0.3 行走（OmniLoco）的现有接口 —— 我们的基准

```
动作     : 29 维关节目标，scale = 1.0，use_default_offset = True，clip = None
关节序   : OMNI_LOCO_JOINT_ORDER（肩优先、左右交错，29 个）
执行器   : beyondmimic_g1_29dof_actuators
时序     : sim.dt = 0.005，decimation = 4 → step_dt = 0.02（50 Hz）
指令     : lin_vel_x(-0.5,1.0) lin_vel_y(-0.5,0.5) ang_vel_z(-1.0,1.0)，保持 10 s
actor obs: height_scan(527) + velocity_command(3) + projected_gravity(3)
           + base_ang_vel(3) + joint_pos_rel(29) + joint_vel(29) + last_action(29) = 623
critic   : 同上 + base_lin_vel(3)，且关闭噪声
```

> ⚠️ **OmniLoco 带 fail-fast 契约校验**（`G1OmniLocoTeacherEnv._validate_teacher_contract`）：会逐项断言 obs 项名/维度/射线数/动作维度/关节序/scale/offset/时序/指令范围/执行器。**因此不能直接改它来加特权项，必须新建任务。**

---

## 1. 目标策略的输入 / 输出定义

### 1.1 动作（输出）

| 项 | 取值 | 与谁对齐 |
|---|---|---|
| 维度 | **29** | instinct_xu |
| 语义 | 关节目标位置 | instinct_xu |
| `scale` | **1.0**（不用 AMP_mjlab 的逐关节 scale） | instinct_xu |
| `use_default_offset` | `True` → `q_target = q_default + 1·a` | instinct_xu |
| `clip` | `None` | instinct_xu |
| 关节序 | `OMNI_LOCO_JOINT_ORDER` | instinct_xu |

### 1.2 Actor 特权观测（建议 **626 维** = 623 + 3）

| 项 | 维度 | 类别 | 来源 |
|---|---|---|---|
| `height_scan` | 527 | **局部高程图（特权）** | `MultiMeshRayCasterCfg` @ `torso_link`，3.0×1.6 m / 0.1 m |
| `velocity_command` | 3 | command | `generated_commands` |
| `projected_gravity` | 3 | 本体感知 | |
| `base_ang_vel` | 3 | 本体感知 | |
| **`base_lin_vel`** | **3** | **← 本次新增的特权项** | `mdp.base_lin_vel` |
| `joint_pos_rel` | 29 | 本体感知 | |
| `joint_vel` | 29 | 本体感知 | |
| `last_action` | 29 | | |

> 真机无法直接测 `base_lin_vel`，所以它是**特权量** —— 正是"特权策略"要利用的。后续蒸馏给学生时该通道会被去掉/被估计器替代。

### 1.3 Critic 观测

沿用 instinct_xu 的 `G1FlatCriticObsCfg`（含 `base_lin_vel`、关闭噪声），并在其基础上按需追加 oracle 量（高程、接触状态等）。

### 1.4 AMP 观测组（`amp_policy` / `amp_reference`）

**两者必须同构**（`get_subobs_size` 必须相等，否则 `init_storage` 里断言失败）。

| 组 | 内容 | 来源 |
|---|---|---|
| `amp_policy` | anchor 相对的各 body 位置 / 朝向 / 线速度 / 角速度 | 机器人当前状态 |
| `amp_reference` | 同上，但取自参考动作 | `*_reference_as_state` 系列函数 |

- anchor 建议 `torso_link`，body 集合沿用 AMP_mjlab 的 13 个（pelvis / 双髋 roll / 双膝 / 双踝 roll / 双臂 roll·elbow·腕）
- **不要把高程图放进 AMP 状态**（会稀释风格判别）

---

## 2. 需要新增 / 修改的代码清单

> 原则：**新增为主，尽量不改学长的原文件**，以免破坏已训好的 Vault 链路。

| # | 文件 | 动作 | 说明 |
|---|---|---|---|
| 1 | `tasks/locomotion/config/g1/amp_walk_env_cfg.py` | **新增** | 继承 `G1OmniLocoEnvCfg`，覆盖：actor obs 加 `base_lin_vel`；新增 `amp_policy`/`amp_reference` 两组；替换为 AMP 奖励集；挂 `motion_reference` 场景实体 |
| 2 | `tasks/locomotion/config/g1/agents/amp_walk_rl_cfg.py` | **新增** | `InstinctRlOnPolicyRunnerCfg` 子类；`algorithm.class_name = "WasabiPPO"` + WASABI 参数；`policy` 用 `InstinctRlActorCriticCfg`（hidden 建议 512/256/128） |
| 3 | `tasks/locomotion/config/g1/__init__.py` | 修改 | 追加 `Instinct-Amp-Walk-G1-v0`(+ Play) 注册（**只加不删**） |
| 4 | `tasks/locomotion/mdp/rewards_amp.py` | **新增** | 从 AMP_mjlab 迁移的奖励函数（见 §3.1） |
| 5 | `tasks/locomotion/mdp/velocity_command.py` | **新增** | 移植 AMP_mjlab 的 `DirectionalVelocityCommandCfg`（见 §3.5） |
| 6 | `tasks/locomotion/mdp/observations_amp.py` | **新增**（可选） | 若需要 AMP_mjlab 那种 anchor-relative body 量 |
| 7 | 参考动作 npz + selection yaml | **新增** | 行走参考动作（见 §3.4、§5.1） |
| 8 | 算法子类 `WasabiPPOAmpMix` | **新增** | 实现 `additive`/`lerp` 两种奖励混合模式（见 §4.2）；**只 override 一个方法**。放置位置见 §4.2（runner 有 1 处 bug，需选 A 或 B） |
| 9 | WASABI / 算法 `@configclass` 子类 | **新增** | 声明 WASABI 字段 + 混合模式字段（`@configclass` 是 dataclass，**额外字段必须显式声明**） |

**不需要改**：`wasabi.py`、`ppo.py`、`discriminator.py`、`amp_storage.py`、`OmniLoco` 三个原任务，以及**全部奖励函数与权重**。

---

## 3. 从 AMP_mjlab 迁移什么 / 不迁移什么

### 3.1 奖励：迁移（你的要求 3）

| AMP_mjlab 奖励项 | 权重 | 迁移建议 |
|---|---|---|
| `track_anchor_linear_velocity` | 1.5 | ✅ 迁（instinct_xu 已有 `track_lin_vel_xy_yaw_frame_exp`，语义等价，可直接改权重） |
| `track_anchor_angular_velocity` | 1.2 | ✅ 迁（对应 `track_ang_vel_z_world_exp`） |
| `track_root_height` | 1.5 | ✅ 迁（instinct_xu 里 `base_height_l2` 被注释掉了，需启用或自写） |
| `body_ang_vel_xy_l2` | 0.7 | ✅ 迁 |
| `track_default_pose_on_stand` | -1.0 | ✅ **建议迁**（低速回 default 站姿 + 速度阻尼；对"停稳"很关键） |
| `feet_lateral_width` | -10.0 | ✅ 迁（防侧向并步过窄） |
| `feet_crossing` | -20.0 | ✅ 迁（防交叉步） |
| `self_collisions` | -0.1 | ✅ 迁 |
| `foot_slip` | — | ✅ 迁（instinct_xu 已有 `feet_slide`） |
| `torso_posture` / `feet_swing_clearance` / `soft_landing` | 已注释 | ⬜ 暂不迁（AMP_mjlab 自己也没用） |

**保留 instinct_xu 原有**的：`termination_penalty(-200)`、`stand_still(-0.8)`、`joint_deviation_*`、`action_rate_l2`、`dof_acc_l2`、`dof_torques_l2`、`dof_pos_limits`、`flat_orientation_l2`、`lin_vel_z_l2`、`feet_air_time(1.0)`。

### 3.2 观测：**不迁移**（你的要求 3/4）

| AMP_mjlab | instinct_xu | 结论 |
|---|---|---|
| actor = **384 维**，且**删掉了 height_scan** | actor = 623 维，**含 527 射线高程图** | ⚠️ **不迁**，保留 instinct_xu 的（高程图是后续越障蒸馏的接口一致性要求） |
| AMP 组用 `body_pos_b`/`body_ori_b`/`body_lin_vel_b`/`body_ang_vel_b` | 无对应 | ✅ **迁这个概念**（构造 `amp_policy`/`amp_reference`） |
| `min_normalized_std=[0.05]*29` | — | ✅ 迁（与动作尺度无关） |

### 3.3 参数对齐清单（你的要求 3）

| 参数 | AMP_mjlab | instinct_xu | 采用 |
|---|---|---|---|
| 动作尺度 | 逐关节 `action_scale` | **1.0** | **instinct_xu** |
| 动作维度 / 关节序 | mjlab 序 | `OMNI_LOCO_JOINT_ORDER` | **instinct_xu** |
| 时序 | mjlab 默认 | dt 0.005 / decim 4 | **instinct_xu** |
| 执行器 | mjlab actuator cfg | `beyondmimic_g1_29dof_actuators` | **instinct_xu** |
| 高程图 | 无 | 527 射线 | **instinct_xu** |
| 奖励项与权重 | 见 §3.1 | — | **继承 AMP_mjlab** |

### 3.4 运动风格数据集（AMP_mjlab 经验，务必参考）

**AMP_mjlab v4 最终只用了 7 个 clip**（清单见 `motion_data_csv/amp_selected.txt`，`WalkandRun/` 下原有 17 个）。
⚠️ 清单里的 `walk_backward_loop_003__A143` **在 `WalkandRun/` 中并不存在**（只有 `walk_backward_loop_001__A022/A024`），说明“选中清单”与“实际部署集”并不一致 —— 这也是我们在迁移时必须**以实测数据为准**、而不是照抄清单的原因。

**已实测的 17 个 clip 特性**（`instinct/tools/convert_amp_motion.py --stats`，本体系速度，单位 m/s）：

| 类别 | clip | vx | vy | ωz (°/s) | 曲线半径 |
|---|---|---|---|---|---|
| 行走直向 | `walk_forward_loop_002__A022` / `__A024` | +0.63 / +0.75 | — | ~0 | 直行 |
| 行走后退 | `walk_backward_loop_001__A022` / `__A024` | −0.58 / −0.50 | — | ~0 | 直行 |
| 侧移 | `walk_sideway_left_loop_002__A021` | ~0 | **+0.55** | ~0 | 直行 |
| | `walk_sideway_right_loop_001__A022` | ~0 | **−0.48** | ~0 | 直行 |
| 慢跑直向 | `jog_forward_loop_003__A021` / `__A022` | +0.90 / +1.02 | — | ~0 | 直行 |
| 慢跑后退 | `jog_backward_loop_002__A022` / `__A023` | −0.90 / −1.07 | — | ~0 | 直行 |
| 弧线行走 | `walk_arc_cw_loop_002__A046` | +0.78 | — | −27.1 | 1.66 m |
| | `arc_walk_left_loop_001__A029` | +1.04 | — | +34.3 | 1.74 m |
| 弧线慢跑 | `jog_arc_cw_loop_004__A045` | +1.29 | — | −44.2 | 1.68 m |
| | `arc_jog_left_loop_002__A029` | **+1.69** | — | +57.0 | 1.71 m |
| 原地转 | `idle_turn_270_001__A048` | 0 | 0 | −14.3 | — |
| | `idle_turn_360_001__A047` | 0 | 0 | −24.1 | — |
| 原地踏步 | `step_rotate_idle_000_002__A026` | 0 | 0 | ~0 | — |

**实测数据集包络（速度指令必须落在此范围内）**：

$$v_x \in [-1.07,\ +1.69]\ \text{m/s},\quad v_y \in [-0.48,\ +0.55]\ \text{m/s},\quad \omega_z \in [-44.2°,\ +57.0°]\ \text{/s}$$

几点直接可用的结论：

1. **前后向覆盖最好**（±0.5 ~ ±1.07 直行共 6 段），**侧向覆盖最薄**（仅 2 段、且 |vy| ≤ 0.55）→ 指令里 `vy` 上限**不应超过 ±0.5**，否则重演“判别器压制侧移”的问题
2. **所有弧线半径统一 ≈1.7 m** → 曲线命令实际是“v 与 ωz 耦合”的一维族，`ωz` 不宜与 `|v|` 解耦采样
3. **最高速度只有 1.69 m/s**，且只有 1 段 → 若命令上限设到 2.0 m/s，高速档将缺乏参考密度（**这正是 AMP_mjlab 排除 `jog_forward_loop_003__A021` 的原因**）；建议初版命令上限取 **±1.2 m/s**
4. 所有 clip 的 `torso_link` 高度一致（0.749 ~ 0.827 m，均值 ≈0.81 m）→ 与 InstinctMJ `_POPSICLE_INIT_STATE` 的 `pos_z=0.82` **吻合**，间接支持“articulation root = `torso_link`”

**选取准则（关键）：clip 的速度分布必须落在速度指令范围内**

- 慢跑 `A022` 保留：44% 帧落在 1.2–2.0 命令带内 → 判别器不再把高速跑姿判为“不自然”，步态切换自然涌现
- 慢跑 `A021` 排除：巡航 2.3–2.4，27% 帧超出命令上限 2.0，带内仅 13% → 对当前目标参考密度低

**教训：数据集覆盖面决定 AMP 会压制什么**

- `WalkandRun` 原 17 段里侧移仅 2 段、后退 4 段（共 ~7417 帧）→ 判别器把横向/慢后退姿态判为“不像参考”而抑制
- **改命令分布无法解决（已实测）**，只能补参考运动数据

→ **对我们的指导**：参考 npz 按同一准则挑 —— 保证**每个命令方向都有足够参考密度**（尤其侧向、后退），否则重演 AMP_mjlab 的问题。

### 3.5 速度指令设计（AMP_mjlab 经验，务必参考）

**现状问题**：instinct_xu 的 OmniLoco 用的是 `UniformVelocityCommandCfg`（三轴独立连续均匀），且 `rel_standing_envs=0`。
→ **正是 AMP_mjlab 诊断出的毛病**：`lin_vel_x` 精确为 0 的概率为零 ⇒ 几乎不出现“纯侧向”命令 ⇒ **左右直移学不出来**。

**AMP_mjlab 的 `DirectionalVelocityCommandCfg`（按概率分档）**：

| 分支 | 概率 | 内容 |
|---|---|---|
| 低速 | `prob_low=0.15` | `‖(vx,vy)‖+|ωz| ≤ low_vel`，其中 `zero_frac_in_low` 比例给精确 (0,0,0)；100% 命中 `track_default_pose_on_stand` 门控 |
| 纯侧向 | `prob_side=0.25` | `lin_vel_x = 0`，y 全范围 |
| 纯前后 | `prob_lin=0.25` | `lin_vel_y = 0`；内部 `prob_lin_fwd=0.6` 给持续直线前向 |
| 持续后退 | `prob_back=0.10` | `lin_vel_y = 0`，vx 取负且幅度较大，ωz≈0 |
| 均匀斜向 | 剩余 0.25 | |

**命令课程**：y 全程不缩小；早期（step<2000）x 上限压到 ±1.0，2000 后恢复全对称。

**对我们的指导**：`DirectionalVelocityCommandCfg` 只是 `UniformVelocityCommandCfg` 的子类，**可直接移植**到 instinct_xu 的 mdp 里并替掉 Uniform。注意 OmniLoco 原任务断言了命令范围与 `rel_standing_envs=0`；新任务可自定，但建议保持 `rel_standing_envs=0`（“站立”交给 `prob_low` 分支，避免双重站立语义）。

**具体数值按 §3.4 实测包络取**：

| 参数 | 初版取值（依据） |
|---|---|
| `lin_vel_x` | ±1.2 m/s（数据集最高 1.69 但仅 1 段，留余量） |
| `lin_vel_y` | ±0.5 m/s（**不得超过 0.55**，侧向参考仅 2 段） |
| `ang_vel_z` | ±0.9 rad/s ≈ ±51°/s（**不得超过 57°/s**） |
| `heading` | 保持开启（弧线半径固定 1.7 m，转向与前进耦合） |

---

## 4. AMP 算法接线的关键差异（⚠️ 最容易出错的地方）

### 4.1 两套 AMP 实现不同，参数不能照搬

| AMP_mjlab（rsl_rl fork） | instinct_xu（WASABI） | 映射说明 |
|---|---|---|
| `class_name="AMPPPO"` | `class_name="WasabiPPO"` | 不同实现 |
| `amp_reward_coef=0.1` | `discriminator_reward_coef`（默认 1.0） | 语义相近，**建议从 0.05~0.1 起**（弱风格先验） |
| `amp_task_reward_lerp=0.75` | → 自建 `style_lerp` | ✅ 通过 `reward_mix_mode="lerp"` 复刻，见 §4.2 |
| `amp_discr_hidden_dims=[1024,512,256]` | `discriminator_kwargs={"hidden_dims":[…]}` | 需查 `Discriminator` 支持的键名 |
| `amp_body_names` + `amp_anchor_name` | `amp_policy`/`amp_reference` obs group 里选的 body 集合 | 概念对应 |
| `amp_num_preload_transitions=200000` | 无 | 略 |
| `amp_disable_style_on_stand` | → 自建同名开关 | ✅ 在 `compute_auxiliary_reward` 里做掩码（见 §4.2） |
| `discriminator_loss_func` | `"BCEWithLogitsLoss"`（= 传统 AMP）/ `"MSELoss"`（= AMP_mjlab 用法） | ⚠️ **AMP_mjlab 用的是 MSELoss 形式**，见 §4.3 |

### 4.2 奖励混合：用「模式 + 权重」两参复刻 AMP_mjlab（零侵入）

**现状差异**

| | 公式 | 特点 |
|---|---|---|
| instinct_xu（`ppo.py::process_env_step`，实测源码） | `R = r_task + c·r_amp` | **加法 + 系数** |
| AMP_mjlab（已验证效果好） | `R = (1-λ)·c·r_amp + λ·r_task` | **凸组合** |

**设计：`reward_mix_mode` + `style_lerp`**

不采用“单个浮点 `amp_mode`”，而是拆成两个正交参数：

| 参数 | 取值 | 对应 AMP_mjlab | 作用 |
|---|---|---|---|
| `reward_mix_mode` | `"additive"`（默认，= instinct_xu 原样）/ `"lerp"` | — | **选哪套公式** |
| `style_lerp` (λ) | 0~1，建议 0.75 | `amp_task_reward_lerp` | **风格/任务配比**（仅 lerp 模式生效） |
| `discriminator_reward_coef` (c) | 建议 0.05~0.1 | `amp_reward_coef` | 风格项整体缩放 |
| `disable_style_on_stand` / `stand_speed_threshold` | bool / 0.1 | 同名 | 低速时关掉风格项 |

**为什么不只用单个 `amp_mode` 浮点**

1. **“用哪套公式”与“风格占比”是两个正交自由度**，一个浮点只能表达后者
2. 保留 `additive` 原样 → 与学长管线、后续 Vault 蒸馏对接时可随时切回，便于消融对比
3. 参数与 AMP_mjlab **一一对应**，迁移可追溯

> 若确实想要单一浮点写法：让 `reward_mix_mode` 兼收 `float λ`，内部等价于 `"lerp"` + `style_lerp=λ`。接口上保留这个便利，但默认值与文档仍以显式两参为准。

**实现：只 override `compute_auxiliary_reward`，其余一律不动**

父类（`ppo.py`，不改）的组合固定为：
```
R = r_task + c · aux · auxiliary_reward_per_env_reward_coefs
```

因此只要**把 aux 从“风格奖励”换成“lerp 修正量”**，就能在不碰父类、不碰任何奖励函数的前提下得到凸组合：

```
令 aux = (1-λ) · ( r_amp − r_task / c )
则 R = r_task + c·aux = r_task + c(1-λ)(r_amp − r_task/c)
      = (1-λ)·c·r_amp + λ·r_task          ← 正是 AMP_mjlab 的公式 ✅
```

```python
class WasabiPPOAmpMix(WasabiPPO):
    """额外支持 AMP_mjlab 风格奖励混合；additive 模式下行为与父类完全一致。"""

    def __init__(self, *a, reward_mix_mode="additive", style_lerp=0.75,
                 disable_style_on_stand=False, stand_speed_threshold=0.1, **kw):
        super().__init__(*a, **kw)
        ...   # 存字段；init_storage 里按 obs_format 缓存 velocity_command 在 policy 组中的切片

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        self._task_rewards = rewards          # 仅供 aux 使用
        try:
            return super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)
        finally:
            self._task_rewards = None

    @torch.no_grad()
    def compute_auxiliary_reward(self, obs_pack):
        base = super().compute_auxiliary_reward(obs_pack)   # {"discriminator_reward": r_amp}
        if self.reward_mix_mode == "additive":
            return base                                      # ← 原样，零行为变化
        r_amp = base["discriminator_reward"]
        if self.disable_style_on_stand:                      # 低速时关风格项
            cmd = slice_command(obs_pack, self._obs_format["policy"], "velocity_command")
            r_amp = r_amp * (cmd.norm(dim=-1, keepdim=True) > self.stand_speed_threshold)
        c = self.discriminator_reward_coef
        base["discriminator_reward"] = (1.0 - self.style_lerp) * (r_amp - self._task_rewards / c)
        return base
```

**零侵入保证**

- `wasabi.py` / `ppo.py` / `discriminator.py` / `amp_storage.py` **一行不改**
- **所有奖励函数与其权重一行不改** —— 混合只发生在算法侧的奖励合成
- `additive` 模式 = 父类行为，可随时回归对照

**放置位置（已核实，有个坑）**

我查了 `instinct_rl/runners/on_policy_runner.py`：
```python
alg_class = importlib.import_module() if ":" in alg_class_name else getattr(algorithms, alg_class_name)
```
`importlib.import_module()` **调用时没传参数** —— 即 **算法 `class_name` 写成 `"module:Class"` 会直接 `TypeError`**。
（`policy_class_name` 和 `discriminator_class_name` 的 `module:class` 分支都是正常的，**只有算法这一处坏了**。）

两个选择：
- **A（推荐，改动最小且更干净）**：把这一行修成与 `build_actor_critic` 一致的形式，`WasabiPPOAmpMix` 就能放在 instinctlab 侧，`instinct_rl` 只动这 1 行：
  ```python
  alg_class = getattr(importlib.import_module(n), c) if ":" in alg_class_name else getattr(algorithms, alg_class_name)
  ```
- **B**：把 `WasabiPPOAmpMix` 放进 `instinct_rl/algorithms/` 并加进 `__init__.py`，走 `getattr(algorithms, ...)`，**完全不碰 runner**

> ⚠️ `auxiliary_reward_per_env_reward_coefs` 需显式设 `[1.0]`（单奖励组），否则默认 `list()` 可能不生效。

### 4.3 判别器损失形式

- `discriminator_loss_func="MSELoss"` → `actor_d → -1`，`reference_d → +1`（**AMP_mjlab 的用法**）
- `discriminator_loss_func="BCEWithLogitsLoss"` → 传统 AMP
- `discriminator_reward_type="log"` → `-log(1-σ(d))`（默认 AMP 奖励）
- `discriminator_gradient_penalty_coef`（默认 10.0）：WASABI 的 Lipschitz 正则，**传统 AMP 可设为 0**

### 4.4 参考状态必须逐步在线提供

`wasabi.py::process_env_step` 会从 `infos["observations"]` 里取 `amp_policy` / `amp_reference` 两个 key，**取不到直接抛 ValueError**。
所以参考动作必须作为**场景实体**（`scene.motion_reference`）由环境逐步采样，而不是离线数据集 —— 这正是 instinct_xu 的 `MotionReferenceManagerCfg` + `*_reference_as_state` 的用途。

---

## 5. 关键风险与待确认项

### 5.1 ✅ 参考动作数据格式（**已解决**，实测 2025-09-23）

**instinctlab 加载器实际只需要 5 个字段**（读 `motion_reference/motion_files/amass_motion.py::_read_retargetted_motion_file`）：
```
framerate, joint_names(29,), joint_pos(T,29), base_pos_w(T,3), base_quat_w(T,4)
```
- `velocity_command` **不是必需的** —— 加载器会自行估计速度（`velocity_estimation_method`）
- `joint_vel` / `base_lin_vel_w` / `base_ang_vel_w` **也不需提供**，由 `_pack_retargetted_motion_sequence` 自动差分估计
- 加载器会**按关节名**映射到 `isaac_joint_names`，并按 `motion_target_framerate` 自动插值

**原计划中的两个“必须处理点”经实测均为误判，已纠正**：

| 原假设 | 实测结果 |
|---|---|
| 要手动翻转四元数 xyzw→wxyz | ❌ **不需要** —— `csv_to_npz.py` 生成 npz 时**已经**做了 `[:, [3,0,1,2]]`，npz 里已是 wxyz（实测四元数模长 = 1.000） |
| 要手动补 `velocity_command` | ❌ **不需要** —— 加载器自动估计 |

**更重要的是：关节列序不需要重排**。已用 FK 严格验证（MJCF 做正运动学，比对 npz 自带的 `body_pos_w`）：

```
17 个 clip 全部：max|FK − body_pos_w| ≤ 1e-6 m
```

即：**npz 的 `joint_pos` 列序 == MJCF hinge 顺序 == `csv_to_npz.py` 的 `joint_names` 列表 == IsaacLab 命名**（三重一致，带 `_joint` 后缀）。

**交付物**：

| 文件 | 说明 |
|---|---|
| `instinct/tools/convert_amp_motion.py` | 转换器（内置 FK 验证、`--verify-only`、`--stats`、`--base-body`） |
| `instinct_xu/.../tasks/locomotion/config/g1/motions/amp_walk/` | 17 个 `*_retargeted.npz` + `amp_walk_motion_selection.yaml` |

**唯一尚需运行时确认的点（低风险）**：`base_pos_w` 的参考系。脚本默认用 **`torso_link`**（依据：`MotionReferenceManagerCfg.prim_path` 指向 `Robot/torso_link`，且实测 `torso_link` 高度 0.81 m 与 InstinctMJ `_POPSICLE_INIT_STATE.pos_z=0.82` 吻合）。若运行时发现参考与机器人有**恒定偏移**，重跑 `--base-body pelvis` 即可（1 行参数）。

### 5.2 其它风险

| # | 项 | 说明 |
|---|---|---|
| 1 | **不要改 OmniLoco 任务** | 它有 fail-fast 契约校验，加 `base_lin_vel` 会直接 `RuntimeError` → 必须新建任务 |
| 2 | **`amp_policy`/`amp_reference` 必须同构** | `get_subobs_size` 不等会在 `init_storage` 断言失败 |
| 3 | **`Discriminator` 的构造键名** | `input_segment` 来自 `obs_format`（env 自动生成），隐藏层参数键名需查 `discriminator.py` 确认 |
| 4 | **`@configclass` 额外字段** | WASABI 参数必须在算法 cfg 子类里显式声明，否则 runner 传不下去 |
| 5 | **参考动作采样窗口** | `motion_bin_length_s` / `motion_start_from_middle_range`：行走是循环步态，**不要用 RSI 随机起点**（会破坏步态相位连续性），建议整段连续播放 |
| 6 | **AMP 状态不要含高程** | 否则判别器在判别地形而不是判别风格 |
| 7 | **PPO hidden dims 不一致** | instinct_xu 行走用 [256,128,128]、越障用 [512,256,128]；AMP 行走建议 **512/256/128**（与 AMP_mjlab 一致） |
| 8 | **License** | instinct_xu 继承了 InstinctLab 的 CC BY-NC 4.0 |
| 9 | **数据集覆盖面决定 AMP 压制什么** | 参考 clip 缺某方向（侧向/后退）时，判别器会把该方向姿态判为“不像参考”而抑制。**改命令分布无法解决（已实测）**，只能补参考数据。见 §3.4 |
| 10 | **`UniformVelocityCommand` 采不出纯侧向命令** | instinct_xu OmniLoco 用的就是 Uniform（三轴独立均匀）→ x 恰为 0 概率为零 → 左右直移学不出来。见 §3.5 |
| 11 | **runner 没法用 `module:class` 指定算法** | `on_policy_runner.py` 里 `importlib.import_module()` 漏传参数 → 写 `"module:Class"` 会 `TypeError`。见 §4.2 的 A/B 方案 |

---

## 6. 执行顺序（分段）

```
✅① 参考动作数据   【已完成】转换器 + 17 个 clip 已产出
                 instinct/tools/convert_amp_motion.py
                 （FK 校验全通过 ≤1e-6 m；四元数无需翻转；无需补 velocity_command；见 §5.1）
                 另：--stats 实测出数据集速度包络，已回填 §3.4/§3.5
② 任务骨架       【已完成】新增任务 Instinct-Amp-Walk-G1-v0 / -Play-v0
                 amp_walk_env_cfg.py（actor obs 626 = 623 + base_lin_vel；保留 527 高程图）
                 + agents/amp_walk_ppo_cfg.py；**复用**学长那套 fail-fast 契约校验器
                 （只覆盖类属性期望值）；先不加 AMP，可独立训练
③ 速度指令       移植 DirectionalVelocityCommandCfg（§3.5），替掉 UniformVelocityCommand
④ AMP 观测组     构造 amp_policy / amp_reference（anchor-relative 13 body），校验同构
⑤ 奖励迁移       AMP_mjlab 奖励函数移植 + 权重对齐（§3.1）
⑥ 算法接线       WasabiPPO + WasabiPPOAmpMix（reward_mix_mode="lerp" + style_lerp），参数映射（§4.1）
⑦ 注册与冒烟     注册新任务 ID；小规模验证（motion_reference 采样 / 判别器前向 / 奖励量级）
```

### 6.1 ⚠️ 本地验证的边界（重要）

本机显存 4 GB，**跑不了 Isaac Sim**（最低 8 GB），所以本地**无法**做真实冒烟测试。
因此每一阶段的验收方式定为：

| 阶段 | 本地可做 | 必须上云才能做 |
|---|---|---|
| ① 数据转换 | ✅ 已全部完成（FK 误差 ≤1e-6 m，可数值验证） | — |
| ②~⑥ 配置与算法 | ✅ **静态契约校验**（`instinct/tools/check_amp_walk_task.py`） | 真实 `gym.make()` + 前向 |
| ⑦ 冒烟 | ⚠️ 只能静态 | ✅ 完整 |

`check_amp_walk_task.py` 做 7 类检查（A 维度算术 / B 字段顺序 / C 符号真实性 /
D 注册一致性 / E 契约继承 / F 配置接线 / G 射线数独立复算），其中 **C 类最有价值**：
它用「该 `<module>.<name>` 必须在仓库其它文件中已出现过」来判定符号真实性，能抓住
`instinct_mdp.base_lin_vel` 这种**看起来对、到 Isaac Sim 里才 `AttributeError`**
的错误（`instinctlab.envs.mdp` 只转发了 `base_ang_vel`，**没有**转发 `base_lin_vel`）。

---

## 7. 一句话总结

> 学长的 `instinct_xu` 已经把 AMP 算法（`wasabi.py`）和参考动作基建（`motion_reference` + `*_reference_as_state`）都备好了，只是**没用起来**。我们只需**新建一个行走任务**：actor 观测在 OmniLoco 的 623 维基础上加 3 维 `base_lin_vel`，补上 `amp_policy`/`amp_reference` 两组观测，算法从 `PPO` 换成 `WasabiPPO`，并用一个**零侵入的 `reward_mix_mode="lerp"`** 精确复刻 AMP_mjlab 的 `(1-λ)·风格 + λ·任务` 混合。
>
> 三处迁移要点：**奖励混合**（模式 + 权重两参，一行奖励函数都不改）、**运动数据集选取**（按命令范围挑 clip，保证每方向参考密度）、**速度指令采样**（换方向分档，否则纯侧向命令采不出来）。
>
> 原以为“最大的实际工作量在参考动作数据格式转换”——**实测后已被证伪**：转换器一天内写完并全部通过 FK 验证（误差 ≤1e-6 m），四元数无需翻转、速度无需手工补、关节序无需重排。真正的风险集中在**数据集速度包络**与**奖励混合的数值一致性**上。
