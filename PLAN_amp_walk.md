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

⚠️ **本节已修正**。原来的方案（2025-09-23 之前）打算照搬 AMP_mjlab 的 *「anchor 相对的 13 body 位姿」*
表示法——那需要新写一批 link 级观测函数并自己处理 anchor 帧，工作量大、风险高。

后来在本地发现 `instinct/repos/InstinctLab` 下有 **Hiking in the Wild 的官方实现**
（`tasks/parkour/config/parkour_env_cfg.py` 的 `AmpPolicyStateObsCfg` / `AmpReferenceStateObsCfg`），
它给出了 InstinctLab 生态里 AMP 状态向量的**权威写法**。现在改为完全采用它：

| 项 | 单帧 | 历史 | flatten | scale |
|---|---|---|---|---|
| `projected_gravity` | 3 | 10 | ✗（所以维度是 (10,3)） | — |
| `joint_pos_rel` | 29 | 10 | ✔ | — |
| `joint_vel_rel` | 29 | 10 | ✔ | **0.05** |
| `base_lin_vel` | 3 | 10 | ✔ | — |
| `base_ang_vel` | 3 | 10 | ✔ | — |

合计 **670 维**，两侧逐项同构（项名/形状/缩放全部一致，只换数据源）。

**为什么这个选择更好**：
1. `reference_as_state.py` 已经把这些项**全部**准备好了（机器人侧 ↔ 参考侧成对），零新代码；
2. `WasabiPPO` + `wasabi.py` 就是围绕这套观测组写的（`actor_state_key="amp_policy"`/
   `reference_state_key="amp_reference"` 就是它的默认值）；
3. 与官方 parkour 实现保持一致，便于日后对照、排错。

**两个 mdp 命名空间的分工（极易写错）**：

| 符号 | 应取自 | 原因 |
|---|---|---|
| `base_lin_vel` / `joint_vel_rel`（机器人侧） | `isaaclab.envs.mdp` | `instinctlab.envs.mdp` **没有**转发这两个 |
| `projected_gravity` / `base_ang_vel` / `joint_pos_rel` | 两边都有，沿用 OmniLoco 的 `instinct_mdp` | — |
| `*_reference_as_state` 全家 | **`instinctlab.envs.mdp`** | 定义在 `envs/mdp/observations/reference_as_state.py`，isaaclab 里没有 |

⚠️ `projected_gravity_reference_as_state` 是个 `ManagerTermBase` 子类（不是普通函数），
IsaacLab 会自动实例化；它内部 `physx.create_simulation_view` 取重力方向。

- **不要把高程图放进 AMP 状态**（会稀释风格判别）
- `amp_policy` / `amp_reference` 保持 `enable_corruption=False`（判别器输入不应加噪）
- 两个组**不参与** actor/critic，步骤⑥ 接入 WASABI 后才被使用

---

## 2. 需要新增 / 修改的代码清单

> 原则：**新增为主，尽量不改学长的原文件**，以免破坏已训好的 Vault 链路。

| # | 文件 | 动作 | 说明 |
|---|---|---|---|
| 1 | `tasks/locomotion/config/g1/amp_walk_env_cfg.py` | **新增** | 继承 `G1OmniLocoEnvCfg`，覆盖：actor obs 加 `base_lin_vel`；新增 `amp_policy`/`amp_reference` 两组；替换为 AMP 奖励集；挂 `motion_reference` 场景实体 |
| 2 | `tasks/locomotion/config/g1/agents/amp_walk_ppo_cfg.py` | ✅ **已完成** | `G1AmpWalkPPORunnerCfg` 子类；`algorithm` = `AmpWalkAlgorithmCfg`（`class_name = "instinctlab.utils.wrappers.instinct_rl:WasabiPPOAmpMix"` + WASABI 参数）；`policy` = 512/256/128 |
| 3 | `tasks/locomotion/config/g1/__init__.py` | 修改 | 追加 `Instinct-Amp-Walk-G1-v0`(+ Play) 注册（**只加不删**） |
| 4 | `tasks/locomotion/mdp/rewards_amp.py` | ✅ **已完成**（新增） | 从 AMP_mjlab v5 迁移的奖励函数（见 §3.1、§3.7） |
| 5 | `tasks/locomotion/mdp/commands.py` | ✅ **已完成**（新增） | `DirectionalVelocityCommand(UniformVelocityCommand)` + Cfg（见 §3.5） |
| 6 | `tasks/locomotion/mdp/observations_amp.py` | **新增**（可选） | 若需要 AMP_mjlab 那种 anchor-relative body 量 |
| 7 | 参考动作 npz + selection yaml | **新增** | 行走参考动作（见 §3.4、§5.1） |
| 8 | 算法子类 `WasabiPPOAmpMix` | ✅ **已完成** | 实现 `additive`/`lerp` 两种奖励混合模式（见 §4.2）；**只覆盖 4 个方法**（`__init__`/`init_storage`/`process_env_step`/`compute_auxiliary_reward`）。落在 `instinctlab/utils/wrappers/instinct_rl/algorithms.py`，走**方案 A** |
| 9 | WASABI / 算法 `@configclass` 子类 | ✅ **已完成** | `AmpWalkAlgorithmCfg(InstinctRlPpoAlgorithmCfg)` 显式声明 WASABI 字段 + 混合模式字段（`@configclass` 是 dataclass，**额外字段必须显式声明**） |

**不需要改**：`wasabi.py`、`ppo.py`、`discriminator.py`、`amp_storage.py`、`OmniLoco` 三个原任务，以及**全部奖励函数与权重**。

**`instinct_rl` 的唯一改动**：`runners/on_policy_runner.py` **1 行**（`importlib.import_module()`
漏传参数 → 算法 `class_name` 写 `"module:Class"` 会 `TypeError`，见 §4.2）。
除此之外**没有往 `instinct_rl` 里加任何文件**。

---

## 3. 从 AMP_mjlab 迁移什么 / 不迁移什么

### 3.1 奖励：迁移（你的要求 3）—— **已实施（步骤⑤）**

实现位置：`instinct_xu/.../tasks/locomotion/mdp/rewards_amp.py`（新增）+ `amp_walk_env_cfg.py::G1AmpWalkRewardsCfg`。
**不新建任何奖励函数的情况**：`track_lin_vel_xy_yaw_frame_exp` / `track_ang_vel_z_world_exp` /
`instinct_mdp.contact_slide` 三者直接复用，只改权重。

| AMP_mjlab 奖励项 | 权重 | 落地结果 |
|---|---|---|
| `track_anchor_linear_velocity` | 1.5 | ✅ 复用 `track_lin_vel_xy_yaw_frame_exp`，权重 1.0 → **1.5**（误差范数在偏航旋转下不变，语义等价） |
| `track_anchor_angular_velocity` | 1.2 | ✅ 复用 `track_ang_vel_z_world_exp`，权重 1.0 → **1.2** |
| `track_root_height` | 1.5 | ✅ 新写 `track_base_height_exp`（weight **+1.5**, std 0.3，目标高度 = `default_root_state[z]` = 0.82） |
| `body_ang_vel_xy_l2` | 0.7 | ✅ 新写 `track_root_ang_vel_xy_exp`（weight **+0.7**, std 3.14）。⚠️ 原名带 `l2` 但**其实是 exp 正奖励** |
| `track_default_pose_on_stand` | −1.0 | ✅ 新写同名函数（L1 + 平滑门控 `gate_tau=0.25` + `waist_yaw_joint: 3.0`） |
| `feet_lateral_width` | −10.0 | ✅ 新写 `feet_lateral_width_cost`（**`sep_floor` 已按本仓库实测重标定为 0.145**，见 §3.7） |
| `feet_crossing` | −20.0 | ✅ 新写 `feet_crossing_cost`。⚠️ **有意偏离**：额外加了方向门控，见 §3.7 |
| `foot_slip` | −0.25 | ✅ 复用 `instinct_mdp.contact_slide`（`feet_slide`），权重 −0.1 → **−0.25** |
| `self_collisions` | −0.1 | ❌ **不迁移**。IsaacLab 的 `ContactSensorCfg.filter_prim_paths_expr` 在 PhysX 上**只支持一对多**（官方 attention：`prim_path` 必须只匹配每环境**一个** prim），所以 `Robot/.*` 过滤 `Robot/.*` 这种自碰撞检测做不出来；且本任务已有更强兜底：资产 `enabled_self_collisions=True` + `G1FlatTerminationsCfg.base_contact` 对躯干/手臂/髋/膝接触直接 **−200 并终止** |
| `torso_posture` / `feet_swing_clearance` / `soft_landing` | 已注释 | ⬜ 不迁（AMP_mjlab 自己也没用） |

**保留 instinct_xu 原有**的（权重原样）：`termination_penalty(-200)`、`stand_still(-0.8)`、
`joint_deviation_*`、`action_rate_l2`、`dof_acc_l2`、`dof_torques_l2`、`dof_pos_limits`、
`flat_orientation_l2`、`lin_vel_z_l2`、`feet_air_time(1.0)`。

> 「不改奖励函数与权重」这条约束的落地方式：**基类 `G1FlatRewardsCfg` 一行不改**，
> 全部改动都在子类 `G1AmpWalkRewardsCfg(G1FlatRewardsCfg)` 里 —— 覆盖 3 项权重 + 新增 5 项。
> `check_amp_walk_task.py` 的 **J 类检查**会逐项断言基类权重未变、子类未偷改其它项权重。


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

**具体数值按 §3.4 实测包络取**（已实现于 `tasks/locomotion/mdp/commands.py` 与
`amp_walk_env_cfg.py::AMP_WALK_COMMAND_RANGES`，并由 `check_amp_walk_task.py` 的 H 类检查
自动对账）：

| 参数 | 实际取值 | 依据 |
|---|---|---|
| `lin_vel_x` | $(-1.0,\ +1.2)$ m/s | 数据集 $[-1.07, +1.69]$，完全在包络内 |
| `lin_vel_y` | $(-0.5,\ +0.5)$ m/s | 数据集 $[-0.48, +0.55]$，✅未超幅值上限 0.55 |
| `ang_vel_z` | $(-0.8,\ +0.8)$ rad/s | 数据集 $[-0.77, +0.99]$，✅未超幅值上限 0.99 |
| `heading` | **`heading_command=False`** | ⚠️ 必须关。若开启，`_update_command` 会用航向误差**重算** $\omega_z$，把方向分档采样到的 $\omega_z$ 覆盖掉；且父类硬编码校验也禁止航向命令 |
| `straight_lin_vel_range` | $(0.5,\ +1.2)$ m/s | 持续直线前向 |
| `straight_ang_vel_range` | $(-0.1,\ +0.1)$ rad/s | ≈0，不转向 |
| `back_lin_vel_range` | $(-1.0,\ -0.5)$ m/s | 持续直线后退 |
| `back_ang_vel_range` | $(-0.1,\ +0.1)$ rad/s | ≈0，不转向 |
| `prob_low / prob_side / prob_lin / prob_back` | 0.15 / 0.25 / 0.25 / 0.10 | 之和 0.75，剩余 0.25 为均匀斜向 |
| `rel_standing_envs` / `rel_heading_envs` | `0.0` / `0.0` | 站立语义交给 `prob_low`，避免两套站立机制打架 |
| `resampling_time_range` | $(10,\ 10)$ s | 与 OmniLoco / AMP_mjlab 一致 |

### 3.6 实现上的最小假设设计（重要）

`DirectionalVelocityCommand._resample_command` 的做法是：

```
super()._resample_command(env_ids)   # 父类先填好默认均匀命令 + 维护 standing/heading
  ↓
按抽签结果**覆盖** self.vel_command_b 中属于各模式的行
```

好处：只需依赖父类的**一个**内部名（`vel_command_b`）与 `cfg.ranges.*`，
**不碰** `is_standing_env` / `is_heading_env` / `heading_target` / `self.robot`。
即使 IsaacLab 版本对它们做改动，本类也不会失效。
`__init__` 里另有 fail-fast 契约校验（检查 `vel_command_b` 是否存在 +
`heading_command`/`rel_standing_envs` 是否满足前提），把问题变成**启动即报错**。

**刻意不移植** AMP_mjlab 的两部分（均为 mjlab 专有 API）：
`init_velocity_prob`（直接把命令速度写进 root state）与 `quat_apply` 世界系换算。

### 3.7 步宽地板 / 交叉惩罚的**实测重标定**（步骤⑤ 的关键发现）

AMP_mjlab 的 `sep_floor = 0.15` 是在**他们自己的模型 + `left_foot`/`right_foot` site** 上
标定的，且只用了**一段前向 clip**（`A022`）。搬到本仓库后参考点、基准帧、数据集都不同，
所以工具 `instinct/tools/calibrate_foot_sep.py` 用**与运行时奖励函数完全相同的量**重标：

$$\text{sep}(t) = \Big[R_{\text{yaw}}\big(q_{\text{root}}\big)^{\top}\,\big(p_{\text{LL\_FOOT}}(t) - p_{\text{LR\_FOOT}}(t)\big)\Big]_y$$

脚部参考点用 URDF 里的 **`LL_FOOT` / `LR_FOOT`**（挂在 `*_ankle_roll_link` 下、偏移
`(0.04, 0, −0.037)`，几乎就是 mjlab `left_foot`/`right_foot` site 的同源物；
实测同一帧下两者世界坐标一致，且横向偏移为 0，所以退化成 ankle body 也几乎不变）。

**实测结果**（17 段 / 7417 帧，输出见 `instinct/tools/amp_walk_foot_sep_stats.json`）：

| 种群 | sep min | p1 | 中位 | sep<0 帧占比 |
|---|---|---|---|---|
| **直线行走** 4 段（前/后 ×2） | **+0.1387** | 0.1455 | 0.2140 | **0.000** |
| 交叉式侧步 2 段（`walk_sideway_*`） | **−0.1503** | −0.144 | 0.2234 | **0.210** |
| 慢跑 4 段 | +0.040 | — | ~0.21 | 0 |
| 弧线 / 原地转 / 踏步 | ≥ +0.12 | — | — | 0 |

（中位 0.2140 与 AMP_mjlab 报告的 0.199 同量级，说明测量管线与他们的口径一致；
用他们的模型 + pelvis 偏航复算 `A022` 得到 p1 = 0.1584 / 中位 0.1990 / sep<0 = 0，
**逐字复现了他们 §v3.5 的标定注记**。）

由此得到两个**必须偏离 AMP_mjlab** 的结论：

1. **`sep_floor` 取 0.145，不是 0.15** —— 取直线行走 p1 向下取整，保证健康步态被罚的帧 ≲1%。
2. ⚠️ **`feet_crossing` 必须加方向门控 `w_dir = |vx|/‖v‖`（AMP_mjlab 原文是不加的）**：
   本仓库参考动作的侧移 clip 用的是**交叉式侧步（grapevine）**，21% 的帧 `sep < 0`，
   这是参考动作**本身的风格**，不是缺陷。对纯侧向指令无条件施加 −20 的交叉惩罚
   会与 AMP 风格先验正面对撞（判别器说「像参考」、奖励说「不许」），直接压制侧移能力 ——
   正是 §3.4 记录的那个坑。宽度项 AMP_mjlab 本来就有 `gate_lateral`，这里让两项共用。

> ⚠️ 踩坑记录：标定脚本第一版忘了**按关节名重映射列序**（npz 的 `joint_pos` 列序 ==
> AMP_mjlab MJCF 的 hinge 顺序，**腿在前**；本仓库 MJCF 是**腰在前**），直接按列喂进去
> 得到的是完全错误的姿态，凭空把 sep 从 +0.15 算成 −0.25、伪造出「7.9% 真交叉」。
> 修正后两侧（本仓库 MJCF / AMP_mjlab MJCF）在同一 clip 上结果一致。
> 这是本项目第三次踩「同名不同序」的坑（前两次见 §5.1）。

---

## 4. AMP 算法接线的关键差异（⚠️ 最容易出错的地方）

### 4.1 两套 AMP 实现不同，参数不能照搬 —— **已实施（步骤⑥）**

最终取值位于 `agents/amp_walk_ppo_cfg.py::AmpWalkAlgorithmCfg`：

| AMP_mjlab（rsl_rl fork） | instinct_xu（WASABI） | 实际取值 |
|---|---|---|
| `class_name="AMPPPO"` | —— | `"instinctlab.utils.wrappers.instinct_rl:WasabiPPOAmpMix"`（自建子类，见 §4.2） |
| `amp_reward_coef=0.1` | `discriminator_reward_coef`（默认 1.0） | **0.1**（AMP_mjlab v5 的值；刻意取弱，见下） |
| `amp_task_reward_lerp=0.75` | 自建 `style_lerp` | **0.75**，`reward_mix_mode="lerp"`（§4.2） |
| `amp_discr_hidden_dims=[1024,512,256]` | `discriminator_kwargs` | **`{"hidden_sizes": [1024, 512], "nonlinearity": "ReLU"}`** |
| `amp_anchor_name` / `amp_body_names` | `amp_policy` / `amp_reference` obs group | 5 项 × 10 帧 = 670 维（§1.4） |
| `amp_num_preload_transitions=200000` | 无 | 略（instinct_xu 逐步在线提供参考状态） |
| `amp_disable_style_on_stand` | 自建同名开关 | 实现完成，**本任务默认关**（见下） |
| `discriminator_loss_func` | `"BCEWithLogitsLoss"` / `"MSELoss"` | **`"MSELoss"`**（AMP_mjlab 用法，§4.3） |
| （AMP_mjlab 的 `amp_reward` 形式） | `discriminator_reward_type` | **`"quad"`** |

**两个关键核实结果**（也是与「照抄上游 parkour」不一致的地方）：

1. **`discriminator_reward_type` 必须是 `"quad"`，不是 `"log"`**。
   AMP_mjlab 的判别器奖励就是
   `amp_reward_coef * clamp(1 - (1/4)*(d-1)^2, min=0)`
   （`rsl_rl/modules/discriminator.py`），即 WASABI 里的 `"quad"`。
   巧合的是**上游 parkour 也用 `"quad"`**，两边一致，不用取舍。
2. **`discriminator_kwargs` 的键名是 `hidden_sizes`（复数），不是 `hidden_dims`**。
   `Discriminator.build_model` 直接 `MlpModel(input_size=..., output_size=..., **kwargs)`，
   而 `MlpModel.__init__(input_size, hidden_sizes, output_size=None, nonlinearity=ReLU)`
   —— `hidden_sizes` 是**必填参数**，写错/漏掉会在 `init_storage` 里 TypeError。

**几个有意取值的理由**

- `discriminator_reward_coef = 0.1`（上游 parkour 用 0.25）：AMP_mjlab 的实测教训是
  「风格先验必须先弱」——他们一度从 0.1 降到 0.05，因为强风格先验会压住任务奖励。
  取弱之后实际权重是 `0.025·r_amp + 0.75·r_task`（风格/任务 = 0.033）；
  跑起来后看 TensorBoard 的 `Step/amp_style_reward` 与任务奖励量级再决定是否上调。
- `disable_style_on_stand` **默认关**：本任务的速度指令有 `prob_low` 档会给精确 `(0,0,0)`，
  而 `track_default_pose_on_stand` 已经用 `gate_tau=0.25` 的平滑门控处理了低速回位；
  两套机制叠加容易互相拉扯。需要时一个 bool 就能打开（实现已就绪）。
- `clip_min_std = 0.05`：等价于 AMP_mjlab 的 `min_normalized_std=[0.05]*29`
  （instinct_rl 的 `ActorCritic.std` 是绝对标准差，`init_noise_std=1.0` 时两者数值相同）。
  防止策略过早塌成确定性策略、让判别器单方面主导。

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

**放置位置（已核实，有个坑）—— 最终采用方案 A**

我查了 `instinct_rl/runners/on_policy_runner.py`：
```python
alg_class = importlib.import_module() if ":" in alg_class_name else getattr(algorithms, alg_class_name)
```
`importlib.import_module()` **调用时没传参数** —— 即 **算法 `class_name` 写成 `"module:Class"` 会直接 `TypeError`**。
（`policy_class_name` 和 `discriminator_class_name` 的 `module:class` 分支都是正常的，**只有算法这一处坏了**。）

两个选择：

- ✅ **A（已采用）**：把这一行修成与 `build_actor_critic` 一致的形式，`WasabiPPOAmpMix` 放在 instinctlab 侧 ——
  `instinct_rl` 只动这 **1 行**，且**不新增任何文件**：
  ```python
  if ":" in alg_class_name:
      module_name, class_name = alg_class_name.split(":")
      alg_class = getattr(importlib.import_module(module_name), class_name)
  else:
      alg_class = getattr(algorithms, alg_class_name)
  ```
- **B（未采用）**：把 `WasabiPPOAmpMix` 放进 `instinct_rl/algorithms/` 并加进 `__init__.py`，完全不碰 runner ——
  但它相当于承认那个 bug 一直存在，且「新增文件 + 改 `__init__.py`」的足迹并不比 A 小。

**A 的落点**：`instinctlab/utils/wrappers/instinct_rl/algorithms.py::WasabiPPOAmpMix`，
由该包的 `__init__.py` 显式转发，配置里写
`class_name = "instinctlab.utils.wrappers.instinct_rl:WasabiPPOAmpMix"`。

为什么落在这里：该包已经是「instinctlab ↔ instinct_rl 桥接层」，且**已经同时容纳配置类与运行时类**
（`vecenv_wrapper.py::InstinctRlVecEnvWrapper`），放一个算法运行时类不引入任何新概念 ——
比在 `instinctlab` 下新开一个与 `instinct_rl/algorithms` 平行的顶层包更保守。

**最终实现与本节的伪代码有 3 处刻意不同**（以代码为准，均为了可调试性与 fail-fast）：

1. `disable_style_on_stand` 的指令切片不是「运行时查 `_obs_format`」，而是
   `init_storage` 里解析一次（`get_obs_slice`），且**只允许精确名 / 唯一含 `command` 的项**两级匹配 ——
   更模糊的猜测会把别的观测当指令用，属于静默错误。
2. 额外往 aux 里放一个**仅供日志**的 `amp_style_reward`（真正的 `r_amp`）。
   因为 lerp 模式下 `discriminator_reward` 这个键承载的是「lerp 修正量」而不是风格奖励，
   不给真值就没法在 TensorBoard 里判断风格项量级。
   ⚠️ 关键点：`PPO.process_env_step` 是 `if coef != 0.0: 才加进奖励`，日志那行在 `if` **外面**，
   所以只要把 `amp_style_reward_coef` 钉成 `0.0`，它就**只进日志、不进奖励**。
3. `init_storage` 里加三条 fail-fast：`num_rewards != 1` / `discriminator_reward_coef == 0` /
   定位不到指令项 —— 全部变成**启动即报错**。

> ⚠️ `auxiliary_reward_per_env_reward_coefs` 必须显式设 `[1.0]`（单奖励组）。
> `PPO.__init__` 收到**空列表**时会把系数退化成标量 `1.0`；显式 `[1.0]`（→ shape `(1,1)`）
> 才与 `transition.rewards` 的 `(num_envs, num_rewards)` 严格广播。

> ⚠️ 另一个咬合点：`PPO.process_env_step` 用 `getattr(self, f"{k}_coef", 1.0)` 找系数，
> 其中 `k = "discriminator_reward"` → 必须命中 `WasabiAlgoMixin.__init__` 的
> `discriminator_reward_coef` 参数。**若这个名字改了，系数会静默退化成 1.0，整套 lerp 差一个因子 c**
> —— 所以检查器 K 类专门断言了这个咬合。

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
| 4 | **`@configclass` 额外字段** | WASABI 参数必须在算法 cfg 子类里声明，否则 runner 传不下去。**已核实**：本仓库大量使用**不带注解**的字段声明（`G1FlatSceneCfg.terrain`、`G1OmniLocoSceneCfg.height_scanner`、以及上游 parkour 的 `AmpAlgoCfg.discriminator_kwargs`），IsaacLab 的 `@configclass` 会把它们收成字段、`to_dict()` 也会带上 → 可以照抄上游风格（无需加类型注解） |
| 5 | **参考动作采样窗口** | `motion_bin_length_s` / `motion_start_from_middle_range`：行走是循环步态，**不要用 RSI 随机起点**（会破坏步态相位连续性），建议整段连续播放 |
| 6 | **AMP 状态不要含高程** | 否则判别器在判别地形而不是判别风格 |
| 7 | **PPO hidden dims 不一致** | instinct_xu 行走用 [256,128,128]、越障用 [512,256,128]；AMP 行走建议 **512/256/128**（与 AMP_mjlab 一致） |
| 8 | **License** | instinct_xu 继承了 InstinctLab 的 CC BY-NC 4.0 |
| 9 | **数据集覆盖面决定 AMP 压制什么** | 参考 clip 缺某方向（侧向/后退）时，判别器会把该方向姿态判为“不像参考”而抑制。**改命令分布无法解决（已实测）**，只能补参考数据。见 §3.4 |
| 10 | **`UniformVelocityCommand` 采不出纯侧向命令** | instinct_xu OmniLoco 用的就是 Uniform（三轴独立均匀）→ x 恰为 0 概率为零 → 左右直移学不出来。见 §3.5 |
| 11 | **runner 没法用 `module:class` 指定算法** | ✅ **已解决（方案 A）**：`on_policy_runner.py` 那一行修成与 `build_actor_critic` 一致的形式；算法子类落在 `instinctlab/utils/wrappers/instinct_rl/algorithms.py`，于是**不需要往 `instinct_rl` 里加文件**。见 §4.2 |
| 12 | **`feet_crossing` 的方向门控不能关** | 参考动作的侧移 clip 是交叉式侧步，21% 帧 sep<0（§3.7 实测）。关掉 `gate_lateral` 会与 AMP 风格先验对撞，压制侧移能力 |
| 13 | **`sep_floor` 必须用本仓库实测值** | 参考 body / 基准帧 / 数据集都不同，用 AMP_mjlab 的 0.15 会误伤本仓库 1% 的健康步态帧。实测值见 §3.7 |
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
③ 速度指令       【已完成】新增 `tasks/locomotion/mdp/commands.py`
                 `DirectionalVelocityCommand`（继承 `UniformVelocityCommand`）+ Cfg
                 参数实测对账通过（H 类检查：命令范围均在参考动作包络内）
④ AMP 观测组     【已完成】`amp_policy` / `amp_reference`（5 项 × 10 帧 = 670 维，逐项同构）
                 采用官方 parkour 的表示法（非 AMP_mjlab 的 13-body 位姿，见 §1.4）
                 新增 G1AmpWalkSceneCfg：+ robot_reference + motion_reference 传感器
                 行走循环步态：已关 RSI，motion_bin_length_s = None
                 I 类检查自动验证两侧同构与维度算术
⑤ 奖励迁移       【已完成】新增 `tasks/locomotion/mdp/rewards_amp.py`
                 + `G1AmpWalkRewardsCfg(G1FlatRewardsCfg)`（覆盖 3 项权重 + 新增 5 项）
                 步宽地板/交叉惩罚已用 `instinct/tools/calibrate_foot_sep.py` **实测重标定**（§3.7）
                 `self_collisions` 经核实**不可迁移**（PhysX 过滤限制，见 §3.1）
⑥ 算法接线       【已完成】
                 - 修 `instinct_rl/runners/on_policy_runner.py` **1 行**（`importlib.import_module()`
                   漏传参 → `"module:Class"` 形式的算法类会 TypeError）
                 - 新增 `instinctlab/utils/wrappers/instinct_rl/algorithms.py::WasabiPPOAmpMix`
                   （只覆盖 4 个方法；`reward_mix_mode` / `style_lerp` / `disable_style_on_stand`）
                 - `agents/amp_walk_ppo_cfg.py`：`AmpWalkPolicyCfg`(512/256/128) +
                   `AmpWalkAlgorithmCfg`（WASABI 判别器参数 + lerp 混合 + `clip_min_std`）
⑦ 注册与冒烟     【已完成，2026-09-25 云端 GPU 实测全绿】
                 `bash /root/cloud_run_bg.sh 'bash /root/cloud_chain_smoke.sh' /root/smoke.log`
                 --task Instinct-Amp-Walk-G1-v0 --num_envs 16 --max_iterations 2 --device cuda:0
                 结果：训练退出码 0；[AMP-WALK PASS] rewards=21；Actor 626→512/256/128→29；
                 Critic 99→512/256/128→1；Discriminator 670→1024/512→1；
                 Step/foot_sep_mean = 0.2411（数据集直线走中位 0.2140）；
                 Step/foot_cross_frac = 0.0000；amp_style_reward = 0.7583；
                 奖励混合实测对账：r_task + c·disc = λ·r_task + (1-λ)c·r_amp ✅
                 详见 CLOUD_ENV_SETUP.md §10（含冒烟暴露并修掉的 2 个真 bug）
```

> ⚠️ **步骤⑦ 实测又清掉两个断言之外的东西**（都不是环境问题，而是我们自己的 bug）：
>
> **1) 诊断量写进了会被整体替换的日志通道** —— 第一轮冒烟在 `OnPolicyRunner.log()`
> 里 `KeyError: 'Metrics/foot_sep_mean'`，**训练根本起不来**。根因是 IsaacLab 的
> `ManagerBasedRLEnv._reset_idx` 里有 `self.extras["log"] = dict()`（整体替换），
> 而 reward 的 `compute()` 在它**之前**；该替换只在「本 step 有 env 终止」时执行，
> 于是 `ep_infos` 各条 dict 的 key 集合不一致，而 `log()` 是拿 `ep_infos[0]` 的 key
> 去索引**全部** dict。
> ✅ 改走 `env.extras.setdefault("step", {})[key] = value`（`extras["step"]`
> 始终是同一个 dict 对象、永不替换），TB 里表现为 `Step/<key>`。
> ⚠️ 副作用：`step_infos` 存的是同一对象的 N 个引用 ⇒ `Step/*` 是**本轮最后一个 step
> 的瞬时值**，不是轮内均值。
>
> **2) 冒烟脚本自己的检查项方向写反了** —— `[A1 PASS]` 是**父类**校验器打的，
> 我们的子类是「先 `super()` 再补 AMP 专项检查」⇒ 它与 `[AMP-WALK PASS]` 同时出现
> **属正常**，且报的是**当前类**的 626（不是 623）。已把判据改成
> 「A1 报的维度是否 = 626」，并新增 `foot_sep_mean` 在线自检（不在 0.05~0.6 就告警）。
>
> ⭐ 顺带被数值自证的一件事：**`LL_FOOT`/`LR_FOOT` 找不到是预期的**。Kit 日志原文
> `link LR_FOOT ... is being merged into right_ankle_roll_link`（URDF 导入器的
> `merge_fixed_joints`）⇒ 退路 `*_ankle_roll_link` 就是脚部参考点。
> 实测 `foot_sep_mean = 0.2411` 落在数据集区间内 ⇒ **§3.7 里 `sep_floor=0.145`
> 可以直接用，不用再重标定**。

### 6.1 ⚠️ 本地验证的边界（重要）

本机显存 4 GB，**跑不了 Isaac Sim**（最低 8 GB），所以本地**无法**做真实冒烟测试。
因此每一阶段的验收方式定为：

| 阶段 | 本地可做 | 必须上云才能做 |
|---|---|---|
| ① 数据转换 | ✅ 已全部完成（FK 误差 ≤1e-6 m，可数值验证） | — |
| ②~⑥ 配置与算法 | ✅ **静态契约校验**（`instinct/tools/check_amp_walk_task.py`） | 真实 `gym.make()` + 前向 |
| ⑦ 冒烟 | ⚠️ 只能静态 | ✅ 完整 |

`check_amp_walk_task.py` 做 11 类检查（A 维度算术 / B 字段顺序 / C 符号真实性 /
D 注册一致性 / E 契约继承 / F 配置接线 / G 射线数独立复算 /
H **速度指令包络 vs 参考动作实测包络对账** /
I **AMP 两侧状态逐项同构** /
J **奖励集迁移对账**：基类权重零改动、迁移表逐项落地、未偷改其它项、
`sep_floor`/`walk_max_lin_vel` 与①产出实测数据数值对账、`asset.data.*` 属性有仓库先例 /
K **算法接线对账**：runner 的 `module:class` bug 已修、自定义算法类可解析到、
只覆盖白名单方法（零侵入）、cfg 字段 ↔ 构造参数咬合、`hidden_sizes` 必填、
aux 系数名咬合、`amp_style_reward_coef=0`、以及**把 λ/c 代进两条公式做数值恒等对账**），
其中 **C 类最有价值**：
它用「本仓库模块**直接查定义**（跟随显式 `from .x import *` 转发）+ 外部模块查仓库先例」
来判定符号真实性，能抓住 `instinct_mdp.base_lin_vel` 这种**看起来对、到 Isaac Sim 里才
`AttributeError`** 的错误（`instinctlab.envs.mdp` 只转发了 `base_ang_vel`，
**没有**转发 `base_lin_vel` / `joint_vel_rel`）。
**H 类**是唯一做跨产物数值对账的检查：拿①产出的 npz 重算速度包络，
再验证②③配置里的命令范围有没有超出去。
**I 类**防的是「改了 history/flatten/scale 导致两侧不同构 → WASABI `init_storage` 断言失败」
这类隐式破坏。

---

## 7. 一句话总结

> 学长的 `instinct_xu` 已经把 AMP 算法（`wasabi.py`）和参考动作基建（`motion_reference` + `*_reference_as_state`）都备好了，只是**没用起来**。我们只需**新建一个行走任务**：actor 观测在 OmniLoco 的 623 维基础上加 3 维 `base_lin_vel`，补上 `amp_policy`/`amp_reference` 两组观测，算法从 `PPO` 换成 `WasabiPPO`，并用一个**零侵入的 `reward_mix_mode="lerp"`** 精确复刻 AMP_mjlab 的 `(1-λ)·风格 + λ·任务` 混合。
>
> 三处迁移要点：**奖励混合**（模式 + 权重两参，一行奖励函数都不改）、**运动数据集选取**（按命令范围挑 clip，保证每方向参考密度）、**速度指令采样**（换方向分档，否则纯侧向命令采不出来）。
>
> 步骤⑤ 补上第四条：**阈值类奖励必须在本仓库重新标定，不能照抄**。步宽地板看似只是一个数，
> 但它同时取决于「脚部参考点」「root 基准帧」「参考数据集」三件本仓库都不同的事；
> 实测后 `sep_floor` 从 0.15 改成 0.145，而且发现**参考动作的侧移 clip 本身就是交叉式侧步**
> （21% 帧 sep<0）→ `feet_crossing` 必须加方向门控，否则奖励会和 AMP 风格先验正面对撞。
> 这类「照抄一个数」的错误在静态检查里是看不出来的，只能靠**把参考数据算一遍**。
>
> 步骤⑥ 补上第五条：**先确认目标公式，再谈参数**。AMP_mjlab 的判别器奖励是
> `clamp(1-0.25(d-1)², 0)`（= WASABI 的 `"quad"`）而不是默认的 `"log"`；
> 而 `discriminator_kwargs` 的键名是 `hidden_sizes`（`MlpModel` 的**必填**参数），
> 写错就会在 `init_storage` 里 TypeError。这两条都是「看源码/看常量」才能定的，
> 不看就会静默地换一个目标函数、或直接崩在启动期。
> lerp 混合那边更隐蔽：`PPO.process_env_step` 是用 `getattr(self, f"{key}_coef", 1.0)`
> 去揀系数的 —— **键名与参数名一旦不咬合，系数会静默变成 1.0，整套公式差一个因子 $c$**。
>
> 原以为“最大的实际工作量在参考动作数据格式转换”——**实测后已被证伪**：转换器一天内写完并全部通过 FK 验证（误差 ≤1e-6 m），四元数无需翻转、速度无需手工补、关节序无需重排。真正的风险集中在**数据集速度包络**与**奖励混合的数值一致性**上。

---

## 8. 实测结果与正式训练配置（步骤⑦⑧，2026-09-25）

### 8.1 步骤⑦：云端端到端冒烟 —— **全绿**（commit `81de929`）

`--num_envs 16 --max_iterations 2 --device cuda:0`：训练**退出码 0**、
`[AMP-WALK PASS] policy=626 rewards=21`、判别器建成（670→1024/512→1）、
`Step/foot_sep_mean = 0.2411`、`foot_cross_frac = 0`。

冒烟抓到 **1 个真 bug**（会让训练在首轮迭代就崩）：诊断量原先写
`env.extras["log"]`，而 IsaacLab 的 `ManagerBasedRLEnv._reset_idx` 会把它
**整体替换**（且只在「本 step 有 env 终止」时执行），于是 `ep_infos` 各条 dict
的 key 集合有的带、有的不带，而 `OnPolicyRunner.log()` 是拿 `ep_infos[0]` 的 key
去索引**全部** dict → `KeyError`。改走 `env.extras["step"]`（同一 dict 对象、
永不替换）。**细节与「Step/* 是瞬时值不是轮内均值」的副作用见
`CLOUD_ENV_SETUP.md` §10.3。**

### 8.2 规模试跑与 `num_envs` 决策

| num_envs | 显存峰值 | 稳态迭代耗时 | 吞吐 |
|---|---|---|---|
| 512 | 4010 MiB | 2.23 s | ≈5 500 步/秒 |
| 2048 | 7030 MiB | 3.09–3.38 s | ≈14 500–15 900 步/秒 |
| **4096** | **11578 MiB** | 3.89–4.32 s | ≈22 700–25 300 步/秒 |

拟合：$\text{显存} \approx 2843 + 2.117\,N$ MiB。
⚠️ 只用 512 单点外推会得出「4096 要 32 GB 会爆」——**固定开销占 512 那点的 75%**，
规模决策必须至少两个点（详见 `CLOUD_ENV_SETUP.md` §11.2）。

**定档**：`num_envs = 4096`（11.6 GB / 24 GB，GPU 利用率 69% → 瓶颈是 CPU 配额 14 核）。
按 ~4 s/轮，`G1OmniLocoPPORunnerCfg` 默认 **`max_iterations = 5000` ≈ 5.6 小时**。

### 8.3 步骤⑨：正式训练 —— 以及一个「假失败」的教训

**先跑了 300 轮做健康检查，结果看起来是失败的**：`Mean reward ≈ −4.12` 横盘不升、
`Mean episode length = 6.96`（0.14 秒）、`Episode_Termination/base_contact = 1.0000`
（所有回合都因躯干触地终止）。看上去像环境配错了。

**同参数 A/B 对照把这个判断推翻了**（`instinct/tools/cloud_ab_baseline.sh`，
跑学长原版 `Instinct-Unified-OmniLoco-G1-v0`，纯 PPO + 原奖励）：

| 迭代 | 原版 OmniLoco 回合长度 / reward | 我们的 AMP 回合长度 / reward |
|---|---|---|
| 0 | 8.06 / −4.54 | 4.40 / −4.25 |
| 10 | 19.27 / −5.45 | 13.09 / −4.97 |
| 20 | 6.49 / −4.36 | 6.07 / −4.41 |
| 30 | 4.03 / −4.18 | 3.83 / −4.12 |
| 40 | — | 6.96 / −4.12 |

**两条曲线几乎重合** ⇒ 环境与 AMP 接线**没有坏**，40~50 轮本来就不够（原版 `max_iterations = 5000`）。

> ⭐ **方法论：训练早期「看上去完全没学」时，先做同参数 A/B 对照，再怀疑自己的改动。**

读数陷阱（同一件事的另一面）：`Mean reward ≈ −4` 几乎全部来自终止惩罚
（`termination_penalty` 归一化后 −0.2 → episodic sum ≈ −4.0），而跟踪项归一化后只有
0.003~0.01 —— 因为回合只有 7 步、分母却是 `max_episode_length_s=20`。
**回合越短，`Episode_Reward/*` 越小，越不能据此判断奖励设计有问题。**
`Episode_Reward/feet_lateral_width ≈ −0.0001` 同理，只能说明这几步里 `sep > 0.145`。

顺带核对了判别器公式（`wasabi.py`）：`clamp(1 − 0.25·(disc−1)², min=0)`，`disc` 是线性层裸输出。
实测反推 `0.25(r_amp − r_task/c)`：`r_amp=0.29`、`r_task=−0.96`、`c=0.1`
→ `0.25×(0.29+9.6)=2.47`，与打印的 2.48 吻合 ⇒ **换算管线正确**。

**最终配置与命令**：

```bash
bash /root/cloud_run_bg.sh 'bash /root/cloud_train_amp_walk.sh 4096 5000 cuda:0' /root/train_full_driver.log
```

`num_envs=4096`、`max_iterations=5000`、`save_interval=500`（可 `--resume` 续训），预计 ≈5.6 h。

**判据要放在迭代 500 / 1000 / 2000 处看，不要在几十轮时下结论**：

| 指标 | 期望 | 不健康的含义 |
|---|---|---|
| `Mean episode length` | 明显上升（几十→几百→上千） | 停滞在个位数 → 真的学不动 |
| `Episode_Termination/base_contact` | 降到 0.1 以下 | 恒为 1.0 → 站都站不住 |
| `Train/mean_reward_0` | 上升（只进 TensorBoard，控制台不打印） | 不升 → 奖励符号/混合公式错 |
| `Step/foot_sep_mean` | 稳在 **0.15–0.30** | 掉向 0 或变负 → 步宽惩罚过猛 |
| `Step/discriminator_reward` | 缓降但**不为 0** | 迅速为 0 → 降 `discriminator_reward_coef` |

### 8.4 步骤⑨ 之后（尚未开始）

1. **看策略行为**：`play.sh --task Instinct-Amp-Walk-G1-Play-v0`（本地做不了，需云端 GUI/录像）。
2. **越障奖励融合**：把 AMP 风格先验与感知越障任务合起来（用户总目标里的「单一固定越障 + 行走」）。
3. **蒸馏**：本策略作教师 → 学生（参考 `VaePpoDistill` 那条链路）。
>
> 步骤⑦ 补上第六条：**「静态检查全过」和「真能训起来」之间还隔着一层框架约定**。
> 我们写的诊断量在纯 Python 里怎么测都对，却因为 IsaacLab 的 `_reset_idx` 会把
> `extras["log"]` **整体替换**、而 `OnPolicyRunner.log()` 又假设 `ep_infos` 里各条 dict
> 的 key 集合完全相同，直接在**首轮迭代**崩掉。这类约定不看源码是猜不出来的，
> 而且症状（`KeyError`）离原因（奖励函数里的一行日志）很远。
> 教训：**往框架的共享字典里写东西前，先确认那个字典的生命周期是谁管的**。

---

## 9. 奖励集返工：v1 → v2（2026-09-25）

### 9.1 触发原因

v1（5000 迭代）出现**单脚支撑 + 跳跃**的异常步态：回合 464/1000 步、**66% 回合触地终止**、
脚距均值 0.348（超过 0.30 上限）。逐项核对 AMP_mjlab 的 `amp_loco` 奖励表后发现：

> **v1 的 21 项奖励里，有 9 项是学长做「越障」动作测试时攒下的 `G1FlatRewardsCfg` 基类奖励，
> AMP_mjlab 的 `amp_loco` 配方里根本没有**；另外 3 项权重与 AMP_mjlab 明显不一致且从未调整。

### 9.2 实测证据（全部来自云端，非推断）

工具：`instinct/tools/cloud_dump_rewards.py`（抽 TB 全量奖励项终值）、
`instinct/tools/cloud_probe_rewards.py`（核对 `RewardManager` 实际加载项）。

1. **21 项奖励无一缺项**：TB 里 `Episode_Reward/*` 恰好 21 个 tag。
   （此前一度怀疑 `feet_air_time` 没进奖励函数 —— **不成立**，它在 21 项里。）
2. **`feet_air_time` 实测仅 +0.018/s**（末段均值），而它在 `sum(in_contact)==1` 才给分。
   若真长期单脚支撑，该项应在 0.25~0.5/s ⇒ **它不可能驱动步态**。
   ⚠️ 这**推翻**了此前「`feet_air_time` 是单脚支撑次因」的推断。
3. **`action_rate_l2` 是最大单项成本 −0.646/s**，而全部任务正向项合计才 ~+1.6/s。
   v1 权重 −0.05 是 AMP_mjlab（−0.01）的 **5 倍**。跳跃恰好是「一次爆发 + 长时间不需要
   动作」的**低动作频率**策略，走路则需持续换腿 ⇒ 过强惩罚本身在鼓励跳跃。
4. 最大正项 `track_base_height` +0.59/s > `track_lin_vel_xy_exp` +0.48/s ⇒ **「稳住」比「走准」值钱**。
5. `feet_lateral_width`(−0.0014/s) / `feet_crossing`(−0.0001/s) **几乎不激活**
   （`max_cost=0.10` 封顶）⇒ 宽站姿**不会**因为 v2 自动解决。

### 9.3 v2 改了什么

新文件 `amp_walk_v2_env_cfg.py`（**不改任何现有文件、不改奖励函数本体、不动基类**）：

**（a）删 9 项**（置 `None`，这是本仓库既定机制 —— `G1FlatEnvCfg.__post_init__` 的 run_name
生成器对其中 4 项都写了 `is not None` 保护）：

| 被删项 | v1 权重 | 删除理由 |
|---|---|---|
| `feet_air_time` | +1.0 | `sum(in_contact)==1` 才给分 → **构造上奖励单脚支撑**；AMP_mjlab 无 |
| `joint_deviation_knee/hip/arms/torso` | −0.05/−0.1/−0.1/−0.1 | ⚠️ **见 §12 的更正**：当时的理由是「把关节往直腿 default 拽、与行走腾空相对抗」，**实测 default 姿态是屈膝的**（膝 0.669 rad ≈ 38°），所以这个理由**站不住** |
| `flat_orientation_l2` | −1.0 | 与新增的 `body_ang_vel_xy` 语义重叠 |
| `stand_still` | −0.8 | 与新增的 `track_default_pose_on_stand`(−1.0) 语义重叠 |
| `lin_vel_z_l2` | −0.1 | 抑制竖直速度 → **直接压制起跳/落地** |
| `dof_torques_l2` | −4e-6 | 实测 −0.018/s，纯装饰 |

**（b）对齐 3 个权重**：`action_rate_l2` −0.05→**−0.01**、`dof_pos_limits` −1.0→**−10.0**、
`dof_acc_l2` −2.0e-7→**−2.5e-7**（`func`/`params` 逐字沿用基类）。

⇒ v2 = **12 项**，与 AMP_mjlab `amp_loco` 项数、权重逐项一致。

**刻意保留的偏离**（避免一次改太多导致归因不清）：速度/角速度跟踪的 `std` 仍为 0.5
（AMP_mjlab 1.0/3.14，与学长同族教师保持形状一致且**更严**）；`dof_pos_limits` 仍只作用于
踝关节、`dof_acc_l2` 仍只作用于髋/膝（AMP_mjlab 均为全关节）。

### 9.4 验证结果（PASS）

```
[配置层] rewards 字段总数 = 21
[配置层] 被删除(None) 9 项: ['dof_torques_l2','feet_air_time','flat_orientation_l2',
        'joint_deviation_arms','joint_deviation_hip','joint_deviation_knee',
        'joint_deviation_torso','lin_vel_z_l2','stand_still']
[配置层] 保留        12 项
[运行层] reward_manager.active_terms 总数 = 12
[REWARD-SET PASS] Instinct-Amp-Walk-V2-G1-v0: active_terms=12 (expect=12)
```

12 项权重实测：`action_rate_l2 −0.01`、`body_ang_vel_xy 0.7`、`dof_acc_l2 −2.5e-07`、
`dof_pos_limits −10`、`feet_crossing −20`、`feet_lateral_width −10`、`feet_slide −0.25`、
`termination_penalty −200`、`track_ang_vel_z_exp 1.2`、`track_base_height 1.5`、
`track_default_pose_on_stand −1`、`track_lin_vel_xy_exp 1.5`。

### 9.5 接口契约**完全不变**（关键）

v2 只替换 `rewards` 字段，动作 29D / 顺序 / `scale=1.0` / default offset / `clip=None`、
观测 626/99/670、时序 50 Hz、执行器 —— **逐项同 `INTERFACE_CONTRACT.md`**。因此：

* checkpoint 结构与 v1 **互换**，可直接从 `model_5000.pt` 续训；
* 环境类**仍是 `G1AmpWalkTeacherEnv`**（期望值全是类属性，`rewards={n}` 只打印不断言）；
* agent cfg **复用同一份** `G1AmpWalkPPORunnerCfg`；
* **蒸馏 teacher 侧接口不受影响**。

### 9.6 启动记录

```bash
pkill -9 -f '[t]rain.py'          # 停掉 v1 续训（Step 9 未存过任何 checkpoint，无损失）
python scripts/instinct_rl/train.py \
  --task Instinct-Amp-Walk-V2-G1-v0 --num_envs 4096 --max_iterations 5000 \
  --device cuda:0 --headless --run_name amp_walk_v2_4096 \
  --resume --load_run 20260925_003514_G1AmpWalkPlane_obs626_directional_rew5_amp_walk_4096 \
  --checkpoint model_5000.pt
```

* 日志：`/root/train_amp_walk_v2_4096.log`
* run 目录：`logs/instinct_rl/g1_amp_walk/20260925_120621_G1AmpWalkV2Plane_obs626_rew12_alignAMP_amp_walk_v2_4096_from20260925_003514`
* 启动日志确认：`[AMP-WALK PASS] policy=626 ... action=29, scale=1.0 ... rewards=12` +
  `Loading model checkpoint from: .../model_5000.pt`

> ⚠️ **预期现象**：奖励语义变了 ⇒ v1 的 value function 不再准确，续训初期**回报会先掉再涨**。
> 不要把这个当作失败（参见 §8.3 的「假失败」教训）。

### 9.7 观察指标（判断 v2 是否真的修好了步态）

| 指标 | v1 末值 | 期望方向 |
|---|---|---|
| `Episode_Monitor`/触地终止比例 | 0.55~0.66 | **↓ 到 <0.1** |
| 回合长度（步） | 464 / 1000 | ↑ 到 900+ |
| `Step/foot_sep_mean` | 0.348 | 回到 0.15~0.30 |
| `Episode_Reward/action_rate_l2` | −0.65 | 绝对值应大幅下降 |
| `Policy/mean_noise_std` | 0.756 | 观察是否随步态变好而下降 |

⚠️ **v2 没碰 `init_noise_std`（仍 1.0，实际 σ≈0.76 rad ≈ ±43° 关节目标噪声）**。
这是刻意的：本次只改奖励集，保证**单变量归因**。若 v2 仍不理想，下一步（v3）按
`init_noise_std 1.0 → ~0.3` 做对照 —— 那一项同样**不碰接口契约**。

---

## 10. v2 结果：**退步**，已停训（2026-09-25）

### 10.1 结论

v2 在迭代 **7330** 处手动停训（GPU 已回落到 74 MiB，进程 0）。
**可比指标全面退步**，且暴露了一个比「奖励权重不对」更根本的病灶。

| 指标（同环境、可比） | v1 @5000 | **v2 @7330** | 判定 |
|---|---|---|---|
| 回合长度（步） | 604.7 | **560.0** | ↓ |
| 触地终止比例 | 0.5498 | **0.6772** | ↑ 更差（目标 <0.1）|
| 超时终止比例 | 0.4504 | 0.3230 | ↓ |
| **脚距均值** | 0.3518 | **0.4616** | ↑ **明显更差**（目标 0.15–0.30）|
| **动作噪声 σ** | 0.7557 | **1.5800** | ↑ **翻倍** |

（回报 5.29 → 3.05 **不可比** —— 奖励集换了，数值没有可比性，不要据此判断。）

### 10.2 ⭐ 首要病灶：σ 全程单调上升（实测）

```
Policy/mean_noise_std   末值 1.57997   全程 max 1.57997   全程 min 0.76582   （234 点）
```

从 v1 交下来的 **0.766 一路涨到 1.580 并平台化**。`action_scale = 1.0` 下
**σ = 1.58 rad ≈ ±90° 的关节目标噪声** —— 策略在**主动放大**探索噪声。

### 10.3 ⚠️ 对 §9 结论的更正（重要）

§9 说「`action_rate_l2 = −0.05` 是 AMP_mjlab 的 5 倍、在鼓励跳跃」，**只对了一半**。

`action_rate_l2 = mean((Δa)²)` 是**二次型、每一步每个关节都有梯度**的**密集**信号；
而 v2 保留的 exp 核跟踪奖励**在误差小时梯度趋近 0**。把 9 个密集惩罚项删掉、
又把 `action_rate_l2` 削到 1/5，等于**抽走了奖励集里绝大部分密集梯度**。
PPO 的 σ 是可训练参数，梯度信号变弱时它会**膨胀**（方差大 → 探索多 → 梯度估计更稳），
这是 PPO 的经典行为。

**证据链自洽**：v1 有 −0.05 时 σ 稳在 0.756；v2 削到 −0.01 后 σ 涨到 1.58，
同时脚距/摔倒率/回合长度一起恶化。

### 10.4 第二个病灶：宽站姿**结构性**无约束（独立于 σ）

`feet_lateral_width` 只在 `sep < sep_floor` 时惩罚（`clamp(sep_floor − sep, 0)`），
**对「过宽」没有任何约束**；实测贡献仅 −0.0014/s。
所以脚距 0.352 → 0.462 的恶化**不可能**靠调它的权重解决 —— 必须**新增一项对「过宽」
的惩罚**（或显著提高 AMP 风格权重，把站姿拉回参考分布）。

### 10.5 重启训练的调整方向

**保留** v2 的 9 项删除（对行走任务方向正确，尤其 `joint_deviation_*` 会把腿拽直、
与行走腾空相正面对抗）。**但必须补两件事**：

1. **把密集梯度补回来**：`action_rate_l2` −0.01 → **−0.02 ~ −0.03**
   （不必全回 −0.05，它确实也偏强）。
2. **给 σ 上限 / 降低起点**：`init_noise_std` 1.0 → **~0.3**。
   注意 `ActorCritic.clip_std(min=None, max=None)` **本身支持 max**，
   但 `PPO.update()` 只传了 `min` ⇒ **代码库里 σ 无上限**；要硬封顶需在
   **我们自己的 wrapper**（`instinctlab/utils/wrappers/instinct_rl.py` 的
   `WasabiPPOAmpMix`）里子类化覆盖，不动学长的 `instinct_rl` 核心。
3. **新增「过宽」惩罚**（见 10.4），否则宽站姿不会好。

⭐ **为什么「从零重训」是对的方向**：跳跃是**在大探索噪声下**才成立的吸引子
（支撑腿自稳、对 ±57° 目标噪声鲁棒）。**从 σ≈0.3 起步，策略根本不会进入那个吸引子**；
而从一个已经跳跃的 checkpoint 续训，很可能就停在这个局部最优里出不来。
这正是「局部收敛」的机制解释，也说明**降 σ 比换奖励更关键**。

### 10.6 保留下来的产物（停训后已确认完整）

| 产物 | 位置 | 内容 |
|---|---|---|
| v1 | `.../20260925_003514_G1AmpWalk..._4096/` | `model_500 … model_5000.pt`（10 个）|
| v2 | `.../20260925_120621_G1AmpWalkV2Plane..._v2_4096_from20260925_003514/` | `model_5500 … model_7000.pt` + `exported/{actor.onnx, policy_normalizer.npz}`（95 M）|
| 本地可回放 | `instinct/eval/v1_5000/`、`instinct/eval/v2_7000/` | 各自带配套 `policy_normalizer.npz`（**子目录隔离，不会串**）|

`g1_amp_walk` 日志总占用 **441 M**，没清。

---

## 11. v3：第二次训练（2026-09-25 启动，从零）

### 11.1 启动前的数据集核对（用户要求）

工具 `instinct/tools/analyze_amp_motions.py`（本地跑，逐 clip **实测**前后/左右/偏航分量）。

⚠️ **实际是 17 个 clip（148 s @ 50 fps），不是 ~6 个**。若「约 6 个」指**类别数**则正好对上 ——
用户列的 5 类**漏了第 6 类「侧向行走」**：

| 类别 | 用户列了 | 实测 clip（括号内是数据算出的中位速度）|
|---|---|---|
| 原地 | ✅ | `idle_turn_270`(vx 0.006, \|wz\|p90 1.17)、`idle_turn_360`(1.57)、**`step_rotate_idle_000_002`(vx 0.006/vy 0.005/\|wz\|p90 **0.019** → 近乎静止)** |
| 前向行走 | ✅ | `walk_forward_002__A022`(+0.78)、`__A024`(+0.93) |
| 前向慢跑 | ✅ | `jog_forward_003__A021`(+0.38，峰值 2.47)、`__A022`(**+1.20**) |
| 后向行走 | ✅ | `walk_backward_001__A022`(−0.62)、`__A024`(−0.60) |
| 后向慢跑 | （归在「后向」）| `jog_backward_002__A022`(−1.17)、`__003__A023`(**−1.34**) |
| 曲线行走/慢跑 | ✅ | `arc_walk_left`(1.06, \|wz\|p90 1.13)、`walk_arc_cw`(0.78, 1.06)、`arc_jog_left`(1.83, 2.26)、`jog_arc_cw`(1.60, **2.80**) |
| ⭐ **侧向行走** | ❌ **没列** | `walk_sideway_left`(vy **+0.62**)、`walk_sideway_right`(vy **−0.50**) |

**覆盖面核对（数据集实测极值 vs 训练指令范围）—— 三项全 ✅ 且有余量**：

| 轴 | 数据集实测 | 指令范围 | 判定 |
|---|---|---|---|
| vx 前后 | **[−1.91, +2.47]** | [−1.0, 1.2] | ✅ 数据**超出**指令范围 |
| vy 左右 | ±1.05 | ±0.50 | ✅ 2× 余量 |
| wz 偏航 | ±2.80 | ±0.80 | ✅ 3.5× 余量 |

⇒ **数据集是指令盒子的超集，AMP 风格先验不会和指令跟踪正面对抗** ✓
（`AmpWalkMotionCfg.velocity_estimation_method = "frontbackward"` 的注释也印证「集合里既有前向也有后退 clip」。）

### 11.2 v3 改了什么（用户确认的「全套」方案）

新文件 `amp_walk_v3_env_cfg.py` + `agents/amp_walk_v3_ppo_cfg.py`（**只新增**）：

| 改动 | v2 | **v3** | 理由 |
|---|---|---|---|
| `rewards.action_rate_l2` | −0.01 | **−0.02** | 补回密集梯度（不回到 v1 的 −0.05，那个确实偏强）|
| `policy.init_noise_std` | 1.0 | **0.3** | 关节目标噪声 **±57° → ±17°** |
| `algorithm.clip_min_std` | 0.05 | **0.015** | = 0.05×(0.3/1.0)，保持 AMP_mjlab「归一化下限 5%」语义 |

**保留** v2 的 9 项删除 + 3 项权重对齐（共 12 项，同 AMP_mjlab 配方）。
**刻意不做**（用户选择，保持单变量归因）：不加「步宽过宽」惩罚 ⇒ **脚距 0.352→0.462 这个
第二病灶本轮不解决**（`feet_lateral_width` 只罚过窄，实测 −0.0014/s，结构性无约束）。

**为什么从零而不用续训**：跳跃是**大探索噪声下才成立的吸引子**（支撑腿自稳、对 ±57° 噪声
鲁棒；交替步态依赖精确摆动腿时序，会被噪声摧毁）。从 σ≈0.3 起步**不会进入**该吸引子；
从已跳跃的 checkpoint 续训则很可能困在局部最优里出不来。这正是「局部收敛」的机制解释。

接口契约**完全不变** ⇒ checkpoint / ONNX / MuJoCo 回放 / **蒸馏 teacher 侧接口**均不受影响。

### 11.3 启动记录与核验

```bash
python scripts/instinct_rl/train.py \
  --task Instinct-Amp-Walk-V3-G1-v0 --num_envs 4096 --max_iterations 5000 \
  --device cuda:0 --headless --run_name amp_walk_v3_4096        # 注意：**不带 --resume**
```

| 核验项 | 结果 |
|---|---|
| 契约 | ✅ `[AMP-WALK PASS] policy=626, action=29, scale=1.0, rewards=12` |
| 迭代计数 | ✅ **`Learning iteration 0/5000`**（确实从零）|
| **`init_noise_std`** | ✅ **`0.3`**（读云端 `params/agent.yaml` 确认）|
| `clip_min_std` | ✅ **`0.015`** |
| run 目录 | `20260925_191029_G1AmpWalkV3Plane_obs626_rew12_ar0.02_ns0.3_amp_walk_v3_4096` |
| 日志 | `/root/train_amp_walk_v3_4096.log` |

> ⚠️ 上传 v3 时踩了一下：上传脚本把 `agents/xxx.py` 拼成了 `agents/agents/xxx.py`，
> 导致 agent cfg 没到位（**探针报 `ModuleNotFoundError` 才发现**）。已修正并清理误传目录。
> 教训：**改配置后必须用探针核验运行层，不能假设上传成功**。

### 11.4 本轮要看什么

| 指标 | v1@5000 | v2@7330 | v3 期望 |
|---|---|---|---|
| **`Policy/mean_noise_std`** | 0.7557 | **1.5800** | 从 **0.3** 起步，**不应单调上升** |
| 回合长度（步） | 604.7 | 560.0 | ↑ |
| 触地终止比例 | 0.5498 | 0.6772 | ↓ 到 <0.1 |
| `Step/foot_sep_mean` | 0.3518 | 0.4616 | ⚠️ 本轮**不指望**改善（没加惩罚）|

⭐ **首要判据是 σ**：如果它不再单调上升（甚至缓慢下降），说明「抽走密集梯度 → PPO σ 膨胀」
这个机制判断成立；如果 σ 仍涨，就要考虑在 `WasabiPPOAmpMix` 里给 σ 硬性封顶
（`ActorCritic.clip_std` 支持 `max`，只是 `PPO.update` 没传）。

### 11.5 首轮 v3 被拦下 → 改成 v3b：数据集 + 指令分布对齐 AMP_mjlab（2026-09-25）

#### （a）用户的两点要求

1. **参考数据集换成 AMP_mjlab 的精选清单**（依据他们的实测经验：**运动数据过多会导致
   风格混乱** —— 判别器的风格分布在过大的参考集上被摊平，反而拉不住步态）。
2. **指令分布参照 `AMP_mjlab/src/tasks/amp_loco/config/g1/env_cfgs.py` 修改**
   （分档为原地 / 纵向 / 横向 / 斜向 / 后退）。

#### （b）数据集 17 → 7 clip

来源：`AMP_mjlab/motion_data_csv/amp_selected.txt` 里**未注释**的 7 行
（文件头写着「v4：6 行走 + 1 慢跑 = 7 个 clip」）—— **用户说的「约 6 个」就是指这个**。
选择清单：`motions/amp_walk/amp_walk_v3_motion_selection.yaml`。

| amp_selected.txt 里的 clip | 我们用的 | 说明 |
|---|---|---|
| `walk_forward_loop_002__A022` | 同 | 前进 ~0.6-1.0 |
| `walk_backward_loop_003__A143` | ⚠️ **`walk_backward_loop_001__A022`** | **代替品**，见下 |
| `walk_sideway_left_loop_002__A021` | 同 | 左侧移 |
| `walk_sideway_right_loop_001__A022` | 同 | 右侧移 |
| `jog_forward_loop_003__A022` | 同 | 慢跑前进（巡航 1.7-1.9）★ |
| `arc_walk_left_loop_001__A029` | 同 | 左弧 |
| `walk_arc_cw_loop_002__A046` | 同 | 右弧 |

⚠️ **为什么用代替品**：`walk_backward_loop_003__A143` 在 AMP_mjlab 的
`src/assets/motions/g1/amp/WalkandRun/` 里**只有 CSV、没有 npz**（那目录正好 17 个 npz
= 我们原来的 17 clip），补齐它要跑 AMP_mjlab 的 `scripts/csv_to_npz.py`
（**仿真式重定向**，属另一个工程；`deploy_walk_v2.py` 还是云端 SSH 脚本）。
而代替品用 `analyze_amp_motions.py` 实测 **vx = −0.62 m/s**，正好落在 AMP_mjlab 给
后退档设的 `back_lin_vel_range = (−0.7, −0.4)` **区间内** ⇒ 对该命令带**功能等价**。

**不选「原地」那一组**（`idle_turn_270/360`、`step_rotate_idle`）：与 amp_selected.txt 一致 ——
静止/原地转的风格由 `prob_low`（20%，其中 60% 给精确 (0,0,0)）+ `track_default_pose_on_stand`
负责，**不靠参考动作**。

#### （c）指令分布换成 AMP_mjlab v4（逐字段对照）

| 字段 | v1 | **v3b**（= AMP_mjlab v4）|
|---|---|---|
| `prob_low`（含 60% 精确 (0,0,0)）| 0.15 | **0.20** |
| `prob_side`（纯侧向 $v_x\equiv0$）| 0.25 | **0.20** |
| `prob_lin`（纯前后 $v_y\equiv0$）| 0.25 | **0.40** |
| `prob_lin_fwd`（其中持续直线前向）| 0.6 | **0.7** |
| `prob_back`（专属持续后退）| 0.10 | 0.10 |
| （余下）均匀**斜向** | 0.25 | **0.10** |
| `straight_lin_vel_range` | (0.5, 1.2) | **(0.4, 2.0)** |
| `back_lin_vel_range` | (−1.0, −0.5) | **(−0.7, −0.4)** |
| `lin_vel_x` | (−1.0, 1.2) | **(−0.7, 2.0)** |
| `lin_vel_y` | ±0.5 | **±1.0** |
| `ang_vel_z` | ±0.8 | **±1.57** |
| `resampling_time_range` | (10, 10) | **(6, 15)** |

⇒ 分档概率和 = 0.20+0.20+0.40+0.10 = **0.90**，余 **0.10** 为均匀斜向，
与 AMP_mjlab 注释「low 0.20 + side 0.20 + lin 0.40(fwd 70%) + back 0.10 + diag 0.10」**逐位一致**。

⚠️ **两项刻意不照抄**（照抄会报错或失效，不是保守）：
`rel_standing_envs` 仍 **0.0**（AMP_mjlab 是 0.05 —— 我们的实现有 **fail-fast 断言**，
站立语义已交给 `prob_low`）；`heading_command` 仍 **False**（AMP_mjlab 是 True，
开了之后 `wz` 会被**航向环重算**，方向分档直接失效）。

⚠️ **已知覆盖缺口**：精选 7 clip 实测 |wz| 上限 ≈ 1.13，而指令范围 ±1.57（照抄 AMP_mjlab）。
vx/vy 刚好覆盖，**wz 超出风格集**。若大角速度命令下步态劣化，先怀疑这里。

#### （d）⭐ 首轮 v3 被契约校验器拦下（探针又救了一次）

改完指令分布后探针直接失败：

```
RuntimeError: Unified OmniLoco teacher contract failed:
  - command ranges ((-0.7, 2.0), (-1.0, 1.0), (-1.57, 1.57)) != ((-1.0, 1.2), (-0.5, 0.5), (-0.8, 0.8))
  - command hold is (6.0, 15.0), expected 10 seconds
```

根因两种，性质不同：

1. `expected_command_ranges` **是类属性** ✓ → 只需给 v3 一个子类覆盖它
   （新建 `G1AmpWalkV3TeacherEnv(G1AmpWalkTeacherEnv)`，与 v1 的做法一致）；
2. ⚠️ 「指令保持时长」原先**写死在方法体里**（`omni_loco_env_cfg.py:256` 硬编码 `(10.0, 10.0)`）
   → 已改成**类属性** `expected_command_hold`，**默认值与原来逐位相同**，
   OmniLoco / v1 / v2 行为**完全不变**，各任务现在可以自行声明期望值。

> 教训（与 §11.3 的上传坑同源）：**改配置后必须跑探针核验运行层**。
> 这次两个问题（上传路径拼错、契约校验拦下）都是探针发现的，都不是靠看代码想出来的。

#### （e）v3b 启动核验：全绿

```
[AMP-WALK PASS] policy=626, action=29, scale=1.0,
                cmd_ranges=((-0.7, 2.0), (-1.0, 1.0), (-1.57, 1.57)) (directional), rewards=12
[运行层] reward_manager.active_terms 总数 = 12
[REWARD-SET PASS] Instinct-Amp-Walk-V3-G1-v0: active_terms=12 (expect=12)
[agent] class = G1AmpWalkV3PPORunnerCfg   init_noise_std = 0.3   clip_min_std = 0.015
[motion] selected_files = 7 个
契约/异常计数 = 0
```

启动：`--task Instinct-Amp-Walk-V3-G1-v0 --num_envs 4096 --max_iterations 5000`
（**从零**，`--run_name amp_walk_v3b_4096`，日志 `/root/train_amp_walk_v3b_4096.log`）。

---

### 11.6 v3c：恢复 `flat_orientation_l2` + `joint_deviation_torso`（2026-09-25）

#### （a）动机（用户实测反馈）

v3b 训练后模型呈现**姿态歪斜** → 难以平衡、速度无法跟踪（**略改变姿态就会摔倒**）。
用户判断应恢复 `flat_orientation_l2` 鼓励躯干水平。

#### （b）先量化了「姿态项该给多大权重」

`flat_orientation_l2` 的 IsaacLab 实现（云端 `envs/mdp/rewards.py:91-98`）：

```python
def flat_orientation_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """Penalize non-flat base orientation using L2 squared kernel."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
```

即 **`Σ projected_gravity_b[:, :2]² = sin²θ`**（θ = **root/pelvis** 相对重力方向的倾斜角）。
v1 实测该项贡献只有 **−0.0160/s**（w = −1.0）⇒ 反推平均倾斜 ≈ **7.2°** ⇒ **−1.0 几乎不构成约束**：

| 倾斜角 θ | sin²θ | w = −1.0（基类原值）| w = −5.0 | w = −10.0 |
|---|---|---|---|---|
| 5° | 0.0076 | −0.008 | −0.038 | −0.076 |
| 10° | 0.0302 | −0.030 | −0.151 | −0.302 |
| 15° | 0.0670 | −0.067 | −0.335 | −0.670 |
| **20°** | 0.1170 | **−0.117** | **−0.585** | −1.170 |
| 30° | 0.2500 | −0.250 | −1.250 | −2.500 |

对照任务正向项合计约 **+1.6/s**：w = −1.0 时歪 20° 才罚 0.117（7%），w = −5.0 时 0.585（36%）。

**用户选定 −1.0（基类原值，最保守）**。若本轮姿态歪斜未改善，第一优先是把
`AMP_WALK_V3_FLAT_ORI_WEIGHT` 改成 **−5.0**（改一个常量即可）。

#### （c）⚠️ 关键区分：它读 root，不读 torso_link

`flat_orientation_l2` 用的是 **root/pelvis** 的 projected gravity。而 `waist_pitch` /
`waist_roll` 的行程只有 **±0.520 rad**（除踝之外最挤的两个）—— **腰部歪斜不会体现在
骨盆倾角上**。因此同时恢复 `joint_deviation_torso`（−0.1，覆盖 `waist_.*`）作为兜底。

AMP_mjlab 有一个专门读 `torso_link` 四元数的 `torso_posture`
（`cos_tilt = 1 − 2(qx² + qy²)`），但它**默认是注释掉的**，本轮未采用（需新写奖励函数）。

#### （d）改了什么（v3b 的 12 项 → **14 项**）

| # | 改动 | 说明 |
|---|---|---|
| 1 | `action_rate_l2`：−0.01 → **−0.02** | 不变（v3 已有）|
| 2 | **恢复** `flat_orientation_l2` = **−1.0** | v2 删掉的 9 项之一，`func` 用基类 |
| 3 | **恢复** `joint_deviation_torso` = **−0.1** | 同上，`func`/`weight`/`params` 逐字沿用基类 |
| 4 | 其余 11 项 | 原样（含 v2 的 `dof_pos_limits` −10.0 / `dof_acc_l2` −2.5e-7）|

⚠️ 这两项**必须写在 `super().__post_init__()` 之后** —— v2 的 `__post_init__` 会把它们
置成 `None`（仓库既定的「删除」机制），写前面等于白写。

`run_name` 由 `rew12` 改成 **`rew14`**。

#### （e）v3c 启动核验：全绿

```
[AMP-WALK PASS] policy=626 (height_scan=527 + base_lin_vel), action=29, scale=1.0,
                cmd_ranges=((-0.7, 2.0), (-1.0, 1.0), (-1.57, 1.57)) (directional), rewards=14
Learning iteration 0/5000                      ← 确实从零
日志 L273: |  4 | flat_orientation_l2   | -1.0 |
日志 L275: |  6 | joint_deviation_torso | -0.1 |
逐项记录已出现：Episode_Reward/flat_orientation_l2: -0.0001
                Episode_Reward/joint_deviation_torso: -0.0001
params/agent.yaml: init_noise_std: 0.3   clip_min_std: 0.015
run 目录: 20260925_203531_G1AmpWalkV3Plane_obs626_rew14_ar0.02_ns0.3_amp_walk_v3c_4096
```

* 启动方式（新增 `instinct/tools/cloud_train_amp_walk_v3.sh`，避免嵌套引号踩坑）：
  ```bash
  bash /root/cloud_run_bg.sh \
    'bash /root/cloud_train_amp_walk_v3.sh 4096 5000 cuda:0 amp_walk_v3c_4096' \
    /root/train_amp_walk_v3c_4096.log
  ```
* 一轮 5.19 s ⇒ ETA ≈ 7.2 h；GPU 10884 MiB。

> ⚠️ 探针验证被跳过：`cloud_probe_rewards.py` 两次卡在 Isaac Sim 启动上（本地没
> `isaaclab`，只能云端跑）。**训练启动日志打印的是同一套 `[AMP-WALK PASS]` / 契约校验**，
> 且额外给出逐项权重表 ⇒ 用训练日志做验证，覆盖面等价且更快。

#### （f）本轮要看的指标

| 指标 | 期望 | 说明 |
|---|---|---|
| `Episode_Reward/flat_orientation_l2` | **绝对值要小** | 若长期在 −0.02 以上 ⇒ 平均倾斜 >8°，说明 −1.0 果然不够，改 −5.0 |
| `Episode_Reward/joint_deviation_torso` | 同样要小 | 反映腰部偏离 default 的程度 |
| `Policy/mean_noise_std` | 不得从 0.3 单调上升 | 上轮 v2 从这里涨到 1.58 ⇒ 退步 |
| 回合长度 / `base_contact` 终止比 | 改善 | v2 时 base_contact 0.5498 → 0.6772 |
| 高速（vx > 1.2）与后向（vx ∈ −0.7..−0.4） | 学会交替步态 | |
| `Step/foot_sep_mean` | 仍会偏差 | 站姿宽度惩罚**本次仍未加**（刻意保持单变量）|

#### （g）已知的下一步候选（本轮未做）

1. `AMP_WALK_V3_FLAT_ORI_WEIGHT` −1.0 → **−5.0**。
2. **`dof_pos_limits` 作用域从「只有踝」扩到全身** —— 现在 hip/knee/waist/arms **完全不罚**。
   按「到最近限位的余量」排序，最挤的 8 个关节里 **6 个**只由已被删除的
   `joint_deviation_*` 覆盖（踝 0.262 / 腰 0.520 / hip_roll 0.524 / 膝 0.756）。
   ⚠️ 但 `joint_pos_limits` 的语义是**越界惩罚**（`(q − limit).clip()`，软限位内恒为 0），
   且 `soft_joint_pos_limit_factor` 默认 **1.0** ⇒ 软限位 = 硬限位。它是「撞上限才罚」，
   **不是「靠近限位就罚」**；真要抑制贴限位得靠 `joint_deviation_*` 或自写 proximity 项。
3. 站姿宽度惩罚（§10.4 的结构性缺口）。

---

### 11.7 v3c 跑完 5000 轮 → **无上限续训**（2026-09-26）

#### （a）5000 轮结果：**没有收敛，仍在上升期**

| iter | 平均回报 | 回合长度 | 动作噪声 σ |
|---|---|---|---|
| 0 | −3.77 | 6.07 | 0.30 |
| **170** | **+16.18** ← 早期峰值 | 549.63 | 0.44 |
| 850 | −0.40 | 401.36 | 0.76 |
| 1870 | −6.90 | 472.95 | 1.00 |
| **2550** | **−10.28** ← 谷底 | 509.63 | 1.08 |
| 3400 | +0.93 | 427.38 | 1.15 |
| 4250 | +3.09 | 480.74 | 1.19 |
| **4990** | **+5.17** | 538.58 | 1.20 |

从 2550 轮的 −10.28 **单调爬回** +5.17；触地终止比 0.6907 → **0.6475**。
⇒ 5000 轮上限是在**恢复上升途中**把训练截断的。

#### （b）⚠️ 对 §11.6(f) 判断的更正（重要）

我在 2210 轮（≈谷底）下过"和 v2 是同一条轨迹"的结论，**下早了**。之后的 2780 轮里
回报单调回升、触地比下降。教训：**在谷底附近取样会得到相反的结论**；判断退步至少要看
2~3 个存点的斜率，或等曲线自证。

#### （c）续训机制（已核实）

| 事实 | 证据 |
|---|---|
| ⭐ `--max_iterations` 是**相对**语义 | `OnPolicyRunner.learn()`：`tot_iter = current_learning_iteration + num_learning_iterations`，`while current < tot_iter` ⇒ 传入值 = **还要再训多少轮**，不是绝对终点 |
| 实测确认 | v2 那次 `--resume`(from 5000) + `max_iterations=5000` 显示的就是 `7410/10000`；v3c_cont 传 10⁹ 显示 `5000/1000005000` |
| `runner.load()` 恢复 `current_learning_iteration` | `train.py:173-211` |
| LR **不**随 `max_iterations` 变化 | `amp_walk_ppo_cfg.py` 里**没有 `schedule` 字段** ⇒ 固定学习率 |
| ⚠️ 续训会重置 AMP 参考动作的播放游标 | 它不在 checkpoint 里 ⇒ 所有环境从参考 clip 第 0 帧重开（小不连续）|

`instinct/tools/cloud_train_amp_walk_v3.sh` 新增 `$5 load_run` / `$6 checkpoint`（非空即自动
`--resume`），并把 `$2`（`max_iterations`）**默认值改成 1000000000 = 实质无上限** ——
按用户要求：训练一直跑到人工叫停，由人盯曲线决定何时停。
`RESUME_FLAGS` 只在脚本**内部**拼装，外层命令行保持"全是单 token"，避免嵌套引号被拆坏
（本项目已多次踩这个坑）。

#### （d）v3c_cont 要看的指标

| 指标 | @4990 | 期望 |
|---|---|---|
| 平均回报 | +5.17 | 继续上升 → 何时平台 |
| 触地终止比 | 0.6475 | 降到 <0.3 |
| 动作噪声 σ | 1.20（仍在涨） | 平台化（digest 期望是"缓慢下降"）|
| 脚距均值 | 0.3445 | 进 0.15~0.30 |
| 判别器损失 | 0.0138 | 不发散 |

#### （e）⭐ AMP 风格权重偏小（本轮最重要的结构性发现）

奖励合成（`utils/wrappers/instinct_rl/algorithms.py:171-195`）：

$$R = \lambda\,r_{\text{task}} + (1-\lambda)\,c\,r_{\text{amp}},\qquad \lambda=0.75,\; c=0.1$$

* ⇒ **风格权重 = (1−λ)·c = 0.025**，只有任务权重 0.75 的 **1/30**
* `r_amp = clamp(1 − 0.25(d−1)², min=0) ∈ [0, 1]`（AMP_mjlab 的 quad 形式）
* ⇒ **风格贡献每步 ∈ [0, 0.025]**；实测 `r_amp ≈ 0.087` ⇒ 每步 **0.0020**，
  对照任务正项合计 ~1.44 ⇒ 风格只占 **~0.14%**

这解释了为什么 `tracking_joint_error_rmse/left_shoulder_roll = 1.36`（29 关节里最大、
是第二名 waist_pitch 0.504 的 2.7 倍）**却没有被纠正** —— 风格项没有足够的力。

> ⚠️ `06_amp_task_split.png` 里的 "AMP contrib" 是**残差** `v_fin − λ·r_task`，不是测量值：
> 由 `amp_contrib + task_contrib ≡ v_fin` 可知它的零交叉是**代数恒等式**，
> 不能据此判断"AMP 退步"。真实的风格奖励已被写进 TB（`aux["amp_style_reward"] = r_amp`），
> 画图应改用它。

#### （f）旁证：肩 roll 的动作确实离群

| 关节 | `tracking_joint_error_rmse` |
|---|---|
| **left_shoulder_roll** | **1.3582** |
| **right_shoulder_roll** | **1.2896** |
| left_elbow | 0.9957 |
| right_elbow | 0.8285 |
| waist_pitch | 0.5044 |
| left_knee | 0.2162 |

另：`flat_orientation_l2` 每步 **−0.0179** ⇒ 反推 `sin²θ=0.0179` ⇒ 平均倾斜 **≈7.7°**，
v1 实测 7.2° ⇒ **−1.0 这个权重确实几乎没改变姿态**（与 §11.6(b) 的预测一致）。
且 `track_default_pose_on_stand` 是**门控**的（`exp(−cmd_speed/0.25)`），
走路时对 70% 以上环境只剩 <20% 强度、对 vx>1 m/s 只剩 2%（`cmd_speed` 含 `|wz|`，
所以原地转身也会把它关掉）。

---

### 11.8 ⭐ 诊断：为什么 `track_default_pose_on_stand` = −0.1 与"静止时单脚跳"**不矛盾**

**背景**：MuJoCo 回放显示静止时仍维持**单脚跳跃**、左右完全不对称；用户据此怀疑
−0.1 与现象矛盾。结论：**−0.1 小是因为门控关着，不是姿态好。**

#### （a）`joint_error_rmse/<joint>` 就是「相对 default pose 的偏差」，且**不门控**

`mdp/loco_metrics.py:24,52`：

```python
joint_error = robot.data.joint_pos - robot.data.default_joint_pos      # 逐关节偏差
log[f"joint_error_rmse/{joint_name}"] = torch.sqrt(torch.mean(joint_error[:, index].square()))
```

⇒ 这才是判断姿态的正确指标（**已经在记录，只是没进 digest 汇总**）。

#### （b）v3c_cont @5390 实测的数值构成

| 量 | 值 |
|---|---|
| 平均关节偏差 RMS | **0.6636 rad = 38.0°**（@2180 是 0.512 rad = 29° ⇒ **在变差**）|
| 位置项上界 `4·mean\|Δq\|` | **2.654 /步**（gate=1 时）|
| TB `Episode_Reward/track_default_pose_on_stand` | **−0.0978 /秒** |
| ⇒ **等效门控质量** | **0.0368** ← 该项 96% 的样本上不生效 |
| `command_bins/vx_low_fraction` | 0.0835 |

#### （c）偏差最大的是**手臂**，不是腿

| 关节 | RMS(rad) | 度 |
|---|---|---|
| **right_shoulder_roll** | **2.0338** | **116.5** |
| left_shoulder_roll | 1.5495 | 88.8 |
| left_elbow / right_elbow | 1.4883 / 1.4860 | 85.3 / 85.1 |
| right_wrist_roll | 1.3359 | 76.5 |
| right_shoulder_yaw / pitch | 1.0356 / 1.0029 | 59.3 / 57.5 |
| （最小）left_ankle_roll / waist_yaw | 0.19 ~ 0.20 | 11 |

腿（相对好）：hip_pitch 0.615/0.294、knee 0.302/0.400、ankle_pitch 0.228/0.354（13~35°）

#### （d）左右不对称：集中在**臂**，且**右臂更大**

| 分组 | Σ(R−L) RMS | 每关节均值 |
|---|---|---|
| 臂 7 组 | **+1.739** | +0.248 |
| 腿 6 组 | +0.159 | +0.026 |

最大项：`shoulder_pitch` **R/L = 2.41**、`shoulder_roll` +0.484、`wrist_roll` +0.399、
**`hip_pitch` −0.322（左>右）**。
⇒ 左 hip_pitch 偏大 + 右 knee/ankle_pitch 偏大 = **一条腿屈/抬、另一条伸**的单脚支撑特征。

#### （e）根因（四条，按重要性）

1. **门控**：`active = exp(−cmd_speed/0.25)`，走路时 →0；实测等效门控质量仅 **0.037**。
   ⇒ `Episode_Reward/track_default_pose_on_stand` 是一个**被关掉的量的平均值**，与姿态好坏几乎无关。
2. **它是 29 关节的 mean，对"哪个关节"盲**：单关节偏 2.03 rad 只贡献 `2.03/29 = 0.070`
   → 罚 `4×0.070 = 0.28`。**且左右互换后均值完全相同 ⇒ 对对称性完全盲目。**
3. **整个任务奖励对镜像变换不变** ⇒ **对称步态与单脚跳在任务奖励上简并**，
   梯度里没有任何破简并的信号。
4. 唯一的破对称信号是 **AMP 判别器**，但它是**全局标量**
   （`clamp(1 − 0.25(d−1)², 0)`，`d` 给**整条 transition** 打分）⇒ 无法定位到关节。
   另外 `pos_joint_weights` 只给 `waist_yaw` ×3，腿/臂都没加权。

#### （f）下一步候选

1. **恢复 `joint_deviation_arms`（−0.1）** —— 无门控、覆盖 7 个手臂关节（L1 sum），
   正对偏差最大的部位 ← **最直接、代价最小**
2. 给 `track_default_pose_on_stand` 的腿/臂加 `pos_joint_weights`（仍受门控 ⇒ 走路时无效）
3. 自写**不门控**的 per-joint 偏差项（唯一"走/站都生效"的做法）
4. **对称性**：⚠️ **不要**直接罚 `|q_L − mirror(q_R)|` —— 正常步态本来就左右交替；
   更稳的是提高 `discriminator_reward_coef` 或换一个能定位到关节的风格项
5. 把「臂/腿/腰分组平均偏差 + 左右不对称度」加进 `digest.txt`，让姿态有可见指标

---

### 11.9 V4：AMP 侧两项修正（镜像损失 + 站立关风格）（2026-09-26）

#### （a）为什么开 V4

v3c_cont 训练跑偏（用户判断 + 实测：`joint_error_rmse` 平均偏差 **38°**、
臂 7 组 `Σ(R−L) = +1.739`、触地终止比 0.72、中位回合长度仅 **68.5 步**）。

#### （b）核对结论（由 AMP_mjlab 工作区 agent 提供，见 `AMP_MJLIB_CHECK*.md`）

| 结论 | 证据 |
|---|---|
| **手臂不一致是数据本身固有** | 17/17 npz 与对方原版**逐帧零差异**（`mean\|Δq\|=0.000000`、`corr=1.0`）|
| **不是选集差异** | 两边 7-clip 精选集几乎相同 |
| ⭐ **真正差异只有两条** | 他们 `use_mirror_loss=True`（**从第 0 轮开、无分阶段**）+ `amp_disable_style_on_stand=True`；**我们两条都没有** |
| 他们的 AMP obs 是 **195 维纯 link 相对位姿** | 不含任何 `joint_pos`/`joint_vel` ⇒ 关节角不一致对其影响较小 |
| 他们的 RSI 是**随机帧** | 不是第 0 帧 ⇒ "每 episode 从迈步相位起步"的担心不成立 |

#### （c）V4 改了什么（环境 / 观测 / 动作 / 奖励 **全部不动**）

| 项 | v3c | **V4** |
|---|---|---|
| `use_mirror_loss` | **无** | **`True`**（`coeff = 0.1`，= AMP_mjlab `mirror_loss_coeff`）|
| `disable_style_on_stand` | `False` | **`True`**（`stand_speed_threshold = 0.1`）|
| 奖励集 / 数据集 / 指令分布 / σ=0.3 / 契约 | — | **完全不变** |

#### （d）实现（刻意做轻）

* `instinctlab/utils/mirror.py` —— 镜像算子。**`height_scan` 的排布从 IsaacLab 源码确认**
  （`GridPatternCfg(size=(3.0,1.6), res=0.1, ordering="xy")` ⇒ 527 = **31(前向)×17(侧向)**，
  扁平化 `idx = iy*31 + ix` ⇒ 镜像 = `reshape(17,31)[::-1]`）。
* **接线方式**：覆写 `compute_losses()` 并放一个 `losses["mirror"]`，系数落在 `self.mirror_coef`
  —— 利用 `PPO.update` 的 `getattr(self, k + "_coef", 1.0)` 机制，
  **不复制那 90 行 `update()`**（避免上游升级漏改）。`Loss/mirror` 自动进 TB。
* 新增 `agents/amp_walk_v4_ppo_cfg.py`、`amp_walk_v4_env_cfg.py`（后者**只换 run_name**）、
  注册 `Instinct-Amp-Walk-V4-G1-v0` / `-Play-v0`。
* 启动脚本 `cloud_train_amp_walk_v3.sh` 新增 `$7` 任务参数（默认仍是 v3）。

#### （e）启动核验（全绿）

```
Learning iteration 0/1000000000     rewards=14     Mean action noise std: 0.30     GPU 11352 MiB
打表损失项: surrogate_loss 0.3073 / value_loss 3.7559 / entropy -6.4231 /
            ⭐ mirror 1.1493 / total_loss 4.1268 / discriminator_loss 0.5948 / amp_total_loss 1.1407
真异常: (无)
```

⇒ **`mirror` 已确认进入 loss 且非 0** ✓；`disable_style_on_stand=True` 未报错 ⇒ 成功定位到指令切片 ✓

#### （f）踩到的三个坑（都值得记）

1. **单测的 import 方式**：`from instinctlab.utils.mirror import ...` 会连带执行
   `instinctlab/__init__.py`，把 **Omniverse/Kit 拉进来** ⇒ 非交互环境**卡在
   `Do you accept the EULA? (Yes/No)`**（现象：输出全空、进程一直挂着，极难排查）。
   改成 `importlib` **按文件路径**加载；且加载后**必须先注册进 `sys.modules`**，
   否则 `@dataclass` 报 `'NoneType' object has no attribute '__dict__'`。
2. ⭐ **CPU/CUDA 混用**：`perm`/`sign` 构造时是 CPU 张量，训练时 `q` 在 CUDA
   ⇒ v4 **第一次启动就崩在 `mirror_joints`**。
   **教训：单测必须在训练设备上跑**（已补第 [7] 节 CUDA 回归，云端 25/25）。
   * 单测第一次还抓到另一个真 bug：`knee`/`elbow` **也是 pitch 类**（名字里没有 "pitch"），
     naive 的 `"pitch" in name` 会误判成反号 ⇒ 被"默认姿态是不动点"这条判据抓到（`max|Δ|=2×0.669`）。
3. **等待循环的 error pattern**：`grep 'Error'` 会被 PhysX 的无害告警
   `[Error] ... Patch buffer overflow` **误触发**，让循环第一次检查就 break（会误判成"没进展"）。
   用 `Traceback|RuntimeError|ImportError|TypeError|ValueError`。

#### （g）V4 要看的指标

| 指标 | 起点 | 期望 |
|---|---|---|
| **`Loss/mirror`** | 1.149 @iter0 | 下降（策略逐渐左右等变）；**恒 0 说明没接上** |
| 左右不对称度 | 臂 Σ(R−L) **+1.739**、腿 +0.159 | 显著下降 |
| 触地终止比 / 中位回合长度 | 0.72 / **68.5 步** | 改善 |
| 姿态平均偏差（`joint_error_rmse` 均值）| 38° | 下降 |
| `Episode_Reward/track_default_pose_on_stand` | — | ⚠️ 开门控后**会更小**，**不能**用它判断姿态 |

---

### 11.10 V4 第 2500 轮核验：`mirror` 系数**数值证明**已生效 + 同轮次对比

#### （a）先怀疑，再查代码，再数值反证

疑点：iter0 的 `total_loss = 4.1268`，而 `surrogate 0.3073 + value 3.7559 = 4.0632`
已经几乎占满总量 —— 担心 `mirror` **只是被打印、没进总损失**。

查代码（三处）：

- `instinct_rl/algorithms/ppo.py:200`：`loss += getattr(self, k + "_coef", 1.0) * v`
- `ppo.py:287`：返回的 `entropy` 是 **`-entropy_batch.mean()`**（负号在算法里）
- 配置 `instinct_rl_ppo_cfg.py`：`value_loss_coef = 1.0`，**`entropy_coef = 0.008`**

数值反证（精确吻合到 4 位）：

```
1.0×0.3073 + 1.0×3.7559 + 0.008×(−6.4231) + 0.1×1.1493 = 4.1268
日志里实际的 total_loss                                  = 4.1268   ✓
（若 mirror 没进总损失：0.3073+3.7559−0.0514 = 4.0118，差 0.115，对不上）
```

⇒ **`mirror` 确实以系数 `mirror_coef = 0.1` 进入总损失**，不是「只进日志」。
同时反过来证明：`mirror_loss()` 返回的是**不含系数**的原始值 `mean(sum(·))`，
系数由 `update()` 按 `"mirror" + "_coef"` 约定补上 —— 与命名设计一致。

#### （b）⭐ 定量解读：打印值换算成「每维 RMS」

打印值 $= \frac{1}{B}\sum_b \sum_{j=1}^{626}\big(\mu_\pi(M_{obs}o_b)_j - M_{act}(\mu_\pi(o_b))_j\big)^2$

$$\text{每维 RMS} = \sqrt{\text{打印值} / 626}$$

| iter | 打印值 | 每维 RMS |
|---|---|---|
| 0 | 1.1493 | 0.0428 rad ≈ **2.45°** |
| 500 | **0.0337**（最低）| 0.0073 rad ≈ **0.42°** |
| 1000 | 0.0435 | 0.0083 rad ≈ 0.48° |
| 2000 | 0.0625 | 0.0100 rad ≈ 0.57° |
| 2500 | 0.0658 | 0.0103 rad ≈ **0.59°** |

含义（两条都要记）：

1. **策略在「镜像等变」意义下已经几乎是左右对称的**（μ 的差只有 0.6°）。
   所以**不能指望镜像损失单独去修「单脚跳」** —— 若不对称模态之后又出现，
   那它来自**闭环对称性破缺 / 观测历史 / 奖励算子的不对称**，而不是「策略本身左右有别」。
2. 注意这个数有个天然偏小的来源：**当状态本身左右对称时**（$o \approx M(o)$）
   打印值也会自动变小。所以它更像「策略在左右对称状态上输出得对不对称」，
   而不是纯粹的策略属性。**拐点要看它随步态形成是涨还是压住。**

#### （c）v3c / v4 **同轮次**对比（新工具 `instinct/tools/cloud_compare_runs.sh`）

| iter | v3c 回报 | v4 回报 | v3c 回合长 | v4 回合长 | v3c 触地 | v4 触地 | v4 mirror |
|---|---|---|---|---|---|---|---|
| 0 | −3.77 | −3.75 | 6.07 | 6.09 | 0.8072 | 0.8078 | 1.1493 |
| 500 | 6.82 | **7.86** | 459.8 | **480.1** | 0.6104 | 0.6614 | 0.0337 |
| 1000 | **0.45** | −1.97 | **435.7** | 387.0 | **0.7057** | 0.7395 | 0.0435 |
| 1500 | **−1.60** | −7.46 | **457.2** | 411.3 | **0.6972** | 0.7416 | 0.0511 |
| 2000 | **−7.96** | −9.52 | 428.5 | **503.0** | 0.6833 | 0.7046 | 0.0625 |
| 2500 | −9.99 | **−7.69** | 496.5 | 464.5 | 0.6868 | 0.7005 | 0.0658 |
| 3000 | 0.57 | — | 481.2 | — | 0.7591 | — | — |
| 3500 | 1.14 | — | 429.5 | — | 0.7338 | — | — |
| 4000 | 4.41 | — | 535.9 | — | 0.7039 | — | — |
| 4500 | 4.83 | — | 516.7 | — | 0.6644 | — | — |
| 5000 | **5.17** | — | 538.6 | — | 0.6475 | — | — |

**读法（重要）**：

- **iter0 两条几乎逐位相同**（−3.77/−3.75，amp_style 0.7631/0.7630）
  ⇒ 初始动力学没被两项改动影响，改动是「干净的」✓
- v4 现在（2500 轮）正落在 **v3c 一模一样的「2000–2900 低谷」**里
  （v3c 在 2000/2500 是 −7.96/−9.99，v4 是 −9.52/−7.69）
  ⇒ **同量级、没有发散**。这个低谷是任务设定的固有现象，不是 V4 引入的退化。
- v3c 是**从 3000 轮才爬升**、到 4500–5000 才稳定到 +4.8 → +5.2。
  ⇒ **V4 现在（2500）远未到可判断的轮次，必须至少跑到 5000 轮再比。**
- v4 触地终止率略高（0.70 vs 0.687）、回合长略短，属于同轮次的正常抖动。

#### （d）日志 tag 真相（`plot_training.py` 之外手查时必须知道）

| 想查什么 | 正确写法 | 坑 |
|---|---|---|
| 迭代号 | 行内有 `Learning iteration <N>/<max>` | **前缀字段数不固定**，`$3` 取到的是字面量 `iteration`；必须正则抽 |
| 回报 | **`Mean reward:`** | 不是 `Train/mean_reward`（那是 TB 名） |
| 回合长度 | **`Mean episode length:`** | 同上 |
| 风格奖励 | `Mean step amp_style_reward:` | `disable_style_on_stand` 会把站立样本置 0 ⇒ 均值天然偏低，**不能**直接与 v3c 比 |
| 触地终止 | `Episode_Termination/base_contact:` | ⚠️ 与 `Episode_Reward/*` 一样是**每秒速率**，**不是比例** |

⇒ 已把解析逻辑固化成 `instinct/tools/cloud_compare_runs.sh`（`SAMPLE=250` 可调采样间隔），
避免每次用「本地 → python → ssh → 远端」四层引号传 `$NF`/`$2` 被吃掉
（症状：只回一行垃圾 `0 |`）。
