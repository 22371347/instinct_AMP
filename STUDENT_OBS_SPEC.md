# 异构蒸馏：学生策略可观测规范（Student Observability Spec）

> 目的：给后续 **异构蒸馏（heterogeneous DAgger）** 的验证提供"学生到底能拿到哪些观测"的权威清单。
> 全部结论都对着源码钉过（`instinct_xu/source/instinct_rl`、`instinct_xu/source/instinctlab`），行号可复查。
> 关联：`INTERFACE_CONTRACT.md`、`PLAN_expert_distill.md`、`OBS_ALIGNMENT`（本文件即是其结论版）。

---

## 0. 一句话结论

学长用的是 **`HeterogeneousMultiExpertDagger`**（`instinct_rl/algorithms/heterogeneous_dagger.py`，
*"Pure action DAgger using one student and any number of routed teachers with independent observations"*）
⇒ **教师观测不需要对齐**（每个教师各带自己的 `obs_group` + `obs_format`）。

但**学生侧**有硬约束，且框架的**默认学生 = 623 维 loco 策略**：

| | 学生（默认） | 我们的 AMP 行走专家 |
|---|---|---|
| 维度 | **623** | **626** |
| 差异 | — | **多了 3 维 `base_lin_vel`** |
| 项名/顺序 | `height_scan, velocity_command, projected_gravity, base_ang_vel, joint_pos_rel, joint_vel, last_action` | 同上但在 `velocity_command` 后**插入** `base_lin_vel` |

⇒ 我们**不需要**改 `height_scan` 的尺寸（上一个问题的答案）；真正要决定的是
**是否把行走专家的 `base_lin_vel` 去掉，从而与 623 的学生契约完全同构**（见 §4）。

---

## 1. 框架硬约束（源码强制，不是建议）

`heterogeneous_dagger.py`：

| # | 约束 | 出处 |
|---|---|---|
| 1 | `student_obs_schema` **必须显式给全**：`if not student_obs_schema: raise ValueError("V0 requires an explicit, complete student_obs_schema.")` | `__init__` |
| 2 | 学生 schema 与 env 的 `observations.policy` **逐项「名字+维度+顺序」完全一致**，否则 raise：`actual != expected` | `_validate_student_obs_format` |
| 3 | 学生总维必须 = `student_obs_dim`（**默认 623**） | 同上 |
| 4 | env 必须额外提供 `teacher_route`（**宽度必须 1**）与 `teacher_valid_mask`（**宽度必须 1**） | `init_storage` |
| 5 | **每个教师各一个唯一 obs 组**（`spec.obs_group`），宽度必须 = 该教师 `expected_obs_dim` | `init_storage` |
| 6 | 每个教师 `expected_action_dim` 必须 = 学生 `num_actions`（我们 = 29） | `init_storage` |
| 7 | 学生 policy 项名**禁止**出现：`teacher_route` / `teacher_valid_mask` / 各教师组名 / `skill_id` / `reference` / `vault_phase` / `global_position` / `distance_to_box` | `forbidden_policy_terms` |
| 8 | 学生必须是**前馈**网络（`is_recurrent` → `NotImplementedError`） | `init_storage` |
| 9 | `action_adapter` **目前只支持 `"identity"`**（其它值 raise）⇒ 教师动作必须与学生同语义同维 | `TeacherSpec.from_config` |
| 10 | `policy_normalizer_mode ∈ {fresh_trainable, loco_frozen, loco_trainable}`；后两者**必须有** `student_init_checkpoint` | `__init__` |
| 11 | `student_init_checkpoint` 只加载 **actor**，且 `load_state_dict(strict=True)` ⇒ **输入维度不符会直接报错**（不会静默截断） | `__init__` |

⚠️⚠️ **框架拦不住特权量泄漏**：`forbidden_policy_terms` **不包含** `height_scan` / `base_lin_vel` /
`global_pelvis_pos` 等 ⇒ 若误把特权项塞进学生观测，框架会**静默通过**。必须在我们 env 侧自查（见 §5）。

---

## 2. 学生可观测 schema（默认版，623）

```python
# 请以 env 的 observations.policy **声明顺序**为准；名字必须逐字一致
STUDENT_OBS_SCHEMA_623 = [
    ("height_scan",        527),   # 31 x 17，3.0(前后) x 1.6(左右) m，res 0.1，torso 居中，yaw 对齐，clip(-1,1)
    ("velocity_command",     3),   # (vx, vy, wz) —— 我们：DirectionalVelocityCommand
    ("projected_gravity",    3),
    ("base_ang_vel",         3),
    ("joint_pos_rel",       29),
    ("joint_vel",           29),
    ("last_action",         29),
]   # sum == 623
```

逐项可得性 / 噪声（我们行走任务侧的实测契约）：

| 项 | 维 | 来源 | 噪声 | 真机可得 | 备注 |
|---|---|---|---|---|---|
| `height_scan` | 527 | `MultiMeshRayCaster` @ `torso_link`，(0,0,20) 垂直下打，`yaw` | 无 | ❌ **特权** | 真机需由深度/激光重建 |
| `velocity_command` | 3 | `DirectionalVelocityCommand`，范围 `((-1.0,1.2),(-0.5,0.5),(-0.8,0.8))` | 无 | ✅ | ⚠️ 第三维是 **wz（偏航率）** |
| `projected_gravity` | 3 | `instinct_mdp.projected_gravity` | ±0.05 | ✅ | 直立 = (0,0,-1) |
| `base_ang_vel` | 3 | `instinct_mdp.base_ang_vel`（body 系） | ±0.2 | ✅ | 无缩放 |
| `joint_pos_rel` | 29 | `mdp.joint_pos_rel` | ±0.01 | ✅ | 相对默认姿态 |
| `joint_vel` | 29 | `mdp.joint_vel` | ±1.5 | ✅ | 无缩放 |
| `last_action` | 29 | `instinct_mdp.last_action` | 无 | ✅ | policy 序，不重排 |

替代版（PHP/越障那条"可部署学生"，供对照，**不是本框架的默认**）：
`depth_image`(48×32=1536) + `velocity_command`(3，第三维被置 0) + `projected_gravity`(3)
+ `base_ang_vel`(3) + `joint_pos`(29) + `joint_vel`(29) + `last_action`(29) —— 单帧无历史。

---

## 3. 我们的行走专家 vs 学生契约（逐项映射）

| 行走专家 626 项 | 学生（623）里是否有 | 结论 |
|---|---|---|
| `height_scan` 527 | ✅ 有（同维同序，**都在最前**） | 直接可用 |
| `velocity_command` 3 | ✅ 有 | ⚠️ 语义待统一（见 §4） |
| `base_lin_vel` 3 | ❌ **没有** | **唯一的差异项**（特权，真机需估计） |
| `projected_gravity` / `base_ang_vel` / `joint_pos_rel` / `joint_vel` / `last_action` | ✅ 全有，名字也一致 | 直接可用 |

⇒ **我们的 policy 组离框架默认学生契约只有 1 项（3 维）之遥**，而且**去掉 `base_lin_vel` 后剩余顺序
恰好等于 623 的顺序**（cmd → grav → ang_vel → jp → jv → la）——即天然同构。

---

## 4. 待你决策的一件事（决定要不要重训）

| 方案 | 做法 | 结果 |
|---|---|---|
| **A. 保持 626** | 行走专家仅作**教师**（`expected_obs_dim = 626`，自己的 `obs_group`） | 异构框架支持 ✓，**现有 V1–V8 权重全部可复用**；但**不能**用作 `student_init_checkpoint`（差 3 维 ⇒ `strict=True` 直接报错） |
| **B. 改成 623** | 新任务/新版本去掉 `base_lin_vel` | 与 `LocoTeacherPolicyObsCfg`(623) **完全同构** ⇒ 既是教师、又能当**学生热启动**（配合 `loco_frozen`/`loco_trainable`）；代价是**要重训**，且与 AMP_mjlab 的 actor 少了那一维对齐 |

> 我的建议：**先按 A 走，不重训**（异构框架本来就为这种情况设计）；等学生的初始化/部署路线定了，
> 再决定是否为了"631→623 的学生热启动"重训一版。

---

## 5. 教师接入清单（`TeacherSpec` 逐字段）

| 字段 | 我们填什么 |
|---|---|
| `name` | `"amp_walk"`（唯一） |
| `route_id` | `0`（越障 = `1`，必须唯一） |
| `checkpoint` | 云上 run 目录里的**确切** `model_xxxx.pt`（不能给 glob/目录） |
| `obs_group` | env 侧新建的 obs 组名，建议 `"teacher_amp_walk"`（唯一、不得与学生项名撞车） |
| `expected_obs_dim` | **626**（方案 A） |
| `expected_action_dim` | **29** |
| `actor_cfg` | 留空即可 ⇒ 自动从 checkpoint 同目录的 `params/agent.yaml` 读 `obs_format`；若该组的项名/顺序与 yaml 不一致，必须显式给出 |
| `loss_weight` | `1.0`（route-balanced 权重） |
| `action_adapter` | `"identity"`（仅此一种受支持） |

教师 normalizer：`TeacherAdapter` 会从教师 run 目录加载 `params/agent.yaml` + normalizer 权重
（我们每个训练 run 都有 ✓）；加载失败会明确报错，不会静默用恒等映射（除非 checkpoint 里确实没有）。

---

## 6. 蒸馏验证清单（提高验证效果的关键，逐条可查）

1. **schema 对齐**：启动时框架会打印/校验 `obs_format`；`student_obs_schema` 的
   `(名字, 维度)` 列表必须与 env `observations.policy` **顺序完全一致**。不一致直接 raise —— 不要绕过。
2. **路由信号**：`teacher_route`(1) + `teacher_valid_mask`(1) 必须存在且宽度 = 1；
   训练日志里应能看到**每个 route 的样本数与损失贡献**（`loss_weight` + `RoutedActionLabelRollout`）。
   ⚠️ 行走技能的样本占比若长期偏低，学生就学不到行走 ⇒ 这是首要监控项。
3. **防泄漏自查**（框架不管）：确认学生 policy 里**没有** `height_scan` / `base_lin_vel` /
   `global_pelvis_*` / `reference` / `skill_id` 等特权或路由信息（若学生就是特权策略，需**显式记录**这是有意为之）。
4. **动作语义一致性**：`expected_action_dim = 29 = num_actions`（框架校验 ✓），
   `action_adapter = "identity"` ⇒ 两边动作必须同义。核实：我们都是
   `q_target = default_q + 1.0 * a`、`use_default_offset=True`、`clip=None`、**同一套默认姿态**
   （`G1_29DOF_TORSOBASE_POPSICLE_CFG`）✅ —— 唯一残留差异是**执行器刚度/阻尼**
   （vault 用 `beyondmimic_g1_29dof_actuators`），动作**语义**一致但**动力学**不同。
5. **时间尺度**：两边都是 `sim.dt = 0.005` × decimation 4 = **50 Hz**，单帧无历史 ⇒ 动作可直接逐帧比对 ✅。
6. **命令语义**（你已决定后续再改）：学生第三维与我们的 `wz` 不同义 —— 蒸馏时**不要**按位置硬对齐，
   要么统一表示，要么在验证时把该项排除。
7. **学生可塑性**：`policy_normalizer_mode` 选 `fresh_trainable` 时学生 normalizer 从零训；
   选 `loco_frozen` 则冻结 loco 的 normalizer（需 `student_init_checkpoint` + 维度匹配，见 §4 方案 B）。

---

## 7. 预留但当前仓库尚未定义的名字

`skill_id`、`reference`、`vault_phase`、`global_position`、`distance_to_box` 只出现在
**禁止进学生 policy 的名单**里，说明设计上预期它们会作为**独立 obs 组**（供路由/验证/日志），
且当前仓库**还没有任何配置**使用这套异构蒸馏（`HeterogeneousMultiExpertDagger` 无引用）。
⇒ 我们是第一个使用者：这些组的**定义权在我们手上**，只要不放进学生 policy 即可。

---

## 8. `height_scan` 扫描区域要不要跟越障/学生对齐？

### 判定规则（一句话）

> **共享网络 / 共享 normalizer ⇒ 必须逐位一致；各自独立网络 ⇒ 不需要。**
> 异构 DAgger 里每个教师各带 `obs_group` + `obs_format`，**天然互相隔离**。

### 实测几何对照

| | 我们（学生 623 契约 & 行走专家 626） | 越障教师（448） |
|---|---|---|
| `GridPatternCfg` | `(0.1, (3.0, 1.6))` | `(0.1, (3.0, 0.8))` |
| 落点范围 | 前后 **±1.5 m**、左右 ±0.8 m | 前后 **±1.5 m**、左右 ±0.4 m |
| 条数 | **527** | **279** |
| mesh 目标 | ⚠️ **仅 `/World/ground`** | `/World/ground` + `VaultBox`（`track_mesh_transforms=True`） |
| 挂载/朝向 | `torso_link`，(0,0,20)，`ray_alignment="yaw"` | 同 |

越障箱实测（`.../motions/vault/g1_speed_vault_retargeted.npz`）：
`obstacle_size = [0.4, 10.0, 0.6]`、`obstacle_pos = [1.7, 0.5, 0.055]`
⇒ 箱面在 **x ≈ 1.5 m**、**10 m 宽的墙**、高 0.6 m。

**结论：不需要对齐。**
- 前后覆盖**完全相同**（都刚好够到 x=1.5 的箱面）；
- 左右我们是**超集**（±0.8 ⊃ ±0.4）；
- 分辨率同为 0.1 ⇒ 数值语义可比。

### 与"学生策略"对齐？—— **已经一致，无需修改**

框架默认学生契约 **623 = `LocoTeacherPolicyObsCfg`**，其 `height_scan` 就是
`GridPatternCfg(resolution=0.1, size=(3.0, 1.6))` = **527**，与我们**逐位相同**。

### 三种"必须一致 / 必须改"的触发条件

1. **学生热启动**（`student_init_checkpoint` + `loco_frozen` / `loco_trainable`）：
   normalizer 逐维统计且被冻结 ⇒ 高度图的**维度、顺序、区域、mesh 目标、噪声**都必须逐位一致。
   （此场景下我们与学生契约本来就一致 ✅，所以仍然不用改。）
2. **同一个 env 里含障碍物**：必须给 `mesh_prim_paths` 加
   `MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/VaultBox", track_mesh_transforms=True)`，
   否则射线会**穿过箱子打到地面**，高度图在箱子处不出现凸起 ⇒ 教师/学生都被喂错输入。
   （我们用的就是 `MultiMeshRayCasterCfg` ✅，本来就支持多目标。）
3. **场景比扫描范围更大**：加大 `size` / `resolution` 会**改维度** ⇒ 必然重训，别为了"对齐"而改。

### ⚠️ 真正要担心的不是"区域对齐"，而是"分布覆盖"

我们的行走专家**只在平地上训过** ⇒ `height_scan` 恒等于常数 `torso_z − 0.5`，
**从未见过任何地形起伏**。一旦在越障场景里被查询，输入即为 **OOD** ⇒
- 蒸馏时**只在行走段采它的标签**（这正是 `teacher_route` 的用途）；或
- 让它在训练场景里见过障碍。

📌 另一个值得确认的线索：学长的越障动作集 docstring 写的是
*"Converted MimicKit G1 vault **and flat locomotion** motions"*
⇒ 这条链路的"行走段"本身可能就是**同一套运动参考框架下的平地 clip**，
而不是我们这种"速度指令 + AMP 风格判别"的专家。定义"行走专家"时这点很关键。
