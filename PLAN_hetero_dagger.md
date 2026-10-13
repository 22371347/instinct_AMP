# 异构 DAgger 蒸馏 —— 方案分析（阶段 D1：双专家 → 单一 height-scan 学生）

> 日期：2026-10-11　状态：**只做方案，不写代码**
> 参考：LLP（arXiv:2608.02653，`论文蒸馏设计.md`）、PHP（arXiv:2602.15827，`refs/php_parkour` + Holosoma `jinkunc/php-release-port` 分支）、本仓库已有实现
> 平台：Unitree G1 29-DoF（torsobase / torso_link 为根）

---

## 0. 一句话方向

**按 LLP 的 V-C1 做"多专家 → 单学生"的合并，但损失采用 PHP 的 DAgger+PPO 混合配方（λ 课程），
学生保持前馈 + 多帧历史（不上 GRU）；环境用"单场景 + 路由划分 + 有效掩码"，
两个专家各带自己的 raw 观测组。** 本阶段产物 = height-scan 学生（还不能上真机），
切换微调（V-C2）与深度蒸馏（Section VI）**明确不在本阶段**。

---

## 1. 三份材料的定位对照

| 维度 | LLP（论文） | PHP（实际代码） | 本仓库已有 |
|---|---|---|---|
| 第一次蒸馏形态 | 特权 → height-scan | 特权 → **深度**（跳过 scan） | `HeterogeneousMultiExpertDagger`（**纯 DAgger**，多教师） |
| 损失 | **纯 DAgger**（V-C1） | **DAgger + PPO，λ 课程**；纯 DAgger 版成功率 0.16/0.03/0.12 | ① 异构 dagger = 纯 DAgger；② `VaePpoAlgoMixin` = **已实现 PHP 的 λ 课程**（单教师） |
| 学生结构 | **必须 RNN/GRU**（无 GRU 消融断崖） | **前馈**（MLP 2048-1024-512-256-128 + 深度 backbone 58×87） | 框架**禁止 RNN**（`is_recurrent` → `NotImplementedError`） |
| 学生时间上下文 | GRU | 帧堆叠（actor_obs `history_length=1`，但 `vel_cmd` 是 15 维 one-hot 速度档） | `policy_obs_frames` **7 项 × 4 帧 = 1500**（我们的 AMP 行走就是 4 帧） |
| 路由 | critic 给 one-hot，actor 不给 | `which_motion` 列塞在 teacher_obs 里，`-1` = 专家丢失 → 零动作 + DAgger 掩码 | `teacher_route`(1) + `teacher_valid_mask`(1) 独立 obs 组 |
| 专家异构 | 允许（观测/奖励/训练方式都可不同） | 允许（每个专家一个 dataset + 一个 checkpoint） | 允许（每教师一个 `obs_group` + 独立 normalizer） |
| 环境分组 | 一个环境分若干组 | 一个 env，每 env 一个专家（**expert-balanced 采样**） | env 侧自己给 route/mask（**无现成配置**） |

**关键结论**：LLP 与 PHP 在"损失"上不一致（纯 DAgger vs DAgger+PPO），而 PHP 有**可复现的消融证据**
说明纯 DAgger 会崩。我们**取 PHP 的损失**、**取 LLP 的 stage 划分与观测目标**。

---

## 2. 本仓库现状盘点（这决定了"要准备什么"）

### 2.1 已有（可直接复用）

| 资产 | 位置 | 说明 |
|---|---|---|
| 异构多专家 DAgger 算法 | `instinct_xu/source/instinct_rl/instinct_rl/algorithms/heterogeneous_dagger.py` | 多教师 + route + valid mask + `student_init_checkpoint`(只载 actor, strict) + normalizer 模式（`fresh_trainable/loco_frozen/loco_trainable`）+ teacher roll-in 课程 |
| **DAgger+PPO λ 课程** | `.../algorithms/vae_ppo_distill.py`（`VaePpoAlgoMixin`） | 已按 **PHP Eq.(3)** 实现：`L = λ_PPO·L_PPO + λ_D·L_D`、`λ_D=max(λ_Dmin,1−k/K)`、`adaptive_schedule_min_ppo_coef` 门控自适应 LR、`distillation_loss_mask` 掩码 |
| **统一蒸馏 env 蓝图（E2）** | `g1_scan_loco_9600_train_sim2sim/.../config/g1/unified_scan_vault_distill_cfg.py`（473 行） | 学生 1009 维 + `loco_teacher`/`vault_teacher`/`teacher_route`/`teacher_valid_mask` + 单场景（越障场景）+ 按路由隐藏箱子 + 分路由 reset + 关掉参考跟踪终止 |
| E2 算法配置 | 同目录 `agents/instinct_rl_unified_scan_vault_distill_cfg.py` | `class_name="HeterogeneousMultiExpertDagger"`；两教师（loco 1009 / vault 448）；`student_init_checkpoint=LOCO_TEACHER` + `policy_normalizer_mode="loco_frozen"` |
| **dagger v2（比我们新 6 天）** | 同目录 `source/instinct_rl/.../heterogeneous_dagger.py` | 相对 `instinct_xu` 版新增：① 支持 **RSL-RL 归一化键** `obs_norm_state_dict`(+`EmpiricalNormalization`)；② `teacher_rollin_uses_invalid_labels`（roll-in 不受 valid 限制，但标签仍被掩码） |
| 越障教师侧全部资产 | `instinct_xu/.../perceptive/config/g1/perceptive_vault_cfg.py` | `PhpTeacherPolicyObsCfg`(**448**)、`VAULT_BOX_PRIM_EXPR`、`reset_vault_box_pose_by_motion_randomized`、`VAULT_MOTION_REFERENCE_CFG` |
| 越障蒸馏既有实现 | `.../perceptive_vault_vae_distill_cfg.py` | `distillation_loss_mask(env)`（跟踪误差门）、`vault_velocity_command(env)`（可部署速度指令）、教师组 `teacher`/掩码组 `distill_mask` |
| 行走专家（我们自己的教师） | `instinct_xu/.../locomotion/config/g1/amp_walk_*` | obs **1500** = 7 项 × **4 帧**（scan 279 + cmd/grav/angvel 各 3 + jp/jv/la 各 29）；hidden (512,256,128)；`EmpiricalNormalization`；动作 29 维 `scale=1.0` + `use_default_offset` |
| 评测/回放 | `instinct/tools/*`、`g1_scan_loco_9600_train_sim2sim/deploy_mujoco` | MuJoCo sim2sim、ONNX 导出、绘图脚本 |

### 2.2 缺口（= 本阶段要"准备"的东西）

1. `instinct_xu` 里**没有**统一（多专家）蒸馏配置 —— 只有 E2 快照里那份，且它绑的是 OmniLoco **1009** 教师，不是我们的 AMP 行走。
2. `HeterogeneousMultiExpertDagger` **没有 PPO 项**（`compute_returns` 被 delete，只做动作回归）⇒ 要么加 PPO 协同项，要么接受纯 DAgger（PHP 证明会崩）。
3. **学生不能被 RNN 化**（框架硬限制）⇒ 时间上下文只能靠帧堆叠（或后续单独开工作项改框架）。
4. 我们**两个教师的观测契约不同源**：行走 1500（torso 根、scan 279、4 帧），越障 448（历史长度 `TEACHER_PROPRIO_HISTORY_LENGTH`）⇒ 学生统一观测需要重新定稿（见 §4 D1）。
5. **前置阻塞**：腰部关节符号（URDF 约定）问题未决 —— 它同时影响两个专家的参考数据与真机部署，**下周组会定案前不宜冻结专家**。

---

## 3. 方向：分阶段，本阶段只做 D1

| 阶段 | 内容 | 参考 | 本阶段 |
|---|---|---|---|
| **D0 前置** | 腰部符号/URDF 约定定案（组会）⇒ 决定是否需要重训两个专家 | —— | ⏳ 阻塞项 |
| **D1 双专家合并** | 行走专家 + 越障专家 → 单学生（height-scan），env 分组 + route/mask | LLP V-C1 + PHP 损失 | ✅ 本方案 |
| **D2 切换微调** | 加 transition 组/相位条件 AMP，学"走↔越"的过渡 | LLP V-C2 | ✖ 后续 |
| **D3 深度蒸馏** | scan 学生 → 深度学生（真机可部署）+ 延迟/噪声对齐 | LLP VI / PHP WARP 相机 | ✖ 后续 |

---

## 4. 关键设计决策（含推荐）

### D1 ⭐ 学生观测契约：`S1 对齐行走专家(1500)` vs `S2 对齐 E2(1009)`

| | **S1（推荐先做）** | S2（E2 原样） |
|---|---|---|
| 学生 obs | = 行走专家现有 1500 布局（scan 279 需把越障箱体加进 ray-caster 的 mesh 列表；4 帧历史） | 1009 = 5 帧本体(15/15/15/145/145/145) + `phase`(2) + scan 527 |
| 能否热启动 | ✅ **可以**：`student_init_checkpoint` = 行走专家 checkpoint，`policy_normalizer_mode="loco_frozen"`（E2 就是这么做 loco 的） | ❌ 只能从零（或先做一个"零列扩维"热启动的小改动） |
| 越障可见性 | 需确认 scan 279 的网格/朝向能不能"看见"箱子（不行则扩维或换网格） | 已验证（MultiMeshRayCaster 地面 + 箱子） |
| 风险 | 没有 `phase`，越障下坡段可能缺相位线索 | 从零训练，收敛更慢 |
| 结论 | **先跑 S1**（省时间、且能直接复用现有行走权重）；若"走→越"切换失败，再上 `phase`（可用"零列扩维热启动"或从零） | 备选 |

> 注：**加 `phase` 会破坏与行走 checkpoint 的维度一致**（strict=True 直接报错），
> 若要用 phase + 热启动，需要"新输入列置零"的扩维技巧（小改动，列为 D1 的可选子项）。

### D2 ⭐ 损失：纯 DAgger vs DAgger+PPO

| 方案 | 依据 | 推荐 |
|---|---|---|
| 纯 DAgger | LLP V-C1 原文；我们已实现 | 只作为**冒烟测试**（1~2k 迭代看 schema/掩码/标签是否通） |
| **DAgger + PPO（λ 课程）** | PHP：纯 DAgger 成功率 0.16/0.03/0.12；`L = L_value + mean(coef·(surrogate − c_e·entropy)) + mean((1−coef)·c_dagger·L_dagger)`，`coef = gate·ppo_coef`，`ppo_coef` 在 `[ppo_start_epoch, dagger_end_epoch]` 内 0→0.9，`c_dagger=10`，`value_loss` **不**被门控（critic 从第 0 迭代就训）；`coef≤0.1` 时 std 的学习率置 0、自适应 KL 关闭 | ✅ **主路线** |

移植落点：把 `VaePpoAlgoMixin` 的 λ 课程 + `_dagger_loss` 逻辑**并入** `HeterogeneousMultiExpertDagger`（多教师 → 共享一个 `distill_mask`/`route`，PPO 项用同一批样本）。PHP 的**残差门控 gate**（按运动相位分段、EMA α=1e-3、warmup 5e5 样本、floor 0.5、分位 0.5/0.95、pre-smear 1 s）属于**二阶优化**，建议**先不做**，用一个常数 `ppo_coef` 或简单线性课程即可。

### D3 ⭐ 路由与有效掩码（环境侧）

沿用 E2 已实现的口径，几乎可直接抄：

| 组件 | E2 做法 | 我们要做的调整 |
|---|---|---|
| 分组 | 固定 env-index 划分：`vault_fraction=0.30`（`route` 稳定，利于日志） | 可先 0.3；观察两个技能的样本比 |
| `teacher_route` | 1 维，0=loco / 1=vault | 不变（教师组各自独立） |
| `teacher_valid_mask` | loco 全 1；vault = 参考 x∈[0,4] ×（跟踪误差门 `distillation_loss_mask`） | 换成我们越障任务的相位窗口（箱子前后） |
| 速度指令 | `unified_velocity_command`：loco 用 `base_velocity`，vault 用 `vault_velocity_command` ⇒ **单一指令通道、不泄露技能** | 保留（这是"学生不许知道自己在哪组"的关键） |
| 隐藏技能 | 学生 obs 里**没有** skill/route/phase（phase 只是时钟） | 保留；`forbidden_policy_terms` 已含 `skill_id/reference/vault_phase` |
| roll-in | E2 用 `teacher_rollin=1.0`（纯专家轨迹，属于诊断档） | 正式跑要开**真 DAgger**：从 1.0 衰减到 ~0.1（或纯学生 on-policy），否则学生没见过自己的偏差分布 |

### D4 学生时间上下文

前馈（框架限制）+ **4~5 帧历史** 起步；GRU 作为**独立后续工作项**（需同时改：dagger 的 recurrent 断言 + RSL-RL 的 TBPTT/序列存储）。判定标准：切换成功率与"越过箱后仍保持步态"的表现。

### D5 PPO 侧的奖励与 critic

- 奖励：PHP 结论是**极不敏感**，`is_alive` 量级即可；我们建议用**每个路由各自的弱任务奖励**（loco：速度跟踪少量权重；vault：存活 + 不撞箱），**避免两路由奖励互相打架**。
- critic：**加 route one-hot**（LLP 与 PHP 都这么做）；actor 不给。注意我们 `student_obs_schema` 与 `critic` 组是分开的，critic 加 one-hot 不影响学生契约。

### D6 前置：腰部符号（D0）与 `base_lin_vel`

- **腰部符号未决 = 专家未冻结**。若最终改 URDF/MJCF ⇒ 两个专家的参考数据与策略全部需要重训/重转 ⇒ **D1 的教师 checkpoint 要等 D0 之后再冻结**。
- `base_lin_vel`：行走专家是 626/1500 口径的一部分（特权）。它只影响"学生热启动"（S1 里学生=行走专家布局 ⇒ 需与学生契约一致）。**建议**：D1 里行走教师保持原样（教师观测不必与学生一致），学生用 S1 布局时按需要决定是否保留该项。

---

## 5. 需要准备的资产清单

### A. 教师侧（冻结的"教师策略"资产）

| # | 资产 | 现状 | 待办 | 阻塞 |
|---|---|---|---|---|
| A1 | 行走专家 checkpoint（确切 `model_*.pt`，非 glob/目录） | 云上 T2/T5 run 有 | 选**一版最终**（建议 D0 定案后重训一版） | D0 |
| A2 | 行走专家的 `params/agent.yaml` + normalizer 权重 | 训练时自动存 | 确认存在（`TeacherAdapter` 硬依赖，缺失会报错） | – |
| A3 | 越障专家 checkpoint + `agent.yaml` | 需确认（E2 引用 `checkpoints/teachers/vault/model_8000.pt`，本地快照**只有 loco**） | 找到/重训并归档到 `checkpoints/teachers/vault/` | D0 |
| A4 | 两教师网络契约 | 行走 hidden (512,256,128)/ELU；越障同为 (512,256,128) | 核对 `agent.yaml` 里 `class_name`+hidden，写进 `teacher_spec.actor_cfg` | – |
| A5 | 动作语义一致性 | 两边都是 29 维、`scale=1.0`、`use_default_offset=True`、同默认姿态 | 逐字核对（`action_adapter` 只支持 `identity`） | – |

### B. 环境侧（**新建配置**，不是新算法）

| # | 资产 | 现状 | 待办 |
|---|---|---|---|
| B1 | 统一场景：越障场景 + 学生 height scanner（`MultiMeshRayCaster`，地面 + 箱体，`track_mesh_transforms=True`） | E2 已有（scan 527） | 按 D1 决定网格（S1 用 279 加密箱体 / S2 用 527） |
| B2 | `teacher_route` / `teacher_valid_mask` 两个 1 维 obs 组 | E2 已有 | 换成我们越障任务的相位窗口 |
| B3 | 学生 `policy` obs 组 | E2 已有（1009） | 按 D1 定稿 |
| B4 | 两个教师 obs 组（`loco_teacher` = 行走 1500 / `vault_teacher` = 448） | 越障组已有；行走组需新建 | 新建 `loco_teacher`，逐项对齐行走 checkpoint 的 obs |
| B5 | 分路由 reset（vault 从参考重置、loco 均匀重置；箱子按路由隐藏） | E2 已有 | 直接迁移 |
| B6 | 终止项裁剪（关掉参考跟踪类终止，对 loco 路由无效）+ episode 15 s + 4096 envs | E2 已有 | 直接迁移 |
| B7 | 速度指令统一通道 | E2 已有 | 直接迁移 |
| B8 | 任务注册（`gym.register`：`Instinct-HeteroDistill-WalkVault-G1-v0` 等） | 快照里有 E2 的注册 | 新注册（并保留 `-Play-` 变体） |

### C. 算法侧（后续写代码时的清单）

| # | 改动 | 依据 |
|---|---|---|
| C1 | 把快照 dagger v2 差异合入 `instinct_xu`（RSL-RL 归一化键 + `teacher_rollin_uses_invalid_labels`） | 已存在的两处修正 |
| C2 | 在异构 dagger 里加入 **PPO 协同项 + λ 课程**（复用 `VaePpoAlgoMixin` 的公式） | D2 |
| C3 | critic 输入加 **route one-hot** | LLP/PHP |
| C4 | （可选）phase 扩维热启动；（后续）RNN 学生支持 | D1/D4 |

### D. 运行与评测

| # | 资产 | 现状 |
|---|---|---|
| D1' | 教师 checkpoint 路径管理（env var，如 `HETERO_LOCO_TEACHER_CHECKPOINT`） | E2 用 `UNIFIED_*_TEACHER_CHECKPOINT` 环境变量 |
| D2' | 评测脚本：走→箱切换成功率、掩码后重放、MuJoCo sim2sim | sim2sim 已有（快照 `deploy_mujoco/`），成功率脚本需新写 |
| D3' | 监控指标 | 每路由样本占比 / valid 掩码率 / 各教师 BC loss / `behavior` / `ppo_coef` / gate 均值与 spread |

---

## 6. 风险与对策（PHP/LLP 踩过的坑，逐条对照）

| 风险 | 证据 | 对策 |
|---|---|---|
| 纯 DAgger 崩 | PHP 消融 0.16/0.03/0.12 | 上 PPO 协同项（D2）；纯 DAgger 只做冒烟 |
| 学生早期被终止项打死 | PHP 把 `bad_tracking` 阈值在 24k→240k(step) 内放宽 2× | 蒸馏环境里**关掉/放宽**参考跟踪类终止；episode 15 s |
| 专家样本失衡 | PHP：600 clip 的 registry 会占 69% env，单个教师压过其他 | **expert-balanced** 采样：按 route 均匀分配 env，不看 clip 数 |
| 标签把"专家也做不到"的样本喂给学生 | PHP `which_motion=-1` → 零动作 + 掩码 | 我们的 `teacher_valid_mask`：跟踪误差门 + 相位窗口 |
| 掩码/门控全 0 导致"无梯度" | PHP 明确警告 floor 不能取 0（自封门） | 若做 gate，floor ≥0.5；先不做 gate 更安全 |
| 学生缺时间上下文 | LLP：无 GRU 断崖（99.9→34.6 等） | 帧堆叠起步；若切换失败再开 RNN 工作项（D4） |
| height-scan 学生**不能上真机** | LLP 明确 | 本阶段不承诺部署；D3 深度蒸馏另开 |
| 学生不知道自己在哪组 | LLP/PHP 一致设计 | 学生 obs 严禁含 route/skill/phase（phase 只作时钟，不算技能标签） |
| 帧堆叠 vs 教师单帧错配 | 我们行走是 4 帧、越障 `TEACHER_PROPRIO_HISTORY_LENGTH` 待确认 | B4 里逐项核对，教师组可以与学生不同（这正是"异构"的意义） |

---

## 7. 验收与监控

**阶段 D1 通过判据（建议）**
1. 两个教师的 label 都在被消费：每路由 `valid_ratio > 0`、`Distill/{name}_loss` 都在下降；
2. `behavior`(BC loss) 单调下降；加 PPO 后 `ppo_coef` 曲线与 `return` 同步上升；
3. 走→箱→走 的**连续成功**（一组 seed 上的成功率），而不只是"各自分开能跑"；
4. MuJoCo sim2sim 回放：过箱后不摔、箱子前后步态连续。

**必须持续看的日志**：每路由样本占比、valid 掩码率、各教师 loss、`ppo_coef`、`episode_length`、`return`。

---

## 8. 下一步（本轮不写代码，只做资产与确认）

1. **等 D0（腰部符号）组会结论** —— 它决定两个专家是否需要重训，是所有 checkpoint 冻结的前提。
2. 把 E2 的 `unified_scan_vault_distill_cfg.py` + `instinct_rl_unified_scan_vault_distill_cfg.py` 挪到 `instinct_xu`（**只搬运/阅读，不改逻辑**），并确认它能在我们树上 import。
3. 找到/确认越障专家 checkpoint 与行走专家 checkpoint 的 `params/agent.yaml`，把 `teacher_spec` 的 6 个字段值先填成一张表（见 §5 A）。
4. 确认行走专家 scan(279) 的 ray-caster mesh 列表能否加入越障箱体（决定 D1-S1 是否可行）。
5. 决定 D1/D2/D4 三个开关（本方案推荐：S1 + DAgger+PPO + 4~5 帧堆叠）。
