# 蒸馏框架选型：为什么优先参考项目内 `instinct_rl` 的那套

> 本文回应团队讨论中的判断 —— **"PHP 的蒸馏框架与我们不同、参考权威性没那么高；项目内本身
> 就有 instinct 论文用的蒸馏框架，那个更该参考"**。
> 结论：**判断成立，但需要一处重要修正** —— 项目内那套**本身就是照 PHP 方法复刻的**，
> 两者不是"竞争关系"，而是"方法 vs 落地实现"的关系。下面给出证据与使用边界。

---

## 1. 结论（先给）

| 问题 | 结论 |
|---|---|
| 工作区里有独立的 "PHP 蒸馏框架" 吗？ | **没有**。搜遍 `instinct_xu/` 找不到 `*php*` 代码文件/包；`PHP` 只出现在**任务名、配置注释、设计文档**里（见 §3 证据 1、2） |
| 那"php 那套"指什么？ | 指 **PHP 论文的方法**（特权教师 + **DAgger + PPO 的 λ 课程 warmup**），以及**照它复刻进项目**的实现 |
| 项目内"instinct 自带的蒸馏框架"是哪个？ | `instinct_rl/algorithms/` 下的 **`TPPO` → `VaeDistill` / `VaePpoDistill` / `MultiTeacherVaePpoDistill`**（主线，有配置在用），另有 **`HeterogeneousMultiExpertDagger`**（已实现但**零配置引用**） |
| 为什么优先参考项目内的？ | ① 它与 InstinctLab 的**观测契约/ckpt 格式/任务注册**天然对齐；② 有 README、接口契约、任务配置三处落地；③ 上游是**固定 commit 的受控 fork** ⇒ 行为可复现（见 §3 证据 6） |
| **关键修正** | "项目内实现" **不是** 与 PHP 并行的另一套方法 —— `PLAN_expert_distill.md` 明写「蒸馏：**PHP 的 DAgger + PPO**（= `instinct_rl/algorithms/tppo.py`）」。所以正确口径是：**照项目内实现走 = 既符合 PHP 方法、又接得上我们的工程接口**；而"直接搬外部 PHP 代码"才是不该做的 |

一句话：**方法学参考 PHP，工程实现只认项目内 `instinct_rl`。**

---

## 2. 能力对照（选型用）

| 能力 | `TPPO` | `VaeDistill` | `VaePpoDistill` | `MultiTeacherVaePpoDistill` | `HeterogeneousMultiExpertDagger` |
|---|---|---|---|---|---|
| 定位 | 教师-学生蒸馏基类 | 纯蒸馏（禁 PPO） | **DAgger + PPO + λ 课程**（本项目主线） | 双教师按 route 选取 | 多教师异构路由 DAgger |
| 多教师 | ✗ | ✗ | ✗ | ✅（2 个） | ✅（任意个） |
| 每教师独立观测 | ✗（单一 `teacher_obs_group`） | 同左 | 同左 | 按 route 切换 | ✅ 每教师独立 `obs_group` |
| 蒸馏目标 | `real` / `mse_sum` / `l1` / `tanh` / `scaled_tanh` / `max_log_prob`（`kl` 未实现） | 继承 | 继承（本项目用 `mse_sum`） | 继承 | 纯 action MSE（不走 `distill_target`） |
| VAE / KL | ✗ | ✅ | ✅ | ✅ | ✗ |
| PPO 同时启用 | ✅ | **✗（强断言）** | ✅（带 λ_PPO/λ_D 课程） | ✅ | ✗（`compute_returns` 被删） |
| loss mask | ✗ | ✗ | ✅ | ✅ | ✅（`teacher_valid_mask`） |
| **有配置在用？** | 基类 | ✓（`VaeDistill` 任务） | ✅ **主线** | 视需要 | **❌ 零引用** |

**本项目当前实际用的**（证据见 §3）：

- 教师：`Instinct-Perceptive-Vault-G1-v0`（`PhpTeacherPolicyObsCfg`：高程图 + 全局状态，**无深度**）
- 学生：`class_name = "VaePpoDistill"`，`distill_target="mse_sum"`，
  `teacher_obs_group="teacher"`、`distillation_mask_obs_group="distill_mask"`，
  `distillation_loss_coef=10.0`（下限 0.1）、`loss_curriculum_iterations=10000`
  ⇒ 完全就是 **PHP 的 `L = λ_PPO·L_PPO + λ_D·L_D`** 与 Table VI 超参

---

## 3. 证据（文件:行）

1. **"PHP" 只作为方法/命名存在，没有独立框架**
   - `instinct_xu/.../perceptive/config/g1/perceptive_vault_cfg.py:206` —
     `class PhpTeacherPolicyObsCfg(ObsGroupCfg)`，docstring：
     *"PHP privileged teacher actor observations (heightmap + global state, no depth)."*
   - 同文件 `:261` `PhpTeacherCriticObsCfg`；`:37` / `:46` / `:73` / `:131` 多处按 PHP 论文设定取值
     （深度分辨率"PHP uses 87x58"、"PHP grid only reaches ±0.35 m"、"PHP: obstacle yaw randomized ±45°"）

2. **蒸馏方法 = PHP；落地实现 = 项目内文件**
   - `PLAN_expert_distill.md:31`（三篇论文对比表）：
     「蒸馏 | **DAgger + PPO（λ 课程 warmup）** | — | DAgger + PPO + 迭代自增广 |
     **PHP 的 DAgger + PPO**（= `instinct_rl/algorithms/tppo.py`）」

3. **主线配置确实在用 `VaePpoDistill`**
   - `instinct_xu/.../agents/instinct_rl_vault_vae_distill_cfg.py:55` `class_name = "VaePpoDistill"`
   - 同文件 `:60` `# PHP Table VI (Distillation).`
   - 同文件 `:~80` `# PHP: L = λ_PPO L_PPO + λ_D L_D, with base DAgger scale 10 and λ_D floored at 0.1.`
   - 同文件 `:~88` `teacher_obs_group = "teacher"` / `distillation_mask_obs_group = "distill_mask"`

4. **算法导出清单（项目内可用集合）**
   - `instinct_rl/algorithms/__init__.py:31-37,51-58` ——
     `TPPO`、`EstimatorPPO`/`EstimatorTPPO`、`HeterogeneousMultiExpertDagger`、
     `VaeDistill`、`VaePpoDistill`、`MultiTeacherVaePpoDistill`

5. **`HeterogeneousMultiExpertDagger` 已实现但无人使用**
   - 全仓库检索只命中它自己的文件与 `__init__.py`：
     `algorithms/heterogeneous_dagger.py:157,381`、`algorithms/__init__.py:32,54`
     ⇒ **没有任何任务配置引用它**（多教师需求出现时，可直接以它为起点扩展）

6. **上游是固定 commit 的受控 fork（可复现性证据）**
   - `instinct_xu/README.md:79-89` —— 原版来自 `ChaoMingSanHua/InstinctLab` 与
     `ChaoMingSanHua/instinct_rl`，并 **checkout 固定提交 `5523e3847cfce578bfcee25a5334256d9cc53e0e`**

7. **README 把 VAE 蒸馏列为包内主线任务**
   - `instinct_xu/README.md:9` —— 「Vault 视觉学生蒸馏 | `Instinct-Perceptive-Vault-VaeDistill-G1-v0`」

---

## 4. 使用边界（重要，避免踩坑）

1. **不要搬外部 PHP 代码进来**：外部实现的观测打包、`obs_group` 命名、loss 掩码语义、
   ckpt 加载格式都与本项目不同；一旦混用，`OnPolicyRunner` 的
   `class_name = "module:Class"` 解析、`teacher_logdir` 加载、契约校验都会对不上。
2. **优先扩展项目内类**：
   - 单教师、要 PPO + 课程 ⇒ `VaePpoDistill`（照现有 vault 配置改）
   - 多教师、每教师观测不同 ⇒ `HeterogeneousMultiExpertDagger`（**注意：它关了 PPO，是纯 DAgger**）
   - 多教师但只需 2 个 + 要 PPO ⇒ `MultiTeacherVaePpoDistill`
3. **蒸馏配置是"按观测项名"索引的**，这是**与换 URDF 直接耦合**的地方：
   `teacher_obs_group` / `distillation_mask_obs_group` / `teacher_policy.obs_format`
   一旦换资产导致**观测项名或维度**变化，这些必须同步 —— 否则会静默用错项（或维度 assert 报错）。
   ⇒ 这也是为什么 URDF 替换指南里把**契约校验器（Step 9）**列为必做项。
4. **VAE 不等于必须用**：本项目 vault 学生的 `kl_loss_coef = 0.0`（注释："No VAE latent;
   keep coef at 0 so the zeroed KL term does not affect the loss"）⇒ 用的是 `VaePpoDistill`
   的 **PPO + DAgger 课程**部分，而非 VAE 潜空间。选型时别被类名误导。

---

## 5. 一句话交接

> 需要蒸馏时：**照 `instinct_rl/algorithms` 里已有的类**（当前主线是 `VaePpoDistill`）**和现有
> vault 配置**改，方法学上它就是 PHP 的 DAgger+PPO λ 课程；`HeterogeneousMultiExpertDagger`
> 是已备好的多教师路线但那套目前**没有配置在用**，启用前请先补配置与契约。
