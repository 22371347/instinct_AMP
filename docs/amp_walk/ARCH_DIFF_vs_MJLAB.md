# 我们 vs AMP_mjlab：**根本性差异清单**

> 生成于 2026-10-08，数据来源：
> * 我们：`docs/amp_walk/EFFECTIVE_REWARDS.md`（**运行期实例化 dump**，65 臂 × 奖励 + 算法侧）
> * mjlab：`AMP_mjlab/src/tasks/amp_loco/{config/g1/rl_cfg.py, amp_env_cfg.py, config/g1/env_cfgs.py}`
>   + `AMP_mjlab/rsl_rl/modules/discriminator.py`（**读实现，不读注释**）
>
> ⚠️ 本文只写**核实过**的差异。数值差 ≠ 强度差（见 §B3 的 λ 教训），凡涉及"强度"都给出**公式与算式**。

---

## 0. 一句话结论

**mjlab 把 AMP 当"弱风格先验"，我们把 AMP 当"强风格约束"，并用一堆显式奖励去补偿它的副作用。**
这个范式差异派生出了下面绝大多数具体差异。

mjlab 的 `rl_cfg.py` 里原文写着：

> `amp_reward_coef` is lowered below 0.1 so task rewards (velocity tracking, height,
> default-pose) dominate; the AMP discriminator acts as a **weak style** [prior]

---

## 1. ⭐⭐⭐ 风格强度：一个**从未对齐**的量（差 4~40 倍）

### 1.1 两边公式（都已读实现确证）

```python
# ---- mjlab（rsl_rl/modules/discriminator.py::predict_amp_reward）----
d = amp_linear(trunk(cat([state, next_state], dim=-1)))
shape = torch.clamp(1 - 0.25 * (d - 1) ** 2, min=0)        # bounded_quad
reward = self.amp_reward_coef * shape                       # = 0.1 × shape
r = (1 - task_reward_lerp) * reward + task_reward_lerp * task_reward

# ---- 我们（instinct_rl/algorithms.py 类 docstring 原文）----
aux = (1 - λ) * (r_amp - r_task / c)
R   = r_task + c * aux
#  展开 ⇒ R = λ·r_task + (1 - λ)·c·r_amp
```

⇒ **两边同构**，**有效风格权重 = `(1−λ) · coef`**
（我们的 `coef` = `algorithm.discriminator_reward_coef`，mjlab = `amp_reward_coef`）

### 1.2 实测对比

| | λ (`style_lerp` / `task_reward_lerp`) | coef | **有效风格权重** | vs mjlab |
|---|---|---|---|---|
| **mjlab** | 0.75 | **0.1** | **0.0250** | — |
| 我们 V22/V23/V27/V30/V31/V33S/V33ACold | 0.75 | **1.0** | **0.2500** | **10×** |
| 我们 T1b | 0.75 | **4.0** | **1.0000** | **40×** |
| 我们 T1c / T1f | 0.90 | 4.0 | **0.4000** | **16×** |
| 我们 **T1i / T1j / T1k / T1l（当前）** | 0.975 | **4.0** | **0.1000** | **4×** |

⚠️ **重要更正**：此前多处（含 `EXPERIMENTS.md §B3` 的推理）说
"λ=0.975 ⇒ 风格强度 = (1−λ)·D = 0.025 = mjlab 等强" —— **漏乘了 `coef`**。
`T1a` 的奖励白名单把 `coef` 设成 **4.0**（V33S 是 1.0），所以：

* T1i 的 λ=0.975 只是把风格从 **10× 降到 4×**，**从未到过 1×**；
* **当前 T1l 的风格强度仍是 mjlab 的 4 倍。**

### 1.3 正确的对齐做法

`coef` 是**纯风格强度旋钮**（展开后它只乘 `r_amp`，不碰 `r_task`）⇒ 对齐 mjlab 应该**调 `coef`**：

```
coef: 4.0 → 1.0     且   λ 保持 0.975      ⇒ (1−0.975)×1.0 = 0.0250 = mjlab **精确值**
```

（继续推 λ 是次优选择：λ 同时是"task 插值权重"，推到 0.99+ 会挤压数值精度且语义混乱。）

---

## 2. ⭐⭐ 判别器的**配套设计**：白化 vs 浅网

| | mjlab | 我们 |
|---|---|---|
| 判别器输入白化 | **有**（`EmpiricalNormalization`）| **无**（`discriminator_kwargs` 只有 `hidden_sizes`/`nonlinearity`）|
| `min_normalized_std` | **`[0.05] * 29`** | **无** |
| 判别器容量 | **`[1024, 512, 256]`**（3 层）| `[1024, 512]`（2 层）|

* 我们缺白化是**已确证的事实**（V31 的 ckpt 里 `discriminator` 只有 3×Linear 的 W/b，
  无任何 `normalizer.*`；`amp_normalizer_expert_only` 是**幽灵开关**）——详见
  `EXPERIMENTS.md` 勘误记录（四）。
* `min_normalized_std` 的作用是**给白化分母设下限**，防"σ≈0 的维把差距放大到离谱"。
  mjlab **同时**有白化和这个下限 ⇒ 它的深网不会因为"某维 σ≈0"而爆掉。
* ⇒ **我们的 2 层浅网不是"独立的选择"，而是"缺白化"的补偿**：没有白化时，
  深层网络会更快把 actor 侧压到饱和（离线 bake-off 实测 `zero_frac` 0.013 → 0.744）。
* ⚠️ **推论（防止误读）**：`EXPERIMENTS.md §A2/A5` 与 memory 里"加第三层更差"的结论，
  **只在我们"无白化"的口径下成立** ⇒ **不能据此判断 mjlab 的 3 层设计有问题**
  （这是 §B3 那类"特定口径失败被当普遍结论"的又一例）。
  ⇒ 要评估"加层"，必须**先接上白化 + `min_normalized_std`**，再加层。

---

## 3. ⭐⭐ 参考批与 episode 初始化

| | mjlab（walk 任务）| 我们 |
|---|---|---|
| actor reset 来源 | **参考帧**（`reset_from_motion_data`）| **参考帧**（`envs/mdp/events/motion_reference.py::reset_robot_state_by_reference`）⇒ **两边都有 RSI** |
| ⭐ **抽哪一帧** | **全池（所有 clip 所有帧 `_concat_frames`）均匀抽**（`torch.randint`）⇒ **任意 clip、任意相位** | 随机选 clip，**但起点固定 = 第 0 帧**（`motion_start_from_middle_range=(0,0)` + `motion_bin_length_s=None`）|
| 对应的开关 | — | `env_starting_stub_sampling_strategy`：我们是**默认 `"independent"`**；`"concat_motion_bins"` 才是“全池拼接”语义 |
| 专家批来源 | 动作数据集 | **离线数据集 i.i.d.**（V30 起，`amp_expert_states_396.pt`，2929×396）|

### ⚠️ 本条曾被我写错，已更正
先前写的“**我们完全没有 motion reset**”是**错的** —— actor 确实从参考帧初始化。
**真正的差异是相位覆盖**：mjlab 每次 reset 从“所有 clip 的所有帧拼成的大池”均匀抽帧，
而我们**总是从某条 clip 的第 0 帧开始**。

⇒ 这正是“mjlab 在**静止状态**也能保持 AMP 风格、且不梯度稀疏”的关键：
它的策略**随时被扔回运动的任何相位**，因此 actor 侧状态分布**始终与参考分布重叠**；
而我们总从“起点”开始，策略容易学成“从头的固定套路”，actor 分布与参考分布逐渐分离
（memory 实测二者联合 gap ≈ **37σ** ⇒ 判别器一步分开、风格奖励变常数）。
⇒ 对比经典 RSI 的作用：**让策略见过各种相位**——它同时也是“AMP 能否把策略拉向参考”的前提。

---

## 4. ⭐ 参考集覆盖面的裁剪

| | mjlab | 我们 |
|---|---|---|
| 动作目录 | `WalkandRun`（我们 17-clip 的来源）+ **`Recovery`（摔倒恢复）** | **5 clip**（`amp_walk_v13_motion_selection.yaml` 精选）|
| 目录内含 | 行走 + 慢跑（jog）| 无慢跑、无恢复、无转身 |

* 我们**大幅裁剪**了参考集（依据："运动数据过多会导致风格混乱"，`amp_walk_v3_env_cfg.py`）。
* ⇒ 直接后果见 `EXPERIMENTS.md §D1`（"数据集覆盖面决定 AMP 会压制什么"），以及
  至今未解的 `wz` 三档 ≈0.01（推测缺"转身"模态——5 clip 里确实没有 `idle_turn_*`）。
* mjlab 的任务范围也更大：`lin_vel_x (−0.7, 2.0)`（含慢跑）+ 地形课程；我们
  `(−0.5, 1.0)` 平坦。

---

## 5. ⭐ 姿态的**责任分工**：显式奖励 vs AMP

mjlab 的 walk 奖励集里**没有**这些（我们的 7 项额外项）：

`flat_orientation_l2` · `lin_vel_z_l2` · `feet_air_time` · `dof_torques_l2` ·
`joint_deviation_hip/arms/torso/knee`

它是靠 **AMP 风格 + `track_root_height`** 保持姿态的。而我们：

* 有 7 项额外奖励去"管姿态"；
* `V23` 还把 `track_default_pose_on_stand` 从"只有 `waist_yaw` 一个关节"扩成
  **全关节 + 逐关节权重表 + 死区表**（`knee`/`ankle` 权重 **2.0**）；
* 结果是**同一件事有两套机制在管**（AMP 想管、显式奖励也在管），互相争夺预算。

⇒ 这与 memory 里的 V30 结论同源：
**"AMP 不可能把『小跳』引导成『行走』"**（判别器必然饱和、风格奖励变常数）
⇒ 所以过去只能不断加显式奖励——但**加显式奖励又让策略离参考分布更远**（hand-arm 姿态冲突就是实例）。

---

## 6. 数值级差异（次要，但要记）

| 项 | mjlab | 我们（T1l 实际生效）| 备注 |
|---|---|---|---|
| `policy.actor/critic_hidden_dims` | `(512, 256, 128)` | `[256, 128, 128]` | 我们更窄 |
| `algorithm.entropy_coef` | `0.005` | `0.008` | |
| `algorithm.learning_rate` | `1e-3` | `1e-3` | ✅ |
| `mirror_loss_coeff` | `0.1` | `0.1` | ✅ |
| `num_steps_per_env` | `24` | `24` | ✅ |
| 判别器 lr | 见 `rl_cfg` | `1e-4` | 两边都低于 policy lr |
| AMP 状态内容 | 13 body × 15 dim × **2 帧拼接** = 390 | 同 + **anchor 重力 6** = **396** | ⭐ 我们多一条"绝对倾角"通道 |
| `track_lin_vel_xy_exp` std | `1.0` | `0.5` | 我们**更严格**（灵敏度 ÷2²）|
| `track_ang_vel_z_exp` std | `3.14` | `0.5` | 我们**严格 ~33×** |
| `track_base_height` std | `0.3` | `0.2`（T1l 已改 0.3）| V9H 收紧 |
| `dof_pos_limits`（= `joint_pos_limits`）| **`−10`，margin 版，全关节** | `−1`，**线性版，仅踝** | 我们轻 10× |
| `action_rate_l2` | `−0.01` | `−0.05` | 我们重 5× |
| `joint_acc_l2`（= `dof_acc_l2`）| `−2.5e-7`，全关节 | `−2e-7`，**hip+knee** | |
| `self_collisions` | `−0.1` | **无**（PhysX 不支持多对多过滤）| |
| `feet_lateral_width` / `feet_crossing` | `−10` / `−20` | `−1` / `−2`（T1k 才开 1/10）| |
| `feet_slide`（= `foot_slip`）| `−0.25` | `−0.1` | |

---

## 7. 归类：哪些是**设计选择**，哪些是**历史漂移**

| 类别 | 项 | 处置 |
|---|---|---|
| **设计选择（有意，保留）** | 浅网（配缺失的正则化）、5-clip 参考集、7 项额外奖励、AMP 状态加 anchor 重力、`std=0.5` 的速度/角速度形状（与 OmniLoco/Vault 教师同族）| 记录理由即可 |
| **⚠️ 历史漂移（应对齐）** | **`coef = 4.0`**（T1a 白名单引入，⇒ 风格 4×）· `action_rate_l2 −0.05` · `dof_pos_limits −1/仅踝/线性` · `track_ang_vel_z_exp 1.0` · `track_base_height std 0.2` · `track_default_pose_on_stand` 的 V23 全关节表 · 白化缺失（+`min_normalized_std`）| 逐项单变量实验 |

---

## 8. 由此得到的**实验优先级**（与本清单一致）

1. **`coef` 4.0 → 1.0**（把风格强度精确对齐 mjlab；**单变量**、零结构改动）⭐ 最高性价比
2. **补白化 + `min_normalized_std`**（然后才谈"是否加第三层"）—— 需冷启动配对
3. **查参考动作集的膝角** vs default `0.669`（零训练成本；若明显更浅则 AMP 在推浅膝）
4. `dof_pos_limits` → `−10` 全关节 margin 版（治踝 roll 卡限位）
5. **加 motion reset**（`reset_from_motion` 的等价物）—— 结构性，但可能根治"不曲膝/不自然姿态"
