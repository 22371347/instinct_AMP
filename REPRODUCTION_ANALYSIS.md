# Light-Loco-Parkour 在 Unitree G1 上的复现可行性分析

> 分析日期：2026-09-19
> 论文：Light-Loco-Parkour: Versatile Perceptive Whole-Body Locomotion via Multi-Skill Distillation
> （arXiv:2608.02653，Light Origins Labs，2026-08-01）
> 社区复现：https://github.com/lucidrains/light-loco-parkour
> 本地代码快照：`llp_reference/`（clone 自 main，最新 commit `963a6ec`）

---

## ⭐ 0. 推荐方案（2026-09-19 更新，优先阅读）

### 0.1 结论：换项目。直接用 Project Instinct 的 `Instinct-Parkour-Target-Amp-G1-v0`

用户真实目标已明确为「**复现一个简易越障技能的跑酷策略**」。据此重新调研后，**推荐放弃以 lucidrains 仓库为基础**，改用：

| 用途 | 仓库 | 说明 |
|---|---|---|
| 环境侧（Isaac Lab / Isaac Sim） | [project-instinct/InstinctLab](https://github.com/project-instinct/InstinctLab) | 822 stars / 113 forks / 4 contributors / 有 tag，**正是 LLP 论文参考文献 [54] 的作者团队** |
| 环境侧（MuJoCo Warp，**云上更轻**） | [project-instinct/InstinctMJ](https://github.com/project-instinct/InstinctMJ) | 251 stars，mjlab-native port |
| RL 侧 | [project-instinct/instinct_rl](https://github.com/project-instinct/instinct_rl) | 180 stars；PPO / **AMP (WASABI)** / **DAgger 蒸馏 (TPPO)** / VAE 蒸馏 / **MoE Actor-Critic** |
| 真机部署 | [project-instinct/instinct_onboard](https://github.com/project-instinct/instinct_onboard) | ONNX → G1 实机 |

**为什么这是决定性的发现：**

1. **目标完全吻合。** `Instinct-Parkour-Target-Amp-G1-v0` 就是 G1 上的越障跑酷任务，已内置地形、深度相机、足底射线、AMP、课程学习、域随机化、ONNX 导出与上机部署链路。
2. **来源权威。** Project Instinct 由 **ZiwenZhuang**（Hang Zhao 组，清华 IIIS + 上海期智研究院）主导，是 *Robot Parkour Learning*、***Deep Whole-Body Parkour***（arXiv:2601.07701，**正是 LLP 论文的参考文献 [54]**）、*Hiking in the Wild*（arXiv:2601.07718）的作者团队。**这就是 LLP 所依赖的同源技术栈，甚至比 LLP 更早、更开源、更成熟。**
3. **工程量下降一个数量级。** 原报告 §5 估算「完整复现 3–5 人 × 6–12 个月」；改走此路线后，**单人 1–3 周即可跑通越障策略**（详见 §7）。
4. **有官方预训练权重**（Google Drive），可以**先验证再训练**，避免盲目投入。

### 0.2 三个必须知道的代价

| # | 事项 | 影响 |
|---|---|---|
| 1 | **License 是 CC BY-NC 4.0（非商用）** | InstinctLab / InstinctMJ 均如此。商用需另寻方案（`instinct_rl` 是 Modified MIT，可商用但需声明）。**务必先确认你的项目性质** |
| 2 | **跑酷任务依赖 AMASS 数据集** | 需要 `parkour_motion_without_run.yaml` 过滤后的重定向 `.npz` 动作文件；AMASS 需向 MPI 申请授权。**这是唯一的硬数据依赖**，但预训练权重可让你先绕过它 |
| 3 | **该任务不含「手撑跳马/攀爬」类全身技能** | 地形是箱阵(≤0.45 m 高)、楼梯(≤0.45 m)、沟壑(0.1–0.7 m)、斜坡——属**感知式接触越障**，不是 LLP 的 climb-and-step / speed-vault。若后续要全身技能，入口是 `Instinct-Perceptive-Shadowing-G1-v0` / `Instinct-Perceptive-HOI-Shadowing-G1-v0` |

> 详细证据与配置解析见文末 **§7**。

---

## 0. 结论摘要（TL;DR）

> ℹ️ 本节是对 **lucidrains 社区仓库**的原始审计结论，仍然有效，但**推荐路线请以上方 §0 为准**。

1. **社区复现的进展远低于"可复现项目"的门槛。** 它目前是一个 ~1600 行的**纯算法/网络层实现**（PyTorch Module + PPO/蒸馏损失函数），**没有任何仿真环境、地形生成、深度相机、动作数据管道和训练循环**。整个仓库不能训练出任何机器人策略；唯一能跑通的验证是 gymnasium 倒立摆上的 PPO sanity check。

2. **论文的实验平台是 Lightbot 0（自研 90 cm / 21 DoF 人形），不是 Unitree G1。** 所以即使在论文作者开源的前提下，G1 上也存在"跨本体迁移"的工作量：动作空间维度、障碍物尺度、相机安装位置、执行器包络都需要重新标定。这一点需要提前纳入预期。

3. **但"在 G1 上复现同类能力"本身是高可行性的**，因为论文所依赖的每一个关键组件在开源生态里都有成熟对应物。**正确的做法不是"补全 lucidrains 的仓库"，而是换用已开源的完整流水线**——经进一步调研，**首选是 Project Instinct 的 `Instinct-Parkour-Target-Amp-G1-v0`**（见上方 §0 与文末 §7），它比手工搭建或补全本仓库都更省时。

4. **如果按本文原始路线（自建流水线）走**，这是多人月级别的工程。云端单次完整流水线约 400–600 GPU·h，含调试预算 2000–6000 GPU·h。**改走 §7 的路线后，可降至单人 1–3 周、数十 GPU·h 量级。**

---

## 1. 论文要复现什么

### 1.1 四个模块的流水线

| 阶段 | 论文章节 | 内容 | 关键输入/输出 |
|---|---|---|---|
| ① 感知式 locomotion | §IV | 用强化学习训练速度跟踪策略；teacher 读**特权高度扫描(scan)**，critic 额外读 oracle 信息（非对称 actor-critic，§IV-B） | 地形课程（10 难度 × 32 列）、域随机化 Table II |
| ② 数据增强 | §V-A | 从**单个种子动作**（网上下载的视频 → GVHMR 提取 SMPL → GMR 重定向）出发，手工对齐障碍物；再用**迭代自增强**（Algorithm 1）让策略 rollout 自己成为下一轮参考，每轮抬高障碍 5–10 cm | 产出"动作-地形配对"参考集 |
| ③ 技能学习与泛化 | §V-B | 先做 **object-interaction mimic**（改造 BeyondMimic：跟踪全局 root 位置、接触部位位置奖励、加入障碍距离/尺寸特权观测），再 **DAgger 蒸馏**到高度扫描观测，最后**去掉专家**做 RL 泛化微调 | 每个技能一个 θ 条件化策略 πᵢ |
| ④ 过渡学习 | §V-C | **多专家 DAgger**（环境分组，每组由对应专家监督，student 无 skill label，critic 有 one-hot）+ **过渡微调**（新增 transition group，用**相位条件 AMP** 在触发位置从 locomotion prior 切到 skill prior） | 单一可部署策略 |
| ⑤ 深度蒸馏 | §VI | 高度扫描策略 → 递归深度策略。GRU + **辅助高度扫描重建损失**(Eq. 13) + 对齐 D435 的噪声/延迟模型 | 上机部署的最终策略 |

### 1.2 论文的硬件（很关键）

- **Lightbot 0**：90 cm 高、18.9 kg、**21 个驱动自由度**、腿部并联四连杆踝关节
- 电机：准直驱，**腰部/下肢 45 N·m，手臂 15 N·m**，峰值角速度 9.42 rad/s
- 传感：胸部 **RealSense D435**（下倾 30°）+ 骨盆 6 轴 IMU + 关节编码器，**无 LiDAR、无动捕、无外部状态估计**
- 算力：机载 **Jetson Orin Nano**，策略 50 Hz
- 论文明确指出：**"执行器能力显著低于 1.3 m 级平台"**，因此不能直接用全尺寸人类动捕重定向

### 1.3 论文自陈的局限（复现时同样会遇到）

- 数据管道的种子收集环节**需要人工介入**对齐视频与障碍物 —— 作者称这是技能集扩张速度的真实上限
- 过渡策略**只能耦合少量离散技能**；障碍物重叠或连续出现时性能退化
- **固定的胸部深度相机**限制了感知覆盖，上半身遮挡相机时尤其严重

---

## 2. 社区复现项目当前进展（基于代码审计）

### 2.1 项目元数据

| 项 | 值 |
|---|---|
| 作者 | lucidrains (Phil Wang)，**单作者**，无其他 contributor |
| 起止 | 2026-08-05 首次提交 → 2026-09-11 最后提交（~5 周，间歇开发） |
| commit 数 | 30 |
| 热度 | 61 stars / 8 forks / 2 watching |
| issue | 共 1 个，已关闭（仅"更新引用"） |
| release | **无** |
| 语言 | Python 100% |
| 状态标注 | README 明确写着 **`(wip)`** |
| 版本号 | `0.0.29`（`0.0.x` 表明处于早期原型） |

### 2.2 代码结构（全部内容）

```
llp_reference/
├── light_loco_parkour/
│   ├── __init__.py                    # 32 行，导出符号
│   └── light_loco_parkour.py          # 1626 行 ← 全部实现都在这一个文件
├── tests/
│   ├── test_agent.py                  # 774 行
│   ├── test_light_loco.py             # 363 行（当前跑不了，见 §2.4）
│   └── test_next_latent_prediction.py # 209 行
├── validate_pendulum.py               # 453 行，倒立摆 PPO 验证脚本
├── pyproject.toml
└── README.md / fig2.png / LICENSE
```

### 2.3 ✅ 已实现的部分（算法层，与论文 Table I / Eq.1–13 对照）

| 论文组件 | 仓库实现 | 对齐度 |
|---|---|---|
| §IV-A 策略架构：MLP 骨干 + 5 帧堆叠 | `StateEncoder`（`num_stacked_frames=5`，可选 GRU + TBPTT） | 对齐 |
| §VI 递归深度策略 | `StateEncoder(use_rnn=True)`，GRU，hiddens 可逐帧透传（已测） | 对齐 |
| §IV-B 非对称 actor/critic | `Agent(actor_state_keys=, critic_state_keys=)` 按 key 路由 | 对齐 |
| §V-C1 critic 收到分组 one-hot | `Critic(skill_groups=)` + `OneHot`（actor 不传即 label-free） | 对齐 |
| §V-B1 Eq.8 DAgger 蒸馏（特权 teacher → 可部署 student，MSE on mean action） | `DistillationWrapper`（`teacher_state_keys` / `student_state_keys`，`detach_teacher`） | 对齐 |
| §VI Eq.13 辅助重建 | `Actor(aux_decoder=)`，但解码自 MLP 输出而非 GRU 隐状态，且目标是拼接的特权状态而非高度扫描 | **近似** |
| Eq.1 线/角速度跟踪（高斯核 σ=0.25） | `reward_linear_velocity_tracking` / `reward_angular_velocity_tracking`，权重 +2.0/+2.0 | 对齐 |
| Eq.2 直立姿态 | `reward_upright_orientation`，权重 +1.0 | 对齐 |
| Eq.3 velocity-slack 奖励 | `reward_velocity_slack`（含对论文"比值定义在零指令下未定义"的修正说明），权重 +1.5 | 对齐（有改进） |
| Eq.4 illegal footstep | `reward_illegal_footstep`，权重 −1.0 | 对齐 |
| Eq.5 足端加速度惩罚（一阶低通滤波，τ=0.06 s） | `FootAccelerationPenalty`（有状态 Module），权重 0.01 | 对齐 |
| Table I 其余惩罚项 | `reward_undesired_contact`(−2.0)、`reward_joint_limit`(−10.0)、`reward_heading_error`(−1.0)、`reward_opposite_direction`(−1.0)、`reward_action_rate`(−0.1) | 对齐 |
| 分组奖励 | `RewardShapingWrapper` 支持 `((组名, 组内权重, ((fn, w), ...)), ...)` 分组；但 `DEFAULT_REWARD_FNS` 是**扁平配置 → 只产出 1 组**，论文的"跟踪组 / 惩罚组"两组需要自己配 | 需配置 |
| §V-C2 Eq.12 AMP | `Discriminator` + `MotionPrior`（`softplus` 奖励 + `gradient_penalty`，可选 modern-GAN 零中心 / WGAN-GP 一中心） | 对齐 |
| §V-C2 相位条件 AMP 切换 | `PhaseConditionalMotionPrior`（硬 one-hot / soft sigmoid 平滑交接，`prior_transition_positions`） | 对齐 |
| §IV PPO | `Agent`：clipped surrogate、entropy bonus、`calc_gae`（`assoc_scan` 加速）、`z_score` 归一化优势 | 对齐 |
| 多 critic 分组加权优势 | `calc_advantages`：逐组 z-score 后加权求和（Mysore et al. ICLR 2022） | 对齐（论文未显式写，但是合理实现） |
| 动作分布 | `Gaussian`、`Beta`（mean-concentration 重参数化，来自 Zhao et al. 2025 自动驾驶论文） | **论文未提及 Beta** |

**积极信号**：奖励项与权重和论文 Table I **逐条对得上**（含 Eq.2 的 `exp(-2‖g‖²)+0.1exp(-‖g‖)` 混合核、Eq.5 的 α=exp(-Δt/τ) 滤波、Eq.4 的射线深度阈值 δ=0.1），说明作者认真读过论文，不是一个敷衍的骨架。

### 2.4 ❌ 未实现的部分（占工程量的大头）

| 缺失项 | 论文对应 | 影响 |
|---|---|---|
| **任何仿真环境** | §III、§IV-C | 致命。无 IsaacLab / IsaacGym / MuJoCo / Gym 机器人环境，无 G1 或 Lightbot 资产 |
| **地形生成器** | §IV-C3 | 9 类地形（斜坡/Perlin/平地/轨道/立柱/踏石/箱阵/楼梯/木板桥）× 10 难度 × 32 列，及地形课程升降级逻辑 |
| **终止条件** | §IV-C2 | 超时/越界(2 m 缓冲)/关节速度守卫(50 rad/s)/躯干接触(>1 N)/过度加速度(40 m/s²)/摔倒(63° 随机 0.01 概率)，以及 10% env 的"免疫"标志 |
| **域随机化** | Table II | 摩擦/恢复系数/连杆质量/躯干负载/质心偏移/armature/库仑+粘滞摩擦/刚度-阻尼耦合/相机外参/推力扰动/初始化随机化 |
| **高度扫描与足端射线** | §IV-C1、Eq.4 | 局部高度扫描（teacher 观测）、**足底小型向下扫描器**（illegal footstep 罚项的数据来源）—— 都没有 |
| **深度相机仿真** | §VI | 胸部相机、D435 噪声模型（σ=0.005+0.02d 的距离相关高斯 + ±5% 尺度抖动 + 分块 dropout + 30–60 ms 延迟 @27–33 Hz） |
| **动作数据管道** | §V-A | GVHMR（视频→SMPL）、GMR（重定向）、障碍物手工对齐工具、参考缓冲/播放、RSI |
| **Object-interaction mimic** | §V-A1 | 全局 root 位置跟踪、接触部位全局位置奖励、障碍距离/尺寸特权观测、goal/mimic 奖励、参考偏离终止（Table III） |
| **迭代自增强循环** | Algorithm 1 | 每轮 15k 预算的 mimic 训练、K 路并行 rollout、按 Score 选优、抬升 5–10 cm 并存储配对 |
| **多专家 DAgger 编排** | §V-C1 | `DistillationWrapper` 只支持**单个** teacher；缺少"环境分组 + 按组匹配专家 + 按组路由"的机制 |
| **训练主循环** | 全局 | 无向量化环境步进、rollout buffer、PPO trainer、checkpoint、日志、LR 调度、观测归一化统计。**只有 `validate_pendulum.py` 里的单环境手写 PPO** |
| **部署栈** | §VII | 无 ONNX/TorchScript 导出、无关节顺序映射、无 50 Hz 实时循环、无安全限幅 |
| **模型默认值** | — | `num_actions=21` 直接编码了 **Lightbot 0 的 21 DoF** 假设 |

### 2.5 ⚠️ 需要注意的偏差与问题

1. **仓库不是论文的 1:1 复现，而是"受论文启发的自研工具箱"。** 作者额外塞进了论文里根本没有的东西：
   - **NLP / SPR 下一隐状态预测**（Schwarzer 2021、Teoh 2026）——两篇完全无关的论文，还专门写了 `test_next_latent_prediction.py`
   - **Beta mean-concentration 动作分布**（Zhao et al. 2025，自动驾驶）
   - **modern GAN baseline 梯度惩罚**（Huang et al. 2025）
   - **多 critic actor learning**（Mysore et al. 2022，BibTeX 里显式引用，但论文正文没提）
   - 还专门做了**倒立摆 / 车杆**上的动作分布收敛性验证
   → 这些和"复现 LLP"是两回事，会分散精力。**直接复用时要主动裁剪。**
2. **`tests/test_light_loco.py` 当前无法运行。** 它 `from validate_transition import (...)`，但仓库里**不存在** `validate_transition.py`（也已从 git 历史中确认从未存在）。这个被注释为 "full pipeline e2e - exercises all modules" 的测试是坏的。**只有 `test_agent.py` 和 `test_next_latent_prediction.py` 大概能独立运行。**
3. **测试全部针对 mock。** `test_light_loco.py` 用的是 `MockTransitionEnv` 和"采样出的假 transitions"——即所有验证都在形状/数值层面，没有一次真实物理仿真。
4. **依赖为作者自研小包**：`env-ssl-wrapper`、`mean-conc-beta`、`x-mlps-pytorch`、`hl-gauss-pytorch`、`assoc-scan`、`torch-einops-utils`。都是可 pip 安装的小工具库（风险低），但意味着**任何一处想改动都要跨仓库**。
5. **`assoc_scan`/`einx`/`beartype` 在 ILP 上运行** —— 这些装饰器与路径模式匹配会带来额外开销，在 4096 并行环境下未必是性能最优选择。

### 2.6 进展判定

```
论文全流水线  ================================================[100%]
lucidrains    ==[   ~15%   ]...................................[   ]
                  ↑ 只完成"网络模块 + 损失函数"这一层
```

**一句话**：这是"把论文里的公式翻译成 PyTorch Module"，而不是"复现论文"。它有价值的部分是**奖励函数定义、分组 critic、相位条件 AMP 的接口设计、蒸馏 wrapper 的签名设计**——这些可以当参考实现抄，但**不能当作项目起点**。

---

## 3. 在 G1 上复现的可行性分析

### 3.1 逐维度评估

| 维度 | 可行性 | 说明 |
|---|---|---|
| 算法层 | 🟢 高 | 论文方法全部基于已发表的公开工作（PPO、DAgger、AMP、BeyondMimic、RSI），无私有魔法 |
| 仿真环境 | 🟢 高 | IsaacLab + Unitree 官方 G1 资产（`unitree_rl_gym` / `unitree_rl_lab`）现成可用 |
| 感知式 locomotion | 🟡 中 | 有直接对标的开源 G1 工作（**BeamDojo** 就是 G1 + 稀疏落脚点），可复用大量地形/奖励/课程代码 |
| 深度蒸馏 | 🟡 中 | 高度扫描→深度的 teacher-student 是 Miki/BeamDojo 系列的标准套路；难点在于**渲染开销**和相机噪声建模 |
| 技能（climb/vault） | 🟡 中 | BeyondMimic（G1，开源）提供了 motion tracking 骨架；但"物体交互 + 迭代自增强"这套需要自研 |
| 动作数据管道 | 🔴 低-中 | GVHMR、GMR 均开源，但**视频→SMPL→重定向→障碍物手工对齐**这一环论文自陈需人工介入，是全流程最脏的一段 |
| 过渡训练 | 🟡 中 | "多专家蒸馏 + RL 微调"有完整模板：**Parkour in the Wild**（Rudin et al. 2025），论文 §V-C 基本就是它的 humanoid 版 |
| 上机部署 sim2real | 🟢 高（G1 优势） | G1 社区部署栈非常成熟，比自研 Lightbot 容易得多 |
| **总的复现难度** | **中-高** | 单点都可做，**串联起来的工程量巨大** |

### 3.2 G1 vs Lightbot 0 —— 跨本体差异带来的具体影响

| 项目 | Lightbot 0 | Unitree G1 | 影响 |
|---|---|---|---|
| 身高 | 90 cm | ~132 cm | **障碍物尺度必须重新标定** |
| 质量 | 18.9 kg | ~35 kg | 动力学时间常数变化，PD 增益需重调 |
| 自由度 | 21 | 23（标准）/ 29（EDU，带 3 自由度腕） | **动作维度、观测维度、`num_actions` 全部要改**；多出的腕关节会引入上机时的额外风险 |
| 下肢力矩 | 45 N·m | 显著更高（请以官方 URDF/spec 为准） | **可行性反而提高** |
| 踝关节 | 并联四连杆 | 串联 | 运动学建模不同，接触力分布不同 |
| 相机 | 胸部 D435，下倾 30° | G1 无标配前置深度相机 | **需要自行加装并做外参标定**，且安装位置决定了遮挡特性 |

**关键推论：**

1. **障碍物高度不能照搬。** 论文的 climb-and-step 做到 75 cm = **0.83 H**（H 为站立身高）。在 G1 上按 0.83H 折算 ≈ **1.1 m**，无论是腿部力矩还是手部支撑都远超能力边界。**必须重新定义 G1 的归一化高度基准**（建议起步取 0.30–0.45 H ≈ 0.4–0.6 m 的 climb/vault），并在论文的归一化指标框架下重新报告。
2. **G1 反而更容易做"人类动捕直采"。** 论文被迫走"视频→SMPL→迭代自增强"是因为**90 cm 平台无法承载全尺寸人类动作**。G1 是近人类尺度，**AMASS / KungfuBot 等现成人类动捕可以直接重定向**。这意味着：
   - ✅ 可以跳过部分"迭代自增强"的复杂度（至少不必从零造参考）
   - ⚠️ 但也就失去了论文第二项贡献（"从稀疏种子扩张"）的复现价值 —— 如果目标是"论文级复现"，这一环仍需保留
3. **深度相机是新增硬件工作量。** G1 无标配 D435，需要自行设计安装件，且 D435 的噪声/延迟标定（论文 §VI 给了具体模型参数）需要实机测量。

### 3.3 云平台训练：算力与预算估算

论文给出的训练预算（§VII-C2）：

- 消融对比：**每个变体 15,000 iteration**，统一预算
- 深度策略：**蒸馏 14,000 iteration + 微调 1,000 iteration**
- 每个表格里的成功率：**500 次随机化试验**；过渡实验 100 次

按 IsaacLab 上 G1 + 4096 envs 的经验推算（单次迭代约 2–5 s，取决于是否开相机渲染）：

| 阶段 | 估算 GPU·h（A100/4090 级） |
|---|---|
| 感知 locomotion teacher（15k iters + 课程） | 10–20 |
| 技能 teacher × 4（含 Algorithm 1 迭代自增强，每个技能 N 轮 mimic 训练） | 240–400 |
| 高度扫描蒸馏（DAgger + PPO） | 15–25 |
| 多专家 DAgger 蒸馏 | 15–25 |
| 过渡微调（含 AMP 判别器） | 20–40 |
| 深度蒸馏（14k + 1k，**开相机渲染，开销显著更高**） | 40–80 |
| **单次干净跑通合计** | **≈ 350–600 GPU·h** |
| **含调试/消融/重跑（经验系数 5–10×）** | **≈ 2000–6000 GPU·h** |

**折算成本**（参考价）：4090 约 $0.3–0.5/h → **$600–3000**；A100 约 $1.5–2.5/h → **$3000–15000**。

**云平台工程注意事项：**

- Isaac Sim 需要 **Vulkan/OpenGL**，无头模式下要 `--headless --enable_cameras`；部分云厂商的容器镜像驱动不满足，**开工前务必先验证能起来**。
- **IsaacGym 已 EOL**（不再维护、对 Py3.10+ 支持差），**不要走 IsaacGym 路线**，直接上 **IsaacLab**。
- **深度渲染是最大的吞吐瓶颈**。深度蒸馏阶段的迭代速度可能只有纯 locomotion 的 1/3–1/5，预算里要留足。
- 若想压低成本，**阶段 0–1 可以先用 MuJoCo / MJX-Brax 做**（有 G1 的 MJCF，且支持 depth rendering，CPU/GPU 都能跑、更便宜），把 IsaacLab 留给需要高质量相机与大规模并行的阶段 3–4。

### 3.4 主要风险（按严重度排序）

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| 1 | **工程量被严重低估**：需要从零搭建仿真环境 + 地形 + 奖励 + 课程 + DR + 训练循环，这是 2–4 人月的工作 | 🔴 | 站在 `unitree_rl_gym` / `BeamDojo` 已有代码上改，不要从 lucidrains 仓库起步 |
| 2 | **动作数据管道的人工环节**：论文自陈 "human-in-the-loop 对齐视频与障碍物" 是技能扩展的真实上限 | 🔴 | 优先用现成人类动捕（AMASS）替代视频管线做 Phase 2 验证，视频管线延后 |
| 3 | **技能收敛性**：论文 §VII-C2 明确说去掉"专家蒸馏"后 teacher **完全不收敛**（"✗ failure to converge"），纯奖励学习在 humanoid 全身接触任务上探索空间过大 | 🟠 | 必须完整实现 object-interaction mimic + RSI，不能简化 |
| 4 | **深度蒸馏掉点**：论文自身报告 depth student 在 climbing 75 cm 时成功率"sharp drop" | 🟠 | 预留最后的 RL 微调阶段；G1 相机位置需要实测优化 |
| 5 | **跨本体超参全失效**：奖励权重、PD 增益、地形难度、DR 范围均需在 G1 上重调 | 🟠 | 按阶段自底向上逐层验证，每层单独出成功率曲线 |
| 6 | 依赖作者自研小包（`env-ssl-wrapper` 等）造成改动跨仓库 | 🟢 | 直接把需要的几个函数内联进自己代码 |
| 7 | `test_light_loco.py` 已损坏 / 无 CI / 无 release | 🟢 | 只用它做"公式对照"，不要依赖其正确性 |

---

## 4. 推荐技术路线（分阶段）

### 4.1 开源资产清单（直接可用的骨架）

| 用途 | 推荐仓库 | 说明 |
|---|---|---|
| 仿真环境 + G1 资产 | **IsaacLab** + **`unitree_rl_gym` / `unitree_rl_lab`** | 官方 G1 URDF/USD、PD 控制、部署示例 |
| 感知式 locomotion（G1 稀疏落脚点） | **BeamDojo**（arXiv:2502.10363） | G1 + 踏石/平衡木，与论文 §IV 目标高度重合 |
| **多专家蒸馏 + RL 微调**（论文 §V-C 的模板） | **Parkour in the Wild**（arXiv:2505.11164） | 论文自己引用的方法论来源，有开源代码 |
| 混合模仿（视频→技能） | **HIL**（arXiv:2505.12619） | PointNet latent + 参考跟踪 |
| **motion tracking 骨架（改造 BeyondMimic）** | **BeyondMimic**（arXiv:2508.08241） | G1 全身跟踪，论文 §V-A1 的起点 |
| 动作重定向 | **GMR**（arXiv:2510.02252）、**OmniRetarget** | 论文 §VII-C1 的对比基线，也是可用工具 |
| 视频→SMPL | **GVHMR**（SIGGRAPH Asia 2024） | 论文的种子动作来源 |
| AMP | **HIL** / **ASAP** | 判别器 + 相位切换可参考 |
| 现成动捕数据（绕过视频管线） | **AMASS**、**KungfuBot** 数据 | G1 近人类尺度，可直接重定向 |

### 4.2 分阶段计划

**Phase 0 — 基础设施（1–2 周）**
- 打通 IsaacLab + G1 headless 训练，确认云平台能跑（含 `--enable_cameras`）
- 复现 baseline 速度跟踪 locomotion（平地），拿到可用的 PD 增益与训练超参

**Phase 1 — 感知式 locomotion（2–4 周）**
- 移植/重建地形生成器（先做斜坡/Perlin/楼梯/踏石四类，不必一次做九类）
- 实现奖励组（直接抄 lucidrains 的 `DEFAULT_REWARD_FNS`，权重照 Table I）+ 足端射线 illegal-footstep
- 高度扫描 teacher（frame stacking）→ 深度 student（GRU + 辅助重建）
- ✅ 里程碑：G1 在踏石/平衡木仿真上的成功率曲线

**Phase 2 — 单个技能（3–6 周）**
- 选 **climb-and-step** 作为第一个技能（接触相位最清晰、论文数据最全）
- 用 AMASS/现成动捕替代视频管线，制作第一个"动作-障碍物配对"种子
- 实现 object-interaction mimic（全局 root 位置奖励 + 接触部位奖励 + 障碍特权观测 + RSI）
- 实现迭代自增强循环（Algorithm 1），**重新标定 G1 的障碍高度区间**（建议从 0.35 H ≈ 0.45 m 起）
- ✅ 里程碑：单障碍高度上的成功率 > 60%

**Phase 3 — 多技能 + 过渡（3–6 周）**
- 增加 2–3 个技能（speed-vault / reverse-vault / step-down）
- 实现多专家 DAgger（**分组环境 + 按组匹配专家**，这部分需要自己在 `DistillationWrapper` 上扩展）
- 过渡微调 + 相位条件 AMP（`PhaseConditionalMotionPrior` 可以直接参考其接口设计）
- ✅ 里程碑：连续课程（locomotion → obstacle → locomotion）端到端成功率，对标论文的 98%

**Phase 4 — 深度蒸馏 + sim2real（3–6 周）**
- D435 噪声/延迟模型（论文 §VI 给了完整参数：σ=0.005+0.02d、±5% 抖动、分块 dropout、30–60 ms 延迟、27–33 Hz）
- G1 实机加装前置深度相机 + 外参标定
- 导出 + 50 Hz 部署
- ✅ 里程碑：室内零样本穿越

### 4.3 关于"要不要用 lucidrains 的仓库"

| 建议 | 内容 |
|---|---|
| ✅ **拿来抄** | `reward_*` 十个函数的实现与权重、`RewardHyperParams` 默认值、`FootAccelerationPenalty` 的滤波实现、`calc_advantages` 的分组 z-score 加权、`PhaseConditionalMotionPrior` 的 phase 解析/平滑交接接口、`DistillationWrapper` 的 actor/critic key 路由设计 |
| ❌ **不要拿来用** | `Beta` 动作分布、NLP/SPR 头、modern-GAN 梯度惩罚、`validate_pendulum.py` —— **论文里没有，属于作者的额外探索** |
| ⚠️ **需要自己重写** | 训练循环、多专家路由、环境/地形/DR/终止、深度噪声模型、部署导出 |

---

## 5. 工作量与产出预期

| 方案 | 范围 | 人力 | 周期 | GPU 预算 |
|---|---|---|---|---|
| **A. 最小可行** | 只在 G1 上做感知式 locomotion（楼梯/踏石/平衡木），**不做全身 parkour 技能** | 1 人 | 1.5–2 个月 | 200–500 GPU·h |
| **B. 部分复现**（推荐） | locomotion + 1 个技能（climb-and-step）+ 深度蒸馏 + 上机 | 1–2 人 | 3–4 个月 | 1000–2500 GPU·h |
| **C. 完整复现** | locomotion + 3–4 技能 + 过渡学习 + 深度蒸馏 + 全部消融 | 3–5 人 | 6–12 个月 | 3000–6000 GPU·h |
| **D. 论文级复现** | C + 论文 Table V 的全部对比基线（PHP/MGMT/BeamDojo/CReF）| 5 人以上团队 | 12 个月+ | 10000+ GPU·h |

**重要现实检查**：论文作者是 Light Origins Labs 的 11 人团队，且他们**自研了硬件**（Lightbot 0）。硬件自研意味着他们对执行器模型、接触特性、相机标定有完全控制权与全部底层数据。在通用 G1 上，**Phase 4 的 sim2real 环节反而可能比他们更顺利（G1 生态成熟），但 Phase 2 的数据管道环节会更难（他们能从硬件侧获得先验）**。

---

## 6. 行动建议

1. **立刻做**：在云平台上验证 IsaacLab + G1 能否 headless 起来（这是所有后续工作的前提，且最容易在环境问题上翻车）。
2. **立刻做**：clone `unitree_rl_gym` 和 `BeamDojo`，跑通一个 G1 平地 locomotion baseline。这会给你一个**真实的工程量感受**，远胜于继续读 lucidrains 的代码。
3. **复现目标要收窄**：先明确"我要的是论文的**哪个贡献**"——是"感知式全身 locomotion 系统"（那 Phase 0–1 + 一个技能就够），还是"从稀疏种子扩张技能"（那数据管道是核心，G1 上因为能直采人类动捕反而削弱了研究意义），还是"纯奖励驱动的技能过渡"（那 Phase 3 是核心）。
4. **不要以 lucidrains 仓库为项目基础**，但**把它当参考实现保留**（已在 `llp_reference/`），尤其是奖励函数与相位 AMP 的接口设计。
5. **障碍高度重新标定**是第一个必须做的实验性决定，不要沿用论文的 0.83 H。

---

## 7. 【重大修正】推荐路线：Project Instinct 的 `Instinct-Parkour-Target-Amp-G1-v0`

### 7.1 Project Instinct 是什么

**由清华 IIIS + 上海期智研究院 ZiwenZhuang（Hang Zhao 组）主导的开源人形全身控制生态**，官网 https://project-instinct.github.io/ 。

其研究线包括：

| 论文 | 出处 | 与本次目标的关系 |
|---|---|---|
| Robot Parkour Learning | CoRL 2023（Zhuang et al.） | 四足跑酷的开山工作 |
| Embrace Collisions | CoRL 2025 | 全身接触型 shadowing |
| **Deep Whole-Body Parkour** | arXiv:2601.07701 | **就是 LLP 论文的参考文献 [54]** |
| **Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids** | arXiv:2601.07718 | **感知式跑酷框架，与本目标最接近** |

**生态组成：**

| 仓库 | 角色 | 规模 | License |
|---|---|---|---|
| `InstinctLab` | 环境侧，基于 **Isaac Lab / Isaac Sim 5.1.0** | 822★ / 113 fork / 4 贡献者 / 3 tag | **CC BY-NC 4.0（非商用）** |
| `InstinctMJ` | 环境侧，基于 **mjlab + MuJoCo Warp** | 251★ / 30 fork / 2 贡献者 | **CC BY-NC 4.0（非商用）** |
| `instinct_rl` | RL 侧（从 rsl_rl 大幅改造） | 180★ / 31 fork | Modified MIT（可商用，需声明） |
| `instinct_onboard` | G1 真机部署 | — | — |
| `robot_trajectory_editor` / `robot-motion-editor` | 动作/轨迹编辑工具 | — | — |

`instinct_rl` 已内置的算法与网络模块，**几乎覆盖了 LLP 论文所需的全部算法件**：

- **算法**：PPO、State Estimator、**AMP (WASABI)**（BCE / Wasserstein / MSE 判别器 + 梯度惩罚）、**Distillation & DAgger (TPPO)**、**VAE Distillation**
- **网络**：Actor-Critic、**MoE Actor-Critic**、VAE Actor-Critic、**Encoder 系列**（图像/多模态）、VQ-VAE、State Estimator
- 观测格式抽象（`obs_format` / `obs_segment` / `obs_pack`），原生支持**非对称 actor/critic 与多段观测**

### 7.2 ✅ 越障能力确认（回答你的第 3 个问题）

**是的，`Instinct-Parkour-Target-Amp-G1-v0` 确实包含越障技能。** 我直接读了任务配置源码，证据如下。

**越障地形**（`tasks/parkour/config/parkour_env_cfg.py` 的 `ROUGH_TERRAINS_CFG`，`num_rows=10` 难度 × `num_cols=20` 列，`curriculum=True`）：

| 地形 key | 生成器 | 关键参数 | 越障含义 |
|---|---|---|---|
| `boxes` | `PerlinDiscreteObstaclesTerrainCfg` | `num_obstacles=20`，宽 `0.8–1.5 m`，**高 `0.05–0.45 m`** | ⭐ **离散障碍箱阵越障**（最贴近你要的"越障"） |
| `mesh_boxes` | `PerlinMeshRandomMultiBoxTerrainCfg` | `box_height_mean=[0.1, 0.4] m`，长/宽 ≈0.4 m，`no_perlin_at_obstacle=True` | ⭐ **随机箱体越障** |
| `square_gaps` | `PerlinSquareGapTerrainCfg` | `gap_distance_range=(0.1, 0.7) m`，`gap_depth=(0.4, 0.6) m` | ⭐ **跨越沟壑/间隙** |
| `pyramid_stairs` | `PerlinPyramidStairsTerrainCfg` | `step_height 0.05–0.23 m`，`step_width=0.3` | 上楼梯 |
| `pyramid_stairs_high` | 同上 | `step_height 0.05–**0.45** m`，`step_width=1.5` | ⭐ **高台阶上楼梯** |
| `pyramid_stairs_inv` / `_inv_high` | `PerlinInvertedPyramidStairsTerrainCfg` | `0.05–0.23` / `0.05–**0.45** m` | ⭐ **下楼梯**（含高台阶） |
| `hf_pyramid_slope_inv` | `PerlinInvertedPyramidSlopedTerrainCfg` | `slope_range=(0.0, 0.7)` | 反向斜坡 |
| `perlin_rough` / `_stand` | `PerlinPlaneTerrainCfg` | `noise_scale 0–0.1`，`noise_frequency=20` | 崎岖地面 |

InstinctLab 侧另有 mesh 地形：`floating_box_terrain`、`random_multi_box_terrain`、`motion_matched_terrain`。

**尺度参照**：G1 站立高约 1.32 m，`0.45 m` 障碍 ≈ **0.34 H**，`0.45 m` 台阶 ≈ 0.34 H。这是**货真价实的越障**（对比 LLP 论文的 climb-and-step 做到 0.83 H，但那是靠**手臂支撑**实现的；本任务是纯下肢+感知的接触越障）。

**越障相关的关键机制（源码实证）：**

- **虚拟障碍物注册**：`virtual_obstacle_source="mesh"` + `register_virtual_obstacle_to_sensor`（`mode="startup"`，绑定到 `leg_volume_points` 传感器），配合 `GreedyconcatEdgeCylinderCfg` 把障碍棱边提取成圆柱体 —— **这相当于 LLP 的"特权障碍信息"**。
- **腿部体积点穿模惩罚**：`VolumePointsCfg(name="leg_volume_points")`，在踝关节上布置 `10×5×2` 网格点；奖励项 `volume_points_penetration` 权重 **−4.0**。这是防止腿撞/穿障的核心项。
- **足底高度扫描器**：`left_height_scanner` / `right_height_scanner`，`RayCastSensorCfg` 挂在 `*_ankle_roll_link`，`GridPatternCfg(resolution=0.12, size=(0.12, 0.0))` —— **这正是 LLP Eq.4 "illegal footstep"/足底射线的等价物**，对应奖励 `feet_at_plane`（权重 −0.1，带 `height_offset=0.058` 鞋底补偿）。
- **课程学习**：`curriculum = {"terrain_levels": CurriculumTermCfg(func=tracking_exp_vel, lin_vel_threshold=(0.3, 0.6))}` —— 按速度跟踪表现升降地形难度，与 LLP §IV-C3 同思路。

### 7.3 传感器 / 观测 / 算法栈

**传感器：**

| 传感器 | 配置 | 作用 |
|---|---|---|
| `camera` | `NoisyGroupedRayCasterCameraCfg`，挂 `torso_link`，**64×36**，`fovx=89.51°`，`fovy=58.29°`，`data_types=["distance_to_image_plane"]`，`update_period=0.02`，`depth_clipping_behavior="max"` | ⭐ **头部深度相机** |
| 深度噪声管道 | `CropAndResizeCfg(crop_region=(18,0,16,16))` → `GaussianBlurNoiseCfg(kernel_size=3, sigma=1)` → `DepthNormalizationCfg(depth_range=(0,2.5))`；另有 `DepthArtifactNoiseCfg` / `RandomGaussianNoiseCfg` / `RangeBasedGaussianNoiseCfg` 可用 | ⭐ 对应 LLP §VI 的 D435 噪声建模 |
| `left/right_height_scanner` | 踝部射线，分辨率 0.12 m | 足底地形感知 |
| `leg_volume_points` | 踝部 3D 点网格 | 穿模检测 |
| `contact_forces` / `torso_contact_forces` / `undesired_contact_forces` | 接触力 | 触点/摔倒判定 |

**观测组**（4 组，`concatenate_terms=False` 便于分段喂给不同网络头）：

| 组 | 内容 | 说明 |
|---|---|---|
| `policy` | `base_ang_vel`(噪声±0.2)、`projected_gravity`(±0.05)、`velocity_commands`、`joint_pos`(±0.01)、`joint_vel`(±0.5, scale 0.05)、`actions`，各 `history_length=8`；**`depth_image`** | 可部署观测 |
| `critic` | 同上 + **`base_lin_vel`**（特权） | ⭐ **非对称 actor-critic**（对应 LLP §IV-B） |
| `amp_policy` / `amp_reference` | `projected_gravity`、`joint_pos_rel`、`joint_vel`、`base_lin_vel`、`base_ang_vel`，`history_length=10` | ⭐ **AMP 判别器的观测对** |
| `amp_reference` | 用 `*_reference_as_state` 系列函数，从 `motion_reference` 实体取状态 | 参考动作侧 |

**`depth_image` 的延迟建模**（很关键，说明这个任务已经是"可部署级"设计）：

```python
func=mdp.delayed_visualizable_image,
data_type="distance_to_image_plane_noised_history",
history_skip_frames=5,          # 隔帧采样 → 模拟低帧率
num_output_frames=8,
delayed_frame_ranges=(0, 1),    # 30–60ms 级延迟
```

**算法与网络**（`config/g1/agents/instinct_rl_amp_cfg.py`）：

```python
policy   = MoEPolicyCfg()    # MoE Actor-Critic, num_moe_experts=4,
                             # depth_encoder = Conv2d output_size=128, channels=[4], kernel=3
                             # actor/critic hidden = [256,128,64], init_noise_std=1.0
algorithm = AmpAlgoCfg()     # WASABI AMP + PPO
```

**训练规模**（`rl_cfg.py`）：

| 项 | 值 |
|---|---|
| `num_envs` | 2048（InstinctMJ）/ 4096（InstinctLab） |
| `max_iterations` | **30000** |
| `num_steps_per_env` | 24 |
| `episode_length_s` | 20.0 |
| `decimation` / `sim.dt` | 4 / 0.005 → **策略 50 Hz** ✅ 与 LLP 一致 |
| `save_interval` | 5000 |

**执行器模型**：`beyondmimic_g1_29dof_delayed_actuators` + `beyondmimic_action_scale` —— **直接复用了 BeyondMimic 的"延迟执行器"建模**（对 sim2real 至关重要，也是 LLP §V-A1 所引用工作的产物）。

**动作参考**：`AmassMotionCfg`，`filtered_motion_selection_filepath="parkour_motion_without_run.yaml"`，`motion_start_from_middle_range=[0.0, 0.9]`（≈ LLP 的 RSI 参考状态初始化），`frame_interval_s=0.02`，`num_frames=10`，`motion_interpolate_bilinear`。

**终止条件**：`time_out`、`terrain_out_bound`（`distance_buffer=2.0`，与 LLP 一致）、`base_contact`（`torso_link` 力 >1 N，与 LLP 一致）、`bad_orientation`（`limit_angle=1.0 rad ≈ 57°`，LLP 是 63° + 随机化）、`root_height`（`minimum_height=0.5`）、`dataset_exhausted`。

**部署链路（已打通）**：`--export-onnx` → `instinct_onboard`（含 G1 关节顺序映射、`joint_signs`/`joint_limits`/`torque_limits` 生成脚本、仿真—真机关节序对齐说明）。官方还提供 **Parkour 预训练权重**（Google Drive）。

### 7.4 选型：InstinctLab 还是 InstinctMJ（云平台视角）

| | **InstinctLab** | **InstinctMJ** |
|---|---|---|
| 仿真后端 | Isaac Lab + **Isaac Sim 5.1.0** | **mjlab 1.6.0 + MuJoCo Warp** |
| 云上部署难度 | 🔴 高：需 Vulkan/OpenGL、Omniverse，镜像体积大，部分云厂商容器不满足 | 🟢 低：纯 Python/MuJoCo，`uv sync` 即可，无 Omniverse 依赖 |
| 成熟度 | 🟢 高：822★、4 贡献者、有 tag、论文主用 | 🟡 中：251★、2 贡献者，是 port |
| `num_envs` | 4096 | 2048 |
| 依赖 | IsaacLab commit `f73c331738`（post-v2.3.2） | `mjlab 1.6.0` 需放在 `../mjlab`；`instinct_rl 1.0.2`（commit `3a28448`） |
| checkpoint 兼容 | — | ⚠️ **与 IsaacLab 关节顺序不同，权重不通用**（MJCF 顺序 vs 肩优先顺序） |
| 建议 | 想稳、想对齐论文 | **云上首选**，成本与上手难度都低得多 |

> `instinct_rl` 主线要求 **Python `>=3.12,<3.13`、PyTorch `2.11.0`**，版本较新，装环境时留意。

**建议：先用 InstinctMJ 在云上把 `Instinct-Parkour-Target-Amp-G1-v0` 跑通（训一个 baseline + 播放预训练权重），再决定是否上 InstinctLab。**

### 7.5 修正后的复现路线

| 阶段 | 内容 | 工期 | 算力 |
|---|---|---|---|
| **P0 环境** | `uv sync` + `uv run instinct-list-envs`；用官方 **Parkour 预训练权重** 跑 `--load-run` 播放，确认地形/深度相机/越障行为符合预期 | 1–3 天 | ~0 |
| **P1 Baseline** | `instinct-train Instinct-Parkour-Target-Amp-G1-v0` 从零训一版，观察成功率与地形课程推进 | 1–3 天（多为等待） | ~20–60 GPU·h |
| **P2 定制** | 按你的越障需求改地形子集（只留 `boxes`/`square_gaps`/`pyramid_stairs*`）、调 `num_rows/num_cols`、调奖励权重；跑 A/B | 1–2 周 | 100–300 GPU·h |
| **P3 部署** | `--export-onnx` → `instinct_onboard`；关节序对齐 + 限幅核对 | 1 周 | ~0 |
| **P4 进阶（可选）** | 若需要**手撑跳马/攀爬**类全身技能 → 走 `Instinct-Perceptive-Shadowing-G1-v0`（地形匹配动作 + 蒸馏）或 `Instinct-Perceptive-HOI-Shadowing-G1-v0`（OMOMO 物体交互） | — | — |

**总投入对比（对"简易越障跑酷策略"这一目标）：**

| 路线 | 人力 | 周期 | GPU 预算 |
|---|---|---|---|
| ❌ 补全 lucidrains 仓库 + 自建流水线（原 §5 方案 A/B） | 1–2 人 | 1.5–4 个月 | 200–2500 GPU·h |
| ❌ 从 IsaacLab + BeamDojo 自建 | 1–2 人 | 1–3 个月 | 200–1000 GPU·h |
| ✅ **InstinctMJ/InstinctLab 直接用** | **1 人** | **1–3 周** | **数十–300 GPU·h** |

### 7.6 必须复现的差距（把话说清楚）

走这条路你得到的是 **「Project Instinct 的感知式越障跑酷」**，而**不是** 「Light-Loco-Parkour 的全身多技能跑酷」。具体差异：

| 能力 | LLP | Instinct-Parkour |
|---|---|---|
| 速度指令跟踪 locomotion | ✅ | ✅ |
| 稀疏落脚点 / 踏石 / 平衡木 | ✅（BeamDojo 类地形） | 🟡 有 `square_gaps`/`boxes`，但无专设踏石地形（可自行加） |
| 楼梯 / 斜坡 / 崎岖 | ✅ | ✅ |
| 箱体越障（下肢接触） | ✅ | ✅（0.05–0.45 m） |
| **手撑跳马 / 攀爬（speed-vault / climb-and-step）** | ✅ 核心贡献 | ❌ **不含**（在 Shadowing/HOI 族） |
| 多专家蒸馏 + 过渡学习（无 skill label 自主切换） | ✅ 核心贡献 | ❌ 单策略端到端训练 |
| 深度相机 onboard + 噪声/延迟建模 | ✅ | ✅ |
| ONNX + G1 真机部署链路 | ❌ 未开源 | ✅ **已打通** |

**结论：如果你的目标就是"简易越障技能的跑酷策略"，那么本任务不是"退而求其次"，而是**比 LLP 更适合的工程起点**——它原生 G1、原生越障、原生可部署。若你最终必须复现 LLP 的**全身技能**与**多技能蒸馏**，则应以 Instinct 做地基，再叠加 `Perceptive-Shadowing`（动作跟踪）与 `instinct_rl` 的 `TPPO`（DAgger 蒸馏）两件现成工具去补那两块。

---

## 附录：本次审计的执行清单

```bash
# 已执行
git clone --depth 50 https://github.com/lucidrains/light-loco-parkour.git llp_reference
git log --oneline -40                       # 30 commits, 2026-08-05 → 2026-09-11
find llp_reference -type f                  # 11 个文件，无环境代码
wc -l light_loco_parkour/*.py tests/*.py    # 主实现 1626 行
grep -r "isaac|mujoco|terrain|depth_camera|G1|onnx|deploy"  # 0 命中
find . -name "validate_transition*"         # 0 命中（test_light_loco.py 的依赖缺失）
```

**已确认的关键事实**：
- 论文平台 = Lightbot 0（90 cm / 21 DoF），非 Unitree G1
- 论文训练框架 = IsaacLab，策略 50 Hz，胸部 RealSense D435
- 论文消融预算 = 每变体 15,000 iters；深度蒸馏 14,000 + 微调 1,000
- 社区仓库 = 纯算法层，无环境；`tests/test_light_loco.py` 依赖文件缺失，不可运行
- 奖励项与 Table I 权重逐条对齐（正面信号）
