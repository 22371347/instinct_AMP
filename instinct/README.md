# InstinctLab / Deep Whole-Body Parkour 复现工程

> 目标：在 Unitree G1 上复现**感知式越障跑酷**策略（Project Instinct 的 `Instinct-Parkour-Target-Amp-G1-v0`）
> 云端训练（RTX 4090）+ 本地轻量验证（无 Isaac Sim）
> 目标平台：**G1 29-DoF (torsoBase_popsicle)**

---

## 0. 关键约束（务必先读）

| 项 | 事实 | 影响 |
|---|---|---|
| 本地 GPU | **RTX 3050 Laptop / 4 GB 显存** | ❌ **本地无法运行 Isaac Sim**（最低要求 8 GB 级 RTX） |
| 本地磁盘 | `/home` 15 GB 可用（已清理 7.4 GB）；`/mnt/winD` 64 GB | 仓库放 `/home`，重资产放 `winD` |
| 云端 | RTX 4090 / 24 GB，**已有 Isaac 镜像** | 所有 Isaac Sim / 训练 / 深度渲染都在云端 |
| Python | 云端 `instinct_rl` 需 **3.12** + **torch 2.11.0**；IsaacLab 5.1 用 **3.11** | ⚠️ 见 §4 环境隔离方案 |

**因此本地的定位是「控制面 + 轻量验证」，不是「仿真面」。**

---

## 1. 目录结构

```
~/rsl_project/llp/
├── REPRODUCTION_ANALYSIS.md          # 可行性分析报告（含越障地形源码实证）
├── llp_reference/                    # lucidrains 社区实现（参考用，已弃用为起点）
└── instinct/                         # ← 本工程
    ├── README.md                     # 本文件
    ├── envs/
    │   └── instinct.sh               # 本地 conda env 激活脚本
    ├── repos/                        # 上游仓库（shallow clone）
    │   ├── InstinctLab/              # 环境侧（Isaac Lab 扩展）
    │   ├── InstinctMJ/               # 环境侧（MuJoCo）★ 本地回放模型来源
    │   ├── instinct_rl/              # RL 侧（PPO / AMP-WASABI / TPPO）
    │   └── instinct_onboard/         # G1 真机部署
    ├── assets/
    │   └── g1_29dof_popsicle_play.xml  # 生成的「可回放」MJCF（注入执行器+地面+keyframe）
    ├── tools/                        # 本地校验与运维脚本
    │   ├── check_env.py              # 环境自检
    │   ├── smoke_test.py             # 物理合理性冒烟测试（19 项）
    │   ├── joint_order.py            # IsaacLab ↔ MuJoCo 关节序映射
    │   ├── g1_params.py              # 执行器 kp/kd/action_scale（可溯源）
    │   ├── build_g1_mjcf.py          # 生成可回放 MJCF
    │   ├── g1_mujoco_play.py         # MuJoCo 轻量回放
    │   └── instinct_ssh.py           # 云端 SSH/scp（复用 AMP_mjlab 凭据）
    ├── patches/                      # 云端配置补丁（数据集路径等）
    └── docs/                         # 笔记 / 故障排查
```

---

## 2. 本地验证环境

### 2.1 conda 环境 `instinct`

```bash
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate instinct
```

内容（**独立于** `AMP_mjlab` 的 `isaaclab` 环境，互不干扰）：

| 包 | 用途 |
|---|---|
| `mujoco` | 本地轻量物理仿真 + 回放（G1 MJCF 已随 InstinctLab 提供） |
| `onnxruntime` | 加载云端导出的 ONNX 策略（**主力路径**） |
| `torch` (CPU) | 加载 `.pt` checkpoint（备用路径） |
| `numpy` / `scipy` / `trimesh` / `matplotlib` / `pyyaml` | 数据与动作校验、绘图 |

### 2.2 自检

```bash
python tools/check_env.py
```

### 2.3 可用资源

**MuJoCo G1 模型 —— 用 InstinctMJ，不要用 InstinctLab 自带的那份：**

| 模型 | 路径 | 自由度 | 可用性 |
|---|---|---|---|
| ✅ **29-DoF popsicle** | `repos/InstinctMJ/src/instinct_mj/assets/resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml` | 29 铰链 | **nu=0（执行器由代码注入）→ 需 `build_g1_mjcf.py`** |
| ✅ **29-DoF popsicle + 鞋**（跑酷任务实际用的） | `repos/InstinctMJ/src/instinct_mj/tasks/parkour/mjcf/g1_29dof_torsoBase_popsicle_with_shoe.xml` | 29 铰链 | 同上 |
| ❌ InstinctLab 自带 `g1.xml` | `InstinctLab/.../unitree_g1/g1.xml` | **nu=37**（23-DoF+双手） | 命名也不同（`torso_joint`）→ **与 29-DoF 策略不匹配，勿用** |
| ❌ 你已有的 `AMP_mjlab/deploy/xmls/g1.xml` | `g1_29dof_rev_1_0` | nu=0 | 无执行器，是部署用模型 |

**你已有的相关资产（可复用）：**

| 资产 | 路径 | 说明 |
|---|---|---|
| 云端运维体系 | `~/rsl_project/AMP_mjlab/tools/`, `deploy/` | SSH 助手、续训、看门狗、曲线回传 |
| AMASS 动作数据 | `~/rsl_project/GMR/AMASS_CMU/`（12 GB） | **Instinct-Parkour 的 AMASS 依赖已具备** |
| GMR 重定向 | `~/rsl_project/GMR/` | 已有重定向工具链 |
| 动作校验 | `AMP_mjlab/tools/validate_motion_csv.py` | 已可直接复用 |
| 走路 ONNX | `~/rsl_project/models/walk_model.onnx` | 可用于打通回放链路 |

---

## 3. 本地轻量回放（无需 Isaac Sim）

### 3.0 第一步：生成「可回放」MJCF（**必做**）

InstinctMJ/InstinctLab 的 G1 MJCF **本身没有执行器（nu=0）** —— 执行器是在代码里
通过 `beyondmimic_g1_29dof_delayed_actuator_cfgs` 注入的。所以先跑：

```bash
conda activate instinct
cd ~/rsl_project/llp/instinct
python tools/build_g1_mjcf.py            # 默认 popsicle
python tools/build_g1_mjcf.py --with-shoe # 跑酷任务实际用的带鞋变体
```

它会生成 `assets/g1_29dof_popsicle_play.xml`，内容：

- **29 个 position 执行器**：`kp`/`kv` 由 BeyondMimic 公式
  `armature × (10Hz·2π)²` 推出（可溯源到 InstinctMJ 源码）
- **地面 + 光照**：裸 MJCF 没有场景，不加会**直接穿地掉落**
- **keyframe `home`**：`_POPSICLE_INIT_STATE` 默认站姿，root 高 0.82 m
- 自动回读校验：`nu==29`、`ctrl 顺序 == MUJOCO_ORDER`、`kp/kv` 写入正确

### 3.1 看模型与关节序

```bash
python tools/g1_mujoco_play.py --dump-model     # 执行器/关节/限位
python tools/joint_order.py --xml assets/g1_29dof_popsicle_play.xml
```

**InstinctMJ(MuJoCo) 与 InstinctLab(IsaacLab) 的关节顺序不同，权重不通用**，
必须先拿到映射再谈回放。实测确认：

| | 顺序 |
|---|---|
| MuJoCo（ctrl 向量） | 腰(pitch,roll,yaw) → 左腿 → 右腿 → 左臂 → 右臂 |
| IsaacLab（策略向量） | **肩优先、左右交错**（`L_shoulder_pitch, R_shoulder_pitch, waist_pitch, ...`） |

`joint_order.policy_to_mujoco_index()` 给出置换索引，用法：
`mujoco_ctrl = [policy_vec[i] for i in policy_to_mujoco_index()]`

### 3.2 回放策略

```bash
# 看策略 IO（先确认观测维度）
python tools/g1_mujoco_play.py --onnx-info --policy /path/to/policy.onnx

# 回放（默认开 viewer）
python tools/g1_mujoco_play.py --policy /path/to/policy.onnx --cmd 0.5 0.0 0.0 --duration 15
# 不开 viewer / 录制
python tools/g1_mujoco_play.py --policy /path/to/policy.onnx --no-view --record out/parkour.mp4
```

> ⚠️ **观测向量的拼接顺序必须与训练时一致**。脚本内置的 `ObsBuilder` 只是
> 「可工作的默认值」（按 `obs_group=policy`，history=8 拼 6 段），
> 请用 `--onnx-info` 打印的输入维度自行核对后调整 `build_observation()`。
> 带 `depth_image` 的策略本地**无法**完整回放（见 3.3）。

### 3.3 验证与边界

```bash
python tools/check_env.py      # 环境自检（35 项）
python tools/smoke_test.py     # 物理合理性（19 项）
```

本地 MuJoCo 回放能验证：
- ✅ 策略网络加载 / 输入输出维度 / 关节序映射
- ✅ PD 跟踪与**逐关节**动作尺度（`beyondmimic_action_scale`，0.0745~0.5476）
- ✅ 模型完整性、地面接触、数值稳定性

**不能**验证：
- ❌ IsaacLab 的**地形生成器 / 课程学习 / 域随机化**
- ❌ **深度相机观测**（`depth_image`，64×36 + 噪声管道）—— MuJoCo 侧无等价输入
- ❌ `NoisyGroupedRayCasterCamera` / `VolumePoints` / `feet_at_plane` 等 InstinctLab 专有传感器

> 所以：**越障行为的真实验证必须在云端完成。** 本地只做「链路打通 + 低风险回归」。

> ℹ️ 零动作下机器人会缓慢倾倒 —— 这是**预期行为**：`_POPSICLE_INIT_STATE` 是 RL 的
> **训练复位姿态**，不是静态平衡姿态，站稳需要策略主动控制。

---

## 4. 云端环境方案（下一步）

### 4.1 版本约束（重要）

| 组件 | 要求 | 备注 |
|---|---|---|
| Isaac Sim | **5.1.0** | InstinctLab README 指定 |
| IsaacLab | commit `f73c331738`（post-v2.3.2） | 必须匹配 |
| `instinct_rl` | **Python 3.12** + **torch 2.11.0** | ⚠️ 你现有 `isaaclab` env 是 py3.11.13 + torch 2.7.0 → **必须新建独立 env** |
| InstinctLab | 以 IsaacLab **扩展**方式安装 | 官方提供 `docker/Dockerfile` |

### 4.2 环境隔离策略

```
云端
├── isaaclab (py3.11.13, torch 2.7.0+cu128)   ← 现有，AMP_mjlab 在用，【不要动】
└── instinct  (py3.12, Isaac Sim 5.1 + IsaacLab f73c3317 + instinct_rl)  ← 新建
```

**理由**：`instinct_rl 1.0.3` 硬性要求 `python_requires=">=3.12,<3.13"` 与 `torch==2.11.0`，
与现有 `mjlab 1.2.0` + `mujoco-warp 3.8.1` + `torch 2.7.0` 栈**直接冲突**，
共用会破坏正在跑的 AMP 训练环境。

### 4.3 需要确认的前置事实

- [ ] 云端镜像的 Isaac Sim 版本是否为 **5.1.0**（若是 4.x，InstinctLab 需降级适配或换镜像）
- [ ] 云端 `/root/gpufree-data` 剩余空间（Isaac Sim + 数据集 + 日志）
- [ ] `instinct_rl` 选 **1.0.3**(main, py3.12) 还是 **1.0.2**(commit `3a28448`, InstinctMJ 锁定版)

---

## 5. 数据准备（已具备大半）

Instinct-Parkour 的参考动作是 **AMASS 经 GMR 重定向**后的 `.npz`，配置项：

```python
# repos/InstinctLab/.../tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py
path = os.path.expanduser("~/Datasets")
filtered_motion_selection_filepath = os.path.expanduser("~/Datasets/parkour_motion_without_run.yaml")
motion_start_from_middle_range = [0.0, 0.9]   # ≈ RSI 参考状态初始化
```

**你已有的**：`GMR/AMASS_CMU/`（12 GB）、`GMR/general_motion_retargeting/`、`AMP_mjlab/tools/validate_motion_csv.py`

**需要做的**（作为 `patches/` 里的补丁，而不是改上游仓库）：
1. 从 AMASS_CMU 选出适用于跑酷的动作子集，生成 `parkour_motion_without_run.yaml`
2. 用 GMR 重定向到 G1 29-DoF popsicle → CSV
3. 用 InstinctLab 的数据管线转成 npz（云端执行）

> 详细流程与现有 AMP 管线一致，参考 `AMP_mjlab/README.md` §9。

---

## 6. 训练与运维

复用你现有的 `AMP_mjlab` 运维范式（同一台云端机器）：

```bash
# 通过 tools/instinct_ssh.py 复用凭据（不重复存放密码）
python tools/instinct_ssh.py "nvidia-smi"
```

后续将补充：
- `tools/train_parkour.sh` — 远程启动训练
- `tools/manage_instinct.sh` — 续训 / 停止 / 状态
- 曲线回传（复用 `AMP_mjlab/tools/refresh_plots.sh` 模式）

---

## 7. 关键坑（本地实测踩到的）

| # | 坑 | 现象 | 处理 |
|---|---|---|---|
| 1 | **MJCF 无执行器** | 加载后 `nu=0`，无法控制 | `tools/build_g1_mjcf.py` 注入（kp/kv 来自 BeyondMimic 公式） |
| 2 | **裸 MJCF 无地面** | 机器人**穿地掉落**（高度 → −15 m） | 生成时注入 plane geom + 光照 |
| 3 | **InstinctLab 自带 `g1.xml` 是 23-DoF+双手** | `nu=37`，关节名 `torso_joint`，与 29-DoF 策略不匹配 | 改用 **InstinctMJ** 的 popsicle 模型 |
| 4 | **动作尺度是逐关节的** | 范围 **0.0745 ~ 0.5476**，不是全局 0.25 | `g1_params.action_scale_for()` |
| 5 | **IsaacLab ↔ MuJoCo 关节序不同** | 直接套用会关节错位、动作全乱 | `joint_order.py` 按名字映射 |
| 6 | **零动作站不稳** | 约 50 步后倾倒 | **预期**：训练复位姿态 ≠ 静态平衡姿态 |
| 7 | `git clone` TLS 中断 | 连接超时 | 国内网络问题，重试即可 |

---

## 8. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| MuJoCo 回放不动 | 观测顺序与训练不一致 | `--onnx-info` 对齐 `build_observation()` |
| 关节乱动/反向 | 关节序/符号不匹配 | 用 `tools/joint_order.py` 按名字映射 |
| 机器人掉进地板 | 没注入地面 | 重新跑 `tools/build_g1_mjcf.py` |
| 云端 import 冲突 | 与 AMP_mjlab 环境串了 | 确认 `conda activate instinct` 且未设 `PYTHONPATH=/root/AMP_mjlab` |
| `conda create` 后包全缺失 | pip 输出被 `tail` 缓冲，看似卡住 | 分批安装、不要用 `\| tail` 观察进度 |

---

## 9. 参考

- 论文：Deep Whole-Body Parkour — arXiv:2601.07701
- 论文：Hiking in the Wild (Scalable Perceptive Parkour) — arXiv:2601.07718
- 上游：https://github.com/project-instinct/InstinctLab
- 生态：https://project-instinct.github.io/
- ⚠️ **License**：InstinctLab / InstinctMJ 为 **CC BY-NC 4.0（非商用）**；`instinct_rl` 为 Modified MIT
