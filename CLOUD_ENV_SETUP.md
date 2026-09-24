# 云端算力平台环境配置规格书

> 目标：为 `instinct_xu`（学长交付包）+ 我们新增的 AMP 行走任务
> （`Instinct-Amp-Walk-G1-v0`）准备一个**可训练、可验证**的 Isaac Sim 环境。
>
> **本文档中的版本号全部有出处**（见每行「依据」列），不是推测。
> 唯一没写死的是 GPU 驱动最低版本 —— 那个必须以 NVIDIA 官方文档为准，我不编数字。

---

## 0. 结论速览（一句话）

> **Python 3.11 + Isaac Sim 5.1.0 + PyTorch 2.7.0/cu128 + IsaacLab 源码 `f73c331738`（= 0.54.4）**

这一组是**硬绑定**的，不能随意升级。原因见 §4「为什么不能动版本」。

---

## 1. 精确版本规格表

| 组件 | 版本 / 提交 | 依据 |
|---|---|---|
| OS | **Ubuntu 22.04 或 24.04**（官方两者都支持）x86_64 | [Isaac Sim 5.1 系统要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) |
| GLIBC | ≥ 2.35（`manylinux_2_35_x86_64`） | 同上（Ubuntu 22.04=2.35、24.04=2.39，均 ✔） |
| **GPU 驱动（Linux）** | **≥ 580.65.06** | 同上（Isaac Sim 5.1 实测于 580.65.06）⚠️ 云主机驱动必须查! |
| RAM / 存储 | 最低 32 GB / 50 GB；**建议 64 GB / 500 GB** | 同上 |
| VRAM | 最低 16 GB（需带 RT Core 的卡） | 同上（4090 24 GB ✔） |
| Python | **3.11** | `instinct_xu/README.md` §2 + Isaac Sim 5.1 文档 |
| Isaac Sim（pip 包） | **`isaacsim[all,extscache]==5.1.0`** | `instinct_xu/README.md` §2 B |
| Isaac Sim 包索引 | `https://pypi.nvidia.com`（额外索引） | 同上 |
| PyTorch | **`torch==2.7.0`**（CUDA **12.8**） | 同上 |
| torchvision | **`torchvision==0.22.0`** | 同上 |
| PyTorch 包索引 | `https://download.pytorch.org/whl/cu128` | 同上 |
| **IsaacLab** | **源码提交 `f73c33173801f5f8afea4142482e47b7710c2b75`**（pip 显示版本号 **0.54.4**） | 同上 |
| 原版 InstinctLab | `ChaoMingSanHua/InstinctLab` @ `8879c06d3f3ae376aa96988a58d2f7f3661ec12d` | `instinct_xu/README.md` §3 |
| 原版 instinct_rl | `ChaoMingSanHua/instinct_rl` @ `5523e3847cfce578bfcee25a5334256d9cc53e0e` | 同上 |
| 参考文档 | Isaac Lab pip 安装 | [链接](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/setup/installation/pip_installation.html) |

**额外 pip 依赖**（README §2 B 末尾，`instinctlab` 的 `INSTALL_REQUIRES` 也要求这些）：

```
setuptools<82  wheel  toml  psutil  pytorch_kinematics  joblib  debugpy
snakeviz  trimesh[all]  scikit-learn  opencv-python  pyvista
tensorboardX  tensorboard  tabulate  GitPython  onnx  onnxscript
```

---

## 2. 两条路线，选哪条

### 路线 A：Docker（与作者环境最一致）

官方 `instinct/repos/InstinctLab/docker/` 的做法是**两层**：

1. 先用 IsaacLab 仓库自带的 docker 流程构建 **`isaac-lab-base`** 镜像
   （它自己再 base 到 `nvcr.io/nvidia/isaac-sim` 的对应版本 tag）
2. 再用 `instinct/repos/InstinctLab/docker/Dockerfile` 在其上加 `instinctlab`

证据：`instinct/repos/InstinctLab/docker/.env.base` 写着
`ISAACLAB_BASE_IMAGE=isaac-lab-base`，`docker-compose.yaml` 用
`--build-arg ISAACLAB_BASE_IMAGE_ARG=${ISAACLAB_BASE_IMAGE}` 消费它。

| 优点 | 缺点 |
|---|---|
| 与作者环境最一致，依赖问题最少 | 镜像 ~25 GB，构建 + 下载耗时长 |
| 环境可复现、可归档 | 云平台对 docker-in-docker / 自定义镜像的支持需确认 |

### 路线 B：conda + pip（**推荐**，README 官方也给的这条）

**关键认识**：Isaac Sim 用 pip 安装时**会把 CUDA 运行库一起装进来**，
容器里**不需要**单独装 CUDA Toolkit —— 只要宿主机的 NVIDIA 驱动 + `nvidia-container-toolkit`。
所以基础镜像用**最朴素的 Ubuntu 22.04 就够**，不必找 CUDA 镜像。

```bash
# 基础镜像: ubuntu:22.04  (或云平台上的 ubuntu22.04 通用镜像)

# 1) 环境
conda create -n instinct_xu python=3.11 -y
conda activate instinct_xu
python -m pip install --upgrade pip

# 2) Isaac Sim 5.1.0 + PyTorch 2.7.0/cu128
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 3) IsaacLab（必须 pin 到具体提交）
git clone --recurse-submodules https://github.com/isaac-sim/IsaacLab.git "$HOME/IsaacLab-instinct-xu"
git -C "$HOME/IsaacLab-instinct-xu" checkout f73c33173801f5f8afea4142482e47b7710c2b75
git -C "$HOME/IsaacLab-instinct-xu" submodule update --init --recursive
cd "$HOME/IsaacLab-instinct-xu" && ./isaaclab.sh -i

# 4) 其它依赖
python -m pip install 'setuptools<82' wheel toml psutil pytorch_kinematics joblib debugpy snakeviz \
  'trimesh[all]' scikit-learn opencv-python pyvista \
  tensorboardX tensorboard tabulate GitPython onnx onnxscript

# 5) 装 instinct_xu 的两份源码（editable，--no-deps 很重要：不要动已装好的 Isaac Sim/Torch）
cd /path/to/instinct_xu
python -m pip install -e source/instinctlab   --no-deps
python -m pip install -e source/instinct_rl    --no-deps
```

---

## 3. 验收清单（分 4 层，逐层过）

```bash
# ── 第 1 层：安装与导入指向（不启动 Isaac Sim）──────────────
python scripts/check_install.py
#   期望: 打印 Python 版本 / torch 版本 / CUDA runtime 2.7.x+cu128 / GPU 名称
#   并确认 instinctlab、instinct_rl 包指向 instinct_xu/source/ 下（不是别处）
python -m pip list --editable      # 应有两行，分别以 source/instinct_rl、source/instinctlab 结尾

# ── 第 2 层：原任务连通性（README §5 官方冒烟）──────────────
CUDA_VISIBLE_DEVICES=0 ./train.sh \
  --task Instinct-Perceptive-Vault-G1-v0 \
  --num_envs 64 --max_iterations 3 --device cuda:0 --headless \
  --run_name instinct_xu_install_smoke

# ── 第 3 层：★ 我们新增任务的冒烟（见 §5）───────────────────
bash /path/to/llp/instinct/tools/cloud_smoke_amp_walk.sh

# ── 第 4 层：正式训练前的规模试跑 ─────────────────────────
#   4096 环境是按 24 GB 显存给的；先跑 512 看显存占用再放大
```

---

## 4. 为什么不能动版本（三个真实的坑）

1. **IsaacLab 必须 pin 提交，不能用 master 或 release tag**
   `instinct_xu` 的代码直接依赖 `f73c331738` 这个提交的 API。
   换任何其它版本都可能出现接口不兼容（本项目已多次遇到这类问题）。

2. **不要为了装别的包而升级/降级 torch 或 numpy**
   README 明确警告「不要为了本包重装 Isaac Sim、Isaac Lab、Torch 或 NumPy」。
   所以第 5 步装源码时 **必须加 `--no-deps`**。

3. ⚠️ **别被同名的参照仓库带偏**（本次实测发现的坑）
   本地参照仓库 `instinct/repos/instinct_rl`（提交 `574c8d50`，**比交付包所用的
   `5523e384` 更新**）里写着：

   ```
   python_requires=">=3.12,<3.13"     torch==2.11.0     torchvision==0.26.0
   ```

   这与交付包 README 的「Python 3.11 + torch 2.7.0」**冲突**。
   经核对，**交付包自己的** `instinct_xu/source/instinct_rl/setup.py` 写的是
   `python_requires=">=3.6"` / `torch>=2.4.0`（宽松），与 README 一致。

   → **结论：以 `instinct_xu/README.md` 为准**。参照仓库的严格 pin 是它自己新版本的要求，
   与我们无关。若云平台上照抄了 `>=3.12` 就会把整个环境搞崩。

---

## 5. 我们新增任务的云端冒烟要点

脚本 `instinct/tools/cloud_smoke_amp_walk.sh` 会依次验证：

| # | 验证项 | 失败说明什么 |
|---|---|---|
| 1 | `import isaaclab.envs.mdp` 里 `UniformVelocityCommand` 类是否存在 | 步骤③ 的方向分档指令**依赖它**（本机无法验证，是最大风险点） |
| 2 | `UniformVelocityCommand` 是否有 `vel_command_b` 缓冲 | 同上；`DirectionalVelocityCommand.__init__` 有 fail-fast 会直接报错 |
| 3 | `--task Instinct-Amp-Walk-G1-v0 --num_envs 16 --max_iterations 2` | 端到端：G1 资产 + 高程图 527 射线 + 626 维观测 + 方向分档指令 + `motion_reference`/`robot_reference` 场景 + `amp_policy`/`amp_reference` 同构 + **步骤⑤ 的奖励集能被 `RewardManager` 实例化并跑通前向** + **步骤⑥ 的 `WasabiPPOAmpMix`/判别器/奖励混合能跑完一个完整迭代** |
| 4 | 契约校验器打印 `[AMP-WALK PASS]` | 说明动作空间/时序/执行器/命令范围全部符合预期 |
| 5 | 日志里出现 `Discriminator Network Structure` | 说明步骤⑥ 的 `module:Class` 解析成了、判别器也建起来了（`init_storage` 跑到） |
| 6 | 不出现 `is not found in the observation dictionary` | 这是 `WasabiAlgoMixin.process_env_step` 对 `amp_policy`/`amp_reference` 缺失的 fail-fast |

> 第 3 项顺带覆盖步骤⑤⑥：`RewardManager` 与判别器都在 `load_managers()` / `init_storage()` 里构建，
> **先于**契约校验；因此任一奖励项 `func` 不存在 / 参数名写错 / `asset.data.<attr>` 不存在，
> 或算法 `class_name` 解不出来 / `discriminator_kwargs` 漏 `hidden_sizes` / 两侧 AMP 状态不同构，
> 都会在 `[AMP-WALK PASS]` 之前报错。跑起来后应顺带看 TensorBoard 里的：
> - `Metrics/foot_sep_mean`（应≈0.19~0.25）、`Metrics/foot_cross_frac`（直线段应≈0）
> - `Step/amp_style_reward`（判别器风格奖励，`"quad"` 形式，值域 [0,1]）
> - `Step/discriminator_reward`（⚠️ lerp 模式下它是**修正量** `(1-λ)(r_amp - r_task/c)`，不是风格奖励）
> - `Loss/discriminator_loss`、`Loss/discriminator_gradient_penalty`

**已知会在云端首次暴露、本机无法验证的假设**（按风险排序）：

1. `isaaclab.envs.mdp.UniformVelocityCommand`（类）是否可导入 —— 仓库内只有 `...Cfg` 的先例
2. 该类的内部缓冲名确实叫 `vel_command_b`
3. `robot_reference` + `motion_reference` 传感器在 `G1OmniLocoSceneCfg` 基础上叠加后能否正常构建
4. `motion_reference` 的关节顺序与机器人是否逐一对齐（两侧 AMP 状态逐项同构依赖它）
5. `isaaclab.envs.mdp.joint_vel_rel` 存在
6. ⭐ **步骤⑤ 新增**：`LL_FOOT` / `LR_FOOT` 是否真的出现在 `robot.body_names` 里
   （URDF 里它们是 `merge_fixed_joints=False` 保留的、质量 1e-9 的纯框架 link）。
   若被 PhysX 丢掉，`feet_lateral_sep` 会自动退到 `left/right_ankle_roll_link`
   （横向偏移为 0，两者数值几乎一致），但**日志里会少掉这个信号**，
   建议在冒烟输出里 `print` 一次 `robot.body_names` 确认。
7. ⭐ **步骤⑤ 新增**：`root_ang_vel_b`（IsaacLab 标准属性，仓库里只是没直接用过）。
8. ⭐ **步骤⑥ 新增**：`WasabiAlgoMixin` 的 `amp_storage` 与 `discriminator_update` 在
   `num_steps_per_env=24`、`4096` envs 下的显存/耗时（判别器 1024/512，输入 670 维）。
   先用 `--num_envs 16` 跑通，再爬规模。

> 这些条每一条都已在本地用**静态契约检查**做过能做的部分
> （`instinct/tools/check_amp_walk_task.py`，**11 类检查 A–K 全过**），
> 但静态检查不能替代真实运行。**建议云环境一通就先跑第 3 层冒烟**，
> 把 ②③④⑤⑥ 五步积累的「未验证断言」一次性清掉，再进 ⑦ 正式小规模训练。

---

## 6. 镜像/环境核对清单（已实操核验，2026-09-23）

### 6.1 `IsaacLab 2.3.2` vs 我们的 pin —— **可用，但不是精确版本**

这是**最容易搞混**的一点：IsaacLab 有**两套独立编号**——

| 编号体系 | 值 | 说明 |
|---|---|---|
| GitHub release tag | `v2.3.x` | 镜像厂商说的“IsaacLab-2.3.2”就是这个 |
| pip 包版本号 | `0.54.4` | 在提交 `f73c331738` 处 `pip show isaaclab` 显示的值 |

**实测的 tag → commit 对应**（`git ls-remote --tags`）：

```
v2.3.0 = 3c6e67bb
v2.3.1 = 5c2ec81c   (2025-12-03)
v2.3.2 = 37ddf626   (2026-01-29)
我们的 pin = f73c331738 (2026-06-02)   ← 不等于任何 tag
```

**祖先关系**（GitHub compare API，`f73c331738...37ddf626`）：

```
status = behind      ahead_by = 0      behind_by = 25
=> v2.3.2 是我们的 pin 的**严格祖先**，落后 25 个提交
```

**那这 25 个提交会不会弄坏我们？** 实测了它们的改动文件（共 77 个），
其中只有 **1 个**落在我们依赖的路径上（`envs/mdp/`、`managers/`、`terrains/`、
`sensors/ray_caster/`、`assets/articulation/`）：

```
modified  2  source/isaaclab/isaaclab/envs/mdp/actions/joint_actions_to_limits.py
```

而它改的是 `JointPositionToLimitsAction` 的 `preserve_order` 标志 ——
我们用的是 `JointPositionAction`（**不是** ToLimits 那个），所以连这一条都不影响。

其余 24 个提交全是文档 / CI-changelog 自动化 / RSL-RL·skrl 集成配置 /
依赖 pin / Franka 等其它任务的修复。

**特别地**，我们最担心的两个东西**没有被触及**：

| 我们依赖的 | 在 25 个提交里被改了吗 |
|---|---|
| `UniformVelocityCommand` / 其内部 `vel_command_b` | ❌ 未触及（`envs/mdp/commands/` 无改动） |
| `CommandManager` / `ObservationManager` | ❌ 未触及 |
| 射线传感器 / 地形 / 关节体 | ❌ 未触及 |

**结论**：

- ✅ **可以先用镜像自带的 2.3.2 开跑**，风险很小（上面的证据支持这个判断）；
- ⚠️ 但若想 100% 复刻作者环境，在镜像里追加：
  ```bash
  cd <IsaacLab目录>
  git fetch --all && git checkout f73c33173801f5f8afea4142482e47b7710c2b75
  git submodule update --init --recursive
  ./isaaclab.sh -i        # 重新装 editable；之后 pip show isaaclab 应为 0.54.4
  ```
  （前提：该目录是个 git 仓库。若是 pip 装的只读包，就直接用 2.3.2。）
- 无论走哪条，都**先用 `cloud_smoke_amp_walk.sh` 跑一遍** —— 25 个提交的差异已被证明
  无关，但「未跑过」才是真正的风险。

### 6.2 Ubuntu 24.04 —— ✅ 官方支持

Isaac Sim 5.1 系统要求写的是 `OS: Ubuntu 22.04/24.04`，两者都官方支持。
只需 `GLIBC ≥ 2.35`（24.04 是 2.39，✔）。

### 6.3 ⚠️ 必须核对的一项：GPU 驱动 ≥ 580.65.06

Isaac Sim 5.1 的实测驱动是 **580.65.06 (Linux)**。这是唯一**外在于镜像**的东西——
云平台宿主机驱动。**启动镜像后第一件事就是查**：

```bash
nvidia-smi   # 右上角 Driver Version 需 >= 580.65.06
ldd --version | head -1   # 需 >= 2.35
```

驱动不达标的话，Isaac Sim 会在初始化阶段挂掉，且报错往往不直白。

---

## 7. 建议的执行顺序（含并行）

```
[t0] 立刻：在云平台开始构建镜像 / 装环境   ← 长尾任务，今天就要启动
      └─ 下载 isaacsim[all]==5.1.0（数 GB）+ 编译 IsaacLab，通常 1~3 小时

[t1] 同时：本地继续做 ⑦ 的训练前准备（纯代码，不依赖云）
      └─ ⑤ 奖励迁移、⑥ 算法接线**已全部完成**（提交 c5139ec / 本次）

[t2] 云环境就绪 → 跑 §3 第 1~3 层验收
      └─ 若第 3 层冒烟挂掉：说明 ②③④ 某处假设错，就地修（代价小）
      └─ 若通过：本地代码「从没跑过」的状态被清除，后面就都是增量风险了

[t3] ⑦ 正式小规模训练（先 512 env，确认显存后再上 4096）
```

---

## 8. 实测结果（2026-09-23，新云平台）

> 平台：`root@183.147.142.40:30528`，镜像 = Ubuntu 24.04 + CUDA 12.8 + 4090×1
> + IsaacSim 5.1.0 + IsaacLab 2.3.2
> 调用方式：`python instinct/tools/instinct_ssh.py --env-check` 等

### 8.1 结论：**可以用，但有 3 个坑已处理**

| 项 | 实测值 | 判定 |
|---|---|---|
| OS / GLIBC | Ubuntu 24.04.4 LTS / GLIBC 2.39 | ✅ |
| GPU / 驱动 | RTX 4090 24564 MiB / **580.126.09** | ✅ ≥ 580.65.06 |
| 容器配额 | **14 核** / 50 GB RAM（宿主 128 核 503 GB） | ✅ |
| Python | 3.11.15（`/opt/conda/envs/isaaclab`） | ✅ |
| isaacsim | **5.1.0.0** | ✅ 精确匹配 |
| torch / torchvision | **2.7.0+cu128** / **0.22.0+cu128** | ✅ 精确匹配 |
| numpy | 1.26.0 | ✅（<2，IsaacLab 要求） |
| isaaclab | **2.3.2.post1** | ⚠️ = release v2.3.2，落后 pin 25 个提交，但 §6.1 已证明差异无关 |
| Kit headless 启动 | ✅ 10.9 s 到 `app ready` | ✅ |
| CUDA 可用 | ✅ `torch.cuda.is_available()=True` | ✅ |

### 8.2 坑 1：Kit 首次运行弹 EULA，非交互 SSH 下直接失败 ⚠️

```
Do you accept the EULA? (Yes/No): Unable to bootstrap inner kit kernel: EOF when reading a line
```

`ssh host "cmd"` **没有 TTY** → 必挂。修复（已固化进 `/root/.llp_env.sh`）：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
```

### 8.3 坑 2（关键）：镜像里 `isaaclab` 被"外壳"遮蔽 ⚠️

镜像把**整个 IsaacLab 仓库快照**装进了 site-packages：

```
site-packages/isaaclab/            ← 外壳（有 __init__.py / __main__.py / apps / source）
site-packages/isaaclab/source/isaaclab/isaaclab/   ← 真正的库
site-packages/isaaclab/source/isaaclab_{tasks,assets,mimic,rl,contrib}/
```

后果：`import isaaclab` **成功**，但 `import isaaclab.envs` 报
`ModuleNotFoundError: No module named 'isaaclab.envs'` —— 因为外壳的
`__init__.py` 把真库挡住了。

**尝试过但无效**：往 site-packages 放 `.pth`
（`.pth` 路径排在 site-packages **之后**，外壳仍然胜出）。
**有效修复**：用 `PYTHONPATH`（它排在 site-packages **之前**）：

```bash
_LLP_SP=/opt/conda/envs/isaaclab/lib/python3.11/site-packages
for _d in "$_LLP_SP"/isaaclab/source/*/; do
    _d="${_d%/}"; _b=$(basename "$_d")
    [ -d "$_d/$_b" ] && export PYTHONPATH="$_d${PYTHONPATH:+:$PYTHONPATH}"
done
```

→ 已固化在 `/root/.llp_env.sh`，用 `instinct/tools/instinct_ssh.py --setup-remote` 生成。

**顺带把步骤③ 的假设也验证了**（这是本次最有价值的收获）：

```
OK isaaclab.envs.mdp 属性: UniformVelocityCommand, UniformVelocityCommandCfg,
                           joint_vel_rel, base_lin_vel, joint_pos_rel,
                           height_scan, generated_commands
--- UniformVelocityCommand 源码探测 ---
     OK vel_command_b       ← 我的 DirectionalVelocityCommand 就依赖它
     OK is_standing_env
     OK is_heading_env
     OK heading_target
```

### 8.4 坑 3：GitHub 不通

| 目标 | 状态 |
|---|---|
| github.com | ❌ timeout |
| pypi.org / pypi.nvidia.com | ✅ |
| 清华 PyPI 镜像（`/etc/pip.conf` 已配好） | ✅ |
| omniverse 资产服务器（S3） | ✅ 200 —— 高程图/材质能正常下载 |

**影响**：README §3 的「git clone 原版 InstinctLab / instinct_rl」这一步做不了。
**对策**：`instinct_xu/source/` 自带这两份源码（README §4 说它们会替换同名包），
**跳过 §3 直接装 §4 即可**；真要参照原版，本地 `instinct/repos/` 已有副本可上传。

### 8.5 磁盘布局（/ 只有 30 G，别装错地方）

| 路径 | 容量 | 类型 |
|---|---|---|
| `/` | 30 G（overlay） | 容器可写层 —— **不要放代码/日志** |
| `/root/gpufree-data` | **49 G** | 本地盘（快）→ 推荐放代码 + 日志 |
| `/root/gpufree-share` | **500 G** | JuiceFS 网络盘 → 放大文件/归档 |

`/root/.llp_env.sh` 里已导出 `LLP_DATA_ROOT=/root/gpufree-data`。
（Isaac Sim 本体在只读镜像层里，不占这 30 G。）

---

## 9. **CPU-only 模式**（2026-09-24 实测）

> 平台 GPU 资源紧张时以**仅 CPU** 形式开了：`nproc=128`、`free=503G`，
> 但 `nvidia-smi -L` **无输出**（没有任何 GPU 可见）。

### 9.1 结论速览

| 能做什么 | 不能做什么 |
|---|---|
| ✅ 装环境（editable 安装两份源码） | ❌ 训练 / 跑迭代（PhysX GPU 求解器起不来） |
| ✅ **读已装 IsaacLab 的源码**做 API 审计 | ❌ 任何需要真实物理步进的验证 |
| ✅ **启动 Kit**（见 9.3，居然可行） | ❌ 渲染 / 相机类任务 |
| ✅ Kit 内**实例化配置 + `to_dict()`** | ⚠️ 建环境/走 step：待验证（见 9.5） |
| ✅ **纯算法侧单元测试**（不需要 Kit，见 9.4） | |

### 9.2 不启动 Kit 时的导入边界（实测）

`pxr`(USD) / `omni` / `carb` 都不在普通 python 进程里，于是：

| 模块 | 结果 |
|---|---|
| `torch` / `numpy` / `gymnasium` / `instinct_rl` | ✅ 可导入 |
| `isaaclab.utils.configclass` | ✅ 可导入（纯 python） |
| `isaaclab.utils.math` / `isaaclab.managers` / `isaaclab.envs` | ❌ `No module named 'pxr'` / `'omni'` |
| `instinctlab` / 任务配置包 | ❌ 同上 |

⇒ **想 import 任务配置，必须先 `AppLauncher` 起 Kit**。别浪费时间在"用普通 python 验证配置"上。

### 9.3 ⭐ Kit 在无 GPU 时**确实能启动**

```
AppLauncher(headless=True)  →  返回成功（~10 s）
报错只有：Warp CUDA error: Could not open libcuda.so
          PhysX error: Could not load libcuda.so.1: ... file too short
```

即：**Kit 框架层能起来，GPU 相关的插件（Warp/PhysX-GPU/渲染）起不来**。
这足以做到「import 任务配置 + 实例化 + `to_dict()`」。

### 9.4 ⭐ 纯算法侧可以在**本机**（4 GB 无 GPU）端到端测

`instinct_rl` 是纯 PyTorch 的库，**奖励混合发生在算法侧**，所以步骤⑥ 的核心
（`WasabiPPOAmpMix` / `PPO.process_env_step` / `OnPolicyRunner` 的类解析）
根本不需要仿真器。工具 `instinct/tools/test_amp_pipeline.py`（需要
`GitPython gymnasium tensorboardX regex`）走**真实**的 `process_env_step` 管线，
读 `RolloutStorage.rewards` 与 AMP_mjlab 的目标公式逐点对账——**14 项全过**。

### 9.5 环境缺口（镜像里漏装的依赖）—— 已在云上修掉

| 现象 | 原因 | 处置 |
|---|---|---|
| 连配置都 import 不了 | `motion_reference_manager.py` 顶层 `import pytorch_kinematics`，镜像没装 | ✅ 补装（连带 `lxml` / `arm_pytorch_utilities` / `pytorch_seed`） |
| `check_install.py` 报 `Missing required package: isaaclab_assets` | 镜像只装了 `isaaclab`，`isaaclab_assets` **没有 pip 元数据** | ⚠️ 未修（见下） |
| `torch.cuda.is_available()` 为假 | 本来就没 GPU | — |

**安装纪律（铁律）**：一律 `pip install --no-deps`，然后**逐轮补传递依赖**，
每一步都比对 `torch` / `numpy` 版本。实测装完后
`torch 2.7.0+cu128` / `numpy 1.26.0` **一字未变**。
（工具：`cloud_fix_deps.sh` → `cloud_fix_deps2.sh` → **`cloud_dep_scan.py`**，
最后这个会**一次性扫完两棵源码树的全部顶层第三方 import**，别再一个一个撞。）

> `check_install.py` 在 CPU-only 下**必然失败**（它自己就要求
> `torch.cuda.is_available()`），所以第 1 层验收在 CPU 模式下要**拆开看**：
> 能验的是「包位置 / 版本」，不能验的是 CUDA 那两行。

### 9.6 由此新增的**云端只读审计**能力（不需要 GPU，价值很高）

本机没有 IsaacLab 源码，所以一批 API 假设一直只能靠文档推断。云上直接读
`site-packages/isaaclab/source/isaaclab/isaaclab/` 的源码就钉死了：

| 问题 | 结论 |
|---|---|
| **`@configclass` 无类型注解的新字段算不算字段？** | **算**。`add_annotations` 会**从默认值推断类型并补注解**（源码注释原文见 `tools/cloud_api_audit.sh`）→ 步骤⑥ 的 WASABI 参数一定能透传到 `to_dict()` |
| `RewardManager` 怎么遍历奖励项 | `cfg.__dict__.items()`，且 `if term_cfg is None: continue` → **把某项设成 `None` 就能关掉它** |
| `ArticulationData` 有哪些属性 | `root_pos_w` / `root_quat_w` / `root_ang_vel_b` / `root_lin_vel_b` / `default_root_state` / `root_state_w` / `body_pos_w` / `body_quat_w` **全部存在** |
| `SceneEntityCfg("robot")` 的 `joint_ids` | `joint_names=None` 时**整段跳过解析** → 保持 `slice(None)`（= 全部关节）；这版**有 `preserve_order` 字段**但默认 `False` |
| `ContactSensorCfg` 过滤 | 源码原文确认**只支持一对多**（`contact_sensor.py:50`）→ 「不迁移 `self_collisions`」的决定成立 |

审计工具：`instinct/tools/cloud_api_audit.sh`（`bash cloud_api_audit.sh Q1 Q3` 可只跑若干项）。

