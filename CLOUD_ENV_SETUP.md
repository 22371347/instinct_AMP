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

---

## 10. GPU 端到端冒烟：**步骤⑦ 实测通过**（2026-09-24 → 09-25）

跑法（后台 + 日志，避免 SSH 断线丢输出）：

```bash
bash /root/cloud_run_bg.sh 'bash /root/cloud_chain_smoke.sh' /root/smoke.log
# cloud_chain_smoke.sh = 环境快照 → cloud_dep_scan.py → cloud_smoke_amp_walk.sh（第 3 层是裁决）
```

### 10.1 本任务的检查项：**全绿**

| 检查 | 结果 |
|---|---|
| 训练进程退出码 | ✅ 0（`--task Instinct-Amp-Walk-G1-v0 --num_envs 16 --max_iterations 2 --device cuda:0`）|
| 契约校验 | ✅ `[AMP-WALK PASS] policy=626 … rewards=21` |
| 判别器构建 | ✅ `Discriminator Network Structure` |
| `amp_policy`/`amp_reference` | ✅ 两者都进了 `infos['observations']`，判别器拿到 670 维 |
| 脚距诊断量 | ✅ `Mean step foot_sep_mean: 0.2411`（在数据集直线走区间内）|
| 异常关键字 | ✅ 无 Traceback / KeyError / OOM |

### 10.2 实测尺寸（与设计逐项对上，可省掉以后的猜测）

```
obs_format（train 启动时打印）
  policy : height_scan(527) + velocity_command(3) + base_lin_vel(3)
         + projected_gravity(3) + base_ang_vel(3) + joint_pos_rel(29)
         + joint_vel(29) + last_action(29)              = 626 ✅
  critic : base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
         + velocity_commands(3) + joint_pos(29) + joint_vel(29) + actions(29) = 99 ✅
  amp_policy / amp_reference : 两侧**完全同构**（wasabi 的 init_storage 断言要求）
         projected_gravity(30) + joint_pos_rel(290) + joint_vel_rel(290)
         + base_lin_vel(30) + base_ang_vel(30)                     = 670 ✅
```

⭐ **由 30/290 反推出一个重要事实**：AMP 观测组是 **10 帧堆叠**，
单帧状态 = 3+29+29+3+3 = **67 维**，×10 = 670。以后调 AMP 输入维数时按这个来。

网络：`Actor 626→512/256/128→29`、`Critic 99→512/256/128→1`、
`Discriminator 670→1024/512→1`（ReLU）。16 环境 24 步/轮 → `Iteration time ≈ 3.3 s`。

### 10.3 冒烟暴露并修掉的 **2 个真 bug**（都不是环境问题）

**① 诊断量写进了会被清空的日志通道（真 bug，会让训练在首轮 log 时直接崩）**

第一轮冒烟在 `OnPolicyRunner.log()` 里 `KeyError: 'Metrics/foot_sep_mean'`，
训练**根本起不来**。根因：

- `ManagerBasedRLEnv._reset_idx` 里有 `self.extras["log"] = dict()`（**整体替换**），
  而 reward 的 `compute()` 在它**之前**跑；
- 该替换只在「本 step 有 env 终止」时执行 ⇒ 终止步写的诊断量被丢掉、非终止步才留下；
- `ep_infos` 里各条 dict 的 key 集合于是不一致，而 `log()` 是拿 `ep_infos[0]` 的 key
  去索引**全部** dict → KeyError。

✅ 修法：诊断量改走 **`env.extras.setdefault("step", {})[key] = value`**。
`InstinctRlEnv.step()` 里 `extras["step"]` 始终是**同一个 dict 对象、永不替换**，
key 集合天然一致；TB 里表现为 `Step/<key>`。
⚠️ 副作用：`step_infos` 存的是同一对象的 N 个引用，所以 `Step/*` 曲线是
**「本轮最后一个 step 的瞬时值」**，不是轮内均值 —— 看曲线时别误读。

**② 冒烟脚本自己的检查项方向写反了（假警报）**

`[A1 PASS]` 是**父类** `G1OmniLocoTeacherEnv._validate_teacher_contract` 打印的，
而我们的子类是 **先 `super()` 再补 AMP 专项检查** ⇒ 它与 `[AMP-WALK PASS]`
**同时出现属正常**，且报的是**当前类**的 626（不是 623）。
判据已改为「A1 报的维度是否 = 626」，并新增 `foot_sep_mean` 的
**在线自检**（不在 0.05~0.6 就告警 → 能自动发现脚部参考 body 选错）。

### 10.4 ⭐ `LL_FOOT`/`LR_FOOT` 找不到是**预期**（数值上已自证）

Kit 日志原文：`link LR_FOOT has body properties (mass, inertia, or collisions)
and is being merged into right_ankle_roll_link` —— URDF 导入器的 `merge_fixed_joints`
把固定脚连杆并进了 ankle roll 连杆，所以运行时 `body_names` 里没有脚连杆。
因为 `sep` 只在 yaw 系取 y 分量、foot→ankle_roll 的横向偏移为 0，
`sep_floor = 0.145`（在 LL_FOOT 上标定）**可原样使用**。
**实测 `foot_sep_mean = 0.2411`**（数据集直线走中位 0.2140 / p1 0.1455）→ 退路正确。

### 10.5 两个云端操作坑（新增）

| 坑 | 表现 | 对策 |
|---|---|---|
| **URDF 导入器扩展是"首次使用才拉"** | 卡在 `UrdfConverter` → `registry_provider._pull_extension_async_impl` → `omni_client_copy`，**无任何日志输出**，极易误判为死锁 | 一次性成本（~1 min+），缓存到 `/root/.local/share/ov`；用 `du -sh /root/.local/share/ov` 和 `.../exts/v2/index/*/packages/*.json` 的 mtime 判断"在涨=在干活"；第二轮起秒过 |
| **`pkill -f <pattern>` 会杀掉自己这条 SSH** | 命令行本身含该字符串 → 自己的 shell 被 kill，后续命令全不执行 | 用括号技巧：`pgrep -af '[c]loud_watch'` / `pkill -f '[c]loud_watch'` |

配套工具：`instinct/tools/cloud_watch.sh`（远端心跳，每 20 s 记 CPU/RSS/日志字节/ov 缓存，
一次轮询就能看出"慢"还是"卡死"）。

### 10.6 唯一残留的非致命项

`scripts/check_install.py`（**交付包自带**，README §5）报
`PackageNotFoundError: No package metadata was found for isaaclab_assets`。
这是**镜像里缺 pip 元数据**，不是缺包、也**不影响本任务**——第 3 层端到端冒烟退出码为 0
就是最直接的证据。已把它降级为提示项（真正的裁决看第 3 层）。

---

## 11. 规模试跑：显存/吞吐实测与 `num_envs` 决策（2026-09-25）

工具：`instinct/tools/cloud_scale_probe.sh <num_envs> <iters> [device]`
摘要落到 `/root/scale_probe_<N>_summary.log`（多次试跑按 N 分文件，不互相覆盖）；
`cloud_chain_probe.sh <目标N> <iters> <等待N>` 可链式跑多点。

### 11.1 实测数据

| num_envs | 显存峰值 | 稳态 `Iteration time` | 吞吐 | 退出码 | `foot_sep_mean` |
|---|---|---|---|---|---|
| 512 | **4010 MiB** | 2.23 s | ≈ 5 500 步/秒 | 0 | 0.2669 → 0.2187 |
| 2048 | **7030 MiB** | 3.09–3.38 s | ≈ 14 500–15 900 步/秒 | 0 | 0.2393 |
| **4096** | **11578 MiB** | 3.89–4.32 s | ≈ 22 700–25 300 步/秒 | 0 | 0.2306 |

三轮都是 30 轮全部跑完；`foot_sep_mean` 全部落在数据集直线走区间内
（实测中位 0.2140 / p1 0.1455）→ 步宽没有塌，退路 body 选得对。
4096 那轮 GPU 利用率 **69%** ⇒ **瓶颈是 CPU（Core Usage Quota 14），不是显存**。

### 11.2 显存拟合与 `num_envs` 决策

三点最小二乘：

$$\text{显存} \approx \underbrace{2843\ \text{MiB}}_{\text{固定开销（Kit/模型/判别器）}} + 2.117\ \text{MiB} \times N_{\text{envs}}$$

（只用前两点是 `3003 + 1.966N`；拿它预测 4096 得 11.0 GB，实测 11.3 GB，偏差 4.7%。）

⚠️ **教训：规模决策必须至少两个点。** 只看 512 那一个点做线性外推会得到
「4096 环境需要 32 GB，24 GB 装不下」—— 因为**固定开销占了 512 那点的 75%**，
把固定开销错算成了「每环境成本」。

**决策：正式训练用 `num_envs = 4096`**（= 镜像推荐的规模）：显存 11.6 GB / 24 GB，
留 52% 余量；且 GPU 利用率已到 69%，再放大只会撞 CPU 配额。
按 ~4 s/轮算，`G1OmniLocoPPORunnerCfg` 默认的 **5000 轮 ≈ 5.6 小时**。

### 11.4 ⭐ 训练早期的「假失败」与 A/B 对照（重要方法论）

先跑了 300 轮做健康检查，迭代 40 时的读数是：

```
Mean reward: -4.12                ← 横盘不升
Mean episode length: 6.96         ← 只有 0.14 秒！
Episode_Termination/base_contact: 1.0000   ← 所有回合都因躯干触地终止
Episode_Termination/time_out:     0.0000   ← 一个超时都没有
```

看上去像「环境/配置错了」。**但同参数的 A/B 对照把这个判断推翻了**：

```bash
# 跑学长原版任务（纯 PPO + 原奖励 + 原 623 维观测），参数完全一致
bash /root/cloud_run_bg.sh 'bash /root/cloud_ab_baseline.sh 4096 50 cuda:0' /root/ab_driver.log
```

| 迭代 | 原版 OmniLoco 回合长度 / reward | 我们的 AMP 回合长度 / reward |
|---|---|---|
| 0 | 8.06 / −4.54 | 4.40 / −4.25 |
| 10 | 19.27 / −5.45 | 13.09 / −4.97 |
| 20 | 6.49 / −4.36 | 6.07 / −4.41 |
| 30 | 4.03 / −4.18 | 3.83 / −4.12 |
| 40 | — | 6.96 / −4.12 |

**两条曲线几乎重合** ⇒ 我们的环境与 AMP 接线**没有坏**，40~50 轮本来就不够
（原版 `max_iterations = 5000`）。工具：`instinct/tools/cloud_ab_baseline.sh`。

⭐ **方法论：训练早期「看上去完全没学」时，先做同参数 A/B 对照，再怀疑自己的改动。**

读数本身的陷阱（同一件事的另一面）：

- `Mean reward ≈ −4` 几乎**全部**来自终止惩罚：`termination_penalty` 归一化后 −0.2
  → episodic sum ≈ **−4.0**。而所有跟踪项归一化后只有 0.003~0.01 量级 ——
  因为回合只有 7 步，分母却是 `max_episode_length_s = 20`。
  **回合越短，`Episode_Reward/*` 越小，越不能据此判断奖励设计有问题。**
- `Episode_Reward/feet_lateral_width ≈ −0.0001` 同理：只能说明这几步里 `sep > 0.145`，
  **不能**据此认为 `sep_floor` 标定正确（真证据是 `Step/foot_sep_mean`）。
- 判别器 `quad` 公式已核对源码（`wasabi.py`）：`clamp(1 − 0.25·(disc − 1)², min=0)`，
  `disc` 是**线性层裸输出**（非 sigmoid）。实测反推一致：
  `discriminator_reward = 0.25(r_amp − r_task/c)`，当 `r_amp=0.29`、`r_task=−0.96`、`c=0.1`
  → `0.25×(0.29+9.6) = 2.47`，与打印的 2.48 吻合 ⇒ **换算管线是对的**。

### 11.5 最终训练配置

```bash
bash /root/cloud_run_bg.sh 'bash /root/cloud_train_amp_walk.sh 4096 5000 cuda:0' /root/train_full_driver.log
```

`num_envs=4096`、`max_iterations=5000`（继承 `G1OmniLocoPPORunnerCfg`）、
`save_interval=500`（可中途看行为，也可 `--resume` 续训）。预计 ≈ 5.6 小时。

途中建议在迭代 500 / 1000 / 2000 处抽查这几个量（**不要在几十轮时就下结论**）：

| 指标 | 期望 | 不健康的含义 |
|---|---|---|
| `Mean episode length` | 明显上升（几十→几百→上千） | 停滞在个位数 → 真的学不动 |
| `Episode_Termination/base_contact` | 下降到 0.1 以下 | 恒为 1.0 → 站都站不住 |
| `Train/mean_reward_0` | 上升（进 TensorBoard 看，控制台不打印） | 不升 → 奖励符号/混合公式错 |
| `Step/foot_sep_mean` | 稳在 **0.15–0.30** | 掉向 0 或变负 → 步宽惩罚过猛 |
| `Step/discriminator_reward` | 缓降但**不为 0** | 迅速为 0 → 降 `discriminator_reward_coef` |

---

## 12. 云端路径速查 + 训练监控（2026-09-25）

### 12.1 关键路径

| 用途 | 路径 |
|---|---|
| 云端登录 | `root@183.147.142.40:30528`（凭据在本地 `~/.llp_cloud_credentials`，chmod 600，**不入库**）|
| 代码根 | `/root/gpufree-data/instinct_xu` |
| 环境变量脚本 | `/root/.llp_env.sh`（`source` 它即可拿到 PATH / `PYTHONPATH`）|
| 云端 python | `/opt/conda/envs/isaaclab/bin/python`（非交互 SSH **PATH 里没有 `python`**）|
| 训练日志目录（TensorBoard）| `/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk/<时间戳>_<run_name>/` |
| 曲线图输出（云端）| `/root/gpufree-data/instinct_xu/logs/instinct_rl/plots/` |
| 曲线图（本地拉回）| `instinct/plots/`（工作区内，VS Code 直接看；已加 .gitignore）|
| 本地工具 | `instinct/tools/`（SSH 助手 / 冒烟 / 试跑 / 训练 / 绘图 / 刷新）|
| 本地 AMP_mjlab（参考实现）| `/home/leo/rsl_project/AMP_mjlab`（绘图脚本蓝本在 `tools/plot_rewards.py`）|

### 12.2 训练监控工作流（对应 AMP_mjlab 那套）

```bash
# 1) 把绘图脚本传上云端 + 自检 tag（改过 plot_training.py 后跑一次）
bash instinct/tools/upload_plot_tool.sh

# 2) 出图并拉回本地（一次性）
bash instinct/tools/refresh_plots.sh once

# 3) 后台循环：每 600 s 自动重画 + 拉回
bash instinct/tools/refresh_plots.sh          # 启动
bash instinct/tools/refresh_plots.sh stop     # 停止
```

产物（`instinct/plots/`，全部由云端生成后 scp 拉回）：

| 文件 | 内容 |
|---|---|
| `01_overview.png` | 平均回报 / 回合长度 / AMP 总损失 / 动作噪声 std |
| `02_reward_terms.png` | 21 项 `Episode_Reward/*` 逐项收敛（缺项自动过滤） |
| `03_amp_diagnostics.png` | 判别器 4 个损失项 + `Step/discriminator_reward` + `amp_style_reward` |
| `04_foot_diagnostics.png` | ⭐ 步宽/交叉专项（**本次迁移最关键的设计风险点**） |
| `05_terminations_perf.png` | 终止方式 / FPS / 速度跟踪误差 |
| `06_amp_task_split.png` | 最终奖励里 task 与 AMP 风格各占多少 |
| `digest_latest.txt` | ⭐ 文本摘要（不开 SSH 就能看健康度） |

`plot_training.py` 支持 `--list-tags`：打印该 run 里**真实存在**的全部 scalar tag
（我们这轮是 **210 个**）。写新图前先跑它，别猜 tag 名。

### 12.3 ⚠️ 一个不能只看均值的设计陷阱（本仓库实测）

`04_foot_diagnostics.png` 第一次出图就抓到：迭代 260 时 `foot_sep_mean` 从 0.27
一路涨到 **0.41 并平台化** —— 远超数据集中位 0.214、也超出预期带 0.15~0.30，
机器人形成了很宽的「牛仔站姿」。

根因：`feet_lateral_width` 只惩罚**窄**步（`clamp(sep_floor − sep, 0)`），
对**宽**步**没有任何约束**；而 AMP 风格权重只有 $(1-\lambda)c = 0.025$，
不足以把站姿拉回参考分布。

→ 这个量必须**专门画出来看**：它在控制台只以 `Mean step foot_sep_mean` 出现一行，
  训练早期还容易误读成「正常」。**均值本身没越界检查，图上才看得见趋势。**

---

## 13. 本地 MuJoCo 回放训练策略（验证闭环，2026-09-25）

本地 env 有 `mujoco 3.13.0` / `onnxruntime 1.30.0` / `imageio`，**不需要云 GPU 就能回放策略**。

```bash
# ① 云端导出（单环境！）→ 得到 actor.onnx + policy_normalizer.npz
bash /root/cloud_eval_policy.sh <run_dir> model_5000.pt 500   # 内含录像 + 导出两步

# ② 拉回本地后自检 + 回放
python instinct/tools/g1_mujoco_play.py --selftest-obs
python instinct/tools/g1_mujoco_play.py --policy instinct/eval/actor_model_5000.onnx \
       --duration 15 --cmd 0.5 0 0 --no-view --record instinct/eval/out.mp4
```

### 13.1 ⭐ 三个必须做对的约定（每个都会让机器人**秒摔**）

| # | 坑 | 后果 / 正确做法 |
|---|---|---|
| 1 | **观测归一化没加载**（最主要） | `OnPolicyRunner.export_as_onnx` 是**先归一化 obs、再把归一化器单独存成 `exported/policy_normalizer.npz`**，而 ONNX 的输入是**归一化后**的观测。直接喂原始观测 = OOD → 标称站姿就输出 ±3.8 rad 疯狂动作、1.6 s 摔倒；加载后同一策略稳定站 **15 s+**。公式 `(x − mean)/(std + eps)` |
| 2 | **`mj_resetDataKeyframe` 后缺 `mj_forward`** | `data.xpos` 未初始化 → 首帧 `height_scan` 拿到 0.5（应 0.32）→ 首帧观测错 → 因 `last_action` 是无裁剪反馈，**指数发散到 NaN** |
| 3 | **动作尺度用错约定** | 本仓库是 `ACTION_SCALE = 1.0`（`target = default_joint_pos + 1.0·action`，云端实测 `action_term._scale == 1.0`）；不要用 `beyondmimic` 的 per-joint `0.25·effort/kp`（小 2~13 倍） |

### 13.2 关节序一共有三套（别混）

| 序 | 前 3 项 | 用途 |
|---|---|---|
| **articulation 序 == OMNI_LOCO 序** | shoulder_pitch(L,R), waist_pitch … | **动作**、`last_action`、**观测里的 `joint_pos_rel`/`joint_vel`**（`SceneEntityCfg("robot")` 默认 → `joint_ids=slice(None)`）|
| MuJoCo MJCF / 执行器序 | 腰 → 左腿 → 右腿 … | MuJoCo `qpos` / `ctrl` |
| 训练 npz 的 `joint_names` 序 | 左腿(6) → 右腿(6) → 腰(3) → 臂(7) | AMP 参考动作 |

**云端实测确认**：`robot.joint_names == action_term._joint_names == OMNI_LOCO_ORDER`
⇒ articulation 序就是肩优先交错序（工具 `instinct/tools/cloud_probe_joint_order.py`）。

### 13.3 云端口径核对（都是读源码/实测，不是推断）

- `height_scan = sensor_z − hit_z − 0.5`，clip(−1,1) → **平地精确退化为常数 `torso_z − 0.5`**（torso_z=0.82 ⇒ 0.32）
- `projected_gravity = R^T·(0,0,−1)` → 直立时 **(0,0,−1)**（旧 `ObsBuilder` 算成了 `R^T·ẑ`，符号反，已修）
- `base_ang_vel` / `joint_vel` **无额外 scale**（原始 rad/s）
- MuJoCo 侧 `MJCF 执行器序 == 硬编码 MUJOCO_ORDER` ✓、`keyframe home == INIT_JOINT_POS` 逐项一致 ✓、root 高度 0.82 ✓

### 13.4 ⚠️ 曾怀疑、**已查实排除**的一项

一度怀疑「npz 是腿优先序、而 motion_reference 不做按名重排」→ AMP 判别器会比较错位向量。
**读加载器源码后确认是虚警**：`motion_reference/motion_files/amass_motion.py:580-588`
按名字把 npz 关节列重排到 `isaac_joint_names`（articulation 序），缺关节还会报错。
⇒ 转换器产出「源序 + 诚实的 `joint_names`」正是它要的契约，**无需重排、无需重训**。

### 13.5 注意：策略动作幅度本身就很大（不是回放的 bug）

训练侧 `Episode_Monitor/tracking_action/std ≈ 2.93`、**`abs_max ≈ 29.7`**（且随训练增大）。
配合 `scale=1.0` ⇒ 策略把大部分目标角**推到关节限位**（bang-bang），
`action_rate_l2 = −0.95` 因此成为最大代价项。
学长的 OmniLoco 基线同样 `scale=1.0` + `init_noise_std=1.0`，属**有意对齐**，不是错误。

### 11.3 观测/统计的 5 个坑（共同点是**都会骗人**）

| 坑 | 现象 | 对策 |
|---|---|---|
| **Python 块缓冲**（最耗时间） | `train.sh ... >> log 2>&1` 时 stdout 默认块缓冲，日志**滞后数分钟**才落盘 → 反复轮询看不到进展，误判"训练卡住" | `export PYTHONUNBUFFERED=1`（`cloud_scale_probe.sh` / `cloud_train_amp_walk.sh` 都已加）。修后实测 20 s 内落盘 113 行 |
| **摘要块自污染** | `{ echo …; grep -c 'X' "$LOG"; } >> "$LOG"` 中块内命令**边跑边写**：前面 `grep -E 'X' "$LOG" \| head -3` 刚写进去的行会被后面的 `grep -c 'X' "$LOG"` 数到（实测 **3 张表被数成 9**）| 摘要写**独立文件**，统计在污染前算完 |
| **「轮数」口径错误**（会直接误导规模决策） | runner **每 10 轮才打一次表**，所以 `grep -c 'Iteration time'` 数的是**表格数不是轮数**；且 `total_timesteps` **只在打表时累加**，比真实步数**少 10 倍**。上一轮据此得出「只跑了 3 轮」的结论是错的（实际 30 轮全跑完）| 稳态吞吐统一用**最后一次 `Iteration time`**（它确实是"一轮"的墙钟时间）；标签写明"表格数（非轮数）" |
| **GPU 采样行带时间戳前缀** | 行是 `HH:MM:SS 7030 MiB, 50 %` → `awk -F', '` 取 `$1` 得 `00:23:53 7030 MiB`，`$1+0` 恒为 0 → 峰值打印成空 | `grep -oE '[0-9]+ MiB' \| awk '{…}'` |
| **框架自身的提示文字被当成异常** | runner 会打印 ``NOTE: you may see a bunch of `NaN or Inf found in input tensor` once …``，含 "NaN" → 异常 grep 误报 | 排掉 `NOTE:` 行；异常检索只留 `Traceback\|KeyError\|RuntimeError\|out of memory\|CUDA error` |

另：`pgrep -f <pattern>` / `pkill -f <pattern>` 会匹配到**自己也含该字符串**的进程
（包括你当前这条 SSH 命令行），必须用 `[s]cale_probe.sh` 括号技巧。
这条坑实测把**我自己的 SSH 会话**kill 掉两次（现象：命令毫无输出、后续步骤压根没执行）。

| 坑 | 现象 | 对策 |
|---|---|---|
| **Kit 不响应普通 SIGTERM** | `pkill -f <训练>` 后 `nvidia-smi` 仍占着 ~5.9 GB，`pgrep` 里进程**还活着** → 和新任务抢显存、存在 OOM 风险 | 用 `pkill -9`（SIGKILL）；杀完务必 `pgrep -af train.py` + `nvidia-smi` 各确认一遍 |

---

## 14. 本地键盘遥控（WASD/QE）平地技能验证（2026-09-25）

### 14.1 结论：本地**已支持**，且是"开窗交互 + 离屏录像"两种都在

依赖逐项审计（`pip list` + `ctypes.util.find_library`，实测）：

| 需要的东西 | 本地状态 | 用途 |
|---|---|---|
| `mujoco 3.13.0` | ✅ | 物理步进（**纯 CPU，不占显存**） |
| `glfw 2.10.2` + `libglfw.so.3` | ✅ | 交互式开窗 |
| `OpenGL 3.1.10`(PyOpenGL) + `libGL.so.1` | ✅ | 窗口渲染 |
| `DISPLAY=:1` + `/tmp/.X11-unix/X1` | ✅ | X11 会话存在（`XDG_SESSION_TYPE=x11`） |
| `libEGL.so.1` | ✅ | 离屏渲染（`--record` 用） |
| `imageio 2.37.4` + `imageio_ffmpeg 0.6.0` | ✅ | 直出 mp4 |
| `onnxruntime 1.30.0` | ✅ | 跑策略 |

### 14.2 用法

```bash
source /home/leo/miniconda3/etc/profile.d/conda.sh && conda activate instinct

# 键盘遥控（--duration 0 = 无限运行，关窗口退出）
python instinct/tools/g1_mujoco_play.py \
  --policy instinct/eval/actor_model_5000.onnx --duration 0

# 调大每次按键的增量（默认 0.1）
python instinct/tools/g1_mujoco_play.py --policy ... --duration 0 --cmd-step 0.2

# 关掉遥控（固定指令）/ 不退窗口地录像
python instinct/tools/g1_mujoco_play.py --policy ... --no-teleop --duration 15
python instinct/tools/g1_mujoco_play.py --policy ... --no-view --record out.mp4
```

**开窗后必须先点一下窗口让它获得焦点**，否则按键不会送到 GLFW。

| 按键 | 作用 |
|---|---|
| `W` / `S` | `vx` += / −= step（前进 / 后退）|
| `A` / `D` | `vy` += / −= step（左移 / 右移）|
| `Q` / `E` | `wz` += / −= step（左转 / 右转）|
| `[` / `]` | 调小 / 调大 step（预设 `0.02/0.05/0.1/0.2/0.4/0.8`）|
| `空格` / `X` | 指令清零（原地站立）|
| `R` | 复位到初始 keyframe |

窗口左上角 HUD 实时显示 `vx/vy/wz` 与 step，左下角显示键位表与训练范围。

### 14.3 实现的 4 个关键取舍（都是踩过的点）

1. **每次按键 = 一次指令增量**，而不是"按住才动"。
   `launch_passive(key_callback=...)` 的签名只声明 `Callable[[int], None]`，
   **没有任何文档保证会同时回传 RELEASE 事件**（实际转发在 C++ 扩展
   `mujoco._simulate` 里）。增量模式在"只给 PRESS"和"PRESS+RELEASE 都给"
   两种情况下**都正确**，所以选了它。
2. ⭐ **两种键码语义都注册**。`glfw.KEY_W == 87 == ord('W')`（字母/空格/括号的
   GLFW 码恰好等于**大写** ASCII），但**小写**字符码是 `ord('w') == 119`。
   由于回传的到底是 GLFW 键码还是字符码没有官方保证，键位表里把
   **GLFW 码 + 大写 ASCII + 小写 ASCII 三套都填上**，避免"按了没反应"。
3. ⭐ **指令必须夹在训练范围内**。`CMD_LIMITS = ((-1.0,1.2), (-0.5,0.5), (-0.8,0.8))`
   = `AMP_WALK_COMMAND_RANGES`（`amp_walk_env_cfg.py:92`）。
   超范围的指令属于**分布外**，摔了**不代表策略差**，只是没训过 —— 不夹住会得出错误结论。
4. ⭐ **`R` 复位必须同时清零 `last_action`**。否则策略会把"上一段生命末期的大动作"
   当作第一帧输入 → 因为 `last_action` 是**无裁剪反馈**，直接指数发散到 NaN（§13 坑 2
   的同一个机制）。复位逻辑统一收进 `reset_state()`，启动时和 `R` 键**共用一份**。

### 14.4 能力边界（平地足够，越障不行）

- ⭐ **平地回放是忠实的**：`height_scan` 在脚本里就是平地精确解
  `np.full(527, torso_z − 0.5)`，与 IsaacLab 的 527 条真射线在**平地上逐位相同**。
  实测 `height_scan[0] = 0.3200`（torso_z=0.82）✅
- ❌ **越障无法本地验证**：需要 MJCF 里有地形 + 527 条**真实射线**扫描，
  当前两者都没有。能力边界写在脚本 docstring 里，不装作能测。
- 本项目当前策略也**只在 Plane 上训练**（`G1AmpWalkPlane…`），所以平地验证**正好是对口**的。

### 14.5 验证记录（2026-09-25，全过）

| 验证 | 手段 | 结果 |
|---|---|---|
| 键位表（两套键码语义） | `instinct/tools/test_keyboard_teleop.py` | ✅ 全过（含 `ord('w')=119` 与 `glfw.KEY_W=87` 双命中） |
| 指令演进 / 夹范围 / step 预设 / 复位标志只生效一次 / 未知键安全忽略 | 同上（**30 项**） | ✅ `[KEYBOARD-TELEOP PASS]` |
| `reset_state` + `mj_forward` | 同上（真实 MJCF，故意弄脏 qpos/qvel 再复位） | ✅ root z=0.82、qvel 归零、`xpos` 已初始化、幂等 |
| 重构后的 reset 路径没破坏观测 | `--selftest-obs`（**不开窗口**） | ✅ 626 维、`height_scan=0.3200`、`projected_gravity=(0,0,-1)` |
| 主循环改动没破坏闭环 | `--no-view --duration 2 --cmd 0.5 0 0` | ✅ 走完 100 步，高度稳在 0.79–0.82，`\|v\|≈0.42~0.50` |

> 单测是**离线**的（不开窗口、不跑策略），所以可以随时跑；
> 但「按键到底有没有送进 GLFW」这一点**只能人工开窗确认** —— 留给你。

### 14.6 已知的非阻塞现象：动作幅度很大

回放日志里 `[act] min=-8.155 max=16.988` —— 这不是 bug：策略把大部分目标角**推到关节限位**
（bang-bang），与训练侧 `tracking_action/abs_max ≈ 29.7`、以及 §13.5 的结论一致。
`scale=1.0` + `init_noise_std=1.0`（绝对 σ）是它的根源，见 `PLAN_amp_walk.md` §9。

### 14.7 两个实测修复：「看不到地面」与「播放太快」（2026-09-25）

#### ① 看不到地面 —— 根因是 geom group（不是 MJCF 缺地面）

实测证据：

```
mjvOption.geomgroup 默认 = [1, 1, 1, 0, 0, 0]     ← group 3 根本不渲染
MJCF <default>: <geom group="3" contype="1" conaffinity="1"/>
floor geom: id=0 group=3 contype=1 conaffinity=1
各 group geom 数: {2: 35, 3: 30}                  ← floor + 29 个碰撞胶囊都在 group 3
```

⇒ **地面存在且碰撞正常**（`contype=1`，所以机器人站得住），只是 `<default>` 把它的
渲染组设成了 **3**，而 **MuJoCo 默认不渲染 group 3**（该组约定用于"仅碰撞/非可视"几何）。

**修法**（只改**加载时的 XML 文本副本**，不碰磁盘文件 —— 那文件由 `build_g1_mjcf.py`
生成，手改会被覆盖）：

1. 给 floor 追加 `group="2"`（可见）；
2. 注入一个 **棋盘材质** `<texture builtin="checker">` + `<material texrepeat="8 8">`。

⭐ 第 2 步不是"顺手美化"：**纯色地面看不出机器人有没有位移/打滑**，
而这两件事恰好是评估行走策略最关心的。

验证（补丁**只影响渲染**）：

| 检查 | 结果 |
|---|---|
| floor 组 / 材质 | ✅ `group=2`、`material=llp_groundplane` |
| floor 碰撞 | ✅ `contype=1` **未变** |
| geom 分布 | ✅ group 2: 35→36、group 3: 30→29（floor 搬了一格） |
| 物理量 | ✅ `timestep=0.002`、`nq=36`、`nu=29` **全未变** |
| 观测 | ✅ `--selftest-obs` 仍 626 维、`height_scan=0.3200` |

⚠️ 顺带纠正一个**我自己的算术错误**：`grep -c '<geom'` 得 66，但其中一行是
`<default>` 里的**原型 geom**（不是真实几何体），真实 geom 是 **65** 个（35+30）。

#### ② 播放太快 —— 根因是**完全没有实时节拍**（不是帧率不匹配）

原代码 `for step in range(n_steps): ... mj_step × decimation` **中间没有任何 sleep**，
所以仿真以 CPU 全速推进。客观测量：

| 模式 | 仿真时间 / 墙上时间 | 实际倍速 |
|---|---|---|
| **修复前**（无节拍） | 4.00 s / **0.12 s** | **33.69× 实时** ← 就是"太快" |
| **修复后**（默认实时） | 4.00 s / 4.00 s | **1.00×** ✅ |
| `--speed 0.5` | 2.00 s / 4.00 s | **0.50×** 慢放 ✅ |

**修法**：按墙上时钟对齐 —— 每个策略步应占 `wall_dt = sim.dt × decimation / speed`
（默认 = 0.002 × 10 / 1.0 = **20 ms** = 50 Hz），用 `time.perf_counter()` 基准累加、
`time.sleep()` 补足；**落后超过 5 步就重置基准**，避免"追赶式暴走"。

新增开关：

| 开关 | 作用 |
|---|---|
| `--speed`（默认 1.0） | 实时倍速；`0.5` 慢放看清细节 |
| `--no-realtime` | 关掉节拍，全速跑（扫参数时用；帧率不匹配的问题在**录像**里本来就没有——录像是按 `fps=policy_hz` 写帧的） |
| `--cam-distance`（默认 3.0） | 初始相机距离 |
| `--show-collision` | 显示 group 3 的碰撞胶囊（排查接触用） |
| `--no-checkered-floor` | 不给地面补棋盘（保持 MJCF 原样） |

另外补了**初始相机** `lookat=(0,0,0.5), distance=3.0, azimuth=145, elevation=-18`，
否则默认视角偏高偏远，第一眼容易看不到地面和脚。

> 参考项目 `~/g1_ws/unitree_rl_mjlab` 也查了：它的场景 XML
> （`src/assets/robots/unitree_g1/xmls/scene_g1.xml:556`）的 floor 同样不设 group ——
> 差别在于**它的 `<default>` 没给 geom 设 `group="3"`**，所以 floor 默认落在
> group 0（可见）。它的实时节拍在 C++ 仿真器里（`simulate/src`），Python 侧只有策略配置。

### 14.8 高亮 height_scan 扫描区域（2026-09-25）

**默认就开**（`--no-show-scan` 可关）。开窗后能看到一个跟着机器人走的橙色矩形框。

#### 几何来源（全是源码，不是估的）

`instinct_xu` 的 `tasks/locomotion/config/g1/omni_loco_env_cfg.py:86`：

```python
height_scanner = MultiMeshRayCasterCfg(
    prim_path = "{ENV_REGEX_NS}/Robot/torso_link",     # 挂在 torso 上
    offset    = OffsetCfg(pos=(0.0, 0.0, 20.0)),       # 从 torso 上方 20 m 垂直向下打
    ray_alignment = "yaw",                              # ⭐ 只跟航向，不随俯仰/横滚倾斜
    pattern_cfg = GridPatternCfg(resolution=0.1, size=(3.0, 1.6)),
    mesh_prim_paths = ["/World/ground"],
)
```

**自洽校验**：`3.0/0.1 + 1 = 31`、`1.6/0.1 + 1 = 17` → `31 × 17 = 527`
✓ = 契约里的 `AMP_WALK_SCAN_RAYS`（`amp_walk_env_cfg.py:87`）。

⇒ **平地时扫描区域就是地面上一个 3.0（前后）× 1.6（左右）m 的矩形**，
中心随机器人 `(x, y)` 平移、随航向（yaw）旋转，**与躯干高度无关**。

#### 画了什么

| 元素 | 含义 |
|---|---|
| **亮橙边框** | 扫描区域的 4 条边 |
| 淡橙内部线 | 每 0.5 m 一条网格（不把 31×17 条射线全画出来遮住机器人）|
| 前边缘外伸的短段 | 提示「这一侧是前方」（`GridPatternCfg` 的 x 是前后方向）|

颜色/粗细常量：`SCAN_COLOR_EDGE` / `SCAN_COLOR_GRID` / `SCAN_LINE_RADIUS = 0.012 m`。

#### 实现要点

1. **用 MuJoCo 的 user geom 叠加层**（`Handle.user_scn`），每帧 `ngeom = 0` 后重填，
   在 `viewer_ctx.lock()` 保护下写、再 `sync()` —— 不碰物理，也不增加碰撞体。
2. 用 `mjv_initGeom(...)` 先设好 `rgba`，再 `mjv_connector(...)` 连两点成一条线。
   （`mjv_connector` 的文档明确要求先调 `mjv_initGeom`。）
3. ⭐ **只取 yaw**：`yaw_frame_basis()` 把机体 x 轴投到水平面再正交化。
   用完整四元数画的话，框会跟着躯干 roll/pitch 翻转 —— **那就把几何画错了**，
   因为 `ray_alignment="yaw"` 的语义就是「网格永远水平扣在地面上」。
4. 机体 x 轴接近竖直（极端姿态）时退回单位阵，不除零。

#### 验证（离线，不需要开窗）

`instinct/tools/test_scan_visual.py` —— **17 项全过**：

| 检查 | 结果 |
|---|---|
| 线段数 = 12 且真的写进 `mjvScene` | ✅ |
| 原地朝向 +x：`x∈[-1.5, 1.8]`、`y∈[-0.8, 0.8]`、`z≡0.012` | ✅ |
| 平移到 (2,3) → 框中心跟到 (2,3) | ✅ |
| **躯干高度 0.82→1.30，框不变**（与身高无关）| ✅ |
| 转向 90° → 跨度互换（`x`=1.6 左右、`y`=3.3 前后+标记）| ✅ |
| ⭐ **俯仰/横滚 30° → 框仍贴地**（yaw 对齐语义）| ✅ |
| 机头朝天（极端姿态）不崩 | ✅ |

> ⚠️ 写这个单测时踩了一下：一开始想从画好的 `mjvGeom` 反推端点，结果 7 项失败 ——
> 因为 MuJoCo 的 **capsule 长轴是局部 `z` 轴**（不是 x），而且总长还要算上两倍半径。
> 修法是把几何抽成纯函数 `scan_footprint_segments()` 直接测，
> **而不是去凑测试的换算公式**。

