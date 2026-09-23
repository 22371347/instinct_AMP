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
| 3 | `--task Instinct-Amp-Walk-G1-v0 --num_envs 16 --max_iterations 2` | 端到端：G1 资产 + 高程图 527 射线 + 626 维观测 + 方向分档指令 + `motion_reference`/`robot_reference` 场景 + `amp_policy`/`amp_reference` 同构 |
| 4 | 契约校验器打印 `[AMP-WALK PASS]` | 说明动作空间/时序/执行器/命令范围全部符合预期 |

**已知会在云端首次暴露、本机无法验证的假设**（按风险排序）：

1. `isaaclab.envs.mdp.UniformVelocityCommand`（类）是否可导入 —— 仓库内只有 `...Cfg` 的先例
2. 该类的内部缓冲名确实叫 `vel_command_b`
3. `robot_reference` + `motion_reference` 传感器在 `G1OmniLocoSceneCfg` 基础上叠加后能否正常构建
4. `motion_reference` 的关节顺序与机器人是否逐一对齐（两侧 AMP 状态逐项同构依赖它）
5. `isaaclab.envs.mdp.joint_vel_rel` 存在

> 这 5 条每一条都已在本地用**静态契约检查**做过能做的部分
> （`instinct/tools/check_amp_walk_task.py`，9 类检查 A–I 全过），
> 但静态检查不能替代真实运行。**建议云环境一通就先跑第 3 层冒烟**，
> 把 ②③④ 三步积累的「未验证断言」一次性清掉，再继续写 ⑤⑥。

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

[t1] 同时：本地继续写 ⑤ 奖励迁移、⑥ 算法接线（纯代码，不依赖云）

[t2] 云环境就绪 → 跑 §3 第 1~3 层验收
      └─ 若第 3 层冒烟挂掉：说明 ②③④ 某处假设错，就地修（代价小）
      └─ 若通过：本地代码「从没跑过」的状态被清除，后面就都是增量风险了

[t3] ⑦ 正式小规模训练（先 512 env，确认显存后再上 4096）
```
