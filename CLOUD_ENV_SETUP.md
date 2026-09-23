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
| OS | Ubuntu 22.04 LTS x86_64 | Isaac Sim 5.1 官方支持矩阵 |
| Python | **3.11** | `instinct_xu/README.md` §2 |
| Isaac Sim（pip 包） | **`isaacsim[all,extscache]==5.1.0`** | `instinct_xu/README.md` §2 B |
| Isaac Sim 包索引 | `https://pypi.nvidia.com`（额外索引） | 同上 |
| PyTorch | **`torch==2.7.0`**（CUDA **12.8**） | 同上 |
| torchvision | **`torchvision==0.22.0`** | 同上 |
| PyTorch 包索引 | `https://download.pytorch.org/whl/cu128` | 同上 |
| **IsaacLab** | **源码提交 `f73c33173801f5f8afea4142482e47b7710c2b75`**（pip 显示版本号 **0.54.4**） | 同上 |
| 原版 InstinctLab | `ChaoMingSanHua/InstinctLab` @ `8879c06d3f3ae376aa96988a58d2f7f3661ec12d` | `instinct_xu/README.md` §3 |
| 原版 instinct_rl | `ChaoMingSanHua/instinct_rl` @ `5523e3847cfce578bfcee25a5334256d9cc53e0e` | 同上 |
| GPU 驱动 | ⚠️ **按官方文档确认最低版本**，不要用 470/525 这类老驱动 | [Isaac Sim 5.1 安装说明](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html) |
| 参考文档 | Isaac Lab pip 安装（**v2.3.1** 分支文档） | [链接](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/setup/installation/pip_installation.html) |

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

## 6. 建议的执行顺序（含并行）

```
[t0] 立刻：在云平台开始构建镜像 / 装环境   ← 长尾任务，今天就要启动
      └─ 下载 isaacsim[all]==5.1.0（数 GB）+ 编译 IsaacLab，通常 1~3 小时

[t1] 同时：本地继续写 ⑤ 奖励迁移、⑥ 算法接线（纯代码，不依赖云）

[t2] 云环境就绪 → 跑 §3 第 1~3 层验收
      └─ 若第 3 层冒烟挂掉：说明 ②③④ 某处假设错，就地修（代价小）
      └─ 若通过：本地代码「从没跑过」的状态被清除，后面就都是增量风险了

[t3] ⑦ 正式小规模训练（先 512 env，确认显存后再上 4096）
```
