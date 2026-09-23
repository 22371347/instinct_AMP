# instinct_xu：G1 三实验训练包

把整个 `instinct_xu` 文件夹交给使用者即可。下面给出**从零安装 Isaac Sim、Isaac Lab、原版 InstinctLab/instinct_rl，再安装本文件夹**的完整顺序；已装好兼容环境的人可复用现有环境。此包包含训练代码、G1 URDF/mesh 和 Vault motion 数据，不含模拟器、训练日志或策略 checkpoint。

| 任务 | Gym ID | 新日志目录 |
| --- | --- | --- |
| 平地 OmniLoco 教师 | `Instinct-Unified-OmniLoco-G1-v0` | `logs/instinct_rl/g1_unified_omni_loco/` |
| 感知 Vault 教师 | `Instinct-Perceptive-Vault-G1-v0` | `logs/instinct_rl/g1_perceptive_vault_shadowing/` |
| Vault 视觉学生蒸馏 | `Instinct-Perceptive-Vault-VaeDistill-G1-v0` | `logs/instinct_rl/g1_perceptive_vault_vae_distill/` |

目录叫 `instinct_xu`，Python 导入名仍是 `instinctlab` 和 `instinct_rl`，以匹配任务注册和 checkpoint 格式。学生依赖先训练出的 Vault 教师；OmniLoco 与 Vault 教师可以分别从零训练。

## 1. 放置收到的 instinct_xu 文件夹

推荐把收到的文件夹放在自己的用户目录中，与 Isaac Lab、原版 InstinctLab 并列。假设当前目录里已有收到的 `instinct_xu/`，执行：

```bash
mkdir -p "$HOME/robot_projects"
cp -a ./instinct_xu "$HOME/robot_projects/"
cd "$HOME/robot_projects/instinct_xu"
export INSTINCT_XU_ROOT="$(pwd -P)"
```

如果文件夹已经在目标位置，直接 `cd` 进去并执行最后一行即可。不要把它放进 Isaac Lab 的 `source/` 目录。后续命令在同一终端运行；换终端后，重新进入 `instinct_xu` 并设置 `INSTINCT_XU_ROOT`。`train.sh` 和 `play.sh` 会自动使用其所在目录的源码，训练输出也写在该目录内。

## 2. 准备 Isaac Sim / Isaac Lab 环境

本包采用 Python 3.11、Isaac Sim 5.1.0、PyTorch 2.7.0 + CUDA 12.8，以及 Isaac Lab 源码提交 `f73c33173801f5f8afea4142482e47b7710c2b75`。Isaac Lab 在该提交的 Python 包版本号显示为 `0.54.4`。有兼容环境时使用 **A**；全新机器使用 **B**。安装前可参照 [Isaac Sim 5.1 的 Python 安装说明](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html) 和 [Isaac Lab 安装说明](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/setup/installation/pip_installation.html) 检查驱动及系统要求。

### A. 已有兼容环境

激活你自己的环境，例如：

```bash
conda activate instinctlab
python -m pip show isaacsim isaaclab
```

如果环境名称不同，使用实际名称。要保留原有两个同名包的 editable 安装位置，可先记录：

```bash
BACKUP_DIR="$HOME/instinct_xu_env_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
python -m pip freeze > "$BACKUP_DIR/pip_freeze.txt"
python -m pip list --editable > "$BACKUP_DIR/editable_packages.txt"
python -m pip show instinctlab instinct_rl > "$BACKUP_DIR/original_packages.txt" || true
echo "$BACKUP_DIR"
```

复用现有环境时，若已经安装原版 InstinctLab/instinct_rl，直接跳到第 4 节；不要为了本包重装 Isaac Sim、Isaac Lab、Torch 或 NumPy。

### B. 全新 Linux x86_64 GPU 环境

以下步骤在同一终端执行；Conda 和 Git 需已安装。它们会在用户主目录另建 Isaac Lab 依赖仓库，`instinct_xu` 自身仍可放在任意位置。

```bash
conda create -n instinct_xu python=3.11 -y
conda activate instinct_xu
python -m pip install --upgrade pip
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

git clone --recurse-submodules https://github.com/isaac-sim/IsaacLab.git "$HOME/IsaacLab-instinct-xu"
git -C "$HOME/IsaacLab-instinct-xu" checkout f73c33173801f5f8afea4142482e47b7710c2b75
git -C "$HOME/IsaacLab-instinct-xu" submodule update --init --recursive
cd "$HOME/IsaacLab-instinct-xu"
./isaaclab.sh -i
cd "$INSTINCT_XU_ROOT"

python -m pip install 'setuptools<82' wheel toml psutil pytorch_kinematics joblib debugpy snakeviz \
  'trimesh[all]' scikit-learn opencv-python pyvista \
  tensorboardX tensorboard tabulate GitPython onnx onnxscript
```

如果 Isaac Sim 5.1 无法在目标机器启动，先按上面的官方说明解决 GPU 驱动或模拟器安装问题，再安装本包。

## 3. 安装原版 InstinctLab 和配套 instinct_rl

从零复现时，先取得[原版 InstinctLab fork](https://github.com/ChaoMingSanHua/InstinctLab)及其[配套 instinct_rl fork](https://github.com/ChaoMingSanHua/instinct_rl)。下面用固定提交建立两个独立目录；它们与收到的 `instinct_xu` 并列，互不覆盖：

```bash
UPSTREAM_ROOT="$HOME/robot_projects/instinct_upstream"
mkdir -p "$UPSTREAM_ROOT"
git clone --recurse-submodules https://github.com/ChaoMingSanHua/InstinctLab.git "$UPSTREAM_ROOT/InstinctLab"
git clone --recurse-submodules https://github.com/ChaoMingSanHua/instinct_rl.git "$UPSTREAM_ROOT/instinct_rl"
git -C "$UPSTREAM_ROOT/InstinctLab" checkout 8879c06d3f3ae376aa96988a58d2f7f3661ec12d
git -C "$UPSTREAM_ROOT/instinct_rl" checkout 5523e3847cfce578bfcee25a5334256d9cc53e0e
git -C "$UPSTREAM_ROOT/InstinctLab" submodule update --init --recursive
git -C "$UPSTREAM_ROOT/instinct_rl" submodule update --init --recursive

python -m pip install --no-deps --no-build-isolation \
  -e "$UPSTREAM_ROOT/instinct_rl" \
  -e "$UPSTREAM_ROOT/InstinctLab/source/instinctlab"
```

这一步安装的是原版项目。`instinct_xu` 的三个实验包含原版基础代码加本实验修改，因此**下一步必须再次安装随包的两份源码**，让同名 Python 包最终指向 `instinct_xu`。原版仓库保留在 `instinct_upstream/`，供对照和其他原版任务使用；运行本包训练入口时以随包源码为准。

## 4. 安装 instinct_xu 的两份 Python 源码

在已激活的 Conda 环境中，从 `instinct_xu` 根目录执行：

```bash
cd "$INSTINCT_XU_ROOT"
python -m pip install --no-deps --no-build-isolation \
  -e "$INSTINCT_XU_ROOT/source/instinct_rl" \
  -e "$INSTINCT_XU_ROOT/source/instinctlab"
python -m pip list --editable
python -m pip check
```

`pip list --editable` 应显示两行路径分别以 `instinct_xu/source/instinct_rl` 和 `instinct_xu/source/instinctlab` 结尾。`--no-deps` 防止这一步改动已准备好的 Isaac Sim / CUDA / Torch 环境。它会切换当前 Conda 环境中的两个同名 Python 包，但不会删除原来的源码文件。

检查实际导入位置和 GPU，无需启动 Isaac Sim：

```bash
python scripts/check_install.py
```

若脚本报告包指向文件夹外，请检查 `pip list --editable`、当前 Conda 环境及已有的 `PYTHONPATH`；直接 `cd` 到新目录不会改变 Python 的安装指向。这里不要单独 `import instinctlab` 做检查，因为任务代码应由训练脚本在 Isaac Sim 启动后导入。三个任务的 Gym 注册已随包提供，无需额外注册。

## 5. 在 GPU 上做小规模连通性测试

```bash
CUDA_VISIBLE_DEVICES=0 ./train.sh \
  --task Instinct-Perceptive-Vault-G1-v0 \
  --num_envs 64 --max_iterations 3 --device cuda:0 --headless \
  --run_name instinct_xu_install_smoke
```

这会检查 G1 资产、Vault motion、任务注册、仿真采样与 PPO 更新，并写入一个新的日志目录。它只是安装连通性测试，不用于评价策略质量。这里的 GPU 测试由接收者运行；包内未附带测试生成的权重。

## 6. 正式训练

在包根目录执行。任务默认并行环境数是 4096；按 GPU 显存调整 `--num_envs`。

```bash
# 平地 OmniLoco 教师：623 维 actor 输入、29 维动作。
CUDA_VISIBLE_DEVICES=0 ./train.sh \
  --task Instinct-Unified-OmniLoco-G1-v0 \
  --num_envs 4096 --device cuda:0 --headless

# Vault 特权教师：随包附有它需要的 G1 资源和动作 NPZ。
CUDA_VISIBLE_DEVICES=0 ./train.sh \
  --task Instinct-Perceptive-Vault-G1-v0 \
  --num_envs 4096 --device cuda:0 --headless
```

训练输出会自动保存在顶部表格列出的 `logs/instinct_rl/` 子目录。先训练好 Vault 教师，再训练视觉学生。列出教师 run：

```bash
find "$INSTINCT_XU_ROOT/logs/instinct_rl/g1_perceptive_vault_shadowing" \
  -mindepth 1 -maxdepth 1 -type d -print
```

把选定的 run 名写入 `VAULT_TEACHER_RUN`，该目录须包含 `model_*.pt` 和 `params/agent.yaml`：

```bash
VAULT_TEACHER_RUN="$INSTINCT_XU_ROOT/logs/instinct_rl/g1_perceptive_vault_shadowing/填入教师run名"
test -f "$VAULT_TEACHER_RUN/params/agent.yaml"
find "$VAULT_TEACHER_RUN" -maxdepth 1 -name 'model_*.pt' -print
export PERCEPTIVE_VAULT_TEACHER_LOGDIR="$VAULT_TEACHER_RUN"

CUDA_VISIBLE_DEVICES=0 ./train.sh \
  --task Instinct-Perceptive-Vault-VaeDistill-G1-v0 \
  --num_envs 4096 --device cuda:0 --headless --enable_cameras
```

蒸馏实现会读取教师目录中**编号最大的** `model_*.pt`。学生深度相机训练需要 `--enable_cameras`。目前的学生配置只蒸馏 Vault 教师；OmniLoco 是独立教师实验。由于本包按要求没有附带策略文件，收到包后可先从零训练两位教师，再训练学生。

## 7. 续训与播放

续训时 `--load_run` 接受**包含 checkpoint 的 run 目录**，`--checkpoint` 是该目录内的文件名。`--max_iterations` 表示从 checkpoint **额外训练**多少轮。可先查看自己生成的 checkpoint：

```bash
find "$INSTINCT_XU_ROOT/logs/instinct_rl" -type f -name 'model_*.pt' -print
```

以下用 OmniLoco 举例；Vault 教师和学生分别换成顶部表格中的任务 ID 与对应 run 目录。学生续训前仍须设置 `PERCEPTIVE_VAULT_TEACHER_LOGDIR`。

```bash
LOCO_RUN="$INSTINCT_XU_ROOT/logs/instinct_rl/g1_unified_omni_loco/填入要继续的run名"
LOCO_CHECKPOINT="model_4500.pt"  # 改为该目录中实际存在的文件名
test -f "$LOCO_RUN/$LOCO_CHECKPOINT"

CUDA_VISIBLE_DEVICES=0 ./train.sh \
  --task Instinct-Unified-OmniLoco-G1-v0 \
  --num_envs 4096 --device cuda:0 --headless \
  --resume --load_run "$LOCO_RUN" --checkpoint "$LOCO_CHECKPOINT" \
  --max_iterations 500

CUDA_VISIBLE_DEVICES=0 ./play.sh \
  --task Instinct-Unified-OmniLoco-G1-Play-v0 \
  --num_envs 1 --device cuda:0 \
  --load_run "$LOCO_RUN" --checkpoint "$LOCO_CHECKPOINT"
```

如果接收者另有兼容的教师 checkpoint，也可以把其 run 目录的**绝对路径**提供给 `--load_run` 或 `PERCEPTIVE_VAULT_TEACHER_LOGDIR`；本包本身不依赖任何固定的外部目录。

## 目录

```text
instinct_xu/
├── README.md
├── train.sh / play.sh
├── scripts/check_install.py
├── scripts/instinct_rl/       训练、播放和专用评估入口
├── source/instinctlab/       三项任务、共享环境代码、G1 资产与 Vault motion
├── source/instinct_rl/       PPO、VaePpoDistill、网络与 runner
└── logs/instinct_rl/        新训练输出，交付时为空
```

包内只注册三项训练任务及三项 Play 变体。原策略与日志按要求未打包；新机器上的 Isaac Sim GPU 初始化和训练效果需要通过第 5 节的运行验证。
