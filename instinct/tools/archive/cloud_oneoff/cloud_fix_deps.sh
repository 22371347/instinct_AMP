#!/usr/bin/env bash
# =============================================================================
# 补齐 README §2 B 列出的额外依赖（镜像里漏装的那些）
# =============================================================================
# 背景：`instinctlab.motion_reference.motion_reference_manager` 顶层 `import
# pytorch_kinematics as pk`；镜像里没装 → 连配置模块都 import 不了，
# 更不用说建环境。它在 **GPU 模式下同样会挡冒烟**，所以是必须先修的缺口。
#
# ⚠️ 铁律：**不能动已装好的 Isaac Sim / Torch / NumPy**（README 明确警告）。
#    所以：装前记录版本 → 全部 `--no-deps` → 装后再比对。
# =============================================================================
set -uo pipefail
PY=/opt/conda/envs/isaaclab/bin/python
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y

hr() { printf '%s\n' "----------------------------------------------------------------"; }

hr
echo "0. 装前版本快照"
$PY - <<'PYEOF'
import torch, numpy
import importlib.metadata as md
print("torch   :", torch.__version__)
print("numpy   :", numpy.__version__)
for p in ("isaacsim", "isaaclab"):
    try:
        print(f"{p:8s}:", md.version(p))
    except Exception as e:  # noqa: BLE001
        print(f"{p:8s}: <无元数据 {type(e).__name__}>")
PYEOF

hr
echo "1. 探测 README §2 B 的依赖哪些缺失"
$PY - <<'PYEOF'
import importlib
mods = {
    "pytorch_kinematics": "pytorch_kinematics", "psutil": "psutil", "toml": "toml",
    "joblib": "joblib", "debugpy": "debugpy", "snakeviz": "snakeviz",
    "trimesh": "trimesh", "sklearn": "sklearn", "cv2": "cv2", "pyvista": "pyvista",
    "tensorboardX": "tensorboardX", "tensorboard": "tensorboard",
    "tabulate": "tabulate", "git": "git", "onnx": "onnx", "onnxscript": "onnxscript",
    "setuptools": "setuptools", "wheel": "wheel", "scipy": "scipy",
}
missing = []
for name, mod in mods.items():
    try:
        importlib.import_module(mod)
        print(f"  OK   {name}")
    except BaseException as e:            # noqa: BLE001
        print(f"  MISS {name}: {type(e).__name__}: {str(e)[:70]}")
        missing.append(name)
print("MISSING=" + " ".join(missing))
PYEOF

hr
echo "2. 用 --no-deps 补装缺失项（不动 Isaac Sim/Torch/NumPy）"
MISSING="pytorch_kinematics psutil toml joblib debugpy snakeviz trimesh scikit-learn opencv-python pyvista tensorboardX tensorboard tabulate GitPython onnx onnxscript"
for pkg in $MISSING; do
  $PY -c "import importlib,sys; importlib.import_module('$pkg')" 2>/dev/null && { echo "  skip $pkg（已装）"; continue; }
  echo "  --- pip install --no-deps $pkg"
  $PY -m pip install --no-deps -q "$pkg" 2>&1 | tail -3
done

hr
echo "3. 装后版本比对（torch / numpy 必须一字不变）"
$PY - <<'PYEOF'
import torch, numpy
print("torch   :", torch.__version__)
print("numpy   :", numpy.__version__)
import importlib
for m in ("pytorch_kinematics", "trimesh", "sklearn", "cv2", "tensorboardX",
          "tabulate", "git", "onnx", "psutil", "joblib"):
    try:
        importlib.import_module(m); print(f"  OK   {m}")
    except BaseException as e:            # noqa: BLE001
        print(f"  MISS {m}: {type(e).__name__}: {str(e)[:70]}")
PYEOF

hr
echo "4. 现在能否 import 任务配置（不需要启动 Kit 吗？不，仍然需要）"
echo "   -> 导入走到哪一层："
$PY - <<'PYEOF'
import importlib
for m in ("instinctlab", "instinctlab.motion_reference",
          "instinctlab.tasks.locomotion.config.g1"):
    try:
        importlib.import_module(m); print(f"  OK   {m}")
    except BaseException as e:            # noqa: BLE001
        print(f"  FAIL {m}: {type(e).__name__}: {str(e)[:90]}")
PYEOF
