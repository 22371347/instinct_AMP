#!/usr/bin/env bash
# 迭代补齐「传递依赖」（--no-deps 装完主包后，它自己的 import 还缺东西）
#
# 为什么要这么绕：README 铁律是**不能动 Isaac Sim / Torch / NumPy**，
# 所以主包一律 `--no-deps`；缺的传递依赖再显式、逐个补，并且每一步都比对版本。
set -uo pipefail
PY=/opt/conda/envs/isaaclab/bin/python

echo "=== 迭代修复 pytorch_kinematics（我们的真实阻塞点）==="
$PY - <<'PYEOF'
import importlib, subprocess, sys
target = "pytorch_kinematics"
for i in range(10):
    try:
        importlib.import_module(target)
        print(f"  ✅ {target} 可导入（第 {i} 轮）")
        break
    except ModuleNotFoundError as e:
        mod = e.name
        print(f"  第 {i} 轮：缺 {mod} → pip install --no-deps {mod}")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", mod],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("   安装失败：", (r.stderr or r.stdout)[-300:])
            break
    except BaseException as e:  # noqa: BLE001
        print(f"  非 ImportError：{type(e).__name__}: {e}")
        break
else:
    print("  ⚠️ 10 轮还没好")
PYEOF

echo
echo "=== 同样修 sklearn（只用它保证 pip 里那堆 extras 自洽；本任务其实不需要）==="
$PY - <<'PYEOF'
import importlib, subprocess, sys
target = "sklearn"
for i in range(6):
    try:
        importlib.import_module(target)
        print(f"  ✅ {target} 可导入（第 {i} 轮）")
        break
    except ModuleNotFoundError as e:
        mod = e.name
        print(f"  第 {i} 轮：缺 {mod} → pip install --no-deps {mod}")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", mod],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("   安装失败：", (r.stderr or r.stdout)[-300:])
            break
    except BaseException as e:  # noqa: BLE001
        print(f"  非 ImportError：{type(e).__name__}: {e}")
        break
PYEOF

echo
echo "=== 版本回归检查（必须与修复前一致：torch 2.7.0+cu128 / numpy 1.26.0）==="
$PY - <<'PYEOF'
import torch, numpy
print("torch :", torch.__version__)
print("numpy :", numpy.__version__)
assert torch.__version__.startswith("2.7.0"), "torch 被改了！"
assert numpy.__version__.startswith("1.26"), "numpy 被改了！"
print("✅ 核心版本未被触动")
import importlib
for m in ("pytorch_kinematics", "trimesh", "scipy", "lxml"):
    try:
        importlib.import_module(m); print(f"  OK   {m}")
    except BaseException as e:  # noqa: BLE001
        print(f"  MISS {m}: {type(e).__name__}: {str(e)[:80]}")
PYEOF

echo
echo "=== 顺手记录：instinctlab 的 motion_reference 是否还缺别的 ==="
$PY - <<'PYEOF'
import ast, pathlib
p = pathlib.Path("/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab/motion_reference/motion_reference_manager.py")
t = ast.parse(p.read_text())
tops = set()
for n in ast.walk(t):
    if isinstance(n, ast.Import):
        for a in n.names:
            tops.add(a.name.split(".")[0])
    elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
        tops.add(n.module.split(".")[0])
print("  顶层 import:", sorted(tops))
PYEOF
