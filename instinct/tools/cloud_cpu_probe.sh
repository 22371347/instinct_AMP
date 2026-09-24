#!/usr/bin/env bash
# =============================================================================
# CPU-only 云平台：可做/不可做边界的硬事实采集
# =============================================================================
# 回答三个问题：
#   P1. 镜像里到底装了哪几个 isaaclab* 分发包？（check_install.py 需要它们的 pip 元数据）
#   P2. 免 Kit 能导入到哪一层？`isaaclab_assets` / `isaaclab_tasks` 能否导入？
#   P3. ★ 没有 GPU 时 Kit（Isaac Sim）到底能不能启动？—— 这决定了
#       「配置实例化 + to_dict()」这类验证在 CPU-only 机器上到底有没有可能做
# 顺带跑源码审计的 Q5（ContactSensor 过滤限制）与 to_dict 实现。
# =============================================================================
set -uo pipefail

PY=/opt/conda/envs/isaaclab/bin/python
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1
hr() { printf '%s\n' "----------------------------------------------------------------"; }
sec() { echo; hr; echo "$1"; hr; }

# shellcheck disable=SC1091
[[ -f /root/.llp_env.sh ]] && source /root/.llp_env.sh
SP=/opt/conda/envs/isaaclab/lib/python3.11/site-packages
IL="$SP/isaaclab/source/isaaclab/isaaclab"

sec "P1. 已装的 isaaclab* 分发包"
$PY -m pip list 2>/dev/null | grep -i "isaac\|torch\|numpy" || true
echo "--- site-packages/isaaclab/source/ 下的快照 ---"
ls -1 "$SP/isaaclab/source/" 2>/dev/null
echo "--- 各自的 dist-info 是否存在 ---"
for p in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic isaaclab_contrib; do
  d=$(ls -d "$SP/${p}"*.dist-info 2>/dev/null | head -1)
  printf '  %-20s %s\n' "$p" "${d:-<无 dist-info>}"
done

sec "P2. 免 Kit 导入探测（补充）"
$PY - <<'PYEOF'
import importlib
for m in ("isaacsim", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl",
          "instinctlab.assets", "instinctlab.assets.unitree_g1"):
    try:
        importlib.import_module(m); print(f"  OK   {m}")
    except BaseException as e:            # noqa: BLE001
        print(f"  FAIL {m}: {type(e).__name__}: {str(e)[:100]}")
import torch
print(f"\n  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}"
      f"  cuda_runtime={torch.version.cuda}")
PYEOF

sec "P3. 没有 GPU 时 Kit 能否启动（硬超时 240s）"
cat > /tmp/try_kit.py <<'PYEOF'
import sys
print("  -> importing isaaclab.app ...", flush=True)
from isaaclab.app import AppLauncher
print("  -> AppLauncher(headless=True) ...", flush=True)
app = AppLauncher(headless=True).app
print("  KIT_OK: app 启动成功（CPU 模式下居然可行）", flush=True)
app.close()
PYEOF
cd /root/gpufree-data/instinct_xu || exit 1
timeout 240 $PY /tmp/try_kit.py 2>&1 | tail -25
rc=${PIPESTATUS[0]}
echo "  exit=$rc $([[ $rc -eq 124 ]] && echo '（超时 → Kit 在无 GPU 下卡住）')"

sec "P_source. to_dict() 的实现位置（configclass）"
grep -n "to_dict\|def __dict__\|asdict" "$IL/utils/configclass.py" | head -20
echo "--- add_annotations 的核心（确认无注解成员会被补注解）---"
sed -n '183,245p' "$IL/utils/configclass.py"

sec "P_source. ContactSensor 一对多限制原文（Q5）"
grep -rn "one-to-many\|single primitive per environment\|matches a single" "$IL/sensors/contact_sensor/" 2>/dev/null | head -8
grep -n "filter_prim_paths_expr" "$IL/sensors/contact_sensor/contact_sensor_cfg.py" | head -6

sec "完成"
