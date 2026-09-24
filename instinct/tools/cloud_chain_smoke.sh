#!/usr/bin/env bash
# =============================================================================
# 云端一键链路：补依赖 → 端到端冒烟
# =============================================================================
# 为什么串成一个脚本：这样只需要**一个后台任务 + 一份日志**，SSH 断了也不影响，
# 轮询一次就能同时看到「缺什么依赖」和「冒烟结论」。
#
# 前置（都在 /root 下，已就位）：
#   cloud_dep_scan.py           一次性扫完两棵源码树的顶层第三方 import 并补齐
#   cloud_smoke_amp_walk.sh     4 层冒烟（第 3 层是真正的裁决）
#   cloud_run_bg.sh             后台跑 + 落盘日志（它会先 source /root/.llp_env.sh）
#
# 启动方式：
#   bash /root/cloud_run_bg.sh 'bash /root/cloud_chain_smoke.sh' /root/smoke.log
# 轮询：
#   tail -40 /root/smoke.log
# =============================================================================
set -uo pipefail
PY=/opt/conda/envs/isaaclab/bin/python

hr() { printf '\n############ %s ############\n' "$1"; }

hr "环境快照"
$PY - <<'PYEOF'
import torch, sys
print("python:", sys.version.split()[0], "->", sys.executable)
print("torch :", torch.__version__, "cuda_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu   :", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
import importlib.metadata as md
for p in ("isaacsim", "isaaclab", "instinctlab", "instinct_rl"):
    try:
        print(f"{p:11s}:", md.version(p))
    except Exception as e:  # noqa: BLE001
        print(f"{p:11s}: <无元数据 {type(e).__name__}>")
PYEOF

hr "1. 依赖扫描 + 补齐（不动 torch / numpy / Isaac Sim）"
$PY /root/cloud_dep_scan.py

hr "2. 端到端冒烟（第 3 层 = 真正的裁决）"
export INSTINCT_XU_ROOT=/root/gpufree-data/instinct_xu
export NUM_ENVS="${NUM_ENVS:-16}"
export MAX_ITERS="${MAX_ITERS:-2}"
export DEVICE="${DEVICE:-cuda:0}"
bash /root/cloud_smoke_amp_walk.sh

hr "链路结束（exit=$?）"
