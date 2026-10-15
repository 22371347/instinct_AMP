#!/usr/bin/env bash
# ============================================================================
# V29 逐维探针（跑在云端，**只需几分钟**）：采 actor / reference 的 AMP 状态逐维 μ/σ
# ============================================================================
# 为什么必须上云：actor 侧的 AMP 状态只存在于仿真里（`obs["amp_policy"]`），离线算不出来。
# 探针已打进 `algorithms.py::compute_auxiliary_reward`（前 5 个 update 各写一个文件，之后不再执行）。
#
# 流程：从 V27 的 model_10000 热启动 V29 → 等 5 个 `/root/amp_dim_probe_*.txt` 出现
#       → 立刻停训（不做任何训练价值的事，纯采样）。
#
# 用法：bash /root/probe_v29_dims.sh
# ============================================================================
set -uo pipefail

RUN_ROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk
LOG=/root/train_amp_walk_v29_probe.log

rm -f /root/amp_dim_probe_*.txt

echo "=== 1) 确保没有训练在跑 ==="
pkill -9 -f 'train[.]py' || true
sleep 2

echo "=== 2) 找 V27 最新 ckpt ==="
V27_DIR=""
for d in $(ls -td "$RUN_ROOT"/*v27* 2>/dev/null); do
  if ls "$d"/model_*.pt >/dev/null 2>&1; then V27_DIR="$d"; break; fi
done
[[ -n "$V27_DIR" ]] || { echo "❌ 找不到 V27 run 目录" >&2; exit 1; }
V27_CK=$(ls "$V27_DIR"/model_*.pt \
         | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
echo "  热启动：$(basename "$V27_DIR") / $(basename "$V27_CK")"

echo "=== 3) 起 V29（打补丁后的 wrapper）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v29.sh 4096 1000000000 cuda:0 amp_walk_v29_probe_4096 '$V27_DIR' $(basename "$V27_CK")" \
  "$LOG"

echo "=== 4) 等 5 个探针文件（最多 12 分钟）==="
for i in $(seq 1 72); do
  n=$(ls /root/amp_dim_probe_*.txt 2>/dev/null | wc -l)
  if [[ "$n" -ge 5 ]]; then echo "  ✅ 已有 $n 个探针文件"; break; fi
  if ! pgrep -f 'train[.]py' >/dev/null 2>&1 && [[ "$i" -gt 3 ]]; then
    echo "  ⚠️ train.py 已退出（可能报错）" ; break
  fi
  sleep 10
done
ls -l /root/amp_dim_probe_*.txt 2>/dev/null || echo "  ❌ 没生成探针文件"

echo "=== 5) 停训（采样完成，不再占用 GPU）==="
pkill -9 -f 'train[.]py' || true
sleep 2
pgrep -af 'train[.]py' | grep -v pgrep || echo "  ✅ 已无 train.py 进程"
echo "=== done ==="
