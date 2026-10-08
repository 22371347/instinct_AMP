#!/usr/bin/env bash
# ============================================================================
# **受控实测**：并行负载对训练速度（每次迭代耗时）的影响（跑在云端）
# ============================================================================
# 背景：用户主观感觉"并行时训练变慢"（V16 训练期间我并行跑过 Isaac Sim 探针 / ONNX 导出）。
# 本脚本用**同一台机、同一训练、同一窗口长度**下换负载条件，量出真实差异。
#
# 为什么用日志里的 `Iteration time` 而不是总时长：
#   * 总时长含建场景/JIT 预热（前几十轮很慢）与停机时刻，噪声大；
#   * 训练日志**每轮**都打一行 `Iteration time: X.XXs`，可直接做窗口统计（中位数/分位）。
#
# 五个条件（每个窗口默认 300 s，约 85 轮 @3.5 s/轮）：
#   A 基线        只训练
#   B +2 CPU hog  2 个忙循环进程
#   C +8 CPU hog  8 个忙循环进程
#   D +探针       **1 个 8-env Isaac Sim 探针**（= 我此前并行跑的那类负载，含建场景尖峰）
#   E 基线（复核）再次只训练，检查漂移/温度降频等时间趋势
#
# ⚠️ 环境事实：容器 **CPU 配额 14 核**（`Core Usage Quota: 14`，见 kit 启动横幅），
#    GPU 24.5 GB（一次 4096-env 训练约用 10~11.5 GB，显存并不紧张）
#    ⇒ 并行的代价主要来自 **CPU 配额被抢**，不是显存。这也是脚本第 0 步打印 cpu.max 的原因。
#
# 用法（云端，训练**已在跑**时执行）：
#   bash /root/cloud_measure_parallel_speed.sh <训练日志路径> [窗口秒数=300] [探针任务id]
# 例：
#   bash /root/cloud_measure_parallel_speed.sh /root/train_amp_walk_v17_4096.log 300
#
# 输出：每个条件一行统计（n / 均值 / 中位 / p90 / 最小 / 最大）
# 判读：把 B/C/D 的**中位数**与 A 比 —— 差值就是该负载的并行代价；
#       E 与 A 应当接近（若差别大，说明测量期间系统本身有漂移，需加大窗口或改夜间测）。
# ============================================================================
set -uo pipefail

LOG="${1:?用法: bash cloud_measure_parallel_speed.sh <训练日志> [窗口秒] [探针任务id]}"
WIN="${2:-300}"
PROBE_TASK="${3:-Instinct-Amp-Walk-V16-G1-v0}"
HOGPIDS="$(mktemp)"

echo "############ 受控测速开始 $(date '+%F %T') ############"
echo "--- 0) 环境 ---"
if [ -f /sys/fs/cgroup/cpu.max ]; then
    echo "cgroup cpu.max = $(cat /sys/fs/cgroup/cpu.max)   （quota/period ⇒ 可用核数）"
else
    echo "cgroup cpu.max 不存在（可能是 v1；nproc=$(nproc)）"
fi
echo "nproc = $(nproc)"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true
echo "训练日志 = $LOG（现有 $(grep -c 'Iteration time' "$LOG" 2>/dev/null || echo 0) 条 Iteration time）"
pgrep -af 'train[.]py' | grep -v pgrep | head -2 || { echo "❌ 没有在跑的训练，先起一个再测"; exit 1; }

# ---- 工具函数 ----------------------------------------------------------------
stats() {  # $1=标签；从 stdin 读一串秒数
    python3 -c "
import sys
tag = sys.argv[1]
xs = [float(l.strip()) for l in sys.stdin if l.strip()]
if not xs:
    print(f'  {tag:<22} ❌ 窗口内没有新样本（训练是否卡住？）'); raise SystemExit
xs.sort(); n = len(xs)
print(f'  {tag:<22} n={n:<4} 均值={sum(xs)/n:6.3f}s  中位={xs[n//2]:6.3f}s  '
      f'p90={xs[int(n*0.9)]:6.3f}s  最小={xs[0]:6.3f}s  最大={xs[-1]:6.3f}s')
" "$1"
}

measure_window() {  # $1=标签  $2=窗口秒
    local n0; n0=$(grep -c 'Iteration time' "$LOG")
    sleep "$2"
    grep -o 'Iteration time: [0-9.]*s' "$LOG" | tail -n +$((n0 + 1)) | sed 's/[^0-9.]//g' | stats "$1"
}

start_hogs() {  # $1=个数
    for _ in $(seq 1 "$1"); do
        nohup bash -c 'while :; do :; done' >/dev/null 2>&1 &
        echo $! >> "$HOGPIDS"
    done
    sleep 5
}
stop_hogs() {
    [ -s "$HOGPIDS" ] && xargs -r kill -9 < "$HOGPIDS" 2>/dev/null || true
    : > "$HOGPIDS"
    sleep 5
}

trap 'stop_hogs; rm -f "$HOGPIDS"' EXIT

# ---- A 基线 -----------------------------------------------------------------
echo
echo "--- A) 基线：只训练（${WIN}s）---"
measure_window "A 基线" "$WIN"

# ---- B/C CPU hog ------------------------------------------------------------
echo
echo "--- B) +2 个 CPU hog（${WIN}s）---"
start_hogs 2; measure_window "B 训练+2hog" "$WIN"; stop_hogs

echo
echo "--- C) +8 个 CPU hog（${WIN}s）---"
start_hogs 8; measure_window "C 训练+8hog" "$WIN"; stop_hogs

# ---- D 真·探针（Isaac Sim，8 envs） -----------------------------------------
echo
echo "--- D) +1 个 8-env Isaac Sim 探针（${WIN}s；含建场景尖峰）---"
if [ -f /root/cloud_run_bg.sh ] && [ -f /root/cloud_probe_amp_append.py ]; then
    bash /root/cloud_run_bg.sh \
        "/opt/conda/envs/isaaclab/bin/python /root/cloud_probe_amp_append.py \
         --task ${PROBE_TASK} --num_envs 8 --steps 6 --device cuda:0 --headless \
         > /root/_speed_probe_out.txt 2>&1" /root/_speed_probe_wrap.log
    measure_window "D 训练+探针" "$WIN"
    # 探针本身可能还在跑，收尾（只杀探针，不动训练）
    pkill -9 -f cloud_probe_amp_append || true
    sleep 5
else
    echo "  （跳过：/root/cloud_probe_amp_append.py 或 cloud_run_bg.sh 不在，先把工具传上去）"
fi

# ---- E 基线复核 -------------------------------------------------------------
echo
echo "--- E) 基线复核：又只训练（${WIN}s）---"
measure_window "E 基线复核" "$WIN"

echo
echo "############ 测速结束 $(date '+%F %T') ############"
echo "读法：把 B/C/D 的中位数与 A 比较 ⇒ 差值 = 该负载的并行代价；"
echo "      E 与 A 若不接近，说明有系统漂移（降频/别的进程），需加长窗口或换时间重测。"
