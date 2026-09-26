#!/bin/bash
# 从多份训练日志里按「相同迭代轮次」采样关键指标，用于不同 run 的同轮次对比。
# 解决：直接用 ssh + awk 时 $NF 等字段引用会被多层引号吃掉（只回一行垃圾）。
#
# 用法:  bash cloud_compare_runs.sh <log1> [log2] ...
#        默认采样间隔 500 iter，可用 SAMPLE=250 覆盖。
set -u

SAMPLE="${SAMPLE:-500}"

if [ "$#" -eq 0 ]; then
  echo "用法: bash cloud_compare_runs.sh <log1> [log2] ...   (SAMPLE=250 可调采样间隔)"
  exit 1
fi

for LOG in "$@"; do
  echo "==================== $(basename "$LOG") ===================="
  if [ ! -f "$LOG" ]; then
    echo "  (文件不存在)"
    continue
  fi

  awk -v sample="$SAMPLE" '
    # ---- 先记录当前迭代号（日志里形如 "[INFO] Learning iteration 2910/1000000000"）----
    # 注意：日志行前缀字段数不固定，不能靠 $3 取，必须用正则抽 "iteration <N>/"
    /Learning iteration/ {
      it = $0
      if (!sub(/.*iteration */, "", it)) next      # 去掉 "iteration " 之前的一切
      if (!sub(/\/.*/, "", it)) next               # 去掉 "/总轮次"
      gsub(/[^0-9]/, "", it)                       # 清掉颜色码/空格
      if (it == "") next
      if (!(it in seen)) { seen[it] = 1; seq[++cnt] = it }
      next
    }

    # ---- 指标行：值一律是最后一个字段 ----
    /Mean reward:/                        { r[it] = $NF }
    /Mean episode length:/                { l[it] = $NF }
    /Episode_Termination\/base_contact:/  { b[it] = $NF }
    /Episode_Termination\/time_out:/      { o[it] = $NF }
    /discriminator_reward:/               { d[it] = $NF }
    /amp_style_reward:/                   { s[it] = $NF }
    # 损失表里的 mirror 是空格分隔、无冒号
    /^[[:space:]]*mirror[[:space:]]/      { m[it] = $2 }

    END {
      if (cnt == 0) { print "  (没解析到任何 Learning iteration，日志格式不对？)"; exit }

      printf "%9s %10s %9s %10s %9s %10s %10s\n", \
             "iter", "reward", "ep_len", "base_cont", "time_out", "amp_style", "mirror"

      shown = 0
      for (idx = 1; idx <= cnt; idx++) {
        itv = seq[idx]
        if (itv % sample != 0) continue
        printf "%9s %10s %9s %10s %9s %10s %10s\n", \
               itv, r[itv], l[itv], b[itv], o[itv], s[itv], m[itv]
        shown = 1
      }

      # 把最后一个已打印轮次之后的点也补上（否则最新进展看不到）
      if (shown && itv != seq[cnt]) {
        itv = seq[cnt]
        printf "%9s %10s %9s %10s %9s %10s %10s   <- 最新\n", \
               itv, r[itv], l[itv], b[itv], o[itv], s[itv], m[itv]
      }
    }
  ' "$LOG"
  echo
done
