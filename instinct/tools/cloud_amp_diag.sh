#!/bin/bash
# AMP 诊断：按迭代采样「判别器输出 / 风格奖励」相关指标。
#
# 背景：`quad` 型风格奖励 r = clamp(1 - 0.25*(d-1)^2, 0)，其中 d 是判别器**原始线性输出**。
# 所以只要知道 r，就能反解 d：d = 1 - 2*sqrt(1 - r)（取 -1 侧那个根）。
# 这个表就是用来确认「判别器是否已经饱和到 -1 ⇒ 风格奖励塌到 0」。
#
# 用法:  bash cloud_amp_diag.sh <log1> [log2] ...      (SAMPLE=250 可调采样间隔)
set -u

SAMPLE="${SAMPLE:-500}"

if [ "$#" -eq 0 ]; then
  echo "用法: bash cloud_amp_diag.sh <log1> [log2] ...   (SAMPLE=250 可调采样间隔)"
  exit 1
fi

for LOG in "$@"; do
  echo "==================== $(basename "$LOG") ===================="
  if [ ! -f "$LOG" ]; then
    echo "  (文件不存在)"
    continue
  fi

  awk -v sample="$SAMPLE" '
    # ---- 迭代号：日志行前缀字段数不固定，必须正则抽 ----
    /Learning iteration/ {
      it = $0
      if (!sub(/.*iteration */, "", it)) next
      if (!sub(/\/.*/, "", it)) next
      gsub(/[^0-9]/, "", it)
      if (it == "") next
      if (!(it in seen)) { seen[it] = 1; seq[++cnt] = it }
      next
    }

    /amp_style_reward:/        { s[it]  = $NF }   # 原始 r_amp（已过"站立关风格"门控）
    /^[[:space:]]*discriminator_loss/ { dl[it] = $NF }   # 0.5*(MSE(d_a,-1)+MSE(d_r,+1))
    /^[[:space:]]*amp_total_loss/     { at[it] = $NF }
    /discriminator_actor:/     { da[it] = $NF }
    /discriminator_reference:/ { dr[it] = $NF }

    END {
      if (cnt == 0) { print "  (没解析到迭代号)"; exit }
      printf "%8s %10s %10s %10s %10s %10s %10s\n", \
             "iter", "style_r", "d(反解)", "disc_loss", "amp_total", "d_actor", "d_ref"

      for (idx = 1; idx <= cnt; idx++) {
        v = seq[idx]
        if (v % sample != 0) continue
        # 反解 d：r = 1 - 0.25*(d-1)^2  ⇒  d = 1 - 2*sqrt(1-r)
        d = ""
        if (s[v] != "") {
          x = 1 - s[v]
          if (x >= 0) d = sprintf("%.3f", 1 - 2 * sqrt(x))
        }
        printf "%8s %10s %10s %10s %10s %10s %10s\n", v, s[v], d, dl[v], at[v], da[v], dr[v]
        shown = 1
      }
      if (shown && v != seq[cnt]) {
        v = seq[cnt]
        d = ""
        if (s[v] != "") { x = 1 - s[v]; if (x >= 0) d = sprintf("%.3f", 1 - 2 * sqrt(x)) }
        printf "%8s %10s %10s %10s %10s %10s %10s   <- 最新\n", v, s[v], d, dl[v], at[v], da[v], dr[v]
      }
    }
  ' "$LOG"
  echo
done
