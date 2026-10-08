#!/usr/bin/env bash
# V9/V8 训练诊断：σ 是否被上限卡住 / 镜像损失趋势 / AMP 项是否真的进了奖励 / 奖励量级
#
# 用法（云端）：
#     bash /root/cloud_v9_diag.sh [/root/train_amp_walk_v9_4096.log]
#
# 输出刻意控制在 ~40 行（SSH 回传会被截断在 ~50 行）。
set -u
L="${1:-/root/train_amp_walk_v9_4096.log}"
if [ ! -f "$L" ]; then
  echo "找不到 $L；现有日志："
  ls -1t /root/train_amp_walk_*.log 2>/dev/null | head -5
  exit 1
fi

# 一行一个键：n / min / max / mean / first / last
STAT='
{
  while (match($0, /(Mean action noise std|Mean reward|Mean episode length|mirror|amp_d|amp_style_reward|amp_style_zero_frac|Learning iteration): *[-0-9.eE+]+/)) {
    s = substr($0, RSTART, RLENGTH);
    k = s; sub(/:.*/, "", k);
    v = s; sub(/^[^:]*: */, "", v); v += 0;
    n[k]++; sum[k] += v;
    if (!(k in mn) || v < mn[k]) mn[k] = v;
    if (!(k in mx) || v > mx[k]) mx[k] = v;
    if (!(k in fst)) fst[k] = v;
    last[k] = v;
    $0 = substr($0, RSTART + RLENGTH);
  }
}
END { for (k in n) printf "  %-22s n=%-6d first=%9.4f min=%9.4f max=%9.4f mean=%9.4f last=%9.4f\n", k, n[k], fst[k], mn[k], mx[k], sum[k]/n[k], last[k] }
'

echo "== 文件 =="
echo "  $L  ($(stat -c%s "$L") bytes)"
echo "  迭代记录 $(grep -c 'Learning iteration' "$L") 条；最新 $(grep -oE 'Learning iteration [0-9]+' "$L" | tail -1)"
echo
echo "== 关键曲线：全时段 =="
awk "$STAT" "$L" | sort
echo
echo "== 关键曲线：最后 1500 行（≈最近几十轮）=="
tail -1500 "$L" | awk "$STAT" | sort
echo
echo "== 奖励分项键（若有逐项日志）=="
grep -oE '(Episode_Reward|Reward|rew)/[A-Za-z0-9_]+' "$L" | sort | uniq -c | sort -rn | head -12
grep -qE 'Episode_Reward/' "$L" || echo "  （无逐项奖励日志）"
echo
echo "== 最新一轮日志尾部 =="
tail -14 "$L" | sed 's/^ *//' | cut -c1-100
echo
echo "== GPU =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  nvidia-smi 不可用"
echo "  train.py 进程：$(pgrep -fc 'train[.]py' 2>/dev/null || echo 0)"
