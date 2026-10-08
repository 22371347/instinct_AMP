#!/bin/bash
# 通用：按迭代采样任意 `Episode_Reward/*` 项（+ 回合长 / 回报）。
#
# 存在理由：
#   ① 直接手写 awk + ssh 会被四层引号吃掉字段引用（只回一行垃圾）；
#   ② `Episode_Reward/*` 是 **每 20 秒的累计量**（= 回合内累加 / max_episode_length_s）
#      ⇒ **回合变长它就会线性变大**，所以必须同时看 `Mean episode length`，
#      并用「每步值 = 值 × 20 / 回合长」来判断"到底是项变差还是回合变长"。
#
# 用法:
#   bash cloud_sample_terms.sh <log> <关键词1> [关键词2 ...]
#     SAMPLE=500 采样间隔（默认 500）
#   例: bash cloud_sample_terms.sh /root/train_amp_walk_v6_4096.log joint_deviation arms torso
set -u

SAMPLE="${SAMPLE:-500}"
LOG="${1:-}"
shift || true

if [ -z "$LOG" ] || [ "$#" -eq 0 ]; then
  echo "用法: bash cloud_sample_terms.sh <log> <关键词1> [关键词2 ...]   (SAMPLE=500)"
  exit 1
fi
[ -f "$LOG" ] || { echo "找不到日志: $LOG"; exit 1; }

# 把关键词拼成 awk 里可 split 的字符串
PATS="$*"

awk -v sample="$SAMPLE" -v pats="$PATS" '
  /Learning iteration/ {
    it = $0
    if (!sub(/.*iteration */, "", it)) next
    if (!sub(/\/.*/, "", it)) next
    gsub(/[^0-9]/, "", it); if (it == "") next
    if (!(it in seen)) { seen[it] = 1; seq[++cnt] = it }
    next
  }
  /Episode_Reward\// {
    tag = $3; sub(/:$/, "", tag); sub(/^Episode_Reward\//, "", tag)
    v[tag, it] = $NF
    if (!(tag in seenTag)) { seenTag[tag] = 1; nt++; tname[nt] = tag }
    next
  }
  /Mean episode length:/ { el[it] = $NF }
  /Mean reward:/         { rw[it] = $NF }

  END {
    if (cnt == 0) { print "  (没解析到迭代号)"; exit }

    nP = split(pats, P, " ")
    # 挑出匹配的项名
    nS = 0
    for (t = 1; t <= nt; t++) {
      for (p = 1; p <= nP; p++) {
        if (index(tname[t], P[p]) > 0) { nS++; sel[nS] = tname[t]; break }
      }
    }
    if (nS == 0) { print "  (没有匹配的 Episode_Reward 项)"; exit }

    printf "%7s", "iter"
    for (s = 1; s <= nS; s++) printf " %14s", substr(sel[s], 1, 14)
    printf " %9s %9s\n", "ep_len", "reward"
    print "  ---- 每秒速率（原始打印值）----"

    for (i = 1; i <= cnt; i++) {
      itv = seq[i]
      if (itv % sample != 0 && i != cnt) continue
      printf "%7s", itv
      for (s = 1; s <= nS; s++) printf " %14s", v[sel[s], itv]
      printf " %9s %9s\n", el[itv], rw[itv]
    }

    # 再来一遍，换成「每步值 = 值 × 20 / 回合长」——判断项本身是否变差
    print ""
    print "  ---- 每步值（= 值 × 20 / 回合长；⭐ 判断\"项变差\"还是\"回合变长\"）----"
    printf "%7s", "iter"
    for (s = 1; s <= nS; s++) printf " %14s", substr(sel[s], 1, 14)
    for (i = 1; i <= cnt; i++) {
      itv = seq[i]
      if (itv % sample != 0 && i != cnt) continue
      printf "%7s", itv
      for (s = 1; s <= nS; s++) {
        x = v[sel[s], itv] + 0
        if (el[itv] + 0 > 0) printf " %14.5f", x * 20.0 / (el[itv] + 0)
        else printf " %14s", "-"
      }
      printf "\n"
    }
  }
' "$LOG"
