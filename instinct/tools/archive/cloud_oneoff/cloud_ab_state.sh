#!/usr/bin/env bash
# 并行训练「现状快照」：对每个日志打印最新一轮的关键量 + 迭代数 + 最新 ckpt
# 用法（云端）： bash /root/cloud_ab_state.sh [log1 log2 ...]
set -u
LOGS=("$@")
if [ ${#LOGS[@]} -eq 0 ]; then
  LOGS=(/root/train_amp_walk_v9b_4096.log /root/train_amp_walk_v9c_2048.log)
fi
LOGROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk

KEYS='Mean action noise std|Mean reward|Mean episode length|amp_d|amp_style_reward|discriminator_reward|Episode_Reward/track_lin_vel_xy_exp|Episode_Reward/track_base_height|Episode_Reward/joint_deviation_arms|Episode_Termination/base_contact'

for L in "${LOGS[@]}"; do
  [ -f "$L" ] || { echo "== $(basename "$L"): 不存在"; continue; }
  echo "════ $(basename "$L")   迭代 $(grep -ac 'Learning iteration' "$L") ════"
  awk -v pat="$KEYS" '
    {
      while (match($0, "(" pat "): *[-0-9.eE+]+")) {
        s = substr($0, RSTART, RLENGTH)
        k = s; sub(/:.*/, "", k)
        v = s; sub(/^[^:]*: */, "", v); v += 0
        n[k]++; last[k] = v
        $0 = substr($0, RSTART + RLENGTH)
      }
    }
    END { for (k in n) printf "    %-44s last=%-12.4f (n=%d)\n", k, last[k], n[k] }
  ' "$L" | sort
  # 最新 ckpt：按目录名匹配（日志名去掉 train_ 前缀与 .log 后缀就是 tag）
  TAG="$(basename "$L")"; TAG="${TAG#train_}"; TAG="${TAG%.log}"
  D="$(ls -1dt "$LOGROOT"/*"$TAG" 2>/dev/null | head -1)"
  if [ -n "$D" ]; then
    echo "    run=$(basename "$D")"
    echo "    ckpt=$(ls -1 "$D"/model_*.pt 2>/dev/null | sed 's#.*/##' | sort -t_ -k2 -n | tail -1)（共 $(ls -1 "$D"/model_*.pt 2>/dev/null | wc -l) 个）"
  fi
  echo
done
