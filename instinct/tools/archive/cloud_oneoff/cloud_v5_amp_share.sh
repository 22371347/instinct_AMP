#!/bin/bash
# V5 专用诊断：① AMP 风格奖励趋势（含"排序信号" std）② AMP 在 mean_reward 中的确切占比
#
# 用法:  bash cloud_v5_amp_share.sh <log>        (SAMPLE=250 可调采样间隔)
#
# ── 占 比 是 怎 么 算 的（❗之前算错过，务必按这个口径读）────────────────────────
# 先记住一个反直觉的事实：**`Mean reward:` 里根本没有 AMP**。
# `rewbuffer` 累加的是 `rollout_step` 返回的 `rewards`，而那是 `env.step()` 的**纯环境奖励**；
# AMP aux 是 `process_env_step` 内部用 `rewards.clone()` 加到 `transition.rewards` 上的，
# **不会改到 runner 手里那个张量** ⇒ `Mean reward` = 纯任务回报（既不含 AMP，也不含 λ）。
# 两条独立证据：(a) Σ Episode_Reward×20s ≈ Mean reward（差 ~1%）；
#                (b) 若含 AMP 则应为 4.7 而非 1.3。
#
# 真正的训练回报： R = λ·r_task + (1-λ)·c·r_amp     λ=0.75, c=0.1 ⇒ AMP 每步 0.025×r_amp
# ❗ 比"占比"时分母里的**任务项也要乘 λ**（之前忘了乘，会把 79.6% 误报成 74.6%）。
#
# 两条路径交叉校验（应该闭合，闭合误差 >20% 说明假设有问题）：
#   A: R = λ·r_task + 0.025·r_amp
#   B: R = r_task + c·aux        ← aux = 日志里的 `Mean step discriminator_reward`（实测）
set -u

SAMPLE="${SAMPLE:-500}"
LOG="${1:-/root/train_amp_walk_v5_4096.log}"
EP_S="${EP_S:-20}"          # max_episode_length_s
LAM="${LAM:-0.75}"          # style_lerp
C="${C:-0.1}"               # discriminator_reward_coef

[ -f "$LOG" ] || { echo "找不到日志: $LOG"; exit 1; }

awk -v sample="$SAMPLE" -v eps="$EP_S" -v lam="$LAM" -v cc="$C" '
  # ---- 迭代号（前缀字段数不固定，必须正则抽）----
  /Learning iteration/ {
    it = $0
    if (!sub(/.*iteration */, "", it)) next
    if (!sub(/\/.*/, "", it)) next
    gsub(/[^0-9]/, "", it); if (it == "") next
    if (!(it in seen)) { seen[it] = 1; seq[++cnt] = it }
    next
  }
  # ---- 任务奖励项（每秒速率）→ 按迭代求和 ----
  /Episode_Reward\// {
    tag = $3; sub(/:$/, "", tag)
    task[it] += $NF
    term[tag] = $NF          # 最后一块的值（每次覆盖 ⇒ 收尾就是最新）
    next
  }
  /Mean reward:/            { rw[it] = $NF }
  /Mean episode length:/    { el[it] = $NF }
  /amp_style_reward:/       { s[it]  = $NF }
  /amp_style_std:/          { sd[it] = $NF }
  /amp_style_zero_frac:/    { z[it]  = $NF }
  /amp_stand_frac:/         { g[it]  = $NF }
  /discriminator_reward:/   { ax[it] = $NF }
  /^[[:space:]]*discriminator_loss/ { dl[it] = $NF }

  END {
    if (cnt == 0) { print "  (没解析到迭代号)"; exit }

    print "===== [1] 趋势（AMP 相关）====="
    printf "%7s %9s %9s %9s %8s %8s %10s %9s %8s\n", \
           "iter", "style_r", "std", "zero_fr", "stand_fr", "aux", "disc_loss", "reward", "ep_len"
    for (i = 1; i <= cnt; i++) {
      v = seq[i]
      if (v % sample != 0) continue
      printf "%7s %9s %9s %9s %8s %8s %10s %9s %8s\n", \
             v, s[v], sd[v], z[v], g[v], ax[v], dl[v], rw[v], el[v]
      last = v
    }
    if (last != seq[cnt]) {
      v = seq[cnt]
      printf "%7s %9s %9s %9s %8s %8s %10s %9s %8s   <- 最新\n", \
             v, s[v], sd[v], z[v], g[v], ax[v], dl[v], rw[v], el[v]
      last = v
    }

    print ""
    print "===== [2] 最新轮次的占比分解（iter " last "）====="
    print "  ❗ 口径：`Mean reward` = 纯任务回报（不含 AMP）；下面是【真正训练回报】的分解"
    print ""
    tv   = eps * task[last]                 # 纯任务回报（= Mean reward 的口径）
    rts  = rw[last] / el[last]              # 每步任务奖励
    ae   = s[last] * (1 - g[last])          # 门控后的有效 r_amp
    wamp = (1 - lam) * cc                   # = 0.025

    a_task = lam * rw[last]                              # λ·任务（参与真实回报的那部分）
    a_amp  = wamp * ae * el[last]                        # AMP per-episode
    print "  -- 路径 A： R = λ·r_task + (1-λ)c·r_amp --"
    printf "     r_task(纯任务,每回合)  = %10.3f      (= Mean reward)\n", rw[last]
    printf "     r_amp(门控前 → 有效)   = %.4f → %.4f   (站立门控剥离 %.2f%%)\n", s[last], ae, 100 * g[last]
    printf "     λ·任务占比的分子        = %10.3f      (= %.2f × %.3f)\n", a_task, lam, rw[last]
    printf "     AMP per-episode        = %10.3f      (= %.3f × %.4f × %s 步)\n", a_amp, wamp, ae, el[last]
    printf "     真实训练回报            = %10.3f\n", a_task + a_amp
    printf "  ⭐ AMP 占真实训练回报 = %.1f %%    (任务项 %.1f %%)\n", \
           100 * a_amp / (a_task + a_amp), 100 * a_task / (a_task + a_amp)

    print ""
    print "  -- 路径 B（独立交叉校验）： R = r_task + c·aux，aux 用日志实测 --"
    if (ax[last] == "") {
      print "     (日志里没找到 Mean step discriminator_reward，跳过)"
    } else {
      printf "     aux(实测)             = %10.5f\n", ax[last]
      printf "     c·aux(每步净增量)      = %10.5f\n", cc * ax[last]
      printf "     R 每步 = r_task + c·aux = %.5f + %.5f = %.5f\n", rts, cc * ax[last], rts + cc * ax[last]
      printf "     → 每回合 %.3f\n", (rts + cc * ax[last]) * el[last]
      printf "     AMP 项每步（扣掉 λ 修正前）= %.5f → 占比 %.1f %%\n", \
             (rts + cc * ax[last]) * el[last] / el[last] - lam * rts, \
             100 * ((rts + cc * ax[last]) - lam * rts) / (rts + cc * ax[last])
      printf "     ❗ 路径 A/B 闭合误差 = %.1f %%（>20%% 说明假设有问题，别直接引用）\n", \
             100 * ((a_task + a_amp) / el[last] - (rts + cc * ax[last])) / (rts + cc * ax[last])
    }

    print ""
    printf "  参考： Σ Episode_Reward×%.0fs = %.3f (于 Mean reward %.3f 对比；应接近 ⇒ 印证 Mean reward 是纯任务)\n", \
           eps, tv, rw[last]

    print ""
    print "===== [3] 任务奖励项（每秒速率，|值| >= 0.02）====="
    for (t in term) {
      x = term[t] + 0
      if (x >= 0.02 || x <= -0.02) printf "  %-42s %10s   (per-ep %.3f)\n", t, term[t], eps * x
    }
  }
' "$LOG"
