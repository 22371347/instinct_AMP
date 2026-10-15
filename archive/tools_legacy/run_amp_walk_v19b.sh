#!/usr/bin/env bash
# ============================================================================
# V19b 启动器（跑在云端）：**从零 + 步态带近零端加密（exp, τ=0.08）+ quad 下端放宽**
# ============================================================================
# 基准 = V19a（10 帧 670 维 / `log_quad`+`wasserstein` / λ0.75 / c1.0 / GP 1.0 /
#               wd 1e-5 / logit_wd 0 / eps 0.05 / 5 clip 全向指令 / `flat −2` /
#               `action_rate −0.10` / `feet_swing_time_penalty 0`）。
#
# 两处改动（用户 2026-10-02 拍板）：
#   ① 步态带下侧：线性 `(0.4−d)/0.4` → **指数** `1 − exp(−d/0.08)`（归一化到 good_lo）
#      ⇒ 在 d=0.01 s 处 score 0.025 → **0.118**、梯度 2.5/s → **≈12.5/s**；
#         在 d=0.35 s 后已≈1 ⇒ 与 [0.4, 1.0] 好区间不打架。
#      （V18 的 `soft=0.3` 在 `d<0.1 s` 处值=0 且梯度=0；V19a 线性化后梯度太小 ⇒
#        `air_time_single_stance` 爬到 0.0111 s 就平台化，只拿到上限 2.4/步 的 1.6%。）
#   ② `amp_quad_k` 0.25 → **0.125**：V19a 实测 5.0% 样本越过 `d ≤ −1` ⇒ style≡0
#      且 dr/dd=0（死区）；放宽后下端到 `d ≤ −1.83`，且 `style_actor` 0.199 → 0.892
#      （风格项/步 0.049 → 0.223，占 r_task 3.4% → ~15%）。
#   ③ **从零训练**（起点也是变量）：V18（从零）与 V19a（热启 V18）都落在小跳模态
#      ⇒ 说明"热启动"不是唯一原因，"小跳在早期无价"才是共同点。
#
# 早期监测（前 ~1k 轮）：
#   * ⭐ `Step/air_time_single_stance`：**主指标**，应从 0.011 s 明显上爬（0.4~1.0 是目标）；
#   * ⭐ `Step/air_time_hop_frac`：**本轮新探针**（`d < 0.1 s` 占比），应显著下降；
#   * `Step/air_time_in_band_frac` 应 > 0.01（V19a 是 0.0005~0.0015）；
#   * `Episode_Reward/feet_air_time`：上限 ≈2.4/步，V19a 只拿 0.038；
#   * `Step/amp_style_zero_frac` 应从 0.050 掉回 ≈ 0（k 放宽的直接验证）；
#   * 守住 `track_lin_vel_xy_exp ≥ 1.0`、`tilt_deg_mean ≤ 12°`、`Mean episode length`。
#
# 用法（云端）：
#   pgrep -af 'train[.]py'        # 必须为空
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v19b.sh' /root/train_amp_walk_v19b_4096.log
#
# 手册：$1 num_envs（默认 4096） $2 iters（默认 10^9） $3 device $4 tag
#       $5 load_run（默认空 = 从零） $6 checkpoint（默认空）
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v19b_scratch_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-}"
TASK="Instinct-Amp-Walk-V19b-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
