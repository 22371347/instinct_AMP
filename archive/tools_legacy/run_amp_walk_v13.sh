#!/usr/bin/env bash
# ============================================================================
# V13 启动器（跑在云端）：**全量测试** —— 指令集全向 + 5-clip 参考集
# ============================================================================
# V13 = V12 逐字（2 帧判别器 134 维 / 21 项奖励含 track_default_pose_on_stand=−1 /
#        λ0.5 c0.5 / σ 上限 0.6 / 镜像增广+损失 / fixed LR）之上，只改**两件套**：
#   ① 指令四档（用户指定）：
#        静止/低速 0.20（‖v‖≤0.1，30% 精确 0）
#        前后向     0.25+0.10（vy=0、ωz=0；前向 0.5~0.9 / 均匀 / 后退 (−0.5,−0.3)）
#        纯侧向     0.25（vx=0、vy∈±0.5、ωz=0）
#        斜向/转向  余量 0.20（xyz 全随机）
#   ② 数据集 2 → 5 clip：直行 + 弧线 + 侧移×2 + 后退（A143 不存在，用 walk_backward_loop_001__A022）
#
# 为什么必须同时改：指令要的模态若在参考集里没有密度，判别器会把**合法姿态**判成
# "不像参考"而抑制（V10b 文档里警告过的"任务要的 vs 风格要的打架"）。
#
# ⚠️ 不要和 V12 同时跑（CPU 是瓶颈）。等 V12 出结论后再发车。
# ⚠️ 热启动：V13 继承 V12 的 runner cfg，已带 `ckpt_manipulator`（丢掉 ckpt 里的旧判别器）
#    ⇒ 从 V12/V11 的 ckpt 热启动不会因判别器形状不匹配而崩（详见 v12 agent cfg 的注释），
#    但**建议等 V12 的结论** —— 若 2 帧判别器无效，先修判别器再说。
#
# 判读（≥500 轮）：
#   侧向：`cmd_true_vy` 分布 / 侧移档的 `track_lin_vel_xy_exp`、`feet_crossing`（应≈0，门控关）
#   后退：后退档的 `track_lin_vel_xy_exp`、`feet_air_time`（不许用"双脚后跳"蹭奖励）
#   全向不退化：前进档 `vx_*_rmse`、`amp_d`（判别器要同时容纳 5 种模态，可能变差，看趋势）
#
# 用法（云端）：
#   pgrep -af 'train[.]py'                      # 必须为空
#   ls -t /root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk/*V12*   # 取 V12 run 目录
#   bash /root/cloud_run_bg.sh 'bash /root/run_amp_walk_v13.sh 4096 1000000000 cuda:0 amp_walk_v13_4096 <V12_run_dir> model_<最新>.pt' \
#        /root/train_amp_walk_v13_4096.log
#   # 从零起（不推荐，除非换判别器）：第 5 个参数留空
#
# 手册：$1 num_envs  $2 iters  $3 device  $4 tag  $5 load_run（可选）  $6 checkpoint
# ============================================================================
set -uo pipefail

INST="${1:-4096}"
ITERS="${2:-1000000000}"
DEV="${3:-cuda:0}"
TAG="${4:-amp_walk_v13_4096}"
LOAD_RUN="${5:-}"
CKPT="${6:-model_2000.pt}"
TASK="Instinct-Amp-Walk-V13-G1-v0"

exec bash /root/cloud_train_amp_walk_v3.sh "$INST" "$ITERS" "$DEV" "$TAG" "$LOAD_RUN" "$CKPT" "$TASK"
