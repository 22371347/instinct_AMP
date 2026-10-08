#!/usr/bin/env bash
# ============================================================================
# S8 端点监听：等到指定轮次（默认 14000 = 本 run 的第一个检查点）落盘后
#   ① 打印稳定性 / 判别器 / 姿态节奏 / 跟踪 全部读数
#   ② 导出并拉取 actor.onnx + policy_normalizer.npz 到 instinct/eval/<tag>_<iter>/
# 用法：bash instinct/tools/cloud_s8_endpoint.sh [轮次]
# 说明：导出 job 与训练共用一张卡是安全的（S5 在 19:41 导出时训练仍在跑到 20:02）。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

TARGET="${1:-14000}"
RUN_SUBSTR="v33S8"
LOG="/root/train_amp_walk_v33S8_coef4_adaptive_from13500.log"

echo "===== 等待 S8 到 ${TARGET} 轮（每 20s 探一次，最长 ~40min）====="
timeout 2500 python3 instinct/tools/instinct_ssh.py "L=${LOG}; T=${TARGET}; \
for i in \$(seq 1 120); do \
  C=\$(grep -oE 'Learning iteration [0-9]+' \$L 2>/dev/null | tail -1 | awk '{print \$3}'); \
  if [ -n \"\$C\" ] && [ \"\$C\" -ge \$T ]; then break; fi; \
  pgrep -f 'train[.]py' >/dev/null 2>&1 || { echo '!! 训练进程已消失'; break; }; \
  sleep 20; \
done; \
echo '===== S8 读数 ====='; date '+%F %T'; \
grep -oE 'Learning iteration [0-9]+' \$L | tail -1; \
echo '--- 红线（ep_len>=300 / action<=0.7 / reward>=20）---'; \
grep -E 'Mean reward:|Mean episode length:|Mean step action_abs_mean:' \$L | tail -3; \
echo '--- 判别器 / 风格 ---'; \
grep -E 'Mean step amp_d:|Mean step amp_d_ref:|Mean step amp_style_reward:|Mean step amp_style_std:' \$L | tail -4; \
echo '--- 姿态 / 节奏 ---'; \
grep -E 'Mean step tilt_deg_mean:|Mean step foot_sep_mean:|Mean step air_time_single_stance:|Mean step air_time_hop_frac:|Mean step air_time_in_band_frac:' \$L | tail -5; \
echo '--- 跟踪 / LR ---'; \
/opt/conda/envs/isaaclab/bin/python -u /root/cloud_tb_tags.py v33S8 \
  --re 'tracking/vxy_rmse|tracking/wz_rmse|command_bins/vx_high_vxy_rmse|Loss/learning_rate' \
  --since 13500 --tail 1 2>&1 | tail -6; \
pgrep -f 'train[.]py' >/dev/null && echo '训练仍在跑' || echo '!! 训练已停止'" 2>&1 | tail -30

echo
echo "===== 导出并拉取 ONNX（tag=v33S8_${TARGET}）====="
bash instinct/tools/pull_latest_onnx.sh "${RUN_SUBSTR}" "model_${TARGET}.pt" \
  --tag "v33S8_${TARGET}" 2>&1 | tail -45
