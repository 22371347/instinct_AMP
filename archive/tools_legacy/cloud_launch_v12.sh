#!/usr/bin/env bash
# ============================================================================
# 一键：**停 V11 → 起 V12**（跑在云端）
# ============================================================================
# 为什么固化成文件：这条命令要经 `local python tool → ssh → cloud_run_bg.sh →
# bash -c → train.py` 五层传递，命令行里引号一旦被拆坏就会静默起错任务
# （本项目已踩过多次）。写进文件再 `bash /root/cloud_launch_v12.sh` 就只有一个 token。
#
# V12 = V11 逐字 + ① AMP 判别器输入 10 帧 → **2 帧**（670 → 134 维，论文 (s_t,s_{t+1})）
#              + ② `track_default_pose_on_stand` −0.3 → **−1.0**
#
# ⚠️ 判别器**在** checkpoint 里 ⇒ 换输入维度必须用 `ckpt_manipulator` 丢掉它
#    （V12 的 agent cfg 已设；否则启动即 `size mismatch [1024,670] vs [1024,134]` 崩）。
# ============================================================================
set -uo pipefail

#: hot start 来源：V11 那次**正在跑**的 run 目录（最新 ckpt = model_9000.pt）
V11_RUN="20260930_093037_G1AmpWalkV11MVPPlane_obs378_rew21_airtime8_ar0.06_bigstep_fromV10b7500_amp_walk_v11_4096_from20260929_232408"
V11_CKPT="model_9000.pt"

echo "=== 1) 停 V11 ==="
pkill -9 -f 'train[.]py' || true
pgrep -af 'train[.]py' || echo "  ✅ 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv || true

echo "=== 3) 起 V12（hot start from V11 ${V11_CKPT}）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v12.sh 4096 1000000000 cuda:0 amp_walk_v12_4096 $V11_RUN $V11_CKPT" \
  /root/train_amp_walk_v12_4096.log
echo "=== done ==="
