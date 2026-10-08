#!/usr/bin/env bash
# ============================================================================
# V30 前置：**导出 mjlab 口径的 AMP 专家数据集 + 采稳态逐维分布**（云端，约 10 分钟）
# ============================================================================
# 两个阶段：
#   ① 导出（~2–3 min）：起 sim（256 envs，不训练）→ 从 `motion_reference` 的数据集张量
#      导出 `(s_t, s_{t+1})` 的 390 维专家状态 → `/root/amp_expert_states.pt`
#      + 逐维 μ/σ 对照 `/root/amp_expert_stats.txt`
#   ② 采稳态（~7 min）：从 V27 model_10000 热启动 V29 训练，探针在 **#5（瞬态）** 与
#      **#50（稳态，episode length≈500）** 各落一份逐维 μ/σ → `/root/amp_dim_probe_steady_*.txt`
#      然后自动停训。
#
# 产物打包：`/root/amp_export_bundle.tgz`
#
# 用法：bash /root/probe_and_export.sh
# ============================================================================
set -uo pipefail

RUN_ROOT=/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk
SRC=/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab
PY=/opt/conda/envs/isaaclab/bin/python
EXPORT_LOG=/root/amp_export_result.txt
TRAIN_LOG=/root/train_amp_walk_v29_steady.log
BUNDLE=/root/amp_export_bundle.tgz

pkill -9 -f 'train[.]py' || true
sleep 2
rm -f /root/amp_dim_probe_steady_*.txt /root/amp_expert_states.pt /root/amp_expert_stats.txt

echo "=== ① 导出 AMP 专家数据集（不训练）==="
cd "$SRC" || exit 1
source /root/.llp_env.sh 2>/dev/null || true
unset PYTHONPATH
"$PY" -u /root/cloud_export_amp_expert.py \
  --task Instinct-Amp-Walk-V29-G1-v0 --num_envs 256 --live_steps 30
RC=$?
echo "  导出退出码 RC=$RC（日志 $EXPORT_LOG）"
if [[ "$RC" -ne 0 ]] || ! grep -q '\[AMP-EXPORT PASS\]' "$EXPORT_LOG"; then
  echo "  ⚠️ 导出未通过 ⇒ 打印尾部供排查"
  tail -40 "$EXPORT_LOG" || true
fi
ls -l /root/amp_expert_states.pt /root/amp_expert_stats.txt 2>/dev/null || true

echo
echo "=== ② 采稳态逐维分布（V29 训练 50 个迭代）==="
V27_DIR=""
for d in $(ls -td "$RUN_ROOT"/*v27* 2>/dev/null); do
  if ls "$d"/model_*.pt >/dev/null 2>&1; then V27_DIR="$d"; break; fi
done
if [[ -z "$V27_DIR" ]]; then echo "❌ 找不到 V27 run 目录 ⇒ 跳过第 ② 阶段" >&2; else
  V27_CK=$(ls "$V27_DIR"/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
  echo "  热启动：$(basename "$V27_DIR") / $(basename "$V27_CK")"
  bash /root/cloud_run_bg.sh \
    "bash /root/run_amp_walk_v29.sh 4096 1000000000 cuda:0 amp_walk_v29_steady_4096 '$V27_DIR' $(basename "$V27_CK")" \
    "$TRAIN_LOG"
  echo "  等稳态探针文件（最多 15 分钟）..."
  for i in $(seq 1 90); do
    if [[ -f /root/amp_dim_probe_steady_050.txt ]]; then echo "  ✅ 稳态探针已落盘"; break; fi
    if [[ "$i" -gt 6 ]] && ! pgrep -f 'train[.]py' >/dev/null 2>&1; then
      echo "  ⚠️ train.py 已退出（可能报错）"; break
    fi
    sleep 10
  done
  pkill -9 -f 'train[.]py' || true
  sleep 2
  ls -l /root/amp_dim_probe_steady_*.txt 2>/dev/null || echo "  ❌ 没生成稳态探针"
fi

echo
echo "=== ③ 打包 ==="
cd /root || exit 1
tar czf "$BUNDLE" \
  amp_expert_states.pt amp_expert_stats.txt amp_export_result.txt \
  amp_dim_probe_steady_*.txt $( [[ -f train_amp_walk_v29_steady.log ]] && echo train_amp_walk_v29_steady.log ) \
  2>/dev/null || true
ls -l "$BUNDLE" 2>/dev/null || echo "❌ 打包失败"

echo
echo "=== ④ 关键结论（从导出日志里直接看）==="
grep -E 'σ 放大倍数中位|expert_dataset|live_reference|live_policy|每帧维数|对状态|数据集：|body 解析|anchor=' \
  "$EXPORT_LOG" 2>/dev/null | tail -20 || true
echo "=== done ==="
