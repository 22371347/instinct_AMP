#!/usr/bin/env bash
# ============================================================================
# P1（换根）云端发车 / 冒烟 / 专家集重导 —— 官方 G1 rev1.0（**pelvis 为根**）
# ============================================================================
# TASK = Instinct-Amp-Walk-Pelvis-G1-v0      TAG = p1_pelvis_official_rev10
#
# 与 T2/T5 的**唯一区别**（细节见 `amp_walk_pelvis_env_cfg.py`）：
#   资产 = official_rev10（pelvis 根）/ init_z = 0.7585 / 高程图挂 pelvis /
#   运动参考挂 pelvis + 数据 = motions/amp_walk_pelvis（base 已换根）/ 判别器 anchor = pelvis /
#   关节序 = PELVIS_JOINT_ORDER
#
# 三种模式（$3）：
#   smoke  —— 只起 2 迭代：**唯一目的**是把契约校验打出的"实际 articulation 关节序"
#             抓回来（新资产的顺序无法推导）。若与 `PELVIS_JOINT_ORDER` 不符，
#             日志里会打印实际顺序 ⇒ 回填常量即可（镜像映射会自动跟着对）。
#   export —— 重导离线专家集（anchor 从 torso 变 pelvis ⇒ 内容变了，维度仍 396）。
#   run    —— 正式开训（**从零**，不 resume；旧 checkpoint 语义已不兼容）。
#
# 用法（云端）：
#   bash /root/cloud_launch_pelvis_remote.sh 1000000000 4096 smoke
#   bash /root/cloud_launch_pelvis_remote.sh 1000000000 4096 export
#   bash /root/cloud_launch_pelvis_remote.sh 1000000000 4096 run
# ============================================================================
set -uo pipefail

BUDGET="${1:-1000000000}"
INST="${2:-4096}"
MODE="${3:-smoke}"
TASK="Instinct-Amp-Walk-Pelvis-G1-v0"
TAG="p1_pelvis_official_rev10"
LOG="/root/train_${TAG}.log"
GUARD_LOG="/root/guard_${TAG}.log"
EXPERT="/root/amp_expert_states_396_pelvis.pt"
#: 专家集**必须**是这个 anchor（换根臂 = pelvis）；导出后按内容校验，
#: 只看"文件存在"会被上一次留下的旧文件骗过（2026-10-13 实测踩过）。
EXPERT_ANCHOR="pelvis"
R=/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab
D="$R/tasks/locomotion/config/g1"
A="$R/assets/resources/unitree_g1/official_rev10"
PY=/opt/conda/envs/isaaclab/bin/python

echo "=== 模式：$MODE（TASK=$TASK  INST=$INST  BUDGET=$BUDGET）==="

# ---------------------------------------------------------------------------
echo "=== 0) 新资产 / 新配置在位检查 ==="
fail=0
for f in "$R/assets/unitree_g1.py" "$A/g1_29dof_rev_1_0.urdf" \
         "$D/amp_walk_pelvis_env_cfg.py" "$D/agents/amp_walk_pelvis_ppo_cfg.py" \
         "$D/__init__.py" \
         "$D/motions/amp_walk_pelvis/amp_walk_v13_motion_selection.yaml"; do
  if [[ -f "$f" ]]; then echo "  OK   $f"; else echo "  FAIL 缺 $f" >&2; fail=1; fi
done
n_mesh=$(ls "$A/meshes" 2>/dev/null | wc -l)
n_npz=$(ls "$D/motions/amp_walk_pelvis"/*_retargeted.npz 2>/dev/null | wc -l)
echo "  网格数 = $n_mesh（应 35）   换根 npz 数 = $n_npz（应 17）"
[[ "$n_mesh" -eq 35 && "$n_npz" -eq 17 ]] || { echo "  FAIL 资产/数据不完整" >&2; fail=1; }
grep -q "G1_29DOF_REV10_PELVIS_CFG" "$R/assets/unitree_g1.py" \
  && echo "  OK   资产 cfg 已在位" || { echo "  FAIL 资产 cfg 缺失" >&2; fail=1; }
grep -q "p1_pelvis_official_rev10_cold" "$D/agents/amp_walk_pelvis_ppo_cfg.py" \
  && echo "  OK   agent cfg 已在位" || { echo "  FAIL agent cfg 缺失" >&2; fail=1; }
[[ "$fail" -eq 0 ]] || exit 1

# ---------------------------------------------------------------------------
if [[ "$MODE" == "export" ]]; then
  echo "=== EXPORT：重导离线专家集（anchor = pelvis，维度仍 396）→ $EXPERT ==="
  for p in $(pgrep -f 'train[.]py'); do echo "  kill -9 $p"; kill -9 "$p"; done
  sleep 2
  # ⚠️ 用 cloud_run_bg.sh 跑：它会 source /root/.llp_env.sh 并 cd 到 /root/gpufree-data/instinct_xu
  #    （导出工具要从那里 import instinctlab/instinct_rl；直接 ssh 前台跑会丢 PYTHONPATH 与 cwd）
  bash /root/cloud_run_bg.sh \
    "AMP_EXPORT_CKPT='$EXPERT' /opt/conda/envs/isaaclab/bin/python -u /root/cloud_export_amp_expert.py --task '$TASK' --num_envs 256 --live_steps 30 --with_anchor_gravity" \
    /root/export_${TAG}.log
  echo "  （导出中；最多等 600s）"
  for _ in $(seq 1 120); do
    grep -qaE "导出完成|已写出|Traceback|Error" /root/export_${TAG}.log 2>/dev/null && break
    sleep 5
  done
  tail -25 /root/export_${TAG}.log

  # ⭐⭐ 内容级校验（**只看"文件存在"会被上一次的旧文件骗过** —— 2026-10-13 实测踩到：
  #      第一次导出留下了一个 anchor=torso_link 的旧文件，光判存在会误判为成功）
  /opt/conda/envs/isaaclab/bin/python - "$EXPERT" "$EXPERT_ANCHOR" <<'PYEOF'
import sys
import torch
p, want = sys.argv[1], sys.argv[2]
m = torch.load(p, map_location="cpu")
anchor = list(m.get("anchor") or [])
shape = tuple(m["states_pair"].shape)
print(f"    [校验] anchor={anchor}（应 [{want}]）  states_pair={shape}  "
      f"num_frames={m.get('num_frames')}  gravity={m.get('with_anchor_gravity')}")
ok = anchor == [want] and shape[1] == 396
raise SystemExit(0 if ok else 1)
PYEOF
  if [[ $? -ne 0 ]]; then
    echo "  ❌ 专家集**内容**校验失败：anchor ≠ $EXPERT_ANCHOR（导出工具没生效，或这是上一次的旧文件）" >&2
    exit 1
  fi
  echo "  ✅ 专家集内容校验通过：anchor=$EXPERT_ANCHOR、维度 396"
  exit 0
fi

# ---------------------------------------------------------------------------
if [[ "$MODE" == "smoke" ]]; then
  echo "=== SMOKE：起 2 迭代（只为拿\"实际关节序\"+ 验证配置能建起来）==="
  for p in $(pgrep -f 'train[.]py'); do kill -9 "$p"; done
  sleep 2
  bash /root/cloud_run_bg.sh \
    "bash /root/run_amp_walk_v33.sh 64 2 cuda:0 'p1_pelvis_smoke' '' '' '$TASK'" \
    "/root/train_p1_pelvis_smoke.log"
  echo "  （起车中；下面等 240s 抓关键行）"
  timeout 240 sh -c 'while ! grep -qaE "AMP-WALK PASS|resolved joint order|Traceback" /root/train_p1_pelvis_smoke.log 2>/dev/null; do sleep 5; done'
  echo "-------------------------- 关键行 --------------------------"
  grep -aE "resolved joint order|实际 =|期望 =|AMP-WALK PASS|\[P1\]|\[PELVIS\]|\[pelvis\]|Anchor|anchor|Traceback|Error" -A 3 \
      /root/train_p1_pelvis_smoke.log | head -60
  echo "-----------------------------------------------------------"
  echo "⚠️ 判读："
  echo "  * 若看到 'resolved joint order differs' + '实际 = (...)' ⇒ 把该顺序回填到"
  echo "    amp_walk_pelvis_env_cfg.py::PELVIS_JOINT_ORDER（**只有这一处**），重跑 smoke。"
  echo "  * 若看到 'AMP-WALK PASS' ⇒ 顺序与 PELVIS_JOINT_ORDER 一致，可以进入 export。"
  exit 0
fi

# ---------------------------------------------------------------------------
echo "=== RUN 前置：专家集必须已重导 ==="
if [[ ! -f "$EXPERT" ]]; then
  echo "  ❌ 缺少 $EXPERT —— 先跑：bash /root/cloud_launch_pelvis_remote.sh 1 1 export" >&2
  exit 1
fi
echo "  OK $EXPERT"

echo "=== 1) 停掉当前训练 / 守护 ==="
for p in $(pgrep -f 'train[.]py'); do echo "  kill -9 $p"; kill -9 "$p"; done
for p in $(pgrep -f 'v29_guard[.]sh'); do echo "  kill -9 $p"; kill -9 "$p"; done
sleep 3
pgrep -af 'train[.]py' || echo "  OK 已无 train.py 进程"

echo "=== 2) GPU 现状 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader \
  || { echo "  FAIL 没有可用 GPU" >&2; exit 1; }

echo "=== 3) ⭐ 从零起车（\$5/\$6 传**空** ⇒ 不加 --resume）==="
bash /root/cloud_run_bg.sh \
  "bash /root/run_amp_walk_v33.sh $INST $BUDGET cuda:0 '$TAG' '' '' '$TASK'" \
  "$LOG"

echo "=== 4) 拉起崩溃守护（冷启动早期\"一启动就倒\"是预期形态 ⇒ 放宽）==="
bash /root/cloud_run_bg.sh "bash /root/v29_guard.sh '$LOG' 30 30 30 2.5" "$GUARD_LOG"

echo "=== 5) 等启动自检（最多 480s）==="
timeout 480 sh -c "while ! grep -qaE 'AMP-WALK PASS|Traceback|RuntimeError' '$LOG' 2>/dev/null; do sleep 5; done"
sleep 20
echo "-------------------------- 关键行 --------------------------"
grep -aE 'resume|\[P1\]|\[pelvis\]|AMP-WALK PASS|AMP-WALK WEIGHTS|wasabi|专家|Expert' "$LOG" | head -30
echo "-------------------------- 存活 ----------------------------"
pgrep -af 'train[.]py' | head -1 || echo "  ❌ train.py 已不在"
echo "-------------------------- 报错（若有）----------------------"
grep -aE 'Traceback|KeyError|RuntimeError|ValueError|FileNotFoundError' -A 4 "$LOG" | tail -20 || true
echo "-----------------------------------------------------------"

cat <<EOF

=== 必须核对（P1 首次启动）===
  * ⚠️ **不应**出现 \`resume    : 续训\`（本臂必须冷启动）
  * \`[P1][换根] 官方 G1 rev1.0（pelvis 为根）\` + \`init_z=0.7585\`
  * \`[P1][ALGO] ... mirror_joint_names = PELVIS_JOINT_ORDER ... 专家集 = $EXPERT\`
  * \`[pelvis] amp_policy: N 个 anchor_cfg → pelvis\`
    （**实测 N=3/组**：body_pos_b / body_ori_b / anchor_gravity_b；body_lin_vel_b 与
      body_ang_vel_b **不带 anchor** ⇒ 两组共 6 处）
  * \`[AMP-WALK PASS] policy=1500 ... rewards=12\`
  * 进程存活、无 Traceback

=== 判读纪律 ===
  * 冷启动初期判别器可能是"死的"（\`amp_d_ref\` 贴死 ≈+0.9）⇒ 那种状态下**姿态不对不算反例**。
  * 先确认 ① 判别器活了（\`amp_d_ref\` 掉到 +0.5 量级）、② \`Step/pelvis_height\` 与 0.7585 同量级。
  * 与 T2-cold 对照时注意：**关节序/观测语义已变**，只能比"行为"，不能比"权重"。
EOF
