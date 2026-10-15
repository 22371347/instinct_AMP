#!/usr/bin/env bash
# ============================================================================
# P1（换根）一键发车 —— 本地侧：同步 → 冒烟（拿关节序）→ 重导专家集 → 正式开训
# ============================================================================
# TASK = Instinct-Amp-Walk-Pelvis-G1-v0      TAG = p1_pelvis_official_rev10
#
# 用法：
#   bash instinct/tools/launch_pelvis.sh smoke     # ① 同步 + 云端 2 迭代冒烟（拿实际关节序）
#   bash instinct/tools/launch_pelvis.sh export    # ② 重导离线专家集（anchor = pelvis）
#   bash instinct/tools/launch_pelvis.sh run       # ③ 正式开训（从零）
#
# 为什么必须先 smoke：
#   新资产的 articulation 关节序**无法推导**（IsaacLab 转 USD 的顺序既不等于 URDF 的
#   `<joint>` 声明序，也不等于 link 序）。契约校验会 fail-fast 并打印**实际顺序**
#   ⇒ 把它回填到 `PELVIS_JOINT_ORDER`（唯一一处），再 export/run。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODE="${1:-smoke}"
TAG="p1_pelvis_official_rev10"
L=instinct_xu/source/instinctlab/instinctlab
R=/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab
SSH=(python3 instinct/tools/instinct_ssh.py)

echo "=== 0) 连通性检查 ==="
"${SSH[@]}" "echo 实例在线" || { echo "❌ 实例未开机 / 端口变了"; exit 1; }

echo "=== 1) 同步代码 + 资产 + 数据 ==="
# 1.1 资产 cfg（新增 G1_29DOF_REV10_PELVIS_CFG）
"${SSH[@]}" --put "$L/assets/unitree_g1.py" "$R/assets/unitree_g1.py" || exit 1
# 1.2+1.3 官方 rev1.0 资产 + 换根后的参考数据：**打包上传 + 云端先删后解**
#   （`scp -r <dir> <已存在的 dir>` 会在目标里套一层同名子目录，重复执行就 dir/dir/…）
TGZ=/tmp/pelvis_p1_assets.tgz
tar czf "$TGZ" -C "$L" \
    assets/resources/unitree_g1/official_rev10 \
    tasks/locomotion/config/g1/motions/amp_walk_pelvis
echo "  资产/数据包：$(du -h "$TGZ" | cut -f1)"
"${SSH[@]}" --put "$TGZ" /root/pelvis_p1_assets.tgz || exit 1
"${SSH[@]}" "rm -rf '$R/assets/resources/unitree_g1/official_rev10' '$R/tasks/locomotion/config/g1/motions/amp_walk_pelvis' && tar xzf /root/pelvis_p1_assets.tgz -C '$R' && echo -n '  网格数：' && ls '$R/assets/resources/unitree_g1/official_rev10/meshes' | wc -l && echo -n '  换根 npz 数：' && ls '$R/tasks/locomotion/config/g1/motions/amp_walk_pelvis'/*_retargeted.npz | wc -l" || exit 1
# 1.4 env / agent / 注册 / 契约基类（关节序期望值可被子类覆盖）
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/amp_walk_pelvis_env_cfg.py" \
                "$R/tasks/locomotion/config/g1/amp_walk_pelvis_env_cfg.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/agents/amp_walk_pelvis_ppo_cfg.py" \
                "$R/tasks/locomotion/config/g1/agents/amp_walk_pelvis_ppo_cfg.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/omni_loco_env_cfg.py" \
                "$R/tasks/locomotion/config/g1/omni_loco_env_cfg.py" || exit 1
"${SSH[@]}" --put "$L/tasks/locomotion/config/g1/__init__.py" \
                "$R/tasks/locomotion/config/g1/__init__.py" || exit 1
echo "=== 1b) 上传云端脚本（发车 + 专家集导出工具）==="
"${SSH[@]}" --put "instinct/tools/cloud_launch_pelvis_remote.sh" /root/cloud_launch_pelvis_remote.sh || exit 1
"${SSH[@]}" --put "instinct/tools/cloud_export_amp_expert.py" /root/cloud_export_amp_expert.py || exit 1
echo "  OK"
echo "  OK 全部已上传（资产 1 个 urdf + 35 网格 + 17 npz + 6 yaml + 5 个 py + 1 个 sh）"

echo "=== 2) 云端执行：$MODE ==="
"${SSH[@]}" "bash /root/cloud_launch_pelvis_remote.sh 1000000000 4096 $MODE" || exit 1

if [[ "$MODE" == "smoke" ]]; then
  cat <<'EOF'

=== 下一步 ===
  * 看上面 \`resolved joint order differs\` 的 \`实际 = (...)\`：
      - 若与 PELVIS_JOINT_ORDER 一致 → 直接跑 export
      - 不一致 → 把实际顺序回填到
        instinct_xu/.../config/g1/amp_walk_pelvis_env_cfg.py::PELVIS_JOINT_ORDER（**唯一一处**）
        然后 `bash instinct/tools/launch_pelvis.sh smoke` 重跑确认
  * 顺序确认后：
      bash instinct/tools/launch_pelvis.sh export     # 重导专家集（anchor = pelvis）
      bash instinct/tools/launch_pelvis.sh run        # 正式开训（从零）
EOF
fi
