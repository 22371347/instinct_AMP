#!/usr/bin/env bash
# =============================================================================
# 云端 IsaacLab **源码级** API 审计（不需要 GPU / 不需要启动 Isaac Sim）
# =============================================================================
#
# 为什么需要它
# ------------
# 本机没有 IsaacLab 源码（只有 instinct_xu/source/instinct_rl 与 instinctlab），
# 所以步骤⑤⑥ 里几个关键假设一直只能靠「文档 + 上游先例」推断。云机器上装着
# 真的 IsaacLab，直接读它的源码就能把这些假设钉死 —— **完全不需要 GPU**。
#
# 要回答的问题
# ------------
#  Q1. `@configclass` 到底怎么收集字段？**不带类型注解**的新类属性算不算字段、
#      `to_dict()` 会不会带上？（这是步骤⑥ 最大的剩余风险：WASABI 参数全是
#      无注解新字段）
#  Q2. `RewardManager` 是按 `cfg.__dict__` 还是按 dataclass fields 遍历奖励项？
#      （同上，决定 G1AmpWalkRewardsCfg 的无注解项会不会生效）
#  Q3. `ArticulationData` 是否有 root_pos_w / root_ang_vel_b / default_root_state /
#      body_pos_w（步骤⑤ 的 track_base_height_exp / track_root_ang_vel_xy_exp /
#      feet_lateral_sep 用到）
#  Q4. `SceneEntityCfg.joint_ids` 在 joint_names=None 时解析成什么？
#      （track_default_pose_on_stand 的 `[:, joint_ids]` 依赖它）
#  Q5. 这个版本的 `ContactSensorCfg` 是否支持 filter_prim_paths_expr，
#      以及是否仍有「只支持一对多」的限制
#  Q6. `load_cfg_from_registry` 用什么把 agent cfg 变成 dict
#
# 用法：
#   bash cloud_api_audit.sh            # 全部
#   bash cloud_api_audit.sh Q1 Q3      # 只跑若干项
# =============================================================================
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y

WANT=("$@")
if [[ ${#WANT[@]} -eq 0 ]]; then WANT=(Q1 Q2 Q3 Q4 Q5 Q6); fi
want() { local x; for x in "${WANT[@]}"; do [[ "$x" == "$1" ]] && return 0; done; return 1; }

hr() { printf '%s\n' "----------------------------------------------------------------"; }
sec() { echo; hr; echo "$1"; hr; }

# --- 定位真正的 isaaclab 库 ---------------------------------------------------
# 镜像把整个 IsaacLab 仓库快照塞进了 site-packages，外面有个「外壳」包；
# 真库在 site-packages/isaaclab/source/isaaclab/isaaclab/
SP=""
for c in /opt/conda/envs/isaaclab/lib/python3*/site-packages "$(python -c 'import site;print(site.getsitepackages()[0])' 2>/dev/null)"; do
  [[ -d "$c" ]] && SP="$c" && break
done
[[ -z "$SP" ]] && { echo "找不到 site-packages"; exit 1; }

IL=""
for c in "$SP/isaaclab/source/isaaclab/isaaclab" "$SP/isaaclab" "$SP/isaaclab/isaaclab"; do
  [[ -d "$c/utils" ]] && IL="$c" && break
done
[[ -z "$IL" ]] && { echo "找不到 isaaclab 库目录（site-packages=$SP）"; exit 1; }

echo "site-packages : $SP"
echo "isaaclab 库   : $IL"
echo "isaaclab 版本 : $(python -c 'import isaaclab;print(getattr(isaaclab,"__version__","?"))' 2>/dev/null)"
python -c "import sys;print('python        :', sys.version.split()[0])"

# ---------------------------------------------------------------------------
if want Q1; then
sec "Q1. @configclass 字段收集语义（无注解类属性算不算字段 / 会不会进 to_dict）"
CFG="$IL/utils/configclass.py"
if [[ -f "$CFG" ]]; then
  echo "[文件] $CFG"
  echo "--- 关键实现片段 ---"
  grep -n "def to_dict\|__dataclass_fields__\|fields(\|__dict__\|dataclass(\|def configclass\|MISSING\|_is_primitive\|annotations" "$CFG" | head -40
  echo
  echo "--- to_dict 全文 ---"
  awk '/def to_dict/,/^$/' "$CFG" | head -40
else
  echo "[缺] $CFG"
  find "$IL" -name "configclass*.py" 2>/dev/null | head
fi
fi

# ---------------------------------------------------------------------------
if want Q2; then
sec "Q2. RewardManager 如何遍历奖励项"
for f in $(find "$IL/managers" -name "reward_manager.py" 2>/dev/null); do
  echo "[文件] $f"
  grep -n "__dict__\|fields(\|_term_cfgs\|for term_name\|cfg_items\|is None\|continue" "$f" | head -30
  echo "--- __init__ 里的遍历 ---"
  awk '/def __init__/,/def /{print NR": "$0}' "$f" | head -45
done
fi

# ---------------------------------------------------------------------------
if want Q3; then
sec "Q3. ArticulationData 的属性"
for f in $(find "$IL/assets" -name "articulation_data.py" 2>/dev/null); do
  echo "[文件] $f"
  for attr in root_pos_w root_quat_w root_lin_vel_w root_ang_vel_w root_ang_vel_b root_lin_vel_b default_root_state root_state_w body_pos_w body_quat_w body_names joint_names; do
    hit=$(grep -c "\b$attr\b" "$f")
    printf '  %-22s %s\n' "$attr" "$([[ $hit -gt 0 ]] && echo "存在 ($hit 处)" || echo '❌ 不存在')"
  done
  echo "--- root_pos_w / root_ang_vel_b 的定义 ---"
  grep -n "def root_pos_w\|def root_ang_vel_b\|root_pos_w:\|root_ang_vel_b:" "$f" | head -10
done
fi

# ---------------------------------------------------------------------------
if want Q4; then
sec "Q4. SceneEntityCfg.joint_ids / body_ids 的解析"
for f in $(find "$IL/managers" -name "scene_entity_cfg.py" 2>/dev/null); do
  echo "[文件] $f"
  grep -n "preserve_order\|def resolve\|joint_ids\|body_ids\|resolve_matching_names" "$f" | head -30
  echo "--- resolve 全文 ---"
  awk '/def resolve/,0' "$f" | head -60
done
for f in $(find "$IL/utils" -name "string.py" 2>/dev/null); do
  echo "--- resolve_matching_names ($f) ---"
  grep -n "def resolve_matching_names" -A 40 "$f" | head -50
done
fi

# ---------------------------------------------------------------------------
if want Q5; then
sec "Q5. ContactSensorCfg.filter_prim_paths_expr 与一对多限制"
for f in $(find "$IL/sensors/contact_sensor" -name "*.py" 2>/dev/null); do
  echo "[文件] $f"
  grep -n "filter_prim_paths_expr" "$f" | head -10
done
echo "--- 一对多限制的原文 ---"
grep -rn "one-to-many\|single primitive per environment\|only one" "$IL/sensors/contact_sensor/" 2>/dev/null | head -10
fi

# ---------------------------------------------------------------------------
if want Q6; then
sec "Q6. agent cfg -> dict 的路径"
grep -rn "load_cfg_from_registry\|to_dict()" "$IL/utils/parse_cfg.py" "$IL/utils/"*.py 2>/dev/null | head -20
find "$IL" -name "parse_cfg.py" 2>/dev/null | while read -r f; do
  echo "[文件] $f"
  awk '/def load_cfg_from_registry/,/return cfg/' "$f" | head -40
done
fi

echo; hr; echo "审计结束"
