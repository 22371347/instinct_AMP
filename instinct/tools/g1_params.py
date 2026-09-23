#!/usr/bin/env python3
"""G1 29-DoF (torsoBase_popsicle) 执行器 / 关节参数

来源（唯一真源）
----------------
从 InstinctMJ 源码摘录并按同样公式推导，便于本地回放时**保真**：

    repos/InstinctMJ/src/instinct_mj/assets/unitree_g1.py
      · 常量      ARMATURE_* / NATURAL_FREQ / DAMPING_RATIO       (L139-155)
      · 执行器组  BEYONDMIMIC_G1_29DOF_*                          (L282-345)
      · 初始姿态  _POPSICLE_INIT_STATE                            (L86-104)
      · 速度限位  G1_29DOF_JOINT_VELOCITY_LIMITS                  (L118-131)
      · 动作尺度  beyondmimic_action_scale = 0.25*effort/stiffness (L536-543)

Key facts（易踩坑）
------------------
1. **MJCF 本身 nu=0**：InstinctLab/InstinctMJ 的执行器都是**代码注入**的，
   所以 `g1_29dof_torsobase_popsicle.xml` 直接加载无法控制。
   → 用 `tools/build_g1_mjcf.py` 生成带 `<actuator>` + `<keyframe>` 的可回放 MJCF。
2. **动作尺度是逐关节的**，不是全局 0.25！范围约 0.075 ~ 0.548。
   用全局 0.25 会导致大关节（knee/hip）幅度偏小、小关节（wrist）幅度偏大。
3. `kv`（阻尼）与 `kp`（刚度）均由 `armature` 与 10 Hz 自然频率推出，非手调。
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# 电机常数（与 InstinctMJ 逐字一致）
# ---------------------------------------------------------------------------

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * math.pi  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# ---------------------------------------------------------------------------
# 执行器分组：regex -> (kp, kd, effort_limit, velocity_limit)
# 顺序与 BEYONDMIMIC_G1_29DOF_* 一致；匹配时**先匹配者优先**。
# ---------------------------------------------------------------------------

#: (regex, kp, kd, effort, vlimit, 组名)
ACTUATOR_GROUPS: tuple[tuple[str, float, float, float, float, str], ...] = (
    (r".*_hip_pitch_joint", STIFFNESS_7520_14, DAMPING_7520_14, 88.0, 32.0, "legs_pitch_yaw"),
    (r".*_hip_yaw_joint",   STIFFNESS_7520_14, DAMPING_7520_14, 88.0, 32.0, "legs_pitch_yaw"),
    (r"waist_yaw_joint",    STIFFNESS_7520_14, DAMPING_7520_14, 88.0, 32.0, "waist_yaw"),
    (r".*_hip_roll_joint",  STIFFNESS_7520_22, DAMPING_7520_22, 139.0, 20.0, "legs_roll_knee"),
    (r".*_knee_joint",      STIFFNESS_7520_22, DAMPING_7520_22, 139.0, 20.0, "legs_roll_knee"),
    (r".*_ankle_pitch_joint", 2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 50.0, 37.0, "feet"),
    (r".*_ankle_roll_joint",  2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 50.0, 37.0, "feet"),
    (r"waist_roll_joint",  2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 50.0, 37.0, "waist"),
    (r"waist_pitch_joint", 2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 50.0, 37.0, "waist"),
    (r".*_shoulder_pitch_joint", STIFFNESS_5020, DAMPING_5020, 25.0, 37.0, "arms_5020"),
    (r".*_shoulder_roll_joint",  STIFFNESS_5020, DAMPING_5020, 25.0, 37.0, "arms_5020"),
    (r".*_shoulder_yaw_joint",   STIFFNESS_5020, DAMPING_5020, 25.0, 37.0, "arms_5020"),
    (r".*_elbow_joint",          STIFFNESS_5020, DAMPING_5020, 25.0, 37.0, "arms_5020"),
    (r".*_wrist_roll_joint",     STIFFNESS_5020, DAMPING_5020, 25.0, 37.0, "arms_5020"),
    (r".*_wrist_pitch_joint", STIFFNESS_4010, DAMPING_4010, 5.0, 22.0, "arms_4010"),
    (r".*_wrist_yaw_joint",   STIFFNESS_4010, DAMPING_4010, 5.0, 22.0, "arms_4010"),
)

#: popsicle 默认站姿（关节角），未列出的关节为 0
INIT_JOINT_POS: dict[str, float] = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

#: root（torso_link）初始世界高度
INIT_ROOT_HEIGHT = 0.82


def _match(regex: str, name: str) -> bool:
    import re

    return re.fullmatch(regex, name) is not None


def lookup(joint_name: str, table: dict[str, float]) -> float | None:
    for regex, value in table.items():
        if _match(regex, joint_name):
            return value
    return None


def actuator_for(joint_name: str) -> tuple[float, float, float, float, str] | None:
    """返回 (kp, kd, effort, vlimit, group)。按 ACTUATOR_GROUPS 顺序，先匹配者优先。"""
    for regex, kp, kd, effort, vlim, group in ACTUATOR_GROUPS:
        if _match(regex, joint_name):
            return kp, kd, effort, vlim, group
    return None


def action_scale_for(joint_name: str) -> float:
    """beyondmimic_action_scale = 0.25 * effort / kp。"""
    act = actuator_for(joint_name)
    if act is None:
        raise KeyError(f"没有匹配到执行器组: {joint_name}")
    kp, _, effort, _, _ = act
    return 0.25 * effort / kp


def init_pos_for(joint_name: str) -> float:
    v = lookup(joint_name, INIT_JOINT_POS)
    return 0.0 if v is None else v


def table(joint_names: list[str]) -> list[dict]:
    """生成逐关节参数表，便于打印/导出。"""
    rows = []
    for name in joint_names:
        act = actuator_for(name)
        if act is None:
            rows.append({"joint": name, "group": "(unmatched)", "kp": None})
            continue
        kp, kd, effort, vlim, group = act
        rows.append(
            {
                "joint": name,
                "group": group,
                "kp": kp,
                "kd": kd,
                "effort_limit": effort,
                "velocity_limit": vlim,
                "action_scale": 0.25 * effort / kp,
                "init_pos": init_pos_for(name),
            }
        )
    return rows


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from joint_order import MUJOCO_ORDER  # type: ignore

    print(f"NATURAL_FREQ = {NATURAL_FREQ:.6f} rad/s (10 Hz)")
    print(f"STIFFNESS_7520_14={STIFFNESS_7520_14:.4f}  STIFFNESS_7520_22={STIFFNESS_7520_22:.4f}")
    print(f"STIFFNESS_5020   ={STIFFNESS_5020:.4f}  STIFFNESS_4010   ={STIFFNESS_4010:.4f}")
    print()
    hdr = f"{'#':>3} {'joint':<26} {'group':<16} {'kp':>8} {'kd':>7} {'eff':>6} {'vlim':>5} {'a_scale':>8} {'init':>7}"
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(table(list(MUJOCO_ORDER))):
        print(
            f"{i:>3} {r['joint']:<26} {r['group']:<16} {r['kp']:>8.3f} {r['kd']:>7.4f} "
            f"{r['effort_limit']:>6.0f} {r['velocity_limit']:>5.0f} {r['action_scale']:>8.5f} {r['init_pos']:>7.3f}"
        )
