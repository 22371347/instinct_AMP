#!/usr/bin/env python3
"""G1 关节序映射：IsaacLab(InstinctLab) ↔ MuJoCo(InstinctMJ)

为什么需要这个
--------------
InstinctMJ 的 README 明确警告：

> Do not directly use checkpoints trained outside `InstinctMJ` with `InstinctMJ`.
> `InstinctMJ` loads the robot from XML / MJCF, and the resulting joint order is not
> the same as the joint order used in IsaacLab.

即：**IsaacLab 侧训练出的策略，其观测/动作向量是按"肩优先、左右交错"顺序排列的；
MuJoCo 侧按 MJCF 的 body 遍历顺序排列（腰 → 左腿 → 右腿 → 左臂 → 右臂）。**
本地回放时若不重排，关节动作会完全错位。

用法
----
    # 打印 MJCF 权威顺序 vs 已知 IsaacLab 顺序，并给出映射
    python tools/joint_order.py --xml <path/to/g1.xml>

    # 作为库使用
    from joint_order import mujoco_order_from_xml, ISAACLAB_ORDER, reorder
    idx = mj_to_isaaclab_index(xml_path)          # policy 向量 -> mujoco ctrl 顺序
"""
from __future__ import annotations

import argparse
import sys

# ---------------------------------------------------------------------------
# 已知顺序（来自 InstinctMJ README，均带 `_joint` 后缀，共 29 个）
# ---------------------------------------------------------------------------

#: MuJoCo / MJCF 顺序：腰 → 左腿 → 右腿 → 左臂 → 右臂
MUJOCO_ORDER: tuple[str, ...] = (
    "waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

#: IsaacLab（旧 InstinctLab 约定）顺序：肩优先、左右交错 —— 训练时策略向量的排列
ISAACLAB_ORDER: tuple[str, ...] = (
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "waist_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "waist_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "waist_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
)

assert len(MUJOCO_ORDER) == 29, len(MUJOCO_ORDER)
assert len(ISAACLAB_ORDER) == 29, len(ISAACLAB_ORDER)
assert set(MUJOCO_ORDER) == set(ISAACLAB_ORDER)


# ---------------------------------------------------------------------------
# 从 MJCF 读出权威顺序
# ---------------------------------------------------------------------------

def mujoco_order_from_xml(xml_path: str) -> list[str]:
    """从 MJCF 读取**受执行器驱动的关节**顺序（即 ctrl 向量的顺序）。"""
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    order: list[str] = []
    for act in range(model.nu):
        # 该执行器作用的关节（transmission=joint 时 trnid[0] 是关节 id）
        jid = int(model.actuator_trnid[act, 0])
        order.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
    return order


def check_against_known(xml_path: str) -> tuple[list[str], list[str], list[str]]:
    """返回 (mjcf_order, missing_in_mjcf, extra_in_mjcf)。"""
    try:
        mj_order = mujoco_order_from_xml(xml_path)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"读取 MJCF 失败: {exc}\n"
            "提示：需在已激活 `instinct` conda 环境（含 mujoco）下运行。"
        ) from None
    missing = [j for j in MUJOCO_ORDER if j not in mj_order]
    extra = [j for j in mj_order if j not in MUJOCO_ORDER]
    return mj_order, missing, extra


# ---------------------------------------------------------------------------
# 映射工具
# ---------------------------------------------------------------------------

def reorder(values: "list | tuple", src_order: tuple[str, ...], dst_order: tuple[str, ...]):
    """按**名字**把 values 从 src_order 排列重排为 dst_order 排列。"""
    if len(values) != len(src_order):
        raise ValueError(f"values 长度 {len(values)} != 源顺序长度 {len(src_order)}")
    by_name = dict(zip(src_order, values))
    return [by_name[n] for n in dst_order]


def policy_to_mujoco_index() -> list[int]:
    """policy 向量(IsaacLab 序) → mujoco ctrl 顺序的索引映射。

    用法: `mujoco_ctrl = [policy_vec[i] for i in policy_to_mujoco_index()]`
    """
    return [ISAACLAB_ORDER.index(n) for n in MUJOCO_ORDER]


def mujoco_to_policy_index() -> list[int]:
    """mujoco 状态向量(MJCF 序) → policy 输入顺序的索引映射。"""
    return [MUJOCO_ORDER.index(n) for n in ISAACLAB_ORDER]


def mirror_index() -> list[int]:
    """左右镜像映射（用于对称性检查）：返回每个关节的镜像关节在 MUJOCO_ORDER 中的下标。"""
    out: list[int] = []
    for name in MUJOCO_ORDER:
        if name.startswith("left_"):
            twin = "right_" + name[len("left_"):]
        elif name.startswith("right_"):
            twin = "left_" + name[len("right_"):]
        else:
            twin = name  # waist_* 自映射
        out.append(MUJOCO_ORDER.index(twin))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", help="G1 MJCF 路径（用于读取权威顺序）")
    ap.add_argument("--dump", action="store_true", help="打印两侧顺序与映射表")
    args = ap.parse_args(argv)

    print(f"IsaacLab 顺序 (策略向量): {len(ISAACLAB_ORDER)} DoF")
    print(f"MuJoCo  顺序 (ctrl 向量): {len(MUJOCO_ORDER)} DoF")

    if args.xml:
        mj_order, missing, extra = check_against_known(args.xml)
        print(f"\n[从 MJCF 读取] act 数量 = {len(mj_order)}")
        ok = not missing and not extra and mj_order == list(MUJOCO_ORDER)
        if ok:
            print("✅ MJCF 顺序与已知 MUJOCO_ORDER 完全一致")
        else:
            print("⚠️  MJCF 顺序与内置常量不一致，请以 MJCF 为准：")
            if missing:
                print(f"   MJCF 中缺失: {missing}")
            if extra:
                print(f"   MJCF 中多出: {extra}")
            print("   MJCF  实际顺序: " + ", ".join(mj_order))

    if args.dump or not args.xml:
        print("\n--- 顺序对照 ---")
        print(f"{'#':>3}  {'IsaacLab (policy)':<30} {'MuJoCo (ctrl)':<30}")
        for i in range(29):
            print(f"{i:>3}  {ISAACLAB_ORDER[i]:<30} {MUJOCO_ORDER[i]:<30}")

        print("\n--- policy 向量 → mujoco ctrl 索引 (policy_to_mujoco_index) ---")
        print(policy_to_mujoco_index())
        print("\n--- 左右镜像索引 (mirror_index) ---")
        print(mirror_index())

    return 0


if __name__ == "__main__":
    sys.exit(main())
