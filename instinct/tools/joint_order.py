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

#: 训练数据 npz 的 `joint_names` 顺序（AMP 参考动作序）。
#: 取自 `.../config/g1/motions/amp_walk/*.npz` 的 `joint_names` 字段：
#: 左腿(6) → 右腿(6) → 腰(3) → 左臂(7) → 右臂(7)，即 Unitree G1 URDF/USD 的声明序。
#: ⚠️ 它**既不等于** `ISAACLAB_ORDER`（肩优先）**也不等于** `MUJOCO_ORDER`（腰优先），
#: 所以「动作」与「关节位置/速度观测」必须用各自的映射。
NPZ_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
assert len(NPZ_ORDER) == 29, len(NPZ_ORDER)
assert set(NPZ_ORDER) == set(ISAACLAB_ORDER), set(NPZ_ORDER) ^ set(ISAACLAB_ORDER)

#: ⭐ P1 臂（官方 `g1_29dof_rev_1_0`，**pelvis 为根**，`Instinct-Amp-Walk-Pelvis-G1-v0`）
#: 的 IsaacLab 侧 articulation 顺序。**云端实测**，不是推出来的：
#:
#: 为什么会变：IsaacLab 的关节序 = 「以根节点为起点逐层向下」的遍历序
#: （每层左右成对，层内按父节点被访问的先后）⇒ **换根必然改序**。
#:   · 旧资产根 = `torso_link` ⇒ 肩部那一层先被访问 → "肩优先、左右交错"（ISAACLAB_ORDER）
#:   · 新资产根 = `pelvis`    ⇒ 髋部那一层先被访问 → 下面这条"髋优先、腰夹在腿中间"
#: 实测来源：2026-10-13 云端 P1 冒烟自检打印的真实 articulation 序，
#: 且与 AMP 专家集导出脚本读到的 `joint_names` 逐项一致（29/29）。
#: ⚠️ 这个顺序**无法从 URDF/XML 文件推出来**（URDF 里是 腿→腰→臂 声明序，
#:    与 IsaacLab 实际给出的完全不同），只能靠实测 —— 改资产后必须重新测。
PELVIS_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)
assert len(PELVIS_ORDER) == 29, len(PELVIS_ORDER)
assert set(PELVIS_ORDER) == set(MUJOCO_ORDER), set(PELVIS_ORDER) ^ set(MUJOCO_ORDER)

#: 按名字取顺序：**唯一真相源**。`g1_mujoco_play.py --asset` 与观测/动作重排
#: 都从这里查表，避免在多个文件里各写一份。
ORDER_CANDIDATES: dict[str, tuple[str, ...]] = {
    "isaaclab": ISAACLAB_ORDER,     # 旧资产(torso 根)：肩优先交错
    "omniloco": ISAACLAB_ORDER,     # == articulation 实测值（旧资产）
    "npz": NPZ_ORDER,               # AMP 参考动作 npz 的序（腿优先）
    "articulation": NPZ_ORDER,      # 历史别名（旧资产下确实等于 npz）
    "mjcf": MUJOCO_ORDER,           # MJCF ctrl 序
    "p1": PELVIS_ORDER,             # ⭐ 新资产(pelvis 根)：髋优先
    "pelvis": PELVIS_ORDER,         # 同义别名
}


def get_order(name: str) -> tuple[str, ...]:
    """按名字取关节顺序；未知名字给出可选列表。"""
    if name not in ORDER_CANDIDATES:
        raise KeyError(f"未知关节序 '{name}'，可选：{sorted(ORDER_CANDIDATES)}")
    return ORDER_CANDIDATES[name]


def mujoco_state_order_from_model(model) -> list[str]:
    """MJCF 的 **状态序**：`qpos[7:]` / `qvel[6:]` 的顺序（按 `jnt_qposadr` 排，跳过 free joint）。

    ⚠️⚠️ **状态序 ≠ ctrl 序**（这是一个会让回放直接抽搐的坑）：
      · 旧 popsicle 资产：两者恰好相同（都是"腰 → 左腿 → 右腿 → 左臂 → 右臂"）
        ⇒ 以前把 `MUJOCO_ORDER` 当状态序用，看不出问题；
      · 官方 `g1_29dof_rev_1_0`：**状态序 = URDF 声明序**（左腿 → 右腿 → 腰 → 左臂 → 右臂，
        即 `NPZ_ORDER`），而 **ctrl 序 = 执行器声明序**（腰 → 腿 → 臂，即 `MUJOCO_ORDER`）。
      ⇒ 观测里 `joint_pos_rel/joint_vel` 读 `qpos/qvel`，**必须**用本函数给的状态序；
        写 `ctrl` 才用 `MUJOCO_ORDER`（见 `policy_to_mujoco_index`）。两处不能混。
    """
    import mujoco

    pairs: list[tuple[int, str]] = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        pairs.append((int(model.jnt_qposadr[j]), name or f"jnt{j}"))
    return [n for _, n in sorted(pairs)]


def mujoco_state_order_from_xml(xml_path: str) -> list[str]:
    """同 `mujoco_state_order_from_model`，但直接从 XML 路径读。"""
    import mujoco

    return mujoco_state_order_from_model(mujoco.MjModel.from_xml_path(xml_path))


def mj_state_to_ctrl_index(model) -> list[int]:
    """状态序 → ctrl 序 的索引映射（老式手动 PD 用：`q_ctrl = q_state[idx]`）。"""
    state = mujoco_state_order_from_model(model)
    return [state.index(n) for n in MUJOCO_ORDER]


def mjcf_to_order_index(dst_order: "tuple[str, ...] | list[str]") -> list[int]:
    """MJCF(MUJOCO_ORDER) 序 → dst_order 序的索引映射。

    用途：把 `qpos[7:]` / `qvel[6:]`（MJCF 序）重排成策略/观测期望的顺序。
    用法：`vec_dst = vec_mjcf[[mjcf_to_order_index(dst)]]`
    """
    return [MUJOCO_ORDER.index(n) for n in dst_order]


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


def policy_to_mujoco_index(order: str = "isaaclab") -> list[int]:
    """policy 向量(dst 序) → mujoco ctrl 顺序的索引映射。

    用法: `mujoco_ctrl = [policy_vec[i] for i in policy_to_mujoco_index()]`
    ⚠️ `order` 必须与该策略**训练时**的关节序一致：
       旧资产(torso 根) = `isaaclab`；新资产(pelvis 根) = `p1`。
       给错 = 动作完全错位（同一个策略也会摔）。
    """
    return [get_order(order).index(n) for n in MUJOCO_ORDER]


def mujoco_to_policy_index(order: str = "isaaclab") -> list[int]:
    """mujoco 状态向量(MJCF 序) → policy 输入顺序的索引映射。"""
    return [MUJOCO_ORDER.index(n) for n in get_order(order)]


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
    ap.add_argument(
        "--order", choices=sorted(ORDER_CANDIDATES), default="isaaclab",
        help="对照表左侧用哪套顺序：isaaclab=旧资产(torso 根)；p1=新资产(pelvis 根)",
    )
    ap.add_argument("--dump", action="store_true", help="打印两侧顺序与映射表")
    args = ap.parse_args(argv)

    _src = get_order(args.order)
    print(f"策略侧顺序 = {args.order}（{len(_src)} DoF）")
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
        print(f"\n--- 顺序对照（{args.order} ↔ mjcf）---")
        print(f"{'#':>3}  {args.order + ' (policy)':<30} {'MuJoCo (ctrl)':<30}")
        for i in range(29):
            print(f"{i:>3}  {_src[i]:<30} {MUJOCO_ORDER[i]:<30}")

        print(f"\n--- policy({args.order}) 向量 → mujoco ctrl 索引 (policy_to_mujoco_index) ---")
        print(policy_to_mujoco_index(args.order))
        print("\n--- 左右镜像索引 (mirror_index) ---")
        print(mirror_index())

    return 0


if __name__ == "__main__":
    sys.exit(main())
