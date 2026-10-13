"""**default 站姿对应的躯干/骨盆高度是多少？**（回答 `track_base_height` 的 target 0.82 是否自洽）

    python3 instinct/tools/analyze_default_pose_height.py

动机：`use_default_offset=True` 的动作原点、`track_default_pose_on_stand` 的靶、
以及 `init_state` 的 `pos_z`，**三者是三个独立来源**：
  * 动作原点 / 站姿靶 = `asset.data.default_joint_pos`（= `init_state.joint_pos` 的 KNEES_BENT）
  * `track_base_height` 的 target = `asset.data.default_root_state[:,2]` = **`init_state.pos_z` = 0.82**
  * 0.82 是**手填的生成高度**，不是"这个关节姿势算出来的离地高度" ⇒ 可能不自洽。

本脚本用**我们自己的 MJCF**（`instinct/assets/g1_29dof_popsicle_play.xml`，回放工具同一个模型）
做正运动学，量几种姿势的"躯干/骨盆离地高度"。
"""

from __future__ import annotations

import mujoco
import numpy as np

XML = "instinct/assets/g1_29dof_popsicle_play.xml"

#: `init_state.joint_pos`（= 动作原点 = `track_default_pose_on_stand` 的靶）
DEFAULT_JOINTS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


def _names(m, objtype):
    return [mujoco.mj_id2name(m, objtype, i) for i in range(
        {mujoco.mjtObj.mjOBJ_BODY: m.nbody,
         mujoco.mjtObj.mjOBJ_SITE: m.nsite}.get(objtype, 0))]


def set_pose(m, d, joints: dict[str, float], base_z: float = 0.82):
    d.qpos[:] = m.qpos0
    d.qvel[:] = 0.0
    for name, val in joints.items():
        for j in range(m.njnt):
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jn is None or m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            hit = (name.replace(".*_", "") == jn.split("_", 1)[-1]) if name.startswith(".*_") else (name == jn)
            if hit:
                d.qpos[m.jnt_qposadr[j]] = val
    # 把 free joint 的 base 抬到 base_z（其余保持原样）
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            d.qpos[m.jnt_qposadr[j] + 2] = base_z
    mujoco.mj_forward(m, d)


def report(m, d, title: str) -> None:
    def bid(nm):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)

    def sid(nm):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, nm)

    # 脚部最低点：优先用 site，其次用 ankle_roll body
    foot_z = []
    for nm in ("LL_FOOT", "LR_FOOT", "left_foot", "right_foot"):
        i = sid(nm)
        if i >= 0:
            foot_z.append((nm, float(d.site_xpos[i, 2])))
    for nm in ("left_ankle_roll_link", "right_ankle_roll_link"):
        i = bid(nm)
        if i >= 0:
            foot_z.append((nm + "(body, 需 −0.037 到鞋底)", float(d.xpos[i, 2])))
    sole = min(v for _, v in foot_z) if foot_z else float("nan")
    print(f"\n--- {title} ---")
    for nm in ("torso_link", "pelvis"):
        i = bid(nm)
        if i >= 0:
            print(f"  {nm:26s} z = {d.xpos[i, 2]:.4f} m   （离地 {d.xpos[i, 2] - sole:+.4f}）")
    for nm, v in foot_z:
        print(f"  脚参考点 {nm:44s} z = {v:.4f}")


def main() -> int:
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    print(f"MJCF: {XML}   nq={m.nq} njnt={m.njnt} nbody={m.nbody}")
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            print("free joint 挂在 body:", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                                              m.jnt_bodyid[j]))
    print("sites:", [s for s in _names(m, mujoco.mjtObj.mjOBJ_SITE) if s and "foot" in s.lower()])

    # ① 关节全零 + base 抬到 0.82
    set_pose(m, d, {})
    report(m, d, "① 关节全零（base 抬到 0.82）")

    # ② 我们的 default（KNEES_BENT）
    set_pose(m, d, DEFAULT_JOINTS)
    report(m, d, "② **我们的 default 站姿**（hip −0.312 / knee 0.669 / ankle −0.363）")

    # ③ default + 参考的腰前倾（waist_pitch +0.201）
    set_pose(m, d, {**DEFAULT_JOINTS, "waist_pitch_joint": 0.201})
    report(m, d, "③ default + 腰前倾 0.201（参考中位）")

    # ④ 参考走路的中位姿态（腰 +0.201 / 膝 0.383）
    set_pose(m, d, {**DEFAULT_JOINTS, "waist_pitch_joint": 0.201, ".*_knee_joint": 0.383})
    report(m, d, "④ 参考中位姿态（腰 +0.201 / 膝 0.383）")

    print("\n【判读】`track_base_height` 的 target 是 `default_root_state[:,2]` = 0.82（手填的生成高度）；")
    print("        把②的\"躯干离地高度\"与 0.82 相减，差多少就是\"站姿靶 vs 高度靶\"的不自洽量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
