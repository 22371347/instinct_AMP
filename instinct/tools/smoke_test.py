#!/usr/bin/env python3
"""本地环境冒烟测试（物理合理性，无需 Isaac Sim / 无需 viewer）

验证什么
--------
A. 模型完整性   nu==29、ctrl 顺序、kp/kv 写入、keyframe、世界地面
B. 关节序映射   policy↔mujoco 双向映射自洽（round-trip 恒等）
C. PD 跟踪      给定目标偏移，关节是否朝目标运动（验证执行器与动作尺度生效）
D. 数值稳定性   自由落体+落地 2 s，无 NaN/Inf，穿地深度在容差内
E. 足端贴合     keyframe 姿态下足端离地高度（判断初始根高是否合理）

⚠️ **不验证**：零动作能否静态站立。
   `_POPSICLE_INIT_STATE` 是 RL 的**训练复位姿态**，不是静态平衡姿态；
   零动作下缓慢倾倒属于**预期行为**，不代表环境有问题。

用法
----
    conda activate instinct
    python tools/smoke_test.py
    python tools/smoke_test.py --xml assets/g1_29dof_popsicle_play.xml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = ROOT / "assets/g1_29dof_popsicle_play.xml"

results: list[tuple[bool | None, str, str]] = []


def check(ok: bool | None, name: str, detail: str = "") -> None:
    results.append((ok, name, detail))
    icon = "✅" if ok else ("⚠️ " if ok is None else "❌")
    print(f"  {icon} {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# A. 模型完整性
# ---------------------------------------------------------------------------

def test_model(mujoco, xml: Path) -> tuple[object, object] | None:
    print("\n[A] 模型完整性")
    from g1_params import actuator_for
    from joint_order import MUJOCO_ORDER

    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)

    check(m.nu == 29, "执行器数量", f"nu={m.nu}")
    if m.nu != 29:
        return None

    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0])) for i in range(m.nu)]
    check(list(names) == list(MUJOCO_ORDER), "ctrl 顺序 == MUJOCO_ORDER",
          "" if list(names) == list(MUJOCO_ORDER) else f"首项 {names[0]} != {MUJOCO_ORDER[0]}")

    exp_kp = np.array([actuator_for(n)[0] for n in names])
    exp_kv = np.array([actuator_for(n)[1] for n in names])
    check(np.allclose(m.actuator_gainprm[:, 0], exp_kp, rtol=1e-6), "kp 写入",
          f"{m.actuator_gainprm[:,0].min():.2f}~{m.actuator_gainprm[:,0].max():.2f}")
    check(np.allclose(-m.actuator_biasprm[:, 2], exp_kv, rtol=1e-6), "kv/kd 写入",
          f"{(-m.actuator_biasprm[:,2]).min():.3f}~{(-m.actuator_biasprm[:,2]).max():.3f}")
    check(m.nkey >= 1, "keyframe", f"nkey={m.nkey}")

    has_plane = any(
        int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE) and int(m.geom_bodyid[g]) == 0
        for g in range(m.ngeom)
    )
    check(has_plane, "世界地面", "plane geom 存在" if has_plane else "缺失 → 会穿地掉落")

    is_position = bool(m.actuator_biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE)
    check(is_position, "执行器类型", "position/affine（ctrl=目标关节角）")

    return m, d


# ---------------------------------------------------------------------------
# B. 关节序映射
# ---------------------------------------------------------------------------

def test_joint_order() -> None:
    print("\n[B] 关节序映射自洽性")
    from joint_order import ISAACLAB_ORDER, MUJOCO_ORDER, mujoco_to_policy_index, policy_to_mujoco_index

    p2m = policy_to_mujoco_index()
    m2p = mujoco_to_policy_index()
    check(sorted(p2m) == list(range(29)) and sorted(m2p) == list(range(29)), "映射为置换",
          "双向映射都是 0..28 的排列")

    # round-trip：policy -> mujoco -> policy 应为恒等
    v = np.arange(29, dtype=float)
    round_trip = np.array([v[i] for i in p2m])[m2p]
    check(np.allclose(round_trip, v), "round-trip 恒等", "policy→mj→policy 复原")

    # 语义抽查：MUJOCO_ORDER[0]=waist_pitch，它在 ISAACLAB_ORDER 中位于下标 2
    # p2m[i] = MUJOCO_ORDER[i] 这个关节在 ISAACLAB_ORDER 中的下标
    check(p2m[0] == ISAACLAB_ORDER.index("waist_pitch_joint"), "抽查 mj[0]=waist_pitch",
          f"muJOCO[0] 对应 policy[{p2m[0]}]")
    check(p2m[MUJOCO_ORDER.index("left_shoulder_pitch_joint")] == 0, "抽查 mj[15]=L_shoulder_pitch",
          f"muJOCO[15] 对应 policy[{p2m[MUJOCO_ORDER.index('left_shoulder_pitch_joint')]}]")
    check(ISAACLAB_ORDER[2] == "waist_pitch_joint", "抽查 policy[2]", ISAACLAB_ORDER[2])


# ---------------------------------------------------------------------------
# C. PD 跟踪
# ---------------------------------------------------------------------------

def test_pd_tracking(mujoco, m, d, xml: Path) -> None:
    print("\n[C] PD 跟踪（执行器与动作尺度是否生效）")
    from g1_params import action_scale_for
    from joint_order import MUJOCO_ORDER

    mujoco.mj_resetDataKeyframe(m, d, 0)
    target = m.qpos0.copy()
    # 把首关节（waist_pitch）与左膝各偏移 0.1 rad
    jids = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
    offs = {"waist_pitch_joint": 0.10, "left_knee_joint": 0.10}
    start = {}
    for name, off in offs.items():
        adr = int(m.jnt_qposadr[jids[name]])
        start[name] = d.qpos[adr]
        target[adr] = d.qpos[adr] + off

    # ctrl 顺序 = MUJOCO_ORDER，逐关节写目标
    ctrl = np.array([target[int(m.jnt_qposadr[jids[n]])] for n in MUJOCO_ORDER])
    for _ in range(250):  # 0.5 s
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)

    for name, off in offs.items():
        adr = int(m.jnt_qposadr[jids[name]])
        moved = d.qpos[adr] - start[name]
        frac = moved / off
        ok = frac > 0.5
        check(ok, f"{name} 朝目标运动", f"位移 {moved:+.4f} / 目标 {off:+.2f}（达成 {frac*100:.0f}%）")
    print("     注：达成率 >100% 属正常 —— PD 是弹簧，重力负载下会有稳态偏差"
          "（kp·(target-q) = -重力力矩），故实际落点会略超目标。")


# ---------------------------------------------------------------------------
# D. 数值稳定性 / 穿地
# ---------------------------------------------------------------------------

def test_stability(mujoco, m, d, seconds: float = 2.0) -> None:
    print(f"\n[D] 数值稳定性（零动作自由落体+落地 {seconds}s）")
    mujoco.mj_resetDataKeyframe(m, d, 0)
    n = int(seconds / m.opt.timestep)
    min_z, max_abs_v = np.inf, 0.0
    nan = False
    for _ in range(n):
        d.ctrl[:] = np.array([m.qpos0[int(m.jnt_qposadr[int(m.actuator_trnid[i, 0])])] for i in range(m.nu)])
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            nan = True
            break
        min_z = min(min_z, float(d.qpos[2]))
        max_abs_v = max(max_abs_v, float(np.abs(d.qvel).max()))

    check(not nan, "无 NaN/Inf", "" if not nan else "出现非有限值")
    check(max_abs_v < 100.0, "速度未爆炸", f"max|qvel|={max_abs_v:.2f} rad/s")
    # 穿地容差：躯干高度不应为负（自由落体后应停在地面上）
    deepest = min_z - m.qpos0[2]
    check(deepest > -0.6, "落地未穿透", f"躯干最低 {min_z:.3f} m（相对初值 {deepest:+.3f} m）")
    print("     注：零动作下缓慢倾倒属**预期**（该姿态是训练复位姿态，非静态平衡姿态）")


# ---------------------------------------------------------------------------
# E. 足端贴合
# ---------------------------------------------------------------------------

def test_foot_clearance(mujoco, m, d) -> None:
    print("\n[E] keyframe 姿态下足端离地高度")
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    for nm in ("left_ankle_roll_link", "right_ankle_roll_link"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid < 0:
            check(None, f"{nm} 不存在", "")
            continue
        z = float(d.xpos[bid, 2])
        # ankle_roll_link 原点略高于鞋底，约 3~6 cm 属正常
        check(0.0 < z < 0.12, f"{nm} 离地", f"z={z:.4f} m")


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", default=str(DEFAULT_XML))
    args = ap.parse_args(argv)

    xml = Path(args.xml)
    print("=" * 70)
    print("G1 本地环境冒烟测试")
    print("=" * 70)
    print(f"MJCF: {xml}")

    if not xml.is_file():
        print(f"\n❌ 找不到 {xml}\n先运行: python tools/build_g1_mjcf.py")
        return 1

    try:
        import mujoco
    except ImportError:
        print("\n❌ 未安装 mujoco。请先 `conda activate instinct`")
        return 1

    md = test_model(mujoco, xml)
    test_joint_order()
    if md is not None:
        m, d = md
        test_pd_tracking(mujoco, m, d, xml)
        test_stability(mujoco, m, d)
        test_foot_clearance(mujoco, m, d)

    print("\n" + "=" * 70)
    n_fail = sum(1 for ok, _, _ in results if ok is False)
    n_warn = sum(1 for ok, _, _ in results if ok is None)
    print(f"通过 {len(results) - n_fail - n_warn} / 提示 {n_warn} / 失败 {n_fail}")
    if n_fail:
        print("\n失败项：")
        for ok, name, detail in results:
            if ok is False:
                print(f"  ❌ {name} — {detail}")
        return 1
    print("✅ 本地环境可用（物理合理性验证通过）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
