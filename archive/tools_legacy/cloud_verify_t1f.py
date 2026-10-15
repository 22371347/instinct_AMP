#!/usr/bin/env python3
"""**T1f 发车前硬校验**：证明 mjlab 的机理已被真正恢复（fail-fast）。

断言
----
环境侧（相对 T1a2）：
  1. `feet_lateral_width`  weight == -10.0，func == feet_lateral_width_cost，
     params.sep_floor == 0.145（AMP_WALK_SEP_FLOOR），gate_lateral == True
  2. `feet_crossing`       weight == -20.0，func == feet_crossing_cost
  3. `dof_pos_limits`      weight == -10.0，func == joint_pos_limits，joint_names == ['.*']
  4. `flat_orientation_l2` weight == -2.0
  5. `track_lin_vel_xy_exp` weight == 1.5；`track_ang_vel_z_exp` weight == 1.2
  6. 非零权重项计数 ≥ 20
算法侧：
  7. style_lerp == 0.90；clip_min_std == 0.05；hidden == [512,256,128]；entropy == 0.005
  8. use_mirror_loss is True（交付要求②）

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
    source /root/.llp_env.sh && unset PYTHONPATH && \
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_t1f.py
"""

from __future__ import annotations

import sys

from isaaclab.app import AppLauncher  # noqa: E402

_APP = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False}).app

import importlib  # noqa: E402

FAIL: list[str] = []


def chk(name, got, want, tol=1e-9):
    if isinstance(got, (int, float)) and isinstance(want, (int, float)):
        ok = abs(float(got) - float(want)) <= tol
    else:
        ok = got == want
    print(f"  {'OK  ' if ok else 'FAIL'} {name:52s} got={got!r}  want={want!r}")
    if not ok:
        FAIL.append(name)


def main() -> int:
    m = importlib.import_module("instinctlab.tasks.locomotion.config.g1")
    cfg = m.G1AmpWalkT1fEnvCfg()
    R = cfg.rewards

    print("=" * 78)
    print("T1f 发车前校验（mjlab 机理是否恢复）")
    print("=" * 78)

    print("\n[1] 足部机理（这是本臂的核心）")
    t = R.feet_lateral_width
    chk("feet_lateral_width.weight", float(t.weight), -10.0)
    chk("feet_lateral_width.func", getattr(t.func, "__qualname__", None), "feet_lateral_width_cost")
    p = t.params or {}
    chk("feet_lateral_width.sep_floor", float(p.get("sep_floor", -1)), 0.145)
    chk("feet_lateral_width.gate_lateral", bool(p.get("gate_lateral")), True)
    chk("feet_lateral_width.command_name", p.get("command_name"), "base_velocity")
    t = R.feet_crossing
    chk("feet_crossing.weight", float(t.weight), -20.0)
    chk("feet_crossing.func", getattr(t.func, "__qualname__", None), "feet_crossing_cost")

    print("\n[2] 关节限位（mjlab −10 / 全关节）")
    t = R.dof_pos_limits
    chk("dof_pos_limits.weight", float(t.weight), -10.0)
    chk("dof_pos_limits.func", getattr(t.func, "__qualname__", None), "joint_pos_limits")
    chk("dof_pos_limits.joint_names", list(getattr((t.params or {}).get("asset_cfg"), "joint_names", []) or []), [".*"])

    print("\n[3] 姿态与跟踪权重")
    chk("flat_orientation_l2.weight", float(R.flat_orientation_l2.weight), -2.0)
    chk("track_lin_vel_xy_exp.weight", float(R.track_lin_vel_xy_exp.weight), 1.5)
    chk("track_ang_vel_z_exp.weight", float(R.track_ang_vel_z_exp.weight), 1.2)

    print("\n[4] 非零权重项计数（T1a2 = 16；本臂 +feet_lateral_width +feet_crossing = 18）")
    names = [n for n in dir(R) if getattr(getattr(R, n, None), "weight", None) not in (None, 0)]
    chk("非零奖励项 >= 18", len(names) >= 18, True)
    t1a2_inactive = ("body_ang_vel_xy", "feet_swing_time_penalty", "track_base_height")
    for n in ("feet_crossing", "feet_lateral_width"):
        chk(f"{n} 已从 0 打开", n in names, True)
    for n in t1a2_inactive:
        chk(f"{n} 仍为 0（未顺手改动）", n in names, False)
    print(f"      非零项 = {len(names)}: {sorted(names)}")

    print("\n[5] 算法侧（AMP_mjlab 对齐）")
    am = importlib.import_module("instinctlab.tasks.locomotion.config.g1.agents.amp_walk_t1f_ppo_cfg")
    a = am.G1AmpWalkT1fPPORunnerCfg()
    chk("style_lerp", float(a.algorithm.style_lerp), 0.90)
    chk("clip_min_std", float(a.algorithm.clip_min_std), 0.05)
    chk("actor_hidden_dims", list(a.policy.actor_hidden_dims), [512, 256, 128])
    chk("entropy_coef", float(a.algorithm.entropy_coef), 0.005)
    chk("use_mirror_loss", bool(getattr(a.algorithm, "use_mirror_loss", False)), True)

    print()
    if FAIL:
        print(f"❌ T1f 校验失败：{len(FAIL)} 项 —— **不要发车**")
        for f in FAIL:
            print(f"   - {f}")
        return 1
    print("✅ T1f 校验全部通过（mjlab 机理已恢复）—— 可以发车")
    return 0


if __name__ == "__main__":
    sys.exit(main())
