#!/usr/bin/env python3
"""Step 0 汇总：把 σ 扫描落盘的序列整理成"σ → 噪声税/探针/步态"的曲线。"""
import json
import math
import statistics as st

SRC = "/root/sigma_scan_series.json"
d = json.load(open(SRC))

# 排序：按 σ_eff 从大到小
modes = sorted(d.items(), key=lambda kv: -kv[1]["sigma_eff"])
ref = d.get("mean")

KEYSET = ["R:action_rate_l2", "R:dof_acc_l2", "R:dof_pos_limits",
          "R:track_default_pose_on_stand", "R:feet_slide", "R:dof_torques_l2",
          "R:flat_orientation_l2", "R:lin_vel_z_l2", "R:joint_deviation_torso",
          "R:joint_deviation_arms", "R:joint_deviation_hip", "R:joint_deviation_knee",
          "R:termination_penalty", "R:feet_lateral_width", "R:feet_crossing"]
#: ⚠️ T2（全权照抄 mjlab）把两个跟踪项改了名字（`track_lin_vel_xy_exp` →
#: `track_anchor_linear_velocity`、`track_ang_vel_z_exp` → `track_anchor_angular_velocity`）
#: ⇒ 两张名字都留着（不存在的项由 `m0` 记 0，见下）。
POS = ["R:track_lin_vel_xy_exp", "R:track_anchor_linear_velocity",
       "R:track_base_height", "R:feet_air_time",
       "R:body_ang_vel_xy", "R:track_ang_vel_z_exp", "R:track_anchor_angular_velocity"]
PROBES = ["hf_acc_all", "hf_acc_legs", "hf_acc_arms",
          "stand_pose_vel_err", "stand_pose_pos_err",
          "air_time_hop_frac", "air_time_in_band_frac", "air_time_single_stance",
          "air_time_lift_frac", "air_time_progress_frac"]


def m(v, k):
    s = v["series"].get(k)
    return st.mean(s) if s else float("nan")


def m0(v, k):
    """同 `m`，但**缺失项记 0** —— 只用于"按项求和"的账目。

    ⚠️ 必须区分两种缺失：该臂**主动删掉了**某一项（如 T2 照着 mjlab 删了
    `feet_air_time`/`flat_orientation_l2`/`joint_deviation_*`）时，它在账目里就该是 0；
    而探针类缺失仍应报 `nan`（别把"没测"当成"测得 0"）⇒ 探针走 `m`、求账走 `m0`。
    （之前统一返回 nan，会让 T2 的"任务项合计"整列变成 NaN。）
    """
    s = v["series"].get(k)
    return st.mean(s) if s else 0.0


def pen_sum(v):
    return sum(m0(v, k) for k in KEYSET)


def pos_sum(v):
    return sum(m0(v, k) for k in POS)


print("=" * 132)
print("① σ → 奖励账目（manager 单位/步；罚项=负项合计，任务=正项合计；env奖励=env.step 返回值）")
print("=" * 132)
hdr = f"{'σ_eff':>7}{'罚项合计':>11}{'任务项合计':>12}{'净':>10}{'env奖励':>10}{'env标准差':>11}" \
      f"{'噪声税':>11}{'税/清洁任务':>12}{'dof_acc':>10}{'act_rate':>10}{'pose罚':>9}{'lim罚':>9}"
print(hdr)
ref_net = pen_sum(ref) + pos_sum(ref) if ref else float("nan")
clean_task = pos_sum(ref) if ref else float("nan")
for name, v in modes:
    pen, pos = pen_sum(v), pos_sum(v)
    net = pen + pos
    tax = net - ref_net if ref else float("nan")
    er, sd = m(v, "_env_reward"), m(v, "_env_reward_std")
    ratio = tax / clean_task if clean_task else float("nan")
    print(f"{v['sigma_eff']:>7.3f}{pen:>11.3f}{pos:>12.3f}{net:>10.3f}{er:>10.4f}{sd:>11.4f}"
          f"{tax:>11.3f}{ratio * 100:>11.0f}%{m(v, 'R:dof_acc_l2'):>10.3f}"
          f"{m(v, 'R:action_rate_l2'):>10.3f}{m(v, 'R:track_default_pose_on_stand'):>9.3f}"
          f"{m(v, 'R:dof_pos_limits'):>9.3f}")

print("\n" + "=" * 132)
print("② σ → 探针 / 步态")
print("=" * 132)
print(f"{'σ_eff':>7}" + "".join(f"{p.replace('air_time_', '').replace('stand_pose_', ''):>18}" for p in PROBES)
      + f"{'jerk_exec':>11}{'jerk_mean':>11}")
for name, v in modes:
    row = "".join(f"{m(v, p):>18.4f}" for p in PROBES)
    print(f"{v['sigma_eff']:>7.3f}{row}{m(v, 'jerk_exec'):>11.4f}{m(v, 'jerk_mean'):>11.4f}")

print("\n" + "=" * 132)
print("③ 罚项的 σ 幂律（log|罚| vs log σ 的斜率：1≈线性于σ，2≈σ²）")
print("=" * 132)
sig = [v["sigma_eff"] for _, v in modes if v["sigma_eff"] > 1e-6]
for k in ["R:action_rate_l2", "R:dof_acc_l2", "R:dof_pos_limits",
          "R:track_default_pose_on_stand", "R:feet_slide",
          "hf_acc_all", "stand_pose_vel_err"]:
    xs, ys = [], []
    for _, v in modes:
        if v["sigma_eff"] <= 1e-6:
            continue
        val = abs(m(v, k))
        if val > 1e-9:
            xs.append(math.log(v["sigma_eff"]))
            ys.append(math.log(val))
    if len(xs) < 3:
        continue
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")
    print(f"{k:<34} 斜率 = {slope:+.2f}   (n={len(xs)})")
print(f"\n（本轮 σ 扫描的点：{['%.3f' % v['sigma_eff'] for _, v in modes]}）")
