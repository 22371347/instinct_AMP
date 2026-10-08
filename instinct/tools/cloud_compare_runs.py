#!/usr/bin/env python3
"""多份训练日志的**同口径对照表**（云端跑，纯文本、不占 GPU）。

背景：并行试了 v9 / v9b / v9c / v9d，只看 MuJoCo 回放很容易记错"哪份会动"
（已经发生过一次）⇒ 用日志里的硬指标对齐：

关键判据（都是**每步**或**比率**，可直接横比）：

- `vx_*_vxy_rmse`（m/s）：**速度跟踪误差**。指令只有 0~0.7 m/s，
  若误差 ≈ 0.9~1.4 ⇒ **根本没在跟指令**（就算"在动"也是原地/乱动），比 `Mean reward` 可靠得多。
- `track_lin_vel_xy_exp`（每步 / 上限 1.5）：速度跟踪奖励占满分的比例。
- `base_contact`：触地终止比例（摔倒率）。
- `amp_d` / `r_amp`：判别器输出与风格奖励 ⇒ 动作离参考动作有多远。
- `σ`：`Mean action noise std` 的**末值/最大值**（是否顶在上限）。
- `mirror`：开了镜像损失才有这行；末值越小越"左右等变"。

用法（云端）：
    /opt/conda/envs/isaaclab/bin/python /root/cloud_compare_runs.py [log ...]
不给参数时自动取 /root/train_amp_walk_v9*.log。
"""

from __future__ import annotations

import glob
import os
import re
import sys

DEFAULT = sorted(glob.glob("/root/train_amp_walk_v9*.log"))

#: 行内 `Key: value` 形式的标量（取**最后一次**出现的值）
#: ⚠️ 日志里的键**带前缀**：奖励项是 `Mean episode Episode_Reward/...`、
#:    步内量是 `Mean step ...`；只写短名会一条都匹不到（踩过）。
SCALAR_KEYS = [
    "Mean reward",
    "Mean episode length",
    "Mean action noise std",
    "Mean episode Episode_Reward/track_lin_vel_xy_exp",
    "Mean episode Episode_Reward/track_ang_vel_z_exp",
    "Mean episode Episode_Reward/track_base_height",
    "Mean episode Episode_Reward/body_ang_vel_xy",
    "Mean episode Episode_Reward/track_default_pose_on_stand",
    "Mean episode Episode_Reward/joint_deviation_torso",
    "Mean episode Episode_Reward/joint_deviation_arms",
    "Mean episode Episode_Reward/feet_slide",
    "Mean episode Episode_Reward/flat_orientation_l2",
    "Mean episode Episode_Reward/action_rate_l2",
    "Mean episode Episode_Reward/dof_acc_l2",
    "Mean episode Episode_Reward/dof_pos_limits",
    "Mean episode Episode_Reward/termination_penalty",
    "Mean episode Episode_Termination/base_contact",
    "Mean step Step_Monitor/tracking_command_bins/vx_mid_vxy_rmse",
    "Mean step Step_Monitor/tracking_command_bins/vx_high_vxy_rmse",
    "Mean step amp_d",
    "Mean step amp_style_reward",
    "Mean step discriminator_reward",
    # ⭐ 诊断（v9f 起才有）
    "Mean step effective_lr",
    "Mean step action_abs_mean",
    "Mean step action_abs_max",
    "Mean episode Episode_Reward/feet_air_time",
]

#: `mirror 0.5279`（**没有冒号**，与损失键一起打印）
BARE_KEYS = ["mirror"]

ITER_RE = re.compile(r"Learning iteration (\d+)/")
KV_RE = re.compile(r"^[ \t]*([A-Za-z][A-Za-z0-9_/ .\-]*): +([-+0-9.eE]+)")


def short_label(path: str) -> str:
    """从路径里猜一个短标签：优先 v<N><后缀>，否则用日志名尾部。"""
    m = re.findall(r"[Vv]([0-9]+[a-z]?)", path)
    if m:
        # 取**最后一个**（文件名里通常是 v9c 这种），并去重
        return "v" + m[-1].lower()
    base = os.path.basename(path).replace("train_amp_walk_", "").replace(".log", "")
    return base[:12]


def parse(path: str) -> dict:
    st: dict[str, object] = {k: None for k in SCALAR_KEYS}
    st["mirror"] = None
    st["σ max"] = None
    st["iters"] = 0
    smax = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            it = ITER_RE.search(line)
            if it:
                st["iters"] = int(it.group(1))
            m = KV_RE.match(line)
            if m:
                key, val = m.group(1).strip(), m.group(2)
                if key in st:
                    st[key] = float(val)
                if key == "Mean action noise std":
                    v = float(val)
                    smax = v if smax is None else max(smax, v)
            for bk in BARE_KEYS:
                if line.lstrip().startswith(bk + " "):
                    try:
                        st[bk] = float(line.split()[1])
                    except (IndexError, ValueError):
                        pass
    st["σ max"] = smax
    return st


def fmt(v: object) -> str:
    if v is None:
        return "–"
    if isinstance(v, int):
        return f"{v}"
    f = float(v)
    return f"{f:.4f}" if abs(f) < 1000 else f"{f:,.0f}"


def main() -> int:
    logs = sys.argv[1:] or DEFAULT
    logs = [p for p in logs if os.path.isfile(p)]
    if not logs:
        print("没有可解析的日志")
        return 1
    labels = [short_label(p) for p in logs]
    data = [parse(p) for p in logs]

    rows = [
        ("iters", "iters（真实轮数）"),
        ("Mean reward", "Mean reward（纯任务回报/回合）"),
        ("Mean episode length", "Mean episode length（步/1000）"),
        ("Mean episode Episode_Termination/base_contact", "base_contact 终止比例（摔倒率）"),
        ("Mean step Step_Monitor/tracking_command_bins/vx_mid_vxy_rmse", "vx_mid 速度跟踪 RMSE (m/s)"),
        ("Mean step Step_Monitor/tracking_command_bins/vx_high_vxy_rmse", "vx_high 速度跟踪 RMSE (m/s)"),
        ("Mean action noise std", "σ 末值"),
        ("σ max", "σ 峰值（贴上限=被卡）"),
        ("mirror", "镜像损失末值（越小越等变）"),
        ("Mean step amp_d", "amp_d（→+1 像参考）"),
        ("Mean step amp_style_reward", "r_amp（风格奖励）"),
        ("Mean step effective_lr", "真实学习率（adaptive 时≤1e-3）"),
        ("Mean step action_abs_mean", "动作幅度 |a| 均值"),
        ("Mean step action_abs_max", "动作幅度 |a| 最大值"),
        ("Mean episode Episode_Reward/feet_air_time", "  feet_air_time（/s；>0=在单脚支撑）"),
        ("Mean episode Episode_Reward/track_lin_vel_xy_exp", "  track_lin_vel_xy（/s，满分75）"),
        ("Mean episode Episode_Reward/track_ang_vel_z_exp", "  track_ang_vel_z（/s）"),
        ("Mean episode Episode_Reward/track_base_height", "  track_base_height（/s）"),
        ("Mean episode Episode_Reward/body_ang_vel_xy", "  body_ang_vel_xy（/s）"),
        ("Mean episode Episode_Reward/track_default_pose_on_stand", "  track_default_pose_on_stand（/s）"),
        ("Mean episode Episode_Reward/joint_deviation_torso", "  joint_deviation_torso（/s）"),
        ("Mean episode Episode_Reward/joint_deviation_arms", "  joint_deviation_arms（/s）"),
        ("Mean episode Episode_Reward/feet_slide", "  feet_slide（/s，罚滑步）"),
        ("Mean episode Episode_Reward/flat_orientation_l2", "  flat_orientation_l2（/s）"),
        ("Mean episode Episode_Reward/action_rate_l2", "  action_rate_l2（/s）"),
        ("Mean episode Episode_Reward/dof_acc_l2", "  dof_acc_l2（/s）"),
        ("Mean episode Episode_Reward/dof_pos_limits", "  dof_pos_limits（/s）"),
        ("Mean episode Episode_Reward/termination_penalty", "  termination_penalty（/s）"),
    ]

    w = 30
    print("=" * (w + 14 * len(logs)))
    print(f"{'指标':<{w}}" + "".join(f"{lab:>14}" for lab in labels))
    print("-" * (w + 14 * len(logs)))
    for key, label in rows:
        print(f"{label:<{w}}" + "".join(f"{fmt(d.get(key)):>14}" for d in data))
    print("-" * (w + 14 * len(logs)))
    print("（奖励口径：`Episode_Reward/x` = **每步加权时间平均**（累计时乘了 dt 再除以 20 s）⇒ "
          "**不是每秒速率**；每回合贡献 = ×20。iters 取日志里**最后的轮号**，"
          "grep -c 会少算（横幅每 10 轮打一次））")
    print("日志：")
    for lab, p in zip(labels, logs):
        print(f"  {lab:<6} {p}")
    print("=" * (w + 14 * len(logs)))
    print("读法：vx_*_rmse 与指令同量级 ⇒ 没在跟指令；base_contact 高 ⇒ 常摔；")
    print("      σ 末值≈上限 ⇒ σ 被卡住（entropy 太强），σ 回落 ⇒ entropy 合适；")
    print("      正奖励占大头的是“站得稳”而不是“走得动” ⇒ 奖励形状在鼓励站着不动。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
