#!/usr/bin/env python3
"""V33 **四臂对比表**：从云端 TensorBoard 读同一批 tag，按臂并列打印（只读，**不占 GPU**）。

为什么单写一个
--------------
V33 的四臂（S0 / S1 / A-cold / S2）结论只能靠**跨臂同 tag 对照**得出，而 `Train/*` 类指标
只写 TB、不打印到训练日志（见 repo memory「日志 vs TensorBoard」）⇒ 每次手工 grep 一个臂
既慢又容易看错行。本脚本把「同 tag × 各臂」拼成一张表，一眼可比。

用法：
    python instinct/tools/v33_arm_table.py                    # 默认 S0 S1 ACold S2 + V31 基线
    python instinct/tools/v33_arm_table.py S0 S1              # 只看两臂
    python instinct/tools/v33_arm_table.py S2 v31             # 加基线
    python instinct/tools/v33_arm_table.py --keys            # 只列出会读的 tag

实现要点：
  * 复用云端 `/root/cloud_tb_tags.py`（它认得 `<run子串>` 与 `--re`），本地只做**解析 + 拼表**；
  * 取每个 tag 的 **last**（最新一个点）而不是 tail_mean —— 四臂的迭代数可能不同，
    "最新值"比"最近 N 点均值"更好对齐（要均值就把 `--agg` 换成 mean）；
  * 远程输出是固定宽度列（tag / tail_mean / min / max / last / n / step），
    tag 名里没有空格 ⇒ 直接 `split()` 即可，**不要**按列位置切。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SSH = REPO / "instinct" / "tools" / "instinct_ssh.py"
REMOTE_PY = "/opt/conda/envs/isaaclab/bin/python"
REMOTE_TOOL = "/root/cloud_tb_tags.py"

#: 打印顺序即分组顺序（空字符串 = 分组标题）
METRICS: list[tuple[str, str]] = [
    ("", "── 风格 / 判别器 ──"),
    ("Step/amp_style_reward", "风格奖励"),
    ("Step/amp_style_std", "风格奖励 std（健康度核心）"),
    ("Step/amp_style_zero_frac", "风格奖励为 0 的比例"),
    ("Step/amp_d", "判别器对策略输出"),
    ("Step/amp_d_ref", "判别器对参考输出"),
    ("Loss/discriminator_loss", "判别器 MSE"),
    ("Loss/discriminator_gradient_penalty", "GP 项"),
    ("Train/disc_grad/gravity", "判别器敏感度：绝对姿态通道"),
    ("Train/disc_grad/pos", "判别器敏感度：位置"),
    ("Train/disc_grad/ori", "判别器敏感度：姿态"),
    ("Train/disc_grad/all", "判别器敏感度：总体"),
    ("", "── 姿态 ──"),
    ("Step/tilt_deg_mean", "躯干倾角均值（度）"),
    ("Step/tilt_gt30_frac", ">30° 比例"),
    ("Step/grav_bx_mean", "重力 x 分量（>0 前倾 / <0 后倾）"),
    ("Step/foot_sep_mean", "双脚间距"),
    ("", "── 步态 / 效率 ──"),
    # ⚠️ TB tag 不是控制台那句 "Mean reward:" —— runner 把它写成 `Train/mean_reward_0`
    #    （带下标 0 = 第 0 个奖励通道）；回合长度是 `Train/mean_episode_length`。
    ("Train/mean_reward_0", "平均回合奖励"),
    ("Train/mean_episode_length", "平均回合长度"),
    ("Step/action_abs_mean", "动作幅度均值（成本代理）"),
    ("Step/air_time_single_stance", "单支撑占比"),
    ("Step/air_time_hop_frac", "小跳占比"),
    ("Step/air_time_in_band_frac", "支撑时长落在目标带的比例"),
    ("", "── 速度跟踪（x/y 拆分 + 分桶，V33 新探针）──"),
    ("Step/lin_track_joint", "联合同（= 原跟踪项）"),
    ("Step/lin_track_fx", "x 轴因子"),
    ("Step/lin_track_fy", "y 轴因子"),
    ("Step/lin_track_side_joint", "侧移档：联合"),
    ("Step/lin_track_side_fy", "侧移档：y 因子"),
    ("Step/lin_track_side_ey_abs", "侧移档：|ey|"),
    ("Step/lin_track_lin_joint", "纯前后档：联合"),
    ("Step/lin_track_back_joint", "后退档：联合"),
    ("Step/lin_track_diag_joint", "对角/转向档：联合"),
    ("", "── 脚手架（S1 拆的是这两项）──"),
    # ⚠️ TB 里的真名就是 `Episode_Reward/<项>`（**没有** `Episode/` 前缀）——
    #    2026-10-06 实测抄出来的（runner 对 `Episode_Reward/*` 这类 key 不额外加前缀）。
    ("Episode_Reward/feet_air_time", "feet_air_time 得分"),
    ("Episode_Reward/flat_orientation_l2", "flat_orientation_l2 得分"),
    ("Episode_Reward/joint_deviation_arms", "joint_deviation_arms 得分"),
    ("Episode_Reward/track_lin_vel_xy_exp", "track_lin_vel_xy 得分"),
]


def ssh(cmd: str) -> str:
    """跑一条远端命令，剔除凭据/密码噪声行。"""
    out = subprocess.run(
        [sys.executable, str(SSH), cmd], capture_output=True, text=True, timeout=300
    ).stdout
    return "\n".join(
        ln for ln in out.splitlines()
        if ln.strip() and "credentials" not in ln and "password" not in ln
    )


def read_arm(arm: str, keys: list[str], agg: str) -> tuple[dict[str, float], str]:
    """读一个臂：返回 {tag: 值} 与被选中的 run 名。"""
    pat = "|".join(re.escape(k) for k in keys)
    raw = ssh(f"{REMOTE_PY} -u {REMOTE_TOOL} {arm} --re '{pat}' --tail 1")
    picked = ""
    vals: dict[str, float] = {}
    for ln in raw.splitlines():
        if ln.startswith("run:"):
            picked = ln.split("run:", 1)[1].strip()
            continue
        parts = ln.split()
        # 数据行：tag  tail_mean  min  max  last  n  step
        if len(parts) != 7 or not parts[0].startswith(("Step/", "Train/", "Loss/", "Episode/")):
            continue
        col = {"last": 4, "mean": 1, "max": 3, "min": 2}.get(agg, 4)
        try:
            vals[parts[0]] = float(parts[col])
        except ValueError:
            continue
    return vals, picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", default=["S0", "S1", "ACold", "S2", "v31"],
                    help="run 子串（默认 S0 S1 ACold S2 v31）")
    ap.add_argument("--agg", default="last", choices=["last", "mean", "max", "min"])
    ap.add_argument("--keys", action="store_true", help="只列出会读的 tag")
    args = ap.parse_args()

    keys = [k for k, _ in METRICS if k]
    if args.keys:
        print("\n".join(keys))
        return 0

    arms = list(args.arms) or ["S0", "S1", "ACold", "S2", "v31"]
    data: dict[str, dict[str, float]] = {}
    picked: dict[str, str] = {}
    for arm in arms:
        vals, run = read_arm(arm, keys, args.agg)
        data[arm] = vals
        picked[arm] = run or "（没匹配到 run / TB 里没有这些 tag）"
        print(f"[{arm:>6}] {len(vals):2d}/{len(keys)} 个 tag ← {picked[arm][:110]}", file=sys.stderr)

    print(f"\n聚合方式 = {args.agg}（last = 最新一个点）\n")
    head = f"{'指标':<34}" + "".join(f"{a:>13}" for a in arms)
    print(head)
    print("-" * len(head))
    for key, label in METRICS:
        if not key:
            print(f"\n{label}")
            continue
        row = f"{label:<34}"
        for a in arms:
            v = data[a].get(key)
            row += f"{'—':>13}" if v is None else f"{v:>13.4g}"
        print(row)
    print("\n提示：`Step/grav_bx_mean` 与 `tilt` 看姿态；`amp_style_std` < 0.03 = 判别器已死；"
          "`disc_grad/gravity` 高于 `pos` ⇒ 判别器真的在用新通道。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
