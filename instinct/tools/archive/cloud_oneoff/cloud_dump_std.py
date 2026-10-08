"""打印 checkpoint 里 actor 的**逐关节 action noise std（σ）**。

## 为什么需要它

训练日志里的 `Mean action noise std` 是 **29 个关节的均值**，而且只打印 **2 位小数**：

    Mean action noise std: 0.39      ← 均值 ∈ [0.385, 0.395)

于是「均值 0.39 + 上限 0.4」有两种完全不同的解释：

1. 29 个关节都 ≈0.39 ⇒ σ 自己收敛到这里，`clip_max_std` 是**虚设**；
2. 少数关节已经顶到 0.40、其余偏小 ⇒ 上限**正在**起作用，均值被"平均"下来了。

两者对"该不该动 σ 上限"的结论**完全相反**，所以必须看逐关节值。
（本来想在训练里打日志，但上游 `PPO.update` 是先聚合/打印 `losses` 再 `return`，
 事后往返回的 dict 里塞键既不显示也会污染 `total_loss` ⇒ 改成离线读 checkpoint。）

## 用法（云端）

    /opt/conda/envs/isaaclab/bin/python /root/cloud_dump_std.py                # 自动找最新 run
    /opt/conda/envs/isaaclab/bin/python /root/cloud_dump_std.py <run_dir>
    /opt/conda/envs/isaaclab/bin/python /root/cloud_dump_std.py <model_xxxx.pt>
"""

from __future__ import annotations

import glob
import os
import sys

import torch

LOG_ROOT = "/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk"

#: articulation 序（与训练时 `action_term._joint_names` / `mirror_joint_names` 一致）
JOINTS_ISAACLAB = (
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


def newest(pattern: str) -> str:
    hits = glob.glob(pattern)
    return max(hits, key=os.path.getmtime) if hits else ""


def resolve(arg: str) -> str:
    if arg and os.path.isfile(arg):
        return arg
    if arg and os.path.isdir(arg):
        return newest(os.path.join(arg, "model_*.pt"))
    run = arg or newest(os.path.join(LOG_ROOT, "*"))
    return newest(os.path.join(run, "model_*.pt"))


def main() -> int:
    path = resolve(sys.argv[1] if len(sys.argv) > 1 else "")
    if not path:
        print("没找到 checkpoint；试过：", sys.argv[1:] or LOG_ROOT)
        return 1
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    std = sd.get("std")
    if std is None:
        print(f"{path}\n  ⚠️ state_dict 里没有 'std'；键示例：{list(sd.keys())[:12]}")
        return 2

    s = std.detach().reshape(-1).float()
    print(f"checkpoint: {path}")
    print(f"  step(文件名推断) = {os.path.basename(path)}")
    print(f"  σ: n={s.numel()}  min={s.min():.4f}  max={s.max():.4f}  mean={s.mean():.4f}  std={s.std():.4f}")
    k = min(6, s.numel())
    top = torch.topk(s, k)
    print(f"  膨胀最大的 {k} 个关节：")
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        name = JOINTS_ISAACLAB[i] if i < len(JOINTS_ISAACLAB) else f"idx{i}"
        print(f"    [{i:2d}] {name:32s} σ = {v:.4f}")
    bot = torch.topk(s, k, largest=False)
    print(f"  最小的 {k} 个关节：")
    for v, i in zip(bot.values.tolist(), bot.indices.tolist()):
        name = JOINTS_ISAACLAB[i] if i < len(JOINTS_ISAACLAB) else f"idx{i}"
        print(f"    [{i:2d}] {name:32s} σ = {v:.4f}")
    for thr in (0.40, 0.39, 0.30, 0.25):
        print(f"  σ >= {thr:.2f} 的关节数: {int((s >= thr - 1e-9).sum())}/{s.numel()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
