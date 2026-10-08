#!/usr/bin/env python3
"""**闭合验证**：专家集里的"根速度"与源数据真实的根速度是不是同一个量。

假设（断点定位的结论）
----------------------
* 源数据 `base_pos_w` **有真实位移**（已实测：直行 clip 6–8 m）；
* loader 的 `link_lin_vel_w = d(link_pos_w)/dt` **含根位移**（`amass_motion.py:744`）；
* loader 的 `link_lin_vel_b = d(link_pos_b)/dt`（`link_pos_b` 是 FK 的 **base 相对**位姿）
  ⇒ **锚点那条恒为 0**；
* 而 `cloud_export_amp_expert.py:219` 用的是 **`link_lin_vel_b`**，训练观测
  (`body_lin_vel_b_reference_as_state`) 用的是 **`link_lin_vel_w` + `_vel_in_body_frame`**。

⇒ 预测：**源数据的 |v_root| ≈ 0.5~1.0 m/s，而专家集里的 |v_root| ≈ 0**（数量级差 30×）。

|v| 是对偏航不变的量（旋转不改变模长），所以两个口径下可以正面对比；逐分量的 σ
才会因各 clip 世界朝向不同而不可比，故这里只比**模长**。

用法（云端，纯 numpy）：
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_verify_expert_vel_convention.py
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch

MOTIONS = (
    "/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab/tasks/locomotion"
    "/config/g1/motions/amp_walk"
)
BUF = "/root/amp_expert_states_396.pt"

#: 396 维布局（term-major / frame-inner，13 body）：锚点自身 = body0
OFF_LIN_VEL_T = 234  # body_lin_vel_b 的「帧 t」块起点


def main() -> int:
    # ---- ① 源数据的真实根速度（世界系，有限差分；取模长）------------------------
    files = sorted(glob.glob(os.path.join(MOTIONS, "*_retargeted.npz")))
    print(f"# 源数据：{len(files)} 个 clip")
    spd = []
    for f in files:
        z = np.load(f)
        bp = np.asarray(z["base_pos_w"], dtype=np.float64)
        fps = float(z["framerate"])
        d = np.diff(bp, axis=0) * fps          # 中心/前向差分的量级近似
        v = np.linalg.norm(d, axis=1)
        spd.append(v)
        name = os.path.basename(f).replace("_retargeted.npz", "")
        print(f"  {name:44s} 平均|v_root|={v.mean():6.3f} m/s   p90={np.quantile(v,0.9):6.3f}")
    allv = np.concatenate(spd)
    print(f"\n# 源数据 |v_root| 汇总：mean={allv.mean():.3f} m/s  "
          f"p50={np.quantile(allv,0.5):.3f}  p90={np.quantile(allv,0.9):.3f}  max={allv.max():.3f}")

    # ---- ② 专家集里的根速度（dims 234:237 = body0 帧 t 的线速度）----------------
    raw = torch.load(BUF, map_location="cpu")
    x = (next(v for v in raw.values() if isinstance(v, torch.Tensor)) if isinstance(raw, dict) else raw).float()
    anchor_v = x[:, OFF_LIN_VEL_T:OFF_LIN_VEL_T + 3]
    n = torch.linalg.norm(anchor_v, dim=1)
    print(f"\n# 专家集 {tuple(x.shape)} 里 anchor（body0）线速度：")
    print(f"  逐轴 σ = {anchor_v.std(dim=0).tolist()}")
    print(f"  |v|  mean={float(n.mean()):.5f} m/s   p50={float(n.median()):.5f}   max={float(n.max()):.5f}")

    # ---- ③ 对照 ---------------------------------------------------------------
    r = float(n.mean()) / max(float(allv.mean()), 1e-9)
    print(f"\n# 比值：专家集 |v_root| / 源数据 |v_root| = {r:.4f}  （即 {1/max(r,1e-9):.0f} 倍偏小）")
    if r < 0.1:
        print("✅ 假设成立：**专家集里的根速度不是源数据的根速度**（差一个数量级以上）。")
        print("   ⇒ 断点 = 导出脚本用了 `link_lin_vel_b`（base 相对差分，锚点恒 0），")
        print("     而训练观测用 `link_lin_vel_w` + `_vel_in_body_frame`。")
        print("   ⇒ 判别器的「真」样本里根速度被清零 ⇒ 风格项在**惩罚根位移**。")
    else:
        print("❌ 假设不成立：两者量级相近 ⇒ 断点在别处，需要重新追踪。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
