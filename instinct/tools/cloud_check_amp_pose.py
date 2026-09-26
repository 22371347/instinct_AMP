"""检查 AMP 参考数据集的「初始帧关节姿态」是否满足正常站立需求。

在云端跑（数据集在云端），无需 GPU。
用法：
  /opt/conda/envs/isaaclab/bin/python cloud_check_amp_pose.py [npz_glob]

对照基准 = 机器人默认姿态（`assets/unitree_g1.py` 里
`G1_29DOF_TORSOBASE_POPSICLE_CFG.init_state.joint_pos`，只列了 8 条正则，
其余关节默认 0.0）。

输出：每个 clip 的
  1) keys / shapes（首次运行用来自查数据结构）
  2) **第 0 帧**逐关节姿态 vs 默认姿态的差
  3) 整个 clip 的逐关节均值 vs 默认姿态
  4) 每个关节在整个 clip 上的 min/max（看是否超限、是否有怪姿态）
"""

from __future__ import annotations

import glob
import sys

import numpy as np

GLOB = sys.argv[1] if len(sys.argv) > 1 else "/root/gpufree-data/instinct_xu/**/*_retargeted.npz"

#: 默认姿态（`G1_29DOF_TORSOBASE_POPSICLE_CFG.init_state.joint_pos`）。
#: 键是关节名子串；未匹配到的关节默认 0.0。
DEFAULT_SUBSTR: list[tuple[str, float]] = [
    ("hip_pitch", -0.312),
    ("knee", 0.669),
    ("ankle_pitch", -0.363),
    ("elbow", 0.600),
    ("shoulder_pitch", 0.200),
    ("left_shoulder_roll", 0.200),
    ("right_shoulder_roll", -0.200),
]


def default_of(name: str) -> float:
    for sub, val in DEFAULT_SUBSTR:
        if sub in name:
            return val
    return 0.0


def pick(d, *must, exclude=()):
    """按子串挑 key（大小写不敏感）。"""
    for k in d:
        kl = k.lower()
        if all(m in kl for m in must) and not any(e in kl for e in exclude):
            return k
    return None


files = sorted(glob.glob(GLOB, recursive=True))
print("=" * 78)
print("找到 npz: %d 个" % len(files))
for f in files:
    print("   ", f)
if not files:
    sys.exit(1)

# ── 数据结构自查（只看第一个）──────────────────────────────────────────────
d0 = np.load(files[0])
print("=" * 78)
print("第 1 个文件的 keys（首次运行用来自查结构）：")
for k in d0.keys():
    a = d0[k]
    print("   %-26s shape=%-18s dtype=%s" % (k, getattr(a, "shape", None), getattr(a, "dtype", None)))

jp_key = pick(d0, "joint", "pos") or pick(d0, "dof", "pos") or pick(d0, "qpos")
jn_key = pick(d0, "joint", "name") or pick(d0, "dof", "name")
fps_key = pick(d0, "fps")
print("-" * 78)
print("推断: joint_pos key = %r   joint_names key = %r   fps key = %r" % (jp_key, jn_key, fps_key))
if jp_key is None:
    print("!! 没找到关节位置字段 —— 请把上面的 keys 贴给我")
    sys.exit(2)

# 关节名
jn = d0[jn_key] if jn_key else None
if jn is not None:
    jn = [str(x) for x in np.asarray(jn).ravel()]
    print("关节名（%d 个）: %s" % (len(jn), jn))

print("=" * 78)
hdr = "%-30s%9s%9s%9s%9s%9s" % ("关节", "默认", "第0帧", "clip均值", "clip最小", "clip最大")
print("%-30s%9s%9s%9s%9s%9s" % ("关节", "默认", "第0帧", "clip均值", "clip最小", "clip最大"))

summ_f0: list[float] = []
for f in files:
    d = np.load(f)
    jp = np.asarray(d[jp_key], dtype=float)          # (T, 29)
    if jp.ndim != 2:
        jp = jp.reshape(jp.shape[0], -1)
    tag = f.split("/")[-1].replace("_retargeted.npz", "")
    print("-" * 78)
    print("【%s】 T=%d  帧率=%s" % (tag, jp.shape[0], d[fps_key] if fps_key else "?"))
    n = jp.shape[1]
    f0, mean, lo, hi = jp[0], jp.mean(axis=0), jp.min(axis=0), jp.max(axis=0)
    for i in range(n):
        nm = jn[i] if (jn and i < len(jn)) else "joint_%d" % i
        dv = default_of(nm)
        print("%-30s%9.3f%9.3f%9.3f%9.3f%9.3f" % (nm, dv, f0[i], mean[i], lo[i], hi[i]))
    dev = float(np.mean(np.abs(f0 - np.array([default_of(jn[i] if (jn and i < len(jn)) else "") for i in range(n)]))))
    summ_f0.append(dev)
    print("  ⇒ 第 0 帧与默认姿态的平均绝对差 = %.4f rad (%.1f deg)" % (dev, np.degrees(dev)))

print("=" * 78)
print("全部 clip 第 0 帧 vs 默认姿态的平均绝对差: %.4f rad (%.1f deg)"
      % (float(np.mean(summ_f0)), np.degrees(float(np.mean(summ_f0)))))
