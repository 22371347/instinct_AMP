#!/usr/bin/env python3
"""镜像算子的**时序堆叠（`history_length > 1`）**回归测试（本机可跑，不需要 isaaclab）。

    source /home/leo/miniconda3/etc/profile.d/conda.sh && conda activate instinct
    python3 instinct/tools/test_mirror_frames.py

覆盖
----
1. 旧 626 布局（`from_joint_names`，`frames` 为空 ⇒ 全部单帧）—— **零回归**
2. 当前 378 布局（all `(dim,)` 单帧）—— **零回归**
3. ⭐ T1o 的 663 布局（本体感受 `(4, dim)` 堆叠 + `height_scan` 单帧）
4. ⭐⭐ T2 的 1500 布局（**`height_scan` 也堆叠 4 帧**：4×279 + 4×96）
4. 逐帧等价性：堆叠项第 f 帧的镜像 == 该帧的单帧镜像
7. fail-fast：堆叠项被谎报成扁平 `(116,)`（T1o 首发崩溃的形态）/ 向量项每帧维度不对

⚠️ 本文件存在的原因
-------------------
`mirror_obs` 里时序项的实现**第一版是错的**：写了 `perm.repeat(n_frames)`，
而 `torch.Tensor.repeat` 平铺的是**值**不是**索引偏移** ⇒ 第 f 帧的关节被映回第 0 帧，
对合性直接崩（实测误差 4.9）。第 3/4 节的对合检验就是抓这个的。
"""

from __future__ import annotations

import importlib.util
import sys

import torch

MIRROR_PY = "instinct_xu/source/instinctlab/instinctlab/utils/mirror.py"


def _load_mirror():
    """按文件路径加载 `mirror.py`（不走 `instinctlab/__init__.py`，避免拖进 isaaclab）。"""
    spec = importlib.util.spec_from_file_location("mirror_mod", MIRROR_PY)
    mod = importlib.util.module_from_spec(spec)
    #: ⚠️ 必须先登记进 `sys.modules`，否则 `@dataclass` 展开时找不到 `cls.__module__`
    #: 会抛 `AttributeError: 'NoneType' object has no attribute '__dict__'`。
    sys.modules["mirror_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_29_joints() -> list[str]:
    """造 29 个关节：14 对左右 + 1 个腰部（无对应项 ⇒ 自映射）。"""
    return [n for i in range(14) for n in (f"left_j{i:02d}", f"right_j{i:02d}")] + ["waist_yaw"]


def main() -> int:
    m = _load_mirror()
    joints = _make_29_joints()
    assert len(joints) == 29, len(joints)
    failures: list[str] = []

    def check_involution(spec, tag: str, expect_dim: int | None = None):
        if expect_dim is not None and spec.obs_dim != expect_dim:
            failures.append(f"[{tag}] obs_dim={spec.obs_dim}，期望 {expect_dim}")
            print(f"  [{tag}] FAIL obs_dim={spec.obs_dim}")
            return spec
        obs = torch.randn(3, spec.obs_dim)
        err = (spec.mirror_obs(spec.mirror_obs(obs)) - obs).abs().max().item()
        ok = err < 1e-6
        if not ok:
            failures.append(f"[{tag}] 对合误差 {err:.2e}")
        print(
            f"  [{tag}] obs_dim={spec.obs_dim} frames={sorted(set(spec.frames)) or [1]} "
            f"对合误差={err:.2e} {'OK' if ok else 'FAIL'}"
        )
        return spec

    print("[1] 旧 626 布局（LEGACY，frames 为空）")
    _ = check_involution(m.MirrorSpec.from_joint_names(joints), "626", 626)

    print("[2] 当前 378 布局（单帧）")
    check_involution(
        m.MirrorSpec.from_obs_format(
            joints,
            {
                "height_scan": (279,),
                "velocity_command": (3,),
                "base_lin_vel": (3,),
                "projected_gravity": (3,),
                "base_ang_vel": (3,),
                "joint_pos_rel": (29,),
                "joint_vel": (29,),
                "last_action": (29,),
            },
        ),
        "378",
        378,
    )

    print("[3] T1o 的 663 布局（本体感受 4 帧 + height_scan 单帧）")
    fmt_663 = {
        "height_scan": (279,),
        "velocity_command": (4, 3),
        "projected_gravity": (4, 3),
        "base_ang_vel": (4, 3),
        "joint_pos_rel": (4, 29),
        "joint_vel": (4, 29),
        "last_action": (4, 29),
    }
    spec663 = check_involution(m.MirrorSpec.from_obs_format(joints, fmt_663), "663", 663)
    if spec663.frames and set(spec663.frames) != {1, 4}:
        failures.append(f"[663] frames 意外：{set(spec663.frames)}")

    print("[4] 逐帧等价性（第 2 帧的镜像 == 该帧的单帧镜像）")
    obs = torch.randn(2, 663)
    sl = dict(spec663.segments)["joint_pos_rel"]
    frames = obs[..., sl].reshape(2, 4, 29)
    single = m.MirrorSpec.from_joint_names(joints).mirror_joints(frames[:, 2, :])
    got = spec663.mirror_obs(obs)[..., sl].reshape(2, 4, 29)[:, 2, :]
    err = (single - got).abs().max().item()
    if err >= 1e-6:
        failures.append(f"[逐帧等价] 误差 {err:.2e}")
    print(f"  误差={err:.2e} {'OK' if err < 1e-6 else 'FAIL'}")

    print("[5] ⭐ 真实路径：obs_format 只给**展平标量**，帧数由 frames_by_term 显式声明")
    #: ⚠️ 本仓库 isaaclab 的 `group_obs_term_dim` 实测就是报展平标量
    #:（`velocity_command`（4 帧×3）→ `(12,)`）⇒ 这是训练里**真正走到**的分支。
    flat_fmt = {
        "height_scan": (279,),
        "velocity_command": (12,),
        "projected_gravity": (12,),
        "base_ang_vel": (12,),
        "joint_pos_rel": (116,),
        "joint_vel": (116,),
        "last_action": (116,),
    }
    frames_by_term = {
        k: 4
        for k in (
            "velocity_command",
            "projected_gravity",
            "base_ang_vel",
            "joint_pos_rel",
            "joint_vel",
            "last_action",
        )
    }
    spec_flat = check_involution(
        m.MirrorSpec.from_obs_format(joints, flat_fmt, frames_by_term=frames_by_term),
        "663-扁平+显式帧数",
        663,
    )
    if set(spec_flat.frames) != {1, 4}:
        failures.append(f"[663-扁平] frames 意外：{set(spec_flat.frames)}")
    #: 与 `(frames, dim)` 写法必须给出**完全相同**的镜像结果（否则两种路径行为不一致）
    _o = torch.randn(2, 663)
    _d = (spec_flat.mirror_obs(_o) - spec663.mirror_obs(_o)).abs().max().item()
    if _d >= 1e-6:
        failures.append(f"[展平 vs 两维] 镜像结果不一致：{_d:.2e}")
    print(f"  与 (frames,dim) 写法的一致性误差={_d:.2e} {'OK' if _d < 1e-6 else 'FAIL'}")

    print("[6] ⭐ T2 的 1500 布局：**height_scan 也堆叠 4 帧**（4×279 + 4×96）")
    fmt_1500 = {
        "height_scan": (1116,),
        "velocity_command": (12,),
        "projected_gravity": (12,),
        "base_ang_vel": (12,),
        "joint_pos_rel": (116,),
        "joint_vel": (116,),
        "last_action": (116,),
    }
    frames_1500 = {
        k: 4
        for k in (
            "height_scan",
            "velocity_command",
            "projected_gravity",
            "base_ang_vel",
            "joint_pos_rel",
            "joint_vel",
            "last_action",
        )
    }
    spec1500 = check_involution(
        m.MirrorSpec.from_obs_format(joints, fmt_1500, frames_by_term=frames_1500),
        "1500",
        1500,
    )
    if set(spec1500.frames) != {4}:
        failures.append(f"[1500] frames 意外：{set(spec1500.frames)}")
    #: 扫描**逐帧**翻转的正确性：第 2 帧的镜像 == 该帧单独镜像
    _o15 = torch.randn(2, 1500)
    _sl15 = dict(spec1500.segments)["height_scan"]
    _fr15 = _o15[..., _sl15].reshape(2, 4, 279)
    _single15 = spec1500.mirror_scan(_fr15[:, 2, :])
    _got15 = spec1500.mirror_obs(_o15)[..., _sl15].reshape(2, 4, 279)[:, 2, :]
    _d15 = (_single15 - _got15).abs().max().item()
    if _d15 >= 1e-6:
        failures.append(f"[1500] 扫描逐帧翻转误差 {_d15:.2e}")
    print(f"  扫描逐帧翻转（第 2 帧 vs 单帧）误差={_d15:.2e} {'OK' if _d15 < 1e-6 else 'FAIL'}")

    print("[7] fail-fast")
    for name, bad_fmt, why in (
        #: 堆叠项**没声明帧数**（这就是 T1o 首发崩溃的形态）：per_frame=116 ≠ 29 ⇒ 兵底报错。
        ("堆叠项谎报成扁平且没声明帧数", {"height_scan": (279,), "joint_pos_rel": (116,)}, "(116,) 无 frames"),
        #: 每帧维度与已知不符（16 ≠ 3）⇒ 也必须拦住。
        ("向量项每帧维度不对", {"height_scan": (279,), "velocity_command": (16,)}, "(16,) ≠ 3"),
    ):
        try:
            m.MirrorSpec.from_obs_format(joints, bad_fmt)
        except KeyError as exc:
            print(f"  [{name}] OK 已拦住（{why}）: {str(exc)[:60]}…")
        else:
            failures.append(f"[fail-fast] {name} 没报错")
            print(f"  [{name}] FAIL —— 没报错！")

    print()
    if failures:
        print(f"❌ 失败 {len(failures)} 项：" + "；".join(failures))
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
