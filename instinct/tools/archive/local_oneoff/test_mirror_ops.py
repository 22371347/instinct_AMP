"""`instinctlab/utils/mirror.py` 的单测（只需 torch，CPU 即可；不用起 Isaac Sim）。

跑法（**云端**优先，本地有 torch 也能跑）：
    /opt/conda/envs/isaaclab/bin/python /root/test_mirror_ops.py
    python instinct/tools/test_mirror_ops.py          # 本地（自动定位仓库里的 mirror.py）
    LLP_MIRROR_PY=/path/to/mirror.py python ...       # 手动指定被测文件

⭐ 覆盖**两种布局**：626（v1~v8，扫描居中）与 378（v9，279 条射线、中心前移 1.5 m）。

⚠️ 这里的 `JOINTS` 用 **npz 序**（左腿→右腿→腰→左臂→右臂）—— 因为它是
AMP_mjlab 核对报告里逐位确认过的顺序，方便对照。
本模块的算子**与关节顺序无关**（置换由关节名现算），所以用它测出的数学性质
对运行时的 articulation（肩优先交错）序**同样成立**。
投产时会在云端用 `action_term._joint_names` 再跑一遍同样的断言。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

# ⚠️ **不要**写 `from instinctlab.utils.mirror import ...`：
#    导入 `instinctlab` 包会连带执行 `instinctlab/__init__.py`，把 Omniverse/Kit
#    拉进来 ⇒ 非交互环境里会**卡在 EULA 交互提示**
#    （实测现象：输出全空、进程一直挂着、`Do you accept the EULA? (Yes/No):`）。
#    这里按**文件路径**直接加载，保持这是纯 CPU 单测。
#: 被测文件：优先环境变量，其次**本地仓库路径**，最后云端固定路径（同一个文件）。
_MIRROR_CANDIDATES = [
    os.environ.get("LLP_MIRROR_PY", ""),
    str(Path(__file__).resolve().parent.parent.parent
        / "instinct_xu/source/instinctlab/instinctlab/utils/mirror.py"),
    "/root/gpufree-data/instinct_xu/source/instinctlab/instinctlab/utils/mirror.py",
]


def _readable(p: str) -> bool:
    """⚠️ `Path.is_file()` 在没权限的路径上会抛 PermissionError（不是返回 False）。"""
    try:
        return bool(p) and Path(p).is_file() and os.access(p, os.R_OK)
    except OSError:
        return False


_MIRROR_PY = next((p for p in _MIRROR_CANDIDATES if _readable(p)), "")
assert _MIRROR_PY, f"找不到 mirror.py，已尝试：{_MIRROR_CANDIDATES}"
print(f"被测文件：{_MIRROR_PY}")
_spec = importlib.util.spec_from_file_location("llp_mirror_under_test", _MIRROR_PY)
assert _spec is not None and _spec.loader is not None, f"无法加载 {_MIRROR_PY}"
mirror_mod = importlib.util.module_from_spec(_spec)
# ⚠️ 必须先注册进 `sys.modules`：`@dataclass` 内部会做
#    `sys.modules.get(cls.__module__).__dict__`，不注册就报
#    `AttributeError: 'NoneType' object has no attribute '__dict__'`（实测踩到）。
sys.modules["llp_mirror_under_test"] = mirror_mod
_spec.loader.exec_module(mirror_mod)

OBS_DIM = mirror_mod.OBS_DIM
SCAN_DIM = mirror_mod.SCAN_DIM
SCAN_NX = mirror_mod.SCAN_NX
SCAN_NY = mirror_mod.SCAN_NY
MirrorSpec = mirror_mod.MirrorSpec

JOINTS = [
    # left leg
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # waist
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # left arm
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

#: 机器人默认姿态（`G1_29DOF_TORSOBASE_POPSICLE_CFG.init_state.joint_pos`）。
#: ⭐ 它是一个**天然不动点**：`M(q_default)` 必须等于 `q_default`。
DEFAULT = {
    "hip_pitch": -0.312, "knee": 0.669, "ankle_pitch": -0.363,
    "elbow": 0.600, "shoulder_pitch": 0.200,
    "left_shoulder_roll": 0.200, "right_shoulder_roll": -0.200,
}


def default_of(name: str) -> float:
    for sub, v in DEFAULT.items():
        if sub in name:
            return v
    return 0.0


PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}   {detail}")


print("=" * 74)
print("mirror.py 单测")
print("=" * 74)

spec = MirrorSpec.from_joint_names(JOINTS)
q_def = torch.tensor([default_of(n) for n in JOINTS], dtype=torch.float32)

print("\n[1] 置换与符号的基本性质")
check("perm 是对合 (perm[perm[i]] == i)", bool((spec.perm[spec.perm] == torch.arange(len(JOINTS))).all()))
check("sign 与 perm 相容 (sign[i]*sign[perm[i]] == +1)",
      bool((spec.sign * spec.sign[spec.perm] == 1).all().item()))
check("pitch 保号", bool((spec.sign[[i for i, n in enumerate(JOINTS) if "pitch" in n]] == 1).all().item()))
check("roll/yaw 反号",
      bool((spec.sign[[i for i, n in enumerate(JOINTS) if ("roll" in n or "yaw" in n)]] == -1).all().item()))
n_self = int((spec.perm == torch.arange(len(JOINTS))).sum())
check("腰部 3 个关节自映射", n_self == 3, f"self-map 数 = {n_self}")

print("\n[2] ⭐ 默认姿态是不动点（最能证明符号规则对）")
m_def = spec.mirror_joints(q_def)
check("M(q_default) == q_default", bool(torch.allclose(m_def, q_def, atol=1e-6)),
      f"max|Δ| = {(m_def - q_def).abs().max():.6f}")

print("\n[3] 关节镜像：对合性 + 左右交换正确")
q = torch.randn(7, len(JOINTS))
check("M(M(q)) == q", bool(torch.allclose(spec.mirror_joints(spec.mirror_joints(q)), q, atol=1e-6)))
# 手算：left_shoulder_roll=+0.9（右=0）⇒ 镜像后应是「右肩外展」，
# 而右肩的符号约定与左肩相反 ⇒ right_shoulder_roll 应为 **−0.9**（roll 反号）
q2 = torch.zeros(len(JOINTS)); q2[JOINTS.index("left_shoulder_roll_joint")] = 0.9
mq2 = spec.mirror_joints(q2)
check("left_shoulder_roll=+0.9（右=0）→ right_shoulder_roll=−0.9（roll 反号）",
      abs(float(mq2[JOINTS.index("right_shoulder_roll_joint")]) + 0.9) < 1e-6
      and abs(float(mq2[JOINTS.index("left_shoulder_roll_joint")])) < 1e-6,
      f"得到 L={float(mq2[JOINTS.index('left_shoulder_roll_joint')]):+.4f}, "
      f"R={float(mq2[JOINTS.index('right_shoulder_roll_joint')]):+.4f}")
# ⚠️ 镜像会**置换索引**，所以断言要看"值落到哪个关节"，不是同索引不变
q3 = torch.zeros(len(JOINTS)); q3[JOINTS.index("left_hip_pitch_joint")] = 0.5
mq3 = spec.mirror_joints(q3)
check("left_hip_pitch=+0.5 → right_hip_pitch=+0.5（pitch 保号）",
      abs(float(mq3[JOINTS.index("right_hip_pitch_joint")]) - 0.5) < 1e-6,
      f"得到 {float(mq3[JOINTS.index('right_hip_pitch_joint')]):+.4f}")
# ⭐ 回归测试：这是单测第一次就抓到的 bug（knee/elbow 是 pitch 类但名字没 "pitch"）
for jn in ("knee_joint", "elbow_joint"):
    q4 = torch.zeros(len(JOINTS)); q4[JOINTS.index("left_" + jn)] = 0.7
    mq4 = spec.mirror_joints(q4)
    check(f"left_{jn}=+0.7 → right_{jn}=+0.7（{jn.removesuffix('_joint')} 是 pitch 类，保号）",
          abs(float(mq4[JOINTS.index("right_" + jn)]) - 0.7) < 1e-6,
          f"得到 {float(mq4[JOINTS.index('right_' + jn)]):+.4f}")
check("waist_yaw 镜像后反号（自映射 + yaw 反号）",
      abs(float(spec.mirror_joints(
          torch.tensor([0.0] * JOINTS.index("waist_yaw_joint") + [0.7] + [0.0] * (28 - JOINTS.index("waist_yaw_joint"))))
          [JOINTS.index("waist_yaw_joint")]) + 0.7) < 1e-6)

print("\n[4] height_scan 镜像（侧向翻转，前向不动）")
check(f"527 = {SCAN_NY}×{SCAN_NX}", SCAN_NY * SCAN_NX == SCAN_DIM == 527)
# 令 scan 的值 = 行号 iy，则镜像后应变成 (NY-1-iy)，且**列号 ix 不变**
scan = torch.arange(SCAN_DIM, dtype=torch.float32).reshape(SCAN_NY, SCAN_NX)
scan_flat = scan.reshape(-1).clone()
m_scan = spec.mirror_scan(scan_flat).reshape(SCAN_NY, SCAN_NX)
check("行（侧向 y）镜像：out[iy,ix] == in[NY-1-iy, ix]",
      bool(torch.equal(m_scan, scan.flip(dims=(0,)))))
check("列（前向 x）不动:", bool(torch.equal(m_scan[:, 0], scan.flip(dims=(0,))[:, 0])))
check("scan 镜像是对合", bool(torch.allclose(spec.mirror_scan(spec.mirror_scan(scan_flat)), scan_flat)))

print("\n[5] 整段观测（626 维）对合性")
obs = torch.randn(5, OBS_DIM)
check("M_obs(M_obs(o)) == o", bool(torch.allclose(spec.mirror_obs(spec.mirror_obs(obs)), obs, atol=1e-6)),
      f"max|Δ| = {(spec.mirror_obs(spec.mirror_obs(obs)) - obs).abs().max():.2e}")
o = torch.zeros(OBS_DIM); o[527] = 1.0; o[528] = 2.0; o[529] = 3.0
mo = spec.mirror_obs(o)
check("cmd (vx,vy,wz) → (vx,−vy,−wz)", (float(mo[527]), float(mo[528]), float(mo[529])) == (1.0, -2.0, -3.0),
      f"得到 {(float(mo[527]), float(mo[528]), float(mo[529]))}")
o = torch.zeros(OBS_DIM); o[536] = 1.0; o[537] = 2.0; o[538] = 3.0
mo = spec.mirror_obs(o)
check("base_ang_vel → (−wx, wy, −wz)", (float(mo[536]), float(mo[537]), float(mo[538])) == (-1.0, 2.0, -3.0))

print("\n[6] 动作镜像与关节镜像一致")
a = torch.randn(4, len(JOINTS))
check("mirror_action == mirror_joints", bool(torch.allclose(spec.mirror_action(a), spec.mirror_joints(a))))

print("\n[7] ⭐ 设备安全（在 CUDA 上复跑核心断言）—— 回归：v4 首次启动就因 CPU/CUDA 混用崩过")
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    qd = q.to(dev)
    check("CUDA 上 M(M(q)) == q",
          bool(torch.allclose(spec.mirror_joints(spec.mirror_joints(qd)).cpu(), q, atol=1e-6)))
    check("CUDA 上 M(q_default) == q_default",
          bool(torch.allclose(spec.mirror_joints(q_def.to(dev)).cpu(), q_def, atol=1e-6)))
    obsd = obs.to(dev)
    check("CUDA 上 M_obs(M_obs(o)) == o",
          bool(torch.allclose(spec.mirror_obs(spec.mirror_obs(obsd)).cpu(), obs, atol=1e-6)))
    check("CUDA 上 mirror_action / mirror_scan 可用且 shape 不变",
          spec.mirror_action(a.to(dev)).shape == a.shape
          and spec.mirror_scan(obsd[:, :SCAN_DIM]).shape == obs[:, :SCAN_DIM].shape)
    check("CUDA 上输出 device == 输入 device",
          spec.mirror_obs(obsd).device == dev and spec.mirror_joints(qd).device == dev)
else:
    print("  (无 CUDA，跳过；⚠️ 训练在 CUDA 上跑，请勿只跑 CPU 就认为安全)")

# ---------------------------------------------------------------------------
# [8] ⭐ 布局参数化：626（v1~v8）与 378（v9，扫描中心前移 1.5 m）都要支持
#     回归背景：v9 首跑在第一次 PPO.update() 崩在
#     `ValueError: policy obs 最后一维应为 626，收到 378`（mirror.py 写死了布局）。
# ---------------------------------------------------------------------------
POLICY_FORMAT_626 = {
    "height_scan": (527,), "velocity_command": (3,), "base_lin_vel": (3,),
    "projected_gravity": (3,), "base_ang_vel": (3,),
    "joint_pos_rel": (29,), "joint_vel": (29,), "last_action": (29,),
}
POLICY_FORMAT_378 = dict(POLICY_FORMAT_626, height_scan=(279,))

print("\n[8] ⭐ from_obs_format：626（居中）与 378（前移 1.5 m）都要支持")
for tag, fmt, dim, ny, nx, vs in (("626", POLICY_FORMAT_626, 626, 17, 31, 527),
                                  ("378", POLICY_FORMAT_378, 378, 9, 31, 279)):
    s = MirrorSpec.from_obs_format(JOINTS, fmt)
    check(f"{tag}: obs_dim = {dim}", s.obs_dim == dim, f"got {s.obs_dim}")
    check(f"{tag}: 扫描布局 = {ny}x{nx}（{vs} 条）", (s.scan_ny, s.scan_nx) == (ny, nx),
          f"got {s.scan_ny}x{s.scan_nx}")
    check(f"{tag}: 分段共 {len(fmt)} 项、首项 = height_scan",
          len(s.segments) == len(fmt) and s.segments[0][0] == "height_scan")
    check(f"{tag}: cmd 段起始偏移 = {vs}", s.segments[1][1].start == vs,
          f"got {s.segments[1][1].start}")

    # --- 扫描镜像：侧向翻转、前向不动（前移量不改变镜像规则）---
    sc = torch.arange(vs, dtype=torch.float32).reshape(ny, nx)
    m = s.mirror_scan(sc.reshape(-1)).reshape(ny, nx)
    check(f"{tag}: scan 侧向翻转 out[iy,ix] == in[{ny}-1-iy,ix]",
          bool(torch.equal(m, sc.flip(dims=(0,)))))
    check(f"{tag}: scan 前向（列）不动", bool(torch.equal(m[:, 0], sc.flip(dims=(0,))[:, 0])))
    check(f"{tag}: scan 镜像对合",
          bool(torch.allclose(s.mirror_scan(s.mirror_scan(sc.reshape(-1))), sc.reshape(-1))))

    # --- 整段观测：对合性 + 关键段符号 ---
    obs_ = torch.randn(4, dim)
    check(f"{tag}: M_obs(M_obs(o)) == o",
          bool(torch.allclose(s.mirror_obs(s.mirror_obs(obs_)), obs_, atol=1e-6)),
          f"max|Δ| = {(s.mirror_obs(s.mirror_obs(obs_)) - obs_).abs().max():.1e}")
    o_ = torch.zeros(dim)
    o_[vs + 0], o_[vs + 1], o_[vs + 2] = 1.0, 2.0, 3.0
    mo_ = s.mirror_obs(o_)
    check(f"{tag}: cmd (vx,vy,wz) → (vx,−vy,−wz)",
          (float(mo_[vs]), float(mo_[vs + 1]), float(mo_[vs + 2])) == (1.0, -2.0, -3.0),
          f"得到 {(float(mo_[vs]), float(mo_[vs + 1]), float(mo_[vs + 2]))}")
    o_d = torch.zeros(dim)
    o_d[vs + 12:vs + 41] = q_def                      # joint_pos_rel
    o_d[vs + 70:vs + 99] = q_def                      # last_action
    mo_d = s.mirror_obs(o_d)
    check(f"{tag}: joint_pos_rel / last_action 里的默认姿态是不动点",
          bool(torch.allclose(mo_d[vs + 12:vs + 41], q_def, atol=1e-6)
               and torch.allclose(mo_d[vs + 70:vs + 99], q_def, atol=1e-6)))

    # --- 错维度必须 fail-fast（而不是静默错镜像）---
    try:
        s.mirror_obs(torch.randn(4, dim + 1))
        check(f"{tag}: 整段维度不符时 fail-fast", False, "没有抛异常")
    except ValueError as e:
        check(f"{tag}: 整段维度不符时 fail-fast", "最后一维" in str(e), str(e)[:60])

print("\n[9] from_obs_format 的 fail-fast（宁可启动报错，也不要静默错镜像）")
try:
    MirrorSpec.from_obs_format(JOINTS, {"height_scan": (527,), "unknown_term": (3,)})
    check("未登记项名 → KeyError", False, "没有抛异常")
except KeyError as e:
    check("未登记项名 → KeyError", "不认识" in str(e), str(e)[:60])
try:
    MirrorSpec.from_obs_format(JOINTS, {"height_scan": (300,)})
    check("未登记的扫描条数 → KeyError", False, "没有抛异常")
except KeyError as e:
    check("未登记的扫描条数 → KeyError", "SCAN_LAYOUTS" in str(e), str(e)[:60])
try:
    MirrorSpec.from_obs_format(JOINTS, {"joint_pos_rel": (29,)})
    check("缺 height_scan → KeyError", False, "没有抛异常")
except KeyError as e:
    check("缺 height_scan → KeyError", "找不到" in str(e), str(e)[:60])
# 兼容入口：from_joint_names 必须仍给出旧 626 布局（v1~v8 的老代码不受影响）
s_legacy = MirrorSpec.from_joint_names(JOINTS)
check("from_joint_names 兼容入口 = 626 / 17x31",
      (s_legacy.obs_dim, s_legacy.scan_ny, s_legacy.scan_nx) == (626, 17, 31),
      f"got {s_legacy.obs_dim}/{s_legacy.scan_ny}x{s_legacy.scan_nx}")

print("\n" + "=" * 74)
print(f"结果：PASS {PASS} / FAIL {FAIL}")
sys.exit(1 if FAIL else 0)
