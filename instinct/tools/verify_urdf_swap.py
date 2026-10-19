#!/usr/bin/env python3
"""URDF 替换后的**训练接口一致性验证器**（本地可跑，不依赖 IsaacLab）。

背景 / 为什么需要它
-------------------
把机器人换成另一份 URDF（换模型、换版本、换根 link）之后，**训练侧接口有 4 处会跟着变**，
而且其中有些**无法从 URDF 推导**，只能实测。这个脚本把你换 URDF 后"到底接不接得上"
一次问清楚：

  ① 关节**名字集合**是否齐全（29 个 G1 关节一个不少）—— 少一个 = 观测/动作维度直接错
  ② 关节**顺序**：URDF 声明序 / MJCF 状态序 / MJCF ctrl 序 / 策略侧(IsaacLab)序
     —— 这四者**互不相同**，混用 = 回放抽搐、训练跑偏
  ③ 与**参考动作数据集**(.npz 的 `joint_names`)能否按名对齐 —— 决定数据管线要不要改
  ④ 镜像对称性（`left_* ↔ right_*`、roll/yaw 反号）—— mirror loss / 对称增强依赖它

用法
----
    # 1) 只看新 URDF 的关节信息与已知顺序的关系
    python instinct/tools/verify_urdf_swap.py \\
        --urdf instinct_xu/.../official_rev10/g1_29dof_rev_1_0.urdf

    # 2) 连本地回放 MJCF 一起查（ctrl 序 / 状态序 / keyframe 根高）
    python instinct/tools/verify_urdf_swap.py \\
        --urdf <...>/g1_29dof_rev_1_0.urdf \\
        --xml  instinct/assets/g1_29dof_rev10_play.xml

    # 3) 再带上参考动作数据集，检查能否按名对齐
    python instinct/tools/verify_urdf_swap.py --urdf <...>.urdf \\
        --npz instinct_xu/.../activations/... .npz

    # 4) 把结论落成 JSON 交给 CI / 存档
    python instinct/tools/verify_urdf_swap.py --urdf <...>.urdf --xml <...>.xml --json out.json

退出码
------
    0 = 接口齐备，可以直接用策略(.pt/.onnx)回放
    1 = 有**致命**问题（关节名缺失/顺序无法映射）⇒ 只能用 .npz 参考动作回放
    2 = 参数/文件问题

⚠️ 本脚本**不能**替你确认"IsaacLab 侧的 articulation 序"。那份顺序由 IsaacLab 在
   把 URDF 转成 USD 之后**运行时**给出，**既不是** URDF 声明序、**也不是** link 序/DFS 序，
   只能实测。实测办法见 `docs/urdf_swap/URDF_REPLACEMENT_GUIDE.md` §5。
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from joint_order import (  # noqa: E402
    ISAACLAB_ORDER,
    MUJOCO_ORDER,
    NPZ_ORDER,
    PELVIS_ORDER,
    mujoco_order_from_xml,
    mujoco_state_order_from_model,
)

#: 已知的"策略侧/数据集侧"关节顺序，键 = 报告里用的短名。
KNOWN_ORDERS: dict[str, tuple[str, ...]] = {
    "isaaclab": ISAACLAB_ORDER,   # 旧资产(torso 根)：肩优先、左右交错
    "p1": PELVIS_ORDER,           # 新资产(pelvis 根)：髋优先（云端实测）
    "npz": NPZ_ORDER,             # URDF/USD 声明序 & AMP 参考动作 npz 序（腿优先）
    "mjcf": MUJOCO_ORDER,         # MJCF ctrl（执行器）序：腰 → 腿 → 臂
}

#: 顺序名的中文释义（报告用）。
ORDER_TITLE = {
    "isaaclab": "IsaacLab 策略序（旧资产 torso 根；肩优先交错）",
    "p1": "IsaacLab 策略序（新资产 pelvis 根；髋优先，云端实测）",
    "npz": "URDF/USD 声明序 & AMP 参考动作 npz 序（腿 → 腰 → 臂）",
    "mjcf": "MJCF ctrl（执行器）序（腰 → 腿 → 臂）",
}


# ---------------------------------------------------------------------------
# URDF 解析（只用标准库，避免 mesh/package:// 之类把 mujoco 加载搞挂）
# ---------------------------------------------------------------------------
@dataclass
class UrdfJoint:
    name: str
    jtype: str
    parent: str
    child: str
    axis: tuple[float, float, float] | None
    origin_xyz: tuple[float, float, float] | None
    limit: tuple[float, float] | None
    effort: float | None

    @property
    def is_actuated(self) -> bool:
        return self.jtype in ("revolute", "continuous", "prismatic")


@dataclass
class UrdfInfo:
    path: Path
    robot_name: str
    joints: list[UrdfJoint] = field(default_factory=list)
    link_masses: dict[str, float] = field(default_factory=dict)
    root_link: str = ""
    mesh_refs: list[str] = field(default_factory=list)

    @property
    def actuated(self) -> list[UrdfJoint]:
        return [j for j in self.joints if j.is_actuated]

    @property
    def total_mass(self) -> float:
        return sum(self.link_masses.values())


def _f(txt: str | None) -> float | None:
    if txt is None:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _triple(txt: str | None) -> tuple[float, float, float] | None:
    if not txt:
        return None
    parts = txt.replace(",", " ").split()
    if len(parts) != 3:
        return None
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None


def parse_urdf(path: Path) -> UrdfInfo:
    """解析 URDF，返回关节/连杆信息（关节**按 <joint> 声明顺序**保留）。"""
    tree = ET.parse(path)
    root = tree.getroot()
    info = UrdfInfo(path=path, robot_name=root.get("name", "<unnamed>"))

    for link in root.findall("link"):
        nm = link.get("name", "")
        inertial = link.find("inertial")
        mass = 0.0
        if inertial is not None:
            m = inertial.find("mass")
            if m is not None:
                mass = _f(m.get("value")) or 0.0
        info.link_masses[nm] = mass

    children: set[str] = set()
    for m in root.iter("mesh"):
        fn = m.get("filename")
        if fn:
            info.mesh_refs.append(fn)
    for j in root.findall("joint"):
        parent = j.find("parent")
        child = j.find("child")
        axis = j.find("axis")
        origin = j.find("origin")
        limit = j.find("limit")
        info.joints.append(
            UrdfJoint(
                name=j.get("name", ""),
                jtype=(j.get("type") or "").lower(),
                parent=(parent.get("link", "") if parent is not None else ""),
                child=(child.get("link", "") if child is not None else ""),
                axis=_triple(axis.get("xyz") if axis is not None else None),
                origin_xyz=_triple(origin.get("xyz") if origin is not None else None),
                limit=(
                    (_f(limit.get("lower")), _f(limit.get("upper")))
                    if limit is not None
                    else None
                ),
                effort=_f(limit.get("effort")) if limit is not None else None,
            )
        )
        if child is not None:
            children.add(child.get("link", ""))

    # 根 link = 从没当过别人 child 的那个
    for nm in info.link_masses:
        if nm not in children:
            info.root_link = nm
            break
    return info


# ---------------------------------------------------------------------------
# 顺序比较工具
# ---------------------------------------------------------------------------
def same_order(a, b) -> bool:
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


def permutation_kind(seq, ref) -> str:
    """seq 与 ref 的关系：完全一致 / 同一集合但顺序不同 / 集合都不同。"""
    if same_order(seq, ref):
        return "same"
    if set(seq) == set(ref):
        return "permutation"
    return "different"


def reorder_index(src, dst) -> list[int]:
    """`vec_dst = vec_src[[reorder_index(src, dst)]]`。"""
    return [src.index(n) for n in dst]


def mirror_pairs(names) -> tuple[list[tuple[str, str]], list[str]]:
    """返回 (成对的 (left, right), 没有左右对应的关节)。"""
    pairs, solo = [], []
    name_set = set(names)
    for n in names:
        if n.startswith("left_"):
            r = "right_" + n[len("left_"):]
            if r in name_set:
                pairs.append((n, r))
    for n in names:
        if n.startswith("left_"):
            continue
        has_l = n.startswith("right_") and ("left_" + n[len("right_"):]) in name_set
        if not has_l:
            solo.append(n)
    return pairs, solo


def resolve_mesh(ref: str, urdf_dir: Path) -> tuple[Path | None, str]:
    """把 URDF 的 `<mesh filename>` 解析成实际路径。

    支持三种常见写法（返回 (路径或 None, 说明)）：
      · `meshes/x.STL`        —— 相对 URDF 所在目录（**自包含式**，本项目新资产用这种）
      · `../meshes/x.STL`     —— 相对 URDF 所在目录的上级（**共享式**，本项目旧资产用这种）
      · `package://pkg/...`   —— 需要 ROS/IsaacLab 侧的包解析器，本脚本只能给出提示
    """
    if ref.startswith("package://"):
        # 剥掉包名，剩余部分当作相对路径试试（多数情况下刚好能对上）
        rest = ref[len("package://"):]
        rest = rest.split("/", 1)[1] if "/" in rest else rest
        cand = (urdf_dir / rest).resolve()
        if cand.is_file():
            return cand, "package:// 已按剩余相对路径解析到（侥幸命中，建议改成显式相对路径）"
        return None, "`package://` 需 ROS/IsaacLab 的包解析器，本脚本无法验证 —— 请确认 IsaacLab 侧能解析"
    cand = (urdf_dir / ref).resolve()
    kind = "自包含式(相对 URDF 目录)" if not ref.startswith("..") else "共享式(上级目录)"
    if cand.is_file():
        return cand, kind
    return None, kind


def check_meshes(info: UrdfInfo) -> tuple[int, list[tuple[str, str]]]:
    """检查 URDF 引用的 mesh 是否都能解析到。返回 (ok 数, [(引用, 原因)])。

    为什么这是必查项：mesh 引用普遍是**相对路径**，换 URDF 时若只换文件、
    不调目录结构（或反过来），加载会直接失败 —— 而报错信息往往只提"找不到文件"，
    很难定位到"是目录布局不匹配"。
    """
    uniq = sorted(set(info.mesh_refs))
    ok = 0
    bad: list[tuple[str, str]] = []
    for ref in uniq:
        p, why = resolve_mesh(ref, info.path.parent)
        if p is None:
            bad.append((ref, why))
        else:
            ok += 1
    return ok, bad


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def hr(txt: str = "") -> None:
    print("\n" + "═" * 78)
    if txt:
        print(f"  {txt}")
        print("═" * 78)


def section(txt: str) -> None:
    print(f"\n── {txt} " + "─" * max(0, 74 - len(txt)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--urdf", required=True, help="待验证的 URDF（换上去的那份）")
    ap.add_argument("--xml", default=None, help="本地回放 MJCF（含 actuator/keyframe 的那份）")
    ap.add_argument("--npz", default=None, help="参考动作数据集（含 joint_names 字段）")
    ap.add_argument("--expect", default=None, choices=sorted(KNOWN_ORDERS),
                    help="期望的策略侧顺序（默认自动匹配所有已知顺序）")
    ap.add_argument("--json", default=None, help="把结论写到 JSON")
    args = ap.parse_args(argv)

    urdf_path = Path(args.urdf)
    if not urdf_path.is_file():
        print(f"❌ 找不到 URDF：{urdf_path}")
        return 2

    report: dict = {"urdf": str(urdf_path)}
    info = parse_urdf(urdf_path)
    act_names = [j.name for j in info.actuated]

    hr("① URDF 概览")
    print(f"  文件      : {info.path}")
    print(f"  robot 名  : {info.robot_name}")
    print(f"  根 link   : **{info.root_link}**   ← 浮动基（FK / 参考动作 base 必须与它一致！）")
    print(f"  link 数   : {len(info.link_masses)}")
    print(f"  关节总数  : {len(info.joints)}（其中受驱动 {len(act_names)}，fixed "
          f"{sum(1 for j in info.joints if not j.is_actuated)}）")
    print(f"  总质量    : {info.total_mass:.4f} kg")
    print(f"  声明序    : {', '.join(act_names[:3])}, … , {', '.join(act_names[-2:])}")
    report["overview"] = {
        "robot_name": info.robot_name,
        "root_link": info.root_link,
        "n_links": len(info.link_masses),
        "n_joints": len(info.joints),
        "n_actuated": len(act_names),
        "total_mass": round(info.total_mass, 6),
        "declaration_order": act_names,
    }

    section("关节明细（URDF <joint> 声明序）")
    print(f"  {'#':>3} {'关节名':<28} {'type':<11} {'parent → child':<44} axis")
    for i, j in enumerate(info.actuated):
        ax = "-" if j.axis is None else f"({j.axis[0]:g},{j.axis[1]:g},{j.axis[2]:g})"
        print(f"  {i:>3} {j.name:<28} {j.jtype:<11} {j.parent + ' → ' + j.child:<44} {ax}")
    report["joints"] = [
        {
            "name": j.name,
            "type": j.jtype,
            "parent": j.parent,
            "child": j.child,
            "axis": j.axis,
            "limit": j.limit,
            "effort": j.effort,
        }
        for j in info.actuated
    ]

    # ---- ①b mesh 引用可解析性 --------------------------------------------
    section("mesh 引用（加载失败最常见的原因）")
    n_ok, bad_mesh = check_meshes(info)
    n_ref = len(set(info.mesh_refs))
    styles = {resolve_mesh(r, urdf_path.parent)[1] for r in set(info.mesh_refs)}
    print(f"  引用 mesh 数 : {n_ref}（去重后）")
    print(f"  可解析       : {n_ok}/{n_ref} " + ("✅" if not bad_mesh else "❌"))
    print(f"  路径风格     : {'; '.join(sorted(styles))}")
    if bad_mesh:
        print("  ❌ 以下引用无法解析（**模型加载会直接失败**）：")
        for ref, why in bad_mesh[:10]:
            print(f"     {ref}   ← {why}")
        print("     ⚠️ 注意 `meshes/x.STL`（自包含）与 `../meshes/x.STL`（共享上级目录）"
              "是两种不同布局，换 URDF 时**目录结构必须与引用写法一致**。")
    report["meshes"] = {
        "n_ref": n_ref, "n_resolved": n_ok, "unresolved": [b[0] for b in bad_mesh],
        "styles": sorted(styles),
    }

    # ---- ② 与已知顺序的关系 ------------------------------------------------
    hr("② 关节顺序对齐（四套顺序互不相同，混用即错位）")
    verdict: dict[str, str] = {}
    for key, ref in KNOWN_ORDERS.items():
        kind = permutation_kind(act_names, ref)
        verdict[key] = kind
        mark = {"same": "✅ 逐项一致", "permutation": "⚠️ 同集合、顺序不同",
                "different": "❌ 集合不同"}[kind]
        print(f"  {key:<9} {mark:<18} {ORDER_TITLE.get(key, '')}")
    report["order_match"] = verdict

    section("与各已知顺序的差异细节")
    for key, ref in KNOWN_ORDERS.items():
        if verdict[key] == "same":
            continue
        miss = [n for n in ref if n not in set(act_names)]
        extra = [n for n in act_names if n not in set(ref)]
        if miss or extra:
            print(f"  [{key}] 缺失(参考里有、本 URDF 没有) {len(miss)}: {miss}")
            if extra:
                print(f"  [{key}] 多余(本 URDF 有、参考里没有) {len(extra)}: {extra}")
        else:
            print(f"  [{key}] 名字齐全，仅顺序不同")
            idx = reorder_index(act_names, ref)
            print(f"       重排索引 act→{key}: {idx}")

    # ---- ③ MJCF 侧（本地回放） --------------------------------------------
    if args.xml:
        xml_path = Path(args.xml)
        hr("③ 本地回放 MJCF")
        if not xml_path.is_file():
            print(f"  ❌ 找不到 MJCF：{xml_path}")
        else:
            try:
                import mujoco

                model = mujoco.MjModel.from_xml_path(str(xml_path))
                ctrl = mujoco_order_from_xml(str(xml_path))
                state = mujoco_state_order_from_model(model)
                # 根 body：自由关节所在的 body
                root_body = "<unknown>"
                for j in range(model.njnt):
                    if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                        root_body = mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j])
                        )
                        break
                key_z = float(model.key_qpos[0][2]) if model.nkey > 0 else None
                print(f"  文件        : {xml_path}")
                print(f"  nq/nv/nu    : {model.nq}/{model.nv}/{model.nu}")
                print(f"  根 body     : **{root_body}**")
                print(f"  keyframe 根高: {key_z}")
                print(f"  ctrl 序     : {' == mjcf 已知序 ✅' if same_order(ctrl, MUJOCO_ORDER) else '⚠️ 与 mjcf 已知序不同'}")
                print(f"  状态序      : {' == npz 序（=URDF 声明序）✅' if same_order(state, NPZ_ORDER) else '⚠️ 与 npz 序不同'}；"
                      f"{' == mjcf(ctrl) 序 ✅（旧资产那种两序重合）' if same_order(state, MUJOCO_ORDER) else '≠ mjcf(ctrl) 序❗两序不同，读 qpos 必须用状态序'}")
                print("  ⚠️ ctrl 序 ≠ 状态序：写 ctrl 用前者，读 qpos/qvel 用后者，别混。")
                report["mjcf"] = {
                    "path": str(xml_path),
                    "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
                    "root_body": root_body,
                    "keyframe_root_z": key_z,
                    "ctrl_order": ctrl,
                    "state_order": state,
                    "ctrl_is_mjcf_order": same_order(ctrl, MUJOCO_ORDER),
                    "state_is_npz_order": same_order(state, NPZ_ORDER),
                    "state_is_mjcf_order": same_order(state, MUJOCO_ORDER),
                }
                if root_body and info.root_link and root_body != info.root_link:
                    print(f"  ❌ 根 body 不一致：MJCF={root_body} vs URDF={info.root_link}"
                          " ⇒ 本地回放与训练侧不是同一根，观测/动作语义会错位")
            except ImportError:
                print("  ⚠️ 未安装 mujoco，跳过 MJCF 检查（pip install mujoco）")
            except Exception as exc:  # noqa: BLE001
                print(f"  ❌ MJCF 加载失败：{type(exc).__name__}: {exc}")

    # ---- ④ 参考动作数据集 -----------------------------------------------
    if args.npz:
        npz_path = Path(args.npz)
        hr("④ 参考动作数据集对齐")
        if not npz_path.is_file():
            print(f"  ❌ 找不到 npz：{npz_path}")
        else:
            import numpy as np

            with np.load(npz_path, allow_pickle=True) as z:
                keys = list(z.keys())
                names = [str(s) for s in z["joint_names"]] if "joint_names" in z else []
            print(f"  文件    : {npz_path}")
            print(f"  字段    : {keys}")
            if not names:
                print("  ⚠️ 没有 joint_names 字段 ⇒ 无法按名对齐，只能靠列序假设（风险高）")
                report["npz"] = {"path": str(npz_path), "has_joint_names": False}
            else:
                missing = [n for n in names if n not in set(act_names)]
                extra = [n for n in act_names if n not in set(names)]
                print(f"  npz 关节数: {len(names)}")
                print(f"  数据集里有、URDF 没有 : {missing if missing else '（无）✅'}")
                print(f"  URDF 有、数据集里没有 : {extra if extra else '（无）✅'}")
                if not missing:
                    idx = reorder_index(act_names, names)
                    print(f"  URDF 声明序 → npz 列序 的重排索引：{idx}")
                    print("  ✅ 可按**名字**对齐（加载器用 joint_names 重排即可，与列序无关）")
                report["npz"] = {
                    "path": str(npz_path),
                    "has_joint_names": True,
                    "n": len(names),
                    "missing_in_urdf": missing,
                    "extra_in_urdf": extra,
                }

    # ---- ⑤ 镜像对称性 ----------------------------------------------------
    hr("⑤ 镜像对称性（mirror loss / 对称增强依赖）")
    #: ⚠️ 判据说明（不要搞反）：**左右同名关节的 axis 在 URDF 里本来就是相同的**
    #: （如 left/right_knee_joint 都是 (0,1,0)）。矢状面镜像的"符号相反"效果来自
    #: ① 各自的局部坐标系互为镜像（由 origin 实现），② 我们在 `utils/mirror.py` 里
    #: 对关节角施加的符号规则：**pitch 保号、roll / yaw 反号**。
    #: 所以这里只检查"成对性 + axis 是否左右一致 + origin.y 是否互为相反数"，
    #: 它们才是换 URDF 时真会被破坏的东西。
    pairs, solo = mirror_pairs(act_names)
    print(f"  左右成对关节: {len(pairs)} 组")
    print(f"  无左右对应  : {solo if solo else '（无）'}")
    axis_bad: list[str] = []
    origin_bad: list[str] = []
    for ln, rn in pairs:
        lj = next((k for k in info.actuated if k.name == ln), None)
        rj = next((k for k in info.actuated if k.name == rn), None)
        if lj is None or rj is None:
            continue
        if lj.axis is not None and rj.axis is not None:
            if not all(abs(a - b) < 1e-6 for a, b in zip(lj.axis, rj.axis)):
                axis_bad.append(f"{ln}={lj.axis} vs {rn}={rj.axis}")
        if lj.origin_xyz is not None and rj.origin_xyz is not None:
            if abs(lj.origin_xyz[1] + rj.origin_xyz[1]) > 1e-4:
                origin_bad.append(
                    f"{ln}.origin.y={lj.origin_xyz[1]:+.6f} vs "
                    f"{rn}.origin.y={rj.origin_xyz[1]:+.6f}"
                )
    print("  axis 左右一致        : " + ("✅" if not axis_bad else f"⚠️ {axis_bad[:4]}"))
    print("  origin.y 互为相反数  : " + ("✅" if not origin_bad else f"⚠️ {origin_bad[:4]}"))
    print("  关节角符号规则（代码侧，不在 URDF 里）:")
    print("    pitch 保号 / roll 反号 / yaw 反号；腰部 yaw−、roll−、pitch+；"
          "left_* ↔ right_* 互换")
    print("    实现见 instinct_xu/.../instinctlab/utils/mirror.py（换资产后要复核这里的假设）")
    report["mirror"] = {
        "n_pairs": len(pairs),
        "solo": solo,
        "axis_mismatch": axis_bad,
        "origin_y_mismatch": origin_bad,
        "sign_rule": "pitch keep; roll/yaw negate; waist yaw-, roll-, pitch+",
    }
    report["symmetry_ok"] = not (axis_bad or origin_bad)

    # ---- 结论 -------------------------------------------------------------
    hr("结论")
    fatal: list[str] = []
    if len(act_names) != 29:
        fatal.append(f"受驱动关节数 = {len(act_names)}（G1 训练接口固定 29）")
    if bad_mesh:
        fatal.append(f"有 {len(bad_mesh)} 个 mesh 引用无法解析（加载会直接失败）")
    for key, ref in KNOWN_ORDERS.items():
        if verdict[key] == "different":
            miss = [n for n in ref if n not in set(act_names)]
            if miss:
                fatal.append(f"相对 {key} 缺失关节 {len(miss)} 个：{miss[:5]}…")
    matched = [k for k, v in verdict.items() if v == "same"]
    report["fatal"] = fatal
    report["matched_orders"] = matched

    if not fatal:
        print("  ✅ 关节名集合齐备（29/29），参考数据集可按名对齐 ⇒ 具备回放条件。")
        print(f"     与已知顺序**逐项一致**的：{matched if matched else '（无，需重排）'}")
        print("\n  ⚠️⚠️ 以上判定只覆盖 **URDF 声明序** 与 **MJCF ctrl/状态序**。")
        print("     **策略侧（IsaacLab）的关节序本脚本无法判定** —— 它由 IsaacLab 把 URDF")
        print("     转成 USD 之后运行时给出，既非声明序、也非 DFS 序（换根必然改序）。")
        print("     ⇒ 必须云端实测（指南 §5 Step 5），再：")
        print("        ① 回填到契约常量（如 `P1_JOINT_ORDER`）；")
        print("        ② 本脚本用 `--expect <序名>` 复核；")
        print("        ③ 同步 `instinct/tools/joint_order.py` 的 ORDER_CANDIDATES（供本地回放）。")
        if any(v == "same" for k, v in verdict.items() if k in ("isaaclab", "p1")):
            print("     （本 URDF 的声明序恰好等于某个已知**策略**序，但这不能替代实测。）")
        rc = 0
    else:
        print("  ❌ 存在致命问题，**不要**直接上策略回放：")
        for f in fatal:
            print(f"     · {f}")
        print("\n  ⇒ 兜底方案：改用 **.npz 参考动作数据集**回放（走按名重排，不依赖策略接口）。")
        print("     见 `docs/urdf_swap/URDF_REPLACEMENT_GUIDE.md` §6。")
        rc = 1

    if args.expect:
        ok = verdict.get(args.expect) == "same"
        print(f"\n  --expect {args.expect}: {'✅ 一致' if ok else '❌ 不一致'}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"\n  JSON 已写入 {args.json}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
