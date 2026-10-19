#!/usr/bin/env python3
"""生成「可回放」的 G1 29-DoF MJCF（注入执行器 + 默认站姿 keyframe）

为什么需要
----------
InstinctMJ / InstinctLab 的 G1 MJCF **本身 nu=0**（无执行器），执行器是
在代码里通过 `beyondmimic_g1_29dof_delayed_actuator_cfgs` 注入的。
因此直接加载原始 MJCF 无法闭环控制，本地回放必须先补上：

  1. `<actuator>`：29 个 position 执行器，kp/kd 由 BeyondMimic 的
     `armature × (10Hz·2π)²` 公式推出（见 `g1_params.py`，可追溯到 InstinctMJ 源码）
  2. `<keyframe>`：popsicle 默认站姿（`_POPSICLE_INIT_STATE`），root 高 0.82 m

用法
----
    python tools/build_g1_mjcf.py                       # 用默认 popsicle 模型
    python tools/build_g1_mjcf.py --with-shoe           # 用带鞋变体（跑酷任务实际用的）
    python tools/build_g1_mjcf.py --out assets/my.xml   # 自定义输出

输出默认写到 `instinct/assets/g1_29dof_popsicle_play.xml`（自包含，meshdir 用绝对路径）。
生成后会**自动回读校验**：nu==29、nkey==1、ctrl 顺序 == MUJOCO_ORDER。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import g1_params as P  # noqa: E402
from joint_order import MUJOCO_ORDER  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INSTINCT_MJ = ROOT / "repos/InstinctMJ/src/instinct_mj"

BASE_POPSICLE = INSTINCT_MJ / "assets/resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml"
BASE_POPSICLE_SHOE = INSTINCT_MJ / "tasks/parkour/mjcf/g1_29dof_torsoBase_popsicle_with_shoe.xml"
DEFAULT_OUT = ROOT / "assets/g1_29dof_popsicle_play.xml"


def build_actuator_block() -> str:
    lines = ["  <actuator>"]
    for name in MUJOCO_ORDER:
        act = P.actuator_for(name)
        if act is None:
            raise SystemExit(f"未匹配到执行器参数: {name}")
        kp, kd, effort, vlim, group = act
        lines.append(
            f"    <!-- {group} | effort_limit={effort:.0f} N·m | velocity_limit={vlim:.0f} rad/s -->"
        )
        # ⭐ `forcerange` 对齐训练侧的 `effort_limit_sim`（MuJoCo 的 <position> 默认不限力矩）。
        lines.append(
            f'    <position name="{name.replace("_joint", "")}_act" joint="{name}" '
            f'kp="{kp:.6f}" kv="{kd:.6f}" ctrllimited="false" '
            f'forcerange="{-effort:.6f} {effort:.6f}"/>'
        )
    lines.append("  </actuator>")
    return "\n".join(lines)


#: 每个执行器组对应的**转子惯量**（与训练侧 `beyondmimic_g1_29dof_actuators` 的 `armature=` 逐组一致）
ARMATURE_BY_GROUP: dict[str, float] = {
    "legs_pitch_yaw": P.ARMATURE_7520_14,     # hip_pitch / hip_yaw
    "legs_roll_knee": P.ARMATURE_7520_22,     # hip_roll / knee
    "waist_yaw": P.ARMATURE_7520_14,
    "feet": 2.0 * P.ARMATURE_5020,            # ankle_pitch / ankle_roll
    "waist": 2.0 * P.ARMATURE_5020,           # waist_roll / waist_pitch
    "arms_5020": P.ARMATURE_5020,             # shoulder_* / elbow / wrist_roll
    "arms_4010": P.ARMATURE_4010,             # wrist_pitch / wrist_yaw
}


def armature_for(joint_name: str) -> float:
    act = P.actuator_for(joint_name)
    if act is None:
        raise KeyError(joint_name)
    group = act[4]
    if group not in ARMATURE_BY_GROUP:
        raise KeyError(f"组 {group!r} 没有对应 armature")
    return ARMATURE_BY_GROUP[group]


def patch_joint_physics(text: str) -> tuple[str, int]:
    """把基础 MJCF 的 `<default><joint damping=0.5 armature=0.01 frictionloss=0.1>`
    **逐关节覆盖**掉（per-joint 属性优先于 `<default>`）。

    为什么要覆盖（对齐训练侧）：
      * 训练侧 IsaacLab 的关节 **armature 是 per-joint**（`beyondmimic_g1_29dof_actuators`
        的 `armature=`，即 0.0036 / 0.0102 / 0.0251 三档）。
      * 训练侧**没有额外的关节阻尼/摩擦**：阻尼完全由 PD 的 `damping=`（即 kv）承担；
        URDF 里也没有 `<dynamics>`（即 friction=armature=0），且执行器配置未设 `friction`。
      * 而基础 MJCF 是**全局** `damping=0.5 / armature=0.01 / frictionloss=0.1`
        ⇒ 回放里关节明显更“粘”、转子惯量也偏大 ⇒ **摆腿更费劲 ⇒ 步幅偏小、频率偏高**
        （这正是“回放看着是小碎步”的可疑来源之一）。
    """
    patched = 0
    missing: list[str] = []

    def repl(m: re.Match) -> str:
        nonlocal patched
        name, rest = m.group(1), m.group(2)
        if 'type="free"' in rest:      # 浮动基的 free joint 不做处理
            return m.group(0)
        try:
            arm = armature_for(name)
        except KeyError:
            missing.append(name)
            return m.group(0)
        for key, val in (("armature", f"{arm:.9f}"), ("damping", "0"), ("frictionloss", "0")):
            if re.search(rf'\b{key}="[^"]*"', rest):
                rest = re.sub(rf'\b{key}="[^"]*"', f'{key}="{val}"', rest)
            else:
                rest = f'{rest} {key}="{val}"'
        patched += 1
        return f'<joint name="{name}"{rest}/>'

    text = re.sub(r'<joint\s+name="([^"]+)"([^>]*?)/>', repl, text)
    if missing:
        print(f"[warn] {len(missing)} 个关节未匹配到执行器组（物理参数未改）: {missing[:6]}", file=sys.stderr)
    print(f"[joint] 逐关节覆盖 armature/damping/frictionloss：{patched} 个")
    return text, patched


def build_keyframe_block(base: Path = BASE_POPSICLE, init_root_height: float = P.INIT_ROOT_HEIGHT) -> str:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(base))
    qpos = m.qpos0.copy()
    # 定位免费关节占用的 qpos 长度
    free_adr = None
    for j in range(m.njnt):
        if int(m.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
            free_adr = int(m.jnt_qposadr[j])
            break
    if free_adr is None:
        raise SystemExit("模型里找不到 free joint（浮动基），无法生成 keyframe")
    free_len = 7
    qpos[free_adr + 2] = init_root_height  # z
    for j in range(m.njnt):
        if int(m.jnt_type[j]) != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        qpos[int(m.jnt_qposadr[j])] = P.init_pos_for(name)
    vec = " ".join(f"{v:.6g}" for v in qpos)
    return (
        "  <keyframe>\n"
        '    <!-- popsicle 默认站姿 (_POPSICLE_INIT_STATE)，root 高 '
        f"{init_root_height} m -->\n"
        f'    <key name="home" qpos="{vec}"/>\n'
        "  </keyframe>"
    )


GROUND_BLOCK = """  <!-- 由 build_g1_mjcf.py 注入：地面 + 光照（裸 MJCF 没有场景，不加会直接穿地掉落） -->
  <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
  <light pos="0 3 2" dir="0 -0.6 -1" directional="true" diffuse="0.4 0.4 0.4"/>
  <geom name="floor" type="plane" size="8 8 0.05" rgba="0.55 0.6 0.65 1" condim="3" friction="1.0 1.0 0.005"/>
"""


def inject_ground(text: str, base: Path) -> tuple[str, bool]:
    """在 <worldbody> 内注入地面与光照。已存在 plane 时跳过。"""
    m = re.search(r"<worldbody[^>]*>", text)
    if m is None:
        raise SystemExit("基础 MJCF 缺少 <worldbody>，无法注入地面")
    # 已有平面地形则不再注入
    body_tail = text[m.end():]
    if re.search(r'type\s*=\s*"plane"', body_tail[: body_tail.find("</worldbody>") if "</worldbody>" in body_tail else len(body_tail)]):
        print("[info] 基础 MJCF 已含 plane 地面，跳过注入")
        return text, False
    return text[: m.end()] + "\n" + GROUND_BLOCK + text[m.end():], True


def generate(
    base: Path,
    out: Path,
    init_root_height: float = P.INIT_ROOT_HEIGHT,
    strip_actuator: bool = False,
    strip_scene: bool = False,
) -> Path:
    if not base.is_file():
        raise SystemExit(f"基础 MJCF 不存在: {base}\n请确认已 clone InstinctMJ 仓库。")

    text = base.read_text(encoding="utf-8")

    # 0) 可选：先剥掉基础模型自带的 <actuator>（官方 G1 MJCF 是 <motor>，本脚本要自建 <position>）
    if strip_actuator and "<actuator" in text:
        text, n_act = re.subn(r"\n?\s*<actuator>.*?</actuator>", "", text, flags=re.S)
        if n_act:
            print(f"[strip] 已移除基础模型自带的 <actuator> 块（{n_act} 处）")

    # 0.5) 可选：剥离基础模型尾部自带的“场景块”（第 2 个及以后的 <asset>/<worldbody>）。
    #      官方 rev1.0 的 xml 把 天空盒/棋盘地面/灯光 追加在机器人后面，
    #      会和本脚本注入的 `floor` 重名（MuJoCo 直接报 repeated name）。
    if strip_scene:
        for tag in ("asset", "worldbody"):
            blocks = list(re.finditer(rf"\n?[ \t]*<{tag}>.*?</{tag}>", text, flags=re.S))
            for b in reversed(blocks[1:]):
                text = text[: b.start()] + text[b.end():]
            if len(blocks) > 1:
                print(f"[strip] 已移除第 2 个及以后的 <{tag}> 块（{len(blocks) - 1} 处）")

    # 1) meshdir 改为绝对路径，使生成文件可放在任意位置
    def abs_meshdir(match: re.Match) -> str:
        raw = match.group(1)
        resolved = (base.parent / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
        return f'meshdir="{resolved}"'

    text, n_sub = re.subn(r'meshdir="([^"]*)"', abs_meshdir, text, count=1)
    if n_sub == 0:
        print("[warn] 基础 MJCF 没有 meshdir 属性，若网格找不到请手工处理", file=sys.stderr)

    # 2) 模型名加后缀，避免与原始模型混淆
    text = re.sub(r'<mujoco model="([^"]*)"', r'<mujoco model="\1_play"', text, count=1)

    # 2.5) ⭐ 逐关节覆盖 armature/damping/frictionloss（对齐训练侧，见 patch_joint_physics）
    text, _n_joint = patch_joint_physics(text)

    # 3) 注入地面 + 光照（裸 MJCF 无场景，不加会穿地掉落）
    text, injected = inject_ground(text, base)
    if injected:
        print("[inject] 已注入地面 + 光照")

    # 4) 插入 actuator + keyframe
    if "<actuator" in text:
        raise SystemExit("基础 MJCF 已含 <actuator>，本脚本会重复注入；请改用 --base 指定无执行器的模型")
    if "<keyframe" in text:
        raise SystemExit("基础 MJCF 已含 <keyframe>，请改用 --base 指定无 keyframe 的模型")

    payload = (
        "\n\n"
        + build_actuator_block()
        + "\n\n"
        + build_keyframe_block(base, init_root_height)
        + "\n"
    )
    if "</mujoco>" not in text:
        raise SystemExit("基础 MJCF 缺少 </mujoco> 结束标签")
    text = text.replace("</mujoco>", payload + "</mujoco>")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def verify(path: Path) -> bool:
    import mujoco

    print(f"\n=== 回读校验 {path.name} ===")
    try:
        m = mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 加载失败: {type(exc).__name__}: {exc}")
        return False

    ok = True
    print(f"  nq={m.nq} nv={m.nv} nu={m.nu} nkey={m.nkey} njnt={m.njnt} dt={m.opt.timestep}")
    if m.nu != 29:
        print(f"  ❌ nu={m.nu}，期望 29")
        ok = False
    else:
        print("  ✅ nu=29")

    if m.nu:
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0])) for i in range(m.nu)]
        if list(names) == list(MUJOCO_ORDER):
            print("  ✅ ctrl 顺序 == MUJOCO_ORDER")
        else:
            print("  ❌ ctrl 顺序与 MUJOCO_ORDER 不符")
            for i, (a, b) in enumerate(zip(names, MUJOCO_ORDER)):
                if a != b:
                    print(f"     [{i}] 实际={a}  期望={b}")
            ok = False

        # kp/kv 是否真的写进去了
        kp = m.actuator_gainprm[:, 0]
        kv = -m.actuator_biasprm[:, 2]
        exp_kp = [P.actuator_for(n)[0] for n in names]
        exp_kv = [P.actuator_for(n)[1] for n in names]
        import numpy as np

        if np.allclose(kp, exp_kp, rtol=1e-6):
            print(f"  ✅ kp 写入正确 (范围 {kp.min():.3f} ~ {kp.max():.3f})")
        else:
            print("  ❌ kp 与期望不符")
            ok = False
        if np.allclose(kv, exp_kv, rtol=1e-6):
            print(f"  ✅ kv/kd 写入正确 (范围 {kv.min():.4f} ~ {kv.max():.4f})")
        else:
            print("  ❌ kv 与期望不符（可能 MuJoCo 的 <position> 不支持 kv）")
            ok = False

        # ⭐ 力矩上限（对齐训练侧 effort_limit_sim）
        limited = int(np.sum(m.actuator_forcelimited.astype(bool)))
        exp_eff = np.array([P.actuator_for(n)[2] for n in names])
        fr = m.actuator_forcerange
        if limited == m.nu and np.allclose(fr[:, 1], exp_eff, rtol=1e-6):
            print(f"  ✅ forcerange 写入正确（{limited}/{m.nu}，上限 {exp_eff.min():.0f}~{exp_eff.max():.0f} N·m）")
        else:
            print(f"  ❌ forcerange 不对：limited={limited}/{m.nu}")
            ok = False

    # ⭐ 逐关节 armature / damping / frictionloss（对齐训练侧：per-joint armature、无额外阻尼/摩擦）
    bad_arm, bad_dmp, bad_fric = [], [], []
    for i in range(m.njnt):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jn is None or int(m.jnt_type[i]) != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        dof = int(m.jnt_dofadr[i])
        try:
            exp_arm = armature_for(jn)
        except KeyError:
            continue
        if abs(float(m.dof_armature[dof]) - exp_arm) > 1e-9:
            bad_arm.append((jn, float(m.dof_armature[dof]), exp_arm))
        if abs(float(m.dof_damping[dof])) > 1e-12:
            bad_dmp.append((jn, float(m.dof_damping[dof])))
        if abs(float(m.dof_frictionloss[dof])) > 1e-12:
            bad_fric.append((jn, float(m.dof_frictionloss[dof])))
    if bad_arm:
        print(f"  ❌ armature 不符（{len(bad_arm)} 个），例：{bad_arm[:3]}")
        ok = False
    else:
        print("  ✅ 逐关节 armature 与训练侧一致")
    if bad_dmp:
        print(f"  ❌ 仍有额外关节阻尼（{len(bad_dmp)} 个），例：{bad_dmp[:3]}")
        ok = False
    else:
        print("  ✅ 无额外关节阻尼（阻尼全部由 PD 的 kv 承担）")
    if bad_fric:
        print(f"  ❌ 仍有 frictionloss（{len(bad_fric)} 个），例：{bad_fric[:3]}")
        ok = False
    else:
        print("  ✅ 无 frictionloss")

    if m.nkey == 1:
        print(f"  ✅ keyframe '{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, 0)}'")
    else:
        print(f"  ⚠️  nkey={m.nkey}")

    # 地面检查（没有地面会直接穿地掉落）
    has_plane = any(
        int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE) and int(m.geom_bodyid[g]) == 0
        for g in range(m.ngeom)
    )
    if has_plane:
        print("  ✅ 世界地面 (plane) 已注入")
    else:
        print("  ❌ 缺少世界地面 → 回放会穿地掉落")
        ok = False

    # 动作尺度汇总
    scales = [P.action_scale_for(n) for n in MUJOCO_ORDER]
    print(f"\n  逐关节 action_scale 范围: {min(scales):.5f} ~ {max(scales):.5f}")
    print("  （注意：不是全局 0.25！）")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-shoe", action="store_true", help="使用跑酷任务的带鞋变体作为基础模型")
    ap.add_argument("--base", help="自定义基础 MJCF（必须无 actuator/keyframe）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help=f"输出路径（默认 {DEFAULT_OUT}）")
    ap.add_argument(
        "--init-root-height",
        type=float,
        default=P.INIT_ROOT_HEIGHT,
        help=(
            f"keyframe 里**根刚体**的离地高度（默认 {P.INIT_ROOT_HEIGHT}）。"
            "torso 根用 0.82；官方 pelvis 根模型用 0.7585（= 默认站位骨盆高度，已实测）"
        ),
    )
    ap.add_argument(
        "--strip-actuator",
        action="store_true",
        help="先剥掉基础模型自带的 <actuator> 块（官方 G1 MJCF 为 <motor>，不剥会与本脚本自建的冲突）",
    )
    ap.add_argument(
        "--strip-scene",
        action="store_true",
        help=(
            "先剥掉基础模型自带的“场景块”（第 2 个及以后的 <asset>/<worldbody>："
            "官方 rev1.0 xml 的天空盒+棋盘地面+灯光），避免与本脚本注入的地面重名"
        ),
    )
    args = ap.parse_args(argv)

    base = Path(args.base) if args.base else (BASE_POPSICLE_SHOE if args.with_shoe else BASE_POPSICLE)
    out = Path(args.out)

    print(f"基础模型: {base}")
    print(f"输出路径: {out}")
    generate(
        base,
        out,
        init_root_height=args.init_root_height,
        strip_actuator=args.strip_actuator,
        strip_scene=args.strip_scene,
    )
    print(f"✅ 已生成 {out}  ({out.stat().st_size / 1024:.1f} KB)")
    return 0 if verify(out) else 1


if __name__ == "__main__":
    sys.exit(main())
