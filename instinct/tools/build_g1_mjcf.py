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
        lines.append(
            f'    <position name="{name.replace("_joint", "")}_act" joint="{name}" '
            f'kp="{kp:.6f}" kv="{kd:.6f}" ctrllimited="false"/>'
        )
    lines.append("  </actuator>")
    return "\n".join(lines)


def build_keyframe_block() -> str:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(BASE_POPSICLE))
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
    qpos[free_adr + 2] = P.INIT_ROOT_HEIGHT  # z
    for j in range(m.njnt):
        if int(m.jnt_type[j]) != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        qpos[int(m.jnt_qposadr[j])] = P.init_pos_for(name)
    vec = " ".join(f"{v:.6g}" for v in qpos)
    return (
        "  <keyframe>\n"
        '    <!-- popsicle 默认站姿 (_POPSICLE_INIT_STATE)，root 高 '
        f"{P.INIT_ROOT_HEIGHT} m -->\n"
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


def generate(base: Path, out: Path) -> Path:
    if not base.is_file():
        raise SystemExit(f"基础 MJCF 不存在: {base}\n请确认已 clone InstinctMJ 仓库。")

    text = base.read_text(encoding="utf-8")

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

    # 3) 注入地面 + 光照（裸 MJCF 无场景，不加会穿地掉落）
    text, injected = inject_ground(text, base)
    if injected:
        print("[inject] 已注入地面 + 光照")

    # 4) 插入 actuator + keyframe
    if "<actuator" in text:
        raise SystemExit("基础 MJCF 已含 <actuator>，本脚本会重复注入；请改用 --base 指定无执行器的模型")
    if "<keyframe" in text:
        raise SystemExit("基础 MJCF 已含 <keyframe>，请改用 --base 指定无 keyframe 的模型")

    payload = "\n\n" + build_actuator_block() + "\n\n" + build_keyframe_block() + "\n"
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
    args = ap.parse_args(argv)

    base = Path(args.base) if args.base else (BASE_POPSICLE_SHOE if args.with_shoe else BASE_POPSICLE)
    out = Path(args.out)

    print(f"基础模型: {base}")
    print(f"输出路径: {out}")
    generate(base, out)
    print(f"✅ 已生成 {out}  ({out.stat().st_size / 1024:.1f} KB)")
    return 0 if verify(out) else 1


if __name__ == "__main__":
    sys.exit(main())
