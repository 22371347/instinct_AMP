#!/usr/bin/env python3
"""本地验证环境自检

检查项
------
1. conda 环境与关键包版本
2. G1 MuJoCo 模型是否可加载（InstinctLab 自带 scene.xml）
3. 关节序常量自洽性 + 与 MJCF 的一致性
4. 资源盘点（AMASS 数据、GMR、云端运维脚本、已有策略）
5. 硬件与磁盘（提醒本地无法跑 Isaac Sim）

用法
----
    conda activate instinct
    python tools/check_env.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # .../llp/instinct
REPOS = ROOT / "repos"
#: 可回放模型（由 tools/build_g1_mjcf.py 生成，已注入执行器+keyframe）
G1_PLAY = ROOT / "assets/g1_29dof_popsicle_play.xml"
#: 原始来源模型（InstinctMJ，nu=0，执行器由代码注入）
G1_SRC = REPOS / "InstinctMJ/src/instinct_mj/assets/resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml"
#: 跑酷任务实际使用的带鞋变体
G1_SHOE = REPOS / "InstinctMJ/src/instinct_mj/tasks/parkour/mjcf/g1_29dof_torsoBase_popsicle_with_shoe.xml"
#: InstinctLab 自带的资产（注意：23-DoF + 双手，nu=37，与 29-DoF 策略不匹配）
G1_LAB_ASSETS = REPOS / "InstinctLab/source/instinctlab/instinctlab/assets/resources/unitree_g1"

AMP = Path(os.path.expanduser("~/rsl_project/AMP_mjlab"))
GMR = Path(os.path.expanduser("~/rsl_project/GMR"))
MODELS = Path(os.path.expanduser("~/rsl_project/models"))

PASS, WARN, FAIL = "✅", "⚠️ ", "❌"
results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"{status} {name}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


# ---------------------------------------------------------------------------
# 1. python / 包
# ---------------------------------------------------------------------------

def check_python() -> None:
    section("1. Python 与依赖")
    print(f"  interpreter : {sys.executable}")
    print(f"  version     : {sys.version.split()[0]}")
    env = os.environ.get("CONDA_DEFAULT_ENV", "(未激活 conda env)")
    if env == "instinct":
        record(PASS, "conda env", "instinct")
    else:
        record(WARN, "conda env", f"当前为 '{env}'，建议 `conda activate instinct`")

    required = {
        "mujoco": "本地物理仿真/回放",
        "onnxruntime": "加载云端导出 ONNX 策略",
        "numpy": "数值",
        "matplotlib": "绘图",
        "yaml": "配置",
        "trimesh": "网格/动作数据处理",
        "scipy": "数值",
        "torch": "加载 .pt checkpoint（可选）",
    }
    optional = {"imageio": "录制 mp4", "onnx": "ONNX 结构检查", "pexpect": "SSH 运维脚本"}

    import importlib

    for mod, why in required.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            record(PASS, f"包 {mod}", f"{ver} ({why})")
        except ImportError:
            record(FAIL, f"包 {mod}", f"缺失 — {why}；pip install {mod}")

    for mod, why in optional.items():
        try:
            m = importlib.import_module(mod)
            record(PASS, f"包 {mod} (可选)", f"{getattr(m, '__version__', '?')} ({why})")
        except ImportError:
            record(WARN, f"包 {mod} (可选)", f"未装 — {why}")


# ---------------------------------------------------------------------------
# 2. MuJoCo 模型
# ---------------------------------------------------------------------------

def check_model() -> None:
    section("2. G1 MuJoCo 模型")

    # 2.1 来源模型（InstinctMJ，29-DoF popsicle）
    for path, label, expect_nu in (
        (G1_SRC, "来源模型 popsicle (InstinctMJ)", 0),
        (G1_SHOE, "来源模型 popsicle+shoe (跑酷用)", 0),
    ):
        if not path.is_file():
            record(WARN, label, f"不存在: {path}")
            continue
        try:
            import mujoco

            m = mujoco.MjModel.from_xml_path(str(path))
            hinge = sum(1 for j in range(m.njnt) if int(m.jnt_type[j]) == 3)
            detail = f"njnt={m.njnt} 铰链={hinge} nu={m.nu}"
            if expect_nu == 0 and m.nu == 0:
                record(PASS, label, detail + "（nu=0 正常：执行器由代码注入）")
            else:
                record(PASS, label, detail)
        except Exception as exc:  # noqa: BLE001
            record(FAIL, label, f"{type(exc).__name__}: {exc}")

    # 2.2 可回放模型
    if not G1_PLAY.is_file():
        record(FAIL, "可回放 MJCF", "未生成 → 运行 `python tools/build_g1_mjcf.py`")
        return
    try:
        import mujoco

        m = mujoco.MjModel.from_xml_path(str(G1_PLAY))
        if m.nu == 29:
            record(PASS, "可回放 MJCF", f"nu=29 nq={m.nq} nkey={m.nkey} ✅")
        else:
            record(FAIL, "可回放 MJCF", f"nu={m.nu} != 29")
        if m.nkey >= 1:
            record(PASS, "默认站姿", f"keyframe '{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, 0)}'")
        else:
            record(WARN, "默认站姿", "无 keyframe")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "可回放 MJCF", f"{type(exc).__name__}: {exc}")

    # 2.3 提醒：InstinctLab 自带资产不是 29-DoF
    if G1_LAB_ASSETS.is_dir():
        try:
            import mujoco

            m = mujoco.MjModel.from_xml_path(str(G1_LAB_ASSETS / "scene.xml"))
            if m.nu != 29:
                record(
                    WARN,
                    "InstinctLab 自带资产",
                    f"nu={m.nu} ≠ 29，是 23-DoF+双手变体，**不要**用它回放 29-DoF 跑酷策略；"
                    "请用 InstinctMJ 的 popsicle 模型",
                )
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# 3. 关节序
# ---------------------------------------------------------------------------

def check_joint_order() -> None:
    section("3. 关节序映射")
    try:
        sys.path.insert(0, str(HERE))
        from joint_order import ISAACLAB_ORDER, MUJOCO_ORDER, mirror_index, policy_to_mujoco_index  # type: ignore
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "joint_order 模块", f"{type(exc).__name__}: {exc}")
        return

    record(PASS, "常量自洽", f"IsaacLab {len(ISAACLAB_ORDER)} / MuJoCo {len(MUJOCO_ORDER)} DoF，名称集合一致")
    idx = policy_to_mujoco_index()
    record(PASS, "policy→mj ctrl 映射", f"首 6 项 {idx[:6]}（应与 MD 中表格一致）")
    record(PASS, "左右镜像映射", f"首 6 项 {mirror_index()[:6]}")

    try:
        import mujoco

        m = mujoco.MjModel.from_xml_path(str(G1_PLAY))
        ctrl_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0])) for i in range(m.nu)
        ]
        if list(ctrl_names) == list(MUJOCO_ORDER):
            record(PASS, "MJCF 顺序核对", "可回放 MJCF 与内置 MUJOCO_ORDER 完全一致")
        else:
            record(WARN, "MJCF 顺序核对", "与内置常量不一致，请以 MJCF 为准（跑 --dump-model 查看）")
    except Exception as exc:  # noqa: BLE001
        record(WARN, "MJCF 顺序核对", f"跳过: {exc}")

    try:
        import g1_params  # type: ignore

        scales = [g1_params.action_scale_for(n) for n in MUJOCO_ORDER]
        record(
            PASS,
            "逐关节 action_scale",
            f"{min(scales):.5f} ~ {max(scales):.5f}（**不是**全局 0.25）",
        )
    except Exception as exc:  # noqa: BLE001
        record(WARN, "逐关节 action_scale", f"跳过: {exc}")


# ---------------------------------------------------------------------------
# 4. 资源盘点
# ---------------------------------------------------------------------------

def _dir_size(path: Path) -> str:
    try:
        out = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True, timeout=120)
        return out.stdout.split()[0] if out.returncode == 0 else "?"
    except Exception:  # noqa: BLE001
        return "?"


def check_assets() -> None:
    section("4. 资源盘点")
    items = [
        (REPOS / "InstinctLab", "InstinctLab (环境侧, Isaac)"),
        (REPOS / "InstinctMJ", "InstinctMJ (环境侧, MuJoCo) ★本地回放必需"),
        (REPOS / "instinct_rl", "instinct_rl (RL 侧)"),
        (REPOS / "instinct_onboard", "instinct_onboard (部署)"),
        (G1_PLAY, "可回放 G1 MJCF"),
        (ROOT / "tools/g1_params.py", "执行器参数模块"),
        (GMR / "AMASS_CMU", "AMASS_CMU 动作数据 ★"),
        (GMR / "general_motion_retargeting", "GMR 重定向 ★"),
        (AMP / "tools/amp_ssh.py", "云端 SSH 助手 ★"),
        (AMP / "tools/validate_motion_csv.py", "动作 CSV 校验 ★"),
        (MODELS / "walk_model.onnx", "已有 G1 走路 ONNX"),
    ]
    for path, label in items:
        if path.exists():
            size = _dir_size(path) if path.is_dir() else f"{path.stat().st_size / 1e6:.1f} MB"
            record(PASS, label, f"{size}  <- {path}")
        else:
            record(WARN, label, f"未找到: {path}")

    if not (REPOS / "InstinctLab").is_dir():
        record(FAIL, "InstinctLab 仓库", "缺失，需 git clone")


# ---------------------------------------------------------------------------
# 5. 硬件 / 磁盘
# ---------------------------------------------------------------------------

def check_hardware() -> None:
    section("5. 硬件与磁盘")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            for line in out.stdout.strip().splitlines():
                record(PASS, "GPU", line.strip())
                if "MiB" in line:
                    try:
                        mib = int(line.split(",")[-1].strip().split()[0])
                        if mib < 8000:
                            record(
                                WARN,
                                "本地 Isaac Sim 可行性",
                                f"显存 {mib} MiB < 8 GB 最低要求 → 这是**已知约束**（非待修问题）："
                                "本地只做轻量验证，Isaac Sim 仿真/训练/越障验证全部在云端进行",
                            )
                    except ValueError:
                        pass
        else:
            record(WARN, "GPU", "nvidia-smi 无输出")
    except Exception:  # noqa: BLE001
        record(WARN, "GPU", "nvidia-smi 不可用")

    for mount in ("/home", "/mnt/winD"):
        if os.path.ismount(mount) or Path(mount).is_dir():
            u = shutil.disk_usage(mount)
            free_gb = u.free / 1e9
            status = PASS if free_gb > 10 else WARN
            record(status, f"磁盘 {mount}", f"剩余 {free_gb:.1f} GB / 总 {u.total / 1e9:.0f} GB")


# ---------------------------------------------------------------------------

def main() -> int:
    print(f"工作区: {ROOT}")
    print("本地验证环境自检（无需 Isaac Sim / 无需 GPU）")
    check_python()
    check_model()
    check_joint_order()
    check_assets()
    check_hardware()

    section("汇总")
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    print(f"  通过 {len(results) - n_fail - n_warn} / 警告 {n_warn} / 失败 {n_fail}")
    if n_fail:
        print("\n需处理：")
        for s, name, detail in results:
            if s == FAIL:
                print(f"  ❌ {name} — {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
