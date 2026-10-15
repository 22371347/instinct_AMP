#!/usr/bin/env python3
"""`flatten_cfg.py` 的**离线**自测（不需要 IsaacLab）。

用纯 `dataclasses` 搭一条"模拟继承链"（含：上游钩子覆盖字段、深层嵌套 cfg、
叶级 term、dict 里嵌 dataclass、列表里嵌 dataclass），然后

  ① `generate` 出展平文件 → 必须语法合法、可 import、可实例化；
  ② `verify` 链式 vs 展平 → **必须报等价**；
  ③ **反向对照**：故意改掉展平文件里的一个字面值 → `verify` **必须报差异**
     （否则这个校验器就是个摆设）。

用法：`python3 instinct/tools/test_flatten_cfg_mock.py`
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
FLATTEN = os.path.join(HERE, "flatten_cfg.py")


# ============================================================================
# 模拟配置链
# ============================================================================


def mock_reward_fn(*_a, **_k) -> float:
    return 0.0


def mock_obs_fn(*_a, **_k) -> float:
    return 0.0


@dataclass
class _Term:
    func: Any = None
    weight: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class _Entity:
    name: str = ""
    body_names: tuple = ()


@dataclass
class _Spawn:
    path: str = ""
    scale: tuple = (1.0, 1.0, 1.0)


@dataclass
class _Robot:
    spawn: Any = None
    init_z: float = 0.0


@dataclass
class _Scanner:
    rays: int = 0
    resolution: float = 0.1


@dataclass
class _Scene:
    robot: Any = None            # 嵌套 cfg → 应生成类
    scanner: Any = None          # 叶级 → 应内联
    extras: list = field(default_factory=list)   # 列表里嵌 dataclass


@dataclass
class _Rewards:
    track: Any = None            # 叶级 term → 内联
    action_rate: Any = None
    slip: Any = None


@dataclass
class MockBase:
    """模拟 `ManagerBasedRLEnvCfg`：**上游钩子会覆盖字段**（展平必须能盖回来）。"""

    decimation: int = 1
    episode_length_s: float = 0.0
    run_name: str = "base"
    scene: Any = None
    rewards: Any = None

    def __post_init__(self):
        self.decimation = 111
        self.episode_length_s = 222.0
        self.run_name = "base_hook"


@dataclass
class MockV1(MockBase):
    def __post_init__(self):
        super().__post_init__()
        self.scene = _Scene(
            robot=_Robot(spawn=_Spawn(path="/World/g1.usd", scale=(1.0, 1.5, 1.0)), init_z=0.7585),
            scanner=_Scanner(rays=279, resolution=0.1),
            extras=[_Entity(name="ground", body_names=("a", "b"))],
        )
        self.rewards = _Rewards(
            track=_Term(func=mock_reward_fn, weight=1.5, params={"std": 0.5, "sensor_cfg": _Entity("robot", ("pelvis",))}),
            action_rate=_Term(func=mock_reward_fn, weight=-0.01),
            slip=_Term(func=mock_reward_fn, weight=-0.25),
        )
        self.decimation = 4
        self.episode_length_s = 20.0
        self.run_name = "G1MockV1"


@dataclass
class MockTarget(MockV1):
    """被测"叶子"：在链尾做深层修改。"""

    def __post_init__(self):
        super().__post_init__()
        # 深层字段替换（展平必须抓到"替换后的对象"，而不是父类的）
        self.rewards.track = _Term(func=mock_obs_fn, weight=-2.0, params={"std": 0.3})
        self.scene.scanner.rays = 527
        self.rewards.slip = None                    # 删除项（None 必须被保留）
        self.decimation = 8
        self.run_name = "G1MockP1"


# ============================================================================
# 测试
# ============================================================================


def _run(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, FLATTEN, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def main() -> int:
    fails: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "mock_flat.py")
        repo_root = os.path.abspath(os.path.join(HERE, "..", ".."))

        # ---- ① generate ---------------------------------------------------
        r = _run(
            [
                "generate",
                "--cls", "test_flatten_cfg_mock:MockTarget",
                "--out", out,
                "--root-name", "MockP1EnvCfg",
                "--decorator", "dataclasses:dataclass",
                "--base", "test_flatten_cfg_mock:MockBase",
                "--no-app",
            ],
            cwd=repo_root,
        )
        print("--- generate stdout ---")
        print(r.stdout.strip() or "(空)")
        if r.returncode != 0:
            print("--- generate stderr ---")
            print(r.stderr.strip())
            return 2
        code = open(out, encoding="utf-8").read()
        print(f"--- 生成 {len(code.splitlines())} 行 ---")

        # 语法必须合法
        try:
            compile(code, out, "exec")
        except SyntaxError as e:
            fails.append(f"生成文件语法错误：{e}")

        # 展平文件里**不得** import 本项目其它 cfg 模块（自包含的核心要求）
        bad = [
            ln for ln in code.splitlines()
            if re.match(r"^(from|import)\s", ln)
            and ("_env_cfg" in ln or "_ppo_cfg" in ln or "tasks.locomotion.config" in ln)
            and "test_flatten_cfg_mock" not in ln
        ]
        if bad:
            fails.append(f"生成文件仍依赖本项目 cfg 模块：{bad}")

        # 关键：字面值必须写死（而不是继续靠继承/引用）
        for needle in ("-2.0", "0.3", "527", "'G1MockP1'", "decimation"):
            if needle not in code:
                fails.append(f"生成文件里找不到期望内容：{needle!r}")

        # ---- ② verify（必须等价）-------------------------------------------
        r = _run(
            [
                "verify",
                "--chain", "test_flatten_cfg_mock:MockTarget",
                "--flat-file", out,
                "--flat-attr", "MockP1EnvCfg",
                "--no-app",
            ],
            cwd=repo_root,
        )
        print("--- verify(应通过) ---")
        print((r.stdout or r.stderr).strip()[:2000])
        if r.returncode != 0:
            fails.append(f"verify 未能证明等价（rc={r.returncode}）")

        # ---- ③ 反向对照（必须报差异）---------------------------------------
        mutated = os.path.join(tmp, "mock_flat_mutated.py")
        # ⚠️ 替换串必须**真的命中**（生成物里是 `self.weight = -2.0`）；
        #    若没命中，本测试会假通过 —— 所以断言"文本确实变了"。
        m_code = code.replace("= -2.0", "= -2.5", 1)
        if m_code == code:
            fails.append("反向对照的替换串没命中，测试本身无效（请检查生成物格式）")
        open(mutated, "w", encoding="utf-8").write(m_code)
        r = _run(
            [
                "verify",
                "--chain", "test_flatten_cfg_mock:MockTarget",
                "--flat-file", mutated,
                "--flat-attr", "MockP1EnvCfg",
                "--no-app",
            ],
            cwd=repo_root,
        )
        print("--- verify(反向对照，应失败) ---")
        print((r.stdout or r.stderr).strip()[:600])
        if r.returncode == 0:
            fails.append("❌ 反向对照失败：改了一个权重，verify 竟然报‘等价’——校验器无效！")

    print()
    if fails:
        print("=== 自测失败 ===")
        for f in fails:
            print("  - " + f)
        return 1
    print("=== ✅ 自测全部通过（生成 + 等价 + 反向对照）===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
