#!/usr/bin/env python3
"""云端 SSH / SCP 助手（InstinctLab 工程版）

设计要点
--------
1. **不重复存放凭据**：默认复用你已经维护好的
   `~/rsl_project/AMP_mjlab/tools/amp_ssh.py`（同一台云端机器），
   必要时可通过环境变量覆盖。
2. 凭据解析优先级：
     环境变量  INSTINCT_SSH_HOST / INSTINCT_SSH_PORT / INSTINCT_SSH_USER / INSTINCT_SSH_PASSWORD
   → 复用 amp_ssh 模块（推荐，凭据只有一份）
   → 配置文件 ~/.config/instinct/ssh.env（KEY=VALUE 形式，自行 chmod 600）
3. 密码**不会**被打印到 stdout。

用法
----
    python tools/instinct_ssh.py "nvidia-smi"                 # 执行远程命令（回显）
    python tools/instinct_ssh.py --capture "nvidia-smi"       # 捕获输出（不污染 stdout）
    python tools/instinct_ssh.py --put  local.txt /root/      # 上传
    python tools/instinct_ssh.py --get  /root/a.log ./        # 下载

也可作为库使用：

    from instinct_ssh import ssh_run, ssh_run_capture, scp_put, scp_get
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 凭据解析
# ---------------------------------------------------------------------------

AMP_SSH_PATH = Path(os.path.expanduser("~/rsl_project/AMP_mjlab/tools/amp_ssh.py"))
CONFIG_PATH = Path(os.path.expanduser("~/.config/instinct/ssh.env"))

_REQUIRED = ("HOST", "PORT", "USER", "PASSWORD")


def _from_env() -> dict | None:
    vals = {k: os.environ.get(f"INSTINCT_SSH_{k}") for k in _REQUIRED}
    return vals if all(vals.values()) else None


def _from_config() -> dict | None:
    if not CONFIG_PATH.is_file():
        return None
    vals: dict[str, str] = {}
    for line in CONFIG_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        vals[k.strip().upper()] = v.strip().strip("'\"")
    return {k: vals.get(k) for k in _REQUIRED} if all(vals.get(k) for k in _REQUIRED) else None


def _from_amp_ssh() -> dict | None:
    """复用 AMP_mjlab 的 amp_ssh 模块（同机，避免第二份密码副本）。"""
    if not AMP_SSH_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_amp_ssh", AMP_SSH_PATH)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 - 只提示，不致命
        print(f"[credentials] 加载 {AMP_SSH_PATH} 失败: {exc}", file=sys.stderr)
        return None
    try:
        return {
            "HOST": getattr(mod, "HOST"),
            "PORT": str(getattr(mod, "PORT")),
            "USER": getattr(mod, "USER"),
            "PASSWORD": getattr(mod, "PASSWORD"),
        }
    except AttributeError:
        return None


def resolve_credentials() -> dict:
    for name, getter in (
        ("环境变量", _from_env),
        ("amp_ssh 模块", _from_amp_ssh),
        ("配置文件", _from_config),
    ):
        creds = getter()
        if creds:
            print(f"[credentials] 来源: {name}", file=sys.stderr)
            return creds
    raise SystemExit(
        "未找到 SSH 凭据。请任选其一：\n"
        f"  1) 设置环境变量 INSTINCT_SSH_HOST/PORT/USER/PASSWORD\n"
        f"  2) 保留 {AMP_SSH_PATH}\n"
        f"  3) 创建 {CONFIG_PATH}（HOST=.. PORT=.. USER=.. PASSWORD=..）"
    )


_CREDS: dict | None = None


def creds() -> dict:
    global _CREDS
    if _CREDS is None:
        _CREDS = resolve_credentials()
    return _CREDS


def _ssh_opts() -> list[str]:
    c = creds()
    return [
        "-p", c["PORT"],
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=30",
        f"{c['USER']}@{c['HOST']}",
    ]


def _scp_opts() -> list[str]:
    c = creds()
    return [
        "-P", c["PORT"],
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
    ]


# ---------------------------------------------------------------------------
# 核心操作（依赖 pexpect 做密码交互）
# ---------------------------------------------------------------------------

def _pexpect():
    try:
        import pexpect  # type: ignore
    except ImportError:
        raise SystemExit(
            "需要 pexpect 做密码交互： pip install pexpect\n"
            "（本脚本可在本地 base/instinct 环境中运行，不依赖 Isaac Sim）"
        ) from None
    return pexpect


def ssh_run(command: str, timeout: int = 3600, quiet: bool = False) -> int:
    """执行远程命令并回显。"""
    pexpect = _pexpect()
    child = pexpect.spawn("ssh", _ssh_opts() + [command], encoding="utf-8", timeout=timeout)
    if not quiet:
        child.logfile_read = sys.stdout
    if child.expect(["[Pp]assword:", pexpect.EOF, pexpect.TIMEOUT]) == 0:
        child.sendline(creds()["PASSWORD"])
        child.expect(pexpect.EOF, timeout=timeout)
    child.close()
    sys.stdout.flush()
    return child.exitstatus or 0


def ssh_run_capture(command: str, timeout: int = 3600) -> tuple[int, str]:
    """执行远程命令并返回 (exit_status, output)。"""
    pexpect = _pexpect()
    child = pexpect.spawn("ssh", _ssh_opts() + [command], encoding="utf-8", timeout=timeout)
    if child.expect(["[Pp]assword:", pexpect.EOF, pexpect.TIMEOUT]) == 0:
        child.sendline(creds()["PASSWORD"])
        child.expect(pexpect.EOF, timeout=timeout)
    output = child.before or ""
    child.close()
    return child.exitstatus or 0, output


def _scp(src: str, dst: str, recursive: bool, timeout: int = 3600) -> int:
    pexpect = _pexpect()
    args = ["scp"] + _scp_opts()
    if recursive:
        args.append("-r")
    args += [src, dst]
    child = pexpect.spawn(args[0], args[1:], encoding="utf-8", timeout=timeout)
    child.logfile_read = sys.stdout
    if child.expect(["[Pp]assword:", pexpect.EOF, pexpect.TIMEOUT]) == 0:
        child.sendline(creds()["PASSWORD"])
        child.expect(pexpect.EOF, timeout=timeout)
    child.close()
    sys.stdout.flush()
    return child.exitstatus or 0


def scp_put(local: str, remote: str, recursive: bool = False) -> int:
    c = creds()
    return _scp(local, f"{c['USER']}@{c['HOST']}:{remote}", recursive)


def scp_get(remote: str, local: str, recursive: bool = False) -> int:
    c = creds()
    return _scp(f"{c['USER']}@{c['HOST']}:{remote}", local, recursive)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 0

    first = argv[0]
    if first in ("-h", "--help"):
        print(__doc__)
        return 0

    if first == "--put":
        if len(argv) < 3:
            raise SystemExit("用法: --put <local> <remote> [-r]")
        return scp_put(argv[1], argv[2], recursive="-r" in argv)
    if first == "--get":
        if len(argv) < 3:
            raise SystemExit("用法: --get <remote> <local> [-r]")
        return scp_get(argv[1], argv[2], recursive="-r" in argv)
    if first == "--capture":
        if len(argv) < 2:
            raise SystemExit("用法: --capture \"<command>\"")
        status, out = ssh_run_capture(" ".join(argv[1:]))
        sys.stdout.write(out)
        return status

    return ssh_run(" ".join(argv))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
