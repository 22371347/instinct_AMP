#!/usr/bin/env python3
"""云端 SSH / SCP 助手（InstinctLab 工程版）

设计要点
--------
1. **凭据不入仓库**：密码只存放在仓库**外**、权限 600 的文件里，
   因此不会被 git 收进去（旧的 AMP_mjlab 版把密码硬编码在仓库文件中）。
2. 凭据解析优先级（**从新到旧**）：

   1) 环境变量 `INSTINCT_SSH_HOST/PORT/USER/PASSWORD`
   2) `~/.llp_cloud_credentials`（本项目专用，KEY=VALUE）
   3) `~/.config/instinct/ssh.env`
   4) ⚠️ 旧机器的 `AMP_mjlab/tools/amp_ssh.py`（**会告警**：那台实例的端口与密码
      与当前平台不同，仅在显式设置 `INSTINCT_SSH_ALLOW_AMP_FALLBACK=1` 时启用）

3. 密码**不会**被打印到 stdout。
4. 连接失败会区分「认证失败 / 连接被拒 / 超时」，而不是抛 pexpect 原始异常。

用法
----
    python tools/instinct_ssh.py "nvidia-smi"                 # 执行远程命令（回显）
    python tools/instinct_ssh.py --capture "nvidia-smi"       # 捕获输出（不污染 stdout）
    python tools/instinct_ssh.py --env-check                  # 环境自检（对应 CLOUD_ENV_SETUP.md §3）
    python tools/instinct_ssh.py --shell                      # 交互式 shell
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
#: 本项目（新云平台）专用凭据文件；放在 $HOME 下，不会被 git 收进去
CREDS_PATH = Path(os.path.expanduser(
    os.environ.get("INSTINCT_SSH_CREDS_FILE", "~/.llp_cloud_credentials")
))

_REQUIRED = ("HOST", "PORT", "USER", "PASSWORD")


def _parse_kv_file(path: Path) -> dict | None:
    """解析 KEY=VALUE 文件；同时接受 `LLP_SSH_HOST=..` 与 `HOST=..` 两种写法。"""
    if not path.is_file():
        return None
    raw: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        raw[k.strip().upper()] = v.strip().strip("'\"")

    vals = {}
    for k in _REQUIRED:
        vals[k] = raw.get(k) or raw.get(f"LLP_SSH_{k}") or raw.get(f"INSTINCT_SSH_{k}")
    # 只提示权限问题，不阻断
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        print(f"[credentials] ⚠️ {path} 权限为 {oct(mode)}，建议 chmod 600", file=sys.stderr)
    return vals if all(vals.values()) else None


def _from_env() -> dict | None:
    vals = {k: os.environ.get(f"INSTINCT_SSH_{k}") for k in _REQUIRED}
    return vals if all(vals.values()) else None


def _from_config() -> dict | None:
    return _parse_kv_file(CONFIG_PATH)


def _from_llp_creds() -> dict | None:
    return _parse_kv_file(CREDS_PATH)


def _from_amp_ssh() -> dict | None:
    """旧机器（AMP_mjlab）的凭据——**默认不启用**。

    那台实例的端口/密码与本平台不同，静默回退会连到错的地方，
    因此必须显式设置 INSTINCT_SSH_ALLOW_AMP_FALLBACK=1 才尝试。
    """
    if os.environ.get("INSTINCT_SSH_ALLOW_AMP_FALLBACK") != "1":
        return None
    print(
        f"[credentials] ⚠️ 正在回退到旧机器的 {AMP_SSH_PATH}；"
        "它指向的是**上一台**实例（端口/密码可能已失效）",
        file=sys.stderr,
    )
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
        (f"凭据文件 {CREDS_PATH}", _from_llp_creds),
        (f"配置文件 {CONFIG_PATH}", _from_config),
        (f"旧机器 {AMP_SSH_PATH.name}", _from_amp_ssh),
    ):
        creds = getter()
        if creds:
            print(f"[credentials] 来源: {name}  ({creds['USER']}@{creds['HOST']}:{creds['PORT']})",
                  file=sys.stderr)
            return creds
    raise SystemExit(
        "未找到 SSH 凭据。请任选其一：\n"
        f"  1) 创建 {CREDS_PATH}（chmod 600），内容：\n"
        f"       LLP_SSH_HOST=..\n       LLP_SSH_PORT=..\n"
        f"       LLP_SSH_USER=..\n       LLP_SSH_PASSWORD=..\n"
        f"  2) 设置环境变量 INSTINCT_SSH_HOST/PORT/USER/PASSWORD\n"
        f"  3) 创建 {CONFIG_PATH}"
    )


_CREDS: dict | None = None

#: ⚠️ 关键：Isaac Sim 首次运行时 Kit 会弹 EULA 交互提示，
#: 而 `ssh host "cmd"` **没有 TTY** → 会直接报
#:     "Do you accept the EULA? (Yes/No): Unable to bootstrap inner kit kernel: EOF"
#: 通过环境变量预接受即可。这些变量同时写到 /root/.llp_env.sh（见 --setup-remote）。
OMNI_ENV_EXPORTS = (
    "export OMNI_KIT_ACCEPT_EULA=YES",
    "export ACCEPT_EULA=Y",
    "export PRIVACY_CONSENT=Y",
)

#: 远端环境引导文件（写入这些变量 + conda 激活）
REMOTE_ENV_FILE = "/root/.llp_env.sh"
CONDA_ENV_NAME = "isaaclab"

REMOTE_ENV_SNIPPET = "\n".join(OMNI_ENV_EXPORTS) + f"""
\n
# 激活镜像自带的 Isaac Sim / IsaacLab 环境
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate {CONDA_ENV_NAME}
fi

# ⚠️ 修复镜像的 isaaclab 遮蔽问题
#    镜像把整个 IsaacLab 仓库快照装到了 site-packages/isaaclab/{{apps,source}}，
#    且 site-packages/isaaclab/__init__.py 是个"外壳"，会**遮蔽**真正的库：
#      real: site-packages/isaaclab/source/isaaclab/isaaclab/__init__.py
#      shell: site-packages/isaaclab/__init__.py
#    结果 `import isaaclab` 成功、但 `isaaclab.envs` 报 ModuleNotFoundError。
#    .pth 无法修复（.pth 路径排在 site-packages 之后，外壳仍胜出）；
#    必须用 PYTHONPATH —— 它排在 site-packages **之前**。
_LLP_SP=/opt/conda/envs/{CONDA_ENV_NAME}/lib/python3.11/site-packages
_LLP_SRC="$_LLP_SP/isaaclab/source"
if [ -d "$_LLP_SRC" ]; then
    for _d in "$_LLP_SRC"/*/; do
        _d="${{_d%/}}"; _b=$(basename "$_d")
        [ -d "$_d/$_b" ] && export PYTHONPATH="$_d${{PYTHONPATH:+:$PYTHONPATH}}"
    done
fi
unset _LLP_SP _LLP_SRC _d _b

# 训练日志等大文件放本地数据盘（49G），/ 只有 30G
[ -d /root/gpufree-data ] && export LLP_DATA_ROOT=/root/gpufree-data
"""


def creds() -> dict:
    global _CREDS
    if _CREDS is None:
        _CREDS = resolve_credentials()
    return _CREDS


def _ssh_opts(tty: bool = False) -> list[str]:
    c = creds()
    opts = [
        "-p", c["PORT"],
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=30",
        "-o", "ServerAliveInterval=30",
    ]
    if tty:
        opts.append("-t")
    return opts + [f"{c['USER']}@{c['HOST']}"]


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


def _authenticate(child, pexpect, timeout: int, stream: bool):
    """喂密码，并把连接类错误翻译成人能看懂的提示。

    pexpect 默认抛的是 EOF/TIMEOUT 这类底层异常，不区分“密码错”还是“机器没开”，
    对排查很不利，所以这里显式判别。
    """
    idx = child.expect([
        "[Pp]assword:",
        r"(?i)permission denied",
        r"(?i)connection refused",
        r"(?i)no route to host",
        r"(?i)connection timed out",
        pexpect.EOF,
        pexpect.TIMEOUT,
    ])

    if idx == 0:
        child.sendline(creds()["PASSWORD"])
        j = child.expect(["[Pp]assword:", r"(?i)permission denied", pexpect.EOF, pexpect.TIMEOUT])
        if j == 0:
            child.close(force=True)
            raise SystemExit("[ssh] ❌ 认证失败：密码被拒绝（再次出现 password 提示）")
        if j == 1:
            child.close(force=True)
            raise SystemExit("[ssh] ❌ 认证失败：Permission denied（检查用户名/密码）")
        if j == 3:
            child.close(force=True)
            raise SystemExit(f"[ssh] ⏱️ 密码后超时（{timeout}s）")
    elif idx == 1:
        child.close(force=True)
        raise SystemExit("[ssh] ❌ 认证失败：Permission denied（检查用户名/密码）")
    elif idx == 2:
        child.close(force=True)
        raise SystemExit(f"[ssh] ❌ 连接被拒（{creds()['HOST']}:{creds()['PORT']}）——实例可能未开机")
    elif idx == 3:
        child.close(force=True)
        raise SystemExit("[ssh] ❌ 主机不可达（no route to host）")
    elif idx == 4:
        child.close(force=True)
        raise SystemExit("[ssh] ❌ 连接超时——检查云平台安全组/端口/实例状态")
    elif idx == 6:
        child.close(force=True)
        raise SystemExit(
            f"[ssh] ⏱️ 超时（{timeout}s）：既没等到密码提示也没连上。\n"
            f"       当前目标 {creds()['HOST']}:{creds()['PORT']}，"
            "若刚重建过实例，端口可能变了（重新更新凭据文件）"
        )
    return child


def wrap_env(command: str) -> str:
    """给远端命令加上 Isaac 所需的环境变量（避免 EULA 交互提示卡死）。"""
    prefix = "; ".join(OMNI_ENV_EXPORTS)
    return f"{prefix}; {command}"


def setup_remote_env() -> int:
    """在远端生成 /root/.llp_env.sh，供后续命令 `source` 使用。"""
    payload = REMOTE_ENV_SNIPPET
    cmd = (
        f"cat > {REMOTE_ENV_FILE} <<'LLP_EOF'\n{payload}\nLLP_EOF\n"
        f"chmod 644 {REMOTE_ENV_FILE}\n"
        f"echo '--- {REMOTE_ENV_FILE} ---'\n"
        f"cat {REMOTE_ENV_FILE}\n"
        f"echo '--- 自检 ---'\n"
        f"bash -lc 'source {REMOTE_ENV_FILE}\n"
        f"echo \"  OMNI_KIT_ACCEPT_EULA=$OMNI_KIT_ACCEPT_EULA\"\n"
        f"echo \"  which python=$(which python)\"\n"
        f"echo \"  python=$(python --version 2>&1)\"\n"
        f"echo \"  LLP_DATA_ROOT=$LLP_DATA_ROOT\"'\n"
    )
    return ssh_run(cmd, timeout=300)


def ssh_run(command: str, timeout: int = 3600, quiet: bool = False, tty: bool = False) -> int:
    """执行远程命令并回显。"""
    pexpect = _pexpect()
    child = pexpect.spawn("ssh", _ssh_opts(tty) + [command], encoding="utf-8", timeout=timeout)
    if not quiet:
        child.logfile_read = sys.stdout
    _authenticate(child, pexpect, timeout, not quiet)
    try:
        child.expect(pexpect.EOF, timeout=timeout)
    except pexpect.TIMEOUT:
        print(f"\n[ssh_run] ⏱️ 命令超时（{timeout}s）", file=sys.stderr)
        child.close(force=True)
        return 124
    code = child.exitstatus if child.exitstatus is not None else (
        128 + (child.signalstatus or 0) if child.signalstatus else 255)
    child.close()
    sys.stdout.flush()
    return int(code)


def ssh_run_capture(command: str, timeout: int = 3600) -> tuple[int, str]:
    """执行远程命令并返回 (exit_status, output)。"""
    pexpect = _pexpect()
    child = pexpect.spawn("ssh", _ssh_opts() + [command], encoding="utf-8", timeout=timeout)
    _authenticate(child, pexpect, timeout, False)
    try:
        child.expect(pexpect.EOF, timeout=timeout)
    except pexpect.TIMEOUT:
        out = child.before or ""
        child.close(force=True)
        return 124, out
    output = child.before or ""
    code = child.exitstatus if child.exitstatus is not None else (
        128 + (child.signalstatus or 0) if child.signalstatus else 255)
    child.close()
    return int(code), output


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
# 环境自检（对应 CLOUD_ENV_SETUP.md §3 的前两层）
# ---------------------------------------------------------------------------

ENV_CHECK_SCRIPT = r"""
set +e
echo "===== 1. 系统 ====="
grep -E '^(NAME|VERSION)=' /etc/os-release 2>/dev/null | sed 's/^/  /'
echo "  kernel : $(uname -r)"
echo "  arch   : $(uname -m)"
echo "  GLIBC  : $(ldd --version 2>/dev/null | head -1)"
echo "  CPU    : $(nproc) cores"
free -g 2>/dev/null | awk '/^Mem:/{print "  RAM    : "$2" GB total, "$7" GB avail"}'
echo
echo "===== 2. GPU / 驱动 (Isaac Sim 5.1 需 >= 580.65.06) ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
  if [ "${DRV:-0}" -ge 580 ] 2>/dev/null; then echo "  [OK] 驱动主版本 $DRV >= 580"; else echo "  [!!] 驱动主版本 ${DRV:-?} < 580"; fi
else
  echo "  [!!] 无 nvidia-smi（容器可能没挂 GPU，需 --gpus all）"
fi
echo
echo "===== 3. 磁盘 ====="
df -h / /root /workspace 2>/dev/null | grep -v '^Filesystem' | sort -u | sed 's/^/  /'
echo
echo "===== 4. Python / conda ====="
for p in python3 python; do command -v $p >/dev/null 2>&1 && echo "  $p -> $($p --version 2>&1)  ($(command -v $p))"; done
if command -v conda >/dev/null 2>&1; then
  echo "  conda : $(conda --version 2>&1)"
  echo "  envs  : $(conda env list 2>/dev/null | grep -v '^#' | awk '{print $1}' | tr '\n' ' ')"
else echo "  conda : 未安装"; fi
echo
echo "===== 5. Isaac Sim / IsaacLab ====="
python3 -m pip show isaacsim 2>/dev/null | grep -E '^(Name|Version|Location):' | sed 's/^/  /' || echo "  isaacsim : 未安装"
python3 -m pip show isaaclab 2>/dev/null | grep -E '^(Name|Version|Location|Editable):' | sed 's/^/  /' || echo "  isaaclab : 未安装"
echo "  -- 常见目录 --"
for d in /workspace /workspace/isaaclab /root/IsaacLab /root/isaaclab /opt/IsaacLab /root/IsaacLab-instinct-xu; do
  [ -e "$d" ] && echo "    存在: $d"
done
echo
echo "===== 6. IsaacLab git 提交 (目标 f73c331738) ====="
for d in /workspace/isaaclab /workspace/IsaacLab /root/IsaacLab /opt/IsaacLab /root/IsaacLab-instinct-xu; do
  if [ -d "$d/.git" ]; then
    echo "  $d"
    echo "     HEAD : $(git -C \"$d\" rev-parse HEAD 2>/dev/null)"
    echo "     tag  : $(git -C \"$d\" describe --tags --always 2>/dev/null)   ($(git -C \"$d\" log -1 --format=%cd 2>/dev/null))"
  fi
done
echo
echo "===== 7. torch ====="
python3 -c "import torch;print('  torch',torch.__version__,'| cuda',torch.version.cuda,'| avail',torch.cuda.is_available())" 2>/dev/null || echo "  torch : 未安装或不可导入"
echo
echo "===== 8. 目录一览 ====="
ls -d /workspace/* /root/* 2>/dev/null | head -40 | sed 's/^/  /'
"""


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", help="要执行的远端命令")
    ap.add_argument("--capture", action="store_true", help="捕获输出而非回显")
    ap.add_argument("--env-check", action="store_true", help="环境自检")
    ap.add_argument("--setup-remote", action="store_true",
                    help=f"生成远端环境引导 {REMOTE_ENV_FILE}（EULA + conda 激活）")
    ap.add_argument("--tty", action="store_true", help="分配 TTY（ssh -t）")
    ap.add_argument("--wrap-env", action="store_true",
                    help="自动前缀 Isaac 所需环境变量（避免 EULA 卡死）")
    ap.add_argument("--shell", action="store_true", help="交互式 shell")
    ap.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"))
    ap.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL"))
    ap.add_argument("--recursive", "-r", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    if args.put:
        return scp_put(args.put[0], args.put[1], args.recursive)
    if args.get:
        return scp_get(args.get[0], args.get[1], args.recursive)
    if args.setup_remote:
        return setup_remote_env()
    if args.env_check:
        return ssh_run(ENV_CHECK_SCRIPT, timeout=600)
    if args.shell:
        return ssh_run(f"source {REMOTE_ENV_FILE} 2>/dev/null; bash -l", timeout=args.timeout, tty=True)
    cmd = args.command or "hostname"
    if args.wrap_env:
        cmd = wrap_env(cmd)
    if args.capture:
        code, out = ssh_run_capture(cmd, timeout=args.timeout)
        sys.stdout.write(out)
        return code
    return ssh_run(cmd, timeout=args.timeout, tty=args.tty)


if __name__ == "__main__":
    sys.exit(main())


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
