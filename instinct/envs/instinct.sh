#!/usr/bin/env bash
# 激活本地验证环境
#
#   source envs/instinct.sh
#
# 注意：本地 env 只用于「轻量验证」，不含 Isaac Sim。
# 4 GB 显存无法运行 Isaac Sim —— 仿真/训练/越障验证全部在云端进行。

_INSTINCT_ENV_NAME="instinct"
_CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

if [ ! -f "$_CONDA_SH" ]; then
    echo "❌ 找不到 conda: $_CONDA_SH" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck source=/dev/null
source "$_CONDA_SH"

if ! conda env list | grep -q "^${_INSTINCT_ENV_NAME} "; then
    echo "❌ conda 环境 '${_INSTINCT_ENV_NAME}' 不存在。创建方式：" >&2
    echo "   conda create -y -n ${_INSTINCT_ENV_NAME} python=3.11" >&2
    echo "   conda activate ${_INSTINCT_ENV_NAME}" >&2
    echo "   pip install mujoco onnxruntime numpy matplotlib pyyaml trimesh scipy tqdm imageio imageio-ffmpeg onnx pexpect" >&2
    echo "   pip install torch --index-url https://download.pytorch.org/whl/cpu" >&2
    return 1 2>/dev/null || exit 1
fi

conda activate "$_INSTINCT_ENV_NAME" || {
    echo "❌ 激活失败" >&2
    return 1 2>/dev/null || exit 1
}

# 切到工程根目录，方便直接跑 tools/*.py
_INSTINCT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$_INSTINCT_ROOT" || return 1 2>/dev/null || exit 1

echo "✅ conda env: $_INSTINCT_ENV_NAME   ($(python --version 2>&1))"
echo "✅ 工作目录 : $_INSTINCT_ROOT"
echo
echo "常用命令："
echo "  python tools/check_env.py       # 环境自检"
echo "  python tools/smoke_test.py      # 物理合理性冒烟测试"
echo "  python tools/build_g1_mjcf.py   # 生成可回放 MJCF"
echo "  python tools/g1_mujoco_play.py --dump-model"
echo "  python tools/g1_mujoco_play.py --policy <onnx> --no-view --duration 10"
echo "  python tools/instinct_ssh.py \"nvidia-smi\"   # 云端运维"
