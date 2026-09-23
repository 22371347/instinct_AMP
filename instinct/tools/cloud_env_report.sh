#!/usr/bin/env bash
# 云端环境探测 4（精简版）：同名包冲突、磁盘规划、conda 激活、依赖齐备性
set +e
E=/opt/conda/envs/isaaclab

echo "===== A) 是否已有 instinctlab / instinct_rl（同名包冲突风险）====="
$E/bin/python -m pip show instinctlab 2>&1 | grep -E "^(Name|Version|Location|Editable):" | sed 's/^/  /'
$E/bin/python -m pip show instinct_rl  2>&1 | grep -E "^(Name|Version|Location|Editable):" | sed 's/^/  /'
echo "  (以上无内容 = 未安装，直接装 instinct_xu 的两份 source 即可)"

echo
echo "===== B) 已有 editable 安装 ====="
$E/bin/python -m pip list --editable 2>/dev/null | sed 's/^/  /'

echo
echo "===== C) 空间规划 ====="
for d in /root/gpufree-data /root/gpufree-share /; do
  line=$(df -h "$d" 2>/dev/null | awk 'NR==2{printf "%-6s 总 / %-6s 可用 (%s 已用)", $2, $4, $5}')
  printf "  %-24s %s\n" "$d" "$line"
done

echo
echo "===== D) 可写性 ====="
for d in /root/gpufree-data /root/gpufree-share; do
  if touch "$d/.wtest" 2>/dev/null; then echo "  $d 可写 OK"; rm -f "$d/.wtest"; else echo "  $d 不可写 FAIL"; fi
done

echo
echo "===== E) conda 激活方式 ====="
if source /opt/conda/etc/profile.d/conda.sh 2>/dev/null; then
  conda activate isaaclab && {
    echo "  OK: source /opt/conda/etc/profile.d/conda.sh && conda activate isaaclab"
    echo "      which python = $(which python)"
    echo "      version      = $(python --version 2>&1)"
  }
fi

echo
echo "===== F) 关键模块可见性 + torch CUDA（不做完整 Isaac Sim import）====="
$E/bin/python - <<'PY' 2>&1 | sed 's/^/  /'
import importlib.util as u
for m in ["isaacsim", "isaaclab", "isaaclab.envs", "isaaclab.sensors", "torch"]:
    try:
        spec = u.find_spec(m)
        print(f"{'OK ' if spec else 'NO '} {m}")
    except Exception as e:
        print(f"ERR {m}: {type(e).__name__}: {e}")
try:
    import torch
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | cuda_available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"torch ERR: {e}")
PY

echo
echo "===== 探测结束 ====="
