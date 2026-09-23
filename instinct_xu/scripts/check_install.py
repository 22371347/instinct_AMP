"""Check that the active Python environment points to this portable package.

This script does not import instinctlab or start Isaac Sim.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "instinctlab": root / "source" / "instinctlab",
        "instinct_rl": root / "source" / "instinct_rl",
    }

    print(f"Package root: {root}")
    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    for name in ("isaacsim", "isaaclab", "isaaclab_assets", "isaaclab_tasks"):
        try:
            print(f"{name}: {version(name)}")
        except PackageNotFoundError as exc:
            raise RuntimeError(f"Missing required package: {name}") from exc

    for name, source_root in expected.items():
        spec = find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Cannot locate Python package: {name}")
        actual = Path(spec.origin).resolve()
        print(f"{name}: {actual}")
        if not actual.is_relative_to(source_root.resolve()):
            raise RuntimeError(f"{name} points outside this package: {actual}")

    import numpy as np
    import torch

    print(f"NumPy: {np.__version__}")
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA in this environment")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Install path and CUDA check passed. Isaac Sim training still needs a GPU smoke run.")


if __name__ == "__main__":
    main()
