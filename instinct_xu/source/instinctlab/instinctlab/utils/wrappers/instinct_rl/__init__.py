from .module_cfg import *
from .rl_cfg import *
from .vecenv_wrapper import InstinctRlVecEnvWrapper

# 项目自定义的 instinct_rl 算法子类（见 algorithms.py 的模块 docstring）。
# 放在这里是为了让 `class_name="instinctlab.utils.wrappers.instinct_rl:WasabiPPOAmpMix"`
# 可以直接 `importlib.import_module(...)` + `getattr(...)` 解析到。
from .algorithms import WasabiPPOAmpMix
