#!/usr/bin/env python3
"""云端探针 v3：**真的构造环境**，对比 展平版 P1 与 链式旧版 谁炸、炸在哪。

v2 已确认（构造 cfg 后）：
    [flat] motion_reference.prim_path           = '{ENV_REGEX_NS}/Robot/pelvis'
    [flat] motion_reference.reference_prim_path = '/World/envs/env_.*/RobotReference/pelvis'
而报错值 `{ENV_REGEX_NS}/Robot` 正好是
    SensorBase._initialize_impl: self.cfg.prim_path.rsplit("/", 1)[0]
的产物 ⇒ 说明 `format(ENV_REGEX_NS=...)` 没作用到这个 sensor 上。
本脚本用 InstinctRlEnv(cfg) 真建场景（各 2 个 env），看两边是否同样失败。
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args([])
args.headless = True
app = AppLauncher(args).app

import importlib  # noqa: E402

from instinctlab.envs import InstinctRlEnv  # noqa: E402

CASES = [
    ("flat", "instinctlab.tasks.locomotion.config.g1.amp_walk_p1_env_cfg", "G1AmpWalkP1EnvCfg"),
    ("chain", "instinctlab.tasks.locomotion.config.g1.amp_walk_pelvis_env_cfg", "G1AmpWalkPelvisEnvCfg"),
]

ENV_NS = "/World/envs/env_.*"


def _dump(tag, what, scene_cfg):
    for name in ("motion_reference", "robot", "height_scanner"):
        val = getattr(scene_cfg, name, None)
        pp = getattr(val, "prim_path", None) if val is not None else None
        bad = "   <-- 仍是占位符（未替换）" if isinstance(pp, str) and "{ENV_REGEX_NS}" in pp else ""
        print(f"[{tag}] {what:10s} {name:18s} = {pp!r}{bad}")


# ⭐ 直接看 sensor 实例收到了什么 prim_path（决定性证据）
from instinctlab.motion_reference import motion_reference_manager as _mrm  # noqa: E402

_orig_init = _mrm.MotionReferenceManager.__init__
_orig_cfg_init = _mrm.MotionReferenceManagerCfg.__init__ if hasattr(_mrm, "MotionReferenceManagerCfg") else None


def _patched_init(self, cfg):
    print(f"[PATCH] MotionReferenceManager.__init__ 收到 cfg={type(cfg).__name__} "
          f"id={id(cfg)} prim_path={getattr(cfg, 'prim_path', None)!r}")
    return _orig_init(self, cfg)


_mrm.MotionReferenceManager.__init__ = _patched_init

# ⭐⭐ 再看 SensorBase._initialize_impl 里实际用的 cfg.prim_path / cfg 身份
from isaaclab.sensors import SensorBase  # noqa: E402

_orig_sb_init_impl = SensorBase._initialize_impl


def _patched_sb_init_impl(self):
    cfg = getattr(self, "cfg", None)
    pp = getattr(cfg, "prim_path", None)
    flag = "  <<< 未替换!" if isinstance(pp, str) and "{ENV_REGEX_NS}" in pp else ""
    print(f"[PATCH] {type(self).__name__}._initialize_impl: prim_path={pp!r} "
          f"id(cfg)={id(cfg)}{flag}")
    return _orig_sb_init_impl(self)


SensorBase._initialize_impl = _patched_sb_init_impl

# ⭐⭐⭐ 看 self.cfg 到底何时被换（SensorBase.__init__ / _resample_update_period）
_orig_sb_init = SensorBase.__init__


def _patched_sb_init(self, cfg):
    pp = getattr(cfg, "prim_path", None)
    print(f"[PATCH] SensorBase.__init__ 入参 prim_path={pp!r} id={id(cfg)}")
    r = _orig_sb_init(self, cfg)
    print(f"[PATCH] SensorBase.__init__ 后 self.cfg.prim_path="
          f"{getattr(self.cfg, 'prim_path', None)!r} id={id(self.cfg)} same_obj={self.cfg is cfg}")
    return r


SensorBase.__init__ = _patched_sb_init

_orig_resample = _mrm.MotionReferenceManager._resample_update_period


def _patched_resample(self, env_ids=None):
    print(f"[PATCH] _resample_update_period 前: prim_path={self.cfg.prim_path!r} id={id(self.cfg)}")
    r = _orig_resample(self, env_ids)
    print(f"[PATCH] _resample_update_period 后: prim_path={self.cfg.prim_path!r} id={id(self.cfg)}")
    return r


_mrm.MotionReferenceManager._resample_update_period = _patched_resample

for tag, mod_name, cls_name in CASES[:1]:  # 第二个 case 会因 sim context 冲突，另跑
    print("=" * 70)
    cls = getattr(importlib.import_module(mod_name), cls_name)
    cfg = cls()
    cfg.scene.num_envs = 2
    cfg.scene.replicate_physics = False
    mr = cfg.scene.motion_reference
    print(f"[{tag}] cfg.scene.motion_reference: id={id(mr)} prim_path={mr.prim_path!r}")
    _dump(tag, "before", cfg.scene)
    try:
        env = InstinctRlEnv(cfg)
        print(f"[{tag}] OK InstinctRlEnv 构造成功")
        env.close()
    except Exception as e:  # noqa: BLE001
        print(f"[{tag}] FAIL {type(e).__name__}: {e}")
    print(f"[{tag}] -- 真实路径后回读 --")
    print(f"[{tag}] cfg.scene.motion_reference: id={id(mr)} prim_path={mr.prim_path!r}")

app.close()
