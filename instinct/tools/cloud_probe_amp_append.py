#!/usr/bin/env python3
"""定向诊断：`amp_policy` 的 history 缓冲**每步被 append 几次、帧内容是什么**。

为什么需要它
------------
`cloud_probe_amp_frames.py` 实测（V15，2 帧）：
  * 同一 step 内 `frame0 == frame1`（|Δ| 全零、逐 env 全零比例 1.000）
  * 滑窗性质 `frame0@t+1 == frameLast@t` **不成立**
两种机制都能造成"帧重复"，必须区分：
  (A) 缓冲从未被 append（`_buffer is None` 恒真）⇒ 每次读数都是
      `compute_group` 里 `elif circular_buffer._buffer is None:` 分支现造的临时缓冲
      （只 append 一次 ⇒ **所有槽位 = 当前值**）⇒ 判别器其实只看到 **1 帧**；
  (B) 同一控制步内 `compute(update_history=True)` 被调了 2 次（例如 env.step 里一次 +
      别的路径一次）⇒ h=2 时两槽位同为当前 step 的值（也是重复，但 h=10 会呈
      "两两成对"而非"全相同"）。

区分办法（本脚本）
------------------
1. **数 append 次数**：在 `CircularBuffer.append` 上打计数器（类级 monkeypatch），
   每步打印总次数 ⇒ 每步 +5（5 个 term 各 1 次）还是 +10。
2. 每步打印每个 term 缓冲的 `_buffer is None` / `_pointer` / `_num_pushes`
   （`append` 里对 num_pushes 有 clamp，所以只能作辅助证据）。
3. 每步打印 `projected_gravity` 的 **h 个帧各自的范数**，以及
   `|每一帧 − 当前值|` ⇒ 一眼看出：全等于当前值（复制）还是只有末帧等于当前值（正常滑窗）。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--task", default="Instinct-Amp-Walk-V16-G1-v0")
ap.add_argument("--num_envs", type=int, default=4)
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--group", default="amp_policy")
ap.add_argument("--probe-term", default="projected_gravity")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import instinctlab.tasks  # noqa: F401,E402
from isaaclab.utils.buffers.circular_buffer import CircularBuffer  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# --- 1) append 计数器（类级 monkeypatch，统计所有实例的 append 调用）------------
_APPEND_LOG: list[int] = []
_orig_append = CircularBuffer.append


def _counting_append(self, data):
    _APPEND_LOG.append(self.max_length)
    return _orig_append(self, data)


CircularBuffer.append = _counting_append

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=cfg).unwrapped
om = env.observation_manager

print("=" * 100)
print(f"TASK = {args.task}")
print(f"env.step 来自        = {type(env).step.__qualname__}")
print(f"recorder 活跃项      = {len(env.recorder_manager.active_terms)}")
print(f"step_dt={env.step_dt} decimation={env.cfg.decimation}")
print(f"group_obs_term_dim[{args.group}] = {list(om.group_obs_term_dim[args.group])}")
bufs = om._group_obs_term_history_buffer[args.group]
term_cfgs = dict(zip(om.active_terms[args.group], om._group_obs_term_cfgs[args.group]))
print(f"缓冲里登记的 term    = {list(bufs.keys())}")
print(f"探针 term            = {args.probe_term}")


_PREV_WIN: dict[str, torch.Tensor] = {}   # term -> 上一步的整窗 (N, h, f)
_PREV_PUSH: dict[str, torch.Tensor] = {}   # term -> 上一步的 num_pushes(N)


def report(tag: str):
    n = len(_APPEND_LOG)
    print(f"\n--- {tag} ｜ append 累积次数={n} "
          f"（max_length 分布={ {m: _APPEND_LOG.count(m) for m in set(_APPEND_LOG)} }）---")
    for name, cb in bufs.items():
        tc = term_cfgs[name]
        h = tc.history_length
        cur = tc.func(env, **tc.params).detach().reshape(env.num_envs, -1)
        if cb._buffer is None:                      # 还没 push 过（构造后 / reset 后）
            print(f"  {name:<18} h={h:<3} _buffer_is_none=True   pointer={cb._pointer} "
                  f"num_pushes={cb._num_pushes.tolist()}")
            continue
        buf = cb.buffer                              # (N, h, f) 老→新
        f = cur.shape[1]
        win = buf.reshape(env.num_envs, buf.shape[1], -1) if buf.dim() == 3 else None
        if name in _PREV_WIN and win.shape[1] == _PREV_WIN[name].shape[1]:
            # 正确不变量：**当前窗的第 k 帧 == 上一窗的第 k+1 帧**（窗口整体向"更老"滑动一格）
            # ⚠️ 早先版本误写成 w_t(0) == w_{t-1}(h-1)，那只有在 h=2 时才等价 ⇒ 对 h=10 恒 ❌
            prev = _PREV_WIN[name]
            # ⚠️ 必须 `.amax(dim=-1)` 保留 env 维，否则得到 0 维张量（曾因此 IndexError）
            per_k = [
                (win[:, k] - prev[:, k + 1]).abs().amax(dim=-1)   # (N,)
                for k in range(win.shape[1] - 1)
            ]
            sw = torch.stack(per_k).amax(dim=0)      # (N,) 逐 env
            reset = (cb._num_pushes < _PREV_PUSH[name]).nonzero().flatten().tolist()
            ok_envs = [i for i in range(env.num_envs) if i not in reset]
            sw_ok = max((sw[i].item() for i in ok_envs), default=0.0)
            print(f"  {name:<18} h={h:<3} 滑窗 max_k|w_t(k)−w_{{t-1}}(k+1)| 逐env="
                  f"{[f'{v:.2e}' for v in sw.tolist()]}  本步 reset 的 env={reset}  "
                  f"⇒ 非 reset env 的 max={sw_ok:.2e} {'✅ 连续控制步' if sw_ok < 1e-5 else '❌'}")
        else:
            frames = [win[:, k] for k in range(win.shape[1])]
            d = [f"{((fr - cur).abs().max().item()):.2e}" for fr in frames]
            print(f"  {name:<18} h={h:<3} 首步（缓冲 prefill）|帧k−当前|(k=0..h-1)={d}")
        _PREV_WIN[name] = win.detach().clone()
        _PREV_PUSH[name] = cb._num_pushes.detach().clone()
        # 帧区分度：最老帧 vs 最新帧（全零 ⇒ 整个窗口是同一帧的复制）
        dup = (win[:, -1] - win[:, 0]).abs().amax(dim=-1)
        print(f"  {'':<18}      num_pushes={cb._num_pushes.tolist()}  "
              f"|末帧−首帧| 逐env={[f'{v:.2e}' for v in dup.tolist()]}")


# --- 2) reset + 逐步 step ------------------------------------------------------
with torch.no_grad():
    env.reset()
report("env.reset() 之后")

zeros = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
for i in range(args.steps):
    with torch.no_grad():
        env.step(zeros)
    report(f"第 {i + 1} 次 env.step() 之后")

print("\n=== 诊断结束 ===")
env.close()
simulation_app.close()
