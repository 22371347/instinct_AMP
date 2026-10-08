#!/usr/bin/env python3
"""诊断：`amp_policy` 的 history 缓冲**到底有没有被推进**（直接读内部状态，不做推断）。

背景
----
`cloud_probe_amp_frames.py` 实测到：2 帧输入的两半**逐位相等**（|Δ| 全零），
且滑窗性质 `frame0@t+1 == frameLast@t` 不成立 ⇒ 说明我们读到的可能是"同一帧的复制"。

两条可能的机制，必须区分：
(A) 缓冲的 `_buffer is None` 一直为真 ⇒ `compute_group` 每次走
    `elif circular_buffer._buffer is None:` 分支 ⇒ 现造一个临时缓冲 append 一次
    ⇒ **所有帧槽位都是当前值**（复制）⇒ 判别器实际只看到 1 帧；
(B) 缓冲正常 append，但由于"同一控制步内 compute 被调用了两次（都带
    update_history=True）"⇒ h=2 时两个槽位都是同一 step 的值（也是复制）。

本脚本逐步打印每个 term 缓冲的
  * `_buffer is None` / `_pointer` / `_num_pushes`（是否在累加）
  * `buffer[0]`（老→新）与**直接算出来的当前值**的差（槽位 == 当前值 ⇒ 复制）
  * 槽位之间（老帧 vs 新帧）的差
并额外打印 env 的类继承链与 recorder 是否开启（recorder 会多调一次 compute）。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--task", default="Instinct-Amp-Walk-V16-G1-v0")
ap.add_argument("--num_envs", type=int, default=4)
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--group", default="amp_policy")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import instinctlab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=cfg).unwrapped
om = env.observation_manager

print("=" * 92)
print(f"TASK = {args.task}")
print(f"env 类继承链 = {[c.__name__ for c in type(env).__mro__[:6]]}")
print(f"env.step 来自   = {type(env).step.__qualname__}")
print(f"recorder 活跃项 = {len(env.recorder_manager.active_terms)}  (>0 会在 step 里多调一次 compute())")
print(f"step_dt = {env.step_dt}   decimation = {env.cfg.decimation}")
print(f"active_terms[{args.group}] = {list(om.active_terms[args.group])}")
print(f"group_obs_term_dim[{args.group}] = {list(om.group_obs_term_dim[args.group])}")
print(f"concatenate_terms[{args.group}] = {om._group_obs_concatenate[args.group]}")

bufs = om._group_obs_term_history_buffer[args.group]
term_cfgs = dict(zip(om.active_terms[args.group], om._group_obs_term_cfgs[args.group]))


def snapshot(tag: str):
    for name, cb in bufs.items():
        buf = cb.buffer  # (N, h, f) 老→新
        cur = term_cfgs[name].func(env, **term_cfgs[name].params).detach()  # 当前值（不碰缓冲）
        cur = cur.reshape(cur.shape[0], -1)
        h = buf.shape[1]
        f = cur.shape[1]
        # 每帧与"当前值"的差（越小 ⇒ 该槽位就是当前步的值）
        d_last = (buf[:, -1].reshape(-1, f) - cur).abs().max().item() if h else float("nan")
        d_first = (buf[:, 0].reshape(-1, f) - cur).abs().max().item() if h else float("nan")
        d_int = (buf[:, -1].reshape(-1, f) - buf[:, 0].reshape(-1, f)).abs().max().item()
        print(f"  [{tag}] {name:<18} h={h} _buffer_is_none={cb._buffer is None!s:<5} "
              f"pointer={cb._pointer:<3} num_pushes={cb._num_pushes[:3].tolist()}  "
              f"|末帧−当前|={d_last:.3e} |首帧−当前|={d_first:.3e} |末−首|={d_int:.3e}")


print("\n--- 构造后（reset 之后）、未 step ---")
snapshot("reset")

zeros = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
for i in range(args.steps):
    with torch.no_grad():
        env.step(zeros)
    print(f"\n--- 第 {i + 1} 次 env.step() 之后 ---")
    snapshot(f"s{i + 1}")

print("\n--- 额外手动 compute(update_history=True) 一次（诊断用：看能否修好滑窗）---")
with torch.no_grad():
    pack = om.compute(update_history=True)
g = pack[args.group]
for name, cb in bufs.items():
    cb_ = cb
    print(f"  {name:<18} _buffer_is_none={cb_._buffer is None!s:<5} pointer={cb_._pointer} "
          f"num_pushes={cb_._num_pushes[:3].tolist()}")
snapshot("after-manual")

print("\n=== 诊断结束 ===")
env.close()
simulation_app.close()
