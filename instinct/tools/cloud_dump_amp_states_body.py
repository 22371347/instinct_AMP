#!/usr/bin/env python3
"""抽取 **V20/刚体空间 195 维** AMP 状态的逐帧原始序列（供离线"帧数 h 扫描"）。

与 `cloud_dump_amp_states.py`（67 维关节空间版）的关系
----------------------------------------------------
那份 dump（`instinct/eval/amp_state_dump.npz`，V18 环境）是**关节空间**的，
本脚本抽**刚体空间**的，两者用**同一套协议**扫 h ⇒ 可以直接回答：

    "刚体表征需要几帧，才能达到关节空间表征在 h=8~10 的水平？"

⚠️ 刻意**不**在同一个 dump 里同时算"关节空间 5 项"：参考侧的
`projected_gravity_reference_as_state` 是 `ManagerTermBase` **类**（不能当普通函数调），
手工重算又有"定义漂移"的风险。刚体 4 项是本环境的 obs 组自带的**普通函数**，
直接调用 ⇒ 逐位等于训练时的定义。姿势分档所需的基座速度另用 3 维单独记录。

数据源
------
* `ref_frame`：(T, N, 195) 来自 **运动参考**（`amp_reference` 的 4 个 term 逐帧拼接）
  —— **与策略质量无关**，是本研究的主数据源；
* `pol_frame`：(T, N, 195) 来自机器人本体（零动作跑，仅作"状态定义对照"）。

195 维布局（term-major，与训练时的 `AMP_V20_STATE_TERMS` 一致）：
    [body_pos_b(39) | body_ori_b(78) | body_lin_vel_b(39) | body_ang_vel_b(39)]
每项内部：**逐刚体连续**（13 体 × (3|6|3|3)），刚体顺序见 `AMP_V20_BODY_NAMES`。

用法（云端，需 GPU；16 envs × 2000 步 ≈ 1~2 分钟）：
    bash /root/cloud_run_bg.sh \
      "/opt/conda/envs/isaaclab/bin/python /root/cloud_dump_amp_states_body.py \
        --task Instinct-Amp-Walk-V20-G1-v0 --num_envs 16 --steps 2000 \
        --out /root/amp_state_dump_body.npz --device cuda:0 --headless" \
      /root/_dump_body_wrap.log
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--task", default="Instinct-Amp-Walk-V20-G1-v0")
ap.add_argument("--num_envs", type=int, default=16)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--out", default="/root/amp_state_dump_body.npz")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import instinctlab.tasks  # noqa: F401,E402

#: 刚体状态的 **term 顺序与每体分量数**（契约 = `AMP_V20_STATE_TERMS` + `AMP_V20_TERM_DIMS`）
TERM_PER_BODY = (
    ("body_pos_b", 3),
    ("body_ori_b", 6),
    ("body_lin_vel_b", 3),
    ("body_ang_vel_b", 3),
)
PER_BODY = sum(f for _, f in TERM_PER_BODY)              # == 15


def current_frame(env, om, group: str, n_bodies: int) -> torch.Tensor:
    """返回**当前一帧**的 (N, n_bodies*15) 刚体状态。

    ⚠️ 必须手动乘 `term_cfg.scale`：管理器的历史缓冲会缩放，直接调 `func` 不会。
    """
    terms = list(om.active_terms[group])
    cfgs = dict(zip(terms, om._group_obs_term_cfgs[group]))
    parts = []
    for name, per_body in TERM_PER_BODY:
        assert name in cfgs, f"{group} 缺少 term {name}（观测契约变了？）"
        tc = cfgs[name]
        v = tc.func(env, **tc.params).detach().reshape(env.num_envs, -1)
        expect = n_bodies * per_body
        assert v.shape[1] == expect, f"{name} 维度 {v.shape[1]} != {expect}"
        if tc.scale is not None:
            v = v * tc.scale
        parts.append(v)
    return torch.cat(parts, dim=-1)


def reference_base_lin_vel_b(env) -> torch.Tensor:
    """参考侧基座线速度（base 系，3 维）—— 仅用于离线按 vx/vy 分档。"""
    mr = env.scene["motion_reference"]
    frame = mr.reference_frame
    return math_utils.quat_apply_inverse(frame.base_quat_w[:, 0], frame.base_lin_vel_w[:, 0])


cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=cfg).unwrapped
om = env.observation_manager

# 刚体数：从任一 term 的 asset_cfg.body_names 反推（并核对 4 项一致）
pol_terms = list(om.active_terms["amp_policy"])
pol_cfgs = dict(zip(pol_terms, om._group_obs_term_cfgs["amp_policy"]))
body_names = tuple(pol_cfgs["body_pos_b"].params["asset_cfg"].body_names)
n_bodies = len(body_names)
frame_dim = n_bodies * PER_BODY
for name, _ in TERM_PER_BODY:
    bn = tuple(pol_cfgs[name].params["asset_cfg"].body_names)
    assert bn == body_names, f"{name} 的 body_names 与 body_pos_b 不一致"
    refc = dict(zip(om.active_terms["amp_reference"], om._group_obs_term_cfgs["amp_reference"]))[name]
    assert tuple(refc.params["asset_cfg"].body_names) == body_names, f"{name} 参考侧体名单不一致"

print(f"[DUMP] env={type(env).__name__} num_envs={env.num_envs} steps={args.steps}")
print(f"[DUMP] 刚体数 = {n_bodies}，单帧维度 = {frame_dim}")
print(f"[DUMP] body_names = {list(body_names)}")
print(f"[DUMP] amp_policy terms   = {pol_terms}")
print(f"[DUMP] amp_reference terms = {list(om.active_terms['amp_reference'])}")

# 参考管理器（记录 clip 标识）
ref_mgr = None
for attr in ("motion_reference_manager", "reference_manager", "motion_ref_manager"):
    if hasattr(env, attr):
        ref_mgr = getattr(env, attr)
        break

zeros = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
ref_hist, pol_hist, clip_hist, blv_hist = [], [], [], []
with torch.no_grad():
    for t in range(args.steps):
        env.step(zeros)
        if t < 5:      # 前几步含 reset 的 prefill，丢弃
            continue
        ref_hist.append(current_frame(env, om, "amp_reference", n_bodies).cpu().numpy().astype(np.float32))
        pol_hist.append(current_frame(env, om, "amp_policy", n_bodies).cpu().numpy().astype(np.float32))
        blv_hist.append(reference_base_lin_vel_b(env).cpu().numpy().astype(np.float32))
        if ref_mgr is not None:
            try:
                ids = ref_mgr.get_current_motion_identifiers()
                clip_hist.append(np.array([str(x) for x in ids], dtype=object))
            except Exception:
                clip_hist.append(np.array(["?"] * env.num_envs, dtype=object))
        else:
            clip_hist.append(np.array(["?"] * env.num_envs, dtype=object))

ref_arr = np.stack(ref_hist)          # (T, N, 195)
pol_arr = np.stack(pol_hist)          # (T, N, 195)
clip_arr = np.stack(clip_hist)        # (T, N)
blv_arr = np.stack(blv_hist)          # (T, N, 3)

np.savez_compressed(
    args.out,
    ref_frame=ref_arr,
    pol_frame=pol_arr,
    ref_base_lin_vel_b=blv_arr,
    clip_id=clip_arr,
    state_terms=np.array([n for n, _ in TERM_PER_BODY], dtype=object),
    term_per_body=np.array([f for _, f in TERM_PER_BODY], dtype=np.int32),
    body_names=np.array(list(body_names), dtype=object),
    frame_dim=np.int32(frame_dim),
    per_body=np.int32(PER_BODY),
    step_dt=np.float32(env.step_dt),
    task=np.array([args.task], dtype=object),
    layout=np.array(["body"], dtype=object),
)

uniq, cnt = np.unique(clip_arr, return_counts=True)
print(f"[DUMP] 保存 → {args.out}  ref={ref_arr.shape} pol={pol_arr.shape}")
print(f"[DUMP] clip 分布: {dict(zip(uniq.tolist(), cnt.tolist()))}")
print(f"[DUMP] 参考 base_lin_vel_b: vx 均值={blv_arr[..., 0].mean():+.3f}  "
      f"vy 均值={blv_arr[..., 1].mean():+.3f}（离线可按此只筛\"前向\"窗口）")
# 各 term 的原始量级（离线定 scale / 归一化 时会用到）
for i, (name, per_body) in enumerate(TERM_PER_BODY):
    lo = sum(f for _, f in TERM_PER_BODY[:i]) * n_bodies
    blk = ref_arr[..., lo : lo + per_body * n_bodies]
    print(f"[DUMP] 参考侧 {name:<16} std={blk.std():.4f} "
          f"p1={np.percentile(blk, 1):+.3f} p99={np.percentile(blk, 99):+.3f}")

env.close()
simulation_app.close()
