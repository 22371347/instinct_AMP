#!/usr/bin/env python3
"""抽取 AMP 判别器状态的**逐帧原始序列**，供离线"帧数 h 扫描"研究使用。

为什么这样设计
--------------
要研究"判别器输入该用几帧（h=1/2/.../20）"，需要的是**时间序列**，不是训练时那份
"已经压成 h 帧历史"的张量（那会把 h 写死）。

所以本脚本每步记录**当前一帧**的 67 维状态（5 个 term 按契约顺序拼接、逐项缩放与
训练时一致）：
    [projected_gravity(3) | joint_pos_rel(29) | joint_vel_rel(29)×0.05 |
     base_lin_vel(3) | base_ang_vel(3)]
两侧都记：
  * `ref_frame`  ：来自 `amp_reference`（运动参考）—— **与策略质量无关**，是本研究的主数据源；
  * `pol_frame`  ：来自 `amp_policy`（机器人本体，本轮用零动作跑，仅作"状态定义对照"）。

⚠️ 状态定义取法与训练完全一致：直接调 term 的 `func(env, **params)` **并手动乘
`term_cfg.scale`**（管理器的历史缓冲会做缩放，直接调函数不会）—— 这样避免我去解析
"老→新"的 h 帧历史缓冲。

⚠️ 参考集是"每 env 分配 clip"（权重随机）⇒ 同时记录每步的 clip 标识（能拿到就记，
拿不到则退回记录参考速度供离线按 vx/vy 分档 —— 例如"只取前向 clip"用于与
V11/V16（它们只有 <1 m/s 前向速度）类比）。

用法（云端，需 GPU；建议 16 envs / 2000 步 ≈ 1~2 分钟）：
    bash /root/cloud_run_bg.sh \
      "/opt/conda/envs/isaaclab/bin/python /root/cloud_dump_amp_states.py \
        --task Instinct-Amp-Walk-V18-G1-v0 --num_envs 16 --steps 2000 \
        --out /root/amp_state_dump.npz --device cuda:0 --headless" \
      /root/_dump_wrap.log
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--task", default="Instinct-Amp-Walk-V18-G1-v0")
ap.add_argument("--num_envs", type=int, default=16)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--out", default="/root/amp_state_dump.npz")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import instinctlab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

#: 判别器状态的 **term 顺序与每帧特征数**（契约：与 `AMP_STATE_TERMS` 一致，顺序即契约）
STATE_TERMS = (
    ("projected_gravity", 3),
    ("joint_pos_rel", 29),
    ("joint_vel_rel", 29),
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
)
FRAME_DIM = sum(f for _, f in STATE_TERMS)          # == 67


def current_frame(env, om, group: str) -> torch.Tensor:
    """返回**当前一帧**的 67 维状态（与训练时的状态定义逐位一致）。

    ⚠️ 必须手动乘 `term_cfg.scale`：管理器的历史缓冲会缩放，直接调 `func` 不会。
    """
    terms = list(om.active_terms[group])
    cfgs = dict(zip(terms, om._group_obs_term_cfgs[group]))
    parts = []
    for name, feat in STATE_TERMS:
        assert name in cfgs, f"{group} 缺少 term {name}（观测契约变了？）"
        tc = cfgs[name]
        v = tc.func(env, **tc.params).detach().reshape(env.num_envs, -1)
        assert v.shape[1] == feat, f"{name} 维度 {v.shape[1]} != {feat}"
        if tc.scale is not None:
            v = v * tc.scale
        parts.append(v)
    return torch.cat(parts, dim=-1)                 # (N, 67)


cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=cfg).unwrapped
om = env.observation_manager

# 找 motion reference manager（用于记录 clip 标识；找不到就退回"用参考速度分档"）
ref_mgr = None
for attr in ("motion_reference_manager", "reference_manager", "motion_ref_manager"):
    if hasattr(env, attr):
        ref_mgr = getattr(env, attr)
        break
print(f"[DUMP] env={type(env).__name__} num_envs={env.num_envs} steps={args.steps}")
print(f"[DUMP] 参考管理器 = {type(ref_mgr).__name__ if ref_mgr is not None else '未找到（将用速度分档）'}")
print(f"[DUMP] amp_policy terms = {list(om.active_terms['amp_policy'])}")
print(f"[DUMP] amp_reference terms = {list(om.active_terms['amp_reference'])}")

zeros = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
ref_hist, pol_hist, clip_hist = [], [], []
with torch.no_grad():
    for t in range(args.steps):
        env.step(zeros)
        if t < 5:      # 前几步含 reset 的 prefill，丢弃
            continue
        ref_hist.append(current_frame(env, om, "amp_reference").cpu().numpy().astype(np.float32))
        pol_hist.append(current_frame(env, om, "amp_policy").cpu().numpy().astype(np.float32))
        if ref_mgr is not None:
            try:
                ids = ref_mgr.get_current_motion_identifiers()
                clip_hist.append(np.array([str(x) for x in ids], dtype=object))
            except Exception:
                clip_hist.append(np.array(["?"] * env.num_envs, dtype=object))
        else:
            clip_hist.append(np.array(["?"] * env.num_envs, dtype=object))

ref_arr = np.stack(ref_hist)          # (T, N, 67)
pol_arr = np.stack(pol_hist)          # (T, N, 67)
clip_arr = np.stack(clip_hist)        # (T, N) 字符串

np.savez_compressed(
    args.out,
    ref_frame=ref_arr,
    pol_frame=pol_arr,
    clip_id=clip_arr,
    state_terms=np.array([n for n, _ in STATE_TERMS], dtype=object),
    state_feats=np.array([f for _, f in STATE_TERMS], dtype=np.int32),
    frame_dim=np.int32(FRAME_DIM),
    step_dt=np.float32(env.step_dt),
    task=np.array([args.task], dtype=object),
)

uniq, cnt = np.unique(clip_arr, return_counts=True)
print(f"[DUMP] 保存 → {args.out}  ref={ref_arr.shape} pol={pol_arr.shape}")
print(f"[DUMP] clip 分布: {dict(zip(uniq.tolist(), cnt.tolist()))}")
# 参考速度分档（离线"只取前向 clip"用）：67 维布局 = pg[0:3] jpos[3:32] jvel[32:61] blv[61:64] bav[64:67]
print(f"[DUMP] 参考 base_lin_vel: vx 均值={ref_arr[..., 61].mean():+.3f}  "
      f"vy 均值={ref_arr[..., 62].mean():+.3f}（离线可按此只筛\"前向\"窗口）")

env.close()
simulation_app.close()
