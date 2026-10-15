#!/usr/bin/env python3
"""**导出 mjlab 口径的 AMP 专家数据集** + 采一份"活体参考/策略"分布（跑在云端，约 2–3 分钟）。

## 为什么要它

V29 逐维诊断发现：判别器 20 步内就分开，根因是**专家批≈单个点**（4096 个影子机器人近同相位，
参考侧 `σ_ref` 中位数只有 0.003~0.04）。mjlab 的做法是**从整个运动数据集随机抽帧**
（`AMPLoader`），专家批因此覆盖**整个运动流形**。本脚本导出这份数据集，并同时：
  * 导出 `amp_expert_states.pt`：`(N, 390)` 的 `(s_t, s_{t+1})` 对（= mjlab 的输入口径）；
  * 落 `amp_expert_stats.txt`：数据集 / 活体参考 / 活体策略 三者的**逐维 μ/σ**
    ⇒ 本地即可算"换成全数据集抽帧后，判别器还会不会平凡可分"（**零训练成本的预检**）。

## 与训练同口径的保证

参考侧 4 项 = `amp_body_state.py` 里的
`body_{pos,ori,lin_vel,ang_vel}_b_reference_as_state`，定义：
  * `pos_b/ori_b`：`_anchor_relative_pose(link_pos_w/quat_w, anchor=torso_link)`，
    `ori_b` 再取 `_rotation_6d`（**旋转矩阵前两列**）；
  * `lin_vel_b/ang_vel_b`：由 `_vel_in_body_frame(link_quat_w, link_lin_vel_w/ang_vel_w)`
    得到 —— **世界系速度转各刚体自身系**，与训练观测**同一函数**。
    ⚠️ 曾（2026-10-07 前）错用缓冲里的 `link_lin_vel_b`（那是 `d(base 相对位姿)/dt`，
    锚点恒为 0）⇒ 专家集根速度被清零 57×，导致风格项惩罚前进。
本脚本**直接 import 这两个私有函数**（`_anchor_relative_pose` / `_rotation_6d`）对
数据集的 `link_*_b` 张量做同样运算，并用 `manager.find_bodies(AMP_V20_BODY_NAMES, preserve_order=True)`
取 body 顺序（与 obs 组同一套解析）⇒ 与训练**逐位同口径**。
另外会打印"数据集 vs 活体 `obs["amp_reference"]`"的逐维 μ/σ 对照，用于**自校验**。

用法（云端）：
    cd /root/gpufree-data/instinct_xu/source/instinctlab/instinctlab && \
    source /root/.llp_env.sh && unset PYTHONPATH && \
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_export_amp_expert.py \
        --task Instinct-Amp-Walk-V29-G1-v0 --num_envs 256 --live_steps 30

    # ⭐ V33-S：多导 3 维/帧「anchor 投影重力」（390 → **396**）
    AMP_EXPORT_CKPT=/root/amp_expert_states_396.pt \
    AMP_EXPORT_STATS=/root/amp_expert_stats_396.txt \
    AMP_EXPORT_OUT=/root/amp_export_result_396.txt \
    /opt/conda/envs/isaaclab/bin/python -u /root/cloud_export_amp_expert.py \
        --task Instinct-Amp-Walk-V33S-G1-v0 --num_envs 256 --live_steps 30 --with_anchor_gravity

产物：`/root/amp_expert_states[_396].pt`、`/root/amp_expert_stats[_396].txt`、
      日志 `/root/amp_export_result[_396].txt`

⚠️ `--with_anchor_gravity` 必须配 **V33S 的 task**（活体 obs 才是 396）——否则脚本会
    打印维度不匹配的警告，新通道的口径自检会跳过。"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback

OUT = os.environ.get("AMP_EXPORT_OUT", "/root/amp_export_result.txt")
CKPT = os.environ.get("AMP_EXPORT_CKPT", "/root/amp_expert_states.pt")
STATS = os.environ.get("AMP_EXPORT_STATS", "/root/amp_expert_stats.txt")

#: 与 `amp_walk_v20_env_cfg.py` 里的口径一致（13 体 + anchor）——**从 cfg 模块直接导入**，
#: 不靠猜名字（名称对不上会造成 reshape 失败而不是静默错误，但仍以导入为准）
def _load_names():
    from instinctlab.tasks.locomotion.config.g1.amp_walk_v20_env_cfg import (
        AMP_V20_ANCHOR_NAME,
        AMP_V20_BODY_NAMES,
    )

    return list(AMP_V20_BODY_NAMES), AMP_V20_ANCHOR_NAME


def _anchor_from_env_cfg(env_cfg) -> str | None:
    """从 env cfg 的 `amp_policy` / `amp_reference` 观测组里读出**实际使用的 anchor 刚体名**。

    ⚠️ 为什么必须这么做（2026-10-13 踩到的静默错误）：
    换根臂（pelvis 根）把 anchor 从 `torso_link` 改成了 `pelvis` —— 但那**只改了**
    `ObsTermCfg.params["anchor_cfg"]`，模块常量 `AMP_V20_ANCHOR_NAME` 仍是 `torso_link`。
    本脚本原先直接 import 那个常量 ⇒ 导出的专家集按 **torso** 计算，而判别器
    策略侧按 **pelvis** ⇒ 两侧**不同帧**，维度一样、不报错，但判别器比对的是错位的量。
    以 env cfg（= obs 组真正用的东西）为准就不会再有这个偏差。
    """
    import dataclasses

    obs = getattr(env_cfg, "observations", None)
    for grp_name in ("amp_policy", "amp_reference"):
        grp = getattr(obs, grp_name, None)
        if grp is None or not dataclasses.is_dataclass(grp):
            continue
        for f in dataclasses.fields(grp):
            params = getattr(getattr(grp, f.name), "params", None)
            if isinstance(params, dict) and "anchor_cfg" in params:
                names = getattr(params["anchor_cfg"], "body_names", None)
                if names:
                    return str(names[0])
    return None


def stats_block(tag: str, x):
    """打印并返回逐维 μ/σ（x: (B, D)）。"""
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    mu, sd = x.mean(0), x.std(0)
    print(f"  [{tag}] shape={x.shape}  |μ| 中位={np.median(np.abs(mu)):.4g}  "
          f"σ 中位={np.median(sd):.4g}  σ<0.01 的维数={int((sd < 0.01).sum())}/{x.shape[1]}",
          flush=True)
    return mu, sd


def main(log: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Instinct-Amp-Walk-V29-G1-v0")
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--live_steps", type=int, default=30, help="采多少步的活体 amp_policy/amp_reference")
    ap.add_argument(
        "--anchor",
        default=None,
        help="覆盖 anchor 刚体名。默认从 env cfg 的 amp_policy/amp_reference 观测组读"
        "（即 obs 组真正的口径）；读不到才退回模块常量 AMP_V20_ANCHOR_NAME。"
        "⚠️ 换根臂必须让这里等于判别器策略侧的 anchor，否则专家批与策略批不同帧。",
    )
    ap.add_argument(
        "--with_anchor_gravity",
        action="store_true",
        help="⭐ V33-S：多导 3 维/帧「anchor 投影重力」（390 → 396）。与 env 侧新增的 "
        "`anchor_gravity_b` 通道**同一函数**计算（直接 import，不重写公式）。",
    )
    args = ap.parse_args()
    log.append(f"参数：{vars(args)}")

    import gymnasium as gym
    import numpy as np
    import torch

    import instinctlab.tasks  # noqa: F401
    from instinctlab.envs.mdp.observations.amp_body_state import (
        _anchor_gravity_b,
        _anchor_relative_pose,
        _rotation_6d,
        _vel_in_body_frame,
    )
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

    spec = gym.spec(args.task)
    env_cfg = spec.kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = 0

    env = InstinctRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
    raw = env.unwrapped
    mgr = raw.scene["motion_reference"]
    print(f"num_envs={args.num_envs}  manager={type(mgr).__name__}", flush=True)

    # ---------------- 0) 检查 manager 里能找到什么 ----------------
    print("\n[0] manager 自检", flush=True)

    def find_dataset(m):
        """实测：`mgr._all_motion_sequences` 为 None ⇒ 数据集在 `_motion_buffers` 里的 reader 上。"""
        s = getattr(m, "_all_motion_sequences", None)
        if s is not None:
            return s, "mgr._all_motion_sequences", None
        bufs = getattr(m, "motion_buffers", None) or getattr(m, "_motion_buffers", None) or {}
        print(f"  motion buffers = {list(bufs)}", flush=True)
        for name, b in bufs.items():
            print(f"  · buffer[{name}] = {type(b).__name__}", flush=True)
            for cand in ("_all_motion_sequences", "all_motion_sequences",
                         "_motion_sequences", "motion_sequences"):
                sv = getattr(b, cand, None)
                if sv is not None:
                    print(f"      ✅ {cand} = {type(sv).__name__}", flush=True)
                    return sv, f"{name}.{cand}", b
            rel = [a for a in dir(b) if "sequence" in a.lower() or "motion" in a.lower()]
            print(f"      （无 sequence 属性；相关属性：{rel[:14]}）", flush=True)
        return None, "not found", None

    seq, where, buf = find_dataset(mgr)
    print(f"  数据集位置：{where}", flush=True)
    if seq is None:
        print("  ⚠️ 找不到数据集张量", flush=True)
        log.append("FAIL: 找不到数据集属性")
        return 1
    fields = ["link_pos_w", "link_quat_w", "link_pos_b", "link_quat_b",
              "link_lin_vel_b", "link_ang_vel_b", "link_lin_vel_w", "link_ang_vel_w"]
    have = {}
    for f in fields:
        v = getattr(seq, f, None)
        if v is None:
            print(f"    ❌ {f}: 无", flush=True)
            continue
        have[f] = v
        d = getattr(v, "_concatenated_tensor", None)
        bs = getattr(v, "_batch_sizes", None)
        if d is not None:
            print(f"    ✅ {f}: {type(v).__name__} dense={tuple(d.shape)} {d.dtype}"
                  f" clips={bs.tolist() if bs is not None else None}", flush=True)
        else:
            print(f"    ✅ {f}: {type(v).__name__}（无 _concatenated_tensor；属性 "
                  f"{[a for a in dir(v) if not a.startswith('_')][:14]}）", flush=True)
    for a in ("lengths", "motion_lengths", "complete_motion_lengths", "_lengths",
              "num_frames", "num_motions", "batch_shape", "shape"):
        v = getattr(seq, a, None)
        if v is not None:
            try:
                print(f"    · {a} = {tuple(v.shape) if hasattr(v, 'shape') else v}", flush=True)
            except Exception:  # noqa: BLE001
                print(f"    · {a} = {v!r}", flush=True)

    # ---------------- 1) body / anchor 索引（与 obs 组同一套解析）----------------
    BODY_NAMES, ANCHOR_CONST = _load_names()
    # ⭐ anchor 以 **env cfg 实际口径** 为准（CLI > env cfg > 模块常量）
    _anchor_env = _anchor_from_env_cfg(env_cfg)
    ANCHOR = args.anchor or _anchor_env or ANCHOR_CONST
    _src = ("--anchor" if args.anchor else
            ("env cfg 的 amp_* 观测组" if _anchor_env else "模块常量(兜底)"))
    print(f"\n[1] 从 cfg 导入口径：{len(BODY_NAMES)} body，anchor={ANCHOR}（来源：{_src}）", flush=True)
    if _anchor_env and _anchor_env != ANCHOR_CONST:
        print(f"    ⚠️ 模块常量 AMP_V20_ANCHOR_NAME={ANCHOR_CONST} 已被 env cfg 覆盖为 {_anchor_env}"
              "（换根臂的正常情形）", flush=True)
    print(f"    BODY_NAMES={BODY_NAMES}", flush=True)
    body_ids, body_names = mgr.find_bodies(BODY_NAMES, preserve_order=True)
    anchor_ids, anchor_names = mgr.find_bodies([ANCHOR], preserve_order=True)
    print(f"\n[1] body 解析：找到 {len(body_ids)}/{len(BODY_NAMES)} 个", flush=True)
    print(f"    ids={list(body_ids)}", flush=True)
    print(f"    names={list(body_names)}", flush=True)
    print(f"    anchor={anchor_names} ids={list(anchor_ids)}", flush=True)
    if len(body_ids) != len(BODY_NAMES):
        print("  ⚠️ body 数量不符 ⇒ 名称可能需要调整（打印 manager 的 link_of_interests 供核对）", flush=True)
    print(f"    manager.cfg.link_of_interests = {list(getattr(mgr.cfg, 'link_of_interests', []))}",
          flush=True)

    # ---------------- 2) 从数据集张量算 195 维/帧 ----------------
    def to_dense(x):
        """`ConcatBatchTensor` 的稠密张量在 `._concatenated_tensor`；普通 tensor 直接用。"""
        if x is None:
            return None
        d = getattr(x, "_concatenated_tensor", None)
        if d is not None:
            return d
        if isinstance(x, torch.Tensor):
            return x
        try:
            return torch.cat([t for t in x], dim=0)
        except Exception as exc:  # noqa: BLE001
            raise TypeError(f"无法把 {type(x).__name__} 转成 tensor: {exc}") from exc

    def dev(x):
        return x.to(device=torch.device("cuda:0"), dtype=torch.float32)

    # clip 边界（每个 clip 的帧数）—— 用来**只在 clip 内**配对，避免接缝处的伪对
    bs_raw = getattr(have.get("link_pos_w") or have.get("link_pos_b"), "_batch_sizes", None)
    clip_lens = [int(x) for x in bs_raw.tolist()] if bs_raw is not None else None

    if "link_pos_b" in have and "link_quat_b" in have:
        pos_b = dev(to_dense(have["link_pos_b"]))[:, body_ids]
        quat_b = dev(to_dense(have["link_quat_b"]))[:, body_ids]
        ori_b = _rotation_6d(quat_b)
    else:
        pos_w, quat_w = dev(to_dense(have["link_pos_w"])), dev(to_dense(have["link_quat_w"]))
        pos_b, quat_b = _anchor_relative_pose(
            pos_w[:, anchor_ids[0]], quat_w[:, anchor_ids[0]],
            pos_w[:, body_ids], quat_w[:, body_ids],
        )
        ori_b = _rotation_6d(quat_b)
    # ⚠️⚠️ 2026-10-07 修正：**必须与训练观测逐字同口径**。
    #    训练侧 `body_lin_vel_b_reference_as_state` / `body_ang_vel_b_reference_as_state`
    #    的计算是：`_vel_in_body_frame(frame.link_quat_w, frame.**link_lin_vel_w**)`
    #    ⇒ 「**世界系速度** 转 各刚体自身系」。
    #    旧版本用的是缓冲里的 `link_lin_vel_b` / `link_ang_vel_b`，而按
    #    `amass_motion.py` 的定义 `link_lin_vel_b = d(link_pos_b)/dt`，其中
    #    `link_pos_b` 是 FK 出来的 **base 相对**位姿 ⇒ **锚点那条恒为 0**。
    #    实测偏差：专家集 anchor |v| = 0.013 m/s，源数据 0.750 m/s ⇒ **差 57×**。
    #    后果：判别器的「真」样本里根速度被清零 ⇒ 学到「根不动 = 真」⇒ 风格项
    #    **惩罚根位移** ⇒ 一旦风格权重非零，策略就被推向原地不动
    #    （T1b λ=0.75 / T1c λ=0.9 实测崩塌，且与权重无关）。
    quat_w = dev(to_dense(have["link_quat_w"]))
    lin_b = _vel_in_body_frame(
        quat_w[:, body_ids], dev(to_dense(have["link_lin_vel_w"]))[:, body_ids]
    )
    ang_b = _vel_in_body_frame(
        quat_w[:, body_ids], dev(to_dense(have["link_ang_vel_w"]))[:, body_ids]
    )
    F = pos_b.shape[0]
    print(f"\n[2] 数据集：{F} 帧；pos_b{tuple(pos_b.shape)} ori_b{tuple(ori_b.shape)} "
          f"lin_b{tuple(lin_b.shape)} ang_b{tuple(ang_b.shape)}", flush=True)
    print(f"    clip 长度：{clip_lens}", flush=True)

    frame195 = torch.cat(
        [pos_b.reshape(F, -1), ori_b.reshape(F, -1), lin_b.reshape(F, -1), ang_b.reshape(F, -1)],
        dim=-1,
    )
    assert frame195.shape[1] == 195, f"每帧维数应为 195，实际 {frame195.shape[1]}"

    # mjlab 口径：`(s_t, s_{t+1})` ⇒ **帧序 term-major**（老→新），与 obs 组一致。
    # 只在**同一 clip 内**配对（接缝处的 t→t+1 跨clip，丢掉）。
    def pair_indices(total: int, lens):
        if lens is None:
            return torch.arange(0, total - 1, dtype=torch.long)
        idx = []
        s = 0
        for L in lens:
            if L >= 2:
                idx += list(range(s, s + L - 1))
            s += L
        return torch.tensor(idx, dtype=torch.long)

    ii = pair_indices(F, clip_lens).to(pos_b.device)
    jj = ii + 1

    # ⭐ V33-S：唯一新增的通道 —— anchor 刚体**自身系**下的投影重力（3 维/帧）。
    #    与 env 侧的 `anchor_gravity_b` 完全同一函数（`_anchor_gravity_b`，直接 import）。
    #    ⚠️ 必须用 `link_quat_w`（**世界系**）；`link_quat_b` 是相对 base 的，会把绝对姿态丢掉。
    grav = None
    if args.with_anchor_gravity:
        if "link_quat_w" not in have:
            raise RuntimeError("--with_anchor_gravity 需要数据集里的 `link_quat_w`（世界系），但没找到")
        quat_w = dev(to_dense(have["link_quat_w"]))
        q_anchor_w = quat_w[:, anchor_ids[0]]
        grav = _anchor_gravity_b(q_anchor_w)  # (F, 3)
        assert grav.shape == (F, 3), f"投影重力形状应为 (F,3)，实际 {tuple(grav.shape)}"
        print(f"\n[2b] anchor 投影重力通道（{ANCHOR}，自身系）：{tuple(grav.shape)}  "
              f"z 分量均值 {float(grav[:, 2].mean()):+.4f}（近 −1 ⇒ 姿态接近直立）"
              f"  |g_xy| 均值 {float(grav[:, :2].norm(dim=-1).mean()):.4f}（⇒ 倾角中位 "
              f"{float(grav[:, :2].norm(dim=-1).median().clamp(-1, 1).asin().abs() * 180 / 3.14159265):.2f}°）",
              flush=True)

    _extra = [] if grav is None else [grav[ii].reshape(-1, 3), grav[jj].reshape(-1, 3)]
    pairs = torch.cat(
        [
            pos_b[ii].reshape(-1, 39), pos_b[jj].reshape(-1, 39),
            ori_b[ii].reshape(-1, 78), ori_b[jj].reshape(-1, 78),
            lin_b[ii].reshape(-1, 39), lin_b[jj].reshape(-1, 39),
            ang_b[ii].reshape(-1, 39), ang_b[jj].reshape(-1, 39),
        ]
        + _extra,
        dim=-1,
    )
    expect_dim = 396 if grav is not None else 390
    assert pairs.shape[1] == expect_dim, f"对状态维度应为 {expect_dim}，实际 {pairs.shape[1]}"
    print(f"    帧状态 (N,195) = {tuple(frame195.shape)}；对状态 (N,{pairs.shape[1]}) = {tuple(pairs.shape)}"
          f"（clip 内配对，clip 长度 {clip_lens}）", flush=True)

    # ---------------- 3) 采活体参考 / 策略（同样是训练口径）----------------
    print(f"\n[3] 采活体分布（{args.live_steps} 步，零动作）", flush=True)
    live_ref, live_pol = [], []
    n_act = getattr(getattr(raw, "action_manager", None), "total_action_dim", None)
    if n_act is None:
        n_act = int(raw.action_space.shape[-1])
    zero = torch.zeros(args.num_envs, n_act, device=raw.device)
    if hasattr(env, "get_observations"):
        env.get_observations()
    for i in range(args.live_steps):
        _, _, _, infos = env.step(zero)
        obss = infos.get("observations") if isinstance(infos, dict) else None
        if obss is None:
            print("  ⚠️ infos 里没有 observations", flush=True)
            break
        if "amp_reference" in obss:
            live_ref.append(obss["amp_reference"].detach().float().cpu())
        if "amp_policy" in obss:
            live_pol.append(obss["amp_policy"].detach().float().cpu())

    import numpy as _np
    lines = [f"# AMP 逐维分布对照（μ/σ，逐维；D={pairs.shape[1]}）", f"# 数据集帧数 F={F}"]
    print("\n[4] 逐维分布对照（关键：**数据集 σ** 比活体参考 σ 大多少）", flush=True)
    musd = {}
    musd["expert_dataset"] = stats_block("expert_dataset", pairs)
    if live_ref:
        lr = torch.cat(live_ref, 0)
        musd["live_reference"] = stats_block("live_reference", lr)
    if live_pol:
        lp = torch.cat(live_pol, 0)
        musd["live_policy"] = stats_block("live_policy", lp)

    for k, (mu, sd) in musd.items():
        lines.append(f"\n== {k} μ ==")
        lines.append(" ".join(f"{v:.6g}" for v in mu))
        lines.append(f"== {k} σ ==")
        lines.append(" ".join(f"{v:.6g}" for v in sd))

    # ⭐ V33-S 专属：新通道（最后 6 维 = [grav(老), grav(新)]）的**口径自检**
    #   数据集路径（本脚本）与训练时活体 obs 路径（`anchor_gravity_b`）必须“长得一样”。
    if grav is not None and live_ref:
        lr_all = torch.cat(live_ref, 0)
        if lr_all.shape[1] == pairs.shape[1]:
            d_mu = (pairs[:, -6:].mean(0).cpu() - lr_all[:, -6:].mean(0)).abs().max()
            d_sd = (pairs[:, -6:].std(0).cpu() - lr_all[:, -6:].std(0)).abs().max()
            print(f"\n  ⭐ 新通道口径自检（最后 6 维）：|Δμ|max={float(d_mu):.4f}  "
                  f"|Δσ|max={float(d_sd):.4f}"  f"（需同量级；差很大 ⇒ 两条路径口径不一致）", flush=True)
            lines.append(f"\n# anchor_gravity_channel |d_mu_max|={float(d_mu):.6g} |d_sigma_max|={float(d_sd):.6g}")
        else:
            print(f"\n  ⚠️ 活体 obs 维度 {lr_all.shape[1]} != 数据集 {pairs.shape[1]} ⇒ "
                  "跳新通道自检（请确认 --task 用的是带重力通道的 V33S）", flush=True)
    if "expert_dataset" in musd and "live_reference" in musd:
        sd_ds = musd["expert_dataset"][1]
        sd_lr = musd["live_reference"][1]
        r = _np.median(sd_ds / _np.maximum(sd_lr, 1e-12))
        print(f"\n  ⭐ **σ 放大倍数中位（数据集 / 活体参考）= {r:.1f}×**  "
              f"（≫1 ⇒ 换成全数据集抽帧后判别器不再「对单点」）", flush=True)
        lines.append(f"\n# sigma_ratio_median(expert_dataset/live_reference) = {r:.4f}")

    torch.save(
        {
            "states_pair": pairs.cpu(),          # (N, 390) 或 (N, 396) mjlab 口径
            "states_frame": frame195.cpu(),      # (F, 195) 单帧（不含新通道）
            "body_names": list(body_names),
            "anchor": list(anchor_names),
            "layout": "[pos_b(老),pos_b(新),ori_b(老),ori_b(新),lin_vel_b(老),lin_vel_b(新),"
                      "ang_vel_b(老),ang_vel_b(新)]"
                      + (",grav_b(老),grav_b(新)" if grav is not None else ""),
            "with_anchor_gravity": bool(grav is not None),
            "num_frames": F,
        },
        CKPT,
    )
    with open(STATS, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[5] 已写 {CKPT}（{os.path.getsize(CKPT) / 1e6:.1f} MB）与 {STATS}", flush=True)
    log.append(f"OK: frames={F} pairs={pairs.shape[0]}")
    return 0


if __name__ == "__main__":
    _log: list[str] = []
    _fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(_fd, 1)
    os.dup2(_fd, 2)
    print("===== 导出 AMP 专家数据集（mjlab 口径）=====", flush=True)

    from isaaclab.app import AppLauncher

    rc = 1
    _err = None
    try:
        _app = AppLauncher({"headless": True, "livestream": -1, "enable_cameras": False}).app
        rc = main(_log)
    except BaseException:  # noqa: BLE001
        _err = traceback.format_exc()
        print("抛异常：\n" + _err, flush=True)
    if _err is not None:
        rc = 1
    print(f"RC={rc}", flush=True)
    print(f"[AMP-EXPORT {'PASS' if rc == 0 else 'FAIL'}]", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # ⚠️ 不调 `_app.close()`（在这个环境里会挂住不返回）⇒ 直接退出，OS 回收
    os._exit(rc)
