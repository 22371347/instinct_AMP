#!/usr/bin/env python3
"""运行时验证 AMP 判别器输入的**帧结构**（2 帧 vs 10 帧）—— 必须实测，不能靠读代码推断。

为什么必须实测
--------------
判别器输入不是"一个历史张量"，而是 IsaacLab 的**逐 term 环形缓冲**：

1. `ObservationManager` 给每个 `history_length > 0` 的 obs term 建 **一个**
   `CircularBuffer`（`isaaclab/utils/buffers/circular_buffer.py`），其 `.buffer`
   属性的语义是「**最老的一帧在 dim=1 的开头，最新的一帧在末尾**」
   ⇒ `history_length=2` 的张量 = `[s_{t-1}, s_t]`（老→新）。
2. 判别器输入是 `vecenv_wrapper._flatten_obs_group()` 把组内各 term
   **按 cfg 顺序** `flatten(start_dim=1)` 再 `cat` 起来的 ⇒ **term-major**：
   每个 term 内部"帧连续"，**不是**"前半段 = 老帧、后半段 = 新帧"。
   例：2 帧 134 维 = `[pg(t-1) 3][pg(t) 3][jpos(t-1) 29][jpos(t) 29][jvel(t-1) 29]
   [jvel(t) 29][blv(t-1) 3][blv(t) 3][bav(t-1) 3][bav(t) 3]`。
3. 历史只在 `compute(update_history=True)` 时**追加**；`ManagerBasedRLEnv.step()`
   里就是这么调的（`manager_based_rl_env.py:238`），而 vecenv wrapper 的
   `compute()`（默认 `update_history=False`）只是**重读**当前缓冲、不会推进历史。
4. 参考侧（`amp_reference`）必须与策略侧**同 Δt**（0.02 s/步）才构成同一个"状态转移"；
   参考游标由 `motion_reference.update_period = 0.02` 推进 ⇒ 应为 **1 帧/步**。

本脚本实测以下断言（跑真 env，不是静态分析）
--------------------------------------------
A. 每个 term 的 `group_obs_term_dim` ⇒ 反推 h 帧（134 = 2×67 / 670 = 5 项 10 帧）。
B. **滑窗性质**：`frame[h-1] @ t  ==  frame[h-2] @ t+1`（逐 env、逐 term）⇒ 证明
   这 h 帧是**连续时间步**，即真的是"状态转移"而不是重复填充。
C. **非退化**：`frame[h-1] - frame[h-2]` 的幅值 / 恰为 0 的比例（若恒 0 ⇒ 帧是复制的，
   判别器根本看不到转移）。
D. 参考侧帧推进：`frame_interval_s` vs `env.step_dt`、`time_passed_from_update` 的
   每步增量 ⇒ 证实 Δframe = 1/步。
E. `--task-b` 可与 `--task` 不同 ⇒ 直接对比 2 帧版与 10 帧版**同一套断言**的结果。

用法（云端，需 GPU）：
    /opt/conda/envs/isaaclab/bin/python /root/cloud_probe_amp_frames.py \
        --task Instinct-Amp-Walk-V15-G1-v0 --num_envs 8 --steps 6
    # 对比 10 帧版（V16 / V11）：
    /opt/conda/envs/isaaclab/bin/python /root/cloud_probe_amp_frames.py \
        --task Instinct-Amp-Walk-V16-G1-v0 --compare-task Instinct-Amp-Walk-V11-G1-v0
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--task", default="Instinct-Amp-Walk-V15-G1-v0", help="主任务（默认 V15 = 2 帧）")
ap.add_argument("--compare-task", default=None, help="可选：再跑一个任务做对照（如 V16/V11 = 10 帧）")
ap.add_argument("--num_envs", type=int, default=8)
ap.add_argument("--steps", type=int, default=18,
                help="推进的环境步数（需 > h+1 才能跳过 reset 的'冻结填充'，默认 18）")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import instinctlab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

#: amp 两组每个 term 的**单帧特征数**（必须与 `AmpPolicyStateObsCfg` 一致）
FEAT_PER_FRAME = {
    "projected_gravity": 3,
    "joint_pos_rel": 29,
    "joint_vel_rel": 29,
    "base_lin_vel": 3,
    "base_ang_vel": 3,
}


def fmt(x: float) -> str:
    return f"{x:.6g}"


def probe(task: str) -> None:
    print("\n" + "=" * 88)
    print(f"### TASK = {task}")
    print("=" * 88)

    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=cfg).unwrapped
    om = env.observation_manager

    print(f"sim.dt            = {env.cfg.sim.dt}")
    print(f"decimation        = {env.cfg.decimation}")
    print(f"env.step_dt       = {env.step_dt}   (= decimation × sim.dt = 控制步长)")

    # --- 参考侧游标（用于 D） -------------------------------------------------
    ref_mgr = None
    for attr in ("motion_reference_manager", "reference_manager", "motion_ref_manager"):
        if hasattr(env, attr):
            ref_mgr = getattr(env, attr)
            print(f"motion reference manager: env.{attr}  ({type(ref_mgr).__name__})")
            break
    if ref_mgr is None:
        cands = [k for k in vars(env) if "refer" in k.lower() or "motion" in k.lower()]
        print(f"⚠️ 未找到 motion reference manager 属性；env 里相关键 = {cands}")
    if ref_mgr is not None:
        try:
            fi = ref_mgr.frame_interval_s
            print(f"frame_interval_s  = {[round(float(v), 6) for v in fi[:4].tolist()]}  (前 4 env)")
        except Exception as exc:  # pragma: no cover
            print(f"  frame_interval_s 读取失败: {exc!r}")
        print(f"cfg.update_period = {ref_mgr.cfg.update_period}")
        print(f"cfg.num_frames    = {ref_mgr.cfg.num_frames}")

    # --- A. 每个 term 的维度 ⇒ 反推帧数 --------------------------------------
    for group in ("amp_policy", "amp_reference"):
        terms = list(om.active_terms[group])
        dims = list(om.group_obs_term_dim[group])
        print(f"\n[{group}] terms={terms}")
        total = 0
        for name, dim in zip(terms, dims):
            size = 1
            for d in dim:
                size *= int(d)
            total += size
            f = FEAT_PER_FRAME.get(name)
            h = (size / f) if f else float("nan")
            print(f"    {name:<20} dim={tuple(int(d) for d in dim)} size={size:<4} "
                  f"feat/帧={f} ⇒ 帧数={h:g}")
        print(f"    ---- 合计 {total} 维")
        h_implied = total / sum(FEAT_PER_FRAME[n] for n in terms)
        print(f"    ⇒ 全局帧数 = {total} / {sum(FEAT_PER_FRAME[n] for n in terms)} = {h_implied:g}")

    # --- 采样：逐步读 amp 两组 -------------------------------------------------
    def read_amp():
        pack = om.compute()  # update_history=False ⇒ 只重读，不推进历史 ✅
        return {g: pack[g] for g in ("amp_policy", "amp_reference")}

    rows = []  # 每步一个 dict：group -> term -> tensor (N, h*feat)
    for _ in range(args.steps):
        with torch.no_grad():
            actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
            env.step(actions)
        rows.append(read_amp())

    N = env.num_envs
    print(f"\n采样 {len(rows)} 步 × {N} env")

    # --- B/C. 滑窗性质 + 非退化 ----------------------------------------------
    for group in ("amp_policy", "amp_reference"):
        terms = list(om.active_terms[group])
        print(f"\n[{group}] 滑窗 & 转移量级")
        for name in terms:
            size = rows[0][group][name].shape[-1]
            f = FEAT_PER_FRAME[name]
            h = size // f
            assert h * f == size, (name, size)
            # 逐帧切片：帧 k 在 [k*f, (k+1)*f)
            def frame(step_idx, k):
                return rows[step_idx][group][name][:, k * f:(k + 1) * f]

            # B: frame[h-1] @ t == frame[h-2] @ t+1  （若 h>=2）
            # ⚠️ reset 的语义：`CircularBuffer.reset()` 把该 env 的整块 buffer 清零，
            #    随后**第一次 append** 会把 `max_length` 个槽位**全部填成同一个值**
            #    （`is_first_push` 分支）⇒ reset 后的前 h-1 步里，输入里的"老帧"
            #    是**重复的冻结值**，滑窗断言在**那段**必然不成立（这是 IsaacLab 的设计，
            #    不是我们的 bug；两边 obs 群一起 reset ⇒ policy/reference 对称）。
            #    ⇒ 只有 `t+1 >= h`（缓冲已填满真实序列）的 pair 才是有效判据。
            if h >= 2:
                steady, prefill = [], []
                for t in range(len(rows) - 1):
                    d = (frame(t + 1, 0) - frame(t, h - 1)).abs().max().item()
                    (steady if (t + 1) >= h else prefill).append(d)
                smax = max(steady) if steady else float("nan")
                pmax = max(prefill) if prefill else float("nan")
                ok = (steady and smax < 1e-6)
                print(f"    {name:<20} h={h:<3} 滑窗(稳态 t+1≥h) max={smax:.3e} ⇒ "
                      f"{'✅ 连续时间步（真·状态转移）' if ok else '❌ 不连续!'}"
                      f"   ｜reset 后 {h - 1} 步内的冻结填充 max={pmax:.3e}")
                print(f"    {'':<20}     稳态 pair 数={len(steady)} / 含冻结 pair 数={len(prefill)}")
            # C: 转移量级（帧间差）
            d = (frame(0, h - 1) - frame(0, 0))
            d_abs = d.abs()
            zero_frac = (d_abs.max(dim=-1).values == 0).float().mean().item()
            print(f"    {'':<20} h={h:<3} |Δ| mean={fmt(d_abs.mean().item()):>10} "
                  f"max={fmt(d_abs.max().item()):>10}  逐env全零比例={zero_frac:.3f} "
                  f"{'✅' if d_abs.mean().item() > 0 else '❌ 帧被复制!'}")

    # --- D. 参考侧帧推进速度 --------------------------------------------------
    if ref_mgr is not None:
        try:
            tp = []
            for _ in range(3):
                with torch.no_grad():
                    actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
                    env.step(actions)
                    tp.append(ref_mgr.time_passed_from_update[:4].detach().clone())
            print("\n[reference cursor] time_passed_from_update（前 4 env，每步两次采样）:")
            for i, t in enumerate(tp):
                print(f"    步 {i}: {[round(float(v), 5) for v in t.tolist()]}")
        except Exception as exc:  # pragma: no cover
            print(f"⚠️ time_passed_from_update 读取失败: {exc!r}")

    env.close()


probe(args.task)
if args.compare_task:
    probe(args.compare_task)

print("\n" + "=" * 88)
print("### 探针结束")
print("=" * 88)

simulation_app.close()
