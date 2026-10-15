#!/usr/bin/env python3
"""**底层对照测试**：同一份观测，云端 `.pt`（torch）与本地 `.onnx`（onnxruntime）输出是否一致。

用户的诉求：不要只看权重是否相同，而是**喂给定观测**、逐步对比两边输出。
本脚本做三件事：

1. 把 ckpt 里的 `model_state_dict['actor.*']` + `policy_normalizer_state_dict` 读出来，
   在本地用 numpy 复现 actor 前向（ELU，隐藏层 (512,256,128)）；
2. 对比 ckpt 的归一化器统计量 与 导出的 `policy_normalizer.npz`；
3. 对以下输入逐项比较 `pt(norm(obs))` 与 `onnx(obs)`：
   · 出生帧（默认站姿 + 平地）
   · 出生帧 + N(0,1) 噪声（多个种子）
   · **真实 rollout 的观测序列**（在本地 MuJoCo 里跑策略，记录它真正看到的 1500 维）

用法：
    python instinct/tools/compare_pt_onnx.py \
        --ckpt /tmp/p1_model_2000.pt \
        --onnx instinct/eval/p1_2000/actor.onnx \
        --norm instinct/eval/p1_2000/policy_normalizer.npz [--rollout-steps 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

OBS_DIM = 1500
PELVIS_Z_DEFAULT = 0.7585
SCAN_OFFSET = 0.5


def elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))


def pt_actor_forward(sd: dict, x: np.ndarray) -> np.ndarray:
    """用 numpy 复现 instinct_rl ActorCritic.actor（ELU，隐藏层 512/256/128）。"""
    h = x
    for i in (0, 2, 4):
        w = sd[f"actor.{i}.weight"].numpy().astype(np.float64)
        b = sd[f"actor.{i}.bias"].numpy().astype(np.float64)
        h = elu(h @ w.T + b)
    w = sd["actor.6.weight"].numpy().astype(np.float64)
    b = sd["actor.6.bias"].numpy().astype(np.float64)
    return h @ w.T + b


def ckpt_normalizer(ckpt: dict) -> "tuple[np.ndarray, np.ndarray, float] | None":
    st = ckpt.get("policy_normalizer_state_dict")
    if st is None:
        return None
    mean = st.get("_mean", st.get("mean"))
    std = st.get("_std", st.get("std"))
    eps = st.get("eps", 1e-2)
    if mean is None or std is None:
        print(f"[norm] ⚠️ ckpt 里的归一化器字段没找到，keys={list(st)}")
        return None
    return (mean.detach().cpu().numpy().reshape(1, -1).astype(np.float64),
            std.detach().cpu().numpy().reshape(1, -1).astype(np.float64),
            float(np.asarray(eps).reshape(-1)[0]))


def birth_obs(cmd=(0.0, 0.0, 0.0)) -> np.ndarray:
    o = np.zeros(OBS_DIM, dtype=np.float64)
    o[0:1116] = PELVIS_Z_DEFAULT - SCAN_OFFSET
    for f in range(4):
        o[1116 + 3 * f: 1119 + 3 * f] = cmd
        o[1128 + 3 * f: 1131 + 3 * f] = (0.0, 0.0, -1.0)
    return o


def rollout_obs(n_steps: int) -> np.ndarray:
    """在本地 MuJoCo 里跑 p1 策略，记录它**真正看到**的原始 1500 维序列。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mujoco

    import g1_mujoco_play as T

    xml = Path(__file__).resolve().parent.parent / "assets/g1_29dof_rev10_play.xml"
    pol_path = Path(__file__).resolve().parent.parent / "eval/p1_2000/actor.onnx"
    norm_path = Path(__file__).resolve().parent.parent / "eval/p1_2000/policy_normalizer.npz"
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    lay = T.AMP_WALK_LAYOUTS["amp1500"]
    T.apply_scan_layout(lay)
    T.set_scan_body("pelvis")
    key = m.key_qpos[0]
    ob = T.AmpWalkObsBuilder(m, d, key[7:7 + 29].copy(), n_joint=29,
                             torso_body_name="pelvis", obs_joint_order="p1", layout=lay)
    pol = T.load_policy(str(pol_path), "auto")
    norm = T.ObsNormalizer(str(norm_path))
    from joint_order import MUJOCO_ORDER, mujoco_state_order_from_model
    st2c = [mujoco_state_order_from_model(m).index(n) for n in MUJOCO_ORDER]
    dflt = key[7:7 + 29][st2c]
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    last = np.zeros(29)
    out = []
    for _ in range(n_steps):
        raw = ob.build(np.zeros(3), last)
        out.append(raw.copy())
        a = np.asarray(pol(norm(raw.astype(np.float32).reshape(1, -1)))).reshape(-1)
        last = a.copy()
        tgt = dflt + a[st2c]
        for _ in range(10):
            d.ctrl[:] = tgt
            mujoco.mj_step(m, d)
    return np.array(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="/tmp/p1_model_2000.pt")
    ap.add_argument("--onnx", default="instinct/eval/p1_2000/actor.onnx")
    ap.add_argument("--norm", default="instinct/eval/p1_2000/policy_normalizer.npz")
    ap.add_argument("--rollout-steps", type=int, default=200,
                    help="本地 rollout 采样的观测帧数（0 = 跳过）")
    ap.add_argument("--seeds", type=int, default=3, help="出生帧+噪声的随机种子个数")
    args = ap.parse_args(argv)

    import onnxruntime as ort
    import torch

    print(f"[1/4] 读 ckpt : {args.ckpt}")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    print(f"      iter = {ck.get('iter')}   actor 层 = "
          + ", ".join(f"{k}:{tuple(sd[k].shape)}" for k in sorted(sd) if k.endswith(".weight")))

    print(f"[2/4] 归一化器一致性：ckpt 内部 vs {args.norm}")
    ck_norm = ckpt_normalizer(ck)
    z = np.load(args.norm, allow_pickle=True)
    mean_np, std_np = z["mean"].reshape(1, -1), z["std"].reshape(1, -1)
    eps_np = float(np.asarray(z["eps"]).reshape(-1)[0]) if "eps" in z.files else 1e-2
    if ck_norm is not None:
        dm = float(np.abs(ck_norm[0] - mean_np).max())
        ds = float(np.abs(ck_norm[1] - std_np).max())
        print(f"      Δmean(max) = {dm:.3e}   Δstd(max) = {ds:.3e}   "
              f"eps: ckpt={ck_norm[2]:g} npz={eps_np:g}"
              + ("   ✅ 一致" if (dm < 1e-6 and ds < 1e-6) else "   ❌ 不一致！"))
    def norm_apply(x, mean, std, eps):
        return ((np.asarray(x, dtype=np.float64).reshape(1, -1) - mean) / (std + eps)).astype(np.float32)

    print(f"[3/4] 载入 ONNX : {args.onnx}")
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    def onnx_forward(x_raw: np.ndarray) -> np.ndarray:
        xn = norm_apply(x_raw, mean_np, mean_np * 0 + std_np, eps_np)
        return np.asarray(sess.run(None, {in_name: xn})[0]).reshape(-1)

    print("[4/4] 逐输入对比（pt 走 ckpt 自己的归一化器；onnx 走导出的 npz）")
    cases: list[tuple[str, np.ndarray]] = []
    b = birth_obs()
    cases.append(("出生帧", b))
    rng = np.random.default_rng(0)
    for i in range(args.seeds):
        cases.append((f"出生帧+N(0,1) 噪声 seed={i}",
                      b + rng.standard_normal(OBS_DIM) * 0.5))
    if args.rollout_steps > 0:
        try:
            obs_seq = rollout_obs(args.rollout_steps)
            print(f"      本地 rollout 采样 {len(obs_seq)} 帧（未归一化的原始观测）")
            cases += [(f"rollout 第 {i} 帧", obs_seq[i]) for i in range(0, len(obs_seq), max(1, len(obs_seq) // 10))]
        except Exception as exc:  # noqa: BLE001
            print(f"      ⚠️ rollout 采样失败（跳过）：{type(exc).__name__}: {exc}")

    worst = 0.0
    print(f"\n{'输入':26s} {'|a_pt|max':>10s} {'|a_onnx|max':>12s} {'Δmax':>10s} {'Δmax/|a|':>10s}")
    for name, x in cases:
        a_pt = pt_actor_forward(sd, norm_apply(
            x, ck_norm[0] if ck_norm else mean_np, ck_norm[1] if ck_norm else std_np,
            ck_norm[2] if ck_norm else eps_np))
        a_onnx = onnx_forward(x)
        d = float(np.abs(a_pt - a_onnx).max())
        rel = d / max(1e-9, float(np.abs(a_pt).max()))
        worst = max(worst, d)
        print(f"{name:26s} {np.abs(a_pt).max():10.4f} {np.abs(a_onnx).max():12.4f} "
              f"{d:10.3e} {rel:10.3e}")
    print(f"\n⇒ 全用例最大绝对差 = {worst:.3e} rad"
          + ("   ✅ 本地 ONNX 与云端 ckpt 在同一观测下**输出一致**（差异在 fp32 噪声级）"
             if worst < 1e-4 else "   ❌ 两边输出不同！问题就在策略/归一化这条链上"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
