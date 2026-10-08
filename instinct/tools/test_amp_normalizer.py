#!/usr/bin/env python3
"""白化（normalizer）改造自检 —— 纯 torch，**不需要** Isaac Sim / GPU / 云端。

被测改动（"新版 AMP 风格方案"，2026-10-03）：
  instinct_xu/source/instinct_rl/instinct_rl/modules/normalizer.py
      → EmpiricalNormalization.normalize(x, update=False)      （只看不改的白化入口）
  instinct_xu/source/instinct_rl/instinct_rl/modules/discriminator.py
      → forward(..., update_normalizer=True) / normalize_input() / forward_normalized()
  instinct_xu/source/instinct_rl/instinct_rl/algorithms/wasabi.py
      → amp_normalizer_expert_only / amp_grad_pen_space
  instinct_xu/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/algorithms.py
      → 两个 kwarg 的声明与透传（AST 静态检查）

断言：
  T0 接线     —— 加载的确实是 `instinct_xu/source/instinct_rl` 那棵树；新 API 存在；默认值 = 旧行为
  T1 白化     —— 白化后常规维 ≈ 零均值/单位方差；近常数维被 eps(σ 下限) 压住而**不**被放大
  T2 专家独占 —— actor 侧前向（update_normalizer=False）不改统计量；漏传则统计量被策略带偏
  T3 GP 空间  —— 同一判别器下 ‖∂d/∂x‖(raw) 远大于 ‖∂d/∂x̃‖(normalized)（放大倍数被 σ 下限封顶
                 ≤ 1/eps）；平方后 raw 空间 GP 项在判别器饱和后完全支配损失 ⇒ 必须用 "normalized"
  T4 真实路径 —— 直接调 WasabiAlgoMixin.compute_discriminator_gradient 两条分支，复现 T3；并确认
                 白化 + "normalized" 的 GP 量级 ≈ 无白化时的旧量级（⇒ coef=1.0 可原样沿用）

用法：
    conda activate instinct && python instinct/tools/test_amp_normalizer.py
"""

from __future__ import annotations

import ast
import inspect
import math
import sys
import types
from pathlib import Path

import torch

# ----------------------------------------------------------------------------
# 0. 让 `import instinct_rl` 指向被改的那棵树（而不是 site-packages / repos 里的旧拷贝）
# ----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
SRC_RL = REPO / "instinct_xu" / "source" / "instinct_rl"
sys.path.insert(0, str(SRC_RL))

import instinct_rl  # noqa: E402
import instinct_rl.modules as M  # noqa: E402
from instinct_rl.algorithms.wasabi import WasabiAlgoMixin  # noqa: E402
from instinct_rl.modules.normalizer import EmpiricalNormalization  # noqa: E402

BRIDGE_PY = (
    REPO
    / "instinct_xu/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/algorithms.py"
)
WASABI_PY = REPO / "instinct_xu/source/instinct_rl/instinct_rl/algorithms/wasabi.py"

# ----------------------------------------------------------------------------
# 1. 合成 AMP 单帧数据：与真实布局同构（4 个观测项 × 13 刚体，共 195 维）
#    真实实测单帧标准差（`amp_scale_sensitivity.py`）：
#        body_pos_b 0.225 / body_ori_b 0.532 / body_lin_vel_b 0.119 / body_ang_vel_b 0.307
#    ⚠️ 真实数据里存在**近常数维**（静止刚体的姿态/零速分量）⇒ 不设 σ 下限时
#       `1/σ → ∞`，这正是"raw 空间 GP 爆量级"的物理来源。
# ----------------------------------------------------------------------------
N_FRAME = 195
BLOCKS = ((0, 39, 0.225), (39, 117, 0.532), (117, 156, 0.119), (156, 195, 0.307))
SEGMENT = {
    "body_pos_b": (39,),
    "body_ori_b": (78,),
    "body_lin_vel_b": (39,),
    "body_ang_vel_b": (39,),
}
N_NEAR_CONST = 8  # 近常数维个数
# ⚠️ 真实数据里这些维的 σ 极小（≈1e-5，例如完全对齐的刚体姿态/零速分量），
#    这正是实测 `raw ‖∇d‖=355~792` vs `normalized ‖∇d‖=0.0049~0.0073`（约 1e5 倍）
#    的来源 ⇒ 合成数据必须用同量级的 σ，否则复现不出"raw 空间爆量级"。
SIGMA_NEAR_CONST = 1e-5
EPS = 0.01  # ⇒ EmpiricalNormalization 的 eps = σ 下限


def make_frames(n: int, seed: int, *, shift: float = 0.0, near_const: bool = True) -> torch.Tensor:
    """造 `n` 帧 AMP 状态。`shift` 模拟"策略偏离专家分布"。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, N_FRAME)
    for a, b, s in BLOCKS:
        x[:, a:b] = torch.randn(n, b - a, generator=g) * s + shift
    if near_const:
        # 放在最后覆盖：不受 shift 影响，始终保持"近常数"（真实数据里的静止分量）
        x[:, :N_NEAR_CONST] = SIGMA_NEAR_CONST * torch.randn(n, N_NEAR_CONST, generator=g)
    return x


def build_disc(normalizer: bool, *, seed: int = 0):
    kw = {}
    if normalizer:
        kw = dict(
            normalizer_class_name="EmpiricalNormalization",
            normalizer_kwargs={"eps": EPS, "until": None},
        )
    torch.manual_seed(seed)
    return M.Discriminator(
        input_segment=SEGMENT,
        output_size=1,
        hidden_sizes=[256, 128],
        **kw,
    )


def disc_mse_loss(d, actor, expert):
    """与 `wasabi.py` 中 `MSELoss` 分支一致的判别器主损失（用于量级比较）。"""
    a = d(actor, update_normalizer=False)
    r = d(expert, update_normalizer=False)
    return 0.5 * (
        torch.nn.MSELoss()(a, -torch.ones_like(a)) + torch.nn.MSELoss()(r, torch.ones_like(r))
    )


def grad_norms(d, x: torch.Tensor, space: str):
    """按 `space` 求 ‖∂d/∂x‖（逐样本 L2），返回 (grad_norm_mean, disc_out)。"""
    if space == "normalized":
        with torch.no_grad():
            x = d.normalize_input(x)
        x = x.detach().requires_grad_(True)
        out = d.forward_normalized(x)
    else:
        x = x.detach().clone().requires_grad_(True)
        out = d(x, update_normalizer=False)
    grad = torch.autograd.grad(out.sum(), x, create_graph=False)[0]
    return grad.norm(2, dim=1).mean(), out


def fake_algo(disc, space: str, expert_only: bool = True):
    """只带 `compute_discriminator_gradient` 所需属性的假算法对象。"""
    obj = WasabiAlgoMixin.__new__(WasabiAlgoMixin)
    obj.discriminator = disc
    obj.amp_grad_pen_space = space
    obj.amp_normalizer_expert_only = expert_only
    obj.discriminator_gradient_torlerance = 0.0
    obj.device = torch.device("cpu")
    return obj


def minibatch(actor, expert):
    return types.SimpleNamespace(actor_states=actor, reference_states=expert, masks=None)


def self_disc_calls(src: str):
    """静态找出所有 `self.discriminator(...)` 调用：[(所在方法, 是否显式给了 update_normalizer, 行号)]。"""
    tree = ast.parse(src)
    out = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for call in [c for c in ast.walk(fn) if isinstance(c, ast.Call)]:
            f = call.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr == "discriminator"
                and isinstance(f.value, ast.Name)
                and f.value.id == "self"
            ):
                has_kw = any(k.arg == "update_normalizer" for k in call.keywords)
                out.append((fn.name, has_kw, call.lineno))
    return out


def train_disc_briefly(d, actor, expert, steps: int = 400, lr: float = 1e-3) -> float:
    """把判别器训到接近饱和（复现 V20 的 `MSE→0.003` 动力学），返回最终 MSE。"""
    opt = torch.optim.Adam(d.parameters(), lr=lr)
    loss = None
    for _ in range(steps):
        loss = disc_mse_loss(d, actor, expert)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.item())


# ----------------------------------------------------------------------------
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  —— {detail}" if detail else ""))


def main() -> int:
    print("=" * 78)
    print("T0 接线检查")
    print("=" * 78)
    print(f"  instinct_rl 源码树: {Path(instinct_rl.__file__).parent}")
    check(
        "加载的是 instinct_xu/source/instinct_rl 那棵树",
        str(Path(instinct_rl.__file__).parent).startswith(str(SRC_RL)),
        str(Path(instinct_rl.__file__).parent),
    )
    check("EmpiricalNormalization.normalize 存在", hasattr(EmpiricalNormalization, "normalize"))
    fwd_params = inspect.signature(M.Discriminator.forward).parameters
    check("Discriminator.forward(update_normalizer=...) 存在", "update_normalizer" in fwd_params)
    check("Discriminator.normalize_input 存在", hasattr(M.Discriminator, "normalize_input"))
    check("Discriminator.forward_normalized 存在", hasattr(M.Discriminator, "forward_normalized"))

    algo_params = inspect.signature(WasabiAlgoMixin.__init__).parameters
    check(
        "wasabi.amp_normalizer_expert_only 默认 True（= 旧行为不回退）",
        algo_params.get("amp_normalizer_expert_only") is not None
        and algo_params["amp_normalizer_expert_only"].default is True,
    )
    check(
        "wasabi.amp_grad_pen_space 默认 'raw'（= 旧行为）",
        algo_params.get("amp_grad_pen_space") is not None
        and algo_params["amp_grad_pen_space"].default == "raw",
    )

    # 桥接层（instinctlab 依赖 isaac ⇒ 不能 import，只能静态检查）
    tree = ast.parse(BRIDGE_PY.read_text())
    bridge_args: set[str] = set()
    bridge_init = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            names = {a.arg for a in node.args.args + node.args.kwonlyargs}
            if {"amp_quad_k"} & names:
                bridge_args |= names
                bridge_init = node
    check("桥接层 __init__ 声明 amp_normalizer_expert_only", "amp_normalizer_expert_only" in bridge_args)
    check("桥接层 __init__ 声明 amp_grad_pen_space", "amp_grad_pen_space" in bridge_args)
    bridge_src = BRIDGE_PY.read_text()
    check(
        "桥接层把两个 kwarg 透传给父类（kwargs[...] = ...）",
        bridge_src.count('kwargs["amp_normalizer_expert_only"]') >= 1
        and bridge_src.count('kwargs["amp_grad_pen_space"]') >= 1,
    )
    # ⚠️ 关键回归检查：`Discriminator.forward` 默认会更新统计量，而 rollout 期的
    #    `compute_auxiliary_reward` 每步都喂**策略数据** ⇒ 漏改一处，白化就被污染。
    calls = self_disc_calls(WASABI_PY.read_text())
    print("  wasabi.py 中 self.discriminator(...) 调用：")
    for name, has_kw, lineno in calls:
        print(f"    · line {lineno:>4}  {name}()  update_normalizer={'✓' if has_kw else '✗'}")
    check(
        "wasabi.py 所有判别器前向都显式指定 update_normalizer",
        all(c[1] for c in calls) and len(calls) >= 3,
        f"{len(calls)} 处调用",
    )
    check(
        "桥接层诊断探针（amp_d_ref）也指定了 update_normalizer=False",
        "update_normalizer=False" in bridge_src,
    )

    print()
    print("=" * 78)
    print("T1 白化正确性：零均值 / 单位方差；近常数维被 σ 下限压住")
    print("=" * 78)
    d = build_disc(True)
    d.train()
    expert = make_frames(16384, seed=1)
    with torch.no_grad():
        d.normalizer.update(expert)
        z = d.normalize_input(expert)
    mu, sd = z.mean(0), z.std(0)
    normal_dims = torch.ones(N_FRAME, dtype=torch.bool)
    normal_dims[:N_NEAR_CONST] = False  # 前 8 维是近常数维
    print(f"  常规维   |mean| max = {mu[normal_dims].abs().max():.3e}   std ∈ "
          f"[{sd[normal_dims].min():.3f}, {sd[normal_dims].max():.3f}]")
    print(f"  近常数维 |mean| max = {mu[~normal_dims].abs().max():.3e}   std ∈ "
          f"[{sd[~normal_dims].min():.3f}, {sd[~normal_dims].max():.3f}]")
    check("常规维 |mean| < 1e-5", mu[normal_dims].abs().max() < 1e-5)
    check(
        "常规维 std ≈ 1/（1+eps·/σ）∈ [0.85, 1.02]（eps 越小越接近 1）",
        0.85 <= sd[normal_dims].min() and sd[normal_dims].max() <= 1.02,
    )
    check(
        "近常数维 std 被 eps 压住（< 0.2，即未被放大到单位方差）",
        sd[~normal_dims].max() < 0.2,
        f"max = {sd[~normal_dims].max():.3f}（无 eps 下限时会 ≈ 1.0，但原始空间的 1/σ 会放大 ~1000×）",
    )
    # 白化是"只看不改"的：再次调用不改变统计量
    mean_before = d.normalizer.mean.clone()
    d.normalize_input(expert)
    check("normalize_input 不更新统计量", torch.equal(mean_before, d.normalizer.mean))

    print()
    print("=" * 78)
    print("T2 专家独占：actor 侧前向不得污染统计量")
    print("=" * 78)
    d = build_disc(True)
    d.train()
    expert2 = make_frames(4096, seed=2)
    actor2 = make_frames(4096, seed=3, shift=3.0)  # 策略跑偏（模拟 V20 的"原地踏步"分布）
    with torch.no_grad():
        d.normalizer.update(expert2)
    mean_before = d.normalizer.mean.clone()
    with torch.no_grad():
        d(actor2, update_normalizer=False)  # ✅ 正确用法：只读
    drift_readonly = (d.normalizer.mean - mean_before).abs().max().item()
    if getattr(d.normalizer, "until", None) is None:  # until=None ⇒ still updating
        d(actor2)  # ❌ 漏传：默认会更新（策略数据污染）
    drift_leak = (d.normalizer.mean - mean_before).abs().max().item()
    print(f"  update_normalizer=False ⇒ 统计量漂移 {drift_readonly:.3e}")
    print(f"  漏传（默认 True）        ⇒ 统计量漂移 {drift_leak:.3f}（被策略分布带偏）")
    check("actor 侧（update_normalizer=False）统计量不变", drift_readonly == 0.0)
    check(
        "漏传时统计量确实会被带偏（说明这个开关是必要的）",
        drift_leak > 0.1,
        f"漂移 {drift_leak:.3f}",
    )

    print()
    print("=" * 78)
    print("T3 GP 空间：raw vs normalized 的 ‖∂d/∂x‖ 量级")
    print("=" * 78)
    d = build_disc(True)
    d.eval()  # 冻结统计量，比较的是同一个判别器
    with torch.no_grad():
        d.normalizer.update(expert2)
    mixed = torch.cat([actor2, expert2], dim=0)
    g_raw, _ = grad_norms(d, mixed, "raw")
    g_nrm, _ = grad_norms(d, mixed, "normalized")
    ratio = g_raw.item() / max(g_nrm.item(), 1e-12)
    gp_raw, gp_nrm = g_raw**2, g_nrm**2  # GP 项 = (‖∇‖²).mean()
    print(f"  raw        ‖∇d‖ = {g_raw:.4e}   （GP 项 = {gp_raw:.4e}）")
    print(f"  normalized ‖∇d‖ = {g_nrm:.4e}   （GP 项 = {gp_nrm:.4e}）")
    print(f"  ‖∇d‖ 比值 = {ratio:.1f}×  ⇒ GP 项（平方）放大 {ratio ** 2:.3g}×")
    print(
        f"  ⚠️ 该比值被 σ 下限封顶：`∂d/∂x = (∂d/∂x̃)/(σ+eps)`，故单维最大放大 = 1/eps = {1 / EPS:.0f}×；\n"
        f"     eps 越小、白化越彻底，raw 空间 GP 越爆（无下限的 raw 空间训练网络实测 ~1e5×）"
    )
    check(
        "raw 空间 ‖∇d‖ 至少是 normalized 的 10 倍（eps=0.01 时理论上限 ≈20×）",
        ratio > 10,
        f"{ratio:.1f}×",
    )
    check(
        "normalized 空间 GP 项落在 v19（无白化 raw）的 5e-3 ~ 1e-2 量级",
        5e-4 < gp_nrm.item() < 5e-2,
        f"{gp_nrm.item():.4e}",
    )

    # 判别器真跑起来后会**饱和**（V20 实测：2000 轮 MSE→0.003），此时 MSE 极小、GP 完全支配判别器损失。
    d_sat = build_disc(True, seed=11)
    d_sat.train()
    with torch.no_grad():
        d_sat.normalizer.update(expert2)
    mse0 = disc_mse_loss(d_sat, actor2, expert2).item()
    mse_sat = train_disc_briefly(d_sat, actor2, expert2)
    mb = minibatch(actor2, expert2)
    gp_raw_sat = WasabiAlgoMixin.compute_discriminator_gradient(fake_algo(d_sat, "raw"), mb).item()
    gp_nrm_sat = WasabiAlgoMixin.compute_discriminator_gradient(fake_algo(d_sat, "normalized"), mb).item()
    # 对照组：**不白化**的判别器，同样训到饱和 ⇒ 用它标定 coef 是否需要改
    d_old_sat = build_disc(False, seed=11)
    d_old_sat.train()
    mse_old_sat = train_disc_briefly(d_old_sat, actor2, expert2)
    gp_old_sat = WasabiAlgoMixin.compute_discriminator_gradient(fake_algo(d_old_sat, "raw"), mb).item()
    gp_nrm_old = WasabiAlgoMixin.compute_discriminator_gradient(fake_algo(build_disc(False, seed=11), "raw"), mb).item()
    print()
    print(f"  判别器饱和前后（同样数据）：MSE {mse0:.4e} → {mse_sat:.4e}")
    print(f"    饱和后 raw 空间 GP 项        = {gp_raw_sat:.4e}  ⇒ 是 MSE 的 {gp_raw_sat / max(mse_sat, 1e-12):.3g}×（完全支配损失）")
    print(f"    饱和后 normalized 空间 GP 项 = {gp_nrm_sat:.4e}  ⇒ 是 MSE 的 {gp_nrm_sat / max(mse_sat, 1e-12):.3g}×")
    print(f"  【对照组】无白化判别器同样训 400 步：MSE {mse_old_sat:.4e}，GP = {gp_old_sat:.4e}")
    print(f"  ⚠️ 白化让\"饱和后的输入空间斜率\"变大 ⇒ coef 可能需要下调（见下）")
    check("判别器确实会饱和（MSE 降到 < 0.05）", mse_sat < 0.05, f"{mse_sat:.4e}")
    check(
        "饱和后 raw 空间 GP 项 ≫ MSE ⇒ 必须用 \"normalized\"",
        gp_raw_sat > 10 * mse_sat,
        f"{gp_raw_sat / max(mse_sat, 1e-12):.3g}×",
    )
    check(
        "饱和后 normalized 空间 GP 项仍 ≪ raw 空间 GP 项（≥10×）",
        gp_raw_sat > 10 * gp_nrm_sat,
        f"ratio = {gp_raw_sat / max(gp_nrm_sat, 1e-12):.3g}×",
    )
    coef_ratio = gp_nrm_sat / max(gp_old_sat, 1e-12)
    print(f"  饱和后 normalized GP / 无白化对照组的 GP = {coef_ratio:.2f}×  ⇒ coef=1.0 是否沿用看这个数（建议 0.3~3× 内沿用）")
    check(
        "白化后饱和期的 GP 与无白化旧配置同量级（0.1×~10×）⇒ coef=1.0 可沿用",
        0.1 < coef_ratio < 10.0,
        f"{coef_ratio:.2f}×",
    )

    print()
    print("=" * 78)
    print("T4 真实代码路径：WasabiAlgoMixin.compute_discriminator_gradient 两条分支")
    print("=" * 78)
    d_w = build_disc(True, seed=7)
    d_w.eval()
    with torch.no_grad():
        d_w.normalizer.update(expert2)
    mb = minibatch(actor2, expert2)
    gp_n = WasabiAlgoMixin.compute_discriminator_gradient(fake_algo(d_w, "normalized"), mb)
    gp_r = WasabiAlgoMixin.compute_discriminator_gradient(fake_algo(d_w, "raw"), mb)
    print(f"  amp_grad_pen_space='normalized' ⇒ GP = {gp_n.item():.4e}")
    print(
        f"  amp_grad_pen_space='raw'        ⇒ GP = {gp_r.item():.4e}   "
        f"（GP 项 {gp_r.item() / max(gp_n.item(), 1e-12):.3g}× ⇒ ‖∇d‖ {math.sqrt(gp_r.item() / max(gp_n.item(), 1e-12)):.1f}×）"
    )
    check("真实路径：raw 分支 GP ≫ normalized 分支 GP", gp_r.item() > 20 * gp_n.item())

    # 回归：无白化（v20/v21 的现状）时的旧 GP 量级 —— 白化方案必须与它同量级，才能沿用 coef=1.0
    d_old = build_disc(False, seed=7)
    d_old.eval()
    gp_old = WasabiAlgoMixin.compute_discriminator_gradient(fake_algo(d_old, "raw"), mb)
    gp_old_white = WasabiAlgoMixin.compute_discriminator_gradient(
        fake_algo(build_disc(False, seed=7), "normalized"), mb
    )
    print(f"  无白化（旧配置，coef=1.0 面对的）GP = {gp_old.item():.4e}")
    print(f"  白化 + 'normalized'            GP = {gp_n.item():.4e}")
    order_ratio = gp_n.item() / max(gp_old.item(), 1e-12)
    print(f"  两者比值 = {order_ratio:.3f}×")
    check(
        "白化+'normalized' 的 GP 与旧配置同量级（0.1× ~ 10×）⇒ coef=1.0 原样沿用",
        0.1 < order_ratio < 10.0,
        f"{order_ratio:.3f}×",
    )
    check(
        "无白化时 'normalized' 分支退化为旧行为（无 normalizer ⇒ 原样返回）",
        abs(gp_old_white.item() - gp_old.item()) < 1e-9,
    )

    print()
    print("=" * 78)
    print("T5 V22 配置接线（静态，无需 Isaac）")
    print("=" * 78)
    v22_py = (
        REPO
        / "instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/amp_walk_v22_ppo_cfg.py"
    )
    base_py = (
        REPO
        / "instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/amp_walk_ppo_cfg.py"
    )
    reg_py = (
        REPO
        / "instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py"
    )
    v22_src, base_src, reg_src = v22_py.read_text(), base_py.read_text(), reg_py.read_text()
    check(
        "V22 在 discriminator_kwargs 里接上 EmpiricalNormalization",
        'normalizer_class_name": "EmpiricalNormalization"' in v22_src
        and '"normalizer_kwargs"' in v22_src,
    )
    check("V22 显式设 amp_normalizer_expert_only = True", "amp_normalizer_expert_only = True" in v22_src)
    check(
        'V22 显式设 amp_grad_pen_space = "normalized"',
        'amp_grad_pen_space = "normalized"' in v22_src,
    )
    check("V22 从零训练（ckpt_manipulator = None）", "ckpt_manipulator = None" in v22_src)
    check(
        "算法 configclass 已声明两个新字段（否则 to_dict() 不带上）",
        "amp_normalizer_expert_only = " in base_src and "amp_grad_pen_space = " in base_src,
    )
    check(
        "V22 已注册为 gym 任务，且 env 侧复用 V21（⇒ 单变量）",
        '"Instinct-Amp-Walk-V22-G1-v0"' in reg_src
        and "G1AmpWalkV21EnvCfg" in reg_src
        and "amp_walk_v22_ppo_cfg:G1AmpWalkV22PPORunnerCfg" in reg_src,
    )

    print()
    n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
    print("=" * 78)
    print(f"结果：{len(CHECKS) - n_fail}/{len(CHECKS)} 通过" + ("" if n_fail == 0 else f" —— {n_fail} 项失败"))
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
