"""**T1n 算法侧（判别器修复臂）**：判别器输入**白化** + **重置判别器**热启动。

诊断（2026-10-09，云端 `params/agent.yaml` + TB 实测）
-----------------------------------------------------
T1l/T1m 的判别器**已经彻底饱和**：

    amp_d（策略侧）        = −0.94
    amp_d_ref（参考侧）    = +0.93
    amp_style_zero_frac    = 0.35 ~ 0.38

而 `quad = clamp(1 − 0.25(d−1)², 0, 1)` **在 `d ≤ −1` 处恰为 0 且 `dr/dd = 0`**
⇒ **35%+ 的样本落在死区**（既不进奖励也不给梯度），余下部分也基本只剩一个常数偏置。
⇒ 这解释了为什么 `discriminator_reward_coef` 从 0.05 抬到 4.0（80×）却毫无塑形效果：
   **放大一个常数不是塑形**。

AMP_mjlab 为什么不会饱和（已核源码）
-----------------------------------
1. `rsl_rl/utils/utils.py:67 class Normalizer(RunningMeanStd)`：
   `clamp((x − mean) / sqrt(var + 1e-4), ±10.0)` —— **真白化 + ±10 夹紧**；
   奖励路径（`modules/discriminator.py:131-133`）与损失路径（`algorithms/amp_ppo.py:438-443`）
   **都**过它。
2. 白化之后 GP(`lambda_=10`) 才真的约束决策面的陡度；我们先前的
   `amp_grad_pen_space="raw"` 作用在 396 维、量纲差 1~2 个数量级的输入上
   **形同虚设** —— 判别器只要在某个大量纲维上走极短一段就能做出 ±1 的分隔。

本臂改动（算法侧，**一个变量：判别器的输入度量空间**）
----------------------------------------------------
    ① `discriminator_kwargs` += `normalizer_class_name="EmpiricalNormalization"`
       + `normalizer_kwargs={"eps": 0.01, "until": None}`
    ② `amp_grad_pen_space` : `"raw"` → **`"normalized"`**
    ③ `fresh_discriminator_on_load` : `False` → **`True`**（丢弃预训练判别器）

其余一律 T1i 口径（λ=0.975 / `discriminator_reward_coef`=4.0 / `bounded_quad` /
`wasserstein` / `disable_style_on_stand` / `clip_min_std`=1e-12 / mirror loss /
`amp_quad_k`=0.25 / GP coef 5.0 —— **本臂不动 GP 系数**）。

⚠️ `amp_normalizer_expert_only` **保持 `True`**（**故意与 mjlab 不同**）
    我们的 `EmpiricalNormalization.forward` 会**顺手 update 统计量**，而 rollout 期
    **每个 env step** 都会拿策略数据前向一次算风格奖励 ⇒ 打开"双端"会让统计量被
    策略分布带偏（`amp_walk_ppo_cfg.py` 里记录的实测漂移 1.51）。
    mjlab 能双端，是因为它在**每个 iteration 内**用
    `vstack(policy_batch + expert_batch)` **批量**更新，而不是逐步更新。

⚠️ 环境侧**零改动**：复用 `G1AmpWalkT1lEnvCfg`。
    - `track_base_height` 已经是 mjlab 值（`std 0.3 / weight 1.5`，云端 `env.yaml` 实测）；
      T1l 的 run 名 `hstd0p3` 就是这个改动。
    - T1m 的 RSI 相位随机化**已撤回**（实测 `amp_style_zero_frac` 0.377→0.35、步态无改善
      —— 它的收益本来就依赖"风格信号还活着"，先把判别器修好再说）。

起点
----
热启动自 **T1a2 的 `model_7000.pt`**（与 T1j/T1k/T1l/T1m 同起点可比）。
⚠️ 但判别器是**从零初始化**的 ⇒ `amp_d` / `Loss/discriminator_loss` /
`amp_style_zero_frac` / `discriminator_reward` **不能与 T1l/T1m 直接比数值**，
要看**绝对水平**是否从"饱和"变成"有分辨力且不卡死区"。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_t1i_ppo_cfg import AMP_T1I_STYLE_LERP, G1AmpWalkT1iPPORunnerCfg

#: ① 判别器白化：与 AMP_mjlab 的 `Normalizer(obs_dim, epsilon=1e-4, clip_obs=10.0)` 对应。
#: ⚠️ 我们的 `EmpiricalNormalization` 是 `(x − mean)/(std + eps)`，**没有 ±10 夹紧**
#:    （差异已记录；`expert_only=True` 下输入不会爆量级，暂时不补）。
AMP_T1N_NORMALIZER = {
    "normalizer_class_name": "EmpiricalNormalization",
    "normalizer_kwargs": {"eps": 0.01, "until": None},
}

#: 其余项的回读守卫（本臂**不动**它们，只防止"顺手改了别的"）。
AMP_T1N_KEEP = {
    "style_lerp": AMP_T1I_STYLE_LERP,          # 0.975
    "discriminator_reward_coef": 4.0,
    "amp_reward_form": "bounded_quad",
    "amp_quad_k": 0.25,
    "discriminator_reward_type": "wasserstein",
    "clip_min_std": 1e-12,
    "entropy_coef": 0.008,
}


@configclass
class G1AmpWalkT1nPPORunnerCfg(G1AmpWalkT1iPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        a = self.algorithm

        # ---- ① 判别器输入白化（保留原有 kwargs，只**增加**两个键）----
        dk = dict(a.discriminator_kwargs)
        dk.update(AMP_T1N_NORMALIZER)
        a.discriminator_kwargs = dk
        # ---- ② GP 必须在白化空间求（否则 GP 项被放大 311×~1.2e6×，支配整个损失）----
        a.amp_grad_pen_space = "normalized"
        # ---- ③ 丢弃预训练判别器（饱和不动点救不回来）----
        a.fresh_discriminator_on_load = True
        self.run_name = "t1n_whiten_freshdisc_from_t1a2"

        # ------------------------------------------------------------------
        # fail-fast 回读断言（烘进配置 ⇒ "不校验直接启动"也安全）
        # ------------------------------------------------------------------
        if str(dk.get("normalizer_class_name") or "") != "EmpiricalNormalization":
            raise ValueError(f"[T1n] 白化未生效：normalizer_class_name={dk.get('normalizer_class_name')!r}")
        if str(a.amp_grad_pen_space) != "normalized":
            raise ValueError(f"[T1n] amp_grad_pen_space 未生效：{a.amp_grad_pen_space!r}")
        if not bool(a.fresh_discriminator_on_load):
            raise ValueError("[T1n] fresh_discriminator_on_load 未生效")
        if not bool(getattr(a, "amp_normalizer_expert_only", False)):
            raise ValueError(
                "[T1n] amp_normalizer_expert_only 必须保持 True —— 我们的 forward 会**逐步**"
                " update 统计量，打开双端会让它被策略分布带偏（mjlab 是每 iteration 批量更新）"
            )
        if bool(getattr(a, "disable_style_on_stand", False)) is not True:
            raise ValueError("[T1n] disable_style_on_stand 意外为 False")
        for _k, _v in AMP_T1N_KEEP.items():
            _got = getattr(a, _k, None)
            if isinstance(_v, float):
                if _got is None or abs(float(_got) - _v) > 1e-12:
                    raise ValueError(f"[T1n] {_k} 被改动：{_got!r}（应 {_v!r}）")
            elif _got != _v:
                raise ValueError(f"[T1n] {_k} 被改动：{_got!r}（应 {_v!r}）")

        print("[T1n][DISCRIMINATOR WHITENING] 判别器输入白化 + 重置判别器：")
        print(f"    discriminator_kwargs = {a.discriminator_kwargs}")
        print(
            f"    amp_grad_pen_space = {a.amp_grad_pen_space!r}"
            f"（GP coef 仍 {a.discriminator_gradient_penalty_coef}，本臂**不动**它）"
        )
        print(
            f"    fresh_discriminator_on_load = {a.fresh_discriminator_on_load}"
            f" ⇒ 热启动**只载 actor/critic**，判别器从零学起"
        )
        print(
            f"    amp_normalizer_expert_only = {a.amp_normalizer_expert_only}"
            f"（统计量仅 expert 侧累积 —— 这点**故意与 mjlab 不同**，理由见文件头）"
        )
        print(
            f"    —— 其余保持 T1i：style_lerp={a.style_lerp}"
            f" coef={a.discriminator_reward_coef} form={a.amp_reward_form!r}"
            f" quad_k={a.amp_quad_k} clip_min_std={a.clip_min_std}"
        )
