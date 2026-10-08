"""**T1i 算法侧**：把**风格项的绝对强度**对齐 AMP_mjlab。

⚠️ 修正（2026-10-08 晚，由发车前校验输出揭示）
------------------------------------------------
初版 T1i 以为要把 `amp_reward_form` / `discriminator_reward_type` /
`disable_style_on_stand` 对齐 mjlab。**实测这三个字段在全字段 diff 里根本没出现**
⇒ **T1a/T1a2 链的真实值已经是 `bounded_quad` / `wasserstein` / `True`**
（我此前读的是 `amp_walk_ppo_cfg.py` 的**默认值**，而 T1a 链实际继承自 V33 系 cfg）。
⇒ 形状早已对齐；唯一的剩余差异是**绝对强度**，且它只能用 λ 调。

强度对照（形式已相同 ⇒ 直接比系数）
----------------------------------
    mjlab : (1−0.75) · amp_reward_coef · D = 0.25 · 0.1 · D = **0.025·D**
    我们   : (1−λ) · D
    ⇒ 令 (1−λ) = 0.025 ⇒ **λ = 0.975** 两者**完全等强**。
    （T1d/T1e 用 λ=0.75 ⇒ 0.25·D，是 mjlab 的 **10×**；T1c 用 0.9 ⇒ 4×。）

本臂改动（**只有一个变量**）
---------------------------
    `style_lerp` : 1.0（T1a2 = AMP 关闭） → **0.975**（风格强度 = mjlab）
    其余一律 T1a2 口径：`clip_min_std 1e-12`、hidden `[256,128,128]`、
    `entropy_coef 0.008`、`init_noise_std 1.0`、mirror loss 保留。
    ⚠️ 环境侧**零改动**（复用 `G1AmpWalkT1a2EnvCfg`）。
    ⚠️ 起点 = **热启动自 T1a2 的 `model_7000.pt`** ⇒ 与 T1b(0.75)/T1c(0.9) 构成
       **风格强度扫描的第三个点**，同起点可比。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .amp_walk_t1a2_ppo_cfg import G1AmpWalkT1a2PPORunnerCfg

AMP_T1I_STYLE_LERP = 0.975          #: 唯一变量：风格绝对强度 = (1−λ) = 0.025 = mjlab
AMP_T1I_KEEP = {
    "clip_min_std": 1e-12,
    "entropy_coef": 0.008,
    "init_noise_std": 1.0,
}


@configclass
class G1AmpWalkT1iPPORunnerCfg(G1AmpWalkT1a2PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        a, p = self.algorithm, self.policy

        #: ⚠️ 必须**显式**覆盖：T1a2 的父类把 `style_lerp` 设成了 **1.0**
        #: （T1a2 = "AMP 关闭"臂）⇒ 不显式覆盖就仍然是"无风格"。
        a.style_lerp = AMP_T1I_STYLE_LERP
        self.run_name = "t1i_style0p975_from_t1a2"

        # ---- 回读断言（fail-fast，烘进配置 ⇒ "不校验直接启动"也安全）----
        if abs(float(a.style_lerp) - AMP_T1I_STYLE_LERP) > 1e-12:
            raise ValueError(f"[T1i] style_lerp 未生效：{a.style_lerp}")
        #: 形状必须已经是 mjlab 形式（本臂**不动**它们，只做守卫）
        if str(getattr(a, "amp_reward_form", None)) != "bounded_quad":
            raise ValueError(f"[T1i] 形状口径意外：amp_reward_form={a.amp_reward_form!r}")
        if str(getattr(a, "discriminator_reward_type", None)) != "wasserstein":
            raise ValueError(f"[T1i] 形状口径意外：discriminator_reward_type={a.discriminator_reward_type!r}")
        if abs(float(getattr(a, "amp_quad_k", 0.0)) - 0.25) > 1e-12:
            raise ValueError(f"[T1i] amp_quad_k 意外：{a.amp_quad_k}")
        if not bool(getattr(a, "disable_style_on_stand", False)):
            raise ValueError("[T1i] disable_style_on_stand 意外为 False")
        if abs(float(a.clip_min_std) - AMP_T1I_KEEP["clip_min_std"]) > 1e-15:
            raise ValueError(f"[T1i] clip_min_std 被改动：{a.clip_min_std}")
        if list(p.actor_hidden_dims) != [256, 128, 128]:
            raise ValueError(f"[T1i] hidden_dims 被改动：{p.actor_hidden_dims}")
        if abs(float(a.entropy_coef) - AMP_T1I_KEEP["entropy_coef"]) > 1e-12:
            raise ValueError(f"[T1i] entropy_coef 被改动：{a.entropy_coef}")

        print("[T1i][STYLE STRENGTH ALIGN] 风格强度已对齐 AMP_mjlab：")
        print(f"    style_lerp = {a.style_lerp}（风格权重 {1-a.style_lerp:.3f} = mjlab 的 0.25×0.1；"
              f"T1d/T1e 的 0.25 是它的 10×）")
        print(f"    形状（本臂不动，守卫）：amp_reward_form={a.amp_reward_form!r} "
              f"quad_k={getattr(a,'amp_quad_k','?')} type={a.discriminator_reward_type!r} "
              f"no_stand_style={bool(getattr(a,'disable_style_on_stand',False))}")
        print(f"    —— 其余保持 T1a2：clip_min_std={a.clip_min_std} "
              f"hidden={list(p.actor_hidden_dims)} entropy={a.entropy_coef}")
        print(f"    use_mirror_loss = {getattr(a, 'use_mirror_loss', None)}（交付要求②）")
