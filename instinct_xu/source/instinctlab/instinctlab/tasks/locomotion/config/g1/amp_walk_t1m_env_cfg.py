"""**T1m**：在 T1l 基础上**只改一个参数** —— **打开 RSI 的随机起点**（相位随机化）。

用户决策（2026-10-09）
----------------------
> "mjlab 确实是从把所有 clip 的所有帧拼成一个大池子、从任意 clip、任意步态相位开始的，
>  这可能是其成功的核心设计……我们可能需要尝试该方案"

## 相对 **T1l** 的差异（**只有 1 个参数**）

    `scene.motion_reference.motion_buffers[*].motion_start_from_middle_range`
        **(0.0, 0.0)  →  (0.0, 1.0)**

其余一切（奖励 17 项、算法侧、指令分布、参考集、判别器）**逐字不动**。

## 为什么改它（与 mjlab 的确切差异）

两边**都有 RSI**（actor 的 reset state 都来自参考帧）—— ⚠️ 此前"我们完全没有 motion reset"
的说法**是错的**，已更正。真正的差异是**抽哪一帧**：

| | mjlab | 我们（T1l 及之前）|
|---|---|---|
| 池子 | **所有 clip 所有帧 `_concat_frames` 拼成一个大池** | 无池 |
| 抽样 | `torch.randint(0, total_frames)` **均匀抽帧** | 随机选 clip，**但起点固定 = 第 0 帧** |
| 结果 | **任意 clip、任意步态相位** | 每条 clip 的**开头** |

mjlab 侧源码：`AMP_mjlab/src/tasks/amp_loco/mdp/events.py::_write_reset_state`
（`total_frames = frames["root_pos"].shape[0]; idx = torch.randint(0, total_frames, ...)`）
与 `MotionResetManager.init`（`self.walk_run_frames[motion_dir] = self._concat_frames(...)`）。

**为什么这可能是"核心设计"**：
* mjlab 的策略**随时被扔回运动的任意相位** ⇒ actor 侧状态分布**始终与参考分布重叠**
  ⇒ 判别器永远有区分空间 ⇒ **风格奖励不会塌成常数**（"AMP tracking 强"）；
* 我们总从"第 0 帧"起 ⇒ 策略容易学成"从头的固定套路"（实测小跳模态）
  ⇒ actor 分布与参考分布逐渐分离（联合 gap ≈ **37σ**）⇒ 判别器一个迭代内就分开
  （`disc_loss 0.42→0.035`）⇒ **AMP 事实上失效**。

## ⚠️ 本文件**推翻**了一条旧结论

`amp_walk_env_cfg.py` 里原本写着：

> `motion_start_from_middle_range = (0.0, 0.0)`：**关闭 RSI 随机起点**。
> **随机起点会破坏步态相位连续性，对循环行走有害**（官方 Vault 的 PLAY 配置也正是这么关掉的）。

* 依据是 **Vault（越障任务，非循环步态）** 的经验外推，**从未在行走上验证**；
* mjlab 的 walk 任务**恰恰用随机起点，并且它是我们已知最好的 AMP 行走配方**。
⇒ 该注释**存疑**，本次实验正是要检验它。

## ⚠️ 已知风险（必须监控）

相位随机化**提高任务难度**（策略要从任意相位起步并稳住）。从热启动引入时，
策略会突然面对大量未见过的初始相位 ⇒ **可能短期退化**。判据：

* ⭐ `Step/amp_d` / `Loss/discriminator_loss`：判别器是否**"重新有话说"**
  （V27 稳态 disc_loss ≈ 0.0014 = 死透；V30 用 mjlab 配方后 ≈ 0.07 = 活）；
* ⭐ `Style/zero_frac`（死区零梯度占比）：应从 ~0.36 下降；
* `Episode_Reward/track_lin_vel_xy_exp`、`vx_high`：不能为了相位牺牲速度跟踪；
* `Mean episode length`：若崩（<100）则说明"热启动 + 相位随机化"组合不可行，
  下一刀应改为**冷启动**或**降低随机化幅度**（如 `(0.0, 0.5)`）。

## 保持不变（守卫）

奖励侧与 T1l 逐字相同：非零 **17** 项、`track_lin_vel_xy_exp = 1.5`、
`track_base_height` = `1.5 / std 0.3`、`feet_lateral_width −1.0` / `feet_crossing −2.0`、
`joint_deviation_torso/knee = 0`、`dof_pos_limits = −1`、`action_rate_l2 = −0.05`。

⚠️ **算法侧不动**：复用 `amp_walk_t1i_ppo_cfg`（`style_lerp = 0.975`）。
   ⇒ 但注意：有效风格权重 = `(1−λ)·coef` = `0.025 × 4.0` = **0.10 = mjlab 的 4×**
   （`coef = 4.0` 由 T1a 白名单引入）—— **本臂不动它**，见 `ARCH_DIFF_vs_MJLAB.md` §1。

起点：**热启动自 T1a2 的 `model_7000.pt`**（与 T1j/T1i/T1k/T1l 同起点，严格可比）。

⚠️ 本文件不改动、不删除任何既有 cfg（非破坏性）。
⚠️ **实现上必须用 `dataclasses.replace` 构造新对象** —— `G1AmpWalkSceneCfg.motion_reference`
   直接引用**模块级** `motion_reference_cfg`，原地修改会污染 V5 及所有臂（共享可变对象陷阱）。
"""

from __future__ import annotations

import dataclasses

from isaaclab.utils import configclass

from .amp_walk_t1l_env_cfg import G1AmpWalkT1lEnvCfg, G1AmpWalkT1lEnvCfg_PLAY

#: 改动前的基线（fail-fast）
T1M_EXPECT_BEFORE = (0.0, 0.0)
#: 改动后的目标值：起点在 clip 的 [0%, 100%] 上均匀采样
T1M_AFTER = (0.0, 1.0)
#: 守卫：这些不得被顺手改动
T1M_GUARDS = {
    "motion_bin_length_s": None,          # 本臂不切 bin
    "track_lin_vel_xy_exp": 1.5,
    "track_base_height": 1.5,
    "feet_lateral_width": -1.0,
    "feet_crossing": -2.0,
    "joint_deviation_torso": 0.0,
    "joint_deviation_knee": 0.0,
    "dof_pos_limits": -1.0,
    "action_rate_l2": -0.05,
}
T1M_EXPECT_NONZERO = 17
T1M_HEIGHT_STD = 0.3


def _apply_t1m(cfg) -> None:
    mr = cfg.scene.motion_reference
    bufs = mr.motion_buffers
    if len(bufs) != 1:
        raise ValueError(f"[T1m] 预期只有 1 个 motion buffer，实际 {list(bufs)}")
    (key, old_buf), = bufs.items()

    got = tuple(old_buf.motion_start_from_middle_range)
    if got != T1M_EXPECT_BEFORE:
        raise ValueError(
            f"[T1m] motion_start_from_middle_range 基线不符：{got}（预期 {T1M_EXPECT_BEFORE}）"
        )
    if old_buf.motion_bin_length_s is not None:
        raise ValueError(f"[T1m] motion_bin_length_s 应为 None，实际 {old_buf.motion_bin_length_s}")

    # ⚠️ 用 replace 构造新对象，避免污染模块级 motion_reference_cfg
    new_buf = dataclasses.replace(old_buf, motion_start_from_middle_range=T1M_AFTER)
    new_bufs = {**bufs, key: new_buf}
    cfg.scene.motion_reference = dataclasses.replace(mr, motion_buffers=new_bufs)

    # ---- 回读断言（fail-fast）------------------------------------------------
    rb = cfg.scene.motion_reference.motion_buffers[key]
    back = tuple(rb.motion_start_from_middle_range)
    if back != T1M_AFTER:
        raise ValueError(f"[T1m] 回读失败：{back}（应为 {T1M_AFTER}）")
    print(f"[T1m][RSI PHASE] motion_buffers['{key}'].motion_start_from_middle_range"
          f"  {got} → {back}   (motion_bin_length_s={rb.motion_bin_length_s})")

    # ---- 奖励侧守卫（与 T1l 相同）--------------------------------------------
    R = cfg.rewards
    for name, want in T1M_GUARDS.items():
        if name.startswith("motion_"):
            continue
        term = getattr(R, name, None)
        val = None if term is None else float(term.weight)
        if val is None or abs(val - want) > 1e-12:
            raise ValueError(f"[T1m] 守卫失败：{name} = {val}（应为 {want}）")
    _std = float((getattr(R.track_base_height, "params", {}) or {}).get("std", float("nan")))
    if abs(_std - T1M_HEIGHT_STD) > 1e-12:
        raise ValueError(f"[T1m] track_base_height.std = {_std}（应为 {T1M_HEIGHT_STD}）")
    n_on = sum(1 for n in dir(R)
               if getattr(getattr(R, n, None), "weight", None) not in (None, 0))
    print(f"[T1m][RSI PHASE] 奖励守卫通过；非零奖励项 = {n_on}（预期 {T1M_EXPECT_NONZERO}）")
    if n_on != T1M_EXPECT_NONZERO:
        raise ValueError(f"[T1m] 非零奖励项数异常：{n_on}（应为 {T1M_EXPECT_NONZERO}）")


@configclass
class G1AmpWalkT1mEnvCfg(G1AmpWalkT1lEnvCfg):
    """T1m = T1l + `motion_start_from_middle_range` (0,0) → (0,1)（单参数）。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_t1m(self)
        self.run_name = "G1AmpWalkT1mMVPPlane_obs378_rew17_rsiphase"


@configclass
class G1AmpWalkT1mEnvCfg_PLAY(G1AmpWalkT1lEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_t1m(self)
        self.run_name = "G1AmpWalkT1mMVPPlane_obs378_rew17_rsiphase"
