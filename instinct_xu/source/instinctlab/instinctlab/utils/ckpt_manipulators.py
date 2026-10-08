"""checkpoint 修补器：`OnPolicyRunner.cfg["ckpt_manipulator"]` 钩子的实现集合。

`instinct_rl/runners/on_policy_runner.py::load()` 支持一个钩子：

```python
loaded_dict = ckpt_manipulator_func(loaded_dict, self.alg.state_dict(), **kwargs)
```

- 传 `"module.path:function_name"` 字符串到 **runner cfg 顶层**的 `ckpt_manipulator` 即启用；
- 额外参数放 `ckpt_manipulator_kwargs`；
- 签名固定：`(loaded_dict, current_dict, **kwargs) -> loaded_dict`。

为什么需要它（本项目踩过的坑）
------------------------------
**AMP 判别器是存在 checkpoint 里的**（`instinct_rl/algorithms/wasabi.py::state_dict()` 会写
`"discriminator"` 与 `"discriminator_optimizer"` 两个键；`load_state_dict()` 里也会读它们）。
所以一旦**判别器输入维度变了**（例：V12 把 AMP 观测从 beyondmimic/parkour 的 10 帧 670 维
换成 LLP 论文的两帧 134 维），直接 `--resume` 会在

    RuntimeError: Error(s) in loading state_dict for Discriminator:
        size mismatch for model.model.0.weight: [1024, 670] vs [1024, 134]

处**启动即崩**（V12 第一次就是这么挂的）。

而 `Wasabi.load_state_dict` 对**键缺失是容忍的**（只打一行 `[Warning] ... not found`）⇒
删掉这两个键，就能拿到"actor/critic + PPO 优化器照常热启动、判别器与其优化器从零重学"的
效果 —— 这正是"换判别器口径但保留已学会的步态"想要的。
"""

from __future__ import annotations

from typing import Any

#: 判别器相关的两个键（与 `wasabi.py::state_dict/load_state_dict` 逐字对应）
AMP_DISCRIMINATOR_KEYS: tuple[str, ...] = ("discriminator", "discriminator_optimizer")


def drop_amp_discriminator(
    loaded_dict: dict[str, Any],
    current_dict: dict[str, Any],
    *,
    verbose: bool = True,
    **_ignored: Any,
) -> dict[str, Any]:
    """从 checkpoint 里**丢掉 AMP 判别器及其优化器**，其余权重原样返回。

    参数
    ----
    loaded_dict:  从磁盘读出来的 checkpoint（**原地修改**后返回）
    current_dict: 当前算法的 `state_dict()`（本修补器不需要它，只为满足钩子签名）
    verbose:      是否打印做了什么（默认 True —— 让日志里留下痕迹）

    行为
    ----
    * `discriminator` / `discriminator_optimizer` 存在 ⇒ 删除，并打印；
    * 不存在 ⇒ 打印"本来就没有"，不做任何事；
    * 其余键（`model_state_dict` / `optimizer_state_dict` / `lr_scheduler_state_dict` /
      `*_normalizer_state_dict` / `iter` / `infos`）**一律不动**。

    ⚠️ 只有在"判别器口径真的换了"时才该用它：若判别器输入维度没变，丢掉它等于白扔
    已经学好的判别器（会显著变慢，`amp_d` 会先跳变一轮）。
    """
    removed: list[str] = []
    missing: list[str] = []
    for key in AMP_DISCRIMINATOR_KEYS:
        if key in loaded_dict:
            removed.append(key)
            del loaded_dict[key]
        else:
            missing.append(key)

    if verbose:
        total = sum(v.numel() for v in current_dict["model_state_dict"].values()) if (
            "model_state_dict" in current_dict
        ) else 0
        print(
            "[ckpt-manipulator] drop_amp_discriminator: "
            f"丢弃 {removed or '（无）'}"
            + (f"；checkpoint 里本来就没有 {missing}" if missing else "")
            + f"。保留 actor/critic（{total} 个参数）+ PPO 优化器 + 所有 normalizer。"
            " ⇒ 判别器从零重学（Wasabi 会各打一行 [Warning] ... not found，属预期）。"
        )
    return loaded_dict


#: 判别器输入白化（`instinct_rl/modules/normalizer.py::EmpiricalNormalization`）的 buffer
#: 名 → 初始默认值。逐字对应 `register_buffer("_mean"/"_var"/"_std"/"count")`。
AMP_NORMALIZER_BUFFER_DEFAULTS: dict[str, float] = {
    "_mean": 0.0,
    "_var": 1.0,
    "_std": 1.0,
    "count": 0,
}


def reset_amp_normalizer(
    loaded_dict: dict[str, Any],
    current_dict: dict[str, Any],
    *,
    verbose: bool = True,
    **_ignored: Any,
) -> dict[str, Any]:
    """把**判别器输入白化**的运行统计量重置为初始值（μ=0 / var=std=1 / count=0），
    但**保留判别器权重与其优化器**、以及 actor/critic/PPO 优化器。

    为什么必须要它（V32 的关键细节）
    --------------------------------
    把 `amp_normalizer_expert_only` 由 `True` 改成 `False`（= 对齐 AMP_mjlab：
    policy **与** expert 两侧都更新统计量）时，若沿用 checkpoint 里的旧统计量，
    这一改动会**几乎完全不生效**：`EmpiricalNormalization.update()` 是**累计平均**
    （`count` 从上一轮起就是千万级）⇒ 新样本对 μ/σ 的影响可忽略。
    ⇒ 换白化口径时必须清零，让它在几百轮内重新收敛到"两侧混合分布"的统计量，
      否则这一实验臂等于"什么都没改"（假阴性）。

    ⚠️ 只在"白化口径变了"时用；口径没变时用它等于白扔已经标定好的输入尺度。
    """
    import torch  # 局部导入：本模块其余部分不依赖 torch

    disc = loaded_dict.get("discriminator")
    if not isinstance(disc, dict):
        if verbose:
            print(
                "[ckpt-manipulator] reset_amp_normalizer: checkpoint 里没有 "
                "`discriminator` ⇒ 无白化统计量可重置（保持原样）。"
            )
        return loaded_dict

    hit: list[str] = []
    for key in list(disc.keys()):
        leaf = key.rsplit(".", 1)[-1]
        if leaf in AMP_NORMALIZER_BUFFER_DEFAULTS and "normalizer" in key:
            value = disc[key]
            if not isinstance(value, torch.Tensor):
                continue
            disc[key] = torch.full_like(value, float(AMP_NORMALIZER_BUFFER_DEFAULTS[leaf]))
            hit.append(key)

    if verbose:
        print(
            "[ckpt-manipulator] reset_amp_normalizer: "
            f"重置 {len(hit)} 个白化 buffer（{'，'.join(hit[:6])}{' …' if len(hit) > 6 else ''}）"
            " ⇒ 判别器权重与优化器保留，只让输入统计量从头重新估计。"
            if hit
            else "[ckpt-manipulator] reset_amp_normalizer: 没找到任何 normalizer buffer（检查判别器是否接了 normalizer）。"
        )
    return loaded_dict


__all__ = [
    "AMP_DISCRIMINATOR_KEYS",
    "AMP_NORMALIZER_BUFFER_DEFAULTS",
    "drop_amp_discriminator",
    "reset_amp_normalizer",
]
