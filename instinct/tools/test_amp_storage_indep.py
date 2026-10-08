#!/usr/bin/env python3
"""`AmpStorage` 专家批解耦（V29 ②）的行为单测。

判据（用**唯一 tag** 编码 (t, b)，所以能直接看出参考批取的是哪一格）：

  令 `actor_states[t, b, :] = reference_states[t, b, :] = tag(t, b) = t*1000 + b`。

  * `independent_reference_sampling=False`（旧口径）⇒ actor 批与 reference 批**逐个相同**
    （因为两边取同一 `(T_select, B_select)`）⇒ 这是**相位俘获**的直接证据。
  * `True`（V29 口径）⇒ 两者**几乎不重合**，且参考批的 `t` 覆盖整个 buffer（不是单一时刻）。

用法：python instinct/tools/test_amp_storage_indep.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/leo/rsl_project/llp/instinct_xu/source/instinct_rl")

import torch  # noqa: E402

from instinct_rl.storage.amp_storage import AmpStorage  # noqa: E402

NUM_ENVS = 64
NUM_TRANS = 24
DIM = 4

fail: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        print(f"  ✅ {label}")
    else:
        print(f"  ❌ {label}   {detail}")
        fail.append(label)


def build(indep: bool) -> AmpStorage:
    st = AmpStorage(
        NUM_ENVS,
        NUM_TRANS,
        (DIM,),
        (DIM,),
        device="cpu",
        independent_reference_sampling=indep,
    )
    t = torch.arange(NUM_TRANS).view(NUM_TRANS, 1, 1).float()
    b = torch.arange(NUM_ENVS).view(1, NUM_ENVS, 1).float()
    tag = (t * 1000.0 + b).expand(NUM_TRANS, NUM_ENVS, DIM).clone()
    st.actor_states.copy_(tag)
    st.reference_states.copy_(tag)
    return st


def main() -> int:
    print("=" * 70)
    print("AmpStorage 专家批解耦单测（V29 ②）")
    print("=" * 70)

    # ---- 旧口径：两者必须逐个相同（证明"相位俘获"的机制）----
    print("\n[1] independent_reference_sampling=False（V27 口径）")
    st = build(False)
    n_same = 0
    n_tot = 0
    for mb in st.mini_batch_generator(num_mini_batches=4, num_epochs=1):
        a = mb.actor_states.reshape(len(mb.actor_states), -1)
        r = mb.reference_states.reshape(len(mb.reference_states), -1)
        n_same += int((a[:, :1] == r[:, :1]).sum())
        n_tot += len(a)
    print(f"  actor 与 reference 的 tag 完全相同的样本比例 = {n_same}/{n_tot}")
    check(n_same == n_tot, "旧口径：参考批与策略批**同 (T,B)**（相位俘获得以证实）")

    # ---- V29 口径：几乎不重合，且 t 覆盖整个 buffer ----
    print("\n[2] independent_reference_sampling=True（V29 口径）")
    st = build(True)
    n_same = 0
    n_tot = 0
    t_seen: set[int] = set()
    for mb in st.mini_batch_generator(num_mini_batches=4, num_epochs=1):
        a = mb.actor_states.reshape(len(mb.actor_states), -1)[:, 0]
        r = mb.reference_states.reshape(len(mb.reference_states), -1)[:, 0]
        n_same += int((a == r).sum())
        n_tot += len(a)
        t_seen |= set((r / 1000.0).floor().to(torch.long).tolist())
    frac = n_same / n_tot
    print(f"  重合样本比例 = {n_same}/{n_tot} = {frac:.4f}（期望 ≈1/{NUM_TRANS}={1/NUM_TRANS:.4f}）")
    print(f"  参考批覆盖到的时刻数 = {len(t_seen)}（buffer 共 {NUM_TRANS} 步）")
    check(frac < 0.15, "V29 口径：参考批与策略批**基本不重合**（解耦生效）", f"frac={frac:.4f}")
    check(len(t_seen) >= NUM_TRANS // 2, "参考批覆盖**多相位**（不是单一时刻）", f"t_seen={len(t_seen)}")

    # ---- 默认值必须保持旧行为（零回归）----
    print("\n[3] 默认构造（不传 flag）")
    st_def = AmpStorage(NUM_ENVS, NUM_TRANS, (DIM,), (DIM,), device="cpu")
    check(st_def.independent_reference_sampling is False, "默认 False ⇒ 既有 run 行为不变")

    # ---- 维度/形状不被破坏 ----
    print("\n[4] 批形状与 dtype")
    st = build(True)
    mb = next(iter(st.mini_batch_generator(num_mini_batches=4, num_epochs=1)))
    check(tuple(mb.actor_states.shape) == (NUM_ENVS * NUM_TRANS // 4, DIM),
          f"actor_states 形状 {tuple(mb.actor_states.shape)}")
    check(tuple(mb.reference_states.shape) == (NUM_ENVS * NUM_TRANS // 4, DIM),
          f"reference_states 形状 {tuple(mb.reference_states.shape)}")

    # ---- V30：离线专家 buffer（mjlab AMPLoader 口径）----
    print("\n[5] V30 离线专家 buffer：参考批必须来自数据集，而不是活体影子机器人")
    st = build(False)  # 注意：**不开** independent_reference_sampling
    n_buf = 500
    buf = (torch.arange(n_buf).view(n_buf, 1).float().expand(n_buf, DIM).clone() + 900000.0)
    st.expert_buffer = buf
    n_from_buf = 0
    n_tot = 0
    n_match_actor = 0
    last_dim = None
    for mb in st.mini_batch_generator(num_mini_batches=4, num_epochs=2):
        a = mb.actor_states.reshape(len(mb.actor_states), -1)[:, 0]
        r = mb.reference_states.reshape(len(mb.reference_states), -1)[:, 0]
        # ⚠️ buffer 的值域是 [900000, 900500)（每行 tag = 900000 + row）⇒ 判据要用区间
        n_from_buf += int(((r >= 900000.0) & (r < 900500.0)).sum())
        n_match_actor += int((a == r).sum())
        n_tot += len(a)
        last_dim = mb.reference_states.shape[-1]
    print(f"  参考批来自 buffer 的比例 = {n_from_buf}/{n_tot}；与 actor 逐格相同的比例 = {n_match_actor}/{n_tot}")
    check(n_from_buf == n_tot, "参考批 100% 来自离线数据集", f"{n_from_buf}/{n_tot}")
    check(n_match_actor == 0, "参考批与策略批**不同格**（不再是活体影子机器人那一步）")
    check(last_dim == DIM, f"参考批维度 = buffer 维度 = {DIM}（维度校验由载入处 wasabi 负责）")
    # buffer 行数无关（这里是 500 行）；确认每次抽样的下标覆盖了 buffer 的多样行
    rows_seen = set()
    for mb in st.mini_batch_generator(num_mini_batches=1, num_epochs=3):
        r = mb.reference_states.reshape(len(mb.reference_states), -1)[:, 0]
        rows_seen |= set((r - 900000.0).to(torch.long).tolist())
    print(f"  三次遍历共抽到的不同 buffer 行数 = {len(rows_seen)}/{n_buf}")
    check(len(rows_seen) > n_buf * 0.5, "抽样覆盖了 buffer 的大部分行（不是只取固定几行）")

    print()
    if fail:
        print(f"[FAIL] {len(fail)} 项未通过：{fail}")
        return 1
    print("[PASS] 专家批解耦生效；默认行为零变化")
    return 0


if __name__ == "__main__":
    sys.exit(main())
