# V22 判别器白化（"新版 AMP 风格方案"）—— 实现与自检记录

> 日期：2026-10-03 · 关联：`RESULTS_amp_vs_mjlab_params.md` §F（方案与离线数据）、
> `RESULTS_amp_frames_body_vs_joint.md`（帧数 × 表征）
> 自检脚本：`instinct/tools/test_amp_normalizer.py`（**33/33 通过**，纯 torch，无需 Isaac/GPU）

## 0. 一句话

把 AMP_mjlab 判别器里的 `amp_normalizer`（状态白化）真正实现进来，**且不照抄 mjlab 那个错的 GP**：
白化统计量**只由 expert 侧**累积，GP（Lipschitz 惩罚）改到**归一化空间**求。

## 1. 改了哪些文件（全部是"新增分支"，默认行为与 V1~V21 **逐字不变**）

| 文件 | 改动 |
|---|---|
| `instinct_xu/source/instinct_rl/instinct_rl/modules/normalizer.py` | 新增 `EmpiricalNormalization.normalize(x, update=False)`：白化但不更新统计量（`forward` 在 training 时会顺手 update） |
| `.../modules/discriminator.py` | `forward(x, hidden_states, masks, update_normalizer=True)`；新增 `normalize_input(x)`（只看不改）、`forward_normalized(x)`（输入已白化时跳过 normalizer，供 GP 用） |
| `.../algorithms/wasabi.py` | 新 kwarg `amp_normalizer_expert_only=True` / `amp_grad_pen_space="raw"`；`compute_amp_losses` 里**只用 reference 更新统计量**；`compute_discriminator_gradient` 增加 `"normalized"` 分支 |
| `.../instinctlab/utils/wrappers/instinct_rl/algorithms.py`（桥接层） | 声明并透传这两个 kwarg（校验取值）；诊断探针 `amp_d_ref` 加 `update_normalizer=False` |
| `.../tasks/locomotion/config/g1/agents/amp_walk_ppo_cfg.py` | 算法 `@configclass` 里**显式声明**这两个字段（否则 `to_dict()` 不带上） |
| `.../agents/amp_walk_v22_ppo_cfg.py`（新） | V22 = V21 + 白化（单变量） |
| `.../config/g1/__init__.py` | 注册 `Instinct-Amp-Walk-V22-G1-v0` / `-Play-v0`（**env 侧复用 V21 的类** ⇒ 环境一字不改） |

## 2. 必须**成对**的三处，缺一个就白做

1. **`discriminator_kwargs` 接上 normalizer**
   `{"normalizer_class_name": "EmpiricalNormalization", "normalizer_kwargs": {"eps": 0.01, "until": None}}`
2. **`amp_grad_pen_space = "normalized"`** —— 见 §3，不改就崩。
3. **`amp_normalizer_expert_only = True`** —— `Discriminator.forward` 默认会更新统计量，而
   rollout 期**每个 env step** 都会拿**策略数据**前向一次算风格奖励
   （`wasabi.py: compute_auxiliary_reward`）。实测：不关掉，统计量 mean 漂移 **1.51**；
   本次实现把 `wasabi.py` 里**全部 4 处**判别器前向都显式指定了 `update_normalizer`（用静态 AST 检查守住）。

## 3. 为什么 GP 必须换空间（本地实测，与真实链路一致）

* 物理：白化后 `∂d/∂x = (∂d/∂x̃)/(σ+eps)`。**`eps` 同时是"σ 下限"和"raw 空间放大的上界 `1/eps`"**。
* 同一判别器、同一批数据：

| `amp_grad_pen_space` | ‖∂d/∂x‖ | GP 项 | 判别器饱和后（MSE→5.6e-4）GP / MSE |
|---|---|---|---|
| `"raw"` | 1.90 | 3.62 | **1.2e6×** ⇒ 判别器损失被 GP 完全支配，学不了"区分" |
| `"normalized"` | 0.108 | 0.0116 | ≈ 0.3×（可控小量） |

* 与**旧配置（无白化）**对齐：

| 配置 | GP 项（未训） | 饱和后 GP 项 |
|---|---|---|
| 无白化（V19d/V20/V21 现状） | 0.0079 | ~0.12 |
| 白化 + `"normalized"` | 0.0082（**1.04×**） | 0.17（**1.48×**） |

⇒ **`discriminator_gradient_penalty_coef = 1.0` 原样沿用**，V22 相对 V21 仍是单变量。
（⚠️ mjlab 的 `compute_grad_pen` 用的是**原始** expert 数据算梯度、主损失却用归一化状态
⇒ 约束的是"另一个函数"，且 `1/σ` 加权偏向小尺度维。**这一点故意不照抄**。）

## 4. 白化本身的效果（本地实测）

| 项 | 结果 |
|---|---|
| 常规维白化后 | `\|mean\| < 2e-8`，`std ∈ [0.92, 0.98]`（`eps=0.01` 带来的 2~8% 压缩） |
| 近常数维（σ≈1e-5） | 被 σ 下限**压住**：std ≈ 1e-3（**不**放大到 1 ⇒ 不把量化噪声当特征） |
| 统计量专家独占 | actor 侧前向（`update_normalizer=False`）漂移 = **0** |
| 离线可分性（`RESULTS_amp_vs_mjlab_params.md` §F） | AUC 0.850 → **0.927**（h=8），`style_std` **×2.7**；拐点前移（AUC≈0.85：raw 需 h=8、白化约 h=5） |

## 5. V22 训练时要盯的探针

| 探针 | 期望 | 异常含义 |
|---|---|---|
| `Loss/discriminator_gradient_penalty` | 与 V21 同量级（~1e-3） | 涨到 1e0 ⇒ GP 空间没生效（`amp_grad_pen_space` 没传进去） |
| `Loss/discriminator_loss` | **不再 2000 轮就饱和到 0.003**（V20 的病） | 仍快速饱和 ⇒ 白化没接上（`discriminator_kwargs` 被覆盖） |
| `Step/amp_style_std` | 期望 > V21 的 0.079（离线预测 ×2.7） | 没涨 ⇒ 白化没进门控前的风格项 |
| `Step/amp_style_zero_frac` | 应保持较低（quad `k=0.125` 的死区） | 升高 ⇒ `d` 分布变差 |
| `amp_d_ref` | 与 `d` 的 gap 明显 | gap ≈ 0 ⇒ 判别器区分不开（AMP 失效） |

## 6. 自检脚本覆盖（`python instinct/tools/test_amp_normalizer.py`，33 项）

* **T0 接线**：实际加载的是 `instinct_xu/source/instinct_rl` 那棵树；新 API 全部存在；
  两个新 kwarg 默认值 = 旧行为；桥接层 AST 检查；**WASABI 全部 4 处判别器前向都显式给了
  `update_normalizer`**（含 rollout 期的 `compute_auxiliary_reward` 与 GP 两条分支）。
* **T1 白化正确性** / **T2 统计量专家独占（含"漏传会漂移 1.51"反证）**。
* **T3 GP 空间**：raw vs normalized 的量级、判别器**饱和后**的支配关系、与旧配置的对齐
  （连同 §3 的两张表）。
* **T4 真实代码路径**：直接调 `WasabiAlgoMixin.compute_discriminator_gradient` 的
  `"raw"` / `"normalized"` 两条分支；并验证**无 normalizer 时 `"normalized"` 退化为旧行为**。
* **T5 V22 配置接线**（静态，无需 Isaac）：三个开关、从零、gym 注册、env 复用 V21。

## 7. 启动 V22（等云端开机）

```bash
# 云端（经 cloud_run_bg.sh；与 V21 同一套工具，改任务 id / 日志名即可）
bash instinct/tools/upload_v21.sh                 # 若只改了上面这几个文件，重跑一次覆盖
bash /root/cloud_launch_v21.sh --task Instinct-Amp-Walk-V22-G1-v0 --log v22
```
⚠️ 云端 `instinct_rl` 是 **editable 安装**（`pip install -e source/instinct_rl`），
   所以上面 `modules/` 与 `algorithms/` 的改动**必须**一并上传，否则白化代码不存在（
   `Discriminator.forward` 会因多一个 kwarg 直接 TypeError）。

## 8. 后续单变量顺序（已排好）

1. **V22**（本代）：白化 + h=8，对 V21 单变量。
2. 若 V22 明显更好 ⇒ **降 h 到 6**（离线已证：白化后 AUC 0.896@h6 ≈ raw 0.850@h8），省 ~25% 算力。
3. 环境侧："方向投影门控" —— 把 `feet_air_time` 的 `motion_threshold` 从速度幅值改成
   `max(0, v_act·v̂_cmd)`，惩罚"踏步前进/后退"。
4. 若 V22 的 GP 在训练中期明显变大 ⇒ 单变量扫 `discriminator_gradient_penalty_coef`（0.3 / 1.0 / 3.0）。
