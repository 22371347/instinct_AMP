Searched for regex `amploss_coef|grad_pen|lambda_|gradient_penalty` (`**/*.py`), 19 results

Read [](file:///home/leo/rsl_project/AMP_mjlab/rsl_rl/algorithms/amp_ppo.py#505-505), lines 505 to 560

Read [](file:///home/leo/rsl_project/AMP_mjlab/rsl_rl/algorithms/amp_ppo.py#105-105), lines 105 to 140

Searched for regex `learning_rate|amp_replay_buffer_size|discriminator_learning|lr=` (`**/src/tasks/amp_loco/**/*.py`), 2 results

## 短答

**没有名为 `discriminator_gradient_penalty` 的配置项**——cfg 里不存在这个字段，**功能有，但系数全部写死**：

| 位置 | 值 | 说明 |
|---|---|---|
| λ（内层） | **10** | 硬编码在调用点 `amp_ppo.py:449`：`compute_grad_pen(*sample_amp_expert, lambda_=10)` |
| 外层权重 | **`amploss_coef = 1.0`** | 硬编码 `amp_ppo.py:114`，对 `amp_loss` 和 `grad_pen_loss` **同等加权** |
| 日志键 | **`Loss/amp_grad_pen`** | 不是 `discriminator_gradient_penalty` |

## 具体表达式

```python
# rsl_rl/modules/discriminator.py:84-112
def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10):
    expert_data = torch.cat([expert_state, expert_next_state], dim=-1)   # 390 维
    expert_data.requires_grad = True
    disc = self.amp_linear(self.trunk(expert_data))
    grad = autograd.grad(disc, expert_data, grad_outputs=ones, create_graph=True)[0]
    return lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()   # 目标：‖∇d‖ → 0

# rsl_rl/algorithms/amp_ppo.py:449-450
grad_pen_loss = self.discriminator.compute_grad_pen(*sample_amp_expert, lambda_=10)
loss += self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss
```

$$\mathcal{L}=\underbrace{1.0}_{\texttt{amploss\_coef}}\cdot\underbrace{\tfrac12\Big[\mathrm{MSE}(d_{exp},+1)+\mathrm{MSE}(d_{pol},-1)\Big]}_{\texttt{Loss/amp}}\;+\;\underbrace{1.0}_{\texttt{amploss\_coef}}\cdot\underbrace{10\cdot\overline{\lVert\nabla_x d(x)\rVert_2^2}}_{\texttt{Loss/amp\_grad\_pen},\ x=[s,s'],\ \text{仅 expert}}$$

三个要点：
1. **惩罚目标是"梯度范数 → 0"**，不是 WGAN-GP 的 → 1 ⇒ 实质是 **R1 正则**，且**只对 expert 数据**求。
2. 求导对象是**整条拼接输入** `x = [expert_state, expert_next_state]`（390 维），不是单帧。
3. **强度全在 λ=10 里**；外层不再额外放大。相对强度 = `10·mean‖∇d‖² / amp_loss`。

## ⚠️ 和你们 tmp.md 的数字不能直接比

你们的 tag `discriminator_gradient_penalty = 0.0142` 外层系数是 **5.0**；我们的 `Loss/amp_grad_pen` **已经含 λ=10**（`compute_grad_pen` 返回值里就乘过了）。所以：

- 若你们的 0.0142 是**未乘**的外层前 `mean‖∇d‖²`，同一量在我们日志里会显示成 `10 × 0.0142 = 0.142`（**大 10 倍**，纯粹是"λ 记在哪一层"的差别）；
- **正确的对比口径是"grad_pen 项占判别器总梯度的比例"**，不是单个数字。你们自测是 62%；我们这边按 `Loss/amp ≈ 0.015` 量级估，占比大概率**同样偏高甚至更高**。

## 另外两个顺带查到的、容易误读的点

```python
params = [
    {"params": policy.parameters(),              "name": "policy"},
    {"params": discriminator.trunk.parameters(),      "weight_decay": 10e-4},  # = 1e-3
    {"params": discriminator.amp_linear.parameters(), "weight_decay": 10e-2},  # = 0.1 ⚠️
]
self.optimizer = optim.Adam(params, lr=learning_rate)
```

1. **`10e-2` = 0.1，不是 1e-2** → 输出层 `amp_linear` 的权重衰减**很大**，会把 logits `d` 持续往 0 拉，**与"判别器饱和/风格奖励水平"直接耦合**（`d→0` 时 `quad(0)=0.75`）。对比你们 tmp.md 的 `3e-4 × 59.6` 与 `0.04 × logit_wd` ⇒ 我们是 **1e-3 / 0.1**，比你们大 **3.3× / 2.5×**。这个写法极易被误读成 `1e-4/1e-2`，建议有意的话加注释。
2. **判别器没有独立学习率**：与策略共用同一个 Adam（`lr = 1.0e-3`，`rl_cfg.py:67/141`）。所以你们 tmp.md 里 **E2 的"降判别器 lr"在我们这份代码里需要先改结构**，现在无法单变量调。

## 汇总对照

| 项 | AMP_mjlab | 你们 |
|---|---|---|
| 字段 `discriminator_gradient_penalty` | ❌ 不存在（写死） | ✅（0.0142） |
| λ / 外层系数 | **10 / 1.0** | ? / 5.0 |
| 目标 | ‖∇d‖ → **0**（R1，仅 expert） | ? |
| 日志键 | `Loss/amp_grad_pen` | `discriminator_gradient_penalty` |
| trunk / head weight decay | **1e-3 / 0.1** | 3e-4 / 0.04 |
| 判别器独立 lr | ❌ 共用 1e-3 | ? |

云上现在仍 `Connection refused`（183.147.142.40:30542），等恢复我可以把 `Loss/amp`、`Loss/amp_grad_pen`、`Loss/amp_policy_pred/expert_pred` 的**真实数值**拉下来，把"惩罚项占比"和"`d` 的饱和水平"都算实，跟你们做个同口径对比。需要我做成对比脚本吗？