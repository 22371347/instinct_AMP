# 镜像损失（mirror loss）设计 —— 仿 AMP_mjlab `symmetry.py`

> 目标：给策略加**左右等变性**约束，从根因上消除**系统性左右偏置 / 单腿偏置**。
> 这是 AMP_mjlab v5 用来**替代**手写落脚对称惩罚的方案，我方**完全没有迁移**。

---

## 1. 机制

$$
\pi\big(M_{\text{obs}}(o)\big) \;\approx\; M_{\text{act}}\big(\pi(o)\big)
$$

⚠️ **不是**"同一时刻左腿 = 右腿"—— 正常行走左右相位不同，同刻左右关节本就不同。
它约束的是"**左摆/右支撑**"与"**右摆/左支撑**"两个状态之间的对应关系。

**为什么对我方有效且安全**：它是**策略层**约束，只要求"镜像输入 → 镜像输出"。
G1 在矢状面反射下是自同构，所以这个约束**对任何参考数据都成立** ⇒
**不受**"我们的 clip 混了前向/后向/侧移/弧线、镜像会产生错误标签"这一顾虑影响
（那个顾虑只适用于 `symmetric_augmentation_*` 这种**数据层**增广）。

## 2. 我方姿态/动作的镜像规则（矢状面反射 y → −y）

| 量 | 变换 |
|---|---|
| 极向量 `[x,y,z]` | `[x, −y, z]` |
| 轴向量 `[ωx,ωy,ωz]` | `[−ωx, ωy, −ωz]` |
| 左右关节 | 交换 `left_*` ↔ `right_*` |
| 关节符号 | **pitch 保号；roll / yaw 反号** |
| 腰部（无左右对应）| 自映射：`yaw −`、`roll −`、`pitch +`（同一条符号规则）|

**✅ 规则已被我方默认姿态验证**（这是个天然的不动点检验）：
`left_shoulder_roll = +0.2`、`right_shoulder_roll = −0.2` ⇒ 交换且反号后仍是
`+0.2 / −0.2` ✓ 自洽。同理 `hip_roll` 的左右行程 `[−0.524, 2.967]` / `[−2.967, 0.524]`
互为镜像 ✓ 佐证"roll 反号"。

### 2.1 策略观测（626 维）逐段镜像

| 段 | 切片 | 维度 | 镜像方式 |
|---|---|---|---|
| `height_scan` | `0:527` | 527 | ⚠️ **reshape 成 (31×17) 后翻转 y 方向的行**（见 §5 待核实）|
| `velocity_command` | `527:530` | 3 | `(vx, −vy, −wz)` |
| `base_lin_vel` | `530:533` | 3 | `(x, −y, z)` |
| `projected_gravity` | `533:536` | 3 | `(x, −y, z)` |
| `base_ang_vel` | `536:539` | 3 | `(−ωx, ωy, −ωz)` |
| `joint_pos_rel` | `539:568` | 29 | 按**关节名**求置换 + 符号 |
| `joint_vel` | `568:597` | 29 | 同上 |
| `last_action` | `597:626` | 29 | 同上 |

⚠️ **关键**：我方 obs/action 的关节顺序是 **articulation（肩优先交错）序**，
**不是** npz 的"左腿→右腿→腰→左臂→右臂"序 ⇒ **置换必须用运行时的 `joint_names` 现算，
绝不能硬编码索引**。

```python
def build_joint_mirror(joint_names: list[str]):
    """返回 (perm, sign)，使 (M q)[i] = sign[i] * q[perm[i]]。
    要求:  M(M(q)) == q  （对合）
           M(q_default) == q_default  （默认姿态是不动点）
    """
    idx = {n: i for i, n in enumerate(joint_names)}
    perm, sign = [], []
    for n in joint_names:
        if n.startswith("left_"):
            m = "right_" + n[len("left_"):]
        elif n.startswith("right_"):
            m = "left_" + n[len("right_"):]
        else:
            m = n                                   # waist 自映射
        if m not in idx:
            raise KeyError(f"{n} 在 joint_names 里没有镜像对应 {m}")
        perm.append(idx[m])
        sign.append(1.0 if "pitch" in n else -1.0)  # pitch 保号，roll/yaw 反号
    return perm, sign
```

动作（29 维）与 `joint_pos_rel` / `joint_vel` / `last_action` 用**同一套** `(perm, sign)`。

## 3. 损失与集成点

对每个 minibatch：

$$
\mathcal{L}_{\text{mirror}} \;=\; c_{\text{mirror}}\cdot \big\lVert\, \mu_\pi\big(M_{\text{obs}}(o)\big) \;-\; M_{\text{act}}\big(\mu_\pi(o)\big) \big\rVert_2^2
$$

* 用**策略均值** `μ_π` 而不是采样动作（降噪；AMP_mjlab 的 docstring 写作 `π`，实现层面同为均值）
* `c_mirror = 0.1`（AMP_mjlab `mirror_loss_coeff = 0.1`）

**集成位置**：`instinct_xu/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/algorithms.py`
（该文件已经承载了 `WasabiAlgoMixin` + lerp 奖励混合的子类）。
需要在 `PPO.update` 的代理目标里多算一项 ⇒ **覆写 `update()`**：
把上游的 surrogate/entropy 部分照抄，再 `loss += mirror_coef * L_mirror`。

⚠️ **代价与风险**：覆写 `update()` 会**复制上游代码** ⇒ 日后 instinct_rl 升级容易漏改。
两种做法取舍：
1. **覆写 `update()`**（推荐先做）：实现直接，但必须加醒目注释标出"从上游复制的范围"
2. 让上游提供 loss hook（更干净，但要改 `instinct_rl` 本身，影响面更大）

## 4. 配置项（新增，**默认关闭**，保持既有实验可复现）

```python
# agents/amp_walk_ppo_cfg.py（或 v4 的 ppo cfg）
use_mirror_loss: bool = False          # 默认关；v4 实验开启
mirror_loss_coeff: float = 0.1
```

⚠️ **不要**同时打开 `symmetric_augmentation_*`（数据层）：我方 clip 含侧移/弧线/后退，
镜像增广会产生错误标签 —— 这一点我方的原有顾虑**依然成立**。

## 5. 待核实项（实现前必须先做）

- [ ] **`height_scan` 的射线排序**：`patterns.GridPatternCfg(resolution=0.1, size=(3.0,1.6))`
      给出 31×17 = 527 条射线，但**行主序/列主序、先 x 后 y 还是先 y 后 x 未知**。
      **核实方法**：造一个已知斜面（如沿世界 +y 单调升高），dump 首帧 scan，
      看数值沿 527 维的递增周期 ⇒ 直接读出 (nx, ny) 与主序，再定 `reshape` 与翻转轴。
      ⚠️ 排序搞错的后果是"镜像后的 scan 不是镜像地形"，会引入**错误监督**（比不加更糟）。
- [ ] `joint_names` 的**镜像配对完整性**：29 个关节里每个都必须找得到对应项
      （上面代码会 `raise`，跑一次单测即可覆盖）。
- [ ] `waist_yaw` 的符号：其行程对称 ⇒ 无法用范围佐证，需用"镜像后姿态世界坐标一致"验证
      （见 §6 测试 3）。

## 6. 验收与自举计划

**单元测试（必须先全绿再开训练）**

1. `M(M(x)) == x` —— 对合性（obs 全段 + 动作）
2. `M(q_default) == q_default` —— 默认姿态是不动点 ⭐（这条最能证明符号规则对）
3. **物理一致性**：构造一个"左脚前、右脚后"的姿态，镜像后应得到"右脚前、左脚后"，
   且两者的**世界坐标 y 分量整体反号**（用 FK 或 MuJoCo 验证）

**自举步骤**

| 步 | 动作 | 判据 |
|---|---|---|
| 1 | 实现 `mirror.py` + 单测（不开训练）| 上述 3 条全绿 |
| 2 | 加 `use_mirror_loss` 开关（默认 False）| 关时与原实验**逐位一致**（跑 50 轮对比）|
| 3 | 50 轮短跑，开 mirror loss | 不崩；`Policy/mean_noise_std` 不暴涨；AMP 分支仍活着 |
| 4 | **v4 单变量全跑**（只加 mirror loss）| 左右不对称度下降：我方指标是臂 7 组 `Σ(R−L)` 当前 **+1.739**、腿 6 组 +0.159 ⇒ 期望显著下降 |
| 5 | 之后再考虑参考数据/手臂不一致的修法 | 见 `AMP_MJLIB_CHECK.md` |

**观察指标（我方已有）**
`joint_error_rmse/<joint>`（不门控，是真正的姿态指标）+ 左右配对差 `R−L`。
⚠️ **不要**用 `Episode_Reward/track_default_pose_on_stand` 判断姿态：它是**门控**的
（实测等效门控质量仅 **0.037**），小值只说明"大部分样本上关着"。
