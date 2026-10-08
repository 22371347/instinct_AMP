# 交给 AMP_mjlab 工作区 agent 的核查清单

> 用途：判定我方（`llp` / `instinct_xu`）AMP 行走复刻里的**参考数据集手臂姿态自相矛盾**
> 与**单脚跳模态**问题，究竟来自**数据迁移/重定向环节**，还是**数据本身**（而 AMP_mjlab
> 用别的手段规避了）。
>
> 请对方 agent **逐条给出结论 + 证据**，不要只回答"看起来没问题"。

---

## 0. 最小上下文（我方现状）

我方把 AMP_mjlab 的 AMP 行走方案迁移到 `instinct_xu`，出现两个问题：

* **A. 参考数据集的手臂姿态在不同 clip 之间互相矛盾** ⇒ 同一个判别器拿到自相矛盾的目标
* **B. 策略收敛到「单脚跳」模态**，左右完全不对称（`joint_error_rmse` 里
  `shoulder_roll` 达 **89°/117°**、`elbow` **85°**，而腿只有 13~35°）

AMP_mjlab 里**没有**出现这两个问题。

## 1. 我方实测数据（请对 AMP_mjlab 原版数据算同一张表）

**我方 npz 位置**
`source/instinctlab/instinctlab/tasks/locomotion/config/g1/motions/amp_walk/*_retargeted.npz`（共 18 个）

**字段（实测）**：`framerate`、`joint_names(29)`、`joint_pos(T,29)`、`base_pos_w(T,3)`、`base_quat_w(T,4)`
⚠️ **我方没有** `joint_vel` / `body_pos_w` / `body_quat_w` / `body_lin_vel_w` / `body_ang_vel_w` 等字段

**关节顺序（29）**：left leg 6（hip_pitch/roll/yaw, knee, ankle_pitch/roll）→ right leg 6 →
waist 3（yaw, roll, pitch）→ left arm 7（shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw）→ right arm 7

**我方逐 clip 关节均值（相对我方 default pose）**

| 关节 | 我方 default | `clip A` 均值 | `clip B` 均值 | **AMP_mjlab 原版 = ?** |
|---|---|---|---|---|
| `left_elbow` | 0.600 | 0.974 | **0.012**（范围 −0.588 ~ 1.026）| ? |
| `left_shoulder_yaw` | 0.000 | −0.596 | −0.402 | ? |
| `right_shoulder_yaw` | 0.000 | **+0.605** | — | ? |
| `left_wrist_roll` | 0.000 | 0.062 | 0.467 | ? |
| `left_knee` | 0.669 | 1.080 | 1.035 | ? |
| `left_hip_pitch` | −0.312 | −0.162 | −0.309 | ? |
| `left_shoulder_roll` | 0.200 | 0.276 | 0.382 | ? |
| **第 0 帧 vs default 平均绝对差** | — | — | **0.219 rad（12.5°）** | ? |

---

## 2. 核查项

### Q1 数据来源与逐文件一致性（最能定性）
- [ ] AMP_mjlab 训练 AMP 行走**具体用哪些 motion 文件**？（列全路径/文件名）
- [ ] 是否包含 `walk_forward_loop_002__A022`、`jog_arc_cw_loop_004__A045` 等同名 clip？
      **逐文件 md5 对比**我方这 18 个 npz。
      ⇒ 同名但内容不同 = **迁移/重定向环节引入**（最强证据）
- [ ] AMP_mjlab npz 的 **keys + shape 全清单**。
      ⚠️ 重点：是否含 `joint_vel`/`body_pos_w`/`body_quat_w`/`body_lin_vel_w`？
      若其 AMP obs 依赖这些而**我方缺失**，我方会拿到常量/零 ⇒ 风格目标失真。

### Q2 手臂姿态在原版数据里是否一致
- [ ] 对原版数据算 **Q1 那张逐 clip 关节均值表**。
      ⇒ 原版也"肘部一支 0.97、一支 0.01" ⇒ **数据本身如此**（转 Q3/Q4 看它怎么规避）
      ⇒ 原版一致 ⇒ **迁移环节**引入（重定向 / 关节映射 / 裁剪）
- [ ] 原版数据的**第 0 帧**是否也是"迈步相位"而非站立姿？

### Q3 判别器（AMP obs）用了什么
- [ ] AMP obs **逐段清单**（joint_pos? joint_vel? body_pos? body_quat?）以及是**绝对量还是相对量**
- [ ] 是否包含**手臂** link / 手臂关节？若**不含**，正好解释它不被手臂不一致拖累
- [ ] 是否把 `joint_pos` **减去 default pose**（相对量）再喂判别器？我方是直接喂绝对值
- [ ] `link_of_interests` / body 集合具体是哪些？我方是 14 个，**含**
      `left/right_shoulder_roll_link`、`elbow_link`、`wrist_roll_link`

### Q4 它用什么手段规避「单脚 / 不对称」
- [ ] `symmetry_cfg` 的**实际完整取值**：`use_data_augmentation` / `use_mirror_loss` /
      `mirror_loss_coeff` / `data_augmentation_func`
- [ ] mirror loss 是否**从第 0 轮就开**？`rl_cfg` 注释提到"阶段1 只开 mirror loss"，
      请给出**完整分阶段计划**（什么时候开 data augmentation、依据是什么）
- [ ] 是否还有**手写的落脚对称奖励**（注释提到 v5 前用过 `feet_lateral_asymmetry`）？
- [ ] 训练集里**是否有 idle / 原地站立 clip**？占多少比例？
- [ ] 是否对**手臂**做特殊处理（屏蔽 / 固定 / 正则化）？

### Q5 复位初始化（RSI）
- [ ] episode reset 时**是否用参考动作某一帧初始化机器人**？
      - 若是：用第 0 帧还是随机帧？`motion_bin_length_s` = ?
      - 若固定用第 0 帧，而第 0 帧是迈步相位 ⇒ **每个 episode 从单腿支撑起步**
- [ ] 参考动作的**播放游标**是否写进 checkpoint？（我方不写 ⇒ 续训时会重置到第 0 帧）

### Q6 默认姿态与关节顺序是否同一个
- [ ] AMP_mjlab 机器人的 `init_state.joint_pos` 具体数值。我方是 8 条正则
      （`hip_pitch −0.312` / `knee 0.669` / `ankle_pitch −0.363` / `elbow 0.600` /
      `shoulder_pitch 0.200` / `shoulder_roll ±0.200`），其余关节 0.0。是否一致？
- [ ] 关节顺序是否与我方一致（left leg → right leg → waist → left arm → right arm）？

---

## 3. 请对方提供的最小产物

1. **逐 clip 29 关节均值 + 第 0 帧姿态表**（原版数据），与本清单 §1 同格式
2. 我方 18 个 npz 与对方同名文件的 **md5 对比结果**
3. 我方**缺失字段**在其 AMP obs 里的用途（用 / 不用）
4. `symmetry_cfg` 实际取值 + 分阶段开启计划
5. **RSI 是否开启 + 初始帧选取规则**
6. Q3 的 AMP obs 逐段清单（带维度）—— 这项决定我方要不要重做 AMP 观测
