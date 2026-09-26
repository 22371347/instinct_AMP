
## 一句话结论

**你们的 `*_retargeted.npz` 与我们的原版数据逐帧完全一致（`mean|Δq| = 0.000000`，误差为 0）——所以"手臂姿态跨 clip 自相矛盾"和"第 0 帧不是站姿"是数据本身固有的，不是迁移/retargeting 引入的。** 我们之所以没被坑，不是数据更干净，而是 ① **只训练精选出的 7 个 clip**（把两个极值 clip 全排除了）、② **风格先验权重很低**。

---

## Q1：训练实际用哪些 motion 文件（决定性证据）

| 项 | AMP_mjlab | instinct_xu |
|---|---|---|
| 目录同名 clip 数 | **17** | **17**（完全同名，无单边缺失） |
| `joint_names` 顺序 | 同 0-28 标准 G1 序 | **与我们的顺序完全一致 ✅** |
| `joint_pos(T,29)` | 345/621/298… 逐 clip 帧数 | **帧数逐 clip 完全相同** |
| **逐帧数值** | — | **17/17 全部 `mean|Δq|=0.0000`、`max|Δq|=0.0000`、`corr=1.0000`** |

> md5 全部不同，但**这只是因为字段集合不同**（你们把 7 字段裁成了 5 字段），内容零差异。

**字段对照（这是你们真正的问题之一）：**

| key | AMP_mjlab | 你们 |
|---|---|---|
| `joint_pos (T,29)` | ✅ | ✅ |
| `joint_vel (T,29)` | ✅ | **缺** |
| `body_pos_w (T,30,3)` | ✅ | **缺** |
| `body_quat_w (T,30,4)` | ✅ | **缺** |
| `body_lin_vel_w (T,30,3)` | ✅ | **缺** |
| `body_ang_vel_w (T,30,3)` | ✅ | **缺** |
| `base_pos_w/base_quat_w` | 从 body 场可得 | ✅ |

我们的 npz 由 csv_to_npz.py 用 mjlab 的 G1 模型做 **FK 重算**生成的（30 link 的位置/姿态/线速度/角速度）。你们只有 base+joints，**若要复刻我们的 AMP 观测就必须自己补 FK**，否则 AMP 参考侧会退化（见 Q3）。

**另外：我们训练的并不是这 17 个文件**，而是 amp_selected.txt 里精选的 **7 个**（`walk_forward_002__A022`、`walk_backward_003__A143`、`walk_sideway_left_002__A021`、`walk_sideway_right_001__A022`、`jog_forward_003__A022`、`arc_walk_left_001__A029`、`walk_arc_cw_002__A046`）。本地 `WalkandRun/` 里那 17 个是开发期全量副本，云端部署时会被清空后只放 7 个。

## Q2：手臂不一致 → **原版数据就有**

你们 §1 的 "clip A / clip B" 我直接定位到了文件名（数字与你表**完全吻合**）：

| clip | 左肘均值 | 左肩yaw | 右肩yaw | 左腕roll |
|---|---|---|---|---|
| **`idle_turn_360_001__A047` ← 你们的 clip A** | **0.974** | −0.596 | **+0.605** | 0.062 |
| **`jog_arc_cw_loop_004__A045` ← 你们的 clip B** | **0.012** | −0.402 | 0.259 | 0.467 |

- 左肘差 **0.96 rad ≈ 55°**；肩 yaw 差 0.19 rad；腕 roll 差 0.4 rad → **判别器确实在吃自相矛盾的风格目标**，你们的观察成立。
- 我们的原版数据**同样**如此（表格逐位一致）。原版极端值：左肘 0.012 ↔ 0.974。
- 第 0 帧偏离 default 为 0.18–0.23 rad（**与你们的 0.219 rad 一致**）→ **原版第 0 帧也是"单腿支撑迈步相位"**，不是站姿。

**我们规避它的关键动作：精选时把极值 clip 全剔除了。** 我们 7 个 clip 的左肘均值 = 1.183 / 1.113 / 1.046 / 1.172 / 0.130 / 0.846 / 1.138 → 集合内范围收敛到 **[0.130, 1.183]**，`jog_arc_cw_004__A045`(0.012)、`idle_turn_270/360`、`step_rotate_idle`、`jog_backward_*`、`arc_jog_*`、`jog_forward_003__A021`、`walk_forward_002__A024`、`walk_backward_001__A022/A024` **全部不在训练集**。

## Q3：AMP 观测到底喂什么

**每帧 195 维，不含任何 `joint_pos`/`joint_vel`：**

| 段 | 维度 | 含义 |
|---|---|---|
| `robot_body_pos_b` | 13×3 = **39** | 各 link 位置 **相对 anchor=`torso_link`，且在 anchor 系下** |
| `robot_body_ori_b` | 13×6 = **78** | 相对姿态的旋转矩阵**前两列**（6D 连续表示） |
| `robot_body_lin_vel_b` | 13×3 = **39** | 世界线速度旋转到**各自 link 自身系** |
| `robot_body_ang_vel_b` | 13×3 = **39** | 同上（角速度） |

13 个 link：`pelvis`、左右 `hip_roll/knee/ankle_roll`、左右 **`shoulder_roll`/`elbow`/`wrist_yaw`**。

**要点纠正**：你们 §1 猜"若不含手臂正好解释"——**我们含手臂**（每侧 3 个 link），但**不含关节角**，所以 "关节角绝对 vs 相对" 这个担忧对我方不存在；手臂姿态是**通过 3 个 link 的相对位姿间接**体现的，且位置/姿态都**相对 torso_link**。所以手臂矛盾对我们**同样可见**——我们能work靠的不是观测屏蔽，而是下面的 Q4。

## Q4：AMP 权重与对称方案（真正的差异点）

```yaml
amp_reward_coef: 0.1          # 风格先验只占总奖励约 10%
amp_task_reward_lerp: 0.75    # τ=0.75：90% 任务奖励 + 10% 风格
symmetry_cfg = {
  use_data_augmentation: False,
  use_mirror_loss: True,
  mirror_loss_coeff: 0.1,
  data_augmentation_func: "src.tasks.amp_loco.mdp.symmetry:g1_data_augmentation",
}
```
- **mirror loss 从第 0 轮就开**（v5 起从头训练，v5.1 续训保持开启）；**没有分阶段**，`use_data_augmentation` 阶段从未启用。实现在 symmetry.py + amp_ppo.py，可直接抄。
- 手写的对称奖励 `feet_lateral_asymmetry` **已在 v5 删除**（只留代码供回滚，未注册）；现存约束只有 `feet_lateral_width`/`feet_crossing`。
- **训练集内没有 idle clip**（`idle_turn_*`、`step_rotate_idle` 全部排除）。
- **手臂没有任何特殊处理**（不屏蔽、不冻结、不正则）。
- 另有一个你们可能没注意的缓解机制：`amp_disable_style_on_stand=True`（`stand_speed_threshold=0.1`）——**站立/低速指令下直接关掉 AMP 风格项**，正好是数据最不一致的工况。

## Q5：RSI / body_idx

- 我们**有**参考初始化：重置时在**全部 clip 拼接后的所有帧中随机取一帧**（`torch.randint`），**不是取第 0 帧**，所以不会每次 episode 都从"单腿支撑迈步相位"起步。
- **没有 motion cursor / phase 变量**（非 tracking 任务），因此 **`body_idx`/cursor 不存在、也不需要写进 checkpoint**；`motion_bin_length_s` 这套设计在我们这里没有对应物。
- walk 任务 `delay_reset_env_ratio=0`、`max_delay_steps=0`（无延迟终止）。

## Q6：default pose / 关节顺序

完全一致 ✅：`hip_pitch −0.312`、`knee 0.669`、`ankle_pitch −0.363`、`elbow 0.6`、`shoulder_pitch 0.2`、`shoulder_roll`(L+0.2 / R−0.2)、其余 0，root 高 0.78；关节顺序已由你们 `joint_names` 字段逐项验证一致（`是 True`）。

---

## 给你们项目的最小修复建议（按优先级）

1. **先裁 AMP 参考集**：至少剔除左肘均值 <0.3 的 `jog_arc_cw_loop_004__A045`（0.012）与 >0.95 的 `idle_turn_270/360`、`step_rotate_idle`；最稳是直接对齐我们那 7 个。**这一步预期收益最大**——矛盾的风格目标消失后"单脚跳/不对称"应显著缓解。
2. **降低 `amp_reward_coef`** 到 ~0.1（我们验证过 0.3 会诱发姿态/模态异常）。
3. **补 FK 字段**（`body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w`，30 link），让 AMP 观测与我们同构；否则判别器拿到的参考态分布与策略态分布**不在同一空间**，是"达不到预期成果"的直接嫌疑。
4. **上 mirror loss**（含 cfg 注入的 YAML 序列化坑，见 `rl/runner.py` 的 copy/restore 处理）。
5. **检查 reset 是否固定第 0 帧**；若是，改成随机帧。

需要我把这份核对结论直接整理成可直接回复给 `instinct_xu` agent 的 markdown（含逐条 Q1–Q6 结论 + 证据表）吗？