一、先回答 §6：AMP_mjlab 判别器的确切形式（与我们逐行相同）
项	AMP_mjlab 实际	你们的
奖励形式	没有可选项，写死 quad（rsl_rl/modules/discriminator.py::predict_amp_reward）：r = c · clamp(1 − 0.25(d−1)², min=0)，d = amp_linear(trunk([s, s'])) 原始线性输出，无 sigmoid	quad ✅ 相同
c = amp_reward_coef	0.1（在函数内部乘进 reward）	0.1 ✅
lerp	R = (1−0.75)·c·r_quad + 0.75·r_task → 风格权重 0.025 = 1/30	同 ✅
损失 & 目标	expert_loss = MSELoss(expert_d, +1)、policy_loss = MSELoss(policy_d, −1)、amp_loss = 0.5(expert+policy)；grad_pen lambda_=10	±1 目标，相同
判别器结构	[1024, 512, 256]；trunk wd 1e-3、head wd 1e-1	相同
暴露原始 d 的日志	✅ Loss/amp_policy_pred = mean(d_policy)、Loss/amp_expert_pred = mean(d_expert)（amp_ppo.py:542-543）	你们有 discriminator_actor/reference
两个校正/提醒：

你们的反解公式是对的（r=0.763 若还被 ×0.1 则 quad=7.63>1 不可能 ⇒ 你们的 tag 是 raw quad）。但我们的 amp_style_reward 类数字是被 ×0.1 的，两边不要直接比这个数，要比 d。
云上机器现在 Connection refused（183.147.142.40:30542），我拉不到我们自己 run 的 amp_policy_pred/expert_pred 实测值，所以下面只谈结构——需要的话云恢复后我补上。
二、结论：跳跃不是"AMP 没生效"，而是任务奖励里有一条通往跳跃的近路，而 AMP 已经没有权利否决它
跳跃 = 三件事同时成立的必然结果：

风格项退化（你们已证明：d→−0.9，r_quad 塌到 0.09，风格在总回报里只剩 ~28%）；
任务奖励对"双脚同时蹬地"给出几乎满额回报；
奖励/约束里没有任何一项要求"左右交替单支撑"。
⇒ 于是"两脚同步蹬一次"成了最低维、最省事的最优解。

三、根因拆解（按贡献排序）
根因 A（主因）：全部"步态形状"压力都压在 AMP 一项上，而它已经饱和
除了风格项，你们的奖励里没有任何一项是关于步态形状的。速度跟踪、姿态、高度全部只约束躯干，不约束接触时序。
关键推论：判别器残余的那点梯度被"姿态匹配"吃掉了 —— 你们的 AMP 参考集里有矛盾的手臂目标（idle_turn 肘 0.974 vs jog_arc 肘 0.012），判别器要花容量去区分这些静态姿态，而腾空/交替这种动态特征拿不到梯度预算。这正好解释你之前看到的现象：手臂 RMSE 85–117°（在拼命拟合矛盾姿态）、腿 13–35°（没人管它怎么动），于是腿选最省事的同步蹬跳。

根因 B：任务奖励形式对跳跃"不敏感"
track_anchor_linear_velocity 只看躯干速度，指数核在 |Δv| 小时就饱和 → 一次蹬地冲量即可命中命令速度，且平均速度也满足。
没有 feet_slip / 左右 air_time 交替 / 周期相位 / 双支撑时长下限 —— 不存在任何"你必须交替迈步"的约束。
根因 C：判别器输入过强 → 500 轮解完 → 死区（你们已发现，我确认机制）
你们的 AMP 观测是 670 维（≈ 67×10 帧历史，含投影重力 + 29 关节 pos + 29 关节 vel + base 线/角速）；我们是 195 维单帧、无关节角、无重力。差别后果：

含关节角 ⇒ 直接看出膝盖/踝的相位；含投影重力 ⇒ 直接看出倾覆；10 帧历史 ⇒ 且喂的是 (s, s') 对，相邻两帧有 9 帧重叠 ⇒ 判别器几乎是在做"求差分"，任务近乎平凡 ⇒ 必然早饱和。
我们的 195 维单帧是"13 个 link 的相对位姿/体坐标速度"，信息更粗、更难解，判别器留在有用区间的概率更高。

根因 D：命令分布让"跳跃蹭平均速度"够用（我们踩过同一个坑，实证）
我们 v4 把前向范围扩到 2.0 后，后退命令占比从 18.5% → 6.4%，策略立刻退化成**"双脚同时蹬地原地后跳"——症状和你现在描述的一模一样，只是方向不同。
修复 = 加专属"持续直线匀速"命令档**（prob_back=0.10，vx∈(−0.7,−0.4)，vy=0，wz≈0，保持 6–15 s）。命令分布里"必须持续保持直线速度"成为高频事件后，单次蹬跳不够用 ⇒ 真交替步涌现。

根因 F（次要，需你们自查）
PD/动作尺度/decimation 差异：跳跃在"踝/膝刚度偏低、action_scale 偏大"的设定下明显更便宜。若迁移 mjlab 配置时改过 stiffness/damping/action_scale/decimation，这本身就是原因之一。另外：终止条件是否宽松、有无 amp_disable_style_on_stand（我们有，threshold=0.1）。

四、我们这边为什么没变成跳跃（可直接复用的机制）
机制	我们的值	对"跳跃"的作用
track_default_pose_on_stand	w=−1.0, gate_tau=0.25, L1, pos_scale=4.0	低速/立定把关节拉回 KNEES_BENT；跳需要持续大幅屈伸踝/膝 ⇒ 便宜解变贵；同时保证"从站姿起步迈步"
feet_lateral_width	w=−10, sep_floor=0.15, 方向门控, walk_max_lin_vel=1.0	限制双足横向间距下限；并足蹬跳常低于数据集 p1(0.157) ⇒ 被罚
feet_crossing	w=−20, max_cost=0.10	真交叉兜底
body_ang_vel_xy_l2	0.7	落地冲击/躯干不稳被罚
专属"持续直线"命令档	prob_back=0.10 + 6–15 s 重采样	⭐ 治跳跃最直接的一招（见根因 E）
精选参考集	7 个（无 idle，jog 仅 1 个前向）	削弱"腾空=人类"
amp_disable_style_on_stand	True	低速纯 task，避免风格拉扯站姿
⚠️ 给你们计划的一个关键提醒：mirror loss / data augmentation 治不了跳跃。 跳跃本身是左右完全对称的，π(M(o)) ≈ M(π(o)) 对它**完全满意**——mirror loss 治"不对称"，不治"两腿同步"。你们的 PLAN_amp_walk 若把希望放在镜像上，跳跃会原样保留。

五、建议的排查顺序（严格单变量）
先加可观测性（E3）：Loss/amp_policy_pred、Loss/amp_expert_pred，再加两个关键诊断 —— r_amp 落死区（d<−1）的样本比例，以及 r_amp 在"行走相位帧 vs 腾空帧"上的均值差（用接触传感器分箱）。这一步就能直接回答"AMP 到底有没有在管步态"，不用再靠反解推测。
参考集切成纯 walk（第一阶段去掉全部 jog_*）⇒ 抹掉"腾空=人类"先验。
加"持续直线匀速"命令档 + resampling 拉到 6–15 s ⇒ 让"单次蹬跳"不够用。
加几何/姿态约束：track_default_pose_on_stand + feet_lateral_width + feet_crossing。
最后才改奖励形式（quad→log，coef 同步 ÷3.7 保持总压力不变，只变形状）。注意 quad 的问题主要不是"梯度为零"——d=−0.9 时 ∂r/∂d=+0.95 其实很大——而是量级塌陷 3.7 倍（风格占比 28%→61%），以及 d<−1 那部分样本 r≡0完全不罚（最不像人的状态反而免责）。
若仍跳：上步态周期项（左右 air_time 交替要求 / phase clock），这是对同步跳最直接的一击。