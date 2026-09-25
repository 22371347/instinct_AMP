# G1 29-DoF 接口契约（Action Contract + 蒸馏接口）

> 所有数值均由**云端运行中的 env 对象**探针实测得到（`instinct/tools/cloud_probe_action_contract.py`），
> 不是从配置文字反推。核对日期 2026-09-25，任务 `Instinct-Amp-Walk-G1-v0`。
> 修改本文件任何数值前，必须重跑探针确认。

---

## A. Action Contract（10 项，全部已固定）

| # | 项目 | **固定值** | 证据 |
|---|------|-----------|------|
| 1 | 输出维度 | **29**（`action_manager.total_action_dim == 29`） | 探针 §2 |
| 2 | 29 维关节顺序 | **USD/articulation 自然序**（见 A.1，必须逐位固定） | `_joint_names` 实测打印 |
| 3 | 每一维含义 | A.1 的 29 个关节名，**按下标一一对应** | 同上 |
| 4 | action→joint target | **`q_target[i] = q_default[i] + 1.0 · a[i]`** | 实测 `processed_actions == default + scale*action`（atol 1e-5 ✓） |
| 5 | action scale | **1.0**（标量，广播到 29） | `cfg.scale = 1.0`，`_scale = 1.0` |
| 6 | default joint offset | **`use_default_offset=True`**，`offset == asset.data.default_joint_pos` | `allclose` 逐项 **True** ✓（值见 A.2） |
| 7 | action clip / squash | **无 clip、无 tanh**（`clip=None`；动作实测可达 ±29.7） | `cfg.clip = None`；`tracking_action/abs_max = 29.6647` |
| 8 | 单位 | 关节角 **rad**，角速度 rad/s，线速度 m/s，高度 m | isaaclab 约定 |
| 9 | decimation / policy Hz | **`sim.dt=0.005` × `decimation=4` = `step_dt=0.02` → 50 Hz**；episode 20 s = 1000 步 | 探针 §1 |
| 10 | actuator / PD 解释 | **5 组 `ImplicitActuatorCfg`**（PhysX 隐式 PD，`ctrl` = **位置目标**）；kp/kd 见 A.3 | 探针 §3 |

**⛔ 铁律**：这 10 项是策略的对外契约。任何一项变更都会使已有 checkpoint / ONNX / MuJoCo 回放**全部失效**，
必须整条链路（训练 → ONNX 导出 → `policy_normalizer.npz` → MuJoCo 回放 → 后续蒸馏）同步重做。
`joint_names=['.*']` 只表示"全部关节"，其**解析结果**才是真正的顺序（A.1），不要依赖正则字符串。

---

### A.1 29 维关节顺序（`_joint_names` 实测，OMNI_LOCO 顺序）

`cfg.joint_names = ['.*']` 按 articulation 的自然序解析如下（shoulder 先、左右交错）：

| 下标 | 关节名 | 下标 | 关节名 | 下标 | 关节名 |
|----|--------|----|--------|----|--------|
| 0 | `left_shoulder_pitch_joint`  | 10 | `left_elbow_joint`        | 20 | `left_hip_yaw_joint` |
| 1 | `right_shoulder_pitch_joint` | 11 | `right_elbow_joint`       | 21 | `right_hip_yaw_joint` |
| 2 | `waist_pitch_joint`          | 12 | `left_hip_pitch_joint`    | 22 | `left_wrist_yaw_joint` |
| 3 | `left_shoulder_roll_joint`   | 13 | `right_hip_pitch_joint`   | 23 | `right_wrist_yaw_joint` |
| 4 | `right_shoulder_roll_joint`  | 14 | `left_wrist_roll_joint`   | 24 | `left_knee_joint` |
| 5 | `waist_roll_joint`           | 15 | `right_wrist_roll_joint`  | 25 | `right_knee_joint` |
| 6 | `left_shoulder_yaw_joint`    | 16 | `left_hip_roll_joint`     | 26 | `left_ankle_pitch_joint` |
| 7 | `right_shoulder_yaw_joint`   | 17 | `right_hip_roll_joint`    | 27 | `right_ankle_pitch_joint` |
| 8 | `waist_yaw_joint`            | 18 | `left_wrist_pitch_joint`  | 28 | `left_ankle_roll_joint` |
| 9 | `left_hip_pitch_joint`       | 19 | `right_wrist_pitch_joint` | 28 | `right_ankle_roll_joint` |

> 注意腰部 3 个关节被**插在 shoulder 之间**（waist_pitch@2、waist_roll@5、waist_yaw@8），
> 不是"先腿后腰"顺序。这与 npz 参考动作文件里的 `joint_names`（腿先）**不同**，
> 参考动作靠**按名字索引**重排（见 `motion_files/amass_motion.py`），不要手工按位置对齐。

### A.2 default joint position（必须保持兼容）

按 A.1 顺序（`_offset` 实测值，rad）：

```
 0 left_shoulder_pitch   +0.2      10 left_elbow          +0.6     20 left_hip_yaw     0.0
 1 right_shoulder_pitch  +0.2      11 right_elbow         +0.6     21 right_hip_yaw    0.0
 2 waist_pitch            0.0      12 left_hip_pitch      -0.312   22 left_wrist_yaw   0.0
 3 left_shoulder_roll    +0.2      13 right_hip_pitch     -0.312   23 right_wrist_yaw  0.0
 4 right_shoulder_roll   -0.2      14 left_wrist_roll      0.0     24 left_knee       +0.669
 5 waist_roll             0.0      15 right_wrist_roll     0.0     25 right_knee      +0.669
 6 left_shoulder_yaw      0.0      16 left_hip_roll        0.0     26 left_ankle_pitch -0.363
 7 right_shoulder_yaw     0.0      17 right_hip_roll       0.0     27 right_ankle_pitch -0.363
 8 waist_yaw              0.0      18 left_wrist_pitch     0.0     28 left_ankle_roll   0.0
 9 left_hip_pitch        -0.312   19 right_wrist_pitch    0.0
```

`INIT_ROOT_HEIGHT = 0.82` m（与云端 keyframe、MuJoCo keyframe 一致）。

### A.3 actuator / PD（云端 vs MuJoCo **逐项一致**）

云端 5 组 `ImplicitActuatorCfg`；MuJoCo 侧 `instinct/assets/g1_29dof_popsicle_play.xml`
有 29 个 `<position>` 执行器，kp/kv **完全等于**下表：

| 组 | 关节 | kp | kd | armature | effort_limit_sim | velocity_limit_sim |
|----|------|----|----|----------|------------------|--------------------|
| legs_pitch_yaw | `.*_hip_pitch`, `.*_hip_yaw` | 40.179238 | 2.557890 | 0.01017752 | 88 | 32 |
| legs_roll_knee | `.*_hip_roll`, `.*_knee` | 99.098428 | 6.308802 | 0.02510 | 139 | 20 |
| waist_yaw | `waist_yaw` | 40.179238 | 2.557890 | 0.01017752 | 88 | 32 |
| feet | `.*_ankle_pitch`, `.*_ankle_roll` | 28.501246 | 1.814446 | 0.00721945 | 50 | 37 |
| waist | `waist_roll`, `waist_pitch` | 28.501246 | 1.814446 | 0.00721945 | 50 | 37 |
| arms_5020 | shoulder p/r/y, elbow, wrist_roll | 14.25062 | 0.90722 | 0.00361 | 25 | 37 |
| arms_4010 | wrist_pitch, wrist_yaw | 16.77833 | 1.06814 | 0.00425 | 5 | 22 |

推导线（`instinct/tools/g1_params.py` 用同一公式，已核对一致）：
`kp = armature · (2π·10)²`，`kd = 2 × 2.0 × armature · (2π·10)`（10 Hz 自然频率，阻尼比 2.0）。

> ⚠️ **残余差异（已知，尚未消除）**：云端用 `effort_limit_sim`（力矩上限）与 `velocity_limit_sim`；
> MuJoCo MJCF 的 `<position>` 带 `ctrllimited="false"` 且**没有 `forcerange`**，即**无力矩限幅**。
> 做纯运动学/稳定性判断时可接受；要复现极限力矩下的行为（例如跳跃落地大冲击）时必须补 `forcerange`。
> 另：MuJoCo `position` 执行器力 = `kp·(ctrl − q) − kv·q̇`，与 PhysX 隐式 PD 的
> `tau = kp·(q_target − q) − kd·q̇` 同构 ✓。

### A.4 观测侧契约（与动作一一对应，改动同样破坏 checkpoint）

| 组 | 术语顺序与维度 | 合计 |
|----|---------------|------|
| `policy`（策略输入） | `height_scan(527)`, `velocity_command(3)`, `base_lin_vel(3)`, `projected_gravity(3)`, `base_ang_vel(3)`, `joint_pos_rel(29)`, `joint_vel(29)`, `last_action(29)` | **626** |
| `critic`（特权值函数） | `base_lin_vel(3)`, `projected_gravity(3)`, `base_ang_vel(3)`, 命令(3), `joint_pos_rel(29)`, `joint_vel(29)`, `last_action(29)` | **99** |
| AMP×2（判别器） | `(30,)`, `(290,)`, `(290,)`, `(30,)`, `(30,)` = 10 帧 × 67 维 | **670** |

语义（已逐条对源码核实）：
- `height_scan = torso_z − ray_hit_z − 0.5`（offset **默认 0.5**，ObsTerm 未传参）⇒ 平面地形上恒为 `torso_z − 0.5`（torso_z=0.82 时 = 0.32）
- `projected_gravity = Rᵀ·(0,0,−1)` ⇒ 直立 = `(0,0,−1)`
- `base_ang_vel = root_ang_vel_b`、`joint_vel` **均不缩放**；`joint_pos_rel = joint_pos − default_joint_pos`
- `last_action = env.action_manager.action`（**原始动作**，非 processed target）
- 归一化：`policy_normalizer.npz` 独立导出，`(x − mean)/(std + eps)`，ONNX **不含**归一化层

---

## B. 蒸馏接口契约（学长初版代码）

### B.1 教师加载契约（`instinct_rl/algorithms/tppo.py::load_teacher_policy_from_logdir`）

```text
teacher_logdir/                       ← 传目录，不是文件
├── model_<N>.pt                      ← 必须存在；扫描所有 model_*.pt，按 N 取【编号最大】
└── params/agent.yaml                 ← 必须存在（只要 checkpoint 里带 policy_normalizer_state_dict 就会读）
```

- `model_<N>.pt` 内需有 `model_state_dict`；若含 `policy_normalizer_state_dict`，则从
  `params/agent.yaml` 的 `normalizers.policy` 段构造教师 normalizer（`class_name` + 其余 kwargs）。
- 顺序敏感：`state_dict` 与 `teacher_policy.obs_format` 不匹配会抛
  `RuntimeError: Teacher checkpoint ... is incompatible with the configured teacher policy observation format`。
- 选取规则：`model_files.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))[-1]` ⇒ **文件名必须是 `model_<纯整数>.pt`**。

### B.2 学生 env 必须提供的观测组（`vae_ppo_distill.py::act_with_obs_pack`）

| 观测组名 | 作用 | 缺失后果 |
|----------|------|----------|
| `teacher` | **教师网络的输入**，必须逐项复刻教师当年 `policy` 组的**术语名 + 形状 + 拼接顺序** | `KeyError: Teacher observation group 'teacher' is required` |
| `distill_mask` | 单标量 DAgger mask（1=该步用教师动作做标签） | `KeyError: Distillation mask observation group ...` |

教师调用链：`obs_pack["teacher"]` → `get_teacher_actions(teacher_obs, None)`
→ `teacher_policy_normalizer(teacher_obs)` → `teacher_actor_critic.act_inference(...)`。
**教师吃的是 `teacher` 组，不是学生 `critic` 组。**

### B.3 算法配置必填项（`AlgorithmCfg(InstinctRlPpoAlgorithmCfg)`）

```python
class_name = "VaePpoDistill"
teacher_obs_group = "teacher"
distillation_mask_obs_group = "distill_mask"
teacher_logdir = <run 目录绝对路径>          # 或由环境变量注入
teacher_policy_class_name = "ActorCritic"
teacher_policy = dict(
    init_noise_std=1.0,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
    obs_format={"policy": {...术语名: 形状...}, "critic": {...}},
    num_actions=29,
    num_rewards=1,
)
```
`obs_format` **不可省**：`build_teacher_policy()` 直接索引 `teacher_policy_cfg_dict["obs_format"]`。

### B.4 当前路径核对（`g1_amp_walk`）

| 检查项 | 结果 |
|--------|------|
| 原始 run `..._20260925_003514_..._4096/` | ✅ 有 `model_500.pt … model_5000.pt` + `params/agent.yaml` |
| 续训 run `..._20260925_112355_..._from20260925_003514/` | ❌ **目录内没有任何 `model_*.pt`**（只有 `events` / `git` / `params`）→ 若现在把它当 `teacher_logdir` 会 `FileNotFoundError`。续训首次落盘发生在超过 `save_interval=500` 的整倍数后 |
| `params/agent.yaml` | ✅ 有；`policy.class_name=ActorCritic`、`activation=elu`、`normalizers.policy.class_name=EmpiricalNormalization` |
| 网络结构 | ✅ Actor 626→512/256/128→29、Critic 99→512/256/128→1，与 B.3 模板**逐项吻合** |
| 动作空间 | ✅ 29 维、scale 1.0、default offset、clip None（= A 节契约） |
| normalizer | ✅ `EmpiricalNormalization`，与 `policy_normalizer.npz` 对应 |
| env 的 `teacher` / `distill_mask` 观测组 | ❌ **本任务尚未定义**（现有 4 组：`policy`/`critic`/2×AMP） |
| walk 侧教师环境变量 | ❌ 无（vault 用 `PERCEPTIVE_VAULT_TEACHER_LOGDIR`，走路任务没有对应项） |
| walk 侧蒸馏 agent cfg / 任务注册 | ❌ 无（现有蒸馏 cfg 只有 `shadowing/perceptive` 的 Vault 系列） |

**结论**：教师侧（目录/checkpoint/网络结构/动作空间）**已满足**契约；
学生侧还差 4 项须新建——`teacher` 观测组、`distill_mask` 观测组、walk 蒸馏 agent cfg
（含 626 的 `obs_format`）、任务注册 + 教师目录注入方式。

> 实现时照抄模板：
> `instinct_xu/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_vault_vae_distill_cfg.py`
> （`TeacherObsCfg` 继承教师 obs 类、`DistillMaskObsCfg` 单标量、`concatenate_terms=False`）
> 与 `.../agents/instinct_rl_vault_vae_distill_cfg.py`。
