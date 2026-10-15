# P1（pelvis 根）策略部署/回放规格

> 生成时间：2026-10-13　　生成脚本：`write_deploy_spec.py`（表从 `instinct/assets/g1_29dof_rev10_play.xml` 自动读出）
> 用途：把训练好的 P1 ckpt 拿到**另一个 MuJoCo 部署/回放项目**里跑。本文只讲"接口"，不含训练奖励。

---

## 1. 任务与产物标识

| 项 | 值 |
|---|---|
| 训练任务（train） | `Instinct-Amp-Walk-Pelvis-G1-v0` |
| 回放/导出任务（play） | `Instinct-Amp-Walk-Pelvis-G1-Play-v0` |
| run 目录 | `<cloud>/logs/instinct_rl/g1_amp_walk/20261013_162510_p1_pelvis_official_rev10_cold_p1_pelvis_official_rev10` |
| 已核对 ckpt | `model_2000.pt`（iter 2000；reward 73.15 / ep_len 984.91 / `fall_rate=0`） |
| ONNX | `instinct/eval/p1_2000/actor.onnx`（3.6 MB；input `[1,1500]` → output `[1,29]`） |
| 归一化器 | `instinct/eval/p1_2000/policy_normalizer.npz`（`mean/std/eps`，eps=0.01） |
| 训练资产（URDF） | `instinct_xu/source/instinctlab/instinctlab/assets/resources/unitree_g1/official_rev10/g1_29dof_rev_1_0.urdf`（sha256 `c0ae739c640c3e2c00d1bdd8810b5d6e59601487bd1a3995859f9543269ee5c8`） |
| 官方 MJCF（同机器人） | `instinct/assets/official_rev10/g1_29dof_rev_1_0.xml` |
| 本地回放模型（已注入执行器/默认站姿） | `instinct/assets/g1_29dof_rev10_play.xml`（本文件的表由它读出） |

**控制频率 / 仿真步长**：策略 50 Hz，`sim.dt = 0.005 s`，`decimation = 4`（物理 200 Hz）。

---

## 2. 机器人本体（关键事实）

- 根节点 = **`pelvis`**（自由关节在 pelvis 上）。**不是** `torso_link`。
- 出生高度：`pelvis.z = 0.7585 m`（= 默认站姿下的骨盆高度）。
- 默认站姿（`init_state.joint_pos`，与 `--asset rev10` 的 keyframe 逐关节一致）：
  `hip_pitch=-0.312`、`knee=0.669`、`ankle_pitch=-0.363`、`elbow=0.6`、
  `shoulder_roll=左 +0.2 / 右 −0.2`、`shoulder_pitch=0.2`，其余 **0.0**。
- 总质量 **33.3411 kg**（与训练 URDF 完全一致；逐 link 主惯量也一致）。
- 运动学已与训练 URDF 逐 body 对比：**只差恒定 758.5 mm 的基座偏移**（= pelvis 高度），即同一机器人。

---

## 3. 关节顺序（**最容易错的一步**）

**训练侧有两套顺序，别混：**

| 用途 | 顺序 | 说明 |
|---|---|---|
| **策略输入/输出的 29 维排列**（obs 的 `joint_pos_rel/joint_vel/last_action` 段、以及 action 向量） | `PELVIS_JOINT_ORDER`（= 本文表格左列，= IsaacLab articulation 序） | **互换根节点会改变它** ⇒ 旧 torso 资产的顺序 (`ISAACLAB_ORDER`) 在这里是**错的** |
| 写进 **ctrl** 的执行器顺序 | `MUJOCO_ORDER`（腰→左腿→右腿→左臂→右臂） | 官方 MJCF 的执行器声明序 |

> ⚠️⚠️ 官方 MJCF 里 **`qpos/qvel` 的顺序与 `ctrl` 的顺序不同**：
> `qpos/qvel` = URDF 声明序（左腿→右腿→腰→左臂→右臂 = `NPZ_ORDER`），
> `ctrl` = 上面这张表的右列（`MUJOCO_ORDER`）。
> 读关节角时**必须**按模型自己报告的顺序读（`jnt_qposadr` 排序），再**按名字**映射到策略序；
> 写控制时按 `MUJOCO_ORDER` 写。混用 ⇒ 观测关节块整块错位 ⇒ 回放抽搐。

### 3.1 顺序对照表

| # | 策略/观测序（`PELVIS_JOINT_ORDER`） | # | MJCF ctrl 序（`MUJOCO_ORDER`） |
|---|---|---|---|
| 0 | `left_hip_pitch_joint` | 0 | `waist_pitch_joint` |
| 1 | `right_hip_pitch_joint` | 1 | `waist_roll_joint` |
| 2 | `waist_yaw_joint` | 2 | `waist_yaw_joint` |
| 3 | `left_hip_roll_joint` | 3 | `left_hip_pitch_joint` |
| 4 | `right_hip_roll_joint` | 4 | `left_hip_roll_joint` |
| 5 | `waist_roll_joint` | 5 | `left_hip_yaw_joint` |
| 6 | `left_hip_yaw_joint` | 6 | `left_knee_joint` |
| 7 | `right_hip_yaw_joint` | 7 | `left_ankle_pitch_joint` |
| 8 | `waist_pitch_joint` | 8 | `left_ankle_roll_joint` |
| 9 | `left_knee_joint` | 9 | `right_hip_pitch_joint` |
| 10 | `right_knee_joint` | 10 | `right_hip_roll_joint` |
| 11 | `left_shoulder_pitch_joint` | 11 | `right_hip_yaw_joint` |
| 12 | `right_shoulder_pitch_joint` | 12 | `right_knee_joint` |
| 13 | `left_ankle_pitch_joint` | 13 | `right_ankle_pitch_joint` |
| 14 | `right_ankle_pitch_joint` | 14 | `right_ankle_roll_joint` |
| 15 | `left_shoulder_roll_joint` | 15 | `left_shoulder_pitch_joint` |
| 16 | `right_shoulder_roll_joint` | 16 | `left_shoulder_roll_joint` |
| 17 | `left_ankle_roll_joint` | 17 | `left_shoulder_yaw_joint` |
| 18 | `right_ankle_roll_joint` | 18 | `left_elbow_joint` |
| 19 | `left_shoulder_yaw_joint` | 19 | `left_wrist_roll_joint` |
| 20 | `right_shoulder_yaw_joint` | 20 | `left_wrist_pitch_joint` |
| 21 | `left_elbow_joint` | 21 | `left_wrist_yaw_joint` |
| 22 | `right_elbow_joint` | 22 | `right_shoulder_pitch_joint` |
| 23 | `left_wrist_roll_joint` | 23 | `right_shoulder_roll_joint` |
| 24 | `right_wrist_roll_joint` | 24 | `right_shoulder_yaw_joint` |
| 25 | `left_wrist_pitch_joint` | 25 | `right_elbow_joint` |
| 26 | `right_wrist_pitch_joint` | 26 | `right_wrist_roll_joint` |
| 27 | `left_wrist_yaw_joint` | 27 | `right_wrist_pitch_joint` |
| 28 | `right_wrist_yaw_joint` | 28 | `right_wrist_yaw_joint` |

### 3.2 策略向量 → ctrl 的映射（`policy_to_mujoco_index()`）

用法：`ctrl[j] = sum( action[i] * (i == map[j]) )`，即 `ctrl = action[map]`。

| ctrl 槽 i | ctrl 写的关节 | 取策略向量的下标 `policy_to_mujoco_index()[i]` |
|---|---|---|
| 0 | `waist_pitch_joint` | 8 |
| 1 | `waist_roll_joint` | 5 |
| 2 | `waist_yaw_joint` | 2 |
| 3 | `left_hip_pitch_joint` | 0 |
| 4 | `left_hip_roll_joint` | 3 |
| 5 | `left_hip_yaw_joint` | 6 |
| 6 | `left_knee_joint` | 9 |
| 7 | `left_ankle_pitch_joint` | 13 |
| 8 | `left_ankle_roll_joint` | 17 |
| 9 | `right_hip_pitch_joint` | 1 |
| 10 | `right_hip_roll_joint` | 4 |
| 11 | `right_hip_yaw_joint` | 7 |
| 12 | `right_knee_joint` | 10 |
| 13 | `right_ankle_pitch_joint` | 14 |
| 14 | `right_ankle_roll_joint` | 18 |
| 15 | `left_shoulder_pitch_joint` | 11 |
| 16 | `left_shoulder_roll_joint` | 15 |
| 17 | `left_shoulder_yaw_joint` | 19 |
| 18 | `left_elbow_joint` | 21 |
| 19 | `left_wrist_roll_joint` | 23 |
| 20 | `left_wrist_pitch_joint` | 25 |
| 21 | `left_wrist_yaw_joint` | 27 |
| 22 | `right_shoulder_pitch_joint` | 12 |
| 23 | `right_shoulder_roll_joint` | 16 |
| 24 | `right_shoulder_yaw_joint` | 20 |
| 25 | `right_elbow_joint` | 22 |
| 26 | `right_wrist_roll_joint` | 24 |
| 27 | `right_wrist_pitch_joint` | 26 |
| 28 | `right_wrist_yaw_joint` | 28 |

（等价的 `numpy`：`np.array(map, dtype=int)`，`mujoco_order = [PELVIS_JOINT_ORDER.index(n) for n in MUJOCO_ORDER]`）

---

## 4. 动作接口

```python
target_joint_pos[ctrl_order] = default_joint_pos[ctrl_order] + action_scale * action[policy_order->ctrl_order]
# 即：target = default + 1.0 * action   （use_default_offset=True，scale=1.0，无 clip）
```
- `action_scale = 1.0`（**不是** 0.25）；契约断言要求 `scale==1.0` 且 `clip is None`。
- `action` 由策略直接给出（raw action），**不需要** tanh/裁剪；实测 `|action| ≲ 3`。
- 位置执行器：`kp/kd` 见表；**力矩上限必须生效**（表里的 `effort_limit`，训练侧是 `effort_limit_sim`）。
- 每个控制周期：把上面算出的 `target` 保持 `decimation=4` 个物理步（或等价的 0.02 s）。

---

## 5. 观测接口（**1500 维**，构造顺序即下表）

| # | 项 | 单帧 | 帧数 | 小计 | 定义 |
|---|---|---|---|---|---|
| 0 | `height_scan` | 279 | **4** | 1116 | 见 5.1 |
| 1 | `velocity_command` | 3 | **4** | 12 | world/body 系命令 `[vx, vy, wz]`（**给的是 body 系指令**，不加缩放） |
| 2 | `projected_gravity` | 3 | **4** | 12 | `R_pelvis^T · (0,0,-1)`（直立时 ≈ `(0,0,-1)`） |
| 3 | `base_ang_vel` | 3 | **4** | 12 | **pelvis 体系**角速度（MuJoCo 自由关节 `qvel[3:6]`），**不缩放** |
| 4 | `joint_pos_rel` | 29 | **4** | 116 | `q − default_joint_pos`，**策略序**（第 3 节左列），**不缩放** |
| 5 | `joint_vel` | 29 | **4** | 116 | `qdot`，**策略序**，**不缩放**（注意：不是 parkour 那套 ×0.05） |
| 6 | `last_action` | 29 | **4** | 116 | **上一帧 raw action**（策略序；首帧/复位后 = 0） |
| | | | | **1500** | |

- ❌ **没有 `base_lin_vel`**（T1o 起已删除，与 mjlab 对齐；真机也没有）。
- **堆叠规则（关键）**：四项 = 4 帧，**项优先**（同一项的 4 帧连续排列），**项内按时间从旧到新**。
  例如 `joint_pos_rel` 段的 116 维 = `[j0(t-3) … j28(t-3), j0(t-2) … , j0(t) … j28(t)]`。
  （该约定已在本地回放工具里验证：同一布局的 T2 策略能正常行走。）
- **归一化**：`x_norm = (x - mean) / (std + eps)`，`eps=0.01`，作用于**整个 1500 维向量**，
  `mean/std` 取自 `policy_normalizer.npz`。**必须**要归一化，否则策略输入 OOD 会输出疯狂动作。

### 5.1 `height_scan`（279 条射线）
- 传感器挂点 = **`pelvis`**（**不是** torso）。
- 网格：`size = (3.0, 0.8) m`（x 前向 3.0 / y 侧向 0.8），`resolution = 0.1` ⇒ **9 × 31 = 279** 条；
  **中心沿 body x 前移 1.5 m**；`ray_alignment = "yaw"`（网格不随躯干 roll/pitch 倾斜，只跟 yaw）。
- 射线起点在挂点上方 20 m，向下投射到地面；观测量 = `pelvis_z − hit_z − 0.5`（clip 到 ±1）。
- **平地近似（精确）**：所有 279 条 = `pelvis_z − 0.5`。
  出生站姿 ⇒ **0.2585**（实测：训练 normalizer 该段均值 0.2458，本地 0.2585 ✅）。
- ⚠️ 上地形/越障时，279 个值的**排布顺序**必须与 IsaacLab `GridPatternCfg` 一致（行优先、网格中心前移 1.5 m、yaw 对齐）——平地回放不受影响。

---

## 6. 关节驱动参数（与训练侧 `beyondmimic_g1_29dof_actuators` 逐项一致）

| # | 关节 | kp | kd | effort_limit | armature | joint damping | 关节范围(rad) | 默认角(rad) |
|---|---|---|---|---|---|---|---|---|
| 0 | `waist_pitch_joint` | 28.5012 | 1.8144 | 50 | 0.007219450 | 0.000 | [-0.520, +0.520] | +0.000 |
| 1 | `waist_roll_joint` | 28.5012 | 1.8144 | 50 | 0.007219450 | 0.000 | [-0.520, +0.520] | +0.000 |
| 2 | `waist_yaw_joint` | 40.1792 | 2.5579 | 88 | 0.010177520 | 0.000 | [-2.618, +2.618] | +0.000 |
| 3 | `left_hip_pitch_joint` | 40.1792 | 2.5579 | 88 | 0.010177520 | 0.000 | [-2.531, +2.880] | -0.312 |
| 4 | `left_hip_roll_joint` | 99.0984 | 6.3088 | 139 | 0.025101925 | 0.000 | [-0.524, +2.967] | +0.000 |
| 5 | `left_hip_yaw_joint` | 40.1792 | 2.5579 | 88 | 0.010177520 | 0.000 | [-2.758, +2.758] | +0.000 |
| 6 | `left_knee_joint` | 99.0984 | 6.3088 | 139 | 0.025101925 | 0.000 | [-0.087, +2.880] | +0.669 |
| 7 | `left_ankle_pitch_joint` | 28.5012 | 1.8144 | 50 | 0.007219450 | 0.000 | [-0.873, +0.524] | -0.363 |
| 8 | `left_ankle_roll_joint` | 28.5012 | 1.8144 | 50 | 0.007219450 | 0.000 | [-0.262, +0.262] | +0.000 |
| 9 | `right_hip_pitch_joint` | 40.1792 | 2.5579 | 88 | 0.010177520 | 0.000 | [-2.531, +2.880] | -0.312 |
| 10 | `right_hip_roll_joint` | 99.0984 | 6.3088 | 139 | 0.025101925 | 0.000 | [-2.967, +0.524] | +0.000 |
| 11 | `right_hip_yaw_joint` | 40.1792 | 2.5579 | 88 | 0.010177520 | 0.000 | [-2.758, +2.758] | +0.000 |
| 12 | `right_knee_joint` | 99.0984 | 6.3088 | 139 | 0.025101925 | 0.000 | [-0.087, +2.880] | +0.669 |
| 13 | `right_ankle_pitch_joint` | 28.5012 | 1.8144 | 50 | 0.007219450 | 0.000 | [-0.873, +0.524] | -0.363 |
| 14 | `right_ankle_roll_joint` | 28.5012 | 1.8144 | 50 | 0.007219450 | 0.000 | [-0.262, +0.262] | +0.000 |
| 15 | `left_shoulder_pitch_joint` | 14.2506 | 0.9072 | 25 | 0.003609725 | 0.000 | [-3.089, +2.670] | +0.200 |
| 16 | `left_shoulder_roll_joint` | 14.2506 | 0.9072 | 25 | 0.003609725 | 0.000 | [-1.588, +2.252] | +0.200 |
| 17 | `left_shoulder_yaw_joint` | 14.2506 | 0.9072 | 25 | 0.003609725 | 0.000 | [-2.618, +2.618] | +0.000 |
| 18 | `left_elbow_joint` | 14.2506 | 0.9072 | 25 | 0.003609725 | 0.000 | [-1.047, +2.094] | +0.600 |
| 19 | `left_wrist_roll_joint` | 14.2506 | 0.9072 | 25 | 0.003609725 | 0.000 | [-1.972, +1.972] | +0.000 |
| 20 | `left_wrist_pitch_joint` | 16.7783 | 1.0681 | 5 | 0.004250000 | 0.000 | [-1.614, +1.614] | +0.000 |
| 21 | `left_wrist_yaw_joint` | 16.7783 | 1.0681 | 5 | 0.004250000 | 0.000 | [-1.614, +1.614] | +0.000 |
| 22 | `right_shoulder_pitch_joint` | 14.2506 | 0.9072 | 25 | 0.003609725 | 0.000 | [-3.089, +2.670] | +0.200 |
| 23 | `right_shoulder_roll_joint` | 14.2506 | 0.9072 | 25 | 0.003609725 | 0.000 | [-2.252, +1.588] | -0.200 |
| 24 | `right_shoulder_yaw_joint` | 14.2506 | 0.9072 | 25 | 0.003609725 | 0.000 | [-2.618, +2.618] | +0.000 |
| 25 | `right_elbow_joint` | 14.2506 | 0.9072 | 25 | 0.003609725 | 0.000 | [-1.047, +2.094] | +0.600 |
| 26 | `right_wrist_roll_joint` | 14.2506 | 0.9072 | 25 | 0.003609725 | 0.000 | [-1.972, +1.972] | +0.000 |
| 27 | `right_wrist_pitch_joint` | 16.7783 | 1.0681 | 5 | 0.004250000 | 0.000 | [-1.614, +1.614] | +0.000 |
| 28 | `right_wrist_yaw_joint` | 16.7783 | 1.0681 | 5 | 0.004250000 | 0.000 | [-1.614, +1.614] | +0.000 |

参数来源（训练侧）：
- `kp = armature × (2π×10)²`，`kd = 2×2.0×armature×(2π×10)`（`NATURAL_FREQ=10 Hz`，`DAMPING_RATIO=2`）。
- `armature` 常量：`ARMATURE_5020=0.003609725`、`ARMATURE_7520_14=0.010177520`、
  `ARMATURE_7520_22=0.025101925`、`ARMATURE_4010=0.00425`。
- `damping = frictionloss = 0`（**不要**额外加关节阻尼）。

---

## 7. 训练环境配置文件（要看细节时按这条链查）

| 文件 | 作用 |
|---|---|
| `instinct_xu/source/instinctlab/instinctlab/tasks/locomotion/config/g1/amp_walk_pelvis_env_cfg.py` | **P1 臂本体**：`_apply_pelvis()`（换资产/扫描挂点/运动参考/AMP anchor）、`PELVIS_JOINT_ORDER`、`PELVIS_INIT_Z=0.7585`、契约断言 |
| `.../g1/amp_walk_t2_env_cfg.py` | **观测/命令契约**：`T2_HISTORY=4`、`T2_OBSERVATION_TERMS/DIMS`（=1500）、`T2_COMMAND_RANGES=((-0.5,1.0),(-1.0,1.0),(-1.57,1.57))`、`T2_SCAN_RAYS=279` |
| `.../g1/amp_walk_t1o_env_cfg.py` | 4 帧历史 + 删 `base_lin_vel` 的由来（对齐 mjlab `history_ordering="time"`） |
| `.../g1/amp_walk_env_cfg.py` | AMP-Walk 基类（奖励/终止/事件）+ 契约校验 `_validate_teacher_contract` |
| `.../g1/amp_walk_v9_env_cfg.py` | `height_scan` 传感器几何（3.0×0.8 / 0.1 / 前移 1.5 / yaw） |
| `.../g1/omni_loco_env_cfg.py` | 动作项：`JointPositionActionCfg(joint_names=[".*"], scale=1.0, use_default_offset=True)`、`dt=0.005`、`decimation=4` |
| `.../assets/unitree_g1.py` | 资产 `G1_29DOF_REV10_PELVIS_CFG`（官方 URDF、`init_state`、beyondmimic 执行器常量）；训练侧关节序常量 |
| `.../config/g1/agents/amp_walk_pelvis_ppo_cfg.py` | agent（`mirror_joint_names=PELVIS_JOINT_ORDER` 等） |
| `.../config/g1/__init__.py` | 任务注册（`Instinct-Amp-Walk-Pelvis-G1-v0` / `-Play-v0`） |
| `instinct/tools/joint_order.py` | 顺序常量：`PELVIS_ORDER` / `MUJOCO_ORDER` / `NPZ_ORDER` + 映射函数 |
| `instinct/tools/g1_mujoco_play.py` | 本地回放参考实现（`--asset rev10` 预设把 4 件事一次切齐） |

---

## 8. 已知差异 / 坑（本地回放实测）

1. **`qpos/qvel` 序 ≠ `ctrl` 序**（见 §3）——已实测会让观测关节块错位、回放抽搐。
2. **脚底接触几何**：官方 MJCF 的踝部碰撞是 **4 个 r=5 mm 小球**；训练 URDF 的踝部 `<collision>` 是**脚掌网格**（PhysX 用凸包=整只脚掌）⇒ 支撑多边形差近一个数量级。
   在部署项目里建议用**训练口径**（脚掌网格/凸包或等效大接触面），不要直接用那 4 个小球。
3. **不要去"关掉自身碰撞"或关掉网格碰撞来止损**（我试过，无效）。
4. **本地 MuJoCo 的零动作纯 PD 保持站姿会倒**（legacy/rev10 都倒）⇒ 别用"零动作能站住"当作模型正确的判据；站立完全靠策略反馈。
5. **2000 步的 ckpt 在训练里只是"站着"**：`vel_mean≈0.019 m/s`、`vxy_rmse≈0.36` ⇒ 回放看到"不太会走"是策略本身，不是回放错。
6. 本地 MuJoCo 回放在**新资产**上仍有未解释的摔倒（`--asset rev10` 预设），根因未完全定位（剩余嫌疑=PhysX↔MuJoCo 接触/求解器差异）——本项目的部署环境可用来做对照实验。

---

## 9. 接入自检清单（建议按顺序做）

1. **顺序**：按名字注入 `left_hip_pitch=+0.3 / waist_yaw=+0.25 / right_elbow=+0.4`，
   确认 `joint_pos_rel` 第 1 帧里**只有槽 0 / 2 / 22** 非零（respectively `+0.612 / +0.25 / −0.2`，因为要减默认角）。
2. **观测出生帧**：默认站姿下 `height_scan` 全部 = `pelvis_z − 0.5`（≈0.2585）、
   `projected_gravity ≈ (0,0,-1)`、`joint_pos_rel/joint_vel/last_action` 全 0、命令 = 你的指令。
3. **归一化反查（最有效）**：把第 2 步的 1500 维用 `policy_normalizer.npz` 归一化，
   逐段看 `|z|`；如果某一段 `|z|` 明显 >3（尤其 `joint_vel`/`base_ang_vel`/`height_scan`）⇒ 该段的**定义或缩放**错了。
   （本地实测：出生帧 `|z|max = 1.5`、`|z|>3` 为 0 个。）
4. **量级**：出生帧策略输出 `|action| ≲ 0.3`；若出现 ±3 级别的大动作 ⇒ 观测或归一化没接对。
5. **力矩上限**：确认 `effort_limit` 生效（否则 e-stop 前会先摔）。

---

## 10. 本地参考命令（本仓库）

```bash
source /home/leo/miniconda3/etc/profile.d/conda.sh && conda activate instinct

# 观测自检（不开窗）：应打印 1500 维、scan 常数 0.2585、状态序≠ctrl 序
python instinct/tools/g1_mujoco_play.py --asset rev10 --obs-mode amp1500 --selftest-obs

# 回放（需 --asset rev10：一次切齐 xml / 扫描挂载 pelvis / 观测序 p1 / 动作序 p1）
python instinct/tools/g1_mujoco_play.py --asset rev10 \
  --policy instinct/eval/p1_2000/actor.onnx --duration 0

# 顺序映射表
python instinct/tools/joint_order.py --order p1 --dump
```

---

## 11. 本目录（`deploy_replay_spec/`）里的东西

| 文件 | 说明 |
|---|---|
| `DEPLOY_SPEC_P1_pelvis.md` | 本文（接口规格） |
| `joint_order_p1.json` | **机器可读**：两套关节序、`policy_to_ctrl_index`、默认角、kp/kd/effort/armature、观测分段与扫描参数、动作约定 |
| `actor.onnx` + `policy_normalizer.npz` | 策略与归一化器（`model_2000.pt` 导出，已核对与 ckpt 逐元素相同） |
| `g1_29dof_rev10_play.xml` | 本地回放用 MJCF（官方 rev1.0 + 位置执行器 + 默认站姿 keyframe + 地面） |
| `joint_order.py` | 顺序常量/映射函数（可直接 import） |
| `verify_policy_io.py` | **自检脚本**（只需 numpy + onnxruntime，不要 MuJoCo） |

自检脚本用法：

```bash
# ① 出生帧参考：打印参考观测各段取值 / 归一化 |z| / 策略输出动作
python verify_policy_io.py --birth

# ② 检查**你自己项目里**拼出的观测（存成 .npy，形状 (1500,) ）
python verify_policy_io.py --obs my_obs.npy
```

判定标准：`|z|max` 应 ≲ 3（>3 的那一段就是定义/缩放接错了）；出生帧策略输出 `|action| ≲ 0.3`。
本仓库实测出生帧：`|z|max = 1.39`、`|z|>3` 的维度 **0/1500**、`|action|max = 0.203`
（与完整 MuJoCo 回放工具的首帧输出逐位一致，可交叉验证）。

---

## 12. "高频 / 打到限位抽搐" 的排查清单（按概率排序）

按经验，这类现象几乎都出在下面 6 条，逐条核对：

| # | 检查 | 正确值 | 错的典型现象 |
|---|---|---|---|
| 1 | **关节序**：策略序 vs ctrl 序 vs qpos 序 三套别混 | 见 §3 | 每个关节拿到别人的指令 ⇒ 乱摆、打到限位 |
| 2 | **default offset 加了吗**（`target = default + scale*action`） | `use_default_offset=True` | 不加 ⇒ 所有关节被命令到 0（伸直/僵住） |
| 3 | **action_scale** | **1.0（全局）** | 用旧 parkour/beyondmimic 的 per-joint `0.25*effort/kp` ⇒ 命令放大 2~13 倍 ⇒ **直接打到限位** |
| 4 | **归一化只做一次** | ONNX 输入 = `(x-mean)/(std+eps)` | 做两次/漏做 ⇒ 输出 ±3 级疯狂动作 |
| 5 | **推理频率** | **50 Hz**（每 4 个 200 Hz 物理步或 10 个 500 Hz 步推理一次） | 每个物理步都推理 ⇒ 等效增益/相位全错 |
| 6 | **力矩限幅** | 见表 `effort_limit` | 不限幅 ⇒ 高频抖振、打限位 |

数组要不要按顺序重排（**逐个回答**）：

| 数组 | 是否要重排 | 权威来源 |
|---|---|---|
| `default_pose` / `offset`（= `default_joint_pos`） | ✅ **必须**：它是"每个 ctrl 槽的目标偏置"，必须按**关节名**映射到该仿真器的 ctrl 序 | 训练 `init_state.joint_pos`（§2） |
| `kp` / `kd` / `armature` / `effort_limit` | ⚠️ 取决于存放方式：写在**模型文件**里（actuator 的 kp/kv/forcerange）⇒ 每关节自带，**不用**重排；写成**29 维平铺数组**且沿用了旧资产顺序 ⇒ **必须**按名重排 | 训练 `beyondmimic_g1_29dof_actuators`（§6） |
| `action_scale` | ❌ 不用重排 | 全局 **1.0** |

时序（实测证据）：

```
训练侧 ：sim.dt = 0.005 s（200 Hz）× decimation 4  ⇒ 策略 step_dt = 0.02 s = 50 Hz
         契约断言：math.isclose(self.step_dt, 0.02)
本地工具：model.opt.timestep = 0.002 s（500 Hz）× decimation 10 ⇒ 0.02 s = 50 Hz
         [run]  100 policy steps @ 50.0 Hz (decimation=10, dt=0.002)
         [rate] 策略推理 100 次 / 仿真 2.00 s = 50.00 Hz  ✅
```
⇒ "200 Hz" 只是**物理子步**频率；**策略推理两边都是 50 Hz**，没有 500 Hz 推理的问题。

### 12.1 现成的两个自检命令（本仓库 `g1_mujoco_play.py` 内置）

```bash
# ① 默认站姿 + ctrl/策略序对照表（无窗；零重力 ⇒ 稳态误差应为 0）
python instinct/tools/g1_mujoco_play.py --asset rev10 --print-ctrl-table
#    实测：最大跟踪误差 = 0.00 mrad、根高度 0.7585 ⇒ ctrl↔策略序映射正确 ✅
#    加 --pose-check-gravity 可在带重力下看（肩/肘会因自重下垂 0.1~0.2 rad，属正常伺服误差）

# ② 逐关节点名"肉眼"校验（开窗；默认零重力、机器人悬在原地不会倒）
python instinct/tools/g1_mujoco_play.py --asset rev10 --pose-check
#    共 59 个姿势：默认站姿 + 每个关节 ±0.4 rad 各一次，终端打印当前关节名/策略序下标，
#    看窗口里**哪个部位在动**即可确认关节序（想带重力加 --pose-check-gravity）

# ③ 资产身份（“我跑的到底是哪套模型”）
python instinct/tools/g1_mujoco_play.py --asset rev10 --fingerprint
#    ⇒ 文件 sha256 + 根 body 名 + nq/nv/nu + 总质量；两套资产总质量恰好都是 33.3411 kg，
#      **根 body 名**（pelvis vs torso_link）才能区分

# ④ 与训练 URDF 逐 link/逐 body 核对（证明就是 g1_29dof_rev_1_0）
python instinct/tools/g1_mujoco_play.py --asset rev10 --verify-model
#    实测：同名 link 质量最大差 1.0 g（0.0128%，XML 合并了 head/logo）、
#          默认站姿与探针姿势下 29 个同名 body 位置差 **0.000 mm** ✅
```

### 12.2 云端对照（同一 ckpt 在 IsaacLab 里跑录像）

```bash
# 云端（不干扰训练，单环境，20 s 视频）
bash /root/cloud_run_bg.sh "./play.sh --task Instinct-Amp-Walk-Pelvis-G1-Play-v0 \
  --load_run <run 目录名> --checkpoint model_2000.pt \
  --num_envs 1 --device cuda:0 --headless --video --video_length 1000" /root/video_p1_2000.log
# 产物：<run>/videos/play/model_<step>-step-0.mp4
```
判定方法：云端视频里机器人**稳站/行走**而本地抽搯 ⇒ 问题在本地仿真侧；
若云端视频**也抽搯/打限位** ⇒ 问题在策略/观测侧（而不在本地回放）。
