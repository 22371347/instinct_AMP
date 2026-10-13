# AMP-Walk 各臂**实际生效**奖励表（自动生成）

> 由 `instinct/tools/audit_reward_table.py` **实例化 cfg 后 dump** ⇒
> 表中是运行期真正生效的值（含所有继承层覆盖），**不是源码字面值**。
> 重新生成方式见脚本 docstring。

匹配 `* `：共 65 个任务。

### `Instinct-Amp-Walk-G1-v0`

- **非零 21** / 存活 21 / 已删除(None) 0
- run_name: `G1AmpWalkPlane_obs626_directional_rew5`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_base_height` | +1.5 | std=0.3 | `track_base_height_exp` |
| `track_lin_vel_xy_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `track_ang_vel_z_exp` | +1.2 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `dof_pos_limits` | -1 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_air_time` | +1 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_default_pose_on_stand` | -1 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `stand_still` | -0.8 | command_name=base_velocity | `stand_still` |
| `body_ang_vel_xy` | +0.7 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_slide` | -0.25 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `lin_vel_z_l2` | -0.1 | — | `lin_vel_z_l2` |
| `action_rate_l2` | -0.05 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -2e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `1.0` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.05` |
| `algorithm.disable_style_on_stand` | `False` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.1` |
| `algorithm.discriminator_reward_type` | `'quad'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `False` |


### `Instinct-Amp-Walk-T1a-G1-v0`

- **非零 16** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkT1aMVPPlane_obs378_rew16_t0equiv_sym`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `dof_pos_limits` | -1 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `feet_air_time` | +1 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_lin_vel_xy_exp` | +1 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `lin_vel_z_l2` | -0.1 | — | `lin_vel_z_l2` |
| `action_rate_l2` | -0.05 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -2e-07 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `body_ang_vel_xy` | +0 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_crossing` | +0 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | +0 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `track_base_height` | +0 | std=0.2 | `track_base_height_exp` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `4.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `1.0` |
| `algorithm.use_mirror_loss` | `True` |


<details><summary>cfg 自检（关键行）</summary>

```
[T1a][REWARD WHITELIST] 字段共 22 项：已删除(None) 1 项 / 存活 21 项（非零 16 项）
```

</details>

### `Instinct-Amp-Walk-T1a2-G1-v0`

- **非零 16** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkT1a2MVPPlane_obs378_rew16_t0impl_sym`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `dof_pos_limits` | -1 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_air_time` | +1 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_lin_vel_xy_exp` | +1 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `lin_vel_z_l2` | -0.1 | — | `lin_vel_z_l2` |
| `action_rate_l2` | -0.05 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -2e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `body_ang_vel_xy` | +0 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_crossing` | +0 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | +0 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `track_base_height` | +0 | std=0.2 | `track_base_height_exp` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[256, 128, 128]` |
| `policy.critic_hidden_dims` | `[256, 128, 128]` |
| `policy.init_noise_std` | `1.0` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `1e-12` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `4.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `1.0` |
| `algorithm.use_mirror_loss` | `True` |


<details><summary>cfg 自检（关键行）</summary>

```
[T1a][REWARD WHITELIST] 字段共 22 项：已删除(None) 1 项 / 存活 21 项（非零 16 项）
[T1a2][REWARD IMPL] 已对齐为 T0 的实现（func + 作用域）：
```

</details>

### `Instinct-Amp-Walk-T1b-G1-v0`

- **非零 16** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkT1a2MVPPlane_obs378_rew16_t0impl_sym`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `dof_pos_limits` | -1 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_air_time` | +1 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_lin_vel_xy_exp` | +1 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `lin_vel_z_l2` | -0.1 | — | `lin_vel_z_l2` |
| `action_rate_l2` | -0.05 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -2e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `body_ang_vel_xy` | +0 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_crossing` | +0 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | +0 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `track_base_height` | +0 | std=0.2 | `track_base_height_exp` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[256, 128, 128]` |
| `policy.critic_hidden_dims` | `[256, 128, 128]` |
| `policy.init_noise_std` | `1.0` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `1e-12` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `4.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


<details><summary>cfg 自检（关键行）</summary>

```
[T1a][REWARD WHITELIST] 字段共 22 项：已删除(None) 1 项 / 存活 21 项（非零 16 项）
[T1a2][REWARD IMPL] 已对齐为 T0 的实现（func + 作用域）：
```

</details>

### `Instinct-Amp-Walk-T1c-G1-v0`

- **非零 16** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkT1a2MVPPlane_obs378_rew16_t0impl_sym`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `dof_pos_limits` | -1 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_air_time` | +1 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_lin_vel_xy_exp` | +1 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `lin_vel_z_l2` | -0.1 | — | `lin_vel_z_l2` |
| `action_rate_l2` | -0.05 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -2e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `body_ang_vel_xy` | +0 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_crossing` | +0 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | +0 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `track_base_height` | +0 | std=0.2 | `track_base_height_exp` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[256, 128, 128]` |
| `policy.critic_hidden_dims` | `[256, 128, 128]` |
| `policy.init_noise_std` | `1.0` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `1e-12` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `4.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.9` |
| `algorithm.use_mirror_loss` | `True` |


<details><summary>cfg 自检（关键行）</summary>

```
[T1a][REWARD WHITELIST] 字段共 22 项：已删除(None) 1 项 / 存活 21 项（非零 16 项）
[T1a2][REWARD IMPL] 已对齐为 T0 的实现（func + 作用域）：
```

</details>

### `Instinct-Amp-Walk-T1e-G1-v0`

- **非零 16** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkT1eMVPPlane_obs378_rew16_fullbodylimit_sym`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `dof_pos_limits` | -1 | asset_cfg=(robot|joints=['.*']) | `joint_pos_limits_margin` |
| `feet_air_time` | +1 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_lin_vel_xy_exp` | +1 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `lin_vel_z_l2` | -0.1 | — | `lin_vel_z_l2` |
| `action_rate_l2` | -0.05 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -2e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `body_ang_vel_xy` | +0 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_crossing` | +0 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | +0 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `track_base_height` | +0 | std=0.2 | `track_base_height_exp` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[256, 128, 128]` |
| `policy.critic_hidden_dims` | `[256, 128, 128]` |
| `policy.init_noise_std` | `1.0` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `1e-12` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `4.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


<details><summary>cfg 自检（关键行）</summary>

```
[T1a][REWARD WHITELIST] 字段共 22 项：已删除(None) 1 项 / 存活 21 项（非零 16 项）
[T1a2][REWARD IMPL] 已对齐为 T0 的实现（func + 作用域）：
[T1e][LIMIT REWARD] dof_pos_limits 已换回**全关节**软边距版：
```

</details>

### `Instinct-Amp-Walk-T1f-G1-v0`

- **非零 18** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkT1fMVPPlane_obs378_rew21_mjlabalign_sym`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `track_lin_vel_xy_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `track_ang_vel_z_exp` | +1.2 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `feet_air_time` | +1 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `lin_vel_z_l2` | -0.1 | — | `lin_vel_z_l2` |
| `action_rate_l2` | -0.05 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -2e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `body_ang_vel_xy` | +0 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `track_base_height` | +0 | std=0.2 | `track_base_height_exp` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `1.0` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.05` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `4.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.9` |
| `algorithm.use_mirror_loss` | `True` |


<details><summary>cfg 自检（关键行）</summary>

```
[T1a][REWARD WHITELIST] 字段共 22 项：已删除(None) 1 项 / 存活 21 项（非零 16 项）
[T1a2][REWARD IMPL] 已对齐为 T0 的实现（func + 作用域）：
[T1f][MJLAB ALIGN] 已恢复/校准的奖励项（回读权重）：
[T1f][MJLAB ALIGN] 非零权重项合计 = 18
```

</details>

### `Instinct-Amp-Walk-T1i-G1-v0`

- **非零 16** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkT1a2MVPPlane_obs378_rew16_t0impl_sym`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `dof_pos_limits` | -1 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_air_time` | +1 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_lin_vel_xy_exp` | +1 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `lin_vel_z_l2` | -0.1 | — | `lin_vel_z_l2` |
| `action_rate_l2` | -0.05 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -2e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `body_ang_vel_xy` | +0 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_crossing` | +0 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | +0 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `track_base_height` | +0 | std=0.2 | `track_base_height_exp` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[256, 128, 128]` |
| `policy.critic_hidden_dims` | `[256, 128, 128]` |
| `policy.init_noise_std` | `1.0` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `1e-12` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `4.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.975` |
| `algorithm.use_mirror_loss` | `True` |


<details><summary>cfg 自检（关键行）</summary>

```
[T1a][REWARD WHITELIST] 字段共 22 项：已删除(None) 1 项 / 存活 21 项（非零 16 项）
[T1a2][REWARD IMPL] 已对齐为 T0 的实现（func + 作用域）：
```

</details>

### `Instinct-Amp-Walk-T1j-G1-v0`

- **非零 14** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkT1jMVPPlane_obs378_rew14_noconstantpull`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `dof_pos_limits` | -1 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_air_time` | +1 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_lin_vel_xy_exp` | +1 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `lin_vel_z_l2` | -0.1 | — | `lin_vel_z_l2` |
| `action_rate_l2` | -0.05 | — | `action_rate_l2` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -2e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `body_ang_vel_xy` | +0 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_crossing` | +0 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | +0 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_knee` | +0 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | +0 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `track_base_height` | +0 | std=0.2 | `track_base_height_exp` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[256, 128, 128]` |
| `policy.critic_hidden_dims` | `[256, 128, 128]` |
| `policy.init_noise_std` | `1.0` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `1e-12` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `4.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.975` |
| `algorithm.use_mirror_loss` | `True` |


<details><summary>cfg 自检（关键行）</summary>

```
[T1a][REWARD WHITELIST] 字段共 22 项：已删除(None) 1 项 / 存活 21 项（非零 16 项）
[T1a2][REWARD IMPL] 已对齐为 T0 的实现（func + 作用域）：
[T1j][NO CONSTANT PULL] 解除"相对夹角的常数拉力"：
[T1j][NO CONSTANT PULL] 守卫通过；非零奖励项 = 14（T1a2/T1i 为 16，本臂 -2 ⇒ 14）
```

</details>

### `Instinct-Amp-Walk-T1k-G1-v0`

- **非零 17** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkT1kMVPPlane_obs378_rew17_height_footsmall`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -2 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `track_base_height` | +1.5 | std=0.2 | `track_base_height_exp` |
| `dof_pos_limits` | -1 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_air_time` | +1 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `feet_lateral_width` | -1 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_lin_vel_xy_exp` | +1 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `lin_vel_z_l2` | -0.1 | — | `lin_vel_z_l2` |
| `action_rate_l2` | -0.05 | — | `action_rate_l2` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -2e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `body_ang_vel_xy` | +0 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_knee` | +0 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | +0 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[256, 128, 128]` |
| `policy.critic_hidden_dims` | `[256, 128, 128]` |
| `policy.init_noise_std` | `1.0` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `1e-12` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `4.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.975` |
| `algorithm.use_mirror_loss` | `True` |


<details><summary>cfg 自检（关键行）</summary>

```
[T1a][REWARD WHITELIST] 字段共 22 项：已删除(None) 1 项 / 存活 21 项（非零 16 项）
[T1a2][REWARD IMPL] 已对齐为 T0 的实现（func + 作用域）：
[T1k][RESTORE MJLAB TERMS] 相对 T1j 恢复三项机理：
[T1k][RESTORE MJLAB TERMS] 守卫通过；非零奖励项 = 17（预期 17）
```

</details>

### `Instinct-Amp-Walk-T1l-G1-v0`

- **非零 17** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkT1lMVPPlane_obs378_rew17_linvel1p5_hstd0p3`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -2 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `track_base_height` | +1.5 | std=0.3 | `track_base_height_exp` |
| `track_lin_vel_xy_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `dof_pos_limits` | -1 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_air_time` | +1 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `feet_lateral_width` | -1 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `lin_vel_z_l2` | -0.1 | — | `lin_vel_z_l2` |
| `action_rate_l2` | -0.05 | — | `action_rate_l2` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -2e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `body_ang_vel_xy` | +0 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_knee` | +0 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | +0 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[256, 128, 128]` |
| `policy.critic_hidden_dims` | `[256, 128, 128]` |
| `policy.init_noise_std` | `1.0` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `1e-12` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `4.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.975` |
| `algorithm.use_mirror_loss` | `True` |


<details><summary>cfg 自检（关键行）</summary>

```
[T1a][REWARD WHITELIST] 字段共 22 项：已删除(None) 1 项 / 存活 21 项（非零 16 项）
[T1a2][REWARD IMPL] 已对齐为 T0 的实现（func + 作用域）：
[T1k][RESTORE MJLAB TERMS] 相对 T1j 恢复三项机理：
[T1k][RESTORE MJLAB TERMS] 守卫通过；非零奖励项 = 17（预期 17）
[T1l][REGRESS TO V33/MJLAB] 把两项历史偏离改回 V33 / AMP_mjlab 原值：
[T1l][REGRESS TO V33/MJLAB] 守卫通过；非零奖励项 = 17（预期 17）
```

</details>

### `Instinct-Amp-Walk-V10-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV10MVPPlane_obs378_rew21_explore_aug_cmdLT`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `feet_air_time` | +2 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_swing_time_penalty` | -0.5 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `track_default_pose_on_stand` | -0.3 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.6` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.7` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.03` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.25` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V10b-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV10bMVPPlane_obs378_rew21_explore_aug_cmd3_clip2`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `feet_air_time` | +2 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_swing_time_penalty` | -0.5 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `track_default_pose_on_stand` | -0.3 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.6` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.7` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.03` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.25` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V11-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV11MVPPlane_obs378_rew21_airtime8_ar0.06_bigstep_fromV10b7500`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +8 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.6 | `feet_air_time_positive_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_swing_time_penalty` | -0.5 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `track_default_pose_on_stand` | -0.3 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.5` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.5` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V12-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV12MVPPlane_obs378_rew21_amp2frame_standpose1.0_fromV11`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +8 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.6 | `feet_air_time_positive_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -1 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_swing_time_penalty` | -0.5 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.5` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.5` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V13-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV13MVPPlane_obs378_rew21_amp2frame_cmd4_clip5_omnidir`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +8 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.6 | `feet_air_time_positive_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -1 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_swing_time_penalty` | -0.5 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.5` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.5` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V14-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV14MVPPlane_obs378_rew21_amp2frame_standpose0.3_limitmargin_fromV11`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +8 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.6 | `feet_air_time_positive_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_swing_time_penalty` | -0.5 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `track_default_pose_on_stand` | -0.3 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.25` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V15-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV15MVPPlane_obs378_rew21_amp2frame_style_x4_c1.0_fromV11`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +8 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.6 | `feet_air_time_positive_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_swing_time_penalty` | -0.5 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `track_default_pose_on_stand` | -0.3 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V16-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV16MVPPlane_obs378_rew21_amp10frame_style_c1.0_fromV11`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +8 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.6 | `feet_air_time_positive_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_swing_time_penalty` | -0.5 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `track_default_pose_on_stand` | -0.3 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V17-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV17MVPPlane_obs378_rew21_amp10frame_flat2.0_airtime4.0_scratch`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.6 | `feet_air_time_positive_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_swing_time_penalty` | -0.5 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `track_default_pose_on_stand` | -0.3 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V18-G1-v0`

- **非零 19** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV18MVPPlane_obs378_rew21_amp10frame_flat2.0_airband_ar0.10_5clip_omnidir_scratch`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3 | `feet_air_time_target_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `track_default_pose_on_stand` | -0.3 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `action_rate_l2` | -0.1 | — | `action_rate_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V19a-G1-v0`

- **非零 19** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV19aMVPPlane_obs378_rew21_amp10frame_flat2.0_airband_soft0.4_ar0.10_5clip_omnidir`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.4 | `feet_air_time_target_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `track_default_pose_on_stand` | -0.3 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `action_rate_l2` | -0.1 | — | `action_rate_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V19b-G1-v0`

- **非零 19** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV19bMVPPlane_obs378_rew21_amp10frame_flat2.0_airband_exptau0.08_ar0.10_5clip_omnidir`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08 | `feet_air_time_target_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `track_default_pose_on_stand` | -0.3 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `action_rate_l2` | -0.1 | — | `action_rate_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.125` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V19c-G1-v0`

- **非零 19** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV19cMVPPlane_obs378_rew21_amp10frame_flat2.0_airband_exptau0.08_motiongate0.15_ar0.06_standpose0.5_5clip_omnidir`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.15, motion_ang_scale=0.5 | `feet_air_time_target_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `track_default_pose_on_stand` | -0.5 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.125` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V19d-G1-v0`

- **非零 19** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV19dMVPPlane_obs378_rew21_amp10frame_flat2.0_airband_exptau0.08_motiongate0.15_standpen0.5_ar0.06_standpose0.5_5clip_omnidir`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.15, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03 | `feet_air_time_target_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `track_default_pose_on_stand` | -0.5 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.125` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V2-G1-v0`

- **非零 12** / 存活 12 / 已删除(None) 9
- run_name: `G1AmpWalkV2Plane_obs626_rew12_alignAMP`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_base_height` | +1.5 | std=0.3 | `track_base_height_exp` |
| `track_lin_vel_xy_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `track_ang_vel_z_exp` | +1.2 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_default_pose_on_stand` | -1 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.7 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_slide` | -0.25 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `action_rate_l2` | -0.01 | — | `action_rate_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |

已删除(None)：`dof_torques_l2`、`feet_air_time`、`flat_orientation_l2`、`joint_deviation_arms`、`joint_deviation_hip`、`joint_deviation_knee`、`joint_deviation_torso`、`lin_vel_z_l2`、`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `1.0` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.05` |
| `algorithm.disable_style_on_stand` | `False` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.1` |
| `algorithm.discriminator_reward_type` | `'quad'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `False` |


### `Instinct-Amp-Walk-V20-G1-v0`

- **非零 19** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV20MVPPlane_obs378_rew21_amp13body2frame_flat2.0_airband_exptau0.08_motiongate0.15_standpen0.5_ar0.06_standpose0.5_5clip_omnidir`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.15, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03 | `feet_air_time_target_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `track_default_pose_on_stand` | -0.5 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.125` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V21-G1-v0`

- **非零 19** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV21MVPPlane_obs378_rew21_amp13body8frame_flat2.0_airband_exptau0.08_motiongate0.15_standpen0.5_ar0.06_standpose0.5_5clip_omnidir`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.15, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03 | `feet_air_time_target_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `track_default_pose_on_stand` | -0.5 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.125` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V22-G1-v0`

- **非零 19** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV21MVPPlane_obs378_rew21_amp13body8frame_flat2.0_airband_exptau0.08_motiongate0.15_standpen0.5_ar0.06_standpose0.5_5clip_omnidir`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.15, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03 | `feet_air_time_target_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `track_default_pose_on_stand` | -0.5 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'normalized'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.125` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU', 'normalizer_class_name': 'EmpiricalNormalization', 'norm…` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V23-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV23MVPPlane_obs378_rew21_amp13body8frame_white_progress0.35_posdead0.8_5clip_omnidir`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'normalized'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.125` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU', 'normalizer_class_name': 'EmpiricalNormalization', 'norm…` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V24-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV24MVPPlane_obs378_rew21_amp13body8frame_white_progress0.35_posdead0.8_accall5clip_omnidir`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'normalized'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.125` |
| `algorithm.clip_max_std` | `0.6` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU', 'normalizer_class_name': 'EmpiricalNormalization', 'norm…` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V26-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV26MVPPlane_obs378_rew21_amp13body8frame_white_progress0.35_posdead0.8_accall_sigma0.20`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'normalized'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.125` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU', 'normalizer_class_name': 'EmpiricalNormalization', 'norm…` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V27-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV27MVPPlane_obs378_rew21_amp13body8frame_nodrop_white_sigma0.20`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.125` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V28-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV28MVPPlane_obs378_rew21_amp13body8frame_nodrop_white_sigma0.20_ampcoef4`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.125` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `4.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V29-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV29MVPPlane_obs378_rew21_amp13body2frame_nodrop_white_sigma0.20_mjlabalign`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V3-G1-v0`

- **非零 14** / 存活 14 / 已删除(None) 7
- run_name: `G1AmpWalkV3Plane_obs626_rew14_ar0.02_ns0.3`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_base_height` | +1.5 | std=0.3 | `track_base_height_exp` |
| `track_lin_vel_xy_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `track_ang_vel_z_exp` | +1.2 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_default_pose_on_stand` | -1 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.7 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_slide` | -0.25 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |

已删除(None)：`dof_torques_l2`、`feet_air_time`、`joint_deviation_arms`、`joint_deviation_hip`、`joint_deviation_knee`、`lin_vel_z_l2`、`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.3` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.015` |
| `algorithm.disable_style_on_stand` | `False` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.1` |
| `algorithm.discriminator_reward_type` | `'quad'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `False` |


### `Instinct-Amp-Walk-V30-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV30MVPPlane_obs378_rew21_amp13body2frame_nodrop_white_sigma0.20_mjlabdisc`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `flat_orientation_l2` | -2 | — | `flat_orientation_l2` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V31-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV31MVPPlane_obs378_rew21_amp13body2frame_nodrop_white_sigma0.20_flat4`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `flat_orientation_l2` | -4 | — | `flat_orientation_l2` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V32B-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV32BMVPPlane_obs378_rew21_amp13body2frame_white_bothside_flat4`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `flat_orientation_l2` | -4 | — | `flat_orientation_l2` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V32C-G1-v0`

- **非零 19** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV32CMVPPlane_obs378_rew19_amp13body2frame_white_bothside_flat4_noscaffold`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `flat_orientation_l2` | -4 | — | `flat_orientation_l2` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `feet_air_time` | +2 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V33ACold-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV33AColdMVPPlane_obs378_rew21_amp13body2frame390_normnone_rawgp_side015_nodisc_fork11500`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `flat_orientation_l2` | -4 | — | `flat_orientation_l2` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V33S-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV33SMVPPlane_obs378_rew21_amp13body2frame396gra_white_normgp_side015_fork11500`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `flat_orientation_l2` | -4 | — | `flat_orientation_l2` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'normalized'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU', 'normalizer_class_name': 'EmpiricalNormalization', 'norm…` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V33S1-G1-v0`

- **非零 19** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV33S1MVPPlane_obs378_rew19_amp13body2frame396gra_white_normgp_side015_noscaffold_fork11500`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `flat_orientation_l2` | -4 | — | `flat_orientation_l2` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `feet_air_time` | +2 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'normalized'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU', 'normalizer_class_name': 'EmpiricalNormalization', 'norm…` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V33S2-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV33S2MVPPlane_obs378_rew21_amp13body2frame396gra_white001_normgp_side015_fork11500`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `feet_air_time` | +4 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `flat_orientation_l2` | -4 | — | `flat_orientation_l2` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'normalized'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU', 'normalizer_class_name': 'EmpiricalNormalization', 'norm…` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V33S3-G1-v0`

- **非零 19** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV33S3MVPPlane_obs378_rew21_amp13body2frame396gra_nowhite_rawgp_side015_weakened_fork11500`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `flat_orientation_l2` | -4 | — | `flat_orientation_l2` |
| `feet_air_time` | +2 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `track_lin_vel_xy_exp` | +2 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V33S4-G1-v0`

- **非零 18** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV33S4MVPPlane_obs378_rew20_amp13body2frame396gra_nowhite_rawgp_noairtime_boostangvel_fork12500`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `flat_orientation_l2` | -4 | — | `flat_orientation_l2` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_air_time` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.7` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V33S5-G1-v0`

- **非零 19** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV33S5MVPPlane_obs378_rew21_amp13body2frame396gra_nowhite_rawgp_side015_air1_boosttrack_fork12500`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `flat_orientation_l2` | -4 | — | `flat_orientation_l2` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `feet_air_time` | +1 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V33S6-G1-v0`

- **非零 18** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV33S6MVPPlane_obs378_rew20_amp13body2frame396gra_nowhite_rawgp_noairtime_ampcoef4_fork12500`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `flat_orientation_l2` | -4 | — | `flat_orientation_l2` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_air_time` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `4.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.7` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V33S7-G1-v0`

- **非零 19** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV33S7MVPPlane_obs378_rew21_amp13body2frame396gra_nowhite_rawgp_side015_air1_mjlabdisc_fork13500`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `flat_orientation_l2` | -4 | — | `flat_orientation_l2` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `feet_air_time` | +1 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `1.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V33S8-G1-v0`

- **非零 19** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV33S8MVPPlane_obs378_rew21_amp13body2frame396gra_nowhite_rawgp_side015_air1_ampcoef4_adaptive_fork13500`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `flat_orientation_l2` | -4 | — | `flat_orientation_l2` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_axis_diag` |
| `dof_pos_limits` | -1.5 | asset_cfg=(robot|joints=['.*']), margin=0.15, power=2.0 | `joint_pos_limits_margin` |
| `track_ang_vel_z_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `feet_air_time` | +1 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), good_lo=0.4, good_hi=1.0, soft=0.3, tau=0.08, motion_threshold=0.0, motion_ang_scale=0.5, stand_penalty=0.5, stand_min_d=0.03, progress_kappa=0.35, progress_tau=0.3, progress_omega_kappa=0.3 | `feet_air_time_target_biped` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_default_pose_on_stand` | -0.8 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.8, gate_tau=0.25, pos_joint_weights={waist_yaw_joint:3.0, left_knee_joint:2.0, left_ankle_pitch_joint:2.0, left_ankle_roll_joint:2.0, right_knee_joint:2.0, right_ankle_pitch_joint:2.0, right_ankle_roll_joint:2.0} (其余 6 项=1.5), pos_joint_tol={共 29 项}, pos_tol_default=0.08, vel_joint_tol={共 29 项}, vel_tol_default=0.5 | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.06 | — | `action_rate_l2` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -1e-06 | asset_cfg=(robot|joints=.*) | `joint_acc_l2` |
| `feet_swing_time_penalty` | +0 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.35` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.2` |
| `algorithm.clip_max_std_schedule` | `300` |
| `algorithm.clip_min_std` | `0.02` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `4.0` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V4-G1-v0`

- **非零 14** / 存活 14 / 已删除(None) 7
- run_name: `G1AmpWalkV4Plane_obs626_rew14_ar0.02_ns0.3_mirror`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_base_height` | +1.5 | std=0.3 | `track_base_height_exp` |
| `track_lin_vel_xy_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `track_ang_vel_z_exp` | +1.2 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_default_pose_on_stand` | -1 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.7 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_slide` | -0.25 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |

已删除(None)：`dof_torques_l2`、`feet_air_time`、`joint_deviation_arms`、`joint_deviation_hip`、`joint_deviation_knee`、`lin_vel_z_l2`、`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.3` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.015` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.1` |
| `algorithm.discriminator_reward_type` | `'quad'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V5-G1-v0`

- **非零 14** / 存活 14 / 已删除(None) 7
- run_name: `G1AmpWalkV5Plane_obs626_rew14_ar0.02_ns0.3_mirror_logamp`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_base_height` | +1.5 | std=0.3 | `track_base_height_exp` |
| `track_lin_vel_xy_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `track_ang_vel_z_exp` | +1.2 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_default_pose_on_stand` | -1 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.7 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_slide` | -0.25 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |

已删除(None)：`dof_torques_l2`、`feet_air_time`、`joint_deviation_arms`、`joint_deviation_hip`、`joint_deviation_knee`、`lin_vel_z_l2`、`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.3` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.015` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.1` |
| `algorithm.discriminator_reward_type` | `'log'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V6-G1-v0`

- **非零 15** / 存活 15 / 已删除(None) 6
- run_name: `G1AmpWalkV6Plane_obs626_rew15_ar0.02_ns0.3_mirror_logamp_arms`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_base_height` | +1.5 | std=0.3 | `track_base_height_exp` |
| `track_lin_vel_xy_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `track_ang_vel_z_exp` | +1.2 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_default_pose_on_stand` | -1 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.7 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_slide` | -0.25 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |

已删除(None)：`dof_torques_l2`、`feet_air_time`、`joint_deviation_hip`、`joint_deviation_knee`、`lin_vel_z_l2`、`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.3` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.015` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.1` |
| `algorithm.discriminator_reward_type` | `'log'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V7-G1-v0`

- **非零 15** / 存活 15 / 已删除(None) 6
- run_name: `G1AmpWalkV7Plane_obs626_rew15_ar0.02_ns0.2_ampc0.05_arms`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_base_height` | +1.5 | std=0.3 | `track_base_height_exp` |
| `track_lin_vel_xy_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `track_ang_vel_z_exp` | +1.2 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_default_pose_on_stand` | -1 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.7 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_slide` | -0.25 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |

已删除(None)：`dof_torques_l2`、`feet_air_time`、`joint_deviation_hip`、`joint_deviation_knee`、`lin_vel_z_l2`、`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.2` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.01` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.05` |
| `algorithm.discriminator_reward_type` | `'log'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V8-G1-v0`

- **非零 15** / 存活 15 / 已删除(None) 6
- run_name: `G1AmpWalkV8Plane_obs626_rew15_ar0.02_ns0.2_ampc0.05_logquad_arms`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_base_height` | +1.5 | std=0.3 | `track_base_height_exp` |
| `track_lin_vel_xy_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `track_ang_vel_z_exp` | +1.2 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_default_pose_on_stand` | -1 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.7 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_slide` | -0.25 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |

已删除(None)：`dof_torques_l2`、`feet_air_time`、`joint_deviation_hip`、`joint_deviation_knee`、`lin_vel_z_l2`、`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.2` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `None` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.01` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.05` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V9-G1-v0`

- **非零 15** / 存活 15 / 已删除(None) 6
- run_name: `G1AmpWalkV9MVPPlane_obs378_rew15_ar0.02_ns0.25_ampc0.05_fwdscan279_mvpcmd`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_base_height` | +1.5 | std=0.3 | `track_base_height_exp` |
| `track_lin_vel_xy_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `track_ang_vel_z_exp` | +1.2 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_default_pose_on_stand` | -1 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.7 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_slide` | -0.25 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |

已删除(None)：`dof_torques_l2`、`feet_air_time`、`joint_deviation_hip`、`joint_deviation_knee`、`lin_vel_z_l2`、`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.25` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.4` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.0125` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.05` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V9b-G1-v0`

- **非零 15** / 存活 15 / 已删除(None) 6
- run_name: `G1AmpWalkV9MVPPlane_obs378_rew15_ar0.02_ns0.25_ampc0.05_fwdscan279_mvpcmd`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_base_height` | +1.5 | std=0.3 | `track_base_height_exp` |
| `track_lin_vel_xy_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `track_ang_vel_z_exp` | +1.2 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_default_pose_on_stand` | -1 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.7 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_slide` | -0.25 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_arms` | -0.1 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |

已删除(None)：`dof_torques_l2`、`feet_air_time`、`joint_deviation_hip`、`joint_deviation_knee`、`lin_vel_z_l2`、`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.25` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.4` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.0125` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.05` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.008` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `False` |


### `Instinct-Amp-Walk-V9c-G1-v0`

- **非零 14** / 存活 15 / 已删除(None) 6
- run_name: `G1AmpWalkV9cMVPPlane_obs378_rew15_ar0.02_ns0.2_maxstd0.3_ent0.0025_ampc0.25_noarms_nomirror`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_base_height` | +1.5 | std=0.3 | `track_base_height_exp` |
| `track_lin_vel_xy_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `track_ang_vel_z_exp` | +1.2 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_default_pose_on_stand` | -1 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.7 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_slide` | -0.25 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`dof_torques_l2`、`feet_air_time`、`joint_deviation_hip`、`joint_deviation_knee`、`lin_vel_z_l2`、`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.2` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.3` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.0125` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.25` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.0025` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `False` |


### `Instinct-Amp-Walk-V9d-G1-v0`

- **非零 14** / 存活 15 / 已删除(None) 6
- run_name: `G1AmpWalkV9dMVPPlane_obs378_rew15_ar0.02_ns0.2_maxstd0.3_ent0.0025_ampc0.25_noarms_mirror`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_base_height` | +1.5 | std=0.3 | `track_base_height_exp` |
| `track_lin_vel_xy_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `track_ang_vel_z_exp` | +1.2 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_default_pose_on_stand` | -1 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.7 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_slide` | -0.25 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`dof_torques_l2`、`feet_air_time`、`joint_deviation_hip`、`joint_deviation_knee`、`lin_vel_z_l2`、`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.2` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.3` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.0125` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.25` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.0025` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V9e-G1-v0`

- **非零 15** / 存活 16 / 已删除(None) 5
- run_name: `G1AmpWalkV9eMVPPlane_obs378_rew16_ar0.02_ns0.2_maxstd0.3_ent0.0025_ampc0.25_noarms_mirror_airtime_halfstand`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_base_height` | +1.5 | std=0.3 | `track_base_height_exp` |
| `track_lin_vel_xy_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `track_ang_vel_z_exp` | +1.2 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `feet_air_time` | +1 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_default_pose_on_stand` | -1 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.7 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_slide` | -0.25 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`dof_torques_l2`、`joint_deviation_hip`、`joint_deviation_knee`、`lin_vel_z_l2`、`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.2` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.3` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.0125` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.25` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.0025` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V9f-G1-v0`

- **非零 15** / 存活 16 / 已删除(None) 5
- run_name: `G1AmpWalkV9fMVPPlane_obs378_rew16_ar0.02_ns0.3_maxstd0.4_fixedlr_ent0.0025_ampc0.25_noarms_mirror_airtime_halfstand`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_base_height` | +1.5 | std=0.3 | `track_base_height_exp` |
| `track_lin_vel_xy_exp` | +1.5 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `track_ang_vel_z_exp` | +1.2 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `feet_air_time` | +1 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_default_pose_on_stand` | -1 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `body_ang_vel_xy` | +0.7 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_slide` | -0.25 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`dof_torques_l2`、`joint_deviation_hip`、`joint_deviation_knee`、`lin_vel_z_l2`、`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.3` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.4` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.0125` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.25` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.0025` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V9g-G1-v0`

- **非零 14** / 存活 16 / 已删除(None) 5
- run_name: `G1AmpWalkV9gMVPPlane_obs378_rew16_rebalance_linvel3_airtime2_nosstill_nomirror0_mirror_ampc0.5`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `feet_air_time` | +2 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_base_height` | +0.5 | std=0.3 | `track_base_height_exp` |
| `body_ang_vel_xy` | +0.2 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `track_ang_vel_z_exp` | +0.2 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |
| `track_default_pose_on_stand` | +0 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |

已删除(None)：`dof_torques_l2`、`joint_deviation_hip`、`joint_deviation_knee`、`lin_vel_z_l2`、`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.3` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.4` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.0125` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.5` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.0025` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.5` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V9h-G1-v0`

- **非零 19** / 存活 20 / 已删除(None) 1
- run_name: `G1AmpWalkV9hMVPPlane_obs378_rew20_linvel3_airtime2_antihop_vz_nohipdev_ns0.3_lerp0.75_ampc0.25_mirror`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `feet_air_time` | +2 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `track_default_pose_on_stand` | -0.3 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.3` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.4` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.0125` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.25` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.0025` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V9i-G1-v0`

- **非零 19** / 存活 20 / 已删除(None) 1
- run_name: `G1AmpWalkV9iMVPPlane_obs378_rew20_linvel3_airtime2_antihop_vz_nohipdev_ns0.6_maxstd0.7_ent0.005_lerp0.75_ampc0.25_mirror`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `feet_air_time` | +2 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `track_default_pose_on_stand` | -0.3 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.6` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.7` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.03` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.25` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


### `Instinct-Amp-Walk-V9j-G1-v0`

- **非零 20** / 存活 21 / 已删除(None) 1
- run_name: `G1AmpWalkV9jMVPPlane_obs378_rew20_linvel3_airtime2_antihop_vz_nohipdev_swingpen_ns0.3_lerp0.75_ampc0.25_mirror`

| term | weight | params | func |
|---|---:|---|---|
| `termination_penalty` | -200 | — | `is_terminated` |
| `feet_crossing` | -20 | command_name=base_velocity, command_threshold=0.1, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_crossing_cost` |
| `dof_pos_limits` | -10 | asset_cfg=(robot|joints=['.*_ankle_pitch_joint', '.*_ankle_roll_joint']) | `joint_pos_limits` |
| `feet_lateral_width` | -10 | command_name=base_velocity, command_threshold=0.1, sep_floor=0.145, gate_lateral=True, max_cost=0.1, walk_max_lin_vel=1.0, walk_gate_width=0.2 | `feet_lateral_width_cost` |
| `track_lin_vel_xy_exp` | +3 | command_name=base_velocity, std=0.5 | `track_lin_vel_xy_yaw_frame_exp` |
| `feet_air_time` | +2 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), threshold=0.5 | `feet_air_time_positive_biped` |
| `flat_orientation_l2` | -1 | — | `flat_orientation_l2` |
| `track_base_height` | +1 | std=0.2 | `track_base_height_exp` |
| `track_ang_vel_z_exp` | +0.8 | command_name=base_velocity, std=0.5 | `track_ang_vel_z_world_exp` |
| `body_ang_vel_xy` | +0.5 | std=3.14 | `track_root_ang_vel_xy_exp` |
| `feet_swing_time_penalty` | -0.5 | command_name=base_velocity, sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), max_air_time=0.6, command_threshold=0.1 | `feet_swing_time_penalty` |
| `track_default_pose_on_stand` | -0.3 | command_name=base_velocity, command_threshold=0.1, pos_scale=4.0, vel_scale=0.5, gate_tau=0.25, pos_joint_weights={'waist_yaw_joint': 3.0} | `track_default_pose_on_stand` |
| `lin_vel_z_l2` | -0.2 | — | `lin_vel_z_l2` |
| `feet_slide` | -0.1 | sensor_cfg=(contact_forces|bodies=.*_ankle_roll_link), asset_cfg=(robot|bodies=.*_ankle_roll_link) | `contact_slide` |
| `joint_deviation_hip` | -0.1 | asset_cfg=(robot|joints=['.*_hip_yaw_joint', '.*_hip_roll_joint']) | `joint_deviation_l1` |
| `joint_deviation_torso` | -0.1 | asset_cfg=(robot|joints=waist_.*) | `joint_deviation_l1` |
| `joint_deviation_knee` | -0.05 | asset_cfg=(robot|joints=['.*_knee_joint']) | `joint_deviation_l1` |
| `action_rate_l2` | -0.02 | — | `action_rate_l2` |
| `dof_torques_l2` | -4e-06 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_torques_l2` |
| `dof_acc_l2` | -2.5e-07 | asset_cfg=(robot|joints=['.*_hip_.*', '.*_knee_joint']) | `joint_acc_l2` |
| `joint_deviation_arms` | +0 | asset_cfg=(robot|joints=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint'…) | `joint_deviation_l1` |

已删除(None)：`stand_still`
**算法侧（实际生效）**

| 字段 | 值 |
|---|---|
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.init_noise_std` | `0.3` |
| `algorithm.amp_grad_pen_space` | `'raw'` |
| `algorithm.amp_normalizer_expert_only` | `True` |
| `algorithm.amp_quad_k` | `0.25` |
| `algorithm.clip_max_std` | `0.4` |
| `algorithm.clip_max_std_schedule` | `None` |
| `algorithm.clip_min_std` | `0.0125` |
| `algorithm.disable_style_on_stand` | `True` |
| `algorithm.discriminator_kwargs` | `{'hidden_sizes': [1024, 512], 'nonlinearity': 'ReLU'}` |
| `algorithm.discriminator_reward_coef` | `0.25` |
| `algorithm.discriminator_reward_type` | `'wasserstein'` |
| `algorithm.entropy_coef` | `0.0025` |
| `algorithm.learning_rate` | `0.001` |
| `algorithm.mirror_loss_coeff` | `0.1` |
| `algorithm.style_lerp` | `0.75` |
| `algorithm.use_mirror_loss` | `True` |


---

成功 65 / 失败 0
