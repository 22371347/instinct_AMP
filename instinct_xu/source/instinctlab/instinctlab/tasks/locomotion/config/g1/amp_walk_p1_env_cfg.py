"""⭐ **自动生成**，不要手改结构 —— 只允许添加注释/文档。

来源（继承链的**解析结果**）：
  * `instinctlab.tasks.locomotion.config.g1.amp_walk_pelvis_env_cfg:G1AmpWalkPelvisEnvCfg` → `G1AmpWalkP1EnvCfg`
  * `instinctlab.tasks.locomotion.config.g1.amp_walk_pelvis_env_cfg:G1AmpWalkPelvisEnvCfg_PLAY` → `G1AmpWalkP1EnvCfg_PLAY`
上游基类：`isaaclab.envs.manager_based_rl_env_cfg.ManagerBasedRLEnvCfg`；另有 42 个自动生成的子 cfg 类。

重新生成（在有 IsaacLab 的机器上）：
    python3 instinct/tools/flatten_cfg.py generate \
        --cls instinctlab.tasks.locomotion.config.g1.amp_walk_pelvis_env_cfg:G1AmpWalkPelvisEnvCfg,instinctlab.tasks.locomotion.config.g1.amp_walk_pelvis_env_cfg:G1AmpWalkPelvisEnvCfg_PLAY \
        --out <本文件> --root-name <根类名>
等价性校验（必须 diff 为空）：
    python3 instinct/tools/flatten_cfg.py verify \
        --chain instinctlab.tasks.locomotion.config.g1.amp_walk_pelvis_env_cfg:G1AmpWalkPelvisEnvCfg --flat-file <本文件> --flat-attr G1AmpWalkP1EnvCfg
"""

from __future__ import annotations

from typing import Any
import os

#: 本文件所在目录（路径字面量都以它为基准，保证生成物换机可用）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import isaaclab.envs as _m0
import isaaclab.envs.mdp as _m15
import isaaclab.managers as _m14
import isaaclab.scene as _m2
import isaaclab.sensors as _m6
from isaaclab.utils import configclass


@configclass
class ViewerCfg:

    """`isaaclab.envs.common.ViewerCfg` 的展平副本（链上位置 `viewer`；字段值 = 链式配置的最终生效值）。"""
    eye: Any = None
    lookat: Any = None
    cam_prim_path: str = None
    resolution: Any = None
    origin_type: str = None
    env_index: int = None
    asset_name: str = None
    body_name: Any = None

    def __post_init__(self):
        self.eye = (2.0, 2.0, 0.5)
        self.lookat = (0.0, 0.0, 0.0)
        self.cam_prim_path = '/OmniverseKit_Persp'
        self.resolution = (1280, 720)
        self.origin_type = 'asset_root'
        self.env_index = 0
        self.asset_name = 'robot'
        self.body_name = None


@configclass
class SimCfg:

    """`isaaclab.sim.simulation_cfg.SimulationCfg` 的展平副本（链上位置 `sim`；字段值 = 链式配置的最终生效值）。"""
    physics_prim_path: str = None
    device: str = None
    dt: float = None
    render_interval: int = None
    gravity: Any = None
    enable_scene_query_support: bool = None
    use_fabric: bool = None
    physx: Any = None
    physics_material: Any = None
    render: Any = None
    create_stage_in_memory: bool = None
    logging_level: str = None
    save_logs_to_file: bool = None
    log_dir: Any = None

    def __post_init__(self):
        import isaaclab.sim as _m1  # 局部 import：避开导入期循环依赖
        self.physics_prim_path = '/physicsScene'
        self.device = 'cuda:0'
        self.dt = 0.005
        self.render_interval = 4
        self.gravity = (0.0, 0.0, -9.81)
        self.enable_scene_query_support = False
        self.use_fabric = True
        self.physx = _m1.PhysxCfg(solver_type=1, solve_articulation_contact_last=False, min_position_iteration_count=1, max_position_iteration_count=255, min_velocity_iteration_count=0, max_velocity_iteration_count=255, enable_ccd=False, enable_stabilization=False, enable_external_forces_every_iteration=False, enable_enhanced_determinism=False, bounce_threshold_velocity=0.5, friction_offset_threshold=0.04, friction_correlation_distance=0.025, gpu_max_rigid_contact_count=8388608, gpu_max_rigid_patch_count=163840, gpu_found_lost_pairs_capacity=2097152, gpu_found_lost_aggregate_pairs_capacity=33554432, gpu_total_aggregate_pairs_capacity=2097152, gpu_collision_stack_size=67108864, gpu_heap_capacity=67108864, gpu_temp_buffer_capacity=16777216, gpu_max_num_partitions=8, gpu_max_soft_body_contacts=1048576, gpu_max_particle_contacts=1048576)
        self.physics_material = _m1.RigidBodyMaterialCfg(func=_m1.spawn_rigid_body_material, static_friction=0.5, dynamic_friction=0.5, restitution=0.0, friction_combine_mode='average', restitution_combine_mode='average', compliant_contact_stiffness=0.0, compliant_contact_damping=0.0)
        self.render = _m1.RenderCfg(enable_translucency=None, enable_reflections=None, enable_global_illumination=None, antialiasing_mode=None, enable_dlssg=None, enable_dl_denoiser=None, dlss_mode=None, enable_direct_lighting=None, samples_per_pixel=None, enable_shadows=None, enable_ambient_occlusion=None, dome_light_upper_lower_strategy=None, carb_settings=None, rendering_mode=None)
        self.create_stage_in_memory = False
        self.logging_level = 'WARNING'
        self.save_logs_to_file = True
        self.log_dir = None


@configclass
class AmpWalkMotionCfg:

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_env_cfg.AmpWalkMotionCfg` 的展平副本（链上位置 `scene.motion_reference.motion_buffers['amp_walk']`；字段值 = 链式配置的最终生效值）。"""
    class_type: Any = None
    clip_joint_ref_to_robot_limits: bool = None
    path: str = None
    supported_file_endings: Any = None
    skip_frames: int = None
    motion_interpolate_func: Any = None
    motion_target_framerate: float = None
    assumed_file_framerate: float = None
    velocity_estimation_method: str = None
    retargetting_func: Any = None
    retargetting_func_kwargs: Any = None
    filtered_motion_selection_filepath: str = None
    subset_selection: Any = None
    motion_start_from_middle_range: Any = None
    env_starting_stub_sampling_strategy: str = None
    motion_start_height_offset: float = None
    buffer_device: str = None
    motion_bin_length_s: Any = None
    ensure_link_below_zero_ground: bool = None

    def __post_init__(self):
        from instinctlab.motion_reference.motion_files import amass_motion as _m11  # 局部 import：避开导入期循环依赖
        from instinctlab.motion_reference import utils as _m12  # 局部 import：避开导入期循环依赖
        self.class_type = _m11.AmassMotion
        self.clip_joint_ref_to_robot_limits = False
        self.path = os.path.normpath(os.path.join(_THIS_DIR, 'motions/amp_walk_pelvis'))
        self.supported_file_endings = ['retargeted.npz',]
        self.skip_frames = 0
        self.motion_interpolate_func = _m12.motion_interpolate_bilinear
        self.motion_target_framerate = 50.0
        self.assumed_file_framerate = 120.0
        self.velocity_estimation_method = 'frontbackward'
        self.retargetting_func = None
        self.retargetting_func_kwargs = {}
        self.filtered_motion_selection_filepath = os.path.normpath(os.path.join(_THIS_DIR, 'motions/amp_walk_pelvis/amp_walk_v13_motion_selection.yaml'))
        self.subset_selection = None
        self.motion_start_from_middle_range = (0.0, 1.0)
        self.env_starting_stub_sampling_strategy = 'independent'
        self.motion_start_height_offset = 0.0
        self.buffer_device = 'output_device'
        self.motion_bin_length_s = None
        self.ensure_link_below_zero_ground = False


@configclass
class MotionReferenceManagerCfg(_m6.SensorBaseCfg):

    """`instinctlab.motion_reference.motion_reference_cfg.MotionReferenceManagerCfg` 的展平副本（链上位置 `scene.motion_reference`；字段值 = 链式配置的最终生效值）。"""
    class_type: Any = None
    prim_path: str = None
    update_period: float = None
    history_length: int = None
    debug_vis: bool = None
    data_class_type: Any = None
    state_class_type: Any = None
    scene_object_names: Any = None
    robot_model_path: str = None
    link_of_interests: Any = None
    num_frames: int = None
    data_start_from: str = None
    frame_interval_s: float = None
    update_period_sample_strategy: str = None
    motion_buffers: Any = None
    mp_split_method: str = None
    symmetric_augmentation_joint_mapping: Any = None
    symmetric_augmentation_joint_reverse_buf: Any = None
    symmetric_augmentation_link_mapping: Any = None
    reference_prim_path: str = None
    visualizer_cfg: Any = None
    visualizing_marker_types: Any = None
    visualizing_robot_offset: Any = None
    visualizing_robot_from: str = None

    def __post_init__(self):
        import isaaclab.sim as _m1  # 局部 import：避开导入期循环依赖
        import instinctlab.motion_reference as _m10  # 局部 import：避开导入期循环依赖
        import isaaclab.markers as _m7  # 局部 import：避开导入期循环依赖
        super().__post_init__()
        self.class_type = _m10.MotionReferenceManager
        self.prim_path = '{ENV_REGEX_NS}/Robot/pelvis'
        self.update_period = 0.02
        self.history_length = 0
        self.debug_vis = False
        self.data_class_type = _m10.MotionReferenceData
        self.state_class_type = _m10.MotionReferenceState
        self.scene_object_names = []
        self.robot_model_path = os.path.normpath(os.path.join(_THIS_DIR, '../../../../assets/resources/unitree_g1/official_rev10/g1_29dof_rev_1_0.urdf'))
        self.link_of_interests = ['pelvis', 'torso_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_elbow_link', 'right_elbow_link', 'left_wrist_roll_link', 'right_wrist_roll_link', 'left_hip_roll_link', 'right_hip_roll_link', 'left_knee_link', 'right_knee_link', 'left_ankle_roll_link', 'right_ankle_roll_link']
        self.num_frames = 1
        self.data_start_from = 'one_frame_interval'
        self.frame_interval_s = 0.0
        self.update_period_sample_strategy = 'uniform'
        self.motion_buffers = {'amp_walk': AmpWalkMotionCfg()}
        self.mp_split_method = 'Even'
        self.symmetric_augmentation_joint_mapping = [1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27]
        self.symmetric_augmentation_joint_reverse_buf = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0]
        self.symmetric_augmentation_link_mapping = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]
        self.reference_prim_path = '/World/envs/env_.*/RobotReference/pelvis'
        self.visualizer_cfg = _m7.VisualizationMarkersCfg(prim_path='/Visuals/MotionReference', markers={'root_frame_ref': _m1.UsdFileCfg(func=_m1.spawn_from_usd, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, deformable_props=None, rigid_props=None, collision_props=None, activate_contact_sensors=False, scale=(0.15, 0.15, 0.15), articulation_props=None, fixed_tendons_props=None, spatial_tendons_props=None, joint_drive_props=None, visual_material_path='material', visual_material=None, usd_path='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/UIElements/frame_prim.usd', variants=None), 'link_ref': _m1.SphereCfg(func=_m1.spawn_sphere, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, rigid_props=None, collision_props=None, activate_contact_sensors=False, visual_material_path='material', visual_material=_m1.PreviewSurfaceCfg(func=_m1.spawn_preview_surface, diffuse_color=(0.0, 1.0, 0.0), emissive_color=(0.0, 0.0, 0.0), roughness=0.5, metallic=0.0, opacity=1.0), physics_material_path='material', physics_material=None, radius=0.04), 'relative_link_ref': _m1.UsdFileCfg(func=_m1.spawn_from_usd, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, deformable_props=None, rigid_props=None, collision_props=None, activate_contact_sensors=False, scale=(0.05, 0.05, 0.05), articulation_props=None, fixed_tendons_props=None, spatial_tendons_props=None, joint_drive_props=None, visual_material_path='material', visual_material=None, usd_path='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/UIElements/frame_prim.usd', variants=None)})
        self.visualizing_marker_types = []
        self.visualizing_robot_offset = (0.0, 2.0, 0.0)
        self.visualizing_robot_from = 'reference_frame'


@configclass
class SceneCfg(_m2.InteractiveSceneCfg):

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_env_cfg.G1AmpWalkSceneCfg` 的展平副本（链上位置 `scene`；字段值 = 链式配置的最终生效值）。"""
    num_envs: int = None
    env_spacing: float = None
    lazy_sensor_update: bool = None
    replicate_physics: bool = None
    filter_collisions: bool = None
    clone_in_fabric: bool = None
    terrain: Any = None
    robot: Any = None
    contact_forces: Any = None
    sky_light: Any = None
    height_scanner: Any = None
    motion_reference: Any = None  # → MotionReferenceManagerCfg
    robot_reference: Any = None

    def __post_init__(self):
        import isaaclab.sim as _m1  # 局部 import：避开导入期循环依赖
        import isaaclab.terrains as _m3  # 局部 import：避开导入期循环依赖
        import isaaclab.assets as _m4  # 局部 import：避开导入期循环依赖
        import isaaclab.actuators as _m5  # 局部 import：避开导入期循环依赖
        import isaaclab.markers as _m7  # 局部 import：避开导入期循环依赖
        from isaaclab.sensors.ray_caster.patterns import patterns_cfg as _m8  # 局部 import：避开导入期循环依赖
        from isaaclab.sensors.ray_caster.patterns import patterns as _m9  # 局部 import：避开导入期循环依赖
        super().__post_init__()
        self.num_envs = 4096
        self.env_spacing = 2.5
        self.lazy_sensor_update = True
        self.replicate_physics = True
        self.filter_collisions = True
        self.clone_in_fabric = False
        self.terrain = _m3.TerrainImporterCfg(class_type=_m3.TerrainImporter, collision_group=-1, prim_path='/World/ground', num_envs=1, terrain_type='plane', terrain_generator=None, usd_path=None, env_spacing=None, use_terrain_origins=True, visual_material=_m1.PreviewSurfaceCfg(func=_m1.spawn_preview_surface, diffuse_color=(0.0, 0.0, 0.0), emissive_color=(0.0, 0.0, 0.0), roughness=0.5, metallic=0.0, opacity=1.0), physics_material=_m1.RigidBodyMaterialCfg(func=_m1.spawn_rigid_body_material, static_friction=1.0, dynamic_friction=1.0, restitution=0.0, friction_combine_mode='multiply', restitution_combine_mode='multiply', compliant_contact_stiffness=0.0, compliant_contact_damping=0.0), max_init_terrain_level=None, debug_vis=False)
        self.robot = _m4.ArticulationCfg(class_type=_m4.Articulation, prim_path='{ENV_REGEX_NS}/Robot', spawn=_m1.UrdfFileCfg(asset_path=os.path.normpath(os.path.join(_THIS_DIR, '../../../../assets/resources/unitree_g1/official_rev10/g1_29dof_rev_1_0.urdf')), usd_dir=None, usd_file_name=None, force_usd_conversion=False, make_instanceable=True, fix_base=False, root_link_name=None, link_density=0.0, merge_fixed_joints=True, convert_mimic_joints_to_normal_joints=False, joint_drive=_m1.UrdfConverterCfg.JointDriveCfg(drive_type='force', target_type='position', gains=_m1.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)), collider_type='convex_hull', self_collision=False, replace_cylinders_with_capsules=True, collision_from_visuals=False, func=_m1.spawn_from_urdf, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, deformable_props=None, rigid_props=_m1.RigidBodyPropertiesCfg(rigid_body_enabled=None, kinematic_enabled=None, disable_gravity=False, linear_damping=0.0, angular_damping=0.0, max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0, max_contact_impulse=None, enable_gyroscopic_forces=None, retain_accelerations=False, solver_position_iteration_count=None, solver_velocity_iteration_count=None, sleep_threshold=None, stabilization_threshold=None), collision_props=None, activate_contact_sensors=True, scale=None, articulation_props=_m1.ArticulationRootPropertiesCfg(articulation_enabled=None, enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4, sleep_threshold=None, stabilization_threshold=None, fix_root_link=None), fixed_tendons_props=None, spatial_tendons_props=None, joint_drive_props=None, visual_material_path='material', visual_material=None), init_state=_m4.ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.7585), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0), joint_pos={'.*_hip_pitch_joint': -0.312, '.*_knee_joint': 0.669, '.*_ankle_pitch_joint': -0.363, '.*_elbow_joint': 0.6, 'left_shoulder_roll_joint': 0.2, 'left_shoulder_pitch_joint': 0.2, 'right_shoulder_roll_joint': -0.2, 'right_shoulder_pitch_joint': 0.2}, joint_vel={'.*': 0.0}), collision_group=0, debug_vis=False, articulation_root_prim_path=None, soft_joint_pos_limit_factor=0.9, actuators={'legs': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['.*_hip_yaw_joint', '.*_hip_roll_joint', '.*_hip_pitch_joint', '.*_knee_joint'], effort_limit=None, velocity_limit=None, effort_limit_sim={'.*_hip_yaw_joint': 88.0, '.*_hip_roll_joint': 139.0, '.*_hip_pitch_joint': 88.0, '.*_knee_joint': 139.0}, velocity_limit_sim={'.*_hip_yaw_joint': 32.0, '.*_hip_roll_joint': 20.0, '.*_hip_pitch_joint': 32.0, '.*_knee_joint': 20.0}, stiffness={'.*_hip_pitch_joint': 40.17923847137318, '.*_hip_roll_joint': 99.09842777666113, '.*_hip_yaw_joint': 40.17923847137318, '.*_knee_joint': 99.09842777666113}, damping={'.*_hip_pitch_joint': 2.5578897650279457, '.*_hip_roll_joint': 6.3088018534966395, '.*_hip_yaw_joint': 2.5578897650279457, '.*_knee_joint': 6.3088018534966395}, armature={'.*_hip_pitch_joint': 0.01017752, '.*_hip_roll_joint': 0.025101925, '.*_hip_yaw_joint': 0.01017752, '.*_knee_joint': 0.025101925}, friction=None, dynamic_friction=None, viscous_friction=None), 'feet': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['.*_ankle_pitch_joint', '.*_ankle_roll_joint'], effort_limit=None, velocity_limit=None, effort_limit_sim=50.0, velocity_limit_sim=37.0, stiffness=28.50124619574858, damping=1.814445686584846, armature=0.00721945, friction=None, dynamic_friction=None, viscous_friction=None), 'waist': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['waist_roll_joint', 'waist_pitch_joint'], effort_limit=None, velocity_limit=None, effort_limit_sim=50, velocity_limit_sim=37.0, stiffness=28.50124619574858, damping=1.814445686584846, armature=0.00721945, friction=None, dynamic_friction=None, viscous_friction=None), 'waist_yaw': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['waist_yaw_joint',], effort_limit=None, velocity_limit=None, effort_limit_sim=88, velocity_limit_sim=32.0, stiffness=40.17923847137318, damping=2.5578897650279457, armature=0.01017752, friction=None, dynamic_friction=None, viscous_friction=None), 'arms': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint', '.*_shoulder_yaw_joint', '.*_elbow_joint', '.*_wrist_roll_joint', '.*_wrist_pitch_joint', '.*_wrist_yaw_joint'], effort_limit=None, velocity_limit=None, effort_limit_sim={'.*_shoulder_pitch_joint': 25.0, '.*_shoulder_roll_joint': 25.0, '.*_shoulder_yaw_joint': 25.0, '.*_elbow_joint': 25.0, '.*_wrist_roll_joint': 25.0, '.*_wrist_pitch_joint': 5.0, '.*_wrist_yaw_joint': 5.0}, velocity_limit_sim={'.*_shoulder_pitch_joint': 37.0, '.*_shoulder_roll_joint': 37.0, '.*_shoulder_yaw_joint': 37.0, '.*_elbow_joint': 37.0, '.*_wrist_roll_joint': 37.0, '.*_wrist_pitch_joint': 22.0, '.*_wrist_yaw_joint': 22.0}, stiffness={'.*_shoulder_pitch_joint': 14.25062309787429, '.*_shoulder_roll_joint': 14.25062309787429, '.*_shoulder_yaw_joint': 14.25062309787429, '.*_elbow_joint': 14.25062309787429, '.*_wrist_roll_joint': 14.25062309787429, '.*_wrist_pitch_joint': 16.77832748089279, '.*_wrist_yaw_joint': 16.77832748089279}, damping={'.*_shoulder_pitch_joint': 0.907222843292423, '.*_shoulder_roll_joint': 0.907222843292423, '.*_shoulder_yaw_joint': 0.907222843292423, '.*_elbow_joint': 0.907222843292423, '.*_wrist_roll_joint': 0.907222843292423, '.*_wrist_pitch_joint': 1.06814150219, '.*_wrist_yaw_joint': 1.06814150219}, armature={'.*_shoulder_pitch_joint': 0.003609725, '.*_shoulder_roll_joint': 0.003609725, '.*_shoulder_yaw_joint': 0.003609725, '.*_elbow_joint': 0.003609725, '.*_wrist_roll_joint': 0.003609725, '.*_wrist_pitch_joint': 0.00425, '.*_wrist_yaw_joint': 0.00425}, friction=None, dynamic_friction=None, viscous_friction=None)}, actuator_value_resolution_debug_print=False)
        self.contact_forces = _m6.ContactSensorCfg(class_type=_m6.ContactSensor, prim_path='{ENV_REGEX_NS}/Robot/.*', update_period=0.005, history_length=3, debug_vis=False, track_pose=False, track_contact_points=False, track_friction_forces=False, max_contact_data_count_per_prim=4, track_air_time=True, force_threshold=1.0, filter_prim_paths_expr=[], visualizer_cfg=_m7.VisualizationMarkersCfg(prim_path='/Visuals/ContactSensor', markers={'contact': _m1.SphereCfg(func=_m1.spawn_sphere, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, rigid_props=None, collision_props=None, activate_contact_sensors=False, visual_material_path='material', visual_material=_m1.PreviewSurfaceCfg(func=_m1.spawn_preview_surface, diffuse_color=(1.0, 0.0, 0.0), emissive_color=(0.0, 0.0, 0.0), roughness=0.5, metallic=0.0, opacity=1.0), physics_material_path='material', physics_material=None, radius=0.02), 'no_contact': _m1.SphereCfg(func=_m1.spawn_sphere, visible=False, semantic_tags=None, copy_from_source=True, mass_props=None, rigid_props=None, collision_props=None, activate_contact_sensors=False, visual_material_path='material', visual_material=_m1.PreviewSurfaceCfg(func=_m1.spawn_preview_surface, diffuse_color=(0.0, 1.0, 0.0), emissive_color=(0.0, 0.0, 0.0), roughness=0.5, metallic=0.0, opacity=1.0), physics_material_path='material', physics_material=None, radius=0.02)}))
        self.sky_light = _m4.AssetBaseCfg(class_type=None, prim_path='/World/skyLight', spawn=_m1.DomeLightCfg(func=_m1.spawn_light, visible=True, semantic_tags=None, copy_from_source=True, prim_type='DomeLight', color=(1.0, 1.0, 1.0), enable_color_temperature=False, color_temperature=6500.0, normalize=False, exposure=0.0, intensity=750.0, texture_file='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr', texture_format='automatic', visible_in_primary_ray=True), init_state=_m4.AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)), collision_group=0, debug_vis=False)
        self.height_scanner = _m6.MultiMeshRayCasterCfg(class_type=_m6.MultiMeshRayCaster, prim_path='{ENV_REGEX_NS}/Robot/pelvis', update_period=0.02, history_length=0, debug_vis=False, mesh_prim_paths=['/World/ground',], offset=_m6.RayCasterCfg.OffsetCfg(pos=(1.5, 0.0, 20.0), rot=(1.0, 0.0, 0.0, 0.0)), attach_yaw_only=None, ray_alignment='yaw', pattern_cfg=_m8.GridPatternCfg(func=_m9.grid_pattern, resolution=0.1, size=[3.0, 0.8], direction=(0.0, 0.0, -1.0), ordering='xy'), max_distance=1000000.0, drift_range=(0.0, 0.0), ray_cast_drift_range={'x': (0.0, 0.0), 'y': (0.0, 0.0), 'z': (0.0, 0.0)}, visualizer_cfg=_m7.VisualizationMarkersCfg(prim_path='/Visuals/RayCaster', markers={'hit': _m1.SphereCfg(func=_m1.spawn_sphere, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, rigid_props=None, collision_props=None, activate_contact_sensors=False, visual_material_path='material', visual_material=_m1.PreviewSurfaceCfg(func=_m1.spawn_preview_surface, diffuse_color=(1.0, 0.0, 0.0), emissive_color=(0.0, 0.0, 0.0), roughness=0.5, metallic=0.0, opacity=1.0), physics_material_path='material', physics_material=None, radius=0.02)}), update_mesh_ids=False, reference_meshes=True)
        self.motion_reference = MotionReferenceManagerCfg()
        self.robot_reference = _m4.ArticulationCfg(class_type=_m4.Articulation, prim_path='{ENV_REGEX_NS}/RobotReference', spawn=_m1.UrdfFileCfg(asset_path=os.path.normpath(os.path.join(_THIS_DIR, '../../../../assets/resources/unitree_g1/official_rev10/g1_29dof_rev_1_0.urdf')), usd_dir=None, usd_file_name=None, force_usd_conversion=False, make_instanceable=True, fix_base=False, root_link_name=None, link_density=0.0, merge_fixed_joints=True, convert_mimic_joints_to_normal_joints=False, joint_drive=_m1.UrdfConverterCfg.JointDriveCfg(drive_type='force', target_type='position', gains=_m1.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)), collider_type='convex_hull', self_collision=False, replace_cylinders_with_capsules=True, collision_from_visuals=False, func=_m1.spawn_from_urdf, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, deformable_props=None, rigid_props=_m1.RigidBodyPropertiesCfg(rigid_body_enabled=None, kinematic_enabled=None, disable_gravity=False, linear_damping=0.0, angular_damping=0.0, max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0, max_contact_impulse=None, enable_gyroscopic_forces=None, retain_accelerations=False, solver_position_iteration_count=None, solver_velocity_iteration_count=None, sleep_threshold=None, stabilization_threshold=None), collision_props=None, activate_contact_sensors=True, scale=None, articulation_props=_m1.ArticulationRootPropertiesCfg(articulation_enabled=None, enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4, sleep_threshold=None, stabilization_threshold=None, fix_root_link=None), fixed_tendons_props=None, spatial_tendons_props=None, joint_drive_props=None, visual_material_path='material', visual_material=None), init_state=_m4.ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.7585), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0), joint_pos={'.*_hip_pitch_joint': -0.312, '.*_knee_joint': 0.669, '.*_ankle_pitch_joint': -0.363, '.*_elbow_joint': 0.6, 'left_shoulder_roll_joint': 0.2, 'left_shoulder_pitch_joint': 0.2, 'right_shoulder_roll_joint': -0.2, 'right_shoulder_pitch_joint': 0.2}, joint_vel={'.*': 0.0}), collision_group=0, debug_vis=False, articulation_root_prim_path=None, soft_joint_pos_limit_factor=0.9, actuators={'legs': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['.*_hip_yaw_joint', '.*_hip_roll_joint', '.*_hip_pitch_joint', '.*_knee_joint'], effort_limit=None, velocity_limit=None, effort_limit_sim={'.*_hip_yaw_joint': 88.0, '.*_hip_roll_joint': 139.0, '.*_hip_pitch_joint': 88.0, '.*_knee_joint': 139.0}, velocity_limit_sim={'.*_hip_yaw_joint': 32.0, '.*_hip_roll_joint': 20.0, '.*_hip_pitch_joint': 32.0, '.*_knee_joint': 20.0}, stiffness={'.*_hip_pitch_joint': 40.17923847137318, '.*_hip_roll_joint': 99.09842777666113, '.*_hip_yaw_joint': 40.17923847137318, '.*_knee_joint': 99.09842777666113}, damping={'.*_hip_pitch_joint': 2.5578897650279457, '.*_hip_roll_joint': 6.3088018534966395, '.*_hip_yaw_joint': 2.5578897650279457, '.*_knee_joint': 6.3088018534966395}, armature={'.*_hip_pitch_joint': 0.01017752, '.*_hip_roll_joint': 0.025101925, '.*_hip_yaw_joint': 0.01017752, '.*_knee_joint': 0.025101925}, friction=None, dynamic_friction=None, viscous_friction=None), 'feet': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['.*_ankle_pitch_joint', '.*_ankle_roll_joint'], effort_limit=None, velocity_limit=None, effort_limit_sim=50.0, velocity_limit_sim=37.0, stiffness=28.50124619574858, damping=1.814445686584846, armature=0.00721945, friction=None, dynamic_friction=None, viscous_friction=None), 'waist': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['waist_roll_joint', 'waist_pitch_joint'], effort_limit=None, velocity_limit=None, effort_limit_sim=50, velocity_limit_sim=37.0, stiffness=28.50124619574858, damping=1.814445686584846, armature=0.00721945, friction=None, dynamic_friction=None, viscous_friction=None), 'waist_yaw': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['waist_yaw_joint',], effort_limit=None, velocity_limit=None, effort_limit_sim=88, velocity_limit_sim=32.0, stiffness=40.17923847137318, damping=2.5578897650279457, armature=0.01017752, friction=None, dynamic_friction=None, viscous_friction=None), 'arms': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint', '.*_shoulder_yaw_joint', '.*_elbow_joint', '.*_wrist_roll_joint', '.*_wrist_pitch_joint', '.*_wrist_yaw_joint'], effort_limit=None, velocity_limit=None, effort_limit_sim={'.*_shoulder_pitch_joint': 25.0, '.*_shoulder_roll_joint': 25.0, '.*_shoulder_yaw_joint': 25.0, '.*_elbow_joint': 25.0, '.*_wrist_roll_joint': 25.0, '.*_wrist_pitch_joint': 5.0, '.*_wrist_yaw_joint': 5.0}, velocity_limit_sim={'.*_shoulder_pitch_joint': 37.0, '.*_shoulder_roll_joint': 37.0, '.*_shoulder_yaw_joint': 37.0, '.*_elbow_joint': 37.0, '.*_wrist_roll_joint': 37.0, '.*_wrist_pitch_joint': 22.0, '.*_wrist_yaw_joint': 22.0}, stiffness={'.*_shoulder_pitch_joint': 14.25062309787429, '.*_shoulder_roll_joint': 14.25062309787429, '.*_shoulder_yaw_joint': 14.25062309787429, '.*_elbow_joint': 14.25062309787429, '.*_wrist_roll_joint': 14.25062309787429, '.*_wrist_pitch_joint': 16.77832748089279, '.*_wrist_yaw_joint': 16.77832748089279}, damping={'.*_shoulder_pitch_joint': 0.907222843292423, '.*_shoulder_roll_joint': 0.907222843292423, '.*_shoulder_yaw_joint': 0.907222843292423, '.*_elbow_joint': 0.907222843292423, '.*_wrist_roll_joint': 0.907222843292423, '.*_wrist_pitch_joint': 1.06814150219, '.*_wrist_yaw_joint': 1.06814150219}, armature={'.*_shoulder_pitch_joint': 0.003609725, '.*_shoulder_roll_joint': 0.003609725, '.*_shoulder_yaw_joint': 0.003609725, '.*_elbow_joint': 0.003609725, '.*_wrist_roll_joint': 0.003609725, '.*_wrist_pitch_joint': 0.00425, '.*_wrist_yaw_joint': 0.00425}, friction=None, dynamic_friction=None, viscous_friction=None)}, actuator_value_resolution_debug_print=False)


@configclass
class RecordersCfg:

    """`isaaclab.managers.recorder_manager.RecorderManagerBaseCfg` 的展平副本（链上位置 `recorders`；字段值 = 链式配置的最终生效值）。"""
    dataset_file_handler_class_type: Any = None
    dataset_export_dir_path: str = None
    dataset_filename: str = None
    dataset_export_mode: Any = None
    export_in_record_pre_reset: bool = None
    export_in_close: bool = None

    def __post_init__(self):
        import isaaclab.utils.datasets as _m13  # 局部 import：避开导入期循环依赖
        self.dataset_file_handler_class_type = _m13.HDF5DatasetFileHandler
        self.dataset_export_dir_path = os.path.normpath(os.path.join(_THIS_DIR, '../../../../../../../../../../tmp/isaaclab/logs'))
        self.dataset_filename = 'dataset'
        self.dataset_export_mode = _m14.DatasetExportMode.EXPORT_ALL
        self.export_in_record_pre_reset = True
        self.export_in_close = False


@configclass
class T2PolicyObsCfg(_m14.ObservationGroupCfg):

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_t2_env_cfg.G1AmpWalkT2PolicyObsCfg` 的展平副本（链上位置 `observations.policy`；字段值 = 链式配置的最终生效值）。"""
    concatenate_terms: bool = None
    concatenate_dim: int = None
    enable_corruption: bool = None
    history_length: Any = None
    flatten_history_dim: bool = None
    height_scan: Any = None
    velocity_command: Any = None
    projected_gravity: Any = None
    base_ang_vel: Any = None
    joint_pos_rel: Any = None
    joint_vel: Any = None
    last_action: Any = None

    def __post_init__(self):
        import isaaclab.utils.noise as _m16  # 局部 import：避开导入期循环依赖
        super().__post_init__()
        self.concatenate_terms = False
        self.concatenate_dim = -1
        self.enable_corruption = True
        self.history_length = None
        self.flatten_history_dim = True
        self.height_scan = _m14.ObservationTermCfg(func=_m15.height_scan, params={'sensor_cfg': _m14.SceneEntityCfg(name='height_scanner', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=None, body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, modifiers=None, noise=_m16.UniformNoiseCfg(func=_m16.uniform_noise, operation='add', n_min=-0.1, n_max=0.1), clip=(-1.0, 1.0), scale=None, history_length=4, flatten_history_dim=True)
        self.velocity_command = _m14.ObservationTermCfg(func=_m15.generated_commands, params={'command_name': 'base_velocity'}, modifiers=None, noise=None, clip=None, scale=None, history_length=4, flatten_history_dim=True)
        self.projected_gravity = _m14.ObservationTermCfg(func=_m15.projected_gravity, params={}, modifiers=None, noise=_m16.UniformNoiseCfg(func=_m16.uniform_noise, operation='add', n_min=-0.05, n_max=0.05), clip=None, scale=None, history_length=4, flatten_history_dim=True)
        self.base_ang_vel = _m14.ObservationTermCfg(func=_m15.base_ang_vel, params={}, modifiers=None, noise=_m16.UniformNoiseCfg(func=_m16.uniform_noise, operation='add', n_min=-0.2, n_max=0.2), clip=None, scale=None, history_length=4, flatten_history_dim=True)
        self.joint_pos_rel = _m14.ObservationTermCfg(func=_m15.joint_pos_rel, params={}, modifiers=None, noise=_m16.UniformNoiseCfg(func=_m16.uniform_noise, operation='add', n_min=-0.01, n_max=0.01), clip=None, scale=None, history_length=4, flatten_history_dim=True)
        self.joint_vel = _m14.ObservationTermCfg(func=_m15.joint_vel, params={}, modifiers=None, noise=_m16.UniformNoiseCfg(func=_m16.uniform_noise, operation='add', n_min=-0.5, n_max=0.5), clip=None, scale=None, history_length=4, flatten_history_dim=True)
        self.last_action = _m14.ObservationTermCfg(func=_m15.last_action, params={}, modifiers=None, noise=None, clip=None, scale=None, history_length=4, flatten_history_dim=True)


@configclass
class FlatCriticObsCfg(_m14.ObservationGroupCfg):

    """`instinctlab.tasks.locomotion.config.g1.flat_env_cfg.G1FlatCriticObsCfg` 的展平副本（链上位置 `observations.critic`；字段值 = 链式配置的最终生效值）。"""
    concatenate_terms: bool = None
    concatenate_dim: int = None
    enable_corruption: bool = None
    history_length: Any = None
    flatten_history_dim: bool = None
    base_lin_vel: Any = None
    base_ang_vel: Any = None
    projected_gravity: Any = None
    velocity_commands: Any = None
    joint_pos: Any = None
    joint_vel: Any = None
    actions: Any = None

    def __post_init__(self):
        super().__post_init__()
        self.concatenate_terms = False
        self.concatenate_dim = -1
        self.enable_corruption = False
        self.history_length = None
        self.flatten_history_dim = True
        self.base_lin_vel = _m14.ObservationTermCfg(func=_m15.base_lin_vel, params={}, modifiers=None, noise=None, clip=None, scale=None, history_length=0, flatten_history_dim=True)
        self.base_ang_vel = _m14.ObservationTermCfg(func=_m15.base_ang_vel, params={}, modifiers=None, noise=None, clip=None, scale=None, history_length=0, flatten_history_dim=True)
        self.projected_gravity = _m14.ObservationTermCfg(func=_m15.projected_gravity, params={}, modifiers=None, noise=None, clip=None, scale=None, history_length=0, flatten_history_dim=True)
        self.velocity_commands = _m14.ObservationTermCfg(func=_m15.generated_commands, params={'command_name': 'base_velocity'}, modifiers=None, noise=None, clip=None, scale=None, history_length=0, flatten_history_dim=True)
        self.joint_pos = _m14.ObservationTermCfg(func=_m15.joint_pos_rel, params={}, modifiers=None, noise=None, clip=None, scale=None, history_length=0, flatten_history_dim=True)
        self.joint_vel = _m14.ObservationTermCfg(func=_m15.joint_vel, params={}, modifiers=None, noise=None, clip=None, scale=None, history_length=0, flatten_history_dim=True)
        self.actions = _m14.ObservationTermCfg(func=_m15.last_action, params={}, modifiers=None, noise=None, clip=None, scale=None, history_length=0, flatten_history_dim=True)


@configclass
class AmpV33PolicyStateObsCfg(_m14.ObservationGroupCfg):

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_v33_env_cfg.AmpV33PolicyStateObsCfg` 的展平副本（链上位置 `observations.amp_policy`；字段值 = 链式配置的最终生效值）。"""
    concatenate_terms: bool = None
    concatenate_dim: int = None
    enable_corruption: bool = None
    history_length: Any = None
    flatten_history_dim: bool = None
    body_pos_b: Any = None
    body_ori_b: Any = None
    body_lin_vel_b: Any = None
    body_ang_vel_b: Any = None
    anchor_gravity_b: Any = None

    def __post_init__(self):
        import instinctlab.envs.mdp as _m17  # 局部 import：避开导入期循环依赖
        super().__post_init__()
        self.concatenate_terms = False
        self.concatenate_dim = -1
        self.enable_corruption = False
        self.history_length = None
        self.flatten_history_dim = True
        self.body_pos_b = _m14.ObservationTermCfg(func=_m17.body_pos_b, params={'asset_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link'), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=True), 'anchor_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis',), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)
        self.body_ori_b = _m14.ObservationTermCfg(func=_m17.body_ori_b, params={'asset_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link'), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=True), 'anchor_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis',), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)
        self.body_lin_vel_b = _m14.ObservationTermCfg(func=_m17.body_lin_vel_b, params={'asset_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link'), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=True)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)
        self.body_ang_vel_b = _m14.ObservationTermCfg(func=_m17.body_ang_vel_b, params={'asset_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link'), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=True)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)
        self.anchor_gravity_b = _m14.ObservationTermCfg(func=_m17.anchor_gravity_b, params={'anchor_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis',), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)


@configclass
class AmpV33ReferenceStateObsCfg(_m14.ObservationGroupCfg):

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_v33_env_cfg.AmpV33ReferenceStateObsCfg` 的展平副本（链上位置 `observations.amp_reference`；字段值 = 链式配置的最终生效值）。"""
    concatenate_terms: bool = None
    concatenate_dim: int = None
    enable_corruption: bool = None
    history_length: Any = None
    flatten_history_dim: bool = None
    body_pos_b: Any = None
    body_ori_b: Any = None
    body_lin_vel_b: Any = None
    body_ang_vel_b: Any = None
    anchor_gravity_b: Any = None

    def __post_init__(self):
        import instinctlab.envs.mdp as _m17  # 局部 import：避开导入期循环依赖
        super().__post_init__()
        self.concatenate_terms = False
        self.concatenate_dim = -1
        self.enable_corruption = False
        self.history_length = None
        self.flatten_history_dim = True
        self.body_pos_b = _m14.ObservationTermCfg(func=_m17.body_pos_b_reference_as_state, params={'asset_cfg': _m14.SceneEntityCfg(name='motion_reference', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link'), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=True), 'anchor_cfg': _m14.SceneEntityCfg(name='motion_reference', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis',), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)
        self.body_ori_b = _m14.ObservationTermCfg(func=_m17.body_ori_b_reference_as_state, params={'asset_cfg': _m14.SceneEntityCfg(name='motion_reference', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link'), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=True), 'anchor_cfg': _m14.SceneEntityCfg(name='motion_reference', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis',), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)
        self.body_lin_vel_b = _m14.ObservationTermCfg(func=_m17.body_lin_vel_b_reference_as_state, params={'asset_cfg': _m14.SceneEntityCfg(name='motion_reference', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link'), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=True)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)
        self.body_ang_vel_b = _m14.ObservationTermCfg(func=_m17.body_ang_vel_b_reference_as_state, params={'asset_cfg': _m14.SceneEntityCfg(name='motion_reference', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link'), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=True)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)
        self.anchor_gravity_b = _m14.ObservationTermCfg(func=_m17.anchor_gravity_b_reference_as_state, params={'anchor_cfg': _m14.SceneEntityCfg(name='motion_reference', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis',), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)


@configclass
class ObservationsCfg:

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_v21_env_cfg.G1AmpWalkV21ObservationsCfg` 的展平副本（链上位置 `observations`；字段值 = 链式配置的最终生效值）。"""
    policy: Any = None  # → T2PolicyObsCfg
    critic: Any = None  # → FlatCriticObsCfg
    amp_policy: Any = None  # → AmpV33PolicyStateObsCfg
    amp_reference: Any = None  # → AmpV33ReferenceStateObsCfg

    def __post_init__(self):
        self.policy = T2PolicyObsCfg()
        self.critic = FlatCriticObsCfg()
        self.amp_policy = AmpV33PolicyStateObsCfg()
        self.amp_reference = AmpV33ReferenceStateObsCfg()


@configclass
class ActionsCfg:

    """`instinctlab.tasks.locomotion.config.g1.omni_loco_env_cfg.G1OmniLocoActionsCfg` 的展平副本（链上位置 `actions`；字段值 = 链式配置的最终生效值）。"""
    joint_pos: Any = None

    def __post_init__(self):
        self.joint_pos = _m15.JointPositionActionCfg(class_type=_m15.JointPositionAction, asset_name='robot', debug_vis=False, clip=None, joint_names=['.*',], scale=1.0, offset=0.0, preserve_order=False, use_default_offset=True)


@configclass
class EventsCfg:

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_t2_env_cfg.G1AmpWalkT2EventsCfg` 的展平副本（链上位置 `events`；字段值 = 链式配置的最终生效值）。"""
    physics_material: Any = None
    base_com: Any = None
    push_robot: Any = None
    reset_from_reference: Any = None

    def __post_init__(self):
        import instinctlab.envs.mdp as _m17  # 局部 import：避开导入期循环依赖
        self.physics_material = _m14.EventTermCfg(func=_m15.randomize_rigid_body_material, params={'asset_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names='.*', body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False), 'static_friction_range': (0.25, 0.8), 'dynamic_friction_range': (0.2, 0.6), 'restitution_range': (0.0, 0.8), 'num_buckets': 64}, mode='startup', interval_range_s=None, is_global_time=False, min_step_count_between_reset=0)
        self.base_com = _m14.EventTermCfg(func=_m15.randomize_rigid_body_com, params={'asset_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names='torso_link', body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False), 'com_range': {'x': (-0.025, 0.025), 'y': (-0.025, 0.025), 'z': (-0.03, 0.03)}}, mode='startup', interval_range_s=None, is_global_time=False, min_step_count_between_reset=0)
        self.push_robot = _m14.EventTermCfg(func=_m15.push_by_setting_velocity, params={'velocity_range': {'x': (-1.0, 1.0), 'y': (-0.5, 0.5), 'z': (-0.4, 0.4), 'roll': (-0.52, 0.52), 'pitch': (-0.52, 0.52), 'yaw': (-0.78, 0.78)}}, mode='interval', interval_range_s=(1.0, 3.0), is_global_time=False, min_step_count_between_reset=0)
        self.reset_from_reference = _m14.EventTermCfg(func=_m17.reset_robot_state_by_reference, params={'position_offset': [0.0, 0.0, 0.0], 'dof_vel_ratio': 1.0, 'base_lin_vel_ratio': 1.0, 'base_ang_vel_ratio': 1.0}, mode='reset', interval_range_s=None, is_global_time=False, min_step_count_between_reset=0)


@configclass
class TeleopDevicesCfg:

    """`isaaclab.devices.device_base.DevicesCfg` 的展平副本（链上位置 `teleop_devices`；字段值 = 链式配置的最终生效值）。"""
    devices: Any = None

    def __post_init__(self):
        self.devices = {}


@configclass
class RewardsCfg:

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_t2_env_cfg.G1AmpWalkT2RewardsCfg` 的展平副本（链上位置 `rewards`；字段值 = 链式配置的最终生效值）。"""
    termination_penalty: Any = None
    track_lin_vel_xy_exp: Any = None
    track_ang_vel_z_exp: Any = None
    feet_air_time: Any = None
    feet_slide: Any = None
    flat_orientation_l2: Any = None
    stand_still: Any = None
    dof_pos_limits: Any = None
    joint_deviation_hip: Any = None
    joint_deviation_arms: Any = None
    joint_deviation_torso: Any = None
    joint_deviation_knee: Any = None
    lin_vel_z_l2: Any = None
    action_rate_l2: Any = None
    dof_acc_l2: Any = None
    dof_torques_l2: Any = None
    track_base_height: Any = None
    body_ang_vel_xy: Any = None
    track_default_pose_on_stand: Any = None
    feet_lateral_width: Any = None
    feet_crossing: Any = None
    track_anchor_linear_velocity: Any = None
    track_anchor_angular_velocity: Any = None

    def __post_init__(self):
        import instinctlab.envs.mdp as _m17  # 局部 import：避开导入期循环依赖
        import instinctlab.tasks.locomotion.mdp as _m18  # 局部 import：避开导入期循环依赖
        self.termination_penalty = _m14.RewardTermCfg(func=_m15.is_terminated, params={}, weight=-200.0)
        self.track_lin_vel_xy_exp = None
        self.track_ang_vel_z_exp = None
        self.feet_air_time = None
        self.feet_slide = _m14.RewardTermCfg(func=_m17.contact_slide, params={'sensor_cfg': _m14.SceneEntityCfg(name='contact_forces', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names='.*_ankle_roll_link', body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False), 'asset_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names='.*_ankle_roll_link', body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, weight=-0.25)
        self.flat_orientation_l2 = None
        self.stand_still = None
        self.dof_pos_limits = _m14.RewardTermCfg(func=_m15.joint_pos_limits, params={'asset_cfg': _m14.SceneEntityCfg(name='robot', joint_names=['.*',], joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=None, body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, weight=-10.0)
        self.joint_deviation_hip = None
        self.joint_deviation_arms = None
        self.joint_deviation_torso = None
        self.joint_deviation_knee = None
        self.lin_vel_z_l2 = None
        self.action_rate_l2 = _m14.RewardTermCfg(func=_m15.action_rate_l2, params={}, weight=-0.01)
        self.dof_acc_l2 = _m14.RewardTermCfg(func=_m15.joint_acc_l2, params={}, weight=-2.5e-07)
        self.dof_torques_l2 = None
        self.track_base_height = _m14.RewardTermCfg(func=_m18.track_base_height_exp, params={'std': 0.3}, weight=1.5)
        self.body_ang_vel_xy = _m14.RewardTermCfg(func=_m18.track_root_ang_vel_xy_exp, params={'std': 3.14}, weight=0.7)
        self.track_default_pose_on_stand = _m14.RewardTermCfg(func=_m18.track_default_pose_on_stand, params={'command_name': 'base_velocity', 'command_threshold': 0.1, 'pos_scale': 4.0, 'vel_scale': 0.5, 'gate_tau': 0.25, 'pos_joint_weights': {'waist_yaw_joint': 3.0}}, weight=-1.0)
        self.feet_lateral_width = _m14.RewardTermCfg(func=_m18.feet_lateral_width_cost, params={'command_name': 'base_velocity', 'command_threshold': 0.1, 'sep_floor': 0.15, 'gate_lateral': True, 'max_cost': 0.1, 'walk_max_lin_vel': 1.0, 'walk_gate_width': 0.2}, weight=-10.0)
        self.feet_crossing = _m14.RewardTermCfg(func=_m18.feet_crossing_cost, params={'command_name': 'base_velocity', 'command_threshold': 0.1, 'gate_lateral': True, 'max_cost': 0.1, 'walk_max_lin_vel': 1.0, 'walk_gate_width': 0.2}, weight=-20.0)
        self.track_anchor_linear_velocity = _m14.RewardTermCfg(func=_m18.track_anchor_lin_vel_exp, params={'command_name': 'base_velocity', 'std': 1.0}, weight=1.5)
        self.track_anchor_angular_velocity = _m14.RewardTermCfg(func=_m18.track_anchor_ang_vel_exp, params={'command_name': 'base_velocity', 'std': 3.14}, weight=1.2)


@configclass
class TerminationsCfg:

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_t2_env_cfg.G1AmpWalkT2TerminationsCfg` 的展平副本（链上位置 `terminations`；字段值 = 链式配置的最终生效值）。"""
    time_out: Any = None
    bad_orientation: Any = None
    bad_base_height: Any = None

    def __post_init__(self):
        self.time_out = _m14.TerminationTermCfg(func=_m15.time_out, params={}, time_out=True)
        self.bad_orientation = _m14.TerminationTermCfg(func=_m15.bad_orientation, params={'limit_angle': 1.2217304763960306}, time_out=False)
        self.bad_base_height = _m14.TerminationTermCfg(func=_m15.root_height_below_minimum, params={'minimum_height': 0.5}, time_out=False)


@configclass
class CurriculumCfg:

    """`instinctlab.tasks.locomotion.config.g1.flat_env_cfg.G1FlatCurriculumCfg` 的展平副本（链上位置 `curriculum`；字段值 = 链式配置的最终生效值）。"""

    def __post_init__(self):
        pass


@configclass
class DirectionalVelocityCommandCfg(_m15.UniformVelocityCommandCfg):

    """`instinctlab.tasks.locomotion.mdp.commands.DirectionalVelocityCommandCfg` 的展平副本（链上位置 `commands.base_velocity`；字段值 = 链式配置的最终生效值）。"""
    class_type: Any = None
    resampling_time_range: Any = None
    debug_vis: bool = None
    asset_name: str = None
    heading_command: bool = None
    heading_control_stiffness: float = None
    rel_standing_envs: float = None
    rel_heading_envs: float = None
    ranges: Any = None
    goal_vel_visualizer_cfg: Any = None
    current_vel_visualizer_cfg: Any = None
    prob_low: float = None
    prob_side: float = None
    prob_lin: float = None
    prob_back: float = None
    side_ang_vel_range: Any = None
    side_vel_min: float = None
    low_vel: float = None
    zero_frac_in_low: float = None
    prob_lin_fwd: float = None
    straight_lin_vel_range: Any = None
    straight_ang_vel_range: Any = None
    back_lin_vel_range: Any = None
    back_ang_vel_range: Any = None

    def __post_init__(self):
        import isaaclab.sim as _m1  # 局部 import：避开导入期循环依赖
        import instinctlab.tasks.locomotion.mdp as _m18  # 局部 import：避开导入期循环依赖
        import isaaclab.markers as _m7  # 局部 import：避开导入期循环依赖
        super().__post_init__()
        self.class_type = _m18.DirectionalVelocityCommand
        self.resampling_time_range = (6.0, 15.0)
        self.debug_vis = True
        self.asset_name = 'robot'
        self.heading_command = False
        self.heading_control_stiffness = 1.0
        self.rel_standing_envs = 0.0
        self.rel_heading_envs = 0.0
        self.ranges = _m15.UniformVelocityCommandCfg.Ranges(lin_vel_x=(-0.5, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.57, 1.57), heading=(-3.141592653589793, 3.141592653589793))
        self.goal_vel_visualizer_cfg = _m7.VisualizationMarkersCfg(prim_path='/Visuals/Command/velocity_goal', markers={'arrow': _m1.UsdFileCfg(func=_m1.spawn_from_usd, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, deformable_props=None, rigid_props=None, collision_props=None, activate_contact_sensors=False, scale=(0.5, 0.5, 0.5), articulation_props=None, fixed_tendons_props=None, spatial_tendons_props=None, joint_drive_props=None, visual_material_path='material', visual_material=_m1.PreviewSurfaceCfg(func=_m1.spawn_preview_surface, diffuse_color=(0.0, 1.0, 0.0), emissive_color=(0.0, 0.0, 0.0), roughness=0.5, metallic=0.0, opacity=1.0), usd_path='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/UIElements/arrow_x.usd', variants=None)})
        self.current_vel_visualizer_cfg = _m7.VisualizationMarkersCfg(prim_path='/Visuals/Command/velocity_current', markers={'arrow': _m1.UsdFileCfg(func=_m1.spawn_from_usd, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, deformable_props=None, rigid_props=None, collision_props=None, activate_contact_sensors=False, scale=(0.5, 0.5, 0.5), articulation_props=None, fixed_tendons_props=None, spatial_tendons_props=None, joint_drive_props=None, visual_material_path='material', visual_material=_m1.PreviewSurfaceCfg(func=_m1.spawn_preview_surface, diffuse_color=(0.0, 0.0, 1.0), emissive_color=(0.0, 0.0, 0.0), roughness=0.5, metallic=0.0, opacity=1.0), usd_path='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/UIElements/arrow_x.usd', variants=None)})
        self.prob_low = 0.2
        self.prob_side = 0.2
        self.prob_lin = 0.4
        self.prob_back = 0.1
        self.side_ang_vel_range = None
        self.side_vel_min = 0.0
        self.low_vel = 0.1
        self.zero_frac_in_low = 0.6
        self.prob_lin_fwd = 0.7
        self.straight_lin_vel_range = (0.4, 1.0)
        self.straight_ang_vel_range = (-0.1, 0.1)
        self.back_lin_vel_range = (-0.7, -0.4)
        self.back_ang_vel_range = (-0.1, 0.1)


@configclass
class CommandsCfg:

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_v3_env_cfg.G1AmpWalkV3CommandsCfg` 的展平副本（链上位置 `commands`；字段值 = 链式配置的最终生效值）。"""
    base_velocity: Any = None  # → DirectionalVelocityCommandCfg

    def __post_init__(self):
        self.base_velocity = DirectionalVelocityCommandCfg()


@configclass
class MonitorTermCfg(_m14.ManagerTermBaseCfg):

    """`instinctlab.monitors.monitor_cfg.MonitorTermCfg` 的展平副本（链上位置 `monitors.tracking`；字段值 = 链式配置的最终生效值）。"""
    func: Any = None
    params: Any = None

    def __post_init__(self):
        import instinctlab.tasks.locomotion.mdp as _m18  # 局部 import：避开导入期循环依赖
        super().__post_init__()
        self.func = _m18.LocoTrackingMonitorTerm
        self.params = {}


@configclass
class MonitorsCfg:

    """`instinctlab.tasks.locomotion.config.g1.omni_loco_env_cfg.G1OmniLocoMonitorCfg` 的展平副本（链上位置 `monitors`；字段值 = 链式配置的最终生效值）。"""
    tracking: Any = None  # → MonitorTermCfg

    def __post_init__(self):
        self.tracking = MonitorTermCfg()


@configclass
class ViewerCfg_1:

    """`isaaclab.envs.common.ViewerCfg` 的展平副本（链上位置 `viewer`；字段值 = 链式配置的最终生效值）。"""
    eye: Any = None
    lookat: Any = None
    cam_prim_path: str = None
    resolution: Any = None
    origin_type: str = None
    env_index: int = None
    asset_name: str = None
    body_name: Any = None

    def __post_init__(self):
        self.eye = (2.0, 2.0, 0.5)
        self.lookat = (0.0, 0.0, 0.0)
        self.cam_prim_path = '/OmniverseKit_Persp'
        self.resolution = (1280, 720)
        self.origin_type = 'asset_root'
        self.env_index = 0
        self.asset_name = 'robot'
        self.body_name = None


@configclass
class SimCfg_1:

    """`isaaclab.sim.simulation_cfg.SimulationCfg` 的展平副本（链上位置 `sim`；字段值 = 链式配置的最终生效值）。"""
    physics_prim_path: str = None
    device: str = None
    dt: float = None
    render_interval: int = None
    gravity: Any = None
    enable_scene_query_support: bool = None
    use_fabric: bool = None
    physx: Any = None
    physics_material: Any = None
    render: Any = None
    create_stage_in_memory: bool = None
    logging_level: str = None
    save_logs_to_file: bool = None
    log_dir: Any = None

    def __post_init__(self):
        import isaaclab.sim as _m1  # 局部 import：避开导入期循环依赖
        self.physics_prim_path = '/physicsScene'
        self.device = 'cuda:0'
        self.dt = 0.005
        self.render_interval = 4
        self.gravity = (0.0, 0.0, -9.81)
        self.enable_scene_query_support = False
        self.use_fabric = True
        self.physx = _m1.PhysxCfg(solver_type=1, solve_articulation_contact_last=False, min_position_iteration_count=1, max_position_iteration_count=255, min_velocity_iteration_count=0, max_velocity_iteration_count=255, enable_ccd=False, enable_stabilization=False, enable_external_forces_every_iteration=False, enable_enhanced_determinism=False, bounce_threshold_velocity=0.5, friction_offset_threshold=0.04, friction_correlation_distance=0.025, gpu_max_rigid_contact_count=8388608, gpu_max_rigid_patch_count=163840, gpu_found_lost_pairs_capacity=2097152, gpu_found_lost_aggregate_pairs_capacity=33554432, gpu_total_aggregate_pairs_capacity=2097152, gpu_collision_stack_size=67108864, gpu_heap_capacity=67108864, gpu_temp_buffer_capacity=16777216, gpu_max_num_partitions=8, gpu_max_soft_body_contacts=1048576, gpu_max_particle_contacts=1048576)
        self.physics_material = _m1.RigidBodyMaterialCfg(func=_m1.spawn_rigid_body_material, static_friction=0.5, dynamic_friction=0.5, restitution=0.0, friction_combine_mode='average', restitution_combine_mode='average', compliant_contact_stiffness=0.0, compliant_contact_damping=0.0)
        self.render = _m1.RenderCfg(enable_translucency=None, enable_reflections=None, enable_global_illumination=None, antialiasing_mode=None, enable_dlssg=None, enable_dl_denoiser=None, dlss_mode=None, enable_direct_lighting=None, samples_per_pixel=None, enable_shadows=None, enable_ambient_occlusion=None, dome_light_upper_lower_strategy=None, carb_settings=None, rendering_mode=None)
        self.create_stage_in_memory = False
        self.logging_level = 'WARNING'
        self.save_logs_to_file = True
        self.log_dir = None


@configclass
class AmpWalkMotionCfg_1:

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_env_cfg.AmpWalkMotionCfg` 的展平副本（链上位置 `scene.motion_reference.motion_buffers['amp_walk']`；字段值 = 链式配置的最终生效值）。"""
    class_type: Any = None
    clip_joint_ref_to_robot_limits: bool = None
    path: str = None
    supported_file_endings: Any = None
    skip_frames: int = None
    motion_interpolate_func: Any = None
    motion_target_framerate: float = None
    assumed_file_framerate: float = None
    velocity_estimation_method: str = None
    retargetting_func: Any = None
    retargetting_func_kwargs: Any = None
    filtered_motion_selection_filepath: str = None
    subset_selection: Any = None
    motion_start_from_middle_range: Any = None
    env_starting_stub_sampling_strategy: str = None
    motion_start_height_offset: float = None
    buffer_device: str = None
    motion_bin_length_s: Any = None
    ensure_link_below_zero_ground: bool = None

    def __post_init__(self):
        from instinctlab.motion_reference.motion_files import amass_motion as _m11  # 局部 import：避开导入期循环依赖
        from instinctlab.motion_reference import utils as _m12  # 局部 import：避开导入期循环依赖
        self.class_type = _m11.AmassMotion
        self.clip_joint_ref_to_robot_limits = False
        self.path = os.path.normpath(os.path.join(_THIS_DIR, 'motions/amp_walk_pelvis'))
        self.supported_file_endings = ['retargeted.npz',]
        self.skip_frames = 0
        self.motion_interpolate_func = _m12.motion_interpolate_bilinear
        self.motion_target_framerate = 50.0
        self.assumed_file_framerate = 120.0
        self.velocity_estimation_method = 'frontbackward'
        self.retargetting_func = None
        self.retargetting_func_kwargs = {}
        self.filtered_motion_selection_filepath = os.path.normpath(os.path.join(_THIS_DIR, 'motions/amp_walk_pelvis/amp_walk_v13_motion_selection.yaml'))
        self.subset_selection = None
        self.motion_start_from_middle_range = (0.0, 1.0)
        self.env_starting_stub_sampling_strategy = 'independent'
        self.motion_start_height_offset = 0.0
        self.buffer_device = 'output_device'
        self.motion_bin_length_s = None
        self.ensure_link_below_zero_ground = False


@configclass
class MotionReferenceManagerCfg_1(_m6.SensorBaseCfg):

    """`instinctlab.motion_reference.motion_reference_cfg.MotionReferenceManagerCfg` 的展平副本（链上位置 `scene.motion_reference`；字段值 = 链式配置的最终生效值）。"""
    class_type: Any = None
    prim_path: str = None
    update_period: float = None
    history_length: int = None
    debug_vis: bool = None
    data_class_type: Any = None
    state_class_type: Any = None
    scene_object_names: Any = None
    robot_model_path: str = None
    link_of_interests: Any = None
    num_frames: int = None
    data_start_from: str = None
    frame_interval_s: float = None
    update_period_sample_strategy: str = None
    motion_buffers: Any = None
    mp_split_method: str = None
    symmetric_augmentation_joint_mapping: Any = None
    symmetric_augmentation_joint_reverse_buf: Any = None
    symmetric_augmentation_link_mapping: Any = None
    reference_prim_path: str = None
    visualizer_cfg: Any = None
    visualizing_marker_types: Any = None
    visualizing_robot_offset: Any = None
    visualizing_robot_from: str = None

    def __post_init__(self):
        import isaaclab.sim as _m1  # 局部 import：避开导入期循环依赖
        import instinctlab.motion_reference as _m10  # 局部 import：避开导入期循环依赖
        import isaaclab.markers as _m7  # 局部 import：避开导入期循环依赖
        super().__post_init__()
        self.class_type = _m10.MotionReferenceManager
        self.prim_path = '{ENV_REGEX_NS}/Robot/pelvis'
        self.update_period = 0.02
        self.history_length = 0
        self.debug_vis = True
        self.data_class_type = _m10.MotionReferenceData
        self.state_class_type = _m10.MotionReferenceState
        self.scene_object_names = []
        self.robot_model_path = os.path.normpath(os.path.join(_THIS_DIR, '../../../../assets/resources/unitree_g1/official_rev10/g1_29dof_rev_1_0.urdf'))
        self.link_of_interests = ['pelvis', 'torso_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_elbow_link', 'right_elbow_link', 'left_wrist_roll_link', 'right_wrist_roll_link', 'left_hip_roll_link', 'right_hip_roll_link', 'left_knee_link', 'right_knee_link', 'left_ankle_roll_link', 'right_ankle_roll_link']
        self.num_frames = 1
        self.data_start_from = 'one_frame_interval'
        self.frame_interval_s = 0.0
        self.update_period_sample_strategy = 'uniform'
        self.motion_buffers = {'amp_walk': AmpWalkMotionCfg_1()}
        self.mp_split_method = 'Even'
        self.symmetric_augmentation_joint_mapping = [1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27]
        self.symmetric_augmentation_joint_reverse_buf = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0]
        self.symmetric_augmentation_link_mapping = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]
        self.reference_prim_path = '/World/envs/env_.*/RobotReference/pelvis'
        self.visualizer_cfg = _m7.VisualizationMarkersCfg(prim_path='/Visuals/MotionReference', markers={'root_frame_ref': _m1.UsdFileCfg(func=_m1.spawn_from_usd, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, deformable_props=None, rigid_props=None, collision_props=None, activate_contact_sensors=False, scale=(0.15, 0.15, 0.15), articulation_props=None, fixed_tendons_props=None, spatial_tendons_props=None, joint_drive_props=None, visual_material_path='material', visual_material=None, usd_path='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/UIElements/frame_prim.usd', variants=None), 'link_ref': _m1.SphereCfg(func=_m1.spawn_sphere, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, rigid_props=None, collision_props=None, activate_contact_sensors=False, visual_material_path='material', visual_material=_m1.PreviewSurfaceCfg(func=_m1.spawn_preview_surface, diffuse_color=(0.0, 1.0, 0.0), emissive_color=(0.0, 0.0, 0.0), roughness=0.5, metallic=0.0, opacity=1.0), physics_material_path='material', physics_material=None, radius=0.04), 'relative_link_ref': _m1.UsdFileCfg(func=_m1.spawn_from_usd, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, deformable_props=None, rigid_props=None, collision_props=None, activate_contact_sensors=False, scale=(0.05, 0.05, 0.05), articulation_props=None, fixed_tendons_props=None, spatial_tendons_props=None, joint_drive_props=None, visual_material_path='material', visual_material=None, usd_path='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/UIElements/frame_prim.usd', variants=None)})
        self.visualizing_marker_types = []
        self.visualizing_robot_offset = (0.0, 2.0, 0.0)
        self.visualizing_robot_from = 'reference_frame'


@configclass
class SceneCfg_1(_m2.InteractiveSceneCfg):

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_env_cfg.G1AmpWalkSceneCfg` 的展平副本（链上位置 `scene`；字段值 = 链式配置的最终生效值）。"""
    num_envs: int = None
    env_spacing: float = None
    lazy_sensor_update: bool = None
    replicate_physics: bool = None
    filter_collisions: bool = None
    clone_in_fabric: bool = None
    terrain: Any = None
    robot: Any = None
    contact_forces: Any = None
    sky_light: Any = None
    height_scanner: Any = None
    motion_reference: Any = None  # → MotionReferenceManagerCfg
    robot_reference: Any = None

    def __post_init__(self):
        import isaaclab.sim as _m1  # 局部 import：避开导入期循环依赖
        import isaaclab.terrains as _m3  # 局部 import：避开导入期循环依赖
        import isaaclab.assets as _m4  # 局部 import：避开导入期循环依赖
        import isaaclab.actuators as _m5  # 局部 import：避开导入期循环依赖
        import isaaclab.markers as _m7  # 局部 import：避开导入期循环依赖
        from isaaclab.sensors.ray_caster.patterns import patterns_cfg as _m8  # 局部 import：避开导入期循环依赖
        from isaaclab.sensors.ray_caster.patterns import patterns as _m9  # 局部 import：避开导入期循环依赖
        super().__post_init__()
        self.num_envs = 1
        self.env_spacing = 2.5
        self.lazy_sensor_update = True
        self.replicate_physics = True
        self.filter_collisions = True
        self.clone_in_fabric = False
        self.terrain = _m3.TerrainImporterCfg(class_type=_m3.TerrainImporter, collision_group=-1, prim_path='/World/ground', num_envs=1, terrain_type='plane', terrain_generator=None, usd_path=None, env_spacing=None, use_terrain_origins=True, visual_material=_m1.PreviewSurfaceCfg(func=_m1.spawn_preview_surface, diffuse_color=(0.0, 0.0, 0.0), emissive_color=(0.0, 0.0, 0.0), roughness=0.5, metallic=0.0, opacity=1.0), physics_material=_m1.RigidBodyMaterialCfg(func=_m1.spawn_rigid_body_material, static_friction=1.0, dynamic_friction=1.0, restitution=0.0, friction_combine_mode='multiply', restitution_combine_mode='multiply', compliant_contact_stiffness=0.0, compliant_contact_damping=0.0), max_init_terrain_level=None, debug_vis=False)
        self.robot = _m4.ArticulationCfg(class_type=_m4.Articulation, prim_path='{ENV_REGEX_NS}/Robot', spawn=_m1.UrdfFileCfg(asset_path=os.path.normpath(os.path.join(_THIS_DIR, '../../../../assets/resources/unitree_g1/official_rev10/g1_29dof_rev_1_0.urdf')), usd_dir=None, usd_file_name=None, force_usd_conversion=False, make_instanceable=True, fix_base=False, root_link_name=None, link_density=0.0, merge_fixed_joints=True, convert_mimic_joints_to_normal_joints=False, joint_drive=_m1.UrdfConverterCfg.JointDriveCfg(drive_type='force', target_type='position', gains=_m1.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)), collider_type='convex_hull', self_collision=False, replace_cylinders_with_capsules=True, collision_from_visuals=False, func=_m1.spawn_from_urdf, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, deformable_props=None, rigid_props=_m1.RigidBodyPropertiesCfg(rigid_body_enabled=None, kinematic_enabled=None, disable_gravity=False, linear_damping=0.0, angular_damping=0.0, max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0, max_contact_impulse=None, enable_gyroscopic_forces=None, retain_accelerations=False, solver_position_iteration_count=None, solver_velocity_iteration_count=None, sleep_threshold=None, stabilization_threshold=None), collision_props=None, activate_contact_sensors=True, scale=None, articulation_props=_m1.ArticulationRootPropertiesCfg(articulation_enabled=None, enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4, sleep_threshold=None, stabilization_threshold=None, fix_root_link=None), fixed_tendons_props=None, spatial_tendons_props=None, joint_drive_props=None, visual_material_path='material', visual_material=None), init_state=_m4.ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.7585), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0), joint_pos={'.*_hip_pitch_joint': -0.312, '.*_knee_joint': 0.669, '.*_ankle_pitch_joint': -0.363, '.*_elbow_joint': 0.6, 'left_shoulder_roll_joint': 0.2, 'left_shoulder_pitch_joint': 0.2, 'right_shoulder_roll_joint': -0.2, 'right_shoulder_pitch_joint': 0.2}, joint_vel={'.*': 0.0}), collision_group=0, debug_vis=False, articulation_root_prim_path=None, soft_joint_pos_limit_factor=0.9, actuators={'legs': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['.*_hip_yaw_joint', '.*_hip_roll_joint', '.*_hip_pitch_joint', '.*_knee_joint'], effort_limit=None, velocity_limit=None, effort_limit_sim={'.*_hip_yaw_joint': 88.0, '.*_hip_roll_joint': 139.0, '.*_hip_pitch_joint': 88.0, '.*_knee_joint': 139.0}, velocity_limit_sim={'.*_hip_yaw_joint': 32.0, '.*_hip_roll_joint': 20.0, '.*_hip_pitch_joint': 32.0, '.*_knee_joint': 20.0}, stiffness={'.*_hip_pitch_joint': 40.17923847137318, '.*_hip_roll_joint': 99.09842777666113, '.*_hip_yaw_joint': 40.17923847137318, '.*_knee_joint': 99.09842777666113}, damping={'.*_hip_pitch_joint': 2.5578897650279457, '.*_hip_roll_joint': 6.3088018534966395, '.*_hip_yaw_joint': 2.5578897650279457, '.*_knee_joint': 6.3088018534966395}, armature={'.*_hip_pitch_joint': 0.01017752, '.*_hip_roll_joint': 0.025101925, '.*_hip_yaw_joint': 0.01017752, '.*_knee_joint': 0.025101925}, friction=None, dynamic_friction=None, viscous_friction=None), 'feet': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['.*_ankle_pitch_joint', '.*_ankle_roll_joint'], effort_limit=None, velocity_limit=None, effort_limit_sim=50.0, velocity_limit_sim=37.0, stiffness=28.50124619574858, damping=1.814445686584846, armature=0.00721945, friction=None, dynamic_friction=None, viscous_friction=None), 'waist': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['waist_roll_joint', 'waist_pitch_joint'], effort_limit=None, velocity_limit=None, effort_limit_sim=50, velocity_limit_sim=37.0, stiffness=28.50124619574858, damping=1.814445686584846, armature=0.00721945, friction=None, dynamic_friction=None, viscous_friction=None), 'waist_yaw': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['waist_yaw_joint',], effort_limit=None, velocity_limit=None, effort_limit_sim=88, velocity_limit_sim=32.0, stiffness=40.17923847137318, damping=2.5578897650279457, armature=0.01017752, friction=None, dynamic_friction=None, viscous_friction=None), 'arms': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint', '.*_shoulder_yaw_joint', '.*_elbow_joint', '.*_wrist_roll_joint', '.*_wrist_pitch_joint', '.*_wrist_yaw_joint'], effort_limit=None, velocity_limit=None, effort_limit_sim={'.*_shoulder_pitch_joint': 25.0, '.*_shoulder_roll_joint': 25.0, '.*_shoulder_yaw_joint': 25.0, '.*_elbow_joint': 25.0, '.*_wrist_roll_joint': 25.0, '.*_wrist_pitch_joint': 5.0, '.*_wrist_yaw_joint': 5.0}, velocity_limit_sim={'.*_shoulder_pitch_joint': 37.0, '.*_shoulder_roll_joint': 37.0, '.*_shoulder_yaw_joint': 37.0, '.*_elbow_joint': 37.0, '.*_wrist_roll_joint': 37.0, '.*_wrist_pitch_joint': 22.0, '.*_wrist_yaw_joint': 22.0}, stiffness={'.*_shoulder_pitch_joint': 14.25062309787429, '.*_shoulder_roll_joint': 14.25062309787429, '.*_shoulder_yaw_joint': 14.25062309787429, '.*_elbow_joint': 14.25062309787429, '.*_wrist_roll_joint': 14.25062309787429, '.*_wrist_pitch_joint': 16.77832748089279, '.*_wrist_yaw_joint': 16.77832748089279}, damping={'.*_shoulder_pitch_joint': 0.907222843292423, '.*_shoulder_roll_joint': 0.907222843292423, '.*_shoulder_yaw_joint': 0.907222843292423, '.*_elbow_joint': 0.907222843292423, '.*_wrist_roll_joint': 0.907222843292423, '.*_wrist_pitch_joint': 1.06814150219, '.*_wrist_yaw_joint': 1.06814150219}, armature={'.*_shoulder_pitch_joint': 0.003609725, '.*_shoulder_roll_joint': 0.003609725, '.*_shoulder_yaw_joint': 0.003609725, '.*_elbow_joint': 0.003609725, '.*_wrist_roll_joint': 0.003609725, '.*_wrist_pitch_joint': 0.00425, '.*_wrist_yaw_joint': 0.00425}, friction=None, dynamic_friction=None, viscous_friction=None)}, actuator_value_resolution_debug_print=False)
        self.contact_forces = _m6.ContactSensorCfg(class_type=_m6.ContactSensor, prim_path='{ENV_REGEX_NS}/Robot/.*', update_period=0.005, history_length=3, debug_vis=False, track_pose=False, track_contact_points=False, track_friction_forces=False, max_contact_data_count_per_prim=4, track_air_time=True, force_threshold=1.0, filter_prim_paths_expr=[], visualizer_cfg=_m7.VisualizationMarkersCfg(prim_path='/Visuals/ContactSensor', markers={'contact': _m1.SphereCfg(func=_m1.spawn_sphere, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, rigid_props=None, collision_props=None, activate_contact_sensors=False, visual_material_path='material', visual_material=_m1.PreviewSurfaceCfg(func=_m1.spawn_preview_surface, diffuse_color=(1.0, 0.0, 0.0), emissive_color=(0.0, 0.0, 0.0), roughness=0.5, metallic=0.0, opacity=1.0), physics_material_path='material', physics_material=None, radius=0.02), 'no_contact': _m1.SphereCfg(func=_m1.spawn_sphere, visible=False, semantic_tags=None, copy_from_source=True, mass_props=None, rigid_props=None, collision_props=None, activate_contact_sensors=False, visual_material_path='material', visual_material=_m1.PreviewSurfaceCfg(func=_m1.spawn_preview_surface, diffuse_color=(0.0, 1.0, 0.0), emissive_color=(0.0, 0.0, 0.0), roughness=0.5, metallic=0.0, opacity=1.0), physics_material_path='material', physics_material=None, radius=0.02)}))
        self.sky_light = _m4.AssetBaseCfg(class_type=None, prim_path='/World/skyLight', spawn=_m1.DomeLightCfg(func=_m1.spawn_light, visible=True, semantic_tags=None, copy_from_source=True, prim_type='DomeLight', color=(1.0, 1.0, 1.0), enable_color_temperature=False, color_temperature=6500.0, normalize=False, exposure=0.0, intensity=750.0, texture_file='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr', texture_format='automatic', visible_in_primary_ray=True), init_state=_m4.AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)), collision_group=0, debug_vis=False)
        self.height_scanner = _m6.MultiMeshRayCasterCfg(class_type=_m6.MultiMeshRayCaster, prim_path='{ENV_REGEX_NS}/Robot/pelvis', update_period=0.02, history_length=0, debug_vis=False, mesh_prim_paths=['/World/ground',], offset=_m6.RayCasterCfg.OffsetCfg(pos=(1.5, 0.0, 20.0), rot=(1.0, 0.0, 0.0, 0.0)), attach_yaw_only=None, ray_alignment='yaw', pattern_cfg=_m8.GridPatternCfg(func=_m9.grid_pattern, resolution=0.1, size=[3.0, 0.8], direction=(0.0, 0.0, -1.0), ordering='xy'), max_distance=1000000.0, drift_range=(0.0, 0.0), ray_cast_drift_range={'x': (0.0, 0.0), 'y': (0.0, 0.0), 'z': (0.0, 0.0)}, visualizer_cfg=_m7.VisualizationMarkersCfg(prim_path='/Visuals/RayCaster', markers={'hit': _m1.SphereCfg(func=_m1.spawn_sphere, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, rigid_props=None, collision_props=None, activate_contact_sensors=False, visual_material_path='material', visual_material=_m1.PreviewSurfaceCfg(func=_m1.spawn_preview_surface, diffuse_color=(1.0, 0.0, 0.0), emissive_color=(0.0, 0.0, 0.0), roughness=0.5, metallic=0.0, opacity=1.0), physics_material_path='material', physics_material=None, radius=0.02)}), update_mesh_ids=False, reference_meshes=True)
        self.motion_reference = MotionReferenceManagerCfg_1()
        self.robot_reference = _m4.ArticulationCfg(class_type=_m4.Articulation, prim_path='{ENV_REGEX_NS}/RobotReference', spawn=_m1.UrdfFileCfg(asset_path=os.path.normpath(os.path.join(_THIS_DIR, '../../../../assets/resources/unitree_g1/official_rev10/g1_29dof_rev_1_0.urdf')), usd_dir=None, usd_file_name=None, force_usd_conversion=False, make_instanceable=True, fix_base=False, root_link_name=None, link_density=0.0, merge_fixed_joints=True, convert_mimic_joints_to_normal_joints=False, joint_drive=_m1.UrdfConverterCfg.JointDriveCfg(drive_type='force', target_type='position', gains=_m1.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)), collider_type='convex_hull', self_collision=False, replace_cylinders_with_capsules=True, collision_from_visuals=False, func=_m1.spawn_from_urdf, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, deformable_props=None, rigid_props=_m1.RigidBodyPropertiesCfg(rigid_body_enabled=None, kinematic_enabled=None, disable_gravity=False, linear_damping=0.0, angular_damping=0.0, max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0, max_contact_impulse=None, enable_gyroscopic_forces=None, retain_accelerations=False, solver_position_iteration_count=None, solver_velocity_iteration_count=None, sleep_threshold=None, stabilization_threshold=None), collision_props=None, activate_contact_sensors=True, scale=None, articulation_props=_m1.ArticulationRootPropertiesCfg(articulation_enabled=None, enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4, sleep_threshold=None, stabilization_threshold=None, fix_root_link=None), fixed_tendons_props=None, spatial_tendons_props=None, joint_drive_props=None, visual_material_path='material', visual_material=None), init_state=_m4.ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.7585), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0), joint_pos={'.*_hip_pitch_joint': -0.312, '.*_knee_joint': 0.669, '.*_ankle_pitch_joint': -0.363, '.*_elbow_joint': 0.6, 'left_shoulder_roll_joint': 0.2, 'left_shoulder_pitch_joint': 0.2, 'right_shoulder_roll_joint': -0.2, 'right_shoulder_pitch_joint': 0.2}, joint_vel={'.*': 0.0}), collision_group=0, debug_vis=False, articulation_root_prim_path=None, soft_joint_pos_limit_factor=0.9, actuators={'legs': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['.*_hip_yaw_joint', '.*_hip_roll_joint', '.*_hip_pitch_joint', '.*_knee_joint'], effort_limit=None, velocity_limit=None, effort_limit_sim={'.*_hip_yaw_joint': 88.0, '.*_hip_roll_joint': 139.0, '.*_hip_pitch_joint': 88.0, '.*_knee_joint': 139.0}, velocity_limit_sim={'.*_hip_yaw_joint': 32.0, '.*_hip_roll_joint': 20.0, '.*_hip_pitch_joint': 32.0, '.*_knee_joint': 20.0}, stiffness={'.*_hip_pitch_joint': 40.17923847137318, '.*_hip_roll_joint': 99.09842777666113, '.*_hip_yaw_joint': 40.17923847137318, '.*_knee_joint': 99.09842777666113}, damping={'.*_hip_pitch_joint': 2.5578897650279457, '.*_hip_roll_joint': 6.3088018534966395, '.*_hip_yaw_joint': 2.5578897650279457, '.*_knee_joint': 6.3088018534966395}, armature={'.*_hip_pitch_joint': 0.01017752, '.*_hip_roll_joint': 0.025101925, '.*_hip_yaw_joint': 0.01017752, '.*_knee_joint': 0.025101925}, friction=None, dynamic_friction=None, viscous_friction=None), 'feet': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['.*_ankle_pitch_joint', '.*_ankle_roll_joint'], effort_limit=None, velocity_limit=None, effort_limit_sim=50.0, velocity_limit_sim=37.0, stiffness=28.50124619574858, damping=1.814445686584846, armature=0.00721945, friction=None, dynamic_friction=None, viscous_friction=None), 'waist': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['waist_roll_joint', 'waist_pitch_joint'], effort_limit=None, velocity_limit=None, effort_limit_sim=50, velocity_limit_sim=37.0, stiffness=28.50124619574858, damping=1.814445686584846, armature=0.00721945, friction=None, dynamic_friction=None, viscous_friction=None), 'waist_yaw': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['waist_yaw_joint',], effort_limit=None, velocity_limit=None, effort_limit_sim=88, velocity_limit_sim=32.0, stiffness=40.17923847137318, damping=2.5578897650279457, armature=0.01017752, friction=None, dynamic_friction=None, viscous_friction=None), 'arms': _m5.ImplicitActuatorCfg(class_type=_m5.ImplicitActuator, joint_names_expr=['.*_shoulder_pitch_joint', '.*_shoulder_roll_joint', '.*_shoulder_yaw_joint', '.*_elbow_joint', '.*_wrist_roll_joint', '.*_wrist_pitch_joint', '.*_wrist_yaw_joint'], effort_limit=None, velocity_limit=None, effort_limit_sim={'.*_shoulder_pitch_joint': 25.0, '.*_shoulder_roll_joint': 25.0, '.*_shoulder_yaw_joint': 25.0, '.*_elbow_joint': 25.0, '.*_wrist_roll_joint': 25.0, '.*_wrist_pitch_joint': 5.0, '.*_wrist_yaw_joint': 5.0}, velocity_limit_sim={'.*_shoulder_pitch_joint': 37.0, '.*_shoulder_roll_joint': 37.0, '.*_shoulder_yaw_joint': 37.0, '.*_elbow_joint': 37.0, '.*_wrist_roll_joint': 37.0, '.*_wrist_pitch_joint': 22.0, '.*_wrist_yaw_joint': 22.0}, stiffness={'.*_shoulder_pitch_joint': 14.25062309787429, '.*_shoulder_roll_joint': 14.25062309787429, '.*_shoulder_yaw_joint': 14.25062309787429, '.*_elbow_joint': 14.25062309787429, '.*_wrist_roll_joint': 14.25062309787429, '.*_wrist_pitch_joint': 16.77832748089279, '.*_wrist_yaw_joint': 16.77832748089279}, damping={'.*_shoulder_pitch_joint': 0.907222843292423, '.*_shoulder_roll_joint': 0.907222843292423, '.*_shoulder_yaw_joint': 0.907222843292423, '.*_elbow_joint': 0.907222843292423, '.*_wrist_roll_joint': 0.907222843292423, '.*_wrist_pitch_joint': 1.06814150219, '.*_wrist_yaw_joint': 1.06814150219}, armature={'.*_shoulder_pitch_joint': 0.003609725, '.*_shoulder_roll_joint': 0.003609725, '.*_shoulder_yaw_joint': 0.003609725, '.*_elbow_joint': 0.003609725, '.*_wrist_roll_joint': 0.003609725, '.*_wrist_pitch_joint': 0.00425, '.*_wrist_yaw_joint': 0.00425}, friction=None, dynamic_friction=None, viscous_friction=None)}, actuator_value_resolution_debug_print=False)


@configclass
class RecordersCfg_1:

    """`isaaclab.managers.recorder_manager.RecorderManagerBaseCfg` 的展平副本（链上位置 `recorders`；字段值 = 链式配置的最终生效值）。"""
    dataset_file_handler_class_type: Any = None
    dataset_export_dir_path: str = None
    dataset_filename: str = None
    dataset_export_mode: Any = None
    export_in_record_pre_reset: bool = None
    export_in_close: bool = None

    def __post_init__(self):
        import isaaclab.utils.datasets as _m13  # 局部 import：避开导入期循环依赖
        self.dataset_file_handler_class_type = _m13.HDF5DatasetFileHandler
        self.dataset_export_dir_path = os.path.normpath(os.path.join(_THIS_DIR, '../../../../../../../../../../tmp/isaaclab/logs'))
        self.dataset_filename = 'dataset'
        self.dataset_export_mode = _m14.DatasetExportMode.EXPORT_ALL
        self.export_in_record_pre_reset = True
        self.export_in_close = False


@configclass
class T2PolicyObsCfg_1(_m14.ObservationGroupCfg):

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_t2_env_cfg.G1AmpWalkT2PolicyObsCfg` 的展平副本（链上位置 `observations.policy`；字段值 = 链式配置的最终生效值）。"""
    concatenate_terms: bool = None
    concatenate_dim: int = None
    enable_corruption: bool = None
    history_length: Any = None
    flatten_history_dim: bool = None
    height_scan: Any = None
    velocity_command: Any = None
    projected_gravity: Any = None
    base_ang_vel: Any = None
    joint_pos_rel: Any = None
    joint_vel: Any = None
    last_action: Any = None

    def __post_init__(self):
        import isaaclab.utils.noise as _m16  # 局部 import：避开导入期循环依赖
        super().__post_init__()
        self.concatenate_terms = False
        self.concatenate_dim = -1
        self.enable_corruption = True
        self.history_length = None
        self.flatten_history_dim = True
        self.height_scan = _m14.ObservationTermCfg(func=_m15.height_scan, params={'sensor_cfg': _m14.SceneEntityCfg(name='height_scanner', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=None, body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, modifiers=None, noise=_m16.UniformNoiseCfg(func=_m16.uniform_noise, operation='add', n_min=-0.1, n_max=0.1), clip=(-1.0, 1.0), scale=None, history_length=4, flatten_history_dim=True)
        self.velocity_command = _m14.ObservationTermCfg(func=_m15.generated_commands, params={'command_name': 'base_velocity'}, modifiers=None, noise=None, clip=None, scale=None, history_length=4, flatten_history_dim=True)
        self.projected_gravity = _m14.ObservationTermCfg(func=_m15.projected_gravity, params={}, modifiers=None, noise=_m16.UniformNoiseCfg(func=_m16.uniform_noise, operation='add', n_min=-0.05, n_max=0.05), clip=None, scale=None, history_length=4, flatten_history_dim=True)
        self.base_ang_vel = _m14.ObservationTermCfg(func=_m15.base_ang_vel, params={}, modifiers=None, noise=_m16.UniformNoiseCfg(func=_m16.uniform_noise, operation='add', n_min=-0.2, n_max=0.2), clip=None, scale=None, history_length=4, flatten_history_dim=True)
        self.joint_pos_rel = _m14.ObservationTermCfg(func=_m15.joint_pos_rel, params={}, modifiers=None, noise=_m16.UniformNoiseCfg(func=_m16.uniform_noise, operation='add', n_min=-0.01, n_max=0.01), clip=None, scale=None, history_length=4, flatten_history_dim=True)
        self.joint_vel = _m14.ObservationTermCfg(func=_m15.joint_vel, params={}, modifiers=None, noise=_m16.UniformNoiseCfg(func=_m16.uniform_noise, operation='add', n_min=-0.5, n_max=0.5), clip=None, scale=None, history_length=4, flatten_history_dim=True)
        self.last_action = _m14.ObservationTermCfg(func=_m15.last_action, params={}, modifiers=None, noise=None, clip=None, scale=None, history_length=4, flatten_history_dim=True)


@configclass
class FlatCriticObsCfg_1(_m14.ObservationGroupCfg):

    """`instinctlab.tasks.locomotion.config.g1.flat_env_cfg.G1FlatCriticObsCfg` 的展平副本（链上位置 `observations.critic`；字段值 = 链式配置的最终生效值）。"""
    concatenate_terms: bool = None
    concatenate_dim: int = None
    enable_corruption: bool = None
    history_length: Any = None
    flatten_history_dim: bool = None
    base_lin_vel: Any = None
    base_ang_vel: Any = None
    projected_gravity: Any = None
    velocity_commands: Any = None
    joint_pos: Any = None
    joint_vel: Any = None
    actions: Any = None

    def __post_init__(self):
        super().__post_init__()
        self.concatenate_terms = False
        self.concatenate_dim = -1
        self.enable_corruption = False
        self.history_length = None
        self.flatten_history_dim = True
        self.base_lin_vel = _m14.ObservationTermCfg(func=_m15.base_lin_vel, params={}, modifiers=None, noise=None, clip=None, scale=None, history_length=0, flatten_history_dim=True)
        self.base_ang_vel = _m14.ObservationTermCfg(func=_m15.base_ang_vel, params={}, modifiers=None, noise=None, clip=None, scale=None, history_length=0, flatten_history_dim=True)
        self.projected_gravity = _m14.ObservationTermCfg(func=_m15.projected_gravity, params={}, modifiers=None, noise=None, clip=None, scale=None, history_length=0, flatten_history_dim=True)
        self.velocity_commands = _m14.ObservationTermCfg(func=_m15.generated_commands, params={'command_name': 'base_velocity'}, modifiers=None, noise=None, clip=None, scale=None, history_length=0, flatten_history_dim=True)
        self.joint_pos = _m14.ObservationTermCfg(func=_m15.joint_pos_rel, params={}, modifiers=None, noise=None, clip=None, scale=None, history_length=0, flatten_history_dim=True)
        self.joint_vel = _m14.ObservationTermCfg(func=_m15.joint_vel, params={}, modifiers=None, noise=None, clip=None, scale=None, history_length=0, flatten_history_dim=True)
        self.actions = _m14.ObservationTermCfg(func=_m15.last_action, params={}, modifiers=None, noise=None, clip=None, scale=None, history_length=0, flatten_history_dim=True)


@configclass
class AmpV33PolicyStateObsCfg_1(_m14.ObservationGroupCfg):

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_v33_env_cfg.AmpV33PolicyStateObsCfg` 的展平副本（链上位置 `observations.amp_policy`；字段值 = 链式配置的最终生效值）。"""
    concatenate_terms: bool = None
    concatenate_dim: int = None
    enable_corruption: bool = None
    history_length: Any = None
    flatten_history_dim: bool = None
    body_pos_b: Any = None
    body_ori_b: Any = None
    body_lin_vel_b: Any = None
    body_ang_vel_b: Any = None
    anchor_gravity_b: Any = None

    def __post_init__(self):
        import instinctlab.envs.mdp as _m17  # 局部 import：避开导入期循环依赖
        super().__post_init__()
        self.concatenate_terms = False
        self.concatenate_dim = -1
        self.enable_corruption = False
        self.history_length = None
        self.flatten_history_dim = True
        self.body_pos_b = _m14.ObservationTermCfg(func=_m17.body_pos_b, params={'asset_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link'), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=True), 'anchor_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis',), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)
        self.body_ori_b = _m14.ObservationTermCfg(func=_m17.body_ori_b, params={'asset_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link'), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=True), 'anchor_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis',), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)
        self.body_lin_vel_b = _m14.ObservationTermCfg(func=_m17.body_lin_vel_b, params={'asset_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link'), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=True)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)
        self.body_ang_vel_b = _m14.ObservationTermCfg(func=_m17.body_ang_vel_b, params={'asset_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link'), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=True)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)
        self.anchor_gravity_b = _m14.ObservationTermCfg(func=_m17.anchor_gravity_b, params={'anchor_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis',), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)


@configclass
class AmpV33ReferenceStateObsCfg_1(_m14.ObservationGroupCfg):

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_v33_env_cfg.AmpV33ReferenceStateObsCfg` 的展平副本（链上位置 `observations.amp_reference`；字段值 = 链式配置的最终生效值）。"""
    concatenate_terms: bool = None
    concatenate_dim: int = None
    enable_corruption: bool = None
    history_length: Any = None
    flatten_history_dim: bool = None
    body_pos_b: Any = None
    body_ori_b: Any = None
    body_lin_vel_b: Any = None
    body_ang_vel_b: Any = None
    anchor_gravity_b: Any = None

    def __post_init__(self):
        import instinctlab.envs.mdp as _m17  # 局部 import：避开导入期循环依赖
        super().__post_init__()
        self.concatenate_terms = False
        self.concatenate_dim = -1
        self.enable_corruption = False
        self.history_length = None
        self.flatten_history_dim = True
        self.body_pos_b = _m14.ObservationTermCfg(func=_m17.body_pos_b_reference_as_state, params={'asset_cfg': _m14.SceneEntityCfg(name='motion_reference', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link'), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=True), 'anchor_cfg': _m14.SceneEntityCfg(name='motion_reference', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis',), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)
        self.body_ori_b = _m14.ObservationTermCfg(func=_m17.body_ori_b_reference_as_state, params={'asset_cfg': _m14.SceneEntityCfg(name='motion_reference', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link'), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=True), 'anchor_cfg': _m14.SceneEntityCfg(name='motion_reference', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis',), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)
        self.body_lin_vel_b = _m14.ObservationTermCfg(func=_m17.body_lin_vel_b_reference_as_state, params={'asset_cfg': _m14.SceneEntityCfg(name='motion_reference', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link'), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=True)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)
        self.body_ang_vel_b = _m14.ObservationTermCfg(func=_m17.body_ang_vel_b_reference_as_state, params={'asset_cfg': _m14.SceneEntityCfg(name='motion_reference', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link'), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=True)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)
        self.anchor_gravity_b = _m14.ObservationTermCfg(func=_m17.anchor_gravity_b_reference_as_state, params={'anchor_cfg': _m14.SceneEntityCfg(name='motion_reference', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=('pelvis',), body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, modifiers=None, noise=None, clip=None, scale=None, history_length=2, flatten_history_dim=True)


@configclass
class ObservationsCfg_1:

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_v21_env_cfg.G1AmpWalkV21ObservationsCfg` 的展平副本（链上位置 `observations`；字段值 = 链式配置的最终生效值）。"""
    policy: Any = None  # → T2PolicyObsCfg
    critic: Any = None  # → FlatCriticObsCfg
    amp_policy: Any = None  # → AmpV33PolicyStateObsCfg
    amp_reference: Any = None  # → AmpV33ReferenceStateObsCfg

    def __post_init__(self):
        self.policy = T2PolicyObsCfg_1()
        self.critic = FlatCriticObsCfg_1()
        self.amp_policy = AmpV33PolicyStateObsCfg_1()
        self.amp_reference = AmpV33ReferenceStateObsCfg_1()


@configclass
class ActionsCfg_1:

    """`instinctlab.tasks.locomotion.config.g1.omni_loco_env_cfg.G1OmniLocoActionsCfg` 的展平副本（链上位置 `actions`；字段值 = 链式配置的最终生效值）。"""
    joint_pos: Any = None

    def __post_init__(self):
        self.joint_pos = _m15.JointPositionActionCfg(class_type=_m15.JointPositionAction, asset_name='robot', debug_vis=False, clip=None, joint_names=['.*',], scale=1.0, offset=0.0, preserve_order=False, use_default_offset=True)


@configclass
class EventsCfg_1:

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_t2_env_cfg.G1AmpWalkT2EventsCfg` 的展平副本（链上位置 `events`；字段值 = 链式配置的最终生效值）。"""
    physics_material: Any = None
    base_com: Any = None
    push_robot: Any = None
    reset_from_reference: Any = None

    def __post_init__(self):
        self.physics_material = _m14.EventTermCfg(func=_m15.randomize_rigid_body_material, params={'asset_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names='.*', body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False), 'static_friction_range': (0.25, 0.8), 'dynamic_friction_range': (0.2, 0.6), 'restitution_range': (0.0, 0.8), 'num_buckets': 64}, mode='startup', interval_range_s=None, is_global_time=False, min_step_count_between_reset=0)
        self.base_com = _m14.EventTermCfg(func=_m15.randomize_rigid_body_com, params={'asset_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names='torso_link', body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False), 'com_range': {'x': (-0.025, 0.025), 'y': (-0.025, 0.025), 'z': (-0.03, 0.03)}}, mode='startup', interval_range_s=None, is_global_time=False, min_step_count_between_reset=0)
        self.push_robot = None
        self.reset_from_reference = None


@configclass
class TeleopDevicesCfg_1:

    """`isaaclab.devices.device_base.DevicesCfg` 的展平副本（链上位置 `teleop_devices`；字段值 = 链式配置的最终生效值）。"""
    devices: Any = None

    def __post_init__(self):
        self.devices = {}


@configclass
class RewardsCfg_1:

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_t2_env_cfg.G1AmpWalkT2RewardsCfg` 的展平副本（链上位置 `rewards`；字段值 = 链式配置的最终生效值）。"""
    termination_penalty: Any = None
    track_lin_vel_xy_exp: Any = None
    track_ang_vel_z_exp: Any = None
    feet_air_time: Any = None
    feet_slide: Any = None
    flat_orientation_l2: Any = None
    stand_still: Any = None
    dof_pos_limits: Any = None
    joint_deviation_hip: Any = None
    joint_deviation_arms: Any = None
    joint_deviation_torso: Any = None
    joint_deviation_knee: Any = None
    lin_vel_z_l2: Any = None
    action_rate_l2: Any = None
    dof_acc_l2: Any = None
    dof_torques_l2: Any = None
    track_base_height: Any = None
    body_ang_vel_xy: Any = None
    track_default_pose_on_stand: Any = None
    feet_lateral_width: Any = None
    feet_crossing: Any = None
    track_anchor_linear_velocity: Any = None
    track_anchor_angular_velocity: Any = None

    def __post_init__(self):
        import instinctlab.envs.mdp as _m17  # 局部 import：避开导入期循环依赖
        import instinctlab.tasks.locomotion.mdp as _m18  # 局部 import：避开导入期循环依赖
        self.termination_penalty = _m14.RewardTermCfg(func=_m15.is_terminated, params={}, weight=-200.0)
        self.track_lin_vel_xy_exp = None
        self.track_ang_vel_z_exp = None
        self.feet_air_time = None
        self.feet_slide = _m14.RewardTermCfg(func=_m17.contact_slide, params={'sensor_cfg': _m14.SceneEntityCfg(name='contact_forces', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names='.*_ankle_roll_link', body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False), 'asset_cfg': _m14.SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names='.*_ankle_roll_link', body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, weight=-0.25)
        self.flat_orientation_l2 = None
        self.stand_still = None
        self.dof_pos_limits = _m14.RewardTermCfg(func=_m15.joint_pos_limits, params={'asset_cfg': _m14.SceneEntityCfg(name='robot', joint_names=['.*',], joint_ids=slice(None, None, None), fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=None, body_ids=slice(None, None, None), object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, weight=-10.0)
        self.joint_deviation_hip = None
        self.joint_deviation_arms = None
        self.joint_deviation_torso = None
        self.joint_deviation_knee = None
        self.lin_vel_z_l2 = None
        self.action_rate_l2 = _m14.RewardTermCfg(func=_m15.action_rate_l2, params={}, weight=-0.01)
        self.dof_acc_l2 = _m14.RewardTermCfg(func=_m15.joint_acc_l2, params={}, weight=-2.5e-07)
        self.dof_torques_l2 = None
        self.track_base_height = _m14.RewardTermCfg(func=_m18.track_base_height_exp, params={'std': 0.3}, weight=1.5)
        self.body_ang_vel_xy = _m14.RewardTermCfg(func=_m18.track_root_ang_vel_xy_exp, params={'std': 3.14}, weight=0.7)
        self.track_default_pose_on_stand = _m14.RewardTermCfg(func=_m18.track_default_pose_on_stand, params={'command_name': 'base_velocity', 'command_threshold': 0.1, 'pos_scale': 4.0, 'vel_scale': 0.5, 'gate_tau': 0.25, 'pos_joint_weights': {'waist_yaw_joint': 3.0}}, weight=-1.0)
        self.feet_lateral_width = _m14.RewardTermCfg(func=_m18.feet_lateral_width_cost, params={'command_name': 'base_velocity', 'command_threshold': 0.1, 'sep_floor': 0.15, 'gate_lateral': True, 'max_cost': 0.1, 'walk_max_lin_vel': 1.0, 'walk_gate_width': 0.2}, weight=-10.0)
        self.feet_crossing = _m14.RewardTermCfg(func=_m18.feet_crossing_cost, params={'command_name': 'base_velocity', 'command_threshold': 0.1, 'gate_lateral': True, 'max_cost': 0.1, 'walk_max_lin_vel': 1.0, 'walk_gate_width': 0.2}, weight=-20.0)
        self.track_anchor_linear_velocity = _m14.RewardTermCfg(func=_m18.track_anchor_lin_vel_exp, params={'command_name': 'base_velocity', 'std': 1.0}, weight=1.5)
        self.track_anchor_angular_velocity = _m14.RewardTermCfg(func=_m18.track_anchor_ang_vel_exp, params={'command_name': 'base_velocity', 'std': 3.14}, weight=1.2)


@configclass
class TerminationsCfg_1:

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_t2_env_cfg.G1AmpWalkT2TerminationsCfg` 的展平副本（链上位置 `terminations`；字段值 = 链式配置的最终生效值）。"""
    time_out: Any = None
    bad_orientation: Any = None
    bad_base_height: Any = None

    def __post_init__(self):
        self.time_out = _m14.TerminationTermCfg(func=_m15.time_out, params={}, time_out=True)
        self.bad_orientation = _m14.TerminationTermCfg(func=_m15.bad_orientation, params={'limit_angle': 1.2217304763960306}, time_out=False)
        self.bad_base_height = _m14.TerminationTermCfg(func=_m15.root_height_below_minimum, params={'minimum_height': 0.5}, time_out=False)


@configclass
class CurriculumCfg_1:

    """`instinctlab.tasks.locomotion.config.g1.flat_env_cfg.G1FlatCurriculumCfg` 的展平副本（链上位置 `curriculum`；字段值 = 链式配置的最终生效值）。"""

    def __post_init__(self):
        pass


@configclass
class DirectionalVelocityCommandCfg_1(_m15.UniformVelocityCommandCfg):

    """`instinctlab.tasks.locomotion.mdp.commands.DirectionalVelocityCommandCfg` 的展平副本（链上位置 `commands.base_velocity`；字段值 = 链式配置的最终生效值）。"""
    class_type: Any = None
    resampling_time_range: Any = None
    debug_vis: bool = None
    asset_name: str = None
    heading_command: bool = None
    heading_control_stiffness: float = None
    rel_standing_envs: float = None
    rel_heading_envs: float = None
    ranges: Any = None
    goal_vel_visualizer_cfg: Any = None
    current_vel_visualizer_cfg: Any = None
    prob_low: float = None
    prob_side: float = None
    prob_lin: float = None
    prob_back: float = None
    side_ang_vel_range: Any = None
    side_vel_min: float = None
    low_vel: float = None
    zero_frac_in_low: float = None
    prob_lin_fwd: float = None
    straight_lin_vel_range: Any = None
    straight_ang_vel_range: Any = None
    back_lin_vel_range: Any = None
    back_ang_vel_range: Any = None

    def __post_init__(self):
        import isaaclab.sim as _m1  # 局部 import：避开导入期循环依赖
        import instinctlab.tasks.locomotion.mdp as _m18  # 局部 import：避开导入期循环依赖
        import isaaclab.markers as _m7  # 局部 import：避开导入期循环依赖
        super().__post_init__()
        self.class_type = _m18.DirectionalVelocityCommand
        self.resampling_time_range = (6.0, 15.0)
        self.debug_vis = True
        self.asset_name = 'robot'
        self.heading_command = False
        self.heading_control_stiffness = 1.0
        self.rel_standing_envs = 0.0
        self.rel_heading_envs = 0.0
        self.ranges = _m15.UniformVelocityCommandCfg.Ranges(lin_vel_x=(-0.5, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.57, 1.57), heading=(-3.141592653589793, 3.141592653589793))
        self.goal_vel_visualizer_cfg = _m7.VisualizationMarkersCfg(prim_path='/Visuals/Command/velocity_goal', markers={'arrow': _m1.UsdFileCfg(func=_m1.spawn_from_usd, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, deformable_props=None, rigid_props=None, collision_props=None, activate_contact_sensors=False, scale=(0.5, 0.5, 0.5), articulation_props=None, fixed_tendons_props=None, spatial_tendons_props=None, joint_drive_props=None, visual_material_path='material', visual_material=_m1.PreviewSurfaceCfg(func=_m1.spawn_preview_surface, diffuse_color=(0.0, 1.0, 0.0), emissive_color=(0.0, 0.0, 0.0), roughness=0.5, metallic=0.0, opacity=1.0), usd_path='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/UIElements/arrow_x.usd', variants=None)})
        self.current_vel_visualizer_cfg = _m7.VisualizationMarkersCfg(prim_path='/Visuals/Command/velocity_current', markers={'arrow': _m1.UsdFileCfg(func=_m1.spawn_from_usd, visible=True, semantic_tags=None, copy_from_source=True, mass_props=None, deformable_props=None, rigid_props=None, collision_props=None, activate_contact_sensors=False, scale=(0.5, 0.5, 0.5), articulation_props=None, fixed_tendons_props=None, spatial_tendons_props=None, joint_drive_props=None, visual_material_path='material', visual_material=_m1.PreviewSurfaceCfg(func=_m1.spawn_preview_surface, diffuse_color=(0.0, 0.0, 1.0), emissive_color=(0.0, 0.0, 0.0), roughness=0.5, metallic=0.0, opacity=1.0), usd_path='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/UIElements/arrow_x.usd', variants=None)})
        self.prob_low = 0.2
        self.prob_side = 0.2
        self.prob_lin = 0.4
        self.prob_back = 0.1
        self.side_ang_vel_range = None
        self.side_vel_min = 0.0
        self.low_vel = 0.1
        self.zero_frac_in_low = 0.6
        self.prob_lin_fwd = 0.7
        self.straight_lin_vel_range = (0.4, 1.0)
        self.straight_ang_vel_range = (-0.1, 0.1)
        self.back_lin_vel_range = (-0.7, -0.4)
        self.back_ang_vel_range = (-0.1, 0.1)


@configclass
class CommandsCfg_1:

    """`instinctlab.tasks.locomotion.config.g1.amp_walk_v3_env_cfg.G1AmpWalkV3CommandsCfg` 的展平副本（链上位置 `commands`；字段值 = 链式配置的最终生效值）。"""
    base_velocity: Any = None  # → DirectionalVelocityCommandCfg

    def __post_init__(self):
        self.base_velocity = DirectionalVelocityCommandCfg_1()


@configclass
class MonitorTermCfg_1(_m14.ManagerTermBaseCfg):

    """`instinctlab.monitors.monitor_cfg.MonitorTermCfg` 的展平副本（链上位置 `monitors.tracking`；字段值 = 链式配置的最终生效值）。"""
    func: Any = None
    params: Any = None

    def __post_init__(self):
        import instinctlab.tasks.locomotion.mdp as _m18  # 局部 import：避开导入期循环依赖
        super().__post_init__()
        self.func = _m18.LocoTrackingMonitorTerm
        self.params = {}


@configclass
class MonitorsCfg_1:

    """`instinctlab.tasks.locomotion.config.g1.omni_loco_env_cfg.G1OmniLocoMonitorCfg` 的展平副本（链上位置 `monitors`；字段值 = 链式配置的最终生效值）。"""
    tracking: Any = None  # → MonitorTermCfg

    def __post_init__(self):
        self.tracking = MonitorTermCfg_1()


@configclass
class G1AmpWalkP1EnvCfg(_m0.ManagerBasedRLEnvCfg):
    """`instinctlab.tasks.locomotion.config.g1.amp_walk_pelvis_env_cfg:G1AmpWalkPelvisEnvCfg` 的**展平**版本：不继承本项目任何其它配置类。

    由 `instinct/tools/flatten_cfg.py` 从链式配置的**解析结果**生成，上游基类 = `isaaclab.envs.manager_based_rl_env_cfg.ManagerBasedRLEnvCfg` (链长 47 层)。
    """

    viewer: Any = None  # → ViewerCfg
    sim: Any = None  # → SimCfg
    scene: Any = None  # → SceneCfg
    recorders: Any = None  # → RecordersCfg
    observations: Any = None  # → ObservationsCfg
    actions: Any = None  # → ActionsCfg
    events: Any = None  # → EventsCfg
    teleop_devices: Any = None  # → TeleopDevicesCfg
    rewards: Any = None  # → RewardsCfg
    terminations: Any = None  # → TerminationsCfg
    curriculum: Any = None  # → CurriculumCfg
    commands: Any = None  # → CommandsCfg
    monitors: Any = None  # → MonitorsCfg
    ui_window_class_type: Any = None
    seed: Any = None
    decimation: int = None
    rerender_on_reset: bool = None
    num_rerenders_on_reset: int = None
    wait_for_textures: bool = None
    xr: Any = None
    export_io_descriptors: bool = None
    log_dir: Any = None
    is_finite_horizon: bool = None
    episode_length_s: float = None
    run_name: str = None

    def __post_init__(self):
        import isaaclab.envs.ui as _m19  # 局部 import：避开导入期循环依赖
        super().__post_init__()  # 只跑上游（ManagerBasedRLEnvCfg）的钩子
        """全部字段按链式配置的**最终生效值**写死。"""
        self.viewer = ViewerCfg()
        self.sim = SimCfg()
        self.scene = SceneCfg()
        self.recorders = RecordersCfg()
        self.observations = ObservationsCfg()
        self.actions = ActionsCfg()
        self.events = EventsCfg()
        self.teleop_devices = TeleopDevicesCfg()
        self.rewards = RewardsCfg()
        self.terminations = TerminationsCfg()
        self.curriculum = CurriculumCfg()
        self.commands = CommandsCfg()
        self.monitors = MonitorsCfg()
        self.ui_window_class_type = _m19.ManagerBasedRLEnvWindow
        self.seed = None
        self.decimation = 4
        self.rerender_on_reset = False
        self.num_rerenders_on_reset = 0
        self.wait_for_textures = True
        self.xr = None
        self.export_io_descriptors = False
        self.log_dir = None
        self.is_finite_horizon = False
        self.episode_length_s = 20.0
        self.run_name = 'p1_pelvis_official_rev10_cold'

        # ---- 展平自检锚点：run_name（链式 vs 展平必须一致，diff 会抓）----
        self.run_name = 'p1_pelvis_official_rev10_cold'


@configclass
class G1AmpWalkP1EnvCfg_PLAY(_m0.ManagerBasedRLEnvCfg):
    """`instinctlab.tasks.locomotion.config.g1.amp_walk_pelvis_env_cfg:G1AmpWalkPelvisEnvCfg_PLAY` 的**展平**版本：不继承本项目任何其它配置类。

    由 `instinct/tools/flatten_cfg.py` 从链式配置的**解析结果**生成，上游基类 = `isaaclab.envs.manager_based_rl_env_cfg.ManagerBasedRLEnvCfg` (链长 48 层)。
    """

    viewer: Any = None  # → ViewerCfg_1
    sim: Any = None  # → SimCfg_1
    scene: Any = None  # → SceneCfg_1
    recorders: Any = None  # → RecordersCfg_1
    observations: Any = None  # → ObservationsCfg_1
    actions: Any = None  # → ActionsCfg_1
    events: Any = None  # → EventsCfg_1
    teleop_devices: Any = None  # → TeleopDevicesCfg_1
    rewards: Any = None  # → RewardsCfg_1
    terminations: Any = None  # → TerminationsCfg_1
    curriculum: Any = None  # → CurriculumCfg_1
    commands: Any = None  # → CommandsCfg_1
    monitors: Any = None  # → MonitorsCfg_1
    ui_window_class_type: Any = None
    seed: Any = None
    decimation: int = None
    rerender_on_reset: bool = None
    num_rerenders_on_reset: int = None
    wait_for_textures: bool = None
    xr: Any = None
    export_io_descriptors: bool = None
    log_dir: Any = None
    is_finite_horizon: bool = None
    episode_length_s: float = None
    run_name: str = None

    def __post_init__(self):
        import isaaclab.envs.ui as _m19  # 局部 import：避开导入期循环依赖
        super().__post_init__()  # 只跑上游（ManagerBasedRLEnvCfg）的钩子
        """全部字段按链式配置的**最终生效值**写死。"""
        self.viewer = ViewerCfg_1()
        self.sim = SimCfg_1()
        self.scene = SceneCfg_1()
        self.recorders = RecordersCfg_1()
        self.observations = ObservationsCfg_1()
        self.actions = ActionsCfg_1()
        self.events = EventsCfg_1()
        self.teleop_devices = TeleopDevicesCfg_1()
        self.rewards = RewardsCfg_1()
        self.terminations = TerminationsCfg_1()
        self.curriculum = CurriculumCfg_1()
        self.commands = CommandsCfg_1()
        self.monitors = MonitorsCfg_1()
        self.ui_window_class_type = _m19.ManagerBasedRLEnvWindow
        self.seed = None
        self.decimation = 4
        self.rerender_on_reset = False
        self.num_rerenders_on_reset = 0
        self.wait_for_textures = True
        self.xr = None
        self.export_io_descriptors = False
        self.log_dir = None
        self.is_finite_horizon = False
        self.episode_length_s = 20.0
        self.run_name = 'p1_pelvis_official_rev10_cold'

        # ---- 展平自检锚点：run_name（链式 vs 展平必须一致，diff 会抓）----
        self.run_name = 'p1_pelvis_official_rev10_cold'
