# G1 29-DoF 关节序 & 执行器参数参考（自动生成，勿手改）

生成命令: `python tools/g1_params.py > docs/g1_reference.md` 相关脚本
来源: InstinctMJ `src/instinct_mj/assets/unitree_g1.py`

## 关节序对照

```
--- 顺序对照 ---
  #  IsaacLab (policy)              MuJoCo (ctrl)                 
  0  left_shoulder_pitch_joint      waist_pitch_joint             
  1  right_shoulder_pitch_joint     waist_roll_joint              
  2  waist_pitch_joint              waist_yaw_joint               
  3  left_shoulder_roll_joint       left_hip_pitch_joint          
  4  right_shoulder_roll_joint      left_hip_roll_joint           
  5  waist_roll_joint               left_hip_yaw_joint            
  6  left_shoulder_yaw_joint        left_knee_joint               
  7  right_shoulder_yaw_joint       left_ankle_pitch_joint        
  8  waist_yaw_joint                left_ankle_roll_joint         
  9  left_elbow_joint               right_hip_pitch_joint         
 10  right_elbow_joint              right_hip_roll_joint          
 11  left_hip_pitch_joint           right_hip_yaw_joint           
 12  right_hip_pitch_joint          right_knee_joint              
 13  left_wrist_roll_joint          right_ankle_pitch_joint       
 14  right_wrist_roll_joint         right_ankle_roll_joint        
 15  left_hip_roll_joint            left_shoulder_pitch_joint     
 16  right_hip_roll_joint           left_shoulder_roll_joint      
 17  left_wrist_pitch_joint         left_shoulder_yaw_joint       
 18  right_wrist_pitch_joint        left_elbow_joint              
 19  left_hip_yaw_joint             left_wrist_roll_joint         
 20  right_hip_yaw_joint            left_wrist_pitch_joint        
 21  left_wrist_yaw_joint           left_wrist_yaw_joint          
 22  right_wrist_yaw_joint          right_shoulder_pitch_joint    
 23  left_knee_joint                right_shoulder_roll_joint     
 24  right_knee_joint               right_shoulder_yaw_joint      
 25  left_ankle_pitch_joint         right_elbow_joint             
 26  right_ankle_pitch_joint        right_wrist_roll_joint        
 27  left_ankle_roll_joint          right_wrist_pitch_joint       
 28  right_ankle_roll_joint         right_wrist_yaw_joint         

--- policy 向量 → mujoco ctrl 索引 (policy_to_mujoco_index) ---
```

## 逐关节执行器参数

```
NATURAL_FREQ = 62.831853 rad/s (10 Hz)
STIFFNESS_7520_14=40.1792  STIFFNESS_7520_22=99.0984
STIFFNESS_5020   =14.2506  STIFFNESS_4010   =16.7783

  # joint                      group                  kp      kd    eff  vlim  a_scale    init
----------------------------------------------------------------------------------------------
  0 waist_pitch_joint          waist              28.501  1.8144     50    37  0.43858   0.000
  1 waist_roll_joint           waist              28.501  1.8144     50    37  0.43858   0.000
  2 waist_yaw_joint            waist_yaw          40.179  2.5579     88    32  0.54755   0.000
  3 left_hip_pitch_joint       legs_pitch_yaw     40.179  2.5579     88    32  0.54755  -0.312
  4 left_hip_roll_joint        legs_roll_knee     99.098  6.3088    139    20  0.35066   0.000
  5 left_hip_yaw_joint         legs_pitch_yaw     40.179  2.5579     88    32  0.54755   0.000
  6 left_knee_joint            legs_roll_knee     99.098  6.3088    139    20  0.35066   0.669
  7 left_ankle_pitch_joint     feet               28.501  1.8144     50    37  0.43858  -0.363
  8 left_ankle_roll_joint      feet               28.501  1.8144     50    37  0.43858   0.000
  9 right_hip_pitch_joint      legs_pitch_yaw     40.179  2.5579     88    32  0.54755  -0.312
 10 right_hip_roll_joint       legs_roll_knee     99.098  6.3088    139    20  0.35066   0.000
 11 right_hip_yaw_joint        legs_pitch_yaw     40.179  2.5579     88    32  0.54755   0.000
 12 right_knee_joint           legs_roll_knee     99.098  6.3088    139    20  0.35066   0.669
 13 right_ankle_pitch_joint    feet               28.501  1.8144     50    37  0.43858  -0.363
 14 right_ankle_roll_joint     feet               28.501  1.8144     50    37  0.43858   0.000
 15 left_shoulder_pitch_joint  arms_5020          14.251  0.9072     25    37  0.43858   0.200
 16 left_shoulder_roll_joint   arms_5020          14.251  0.9072     25    37  0.43858   0.200
 17 left_shoulder_yaw_joint    arms_5020          14.251  0.9072     25    37  0.43858   0.000
 18 left_elbow_joint           arms_5020          14.251  0.9072     25    37  0.43858   0.600
 19 left_wrist_roll_joint      arms_5020          14.251  0.9072     25    37  0.43858   0.000
 20 left_wrist_pitch_joint     arms_4010          16.778  1.0681      5    22  0.07450   0.000
 21 left_wrist_yaw_joint       arms_4010          16.778  1.0681      5    22  0.07450   0.000
 22 right_shoulder_pitch_joint arms_5020          14.251  0.9072     25    37  0.43858   0.200
 23 right_shoulder_roll_joint  arms_5020          14.251  0.9072     25    37  0.43858  -0.200
 24 right_shoulder_yaw_joint   arms_5020          14.251  0.9072     25    37  0.43858   0.000
 25 right_elbow_joint          arms_5020          14.251  0.9072     25    37  0.43858   0.600
 26 right_wrist_roll_joint     arms_5020          14.251  0.9072     25    37  0.43858   0.000
 27 right_wrist_pitch_joint    arms_4010          16.778  1.0681      5    22  0.07450   0.000
 28 right_wrist_yaw_joint      arms_4010          16.778  1.0681      5    22  0.07450   0.000
```
