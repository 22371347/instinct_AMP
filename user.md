仿真代码：
source /home/leo/miniconda3/etc/profile.d/conda.sh && conda activate instinct

python instinct/tools/g1_mujoco_play.py --policy /home/leo/rsl_project/llp/instinct/eval/vt2_11500/actor.onnx --duration 0


python instinct/tools/g1_mujoco_play.py \
  --policy instinct/eval/v11_9000/actor.onnx --duration 0

python instinct/tools/g1_mujoco_play.py --policy /home/leo/rsl_project/llp/instinct/eval/v10b_8500/actor.onnx --duration 0

python instinct/tools/g1_mujoco_play.py --policy /home/leo/rsl_project/llp/instinct/eval/p1_6500/actor.onnx --duration 0

传输图像：
bash instinct/tools/refresh_plots.sh  #10分钟传一次
bash instinct/tools/refresh_plots.sh once  #传一次
bash instinct/tools/refresh_plots.sh stop  
bash instinct/tools/refresh_plots.sh list          # 列出云端所有 run（名字/时间/最新 ckpt）
bash instinct/tools/refresh_plots.sh once p1
      # 只画 v9c，产物进 instinct/plots/v9c/
bash instinct/tools/refresh_plots.sh loop t3
bash instinct/tools/refresh_plots.sh once G1AmpWalkV9MVPPlane   # 也接受更长的子串

onnx回传：
bash instinct/tools/pull_latest_onnx.sh
bash instinct/tools/pull_latest_onnx.sh v9c --list          # 先看会选哪份（不占 GPU）
bash instinct/tools/pull_latest_onnx.sh p1
# → instinct/eval/p1_<step>/（会自动用 Instinct-Amp-Walk-Pelvis-G1-Play-v0 导出）
# 回放（P1 = 官方 rev1.0，pelvis 为根；**必须带 --asset rev10**，否则关节序错位）：
python instinct/tools/g1_mujoco_play.py --asset rev10 --policy instinct/eval/p1_5000/actor.onnx --duration 0
#   --asset 会一次切齐：play MJCF、扫描挂载 body(pelvis)、观测关节序(p1)、动作关节序(p1)
#   V1~T5 的老策略不加 --asset 即可（默认 legacy，行为与以前完全一致）
# 真拉：导出+拉回 instinct/eval/v9c_<step>/
bash instinct/tools/pull_latest_onnx.sh t1a2 model_2000.pt  
# 也可指定 checkpoint
bash instinct/tools/pull_latest_onnx.sh t1a2_t0impl_t0explore model_5500.pt

ssh与密码：
ssh -p 30843 root@183.147.142.40
(x69*v)sg53nOo%f

ssh -p 31392 root@183.147.142.40 
3m9@u#skN!dz57C


逐个 clip 的回放指令（17 个）
cd /home/leo/rsl_project/llp
source /home/leo/miniconda3/etc/profile.d/conda.sh && conda activate instinct

# ① 单个 clip（--clip 给**子串**即可）
python3 instinct/tools/render_ref_motion.py --clip walk_forward   --repeat 3
python3 instinct/tools/render_ref_motion.py --clip sideway_left   --repeat 3
python3 instinct/tools/render_ref_motion.py --clip jog_forward    --repeat 3
python3 instinct/tools/render_ref_motion.py --clip arc_jog_left   --repeat 3
# …子串用下表第 1 列（如 arc_walk_left / idle_turn_270 / step_rotate / walk_arc_cw / backward / A024 …）

# ② 批量：v13 = **我们训练实际用的 5 个**（已渲好一份在 instinct/eval/ref_clips/）
python3 instinct/tools/render_ref_motion.py --selection amp_walk_v13_motion_selection.yaml --repeat 3

# ③ 批量：**全集 17 个**
python3 instinct/tools/render_ref_motion.py --selection amp_walk_motion_selection.yaml --repeat 2 \
        --outdir instinct/eval/ref_clips17

# ④ 排查专用：**两种 base 体假设对照**（下面第三节解释）
python3 instinct/tools/render_ref_motion.py --clip walk_forward --base-body pelvis --repeat 2 --outdir /tmp/ref_pelvis

# ⑤ 只要**定量表**（不渲染，最快，排查用这个）
python3 instinct/tools/render_ref_motion.py --selection amp_walk_motion_selection.yaml --table

python instinct/tools/g1_mujoco_play.py --asset rev10 --policy /home/leo/rsl_project/llp/instinct/eval/p1_2500/actor.onnx --duration 0

# 现在直接这样跑就行（不用再手动加 --asset）
python instinct/tools/g1_mujoco_play.py --policy instinct/eval/p1_2000/actor.onnx \
    --cmd -0.0 0 -0.0 --duration 15



arc_walk_left_A029 
walk_backward_A022
walk_forward_A022
walk_sideway_left_A021



本周计划：1. 拓展数据集与指令集，开展 AMP 运控策略全量训练：在现有单个数据集基础上补齐侧向、后退、转向等运动模态与对应指令分布，训练覆盖全向运动的行走专家，目标是得到可直接对接后续蒸馏等步骤的行走专家模型。
2. 开始向复杂地形等方向探索：阅读并梳理相关论文（复杂地形运动、地形课程学习、教师-学生蒸馏等），分析技术路线，为后续综合训练"具备复杂地形越障技能 + 人类运动风格"的专家策略做准备。

任务完成度：
1、成功开展AMP 运控策略全量指令训练，效果达到预期。目前可实现在平地环境下的全向指令策略的训练和仿真。但发现了所用项目的模型文件与标准文件不同，重新调整模型后，迁移代码成功完成了重训。
2、阅读调研论文Deep Whole-Body Parkour、T-GMP: Terrain-conditioned Generative Motion Priors for Versatile and Natural Humanoid Locomotion、Article HAML: Humanoid Adversarial Multi-Skill Learning via a Single Policy。对多地型、多技能的对抗式算法AMP进行了分析学习，针对多出现的风格踏陷问题及其对策进行了解。
3、蒸馏方面获取了PHP的开源代码并进行了初步分析，但与我们的任务切合度不太高，需要对蒸馏思路进行整理，准备着手开展第一轮蒸馏尝试。


除去以上研究性工作外，关于事务性工作（太极计算机方面那个论文管理平台的注册）可能需要部分资料。
目前完成了大部分硕博研究生和部分老师的信息采集，并按照微信群内备注了名单的成员进行了初步核对。但我这里缺少所有人的总名单，不确定群内是否有未备注的硕博研究生难以进行精确核对。

其项目中不包括视觉部分，修改工程量较大，暂时搁置该方案。依然需要我们本地进行mujoco回放。根据我的部署回放相关经验，模型当前出现高频、到达限位抽搐。可能会在以下方面出现问题，请和我一起进行测试：
1、关节序错乱。检验测试方法：令模型按训练配置的“default_pose”数组进行站立回放，肉眼确认其位置
2、由于关节序改变了，default_pose、offset、刚度等数组是否也需要进行对应顺序的调整。
3、模型的关节变化频率很高，你提到步长为200Hz。仿真器确实如此，但是模型推理的step_dt通常设置为50Hz,可以进行确认

经过观测，关节序应该问题较小，且虽然高频抽搐，但仍有保持站立稳态的趋势。说明顺序可能是正确的。但PhysX与MuJoCo的物理差异不会相差到这样明显的程度。可能的其他待排查项如下：
1、确认回放脚本确实是使用g1_29dof_rev的模型
2、对云端的pt尝试拉取回放视频，根据其回放视频验证问题出现的环节

很遗憾，我认为其不可能是脚掌模型的问题。证据如下：对以往的训练模型，如果呈现摔倒等非脚掌触的模态，也只会进行低频的挣扎或者静止。但目前回放效果是摔倒后会呈现高频而剧烈的挣扎。
1、但是我认为你用训练URDF重建play依然是具有价值的，可以便于控制变量检测
2、我想尝试更底层的测试方法，例如让我们输入给定的观测值，对比云端.pt以及本地onnx模型的输出区别是否存在不同
请执行你认为正确的行动

无需定位了，我们继续执行任务
1、我们现在来训练一个用于蒸馏测试的越障专家数据集，可以直接修改instinct_xu/source/instinctlab/instinctlab/tasks/shadowing/beyondmimic/config/g1/beyondmimic_vault_cfg.py文件，主要修改motion_reference_cfg = MotionReferenceManagerCfg应该就可以了
2、修改完成后我们需要回放instinct_xu/source/instinctlab/instinctlab/tasks/shadowing/beyondmimic/config/g1/motions/vault/g1_speed_vault_retargeted.npz数据集进行验证，导出数据集replay视频人工核对
3、AMP专家任务已经初步完成。我建立了一个还未初始化的github仓库https://github.com/22371347/instinct_AMP.git，我们把当前的仓库上传备份后，可以移除本地大多数确定不再使用的版本配置文件、plots曲线、tool脚本，仅把最终版本p1和训练过程中的调试经验进行保留，对确认有误/不再有效的.md笔记文件也可以归档或者移除。同时由于各个env_cfg间有较复杂的嵌套和继承关系，我希望能在p1的配置脚本中进行整理，不继承我们其他AMP版本的资产而是优化为一个完整独立的配置文件。



