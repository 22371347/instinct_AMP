仿真代码：
source /home/leo/miniconda3/etc/profile.d/conda.sh && conda activate instinct

python instinct/tools/g1_mujoco_play.py --policy /home/leo/rsl_project/llp/instinct/eval/vt2_11500/actor.onnx --duration 0


python instinct/tools/g1_mujoco_play.py \
  --policy instinct/eval/v11_9000/actor.onnx --duration 0

python instinct/tools/g1_mujoco_play.py --policy /home/leo/rsl_project/llp/instinct/eval/v10b_8500/actor.onnx --duration 0

python instinct/tools/g1_mujoco_play.py --policy /home/leo/rsl_project/llp/instinct/eval/v11_9000/actor.onnx --duration 0

传输图像：
bash instinct/tools/refresh_plots.sh  #10分钟传一次
bash instinct/tools/refresh_plots.sh once  #传一次
bash instinct/tools/refresh_plots.sh stop  
bash instinct/tools/refresh_plots.sh list          # 列出云端所有 run（名字/时间/最新 ckpt）
bash instinct/tools/refresh_plots.sh once t5
      # 只画 v9c，产物进 instinct/plots/v9c/
bash instinct/tools/refresh_plots.sh loop t3
bash instinct/tools/refresh_plots.sh once G1AmpWalkV9MVPPlane   # 也接受更长的子串

onnx回传：
bash instinct/tools/pull_latest_onnx.sh
bash instinct/tools/pull_latest_onnx.sh v9c --list          # 先看会选哪份（不占 GPU）
bash instinct/tools/pull_latest_onnx.sh t4
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

arc_walk_left_A029 
walk_backward_A022
walk_forward_A022
walk_sideway_left_A021