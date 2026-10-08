仿真代码：
source /home/leo/miniconda3/etc/profile.d/conda.sh && conda activate instinct

python instinct/tools/g1_mujoco_play.py \
  --policy instinct/eval/v1_500/actor.onnx --duration 0

python instinct/tools/g1_mujoco_play.py \
  --policy instinct/eval/v11_9000/actor.onnx --duration 0

python instinct/tools/g1_mujoco_play.py --policy /home/leo/rsl_project/llp/instinct/eval/v10b_8500/actor.onnx --duration 0

python instinct/tools/g1_mujoco_play.py --policy /home/leo/rsl_project/llp/instinct/eval/v11_9000/actor.onnx --duration 0

传输图像：
bash instinct/tools/refresh_plots.sh  #10分钟传一次
bash instinct/tools/refresh_plots.sh once  #传一次
bash instinct/tools/refresh_plots.sh stop  
bash instinct/tools/refresh_plots.sh list          # 列出云端所有 run（名字/时间/最新 ckpt）
bash instinct/tools/refresh_plots.sh once v9c      # 只画 v9c，产物进 instinct/plots/v9c/
bash instinct/tools/refresh_plots.sh loop t1i
bash instinct/tools/refresh_plots.sh once G1AmpWalkV9MVPPlane   # 也接受更长的子串

onnx回传：
bash instinct/tools/pull_latest_onnx.sh
bash instinct/tools/pull_latest_onnx.sh v9c --list          # 先看会选哪份（不占 GPU）
bash instinct/tools/pull_latest_onnx.sh t1i
# 真拉：导出+拉回 instinct/eval/v9c_<step>/
bash instinct/tools/pull_latest_onnx.sh t1a2 model_2000.pt  
# 也可指定 checkpoint
bash instinct/tools/pull_latest_onnx.sh t1a2_t0impl_t0explore model_5500.pt

ssh与密码：
ssh -p 30843 root@183.147.142.40
(x69*v)sg53nOo%f

ssh -p 31392 root@183.147.142.40 
3m9@u#skN!dz57C




