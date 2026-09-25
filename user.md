仿真代码：
source /home/leo/miniconda3/etc/profile.d/conda.sh && conda activate instinct
python instinct/tools/g1_mujoco_play.py \
  --policy instinct/eval/v2_7000/actor.onnx --duration 0


传输图像：
bash instinct/tools/refresh_plots.sh  #10分钟传一次
bash instinct/tools/refresh_plots.sh once  #传一次
bash instinct/tools/refresh_plots.sh stop  

onnx回传：
bash instinct/tools/pull_latest_onnx.sh
