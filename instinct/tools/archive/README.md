# 归档目录（instinct/tools/archive）

这里放的是**已经完成任务、不再参与流水线**的一次性脚本。刻意**不删除**：
它们记录了当时的实验设计与排查路径（哪个假设、怎么验的），事后复盘很有用。

需要复原时直接把文件移回上一级即可，例如：

```bash
git mv instinct/tools/archive/cloud_oneoff/cloud_probe_rewards.py instinct/tools/
```

## cloud_oneoff/（28 个）

一次性云端探针 / 审计 / 消融脚本：`cloud_probe_*`、`cloud_dump_*`、`cloud_ab_*`、
`cloud_chain_*`、`cloud_v9_diag*`、`cloud_scale_probe`、`cloud_smoke_amp_walk`、
`cloud_api_audit`、`cloud_cpu_config_check`、`cloud_fix_deps*` 等。

**为什么可以归档**：它们的用途已被下面这些更省事的手段取代 ——
* 奖励权重核对 → 训练启动时自动打印的 `[AMP-WALK WEIGHTS]` 一行（`amp_walk_env_cfg.py`）
* 观测/维度契约 → 启动时的 `[AMP-WALK PASS]` + `obs_format to build ActorCritic` 打印
* 指标对比 → `cloud_compare_runs.py`
* 环境搭建 → `cloud_cpu_setup.sh`

## launchers_old/（11 个）

`run_amp_walk_v9.sh` ~ `run_amp_walk_v9j.sh`、`run_amp_walk_v10.sh` —— 历史版本的启动器。
当前在用的是 `run_amp_walk_v10b/v11/v12/v13.sh`（V10b 是 V11/V12 热启动的来源，保留）。

## local_oneoff/（7 个）

一次性本地自检/可视化：`test_amp_pipeline.py`（内含 `AMP_STATE_SIZE=670` 的旧断言，
V12 换成 2 帧 134 维后已失效）、`test_keyboard_teleop.py`、`test_mirror_ops.py`、
`test_scan_visual.py`、`smoke_test.py`、`check_env.py`、`check_default_pose.py`。
