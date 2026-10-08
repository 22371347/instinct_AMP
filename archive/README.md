# 归档目录（仓库根）

放历史记录与旧产物，**不参与当前流水线**，但保留可追溯性。

## docs/（5 个）

* `AMP_MJLIB_CHECK.md` / `AMP_MJLIB_CHECK_DATA.md` / `AMP_MJLIB_CHECK_RESULT.md`
  —— 与 AMP_mjlab 逐项对照的那次审计（结论已并入 `PLAN_amp_walk.md`）。
* `tmp.md` —— 外部 AI 对 AMP_mjlab 的分析原文（我们逐条批注过的底稿）。
* `MUJOCO_LOG.TXT` —— 早期 MuJoCo 回放的日志残留。

**仍留在仓库根**（活跃文档）：`PLAN_amp_walk.md`、`PLAN_expert_distill.md`、
`INTERFACE_CONTRACT.md`、`STUDENT_OBS_SPEC.md`、`CLOUD_ENV_SETUP.md`、
`MIRROR_LOSS_DESIGN.md`、`REPRODUCTION_ANALYSIS.md`、`user.md`。

## eval_legacy/（5 个）

* `actor_model_5000.onnx` + `policy_normalizer.npz`（裸名字、**不带版本号**）——
  正是 `pull_latest_onnx.sh` 头部警告的那类隐患：两个模型共用同一个归一化器文件名，
  后拉的会静默覆盖先拉的 ⇒「旧 onnx + 新 normalizer」错配。
  现在 `instinct/eval/<tag>/` 每个模型独立目录，靠目录天然配对。
* `model_5000_*.mp4`、`mujoco_model5000_cmd_vx0.5.mp4` —— 早期录像（可随时重新生成，
  `.gitignore` 里本来就列为「体积大且可重新生成」）。
