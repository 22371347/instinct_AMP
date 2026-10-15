# V15 vs V16：AMP 判别器输入帧数（2 帧 → 10 帧）消融实验记录

> 记录日期：2026-10-01（训练已按要求停止，明日继续）
> 结论一句话：**10 帧判别器明显优于 2 帧** —— 判别器重新"有话说"（loss 0.39→0.13），
> 且策略行为显著更像人（倾角 30.3°→28.0°、步幅 `feet_air_time` 0.104→0.241 ×2.3、
> 抖动 `action_rate_l2` −0.568→−0.356 ×0.63、速度跟踪 1.302→1.447）。
> 但**退化没有消失**：V16 的 `tilt_deg_mean` 仍在缓慢上升（22.7°→28.0°），
> 且判别器**仍被权重衰减支配**（`weight_decay 105` vs `loss 0.134`，比例比 V15 更悬殊）。

---

## 1. 实验动机

V11（10 帧 AMP，会走路）与 V15（2 帧 AMP，训练到 10.5k 后姿态退化）在 **AMP 侧**的差异
只有两项：

| 项 | V11 | V15 | |
|---|---|---|---|
| λ（`style_lerp`） | 0.5 | 0.75 | 不同 |
| c（`discriminator_reward_coef`） | 0.5 | 1.0 | 不同 |
| **有效风格权重 (1−λ)c** | **0.25** | **0.25** | ✅ 相同 |
| 判别器输入帧数 | 10（670 维） | 2（134 维） | ❌ **结构性差异** |
| 判别器初始化 | 从零训 9000 轮 | 从零重学 | 需对齐 |

⇒ V16 = **V15 逐字沿用**，只把判别器输入回退到 **10 帧**（670 维）⇒ 严格单变量。

## 2. 单变量设计（两 run 完全对齐）

| | V15（2 帧） | V16（10 帧） |
|---|---|---|
| 任务 id | `Instinct-Amp-Walk-V15-G1-v0` | `Instinct-Amp-Walk-V16-G1-v0` |
| env cfg | `amp_walk_v15_env_cfg.py` | `amp_walk_v16_env_cfg.py`（V15 + `_patch_amp_history(self, 10)`） |
| agent cfg | `amp_walk_v15_ppo_cfg.py`（λ 0.75 / c 1.0） | `amp_walk_v16_ppo_cfg.py`（逐字同 V15） |
| 判别器输入 | `[(6,),(58,),(58,),(6,),(6,)]` = **134** | `[(30,),(290,),(290,),(30,),(30,)]` = **670** |
| 起点 ckpt | V11 `model_9000.pt` | V11 `model_9000.pt`（**同一个**） |
| 判别器初始化 | 从零（`drop_amp_discriminator`） | 从零（同样丢弃；670 维本可直载但不丢会引入第二变量） |
| 奖励/指令/数据集 | 21 项 / 三档指令 / 2-clip | 逐字相同（含 `joint_pos_limits_margin`、`track_default_pose_on_stand −0.3`） |

- **V11 起点 run**：`20260930_093037_G1AmpWalkV11MVPPlane_obs378_rew21_airtime8_ar0.06_bigstep_fromV10b7500_amp_walk_v11_4096_from20260929_232408`（`model_9000.pt`）
- **V15 run**：`20260930_222744_G1AmpWalkV15MVPPlane_obs378_rew21_amp2frame_style_x4_c1.0_fromV11_amp_walk_v15_4096_from20260930_093037`（停在 10500；ckpt `model_9500/10000/10500.pt` + `exported/`）
- **V16 run**：`20261001_000308_G1AmpWalkV16MVPPlane_obs378_rew21_amp10frame_style_c1.0_fromV11_amp_walk_v16_4096_from20260930_093037`
  （**停在 10240**；最后一颗 ckpt = `model_10000.pt`，另有 `model_9500.pt`、`exported/`、`params/`）
- run 目录根：`/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk/`
- 日志：`/root/train_amp_walk_v15_4096.log`、`/root/train_amp_walk_v16_4096.log`

## 3. 结果（250 轮窗口，同轮次对照）

### 3.1 早期（各 ~1k 轮，最能看出"退化速度"）

| 量 | V15 9000-9131 | V16 9000-9141 | V15 9131-9381 | V16 9141-9391 | V15 9381-9631 | V16 9391-9641 |
|---|---|---|---|---|---|---|
| `Step/amp_d` | −0.326 | **−0.596** | −0.381 | **−0.640** | −0.386 | **−0.640** |
| `Step/amp_style_reward` | 0.833 | 0.468 | 0.751 | 0.404 | 0.744 | 0.403 |
| `Loss/discriminator_loss` | 0.459 | **0.175** | 0.393 | **0.139** | 0.387 | **0.137** |
| `Step/tilt_deg_mean` | 23.5° | 22.7° | **31.9°** | **23.0°** | **35.4°** | **26.1°** |
| `Step/tilt_p95_deg` | 32.2° | 31.7° | 40.1° | 34.6° | 44.6° | 36.2° |
| `Step/tilt_gt30_frac` | 0.149 | 0.118 | 0.662 | 0.284 | 0.806 | 0.305 |
| `Episode_Reward/flat_orientation_l2` | −0.056 | −0.061 | −0.152 | −0.095 | −0.170 | −0.111 |
| `Episode_Reward/action_rate_l2` | −0.314 | −0.272 | −0.546 | **−0.341** | −0.524 | **−0.344** |
| `Episode_Reward/feet_air_time` | 0.076 | 0.102 | 0.106 | **0.198** | 0.100 | **0.234** |
| `Episode_Reward/track_lin_vel_xy_exp` | 0.895 | 0.995 | 1.319 | **1.426** | 1.223 | **1.429** |

**读法**：2 帧版在头 250 轮就把倾角从 23.5° 推到 31.9°；10 帧版 640 轮才到 26.1°，
且抖动只有 2/3、步幅是 2 倍、跟踪更好 ⇒ **10 帧的退化明显更慢、行为更像人**。

### 3.2 尾段（各 ~10.2–10.6k 轮）

| 量 | V15（2 帧）10311-10561 | V16（10 帧）9981-10231 | 变化 |
|---|---|---|---|
| `Step/amp_d` | −0.387 | **−0.646** | 判别器分辨力↑ |
| `Step/amp_style_reward` | 0.742 | 0.396 | 更低但更"真实" |
| `Loss/discriminator_loss` | 0.386 | **0.134** | **↓ 2.9×**（判别器强得多） |
| `Loss/discriminator_weight_decay` | 74.6 | 105.2 | ⚠️ 两版都被它支配（V16 更甚） |
| `Step/tilt_deg_mean` | 30.31° | **28.00°** | ↓ 2.3° |
| `Step/tilt_p95_deg` | 39.35° | **37.69°** | ↓ |
| `Step/tilt_gt30_frac` | 0.556 | **0.340** | ↓ 显著 |
| `Step/tilt_gt45_frac` | 0.0045 | **0.0010** | ↓ 4.5× |
| `Episode_Reward/flat_orientation_l2` | −0.1398 | −0.1269 | ↑（更直） |
| `Episode_Reward/feet_air_time` | 0.1036 | **0.2407** | **↑ 2.3×**（步幅正常化） |
| `Episode_Reward/action_rate_l2` | −0.5682 | **−0.3559** | ↑（抖动 ↓37%） |
| `Episode_Reward/track_lin_vel_xy_exp` | 1.3022 | **1.4466** | ↑ |
| `Episode_Reward/feet_slide` | −0.0409 | −0.0282 | ↑（滑动 ↓31%） |
| `Loss/mirror` | 0.060 | 0.056 | 相当 |
| 风格入账（0.25×style） | 0.186 | 0.099 | 更低（因 style 更低） |

### 3.3 V16 自身的时间趋势（9000 → 10240）

| 窗口 | 9231-9481 | 9481-9731 | 9731-9981 | 9981-10231 |
|---|---|---|---|---|
| `amp_d` | −0.6400 | −0.6402 | −0.6404 | −0.6456 |
| `amp_style_reward` | 0.4031 | 0.4026 | 0.4022 | 0.3960 |
| `tilt_deg_mean` | 24.56° | 26.93° | 28.09° | 28.00° |
| `feet_air_time` | 0.214 | 0.234 | 0.241 | 0.241 |
| `track_lin_vel_xy` | 1.427 | 1.450 | 1.463 | 1.447 |
| `action_rate_l2` | −0.343 | −0.351 | −0.354 | −0.356 |

⇒ 10000 轮（用户目视："能走出走路姿态，且比 9k 轮更好"）与量化一致：
**步幅/跟踪/滑动已稳定在好水平**，倾角在 28° 附近趋于平缓（没有再像 V15 那样冲到 35°+）。

## 4. 附：运行时对"2 帧实现"的验证（已定案 ✅）

用 `instinct/tools/cloud_probe_amp_append.py` + `cloud_check_amp_history.sh`
（云端跑，V15/V16 各一次）实测：

| 主张 | 证据 | 结论 |
|---|---|---|
| 每 term 每步恰好 push 1 次 | 类级 `CircularBuffer.append` 计数器：`env.reset()` 触发 10 次，之后**每步 +10**（5 term × 2 组）；`pointer`/`num_pushes` 各 +1/步 | ✅ |
| 帧 = 相邻控制步（0.02 s） | 不变量 `w_t(k) == w_{t-1}(k+1)` 在**非 reset env** 上残差 ≤1.3e-07（float32 舍入） | ✅ 真·状态转移 |
| 帧不是复制 | h=2 `\|末帧−首帧\| = 5e-3~4e-1`；h=10 `5e-2~1.0` 且由新到老单调 | ✅ |
| 两侧同构 | 同 history / 同 term 顺序 / 同缩放 | ✅ |
| 唯一瑕疵（非 bug） | reset 后前 h−1 帧是 `is_first_push` 的 prefill 常量（h=10 最多 0.2 s），两侧对称 | 已知 |

⚠️ **踩坑记录（探针判据，勿再犯）**：
1. 别在 **step 0** 判帧间差（那是 `push1` 的全槽位 prefill）；
2. 滑窗不变量是 `w_t(k) == w_{t-1}(k+1)`；写成 `w_t(0) == w_{t-1}(h-1)` **只在 h=2 成立**；
3. 别跨 env 取 `max`（会被"当步被 reset 的 env"污染，其缓冲被清零）；
4. 自写 Isaac Sim 脚本必须经 `cloud_run_bg.sh`（内含 `source /root/.llp_env.sh`，否则卡 EULA）；
5. `cfg.update_period = 0.02` = `env.step_dt` ⇒ 参考游标 1 帧/步（两侧同 Δt）。

## 5. 结论与下一步（待用户拍板）

**结论**
1. **帧数确实是 V11↔V15 的 AMP 侧关键变量**：同样起点、同样 λ/c 下，10 帧让判别器
   损失从 0.39 降到 0.13，并让策略在步幅/抖动/跟踪/倾角上全面占优。
2. **但 10 帧没有根治退化**：倾角仍从 22.7° 爬到 28°（V15 是 30.3°），
   `tilt_gt30_frac` 仍有 34% 的 env 超 30°。
3. ⚠️ **判别器仍被权重衰减支配**，且 10 帧下更悬殊：
   `Loss/discriminator_weight_decay = 105` vs `Loss/discriminator_loss = 0.134`
   （V15 是 74.6 vs 0.386）⇒ 判别器容量没被用起来。

**候选下一步（单变量，二选一）**

| 编号 | 变量 | 依据 |
|---|---|---|
| **V17a** | `discriminator_weight_decay_coef` 3e-4 → **0（或 1e-5）**，其余同 V16 | 权重衰减项已是损失项的 **780 倍** ⇒ 判别器停在"够用即止"的边界解 |
| **V17b** | `flat_orientation_l2` −1 → **−4~−6**（或在 `‖cmd_xy‖>0.1` 时门控的 L1 `torso_pitch_l1`），其余同 V16 | 倾角是唯一仍在恶化的量（22.7°→28°），而它的定价一直很低（28° 时每步只收 0.127，远小于 `track_lin_vel_xy` 1.45） |

**明日操作备忘**

已导出（GPU 空闲时做了，回放自检通过：input[1,378]/output[1,29]、闭环 33× 实时）：
`instinct/eval/v16_10000/{actor.onnx, policy_normalizer.npz}`

```bash
# 回放（本地，用户自己跑）
python instinct/tools/g1_mujoco_play.py --policy instinct/eval/v16_10000/actor.onnx --duration 0
```

```bash
# 续跑 V16（从 10000 那颗）
bash /root/cloud_run_bg.sh "bash /root/run_amp_walk_v16.sh 4096 1000000000 cuda:0 amp_walk_v16_4096 \
  20261001_000308_G1AmpWalkV16MVPPlane_obs378_rew21_amp10frame_style_c1.0_fromV11_amp_walk_v16_4096_from20260930_093037 \
  model_10000.pt" /root/train_amp_walk_v16_4096.log
# 画图 / 导出（Play 任务已注册）
bash instinct/tools/refresh_plots.sh loop v16
bash instinct/tools/pull_latest_onnx.sh v16          # → instinct/eval/v16_<step>/
# 回放（本地，用户自己跑）
python instinct/tools/g1_mujoco_play.py --policy instinct/eval/v16_<step>/actor.onnx --duration 0
# 窗口对比（含 V15）
python instinct/tools/instinct_ssh.py "/opt/conda/envs/isaaclab/bin/python /root/cloud_amp_compare.py v16 v15 --window 250 --last 4"
```
