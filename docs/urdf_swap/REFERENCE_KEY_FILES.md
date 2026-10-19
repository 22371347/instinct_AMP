# 关键代码文件参考清单（换 URDF 后出问题时的对照材料）

> 配套文档见本目录 `README.md`。快照位置：`docs/urdf_swap/reference/`（原始相对路径已保留，可直接与
> 自己的项目 `diff`）；打包版：`docs/urdf_swap/urdf_swap_reference.tar.gz`。
>
> **用法**：对方改完 URDF 后出现异常 ⇒ 先按 §2 的"症状 → 文件"表定位，再用 §1 的快照逐项比对。
> 比对时**优先看加 ⭐ 的文件**。

---

## 1. 快照清单（15 个文件，640 KB）

| # | 文件 | 行数 | 为什么重要 / 看什么 |
|---|---|---|---|
| ⭐1 | `instinct_xu/.../assets/unitree_g1.py` | 664 | **新旧资产的差异本体**。看 `G1_29DOF_REV10_PELVIS_CFG`（≈L576 起）与旧的 `G1_29DOF_TORSOBASE_POPSICLE_CFG` 的**逐字段差异**：`asset_path`、`init_state.pos.z`、默认站姿 `joint_pos`、`actuators` |
| ⭐2 | `instinct_xu/.../official_rev10/g1_29dof_rev_1_0.urdf` | 994 | **权威资产**。用它当"正确答案"比对：关节名、`<parent>/<child>`、`axis`、`origin`、`limit`、总质量（33.3411 kg）。meshes 未纳入快照（自行从官方 rev1.0 包获取） |
| ⭐3 | `instinct_xu/.../locomotion/config/g1/amp_walk_p1_contract.py` | 292 | **关节序与观测契约的唯一真相源** + 运行时校验器。看 `P1_JOINT_ORDER`（换根实测序）、`P1_OBSERVATION_SIZE == 1500`、断言清单 |
| ⭐4 | `instinct/tools/joint_order.py` | 302 | **四套顺序的定义**与重排工具。看 `MUJOCO_ORDER` / `ISAACLAB_ORDER` / `NPZ_ORDER` / `PELVIS_ORDER`、`ORDER_CANDIDATES`、`mujoco_state_order_from_model()`（状态序 ≠ ctrl 序） |
| 5 | `instinct/tools/verify_urdf_swap.py` | 494 | **本次交付的验证脚本**（可独立运行）。出问题时先跑它，看它报哪一节 ❌ |
| 6 | `instinct_xu/.../locomotion/config/g1/amp_walk_p1_env_cfg.py` | 1332 | 展平后的 env cfg。看 `scene.robot.spawn.asset_path`、`init_state`、`height_scanner.prim_path`、`motion_reference.prim_path`、`link_of_interests`、`track_pelvis_height` 靶值 0.7585 |
| 7 | `instinct_xu/.../config/g1/agents/amp_walk_p1_ppo_cfg.py` | 224 | 算法侧：`mirror_joint_names` 必须与 `P1_JOINT_ORDER` 一致（镜像损失依赖） |
| 8 | `instinct_xu/.../locomotion/config/g1/__init__.py` | 817 | 任务注册（≈L804–814）：新 id、`entry_point=G1AmpWalkP1TeacherEnv`、cfg 路径。**换资产时要在这里加新 id，别覆盖旧的** |
| 9 | `instinct_xu/.../shadowing/beyondmimic/config/g1/beyondmimic_vault_cfg.py` | 368 | **越障类必看**：`VAULT_FK_URDF`（FK 必须用 pelvis 根的 URDF，≈L28–45）与数据集契约注释（base 语义、障碍箱同世界系） |
| 10 | `instinct_xu/.../utils/mirror.py` | 373 | 矢状面镜像规则（pitch 保号 / roll·yaw 反号）、扫描布局表 `SCAN_LAYOUTS`、按项名推导布局的写法 |
| 11 | `instinct_xu/scripts/instinct_rl/dump_io.py` | 185 | **云端导出 IO 快照**：打印权威 `[joint_names(articulation order)]` 与 obs layout，用于交叉验证 |
| 12 | `instinct/tools/cloud_p1_smoke.py` | 110 | **云端配置层冒烟**：资产路径 / `init_z` / 挂点 / 观测维 / 关节数 全项自检（泛化到新机器人最省事的起点） |
| 13 | `instinct/tools/build_g1_mjcf.py` | 407 | 本地可回放 MJCF 的生成器：执行器注入公式、逐关节阻尼覆盖、`--init-root-height` / `--strip-*` |
| 14 | `instinct/tools/g1_mujoco_play.py` | 2309 | 本地回放主程序：`ASSET_PROFILES`、`detect_asset_profile`（auto 推断）、`ASSET_FINGERPRINTS`、`verify_model_vs_urdf`、`--asset-check` 自检 |
| 15 | `instinct/assets/g1_29dof_rev10_play.xml` | 324 | 生成好的回放 MJCF：可直接比对 `nu=29`、执行器 kp/kv、`keyframe` 根高 0.7585、ctrl 顺序 |

---

## 2. 症状 → 先看哪个文件

| 症状 | 首选文件 | 次选 |
|---|---|---|
| `AssertionError`/`AssertionError` 关观测维（如期望 1500 收到 626） | ⭐3 契约 | 7 ppo_cfg |
| 训练能起但学不动、奖励不涨 | ⭐3 → ⭐4 | 11 dump_io（云端实测序） |
| 回放抽搐 / 摔倒 / 关节乱动 | ⭐4 | 14 g1_mujoco_play（换 `--asset`）、13 build_g1_mjcf |
| 不知道本地跑的到底是不是训练那台机器 | 14（`--verify-model`） | 15 play.xml |
| 参考动作整体穿地约 4 cm | 9 vault cfg | ⭐1 资产 cfg（根 link） |
| 身高奖励恒负 / 高度总差一截 | 6 env cfg（`track_*_height`） | ⭐1（`init_state.pos.z`） |
| 高程图/运动参考报 prim path 找不到 | 6 env cfg（挂点） | ⭐3 契约 |
| 换根后旧 ckpt 报 `size mismatch` | ——（**预期行为**，需重训） | ⭐3 |
| 镜像损失报维度/关节名错 | 7 ppo_cfg | 10 mirror.py |
| 蒸馏配置报观测组找不到 | 见 `DISTILL_FRAMEWORKS.md` §4.3 | ⭐3 契约 |
| 云端资产加载失败 / `UrdfFileCfg` 报错 | ⭐2 URDF | ⭐1 资产 cfg |

---

## 3. 与新资产对照的三步法

```bash
# 第 1 步：先跑交付脚本，拿到"结构级"结论（秒级、无需 IsaacLab）
python instinct/tools/verify_urdf_swap.py --urdf <你的>.urdf --xml <你的>_play.xml

# 第 2 步：结构级 OK 后再逐字段对照（把 <你的> 换成自己的路径）
diff -u docs/urdf_swap/reference/instinct_xu/.../assets/unitree_g1.py \
        <你的>/instinctlab/assets/<robot>.py | head -80

# 第 3 步：把 <你的> 关键常量与参考契约并排看
grep -n 'JOINT_ORDER\|OBSERVATION_SIZE\|init_state\|prim_path' \
     docs/urdf_swap/reference/instinct_xu/.../amp_walk_p1_contract.py \
     <你的>/.../<your>_contract.py
```

> ⚠️ 快照只是**参考实现**，不是让你整文件覆盖 —— 你的机器人关节名/维度必然不同。
> 参照的是**结构与不变量**（顺序从哪来、契约断言写哪些、资产字段哪些必须实测）。
