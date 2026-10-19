# URDF 替换交付包（G1 pelvis 换根模板）

本目录是**可直接发给组内同学**的 URDF 替换交付物。内容基于一次已通过验证的完整改造：

> **G1：popsicle（torso 为根）→ 官方 rev1.0（pelvis 为根）**
> 已通过 **locomotion（AMP 行走 P1）** 与 **越障（Perceptive Vault）** 两条技能的
> 训练 + 回放验证；`instinct_xu.zip` 是该改造**前**的基线。

---

## 目录内容

| 文件 | 用途 | 谁该看 |
|---|---|---|
| **`URDF_REPLACEMENT_GUIDE.md`** | **主指南**：3 条铁律、5 处影响面、9 步操作、5 级验证、症状排查表 | 所有要换 URDF 的人 |
| **`DISTILL_FRAMEWORKS.md`** | 蒸馏选型：为什么参考项目内 `instinct_rl` 那套（含与 PHP 的关系、能力矩阵、使用边界） | 要做 teacher→student 蒸馏的人 |
| **`REFERENCE_KEY_FILES.md`** | 关键代码文件清单 + 症状→文件索引 + 对照三步法 | 改完出问题时 |
| `reference/` | 15 个关键文件的**原始快照**（保留相对路径，可直接 diff） | 同上 |
| `urdf_swap_reference.tar.gz` | 上面 `reference/` 的打包版（142 KB，便于分发） | 同上 |
| `../../instinct/tools/verify_urdf_swap.py` | **验证脚本**（本地可跑，无需 IsaacLab） | 动手前后各跑一次 |

---

## 上手顺序（对方拿到本包后）

```bash
# 0) 先读主指南的 §0 三条铁律 + §1 影响面（5 分钟，能省几天）

# 1) 把自己的新资产放好（训练侧 URDF + 本地回放 MJCF，见指南 Step 1）

# 2) 跑验证脚本，先拿结构级结论（秒级）
python instinct/tools/verify_urdf_swap.py \
    --urdf <你的>.urdf --xml <你的>_play.xml [--npz <参考动作>.npz]
#    退出码 0 = 关节名齐备、接口可对齐 ⇒ 可以继续
#    退出码 1 = 有致命问题 ⇒ 走指南 §6 的 .npz 兜底路线

# 3) 按指南 Step 2~9 改配置；每改一步就用脚本回归一次

# 4) 验证：本地 .pt 回放（指南 §5 V2）→ 云端冒烟（V3）→ IO 交叉（V4）
```

---

## 这套改造验证到了什么程度

| 项目 | 状态 |
|---|---|
| 静态接口（关节名/顺序/根 body/质量/镜像） | ✅ 由 `verify_urdf_swap.py` 直接验证 |
| 本地回放 MJCF 与训练 URDF 是同一机器人 | ✅ `--verify-model` 逐 link/质量比对 |
| 换根后关节序**实测**并固定进契约 | ✅ 云端 smoke 读数 → `P1_JOINT_ORDER` |
| locomotion（AMP 行走）训练 | ✅ 真机训练通过（6144 timesteps/iter） |
| 越障（Vault）训练 | ✅ 训练 + 云端 IsaacLab 回放通过 |
| 观测构造与云端逐段一致 | ✅ `dump_io.py` + 本地重建对比 |

**已知限制**（务必转达）：
1. 越障类任务的**行为**只能在云端 IsaacLab 回放里看 —— 本地 MuJoCo 没有障碍地形。
2. IsaacLab 的 articulation 关节序**必须云端实测**，脚本无法替代（指南 §5）。
3. 本包快照是**参考实现**，不要整文件覆盖：你的机器人关节名/维度必然不同，
   要参照的是**结构与不变量**。
