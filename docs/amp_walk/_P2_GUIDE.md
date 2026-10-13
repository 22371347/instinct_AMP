# P2 作业指导书：cfg docstring 瘦身

> 本文件是**给执行者（人或 AI 子代理）的操作手册**。目标读者不需要先读别的文档。

---

## 1. 为什么做这件事

`instinct_xu/.../config/g1/amp_walk_*.py` 每个文件顶部有一段很长的模块 docstring，
混着三类东西：① 当前生效的参数 ② 当时的推理过程 ③ **已被证伪的结论**。

**危害**：后层 cfg 会继承前层并**覆盖参数**，所以读某个文件 docstring 里的"参数值"
会**读错实际生效值**。真实事故：有人读 v9h 的注释以为 `track_base_height` 的 std 是 0.3，
而 v9h 的源码其实已改成 0.2 并一路继承到今天的训练臂 ⇒ 配错配置、训练崩掉。

**目标**：改成"一眼看出：**本层改了什么、哪些今天还在生效、其余去哪查**"，且**不丢任何结论**。

---

## 2. 铁律（违反即失败）

1. **只改 docstring，一行代码都不许动。** 必须用脚本改（见 §4），**不要**用编辑器的文本替换。
2. **不许丢结论。** 原 docstring 里凡属"结论 / 教训 / 未解问题 / 参数改动"的内容，
   新稿必须保留（可压成一行/一个表格行）。若已归档到 `docs/amp_walk/EXPERIMENTS.md`，
   写"→ `EXPERIMENTS.md` §X"**并保留一句话摘要**。
3. **不许编造。** 不确定的标"未核实"。查"今天实际生效值"**只能用**
   `docs/amp_walk/EFFECTIVE_REWARDS.md`，**不要**读源码猜。
4. **不碰 `__init__.py` / `mdp/*.py`** 等代码文件（本任务只做 `config/g1/amp_walk_*.py` 的 docstring）。
5. 每个文件改完必须跑 §4 的验证，不通过就修到通过。

---

## 3. 新 docstring 模板（严格遵守）

```python
"""<臂名>：<一行目标>（<依据/日期，若原文有>）。

## ⚠️ 本层对**今天**仍有影响的部分

* <逐条：哪些设定今天还在生效；若完全没有，写"无 —— 已被后层完全覆盖，见表">
* ⭐ 用星号标出最重要的那一两条

## 本层改动一览

| # | 改动 | 现状（今天实际生效）|
|---|---|---|
| ① | <原文改动，压成一行> | <查 EFFECTIVE_REWARDS.md 后的现状；若被覆盖写"**已被 <某层> 覆盖** → <新值>"> |

## 推理与判读（已归档，勿在本文件重复）

<每条一句话摘要> → `docs/amp_walk/EXPERIMENTS.md` §<节号>
（若确实没归档，写"见该版 run 日志"）

⚠️ 其余逐字不动：<原文里"本次不动"的关键项，一句话>

⚠️ 本文件里的 `RewTerm(...)` **字面值不代表最终生效值**（后层会整体覆盖 `params`）。
   查实际生效值 → `docs/amp_walk/EFFECTIVE_REWARDS.md` → `Instinct-Amp-Walk-<版本>-G1-v0`
"""
```

参考两个已完成的样板：`amp_walk_v9h_env_cfg.py`、`amp_walk_v18_env_cfg.py`。

---

## 4. 怎么改（必用脚本）

中文技术文档里混用了 `U+2212`（−）、全角引号（“”）、全角括号等字符，
**基于文本匹配的替换会静默失败**。项目已提供按**行号**替换的脚本（用 `ast` 定位 docstring 边界）：

```bash
cd /home/leo/rsl_project/llp
# 1) 把新 docstring 写进 draft 文件（内容必须自带三引号 """..."""）
#    路径：docs/amp_walk/_p2_drafts/<cfg文件名>.txt
# 2) 预检查（只报告行范围，不写）
python3 instinct/tools/rewrite_docstring.py <cfg相对路径> <draft路径> --check
# 3) 实跑
python3 instinct/tools/rewrite_docstring.py <cfg相对路径> <draft路径>
# 4) 【必做】验证：docstring 之后的代码必须逐行完全一致
python3 - "<cfg相对路径>" <<'PY'
import ast, subprocess, sys
rel = sys.argv[1]
old = subprocess.run(['git','show',f'HEAD:{rel}'],capture_output=True,text=True).stdout
new = open(rel, encoding='utf-8').read()
oe = ast.parse(old).body[0].end_lineno
ne = ast.parse(new).body[0].end_lineno
oc, nc = old.splitlines()[oe:], new.splitlines()[ne:]
print("代码逐行完全一致:", oc == nc, " (共", len(nc), "行)")
PY
# 5) 语法检查：python3 -c "import ast;ast.parse(open(F,encoding='utf-8').read())"
```

⚠️ 注意：`git show HEAD:<path>` 对比的是**清理前快照 `7b3e39d`**（那次提交之后
P2 的改动都还没提交）。所以只要"代码逐行一致 = True"，就说明你只动了 docstring。

---

## 5. 参考资料

1. `docs/amp_walk/EFFECTIVE_REWARDS.md` —— **65 个臂实际生效的完整奖励表**（查"现状"只用这个）
2. `docs/amp_walk/EXPERIMENTS.md` —— 已归档的"已证伪结论 + 方法论教训"
   （§一 参数真相 / §A AMP判别器 / §B 奖励结构 / §C 步态坍塌 / §D 数据集框架 / §E 方法论 /
   附待办 / **勘误记录**）。**若你的内容已被它收录，写指针 + 一句话摘要。**
3. `docs/amp_walk/_p2_drafts/` —— 已完成的 draft（可参考风格）

---

## 6. 已知事实（避免重复踩坑）

* **我们用的机器人**：`G1_29DOF_TORSOBASE_POPSICLE_CFG`（`flat_env_cfg.py:25` 指定），
  `init_state = pos(0,0,0.82)` + `hip_pitch −0.312` + **`knee 0.669`(38.3°)** + `ankle_pitch −0.363`。
  （`0.42` 属于 `G1_29DOF_TORSOBASE_CFG`，**我们不用它**。）
* **`track_base_height` 的 target** = `default_root_state[2]` = **0.82 m = "屈膝 38°"下的 torso 高度**
  ⇒ 该项**鼓励保持屈膝**，直腿会超高被罚。
* **默认屈膝 38°** ⇒ `track_default_pose_on_stand` 的方向也是**鼓励屈膝**。
* **`amp_walk_env_cfg.py` 是 V5**，不是 V33。
* **T1l 的继承链跨越 41 个文件**（V2/V3/V12/V14/V16~V29/V33 全在链上）⇒ 它们的注释都可能误导人。

---

## 7. 报告格式（交给派活的人）

每个文件一行：

```
<文件名>: docstring <旧>→<新> 行 | 代码逐行一致=<True/False> | 语法=<OK/错误>
```

再加一段"**遇到的歧义 / 我判断不了的地方**"（宁可报告也不要猜）。
**不要**在报告里复述 docstring 全文。
