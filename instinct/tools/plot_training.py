#!/usr/bin/env python3
"""G1 AMP 行走训练曲线（在**云端**生成 PNG，再由本地脚本拉回来看）。

设计蓝本：`AMP_mjlab/tools/plot_rewards.py`，保留它几个真正好用的做法：
  1. 参数化实验目录；**只认时间戳 run 目录**（`^\\d{8}_`），过滤 `_bak`/`params`/`git`
     之类非训练目录 —— AMP_mjlab 曾因为 `_` 的 ASCII 比数字大、备份目录排到末尾
     被当成 LATEST，画出一张完全不符实际的曲线。
  2. 两种模式：`resume` = 只画最新 run（快，监控用）；`full` = 合并历史 run，
     重叠 step 取较新者（跨 run 连续，避免续训造成断层）。
  3. **逐子图独立容错**：缺某个 tag 就显示 "no data"，绝不让整张图失败。

相对蓝本做的三处改进：
  A. **EventAccumulator 按 run 缓存**（蓝本每次 `_series()` 都新建一个，90+ 个 tag
     就要把整个 events 文件重读 90 遍）。
  B. **先把 events 文件拷到 /tmp 再读**：训练正在写这个文件，
     直接读偶发「最后一条记录被截断」的解析报错。
  C. 新增第 04 张「步宽/交叉专项」图与文末文本摘要 —— 这两项是本次迁移最关键的
     设计风险点（见 PLAN_amp_walk.md §3.7）。

⚠️ **本仓库的奖励混合公式与 AMP_mjlab 不同**（见 PLAN §4.2）：
      本仓库（`WasabiPPOAmpMix`，lerp 模式）:  R = λ·r_task + (1-λ)·c·r_amp
      AMP_mjlab                            :  R = (1-lerp)·disc + lerp·task
   形式同构，但**风格项额外乘一个 c**（`discriminator_reward_coef=0.1`），
   所以实际风格权重是 (1-λ)·c = 0.025，**不是** (1-λ) = 0.25。
   第 05 图按本仓库的口径拆解，并把 λ 与 c 都标在标题上，防止误读。

环境变量（全都有默认值，按需覆盖）：
  LLP_LOG_ROOT   日志根目录，默认 /root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk
  LLP_RUN        ⭐ 选择要画的 run：**目录名子串**（`v9c` / `amp_walk_v9b_mvp_4096`）
                 或完整目录名；留空 = 最新。命中多个取最新一个。
                 并行跑多份训练（v9b / v9c …）时必须用它，否则画的是“最后启动的那份”
  LLP_OUT        resume 模式输出目录
  LLP_OUT_FULL   full 模式输出目录
  LLP_LATEST     显式指定 run 目录（跳过自动取最新）
  LLP_OLD_RUN    full 模式合并起点（run 目录名子串，可选）
  LLP_SKIP_FULL  =1 时跳过 full 模式（跨 run 合并慢，监控循环建议开）
  LLP_LERP       λ，默认 0.75
  LLP_C          c，默认 0.1
  LLP_EP_LEN_S   episode_length_s，默认 20.0（flat_env_cfg.py 里是 20.0）
  LLP_NUM_REWARDS 奖励组数，默认 1（本任务没走 MultiRewardCfg）
  LLP_SMOOTH     平滑窗，默认 100
  LLP_DPI        出图 DPI，默认 130

用法：
  python plot_training.py                     # 出全部图（resume + full，最新 run）
  python plot_training.py --run v9c           # ⭐ 只画 v9c 那份，输出进 <OUT>/v9c/
  python plot_training.py --run v9b --all-matching   # 命中多个时全部合并（full 模式）
  python plot_training.py --list-runs         # 列出可选 run（目录名 / 修改时间 / 最新 ckpt）
  python plot_training.py --list-tags         # 只列出「选中的 run 里真实存在的 tag」再退出
  LLP_SKIP_FULL=1 python plot_training.py --run v9c

⚠️ 给了 `--run` 时输出目录会自动加子目录（`<OUT>/<run>` / `<OUT_FULL>/<run>_full`），
   这样多份训练互不覆盖；不想用子目录就显式传 `--outdir` / `--outdir-full`。
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

LOG_ROOT = os.environ.get(
    "LLP_LOG_ROOT", "/root/gpufree-data/instinct_xu/logs/instinct_rl/g1_amp_walk"
)
OUT_RESUME = os.environ.get(
    "LLP_OUT", "/root/gpufree-data/instinct_xu/logs/instinct_rl/plots"
)
OUT_FULL = os.environ.get(
    "LLP_OUT_FULL", "/root/gpufree-data/instinct_xu/logs/instinct_rl/plots_full"
)
LATEST_OVERRIDE = os.environ.get("LLP_LATEST", "")
OLD_RUN = os.environ.get("LLP_OLD_RUN", "")
SKIP_FULL = os.environ.get("LLP_SKIP_FULL", "") != ""
LERP = float(os.environ.get("LLP_LERP", "0.75"))
C_COEF = float(os.environ.get("LLP_C", "0.1"))
EP_LEN_S = float(os.environ.get("LLP_EP_LEN_S", "20.0"))
#: 策略频率。⚠️ 两个口径换算倍数不同：
#:   `Episode_Reward/*` = `episode_sum / max_episode_length_s` ⇒ **每秒速率** ⇒ ×EP_LEN_S
#:   `Step/*`            = 每步瞬时值的均值             ⇒ **每步值**   ⇒ ×EP_LEN_S×HZ
#: 搞混会整整差 50 倍。
HZ = float(os.environ.get("LLP_POLICY_HZ", "50"))
EP_LEN_STEPS = EP_LEN_S * HZ
#: 原始风格奖励（= 判别器的 quad 奖励 r_amp ∈ [0,1]）。
#: `algorithms.py` 里 `aux["amp_style_reward"] = r_amp`，**只写日志、不进奖励**
#: （真的进奖励的是 lerp 混合后的 `discriminator_reward`）。
STYLE_TAG_NAME = os.environ.get("LLP_STYLE_TAG", "Step/amp_style_reward")
DISCRIMINATOR_TAG_NAME = os.environ.get("LLP_DISC_TAG", "Step/discriminator_reward")
NUM_REWARDS = int(os.environ.get("LLP_NUM_REWARDS", "1"))
SMOOTH = int(os.environ.get("LLP_SMOOTH", "100"))
DPI = int(os.environ.get("LLP_DPI", "130"))

# `Episode_Reward/*` —— ⚠️ **这只是"展示顺序提示"，不是白名单**。
# 真正画哪些项由 `Trainer.reward_tags()` 从 TB 里**动态发现**（`Episode_Reward/*` 全拿）。
# ⭐ 历史坑（2026-10-09）：T2（全权照抄 AMP_mjlab）把两个跟踪项改成了 mjlab 的名字
#   （`track_lin_vel_xy_exp` → `track_anchor_linear_velocity`、
#    `track_ang_vel_z_exp` → `track_anchor_angular_velocity`），
#   而这里是硬编码的旧名字 ⇒ **这两条曲线整条从图上消失**，看起来像"奖励没设计速度跟踪"，
#   实际是**画图遗漏**（训练日志里 `Episode_Reward/track_anchor_linear_velocity` 一直都在）。
REWARD_TAGS = [
    # ---- T2（mjlab 命名）----
    "Episode_Reward/track_anchor_linear_velocity",
    "Episode_Reward/track_anchor_angular_velocity",
    # ---- T1 及更早 ----
    "Episode_Reward/track_lin_vel_xy_exp",
    "Episode_Reward/track_ang_vel_z_exp",
    "Episode_Reward/track_base_height",
    "Episode_Reward/body_ang_vel_xy",
    "Episode_Reward/track_default_pose_on_stand",
    "Episode_Reward/feet_lateral_width",
    "Episode_Reward/feet_crossing",
    "Episode_Reward/feet_slide",
    "Episode_Reward/feet_air_time",
    "Episode_Reward/flat_orientation_l2",
    "Episode_Reward/stand_still",
    "Episode_Reward/dof_pos_limits",
    "Episode_Reward/joint_deviation_hip",
    "Episode_Reward/joint_deviation_arms",
    "Episode_Reward/joint_deviation_torso",
    "Episode_Reward/joint_deviation_knee",
    "Episode_Reward/lin_vel_z_l2",
    "Episode_Reward/action_rate_l2",
    "Episode_Reward/dof_acc_l2",
    "Episode_Reward/dof_torques_l2",
    "Episode_Reward/termination_penalty",
]

# 步宽标定值（来自 instinct/tools/calibrate_foot_sep.py 的实测，见 PLAN §3.7）
SEP_FLOOR = 0.145
SEP_DATASET_MEDIAN = 0.2140


# ---------------------------------------------------------------------------
# run 发现
# ---------------------------------------------------------------------------


def discover_runs() -> list[str]:
    """只返回**训练 run** 目录（时间戳开头），已排序。"""
    all_dirs = sorted(glob.glob(os.path.join(LOG_ROOT, "*", "")))
    ts_re = re.compile(r"^\d{8}_")
    runs = [d for d in all_dirs if ts_re.match(os.path.basename(os.path.normpath(d)))]
    if not runs:
        raise SystemExit(
            f"未找到训练目录: {LOG_ROOT}\n"
            f"  （目录下现有: {[os.path.basename(os.path.normpath(d)) for d in all_dirs]}）"
        )
    return runs


def slug(token: str) -> str:
    """把 `--run` 的取值变成安全的目录名（`v9c` → `v9c`；带空格/斜杠的会被替换）。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", (token or "").strip()) or "run"


def match_runs(runs: list[str], token: str) -> list[str]:
    """按 **目录名子串**（大小写不敏感）匹配 run；也接受完整目录名。

    用途：并行跑多份训练时（v9b / v9c …）用 `--run v9c` 明确“画哪一份”，
    而不是永远画“最新那一个”。返回按时间顺序（旧→新）。
    """
    if not token:
        return list(runs)
    t = token.strip().lower()
    exact = [r for r in runs if os.path.basename(os.path.normpath(r)).lower() == t]
    if exact:
        return exact
    return [r for r in runs if t in os.path.basename(os.path.normpath(r)).lower()]


def run_info(run_dir: str) -> str:
    """一行描述一个 run：目录名 / 修改时间 / 最新 ckpt / events 文件数。"""
    name = os.path.basename(os.path.normpath(run_dir))
    try:
        mt = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(run_dir)))
    except OSError:
        mt = "?"
    cks = glob.glob(os.path.join(run_dir, "model_*.pt"))

    def _step(p: str) -> int:
        m = re.search(r"model_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    newest_ck = os.path.basename(max(cks, key=_step)) if cks else "(无 ckpt)"
    n_ev = len(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    return f"{name:<72} {mt}  {newest_ck:<16} events={n_ev}"


# ---------------------------------------------------------------------------
# TensorBoard 读取（带缓存 + 拷贝避免读写竞争）
# ---------------------------------------------------------------------------

_ACC_CACHE: dict[str, EventAccumulator] = {}
_TMP_DIR = tempfile.mkdtemp(prefix="llp_tb_")


def _accumulator(run_dir: str) -> EventAccumulator | None:
    """读一个 run 的 EventAccumulator（缓存）。

    ⚠️ 训练正在往这个 events 文件里写，直接读偶尔会碰到「最后一条记录只写了一半」
    导致解析报错。所以先拷到 /tmp 再读 —— 快照一致，且不干扰训练。
    """
    key = os.path.normpath(run_dir)
    if key in _ACC_CACHE:
        return _ACC_CACHE[key]
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        _ACC_CACHE[key] = None  # type: ignore[assignment]
        return None
    src = files[-1]  # 取最新的一个 events 文件
    dst = os.path.join(_TMP_DIR, os.path.basename(key) + "_" + os.path.basename(src))
    try:
        shutil.copy2(src, dst)
    except OSError:
        dst = src  # 拷不动就直接读原文件（降级，不是错误）
    acc = EventAccumulator(dst, size_guidance={"scalars": 0})
    try:
        acc.Reload()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Reload 失败 {key}: {type(exc).__name__}: {exc}")
        _ACC_CACHE[key] = None  # type: ignore[assignment]
        return None
    _ACC_CACHE[key] = acc
    return acc


def available_tags(run_dir: str) -> list[str]:
    acc = _accumulator(run_dir)
    return [] if acc is None else list(acc.Tags().get("scalars", []))


def _drop_warmup(steps, vals, drop_iters: int):
    """隐去开头的**初始化段**（默认前 100 轮，即 10 个 TB 采样点）。

    为什么要裁：训练启动的头 50~100 轮是"初始化/自检"阶段 ——
      * 从零训练：机器人还不会站，奖励在 `−40 ↔ +15` 量级剧烈震荡；
      * 热启动（`--resume`）：判别器被 `drop_amp_discriminator` 重置、参考动作游标
        归零 ⇒ 头几轮 `amp_d`/`amp_style_reward`/回合长度全部跳变。
    这段的振幅比后续"真正的趋势变化"大得多 ⇒ **y 轴被迫拉得很宽，后期细节看不见**。
    用户反馈：用对数轴不符合直觉 ⇒ 正确做法是**裁掉初始化段**。

    ⚠️ 必须裁在**平滑之前**：若只 `set_xlim`，移动平均仍会被初始化段的尖峰污染，
       开头几个点会假性外飞。
    ⚠️ 裁掉的部分会写进 x 轴标签与 suptitle（**不能静默裁**）—— 否则就会重演历史上
       `smooth_series` 那个 bug 的观感："曲线从 500 轮才有数据"。
    ⚠️ 数据本来就短（裁完 < 5 点）则不裁，避免整张图变成空图。
    """
    if drop_iters <= 0 or len(steps) == 0:
        return steps, vals, 0
    keep = steps >= float(drop_iters)
    if int(keep.sum()) < 5:
        return steps, vals, 0
    return steps[keep], vals[keep], int(len(steps) - keep.sum())


def _series(run_dir: str, tag: str, clip: float | None = None):
    """读单 run 一条标量曲线。缺 tag / 空数据都返回空数组，不抛异常。"""
    acc = _accumulator(run_dir)
    if acc is None:
        return np.array([]), np.array([])
    try:
        evs = acc.Scalars(tag)
    except KeyError:
        return np.array([]), np.array([])
    steps = np.array([e.step for e in evs], dtype=float)
    vals = np.array([e.value for e in evs], dtype=float)
    finite = np.isfinite(vals)
    steps, vals = steps[finite], vals[finite]
    if clip is not None and len(vals) > 0:
        keep = np.abs(vals) < clip
        steps, vals = steps[keep], vals[keep]
    return steps, vals


def _merge(run_dirs: list[str], tag: str, clip: float | None = None):
    """合并多个 run 的同一条曲线；step 重叠时**取较新 run**（与蓝本一致）。"""
    parts_s, parts_v, parts_i = [], [], []
    for i, d in enumerate(run_dirs):
        s, v = _series(d, tag, clip)
        if len(s) == 0:
            continue
        parts_s.append(s)
        parts_v.append(v)
        parts_i.append(np.full(len(s), i))
    if not parts_s:
        return np.array([]), np.array([])
    steps = np.concatenate(parts_s)
    vals = np.concatenate(parts_v)
    order = np.argsort(steps, kind="stable")
    steps, vals = steps[order], vals[order]
    # 同一 step 出现多次 → 只保留最后一次出现（即较新 run 覆盖旧的）
    keep = np.ones(len(steps), dtype=bool)
    seen = set()
    for i in range(len(steps) - 1, -1, -1):
        if steps[i] in seen:
            keep[i] = False
        else:
            seen.add(steps[i])
    steps, vals = steps[keep], vals[keep]
    order = np.argsort(steps)
    return steps[order], vals[order]


def smooth_series(steps, vals, window: int):
    """尾部（trailing）移动平均，**保留全部 x**（曲线从 step 0 开始）。

    ⚠️ 旧实现用 `np.convolve(..., mode="valid")` + `steps[w-1:]` ⇒ **丢掉前 w−1 个采样点**，
    而 TB 每 10 轮才记一次 ⇒ 图上曲线的 x 起点变成 `10×(w−1)`（运行早期 w≈50 时正好 ≈500），
    看起来像"奖励项从 500 轮才开始有数据"（其实 TB 里所有标量都从 step 0 开始）。
    现在：前 w−1 点用"到目前为止可用样本"的均值（等价于 pandas `rolling(w, min_periods=1)`），
    于是 x 从 0 起、整体趋势与旧图一致，只是开头不再被截掉。
    """
    if window <= 1 or len(vals) < 5:
        return steps, vals
    w = max(min(window, len(vals) // 5), 1)
    if w <= 1:
        return steps, vals
    n = len(vals)
    cum = np.convolve(vals, np.ones(w) / w, mode="full")[:n]   # 尾部和的 1/w（窗口不足时偏小）
    cnt = np.minimum(np.arange(1, n + 1), w)                    # 每个点实际参与平均的样本数
    return steps, cum * w / cnt                                 # 归一化成"可用样本的均值"


def _align_to(steps_target, series_list):
    """把多条 (steps, vals) 用线性插值对齐到 steps_target 后求和。"""
    total = np.zeros(len(steps_target), dtype=float)
    for s, v in series_list:
        if len(v) == 0:
            continue
        total = total + np.interp(steps_target, s, v, left=v[0], right=v[-1])
    return total


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------


class Plotter:
    def __init__(self, mode: str, outdir: str, runs: list[str], latest: str,
                 drop_iters: int = 100):
        self.mode = mode
        self.outdir = outdir
        self.runs = runs          # full 模式要合并的 run 列表
        self.latest = latest
        #: 隐去的初始化轮数（0 = 不隐）。见 `_drop_warmup` 的 docstring。
        self.drop_iters = int(drop_iters)
        self.tags_cache: list[str] | None = None
        os.makedirs(outdir, exist_ok=True)

    # --- 取数 ---
    def drop_note(self) -> str:
        """图上统一的"已隐去前 N 轮"标注（裁了就必须写出来，否则会被误读）。"""
        return "" if self.drop_iters <= 0 else f"  |  drop-first-{self.drop_iters}-iters"

    def get(self, tag: str, smooth: int | None = None, clip: float | None = None):
        # ⭐ clip 默认 **不裁剪**。蓝本对每张图都传 clip=1000，结果把
        # `Perf/total_fps`（真实值 ~24000）整个裁成空 → 图上显示 no data。
        # 尖峰问题改用「_panel 里按分位数设 y 轴范围」解决（保留全部数据不丢）。
        if self.mode == "resume":
            s, v = _series(self.latest, tag, clip)
        else:
            s, v = _merge(self.runs, tag, clip)
        # ⭐ 先裁初始化段，再平滑（顺序不能反，见 `_drop_warmup`）
        s, v, _ = _drop_warmup(s, v, self.drop_iters)
        return smooth_series(s, v, SMOOTH if smooth is None else smooth)

    def tags(self) -> list[str]:
        if self.tags_cache is None:
            t = set()
            for d in ([self.latest] if self.mode == "resume" else self.runs):
                t.update(available_tags(d))
            self.tags_cache = sorted(t)
        return self.tags_cache

    def reward_tags(self) -> list[str]:
        """本 run（们）**实际存在**的 `Episode_Reward/*`（`REWARD_TAGS` 仅用于排顺序）。

        ⚠️ 以前这里是**硬编码白名单** ⇒ 一旦某个臂改了奖励项名字，那条曲线会**静默消失**，
        很容易被误读成"奖励设计漏了某项"（T2 的 `track_anchor_linear_velocity` 就是这样
        从图上消失的，而它在训练日志里一直存在）。
        现在：按 TB 里真实存在的 tag 发现，未登记的项按字母序排在后面。
        """
        have: set[str] = set(self.tags())
        for d in self.runs:
            have.update(available_tags(d))
        known = [t for t in REWARD_TAGS if t in have]
        extra = sorted(
            t for t in have if t.startswith("Episode_Reward/") and t not in set(known)
        )
        return known + extra

    def find(self, substr: str, *, exclude: str | None = None) -> str | None:
        """按子串找 tag（用于不确定精确名字的量）。"""
        for t in self.tags():
            if substr in t and (exclude is None or exclude not in t):
                return t
        return None

    def latest_value(self, tag: str):
        s, v = self.get(tag, smooth=1)
        return (float(s[-1]), float(v[-1])) if len(v) else (None, None)

    # --- 出图 ---
    def save(self, fig, name: str):
        path = os.path.join(self.outdir, name)
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO][{self.mode}] 已保存: {path}")

    def _panel(self, ax, tag, title, *, smooth=None, hlines=(), color=None, clip=None):
        s, v = self.get(tag, smooth=smooth, clip=clip) if tag else (np.array([]), np.array([]))
        if len(v) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
        else:
            ax.plot(s, v, lw=1.2, color=color)
            # 尖峰不丢数据，但把 y 轴限在 0.5~99.5 分位内，否则一个异常值会把整图拉爆
            # （比蓝本的 clip=1000 好：既能保护视图，又不会把 FPS 这类大值裁成空）
            if len(v) > 10:
                lo, hi = np.nanpercentile(v, [0.5, 99.5])
                pad = 0.08 * (hi - lo) + 1e-9
                ax.set_ylim(lo - pad, hi + pad)
        for y, lab, sty in hlines:
            ax.axhline(y, ls=sty, lw=0.9, color="0.45")
            ax.text(0.99, y, " " + lab, transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=7, color="0.35")
        ax.set_title(title)
        # ⚠️ 裁了初始化段就必须**写在轴上**，否则会被误读成"训练从第 N 轮才开始"。
        # ⚠️ 必须**纯 ASCII**：云端 matplotlib 只有 DejaVu Sans（无 CJK 字形）
        #    ⇒ 中文会渲染成方块/缺字（实测）。
        ax.set_xlabel("iteration" if self.drop_iters <= 0
                      else f"iteration  (first {self.drop_iters} iters hidden)")
        ax.grid(alpha=0.3)

    def fig_overview(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f"AMP Walk Training Overview [{'latest' if self.mode == 'resume' else 'full'}]"
                     f"  {os.path.basename(os.path.normpath(self.latest))}{self.drop_note()}", fontsize=13)
        self._panel(axes[0, 0], f"Train/mean_reward_{0}", "Mean Reward (episode total)")
        self._panel(axes[0, 1], "Train/mean_episode_length", "Episode Length (steps)")
        self._panel(axes[1, 0], "Loss/amp_total_loss", "AMP Total Loss")
        self._panel(axes[1, 1], "Policy/mean_noise_std", "Action Noise Std")
        self.save(fig, "01_overview.png")

    def fig_reward_terms(self):
        # ⭐ 动态发现（不再是硬编码白名单）：T2 改名后的 `track_anchor_*` 以前会被静默丢掉
        tags = [t for t in self.reward_tags()
                if self.latest_value(t)[1] is not None
                or (self.mode == "full" and any(len(_series(d, t)[1]) for d in self.runs))]
        n = max(len(tags), 1)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 2.8 * rows), squeeze=False)
        fig.suptitle(f"Per-Reward-Term Convergence (smooth={SMOOTH}){self.drop_note()}", fontsize=13)
        for i, tag in enumerate(tags):
            self._panel(axes.flat[i], tag, tag.replace("Episode_Reward/", ""))
            axes.flat[i].set_title(tag.replace("Episode_Reward/", ""), fontsize=9)
        for j in range(len(tags), rows * cols):
            axes.flat[j].axis("off")
        if not tags:
            axes[0, 0].text(0.5, 0.5, "no reward data", ha="center", va="center",
                            transform=axes[0, 0].transAxes)
        self.save(fig, "02_reward_terms.png")

    def fig_amp_diagnostics(self):
        """判别器诊断。

        ⭐ 2026-10-10 新增 `amp_d` / `amp_d_ref` 以及两者的**差距图**（第 9 格）。
        这两个量是「AMP 风格到底有没有在干活」的**唯一直接证据**：
          - `amp_d`     = D(策略状态)  → +1 = 完全像参考
          - `amp_d_ref` = D(参考状态)  → 判别器自己的「满分线」，恒 ≈ +1
          - 差距 `amp_d_ref − amp_d` = **判别器还剩多少可分辨度**。
            差距≈0 ⇒ D 贴死/摆烂（T1l：`amp_d_ref +0.93`）；
            差距大 ⇒ D 在工作（T2 末期：`amp_d_ref +0.52` vs `amp_d −0.83`，gap≈1.35）。
        ⚠️ 本仓库的 `amp_d` 是 **wasserstein 原始打分**（不是概率），量纲 ±~1，可为负。
        """
        fig, axes = plt.subplots(3, 3, figsize=(20, 13))
        fig.suptitle("AMP Discriminator Diagnostics", fontsize=13)
        for ax, (tag, title) in zip(axes.flat, [
            ("Loss/discriminator_loss", "discriminator_loss"),
            ("Loss/discriminator_gradient_penalty", "discriminator_gradient_penalty"),
            ("Loss/discriminator_weight_decay", "discriminator_weight_decay"),
            ("Loss/logit_weight_decay", "logit_weight_decay"),
            ("Step/discriminator_reward", "discriminator_reward (aux, this repo)"),
            ("Step/amp_style_reward", "amp_style_reward = r_amp (log only, quad in [0,1])"),
            ("Step/amp_d", "amp_d = D(policy state)   [+1 = reference-like]"),
            ("Step/amp_d_ref", "amp_d_ref = D(reference state)   [should stay ~+1]"),
        ]):
            self._panel(ax, tag, title, smooth=50)
        self._panel_amp_d_gap(axes[2, 2])
        self.save(fig, "03_amp_diagnostics.png")

    def _panel_amp_d_gap(self, ax):
        """`amp_d` 与 `amp_d_ref` **画在同一张图**上，并填色标出差距。

        分开看两个数很费劲：并排一看就知道判别器是「在干活」还是「贴死/摆烂」。
        """
        s_d, v_d = self.get("Step/amp_d", smooth=50)
        s_r, v_r = self.get("Step/amp_d_ref", smooth=50)
        ax.set_title("amp_d vs amp_d_ref  (shaded = how much D prefers reference)", fontsize=10)
        # ⚠️ 必须**纯 ASCII**：云端 matplotlib 只有 DejaVu Sans（无 CJK 字形）
        ax.set_xlabel("iteration" if self.drop_iters <= 0
                      else f"iteration  (first {self.drop_iters} iters hidden)")
        ax.grid(alpha=0.3)
        if len(v_d) == 0 and len(v_r) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            return
        if len(v_r):
            ax.plot(s_r, v_r, lw=1.2, label="amp_d_ref (reference)")
        if len(v_d):
            ax.plot(s_d, v_d, lw=1.2, label="amp_d (policy)")
        # 差距围填 + 末尾数值（读「现在还剩多少可分辨度」）
        if len(v_d) and len(v_r):
            x = np.union1d(s_d, s_r)
            yd = np.interp(x, s_d, v_d)
            yr = np.interp(x, s_r, v_r)
            ax.fill_between(x, yd, yr, color="0.6", alpha=0.22)
            ax.annotate(f"gap@end = {yr[-1] - yd[-1]:+.3f}",
                        xy=(x[-1], 0.5 * (yd[-1] + yr[-1])), xytext=(-8, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=8, color="0.2",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec="0.7", alpha=0.85))
        # y 轴：分位数裁剪（与 `_panel` 同法），但**保证 0 线与 +1 基准线可见**
        if len(v_d) or len(v_r):
            allv = np.concatenate([v for v in (v_d, v_r) if len(v)])
            if len(allv) > 10:
                lo, hi = np.nanpercentile(allv, [0.5, 99.5])
            else:
                lo, hi = float(np.min(allv)), float(np.max(allv))
            lo, hi = min(lo, 0.0), max(hi, 1.0)
            pad = 0.08 * (hi - lo) + 1e-9
            ax.set_ylim(lo - pad, hi + pad)
        for y, lab in ((0.0, " 0 = coin flip"), (1.0, " +1 = reference-like")):
            ax.axhline(y, ls=":", lw=0.9, color="0.45")
            ax.text(0.99, y, lab, transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=7, color="0.35")
        ax.legend(fontsize=8, loc="best")

    def fig_foot_diagnostics(self):
        """⭐ 本任务最关键的设计风险点：步宽地板 / 交叉惩罚是否真的在起作用。

        `foot_sep_mean` 是「脚距在 root 偏航系下的横向间距」，同时充当
        **脚部参考 body 是否选对**的在线自检（退路是 *_ankle_roll_link）。
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        fig.suptitle("Foot Separation / Crossing Diagnostics  "
                     f"(sep_floor={SEP_FLOOR}, dataset median={SEP_DATASET_MEDIAN})", fontsize=13)
        self._panel(
            axes[0, 0], "Step/foot_sep_mean",
            "foot_sep_mean (expect 0.15~0.30)",
            smooth=50,
            hlines=[(SEP_FLOOR, f"sep_floor={SEP_FLOOR}", "--"),
                    (SEP_DATASET_MEDIAN, f"dataset median {SEP_DATASET_MEDIAN}", ":")],
        )
        self._panel(axes[0, 1], "Step/foot_cross_frac",
                    "foot_cross_frac (straight walk ~0; sideways may >0)", smooth=50)
        self._panel(axes[1, 0], "Episode_Reward/feet_lateral_width",
                    "Episode_Reward/feet_lateral_width (narrow-stance penalty)")
        self._panel(axes[1, 1], "Episode_Reward/feet_crossing",
                    "Episode_Reward/feet_crossing (true crossing penalty)")
        self.save(fig, "04_foot_diagnostics.png")

    def fig_terminations_perf(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Terminations / Perf / Velocity Tracking", fontsize=13)
        self._panel(axes[0, 0], "Episode_Termination/base_contact",
                    "Termination: base_contact (should drop below 0.1)")
        self._panel(axes[0, 1], "Episode_Termination/time_out", "Termination: time_out")
        self._panel(axes[1, 0], "Perf/total_fps", "FPS (steps/s)")
        self._panel(axes[1, 1], self.find("error_vel_xy") or "Episode/Metrics/base_velocity/error_vel_xy",
                    "linear velocity tracking error (lower is better)")
        self.save(fig, "05_terminations_perf.png")

    def fig_amp_task_split(self):
        """最终奖励里 task 与 AMP 风格各占多少。

        本仓库：R = λ·r_task + (1-λ)·c·r_amp
          - r_task（episode 总任务奖励）= `Episode_Reward_sum` × EP_LEN_S
            （IsaacLab: `Episode_Reward/x = episode_sum / max_episode_length_s`
             ⇒ 乘回 EP_LEN_S 就还原成 episode 总和）
          - task 贡献 = λ · r_task
          - **风格贡献 = (1-λ)·c·r_amp，其中 r_amp 是实测值**
            （TB 的 `Step/amp_style_reward`；它是 quad 形式 ∈ [0,1]，
             ⇒ **风格贡献恒非负且 ∈ [0, (1-λ)·c]**）
        ⚠️ 单位不同：`Episode_Reward/*` 是每秒速率 ⇒ ×EP_LEN_S（=max_episode_length_s）；
           `Step/*` 是每步均值 ⇒ ×**该回合实际走了多少步**。搞混会差一个量级。
        ⚠️ λ=0.75 不代表风格占 25%：实际风格权重是 (1-λ)·c = 0.025。
        """
        s_fin, v_fin = self.get(f"Train/mean_reward_{0}", smooth=1)
        if len(v_fin) == 0:
            print(f"[INFO][{self.mode}] 05 无 Train/mean_reward 数据，跳过")
            return
        parts = []
        for t in self.reward_tags():          # ⭐ 动态发现（含 T2 的 track_anchor_*）
            s, v = self.get(t, smooth=1)
            if len(v):
                parts.append((s, v))
        v_task_raw = _align_to(s_fin, parts) * EP_LEN_S      # 每秒率 → episode 总奖励
        task_contrib = LERP * v_task_raw

        # ⭐ 风格贡献：**实测**（旧版用残差 `v_fin - λ·r_task`，那不是测量值，
        #    它的零交叉是代数恒等式，会让人误判 AMP 退步）
        s_amp, v_amp = self.get(STYLE_TAG_NAME, smooth=1)
        # ⚠️ 乘的是**实测的平均回合步数**，不是最大步数 EP_LEN_STEPS：`Step/*` 是每步均值，
        #    风格贡献的 episode 总和 = (1−λ)·c·r_amp × 该回合实际走了多少步。
        #    用最大值（EP_LEN_STEPS = 20s×50Hz = 1000）会把不足整长的回合高估 2~3 倍
        #    —— 实测平均回合只有 ~400~520 步，这就是之前残差 −1.40 的来源。
        s_len, v_len = self.get("Train/mean_episode_length", smooth=1)
        if len(v_len):
            v_len_al = _align_to(s_fin, [(s_len, v_len)])
        else:
            v_len_al = np.full(len(s_fin), EP_LEN_STEPS)
        if len(v_amp) == 0:
            amp_contrib = v_fin - task_contrib
            style_src = f"⚠️ 残差 v_fin−λ·r_task（TB 缺 {STYLE_TAG_NAME}，非实测）"
        else:
            amp_contrib = (1.0 - LERP) * C_COEF * _align_to(s_fin, [(s_amp, v_amp)]) * v_len_al
            style_src = "实测 r_amp × (1−λ)·c × 实测回合步数"
        # 自检：两项贡献之和应当 ≈ final，差值就是误差（含 env 平均口径差异）
        resid = v_fin - task_contrib - amp_contrib

        s_s, v_fin_s = smooth_series(s_fin, v_fin, 100)
        _, v_task_s = smooth_series(s_fin, task_contrib, 100)
        _, v_amp_s = smooth_series(s_fin, amp_contrib, 100)
        _, v_sum_s = smooth_series(s_fin, task_contrib + amp_contrib, 100)
        _, v_res_s = smooth_series(s_fin, resid, 100)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f"Task vs AMP Contribution (both MEASURED)   "
            f"(λ={LERP}, c={C_COEF} → 风格权重 (1-λ)·c={ (1 - LERP) * C_COEF :.4f}){self.drop_note()}",
            fontsize=13)

        ax = axes[0]
        ax.plot(s_s, v_fin_s, lw=1.5, color="k", label="final (Train/mean_reward_0)")
        ax.plot(s_s, v_task_s, lw=1.2, color="tab:blue", label="task contrib (λ·r_task)")
        ax.plot(s_s, v_amp_s, lw=1.8, color="tab:orange", label=f"AMP contrib — {style_src}")
        ax.plot(s_s, v_sum_s, lw=1.0, color="tab:green", ls="--", label="task + AMP")
        ax.plot(s_s, v_res_s, lw=0.8, color="gray", alpha=0.7,
                label="残差 final−(task+AMP)  ← 自检，应接近 0")
        if len(s_s):
            ax.annotate(
                f"@last  final={v_fin_s[-1]:.2f}\n"
                f"task={v_task_s[-1]:.2f}   AMP={v_amp_s[-1]:.4f}\n"
                f"task+AMP={v_sum_s[-1]:.2f}   resid={v_res_s[-1]:.2f}",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, va="bottom",
                bbox=dict(fc="w", ec="0.7", alpha=0.85))
        ax.set_title("Contribution Magnitude (episode sum)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("reward")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1]
        # 力比用**非负分母** |task|+|AMP|：两项符号相反时用 v_fin 作分母会出鬼数
        denom = np.abs(v_task_s) + np.abs(v_amp_s)
        valid = denom > 1e-9
        p_task = np.where(valid, np.abs(v_task_s) / np.maximum(denom, 1e-9), np.nan) * 100.0
        p_amp = np.where(valid, np.abs(v_amp_s) / np.maximum(denom, 1e-9), np.nan) * 100.0
        ax.fill_between(s_s, 0, np.nan_to_num(p_task), color="tab:blue",
                        alpha=0.6, label="|task| share")
        ax.fill_between(s_s, np.nan_to_num(p_task),
                        np.nan_to_num(p_task + p_amp), color="tab:orange",
                        alpha=0.6, label="|AMP| share")
        ax.set_ylim(0, 100)
        if len(s_s):
            ax.annotate(f"@last  |task|={p_task[-1]:.1f}%   |AMP|={p_amp[-1]:.3f}%",
                        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, va="bottom",
                        bbox=dict(fc="w", ec="0.7", alpha=0.85))
        ax.set_title("Force Share  |task| vs |AMP|  (non-negative denominator)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("percent")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        self.save(fig, "06_amp_task_split.png")

        # 控制台摘要（方便 grep 自检）
        if len(s_s):
            print(f"[INFO][{self.mode}] 06: @{int(s_s[-1])}  final={v_fin_s[-1]:.2f}  "
                  f"task={v_task_s[-1]:.2f}  AMP={v_amp_s[-1]:.4f}  "
                  f"sum={v_sum_s[-1]:.2f}  resid={v_res_s[-1]:.2f}  |AMP|占={p_amp[-1]:.3f}%")

    # --- 文本摘要 ---
    def digest(self) -> list[str]:
        rows = []
        for tag, label, hint in [
            (f"Train/mean_reward_{0}", "平均回报（episode 总）", "应上升"),
            ("Train/mean_episode_length", "平均回合长度（步）", "应上升：几十→几百→上千"),
            ("Train/median_episode_length", "中位回合长度（步）", ""),
            ("Episode_Termination/base_contact", "触地终止比例", "应下降到 <0.1"),
            ("Episode_Termination/time_out", "超时终止比例", "应上升（能站住）"),
            ("Step/foot_sep_mean", "脚距均值", f"应稳在 0.15~0.30（floor={SEP_FLOOR}）"),
            ("Step/foot_cross_frac", "真交叉帧占比", "直线走≈0（侧移允许>0）"),
            ("Step/discriminator_reward", "判别器奖励（aux）", "缓降但不应为 0"),
            ("Step/amp_style_reward", "r_amp（quad∈[0,1]）", ""),
            ("Loss/discriminator_loss", "判别器损失", "不发散/不为 NaN"),
            ("Policy/mean_noise_std", "动作噪声 std", "缓慢下降"),
            ("Perf/total_fps", "FPS", ""),
        ]:
            step, val = self.latest_value(tag)
            rows.append((label, tag, step, val, hint))
        return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-tags", action="store_true",
                    help="只打印选中 run 里真实存在的 scalar tag 再退出")
    ap.add_argument("--list-runs", action="store_true",
                    help="列出可选 run（目录名 / 修改时间 / 最新 ckpt）再退出")
    ap.add_argument("--run", default=os.environ.get("LLP_RUN", ""),
                    help="选哪个 run：目录名**子串**（v9c / amp_walk_v9b_mvp_4096）"
                         "或完整目录名；留空 = 最新。命中多个时取最新一个")
    ap.add_argument("--all-matching", action="store_true",
                    help="--run 命中多个时，full 模式把它们全部合并（默认只合并选中的那个）")
    ap.add_argument("--outdir", default=None,
                    help=f"resume 模式输出目录（默认 {OUT_RESUME}；给了 --run 则默认 <它>/<run>）")
    ap.add_argument("--outdir-full", default=None,
                    help=f"full 模式输出目录（默认 {OUT_FULL}；给了 --run 则默认 <它>/<run>_full）")
    ap.add_argument("--drop-iters", type=int, default=100,
                    help="隐去开头的初始化段（单位=轮，默认 100；即 TB 前 10 个采样点）。"
                         "初始化段的振幅比后续趋势大得多 ⇒ 会压扁 y 轴。"
                         "给 0 = 完全不裁；要看更晚的阶段可以给 500 / 1000")
    args = ap.parse_args()

    runs = discover_runs()

    if args.list_runs:
        print(f"[INFO] LOG_ROOT = {LOG_ROOT}")
        print(f"[INFO] 共 {len(runs)} 个 run（旧 → 新）：")
        for r in runs:
            print("   ", run_info(r))
        print("[INFO] 用法示例： --run v9c   /   --run amp_walk_v9b_mvp_4096")
        return 0

    hits = match_runs(runs, args.run)
    if not hits:
        print(f"[ERROR] --run {args.run!r} 没匹配到任何 run。可选（新 → 旧）：")
        for r in reversed(runs[-12:]):
            print("   ", os.path.basename(os.path.normpath(r)))
        return 2

    latest = os.path.normpath(LATEST_OVERRIDE) if LATEST_OVERRIDE else os.path.normpath(hits[-1])
    if LATEST_OVERRIDE and os.path.normpath(LATEST_OVERRIDE) not in \
            [os.path.normpath(r) for r in runs]:
        print(f"[WARN] LLP_LATEST={LATEST_OVERRIDE} 不在时间戳 run 列表里，仍按它读")

    tag = slug(args.run) if args.run else ""
    outdir_resume = args.outdir or (os.path.join(OUT_RESUME, tag) if tag else OUT_RESUME)
    outdir_full = args.outdir_full or (os.path.join(OUT_FULL, tag + "_full") if tag else OUT_FULL)

    print(f"[INFO] LOG_ROOT = {LOG_ROOT}")
    print(f"[INFO] run 数 = {len(runs)}；--run = {args.run or '(未指定→最新)'}；命中 {len(hits)} 个")
    print(f"[INFO] 选中 run = {os.path.basename(os.path.normpath(latest))}")
    if len(hits) > 1:
        others = ", ".join(os.path.basename(os.path.normpath(h)) for h in hits[:-1])
        print(f"[INFO] （同命中的更早 run：{others}）")

    if args.list_tags:
        tags = available_tags(latest)
        print(f"[INFO] 选中 run 的 scalar tag 共 {len(tags)} 个：")
        for t in tags:
            print("   ", t)
        return 0

    # full 模式合并范围：默认全历史（或从 LLP_OLD_RUN 起）；一旦指定 --run，就只合并
    # “命中的 run”，避免把别的版本混进同一条曲线（v9b / v9c 对比时尤其重要）。
    old_idx = 0
    if OLD_RUN:
        for i, r in enumerate(runs):
            if OLD_RUN in r:
                old_idx = i
                break
    full_runs = runs[old_idx:]
    if args.run:
        sel = set(hits if args.all_matching else [latest])
        full_runs = [r for r in full_runs if r in sel]
    print(f"[INFO] full 模式合并 run 数 = {len(full_runs)}")
    print(f"[INFO] 输出目录： resume={outdir_resume}   full={outdir_full}")
    if args.drop_iters > 0:
        print(f"[INFO] 隐去初始化段：前 {args.drop_iters} 轮不绘（TB 每 10 轮 1 点"
              f" ⇒ 约前 {max(args.drop_iters // 10, 1)} 个采样点；可在图上 x 轴看到标注）")
    else:
        print("[INFO] --drop-iters 0 ⇒ 绘制全部数据点（含初始化段）")

    for mode, outdir, suffix in (("resume", outdir_resume, "(latest)"),
                                 ("full", outdir_full, "(full)")):
        if mode == "full" and SKIP_FULL:
            print("[INFO] LLP_SKIP_FULL=1，跳过 full 模式")
            continue
        if mode == "full" and len(full_runs) == 1:
            print("[INFO] 只有 1 个 run，full 模式与 resume 等价，跳过")
            continue
        print(f"======== 生成 {mode} 曲线 -> {outdir} ========")
        p = Plotter(mode, outdir, full_runs, latest, drop_iters=args.drop_iters)
        for fn in (p.fig_overview, p.fig_reward_terms, p.fig_amp_diagnostics,
                   p.fig_foot_diagnostics, p.fig_terminations_perf, p.fig_amp_task_split):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 单图失败不拖垮其余
                print(f"[WARN][{mode}] {fn.__name__} 失败: {type(exc).__name__}: {exc}")

        lines = [f"-------- 摘要 [{mode}]  run={os.path.basename(os.path.normpath(latest))} --------"]
        for label, tag, step, val, hint in p.digest():
            vs = "no data" if val is None else f"{val:,.4f}"
            st = "-" if step is None else f"{int(step)}"
            lines.append(f"  {label:<22} @{st:>6}  {vs:>14}   {hint}")
        for ln in lines:
            print(ln)
        print("")
        # 摘要落盘：本地 scp 回来看，不必开 SSH 就能知道训练健康度
        digest_path = os.path.join(outdir, "digest.txt")
        try:
            with open(digest_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            print(f"[INFO][{mode}] 已保存: {digest_path}")
        except OSError as exc:
            print(f"[WARN][{mode}] digest.txt 写入失败: {exc}")

    print("[INFO] 全部图表生成完毕")
    return 0


if __name__ == "__main__":
    sys.exit(main())
