#!/usr/bin/env python3
"""AMP 算法接线的**纯 CPU 端到端测试**（不需要 Isaac Sim / GPU）。

为什么需要它
------------
本机显存 4 GB 跑不了 Isaac Sim，所以 ⑤⑥ 的代码一直处在「从没跑过」的状态。
但其中**很大一部分根本不需要 Isaac Sim**：

- `instinct_rl` 是纯 PyTorch 的 RL 库；
- `WasabiPPOAmpMix` 只依赖 `instinct_rl` 与 torch；
- 奖励混合发生在算法侧，与仿真器无关。

所以本脚本用**真实的** `WasabiPPO` / `AmpStorage` / `PPO.process_env_step` /
`OnPolicyRunner` 配上假环境，在本地把下面三件事验证掉：

T1. `WasabiPPOAmpMix` 的 lerp 混合 —— 走**真实**的 `process_env_step` 管线，
    最后读 `RolloutStorage.rewards` 与 AMP_mjlab 的目标公式对账。
T2. `OnPolicyRunner` 的 `module:Class` 解析 —— 直接验证步骤⑥ 修的那 1 行。
T3. 测试用的 λ/c 等数字**直接取自 `amp_walk_ppo_cfg.py`**（复用静态检查器的解析器），
    所以这个测试不可能与真实配置漂移。

跑不了的部分（留给云上）：Isaac Sim 本身、`instinctlab.*` 的导入链（需要 `isaaclab`）、
`RewardManager` 与传感器构建。这些由 `cloud_smoke_amp_walk.sh` 覆盖。

用法
----
    conda activate instinct          # 需要 torch(CPU) + GitPython + gymnasium
    python tools/test_amp_pipeline.py
"""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent                     # .../llp/instinct
REPO = ROOT.parent                                                # .../llp
INSTINCT_RL_SRC = REPO / "instinct_xu/source/instinct_rl"
INSTINCTLAB_SRC = REPO / "instinct_xu/source/instinctlab"
ALGORITHMS_PY = INSTINCTLAB_SRC / "instinctlab/utils/wrappers/instinct_rl/algorithms.py"
AGENT_CFG = (
    INSTINCTLAB_SRC
    / "instinctlab/tasks/locomotion/config/g1/agents/amp_walk_ppo_cfg.py"
)

for p in (str(INSTINCT_RL_SRC), str(INSTINCTLAB_SRC), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

NUM_ENVS = 8
NUM_ACTIONS = 29
AMP_STATE_SIZE = 670
POLICY_OBS_SIZE = 626
#: `velocity_command` 在 policy 展平观测里的位置（与 AMP_WALK_OBSERVATION_DIMS 一致）
COMMAND_SLICE = slice(527, 530)

#: 假环境的 obs_format。形状与实际任务一致（用静态检查器校验过的那套）。
OBS_FORMAT = {
    "policy": {
        "height_scan": (527,),
        "velocity_command": (3,),
        "base_lin_vel": (3,),
        "projected_gravity": (3,),
        "base_ang_vel": (3,),
        "joint_pos_rel": (29,),
        "joint_vel": (29,),
        "last_action": (29,),
    },
    "critic": {"base_lin_vel": (3,)},
    # 5 项 × history 10 = 670（每项都 flatten 后求和）
    "amp_policy": {
        "projected_gravity": (10, 3),
        "joint_pos_rel": (290,),
        "joint_vel_rel": (290,),
        "base_lin_vel": (30,),
        "base_ang_vel": (30,),
    },
    "amp_reference": {
        "projected_gravity": (10, 3),
        "joint_pos_rel": (290,),
        "joint_vel_rel": (290,),
        "base_lin_vel": (30,),
        "base_ang_vel": (30,),
    },
}

_fails: list[str] = []
_n = 0

#: ⭐ 两种 policy 观测布局（见 `amp_walk_v9_env_cfg.OBSERVATION_DIMS` 与 `mirror.py SCAN_LAYOUTS`）：
#:   v1~v8 = 626（527 射线，扫描居中）、v9 = 378（279 射线，扫描中心前移 1.5 m）。
POLICY_LAYOUTS: dict[str, tuple[int, int]] = {"626": (626, 527), "378": (378, 279)}
POLICY_FORMAT_378 = dict(OBS_FORMAT["policy"], height_scan=(279,))
POLICY_FORMATS = {"626": dict(OBS_FORMAT["policy"]), "378": POLICY_FORMAT_378}


def obs_format_for(tag: str) -> dict:
    """把 `OBS_FORMAT` 的 policy 段换成指定布局（其余专家段不动）。"""
    fmt = dict(OBS_FORMAT)
    fmt["policy"] = POLICY_FORMATS[tag]
    return fmt


def check(name: str, cond: bool, detail: str = "") -> None:
    global _n
    _n += 1
    if cond:
        print(f"  ✅ {name}" + (f"  {detail}" if detail else ""))
    else:
        _fails.append(name)
        print(f"  ❌ {name}  {detail}")


def load_module_by_path(path: Path, name: str):
    """按**文件路径**加载模块，绕过包的 `__init__`。

    `instinctlab.utils.wrappers.instinct_rl.__init__` 会 import `isaaclab`（本机没有），
    但 `algorithms.py` 自己只依赖 `instinct_rl`。绕过包初始化不影响被测代码本身，
    「包 __init__ 是否正确转发」由静态检查器 K 类负责。
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # ⚠️ 必须先注册进 `sys.modules`：`@dataclass` 内部会回查 `cls.__module__`，
    #    不注册就报 `AttributeError: 'NoneType' object has no attribute '__dict__'`。
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def preload_instinctlab_stubs() -> None:
    """让 `algorithms.py` 能在**本机**（没装 isaaclab）加载。

    `algorithms.py` 里有 `from instinctlab.utils.mirror import MirrorSpec`；
    直接 import 会执行 `instinctlab/__init__.py` → 拖进 isaaclab/Kit ⇒ 本机必挂。
    这里把 `instinctlab` 及其子包注册成**空壳包**（只给 `__path__`），
    这样 `instinctlab.utils.mirror` 能按路径加载，而 `__init__.py` 一个都不执行。
    """
    import types

    pkg_root = INSTINCTLAB_SRC / "instinctlab"
    for name, sub in (
        ("instinctlab", ""),
        ("instinctlab.utils", "utils"),
        ("instinctlab.utils.wrappers", "utils/wrappers"),
        ("instinctlab.utils.wrappers.instinct_rl", "utils/wrappers/instinct_rl"),
    ):
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__path__ = [str(pkg_root / sub)]
        sys.modules[name] = mod
    if "instinctlab.utils.mirror" not in sys.modules:
        load_module_by_path(pkg_root / "utils/mirror.py", "instinctlab.utils.mirror")


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


class DummyActorCritic(nn.Module):
    """凑齐 `PPO.act` / `PPO.process_env_step` 用到的最小接口。"""

    is_recurrent = False

    def __init__(self):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))
        self.action_mean = torch.zeros(1, NUM_ACTIONS)
        self.action_std = torch.ones(1, NUM_ACTIONS)

    def act(self, obs, masks=None, hidden_states=None):
        n = obs.shape[0]
        self.action_mean = torch.zeros(n, NUM_ACTIONS, device=obs.device)
        self.action_std = torch.ones(n, NUM_ACTIONS, device=obs.device)
        return self.action_mean.clone()

    def act_inference(self, obs):
        return torch.zeros(obs.shape[0], NUM_ACTIONS, device=obs.device)

    def update_distribution(self, obs, masks=None, hidden_states=None):
        """⚠️ `mirror_loss()` 里会回读：`ac.act(...)` 覆盖分布后要还原。"""
        self.act(obs, masks, hidden_states)

    def evaluate(self, obs, masks=None, hidden_states=None):
        return torch.zeros(obs.shape[0], 1, device=obs.device)

    def get_actions_log_prob(self, actions):
        return torch.zeros(actions.shape[0], 1, device=actions.device)

    def reset(self, dones=None):
        pass


class ReadoutActorCritic(DummyActorCritic):
    """μ_π(o) = o[:, sl]（确定性线性读出）—— 给镜像损失一个**可手算的非退化**取值。

    选不同的 `sl` 就能造出「等变」与「非等变」两种情形：
      · `sl` = joint_pos_rel 段 ⇒ M_π 与 M_act 完全一致 ⇒ 损失应为 **0**（正对照）；
      · `sl` = height_scan 前缀 ⇒ 前缀不是关节向量，两侧变换不同 ⇒ 损失 **> 0**。
    """

    def __init__(self, sl: slice):
        super().__init__()
        self.sl = sl

    def act(self, obs, masks=None, hidden_states=None):
        self.action_mean = obs[:, self.sl].clone()
        self.action_std = torch.ones_like(self.action_mean)
        return self.action_mean.clone()


def make_obs_pack(command: torch.Tensor, obs_dim: int = POLICY_OBS_SIZE,
                  cmd_slice: slice = COMMAND_SLICE) -> dict:
    n = command.shape[0]
    policy = torch.zeros(n, obs_dim)
    policy[:, cmd_slice] = command
    return {
        "policy": policy,
        "critic": torch.zeros(n, 3),
        "amp_policy": torch.zeros(n, AMP_STATE_SIZE),
        "amp_reference": torch.zeros(n, AMP_STATE_SIZE),
    }


def build_alg(mix_module, fmt: dict | None = None, ac=None, **mix_kwargs):
    """构造一个真实的 `WasabiPPOAmpMix`（`fmt` 可换 policy 观测布局）。"""
    alg = mix_module.WasabiPPOAmpMix(
        ac if ac is not None else DummyActorCritic(),
        discriminator_kwargs={"hidden_sizes": [16, 8], "nonlinearity": "ReLU"},
        discriminator_reward_coef=mix_kwargs.pop("discriminator_reward_coef"),
        discriminator_reward_type="quad",
        discriminator_loss_func="MSELoss",
        auxiliary_reward_per_env_reward_coefs=[1.0],
        device="cpu",
        **mix_kwargs,
    )
    alg.init_storage(
        NUM_ENVS,
        num_transitions_per_env=4,
        obs_format=fmt if fmt is not None else OBS_FORMAT,
        num_actions=NUM_ACTIONS,
        num_rewards=1,
    )
    # 把所有判别器参数清零 → d ≡ 0 → quad(d) = 1 - 0.25*(0-1)^2 = 0.75（已知常数）
    with torch.no_grad():
        for p in alg.discriminator.parameters():
            p.zero_()
    return alg


def run_step(alg, r_task: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    """跑一次**真实**的 act + process_env_step，返回写进 storage 的最终奖励。

    ⚠️ 必须用 `storage.step - 1` 取刚刚写入的那一条：`RolloutStorage` 是
    (num_transitions, num_envs, num_rewards) 的环形推进缓冲，同一个 alg 实例
    多次调用会写到不同槽位（第一版测试就踩了这个坑，把「运动门控」的断言读成了
    上一条「静止」的结果）。
    """
    obs_pack = make_obs_pack(command)
    infos = {"observations": obs_pack, "step": {}}
    alg.act(obs_pack["policy"], obs_pack["critic"])
    alg.process_env_step(
        r_task.unsqueeze(1),                                   # (N, num_rewards)
        torch.zeros(NUM_ENVS, dtype=torch.long),
        infos,
        obs_pack["policy"],
        obs_pack["critic"],
    )
    return alg.storage.rewards[alg.storage.step - 1].clone(), infos


# ---------------------------------------------------------------------------
# T1. lerp 奖励混合（走真实的 process_env_step）
# ---------------------------------------------------------------------------


def test_lerp_pipeline(mix_module, lam: float, coef: float) -> None:
    print("T1. WasabiPPOAmpMix 的 lerp 奖励混合（真实 process_env_step 管线）")
    r_task = torch.tensor([0.3, -1.2, 2.5, 0.0, 1.7, -4.0, 0.9, 3.3])
    command = torch.tensor([[0.8, 0.0, 0.0]] * NUM_ENVS)       # 运动指令（门控打开）
    quad = 0.75                                                # d ≡ 0 ⇒ clamp(1-0.25(d-1)^2)

    # --- lerp ---
    alg = build_alg(mix_module, reward_mix_mode="lerp", style_lerp=lam,
                    discriminator_reward_coef=coef)
    got, infos = run_step(alg, r_task, command)
    want = (1.0 - lam) * coef * quad + lam * r_task
    err = float((got.squeeze(-1) - want).abs().max())
    check("lerp: R = (1-λ)·c·r_amp + λ·r_task", err < 1e-6, f"最大偏差 {err:.3e}")
    print(f"     r_task={r_task.tolist()}")
    print(f"     got    ={[round(v, 6) for v in got.squeeze(-1).tolist()]}")
    print(f"     want   ={[round(v, 6) for v in want.tolist()]}")

    # 日志键：原始风格奖励必须能读到，且**不得**被重复计入
    check("lerp: 日志里能读到原始风格奖励 amp_style_reward",
          "amp_style_reward" in infos["step"]
          and float(infos["step"]["amp_style_reward"].max() - quad) < 1e-6,
          f"= {float(infos['step']['amp_style_reward'].flatten()[0]):.6f}（quad(d=0)={quad}）")
    check("lerp: amp_style_reward 未被重复计入奖励（coef 必须为 0）",
          abs(float(alg.amp_style_reward_coef)) == 0.0,
          f"amp_style_reward_coef={alg.amp_style_reward_coef}")

    # --- additive（父类原样，零行为变化）---
    alg2 = build_alg(mix_module, reward_mix_mode="additive", style_lerp=lam,
                     discriminator_reward_coef=coef)
    got2, _ = run_step(alg2, r_task, command)
    want2 = r_task + coef * quad
    err2 = float((got2.squeeze(-1) - want2).abs().max())
    check("additive: R = r_task + c·r_amp（= WasabiPPO 原样）", err2 < 1e-6,
          f"最大偏差 {err2:.3e}")

    # --- disable_style_on_stand（低速时只留 λ·r_task）---
    alg3 = build_alg(mix_module, reward_mix_mode="lerp", style_lerp=lam,
                     discriminator_reward_coef=coef, disable_style_on_stand=True,
                     stand_speed_threshold=0.1)
    zero_cmd = torch.zeros(NUM_ENVS, 3)                        # cmd_speed = 0 ≤ 0.1
    got3, _ = run_step(alg3, r_task, zero_cmd)
    err3 = float((got3.squeeze(-1) - lam * r_task).abs().max())
    check("disable_style_on_stand: 静止时 R = λ·r_task（风格项被丢掉）", err3 < 1e-6,
          f"最大偏差 {err3:.3e}")

    # 运动时门控打开 → 回到 lerp 公式
    got4, _ = run_step(alg3, r_task, command)
    err4 = float((got4.squeeze(-1) - want).abs().max())
    check("disable_style_on_stand: 运动时回到完整 lerp 公式", err4 < 1e-6,
          f"最大偏差 {err4:.3e}")

    # --- fail-fast ---
    try:
        build_alg(mix_module, reward_mix_mode="lerp", style_lerp=lam,
                  discriminator_reward_coef=coef, disable_style_on_stand=True)
        # 这个能成功（policy 里有 velocity_command）；换一个没有 command 的 obs_format 试
        alg5 = mix_module.WasabiPPOAmpMix(
            DummyActorCritic(), discriminator_kwargs={"hidden_sizes": [8]},
            discriminator_reward_coef=coef, device="cpu",
            reward_mix_mode="lerp", disable_style_on_stand=True,
        )
        bad_format = dict(OBS_FORMAT)
        bad_format["policy"] = {"height_scan": (527,), "joint_pos_rel": (29,)}
        alg5.init_storage(NUM_ENVS, 4, obs_format=bad_format,
                          num_actions=NUM_ACTIONS, num_rewards=1)
        check("fail-fast: 定位不到速度指令项时启动即报错", False, "没有抛异常")
    except ValueError as e:
        check("fail-fast: 定位不到速度指令项时启动即报错", "速度指令" in str(e), str(e)[:60])

    try:
        alg6 = mix_module.WasabiPPOAmpMix(
            DummyActorCritic(), discriminator_kwargs={"hidden_sizes": [8]},
            discriminator_reward_coef=coef, device="cpu", reward_mix_mode="nonsense",
        )
        check("fail-fast: reward_mix_mode 非法值时构造即报错", False, "没有抛异常")
    except ValueError:
        check("fail-fast: reward_mix_mode 非法值时构造即报错", True)


# ---------------------------------------------------------------------------
# T3. 镜像损失 × 两种观测布局
# ---------------------------------------------------------------------------


def test_mirror_loss_layouts(mix_module) -> None:
    """⭐ 镜像损失在 626（v1~v8）与 378（v9）两种布局下都要能跑。

    回归背景：`mirror.py` 曾把布局写死成 626 ⇒ v9（378）一开镜像损失就在
    **第一次 `PPO.update()`** 崩：`ValueError: policy obs 最后一维应为 626，收到 378`。
    这里直接用构造出的 obs 打这条路径（不跑完整 `update`，避开 minibatch 内部字段）。
    """
    import joint_order  # （本机工具，已在 sys.path 里）

    print("\nT3. ⭐ 镜像损失 × 两种观测布局（626 / 378）—— 回归 v9 的 PPO.update() 崩溃")
    joints = list(joint_order.ISAACLAB_ORDER)
    check("关节名 = 29 个（= 策略输出维）", len(joints) == NUM_ACTIONS, f"got {len(joints)}")

    # 0) 开了镜像损失却没给关节名 ⇒ 启动即 fail-fast
    try:
        alg0 = build_alg(mix_module, use_mirror_loss=True, discriminator_reward_coef=0.5)
        alg0.mirror_loss(torch.randn(4, POLICY_OBS_SIZE))
        check("fail-fast: use_mirror_loss 却没给 mirror_joint_names", False, "没有抛异常")
    except ValueError as e:
        check("fail-fast: use_mirror_loss 却没给 mirror_joint_names",
              "mirror_joint_names" in str(e), str(e)[:60])

    for tag, (dim, scan) in POLICY_LAYOUTS.items():
        n_jp = scan + 12                      # joint_pos_rel 段起点（scan 后：cmd/lin/grav/ang）
        fmt = obs_format_for(tag)
        obs = torch.randn(6, dim)

        def primed(a):
            """⭐ 先把 `action_mean` 置为 μ_π(o)。

            `mirror_loss()` 读的是 `ac.action_mean` 而不是自己再前向一次：
            它的 docstring 写明「= μ_π(o)（`compute_losses` 刚算过）」。
            ⚠️ 所以忘调 `act()` 不会报错，而是**静默拿上一次（或初始）的均值**
            去算损失 —— 单测里就撞到过（损失变成了 ~29 而不是 0）。
            """
            a.actor_critic.act(obs, masks=None, hidden_states=None)
            return a

        # 1) 非等变读出（取 height_scan 前缀）⇒ 损失应为**有限正数**
        alg = build_alg(
            mix_module, fmt=fmt, ac=ReadoutActorCritic(slice(0, NUM_ACTIONS)),
            use_mirror_loss=True, mirror_joint_names=joints,
            mirror_loss_coeff=0.1, discriminator_reward_coef=0.5,
        )
        l1 = float(primed(alg).mirror_loss(obs))
        check(f"{tag}: mirror_loss 跑通且有限正数", math.isfinite(l1) and l1 > 1e-6, f"got {l1:.6g}")
        check(f"{tag}: mirror_coef 记录 = 0.1", abs(float(alg.mirror_coef) - 0.1) < 1e-12,
              f"got {alg.mirror_coef}")

        # 2) 等变读出（取 joint_pos_rel 段）⇒ 损失应为 0（**正对照**）
        alg2 = build_alg(
            mix_module, fmt=fmt, ac=ReadoutActorCritic(slice(n_jp, n_jp + NUM_ACTIONS)),
            use_mirror_loss=True, mirror_joint_names=joints,
            mirror_loss_coeff=0.1, discriminator_reward_coef=0.5,
        )
        l2 = float(primed(alg2).mirror_loss(obs))
        check(f"{tag}: 等变读出（μ = joint_pos_rel 段）损失 ≈ 0", l2 < 1e-10, f"got {l2:.3g}")

        # 3) 拿错布局的 obs ⇒ fail-fast，而不是静默算错
        try:
            alg.mirror_loss(torch.randn(6, dim + 100))
            check(f"{tag}: 观测维度不符 ⇒ fail-fast", False, "没有抛异常")
        except ValueError as e:
            check(f"{tag}: 观测维度不符 ⇒ fail-fast", "最后一维" in str(e), str(e)[:50])


# ---------------------------------------------------------------------------
# T2. OnPolicyRunner 的 module:Class 解析
# ---------------------------------------------------------------------------


def test_runner_module_class() -> None:
    print("\nT2. OnPolicyRunner 的 'module:Class' 算法解析（步骤⑥ 修的那 1 行）")

    # 2.1 先证明 bug 确实存在过：无参调用 importlib.import_module() 会 TypeError
    try:
        importlib.import_module()
        check("原 bug 复现：importlib.import_module() 无参调用会 TypeError", False,
              "居然没抛异常")
    except TypeError as e:
        check("原 bug 复现：importlib.import_module() 无参调用会 TypeError", True,
              str(e)[:50])

    # 2.2 再证明修好后 runner 真的能按 "module:Class" 建出算法
    try:
        import tensorboardX  # noqa: F401
    except ImportError:
        print("  ⏭  跳过：需要 tensorboardX（pip install tensorboardX）")
        return
    try:
        from instinct_rl.runners.on_policy_runner import OnPolicyRunner
    except ImportError as e:
        print(f"  ⏭  跳过：runner 的依赖没装全（{e}；pip install regex）")
        return

    class FakeEnv:
        num_envs = NUM_ENVS
        num_actions = NUM_ACTIONS
        num_rewards = 1
        num_steps_per_env = 4

        def get_obs_format(self):
            return OBS_FORMAT

        def reset(self):
            return None, {"observations": make_obs_pack(torch.zeros(NUM_ENVS, 3))}

    train_cfg = {
        # ⚠️ 这里**故意**用 instinct_rl 自己的类：本机没有 isaaclab，
        #    无法导入 `instinctlab.utils.wrappers.instinct_rl`。
        #    它验证的是**解析机制**；目标字符串的可达性由静态检查器 K 类负责。
        "algorithm": {
            "class_name": "instinct_rl.algorithms:WasabiPPO",
            "discriminator_kwargs": {"hidden_sizes": [16, 8], "nonlinearity": "ReLU"},
            "auxiliary_reward_per_env_reward_coefs": [1.0],
            "value_loss_coef": 1.0,
            "clip_param": 0.2,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [32, 16],
            "critic_hidden_dims": [32, 16],
            "activation": "elu",
        },
        "num_steps_per_env": 4,
        "save_interval": 100,
        "log_interval": 1,
        "normalizers": {},
    }
    try:
        runner = OnPolicyRunner(FakeEnv(), train_cfg, log_dir=None, device="cpu")
        check("修好后：OnPolicyRunner 接受 'module:Class' 形式的算法 class_name",
              type(runner.alg).__name__ == "WasabiPPO",
              f"实际构建出 {type(runner.alg).__name__}")
    except Exception as e:
        check("修好后：OnPolicyRunner 接受 'module:Class' 形式的算法 class_name", False,
              f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# T3. 测试数字取自真实配置（防漂移）
# ---------------------------------------------------------------------------


def test_config_numbers() -> tuple[float, float]:
    print("\nT3. λ / c / class_name 直接取自 amp_walk_ppo_cfg.py")
    import check_amp_walk_task as chk  # noqa: PLC0415  （同目录的静态检查器）

    cfg = chk._class_literals(AGENT_CFG, "AmpWalkAlgorithmCfg")
    lam = cfg.get("style_lerp")
    coef = cfg.get("discriminator_reward_coef")
    check("能解析出 style_lerp / discriminator_reward_coef",
          isinstance(lam, (int, float)) and isinstance(coef, (int, float)),
          f"λ={lam}, c={coef}")
    check("reward_mix_mode = 'lerp'", cfg.get("reward_mix_mode") == "lerp")
    check("auxiliary_reward_per_env_reward_coefs = [1.0]",
          cfg.get("auxiliary_reward_per_env_reward_coefs") == [1.0],
          f"实际 {cfg.get('auxiliary_reward_per_env_reward_coefs')!r}")
    check("class_name 指向 instinctlab 侧的 WasabiPPOAmpMix",
          isinstance(cfg.get("class_name"), str)
          and cfg["class_name"].endswith(":WasabiPPOAmpMix"))
    return float(lam), float(coef)


# ---------------------------------------------------------------------------

def main() -> int:
    print(f"算法源目录 : {INSTINCT_RL_SRC.relative_to(REPO)}")
    print(f"被测算   : {ALGORITHMS_PY.relative_to(REPO)}")
    print(f"torch      : {torch.__version__}  (cuda={torch.cuda.is_available()})")
    print()
    if not ALGORITHMS_PY.is_file():
        print(f"找不到 {ALGORITHMS_PY}")
        return 1
    preload_instinctlab_stubs()
    mix_module = load_module_by_path(ALGORITHMS_PY, "amp_mix_algorithms")
    lam, coef = test_config_numbers()
    print()
    test_lerp_pipeline(mix_module, lam, coef)
    test_mirror_loss_layouts(mix_module)
    test_runner_module_class()

    print()
    print("=" * 72)
    if _fails:
        print(f"❌ 失败 {len(_fails)}/{_n}：")
        for f in _fails:
            print(f"   - {f}")
        return 1
    print(f"✅ 全部通过（{_n} 项）—— 注意：这仍不能替代云机器上的真实冒烟测试")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
