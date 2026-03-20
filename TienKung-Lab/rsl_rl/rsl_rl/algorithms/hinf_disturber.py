"""
H-Infinity Disturber Plugin for DreamWaQ PPO Training.

实现论文 "Learning H-Infinity Locomotion Control" 中的可学习捣乱者模块。
通过双梯度下降对抗训练，提高机器人在复杂地形（楼梯等）中的鲁棒性。

核心思想:
  - Disturber 网络根据当前物理状态生成对抗性外力 d(s_t)
  - Actor (PPO) 最小化代价，Disturber 最大化代价
  - Lagrangian 乘子 λ 约束外力强度上界 η
  - 梯度严格隔离：Disturber 梯度绝不污染 DreamWaQ VAE 编码器

论文公式对应:
  - 公式(9):  Disturber 目标 max_d E[Σ C_t - γ^t * d(a_t|s_t)]
  - 公式(11): L^{dist}(θ) = L^{PPO}(θ) + λ * Lagrangian_constraint
  - 公式(12): 双梯度交替更新 π, d, λ

作者: 基于 TienKung-Lab DreamWaQ 代码库集成
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Optional


# ===========================================================================
# 1. 超参数配置
# ===========================================================================

@dataclass
class HInfDisturberCfg:
    """H-Infinity Disturber 超参数配置。"""

    # --- 网络结构 ---
    # Disturber Actor 输入维度（纯物理状态，不含 VAE 潜空间）
    # G1 DWAQ 中: ang_vel(3) + proj_gravity(3) + cmd(3) + joint_pos(23) + joint_vel(23) + action(23) = 78
    # 若开启 gait_phase: +4 = 82
    disturber_obs_dim: int = 78
    # Disturber Actor 隐藏层
    actor_hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    # Disturber Critic 隐藏层
    critic_hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    # 输出维度：3D 线性力向量 [Fx, Fy, Fz]
    force_dim: int = 3

    # --- H-Infinity 约束 ---
    # η: 外力 L2 范数上界 (单位: N)
    # 论文建议从 100N 开始，逐步增大
    eta: float = 50.0

    # --- 优化器 ---
    disturber_lr: float = 3e-4
    lagrangian_lr: float = 1e-3   # α: Lagrangian 乘子学习率

    # --- Lagrangian 乘子初始值与范围 ---
    lambda_init: float = 1.0
    lambda_min: float = 0.0
    lambda_max: float = 10.0

    # --- PPO 参数（Disturber Critic 用）---
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95            # GAE lambda
    value_loss_coef: float = 1.0
    num_learning_epochs: int = 1
    num_mini_batches: int = 4
    max_grad_norm: float = 1.0

    # --- Cost 函数权重（仅 tracking error，严格按论文定义）---
    # C_t = w_lin * ||v_cmd - v_actual||^2 + w_ang * ||ω_cmd - ω_actual||^2
    cost_lin_vel_weight: float = 1.0
    cost_ang_vel_weight: float = 0.5

    # --- 施力目标 body name（在 IsaacLab 中对应 torso_link）---
    # 外力施加在机器人躯干质心
    apply_body_name: str = "torso_link"

    # --- 训练开关 ---
    # 前 N 个 iteration 不启用 Disturber（让 Actor 先学会基本行走）
    warmup_iterations: int = 500
    # ε-greedy 探索概率
    epsilon_greedy: float = 0.2


# ===========================================================================
# 2. Disturber Actor 网络
# ===========================================================================

class DisturberActor(nn.Module):
    """
    Disturber Actor 网络：输出对抗性 3D 线性力向量。

    输入: 纯物理状态 s_t（绝对不含 DreamWaQ VAE 潜空间特征）
    输出: d(s_t) ∈ R^3，经 L2 范数截断使 ||d|| ≤ η

    关键设计:
      - 输出层使用 Tanh 激活，将原始输出映射到 [-1, 1]^3
      - 再乘以 η 并做 L2 范数截断，保证约束严格成立
    """

    def __init__(self, obs_dim: int, hidden_dims: list, force_dim: int = 3):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        # 输出层：无激活，后续手动截断
        layers.append(nn.Linear(in_dim, force_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, eta: float) -> torch.Tensor:
        """
        Args:
            obs:  [N, obs_dim]  纯物理状态（已 detach，不含 VAE 特征）
            eta:  float         L2 范数上界 (N)

        Returns:
            force: [N, 3]  对抗力向量，严格满足 ||force||_2 ≤ η
        """
        # [N, 3] 原始输出
        raw = self.net(obs)

        # --- L2 范数截断（论文核心约束）---
        # 计算每个样本的 L2 范数: [N, 1]
        norms = raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        # 若范数超过 η，则等比缩放；否则保持原值
        # scale = min(1, η / ||raw||)
        scale = (eta / norms).clamp(max=1.0)
        force = raw * scale  # [N, 3]，保证 ||force|| ≤ η

        return force


# ===========================================================================
# 3. Disturber Critic 网络
# ===========================================================================

class DisturberCritic(nn.Module):
    """
    Disturber Critic 网络：估计 Cost Value Function V^{cost}(s_t)。

    输入: 纯物理状态 s_t（同 Actor，不含 VAE 特征）
    输出: 标量 cost value

    注意: Disturber Critic 估计的是"代价"（越大越好对 Disturber），
    与 PPO Critic 估计的"奖励"（越大越好对 Actor）方向相反。
    """

    def __init__(self, obs_dim: int, hidden_dims: list):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [N, obs_dim]  纯物理状态（已 detach）

        Returns:
            value: [N, 1]  cost value 估计
        """
        return self.net(obs)


# ===========================================================================
# 4. Rollout Storage（Disturber 专用）
# ===========================================================================

class DisturberRolloutStorage:
    """
    Disturber 专用 Rollout Storage。

    存储内容:
      - disturber_obs:  纯物理状态（不含 VAE 特征）
      - costs:          C_t（仅 tracking error）
      - dones:          episode 终止标志
      - cost_values:    V^{cost}(s_t) 估计值
      - cost_returns:   GAE 计算的 cost returns
      - cost_advantages: GAE 计算的 cost advantages
      - actions_log_prob: Disturber action log prob（用于 PPO 更新）
      - action_mean/sigma: 用于 KL 自适应学习率
    """

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        disturber_obs_dim: int,
        force_dim: int = 3,
        device: str = "cuda",
    ):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        T, N = num_transitions_per_env, num_envs

        # 纯物理状态: [T, N, obs_dim]
        self.disturber_obs = torch.zeros(T, N, disturber_obs_dim, device=device)
        # 对抗力: [T, N, 3]
        self.forces = torch.zeros(T, N, force_dim, device=device)
        # 任务代价 C_t（仅 tracking error）: [T, N, 1]
        self.costs = torch.zeros(T, N, 1, device=device)
        # episode 终止: [T, N, 1]
        self.dones = torch.zeros(T, N, 1, device=device).byte()
        # Disturber Critic 估计的 cost value: [T, N, 1]
        self.cost_values = torch.zeros(T, N, 1, device=device)
        # GAE 计算结果
        self.cost_returns = torch.zeros(T, N, 1, device=device)
        self.cost_advantages = torch.zeros(T, N, 1, device=device)
        # Disturber PPO 所需
        self.actions_log_prob = torch.zeros(T, N, 1, device=device)
        self.action_mean = torch.zeros(T, N, force_dim, device=device)
        self.action_sigma = torch.zeros(T, N, force_dim, device=device)

        self.step = 0

    def add(
        self,
        disturber_obs: torch.Tensor,   # [N, obs_dim]
        forces: torch.Tensor,          # [N, 3]
        costs: torch.Tensor,           # [N]
        dones: torch.Tensor,           # [N]
        cost_values: torch.Tensor,     # [N, 1]
        actions_log_prob: torch.Tensor,# [N]
        action_mean: torch.Tensor,     # [N, 3]
        action_sigma: torch.Tensor,    # [N, 3]
    ):
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError("DisturberRolloutStorage overflow")

        t = self.step
        # 所有操作均为纯 GPU Tensor 赋值，无 for 循环
        self.disturber_obs[t].copy_(disturber_obs)
        self.forces[t].copy_(forces)
        self.costs[t].copy_(costs.view(-1, 1))
        self.dones[t].copy_(dones.view(-1, 1))
        self.cost_values[t].copy_(cost_values)
        self.actions_log_prob[t].copy_(actions_log_prob.view(-1, 1))
        self.action_mean[t].copy_(action_mean)
        self.action_sigma[t].copy_(action_sigma)
        self.step += 1

    def compute_returns(self, last_cost_values: torch.Tensor, gamma: float, lam: float):
        """
        使用 GAE 计算 cost returns 和 advantages。

        注意: Disturber 是"最大化"代价，因此 advantage 符号与 PPO 相反。
        这里我们保持与 PPO 相同的计算方式（最大化等价于最小化负代价），
        在 Disturber loss 中取负号即可。

        Args:
            last_cost_values: [N, 1]  最后一步的 cost value 估计
            gamma: 折扣因子
            lam:   GAE lambda
        """
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_cost_values
            else:
                next_values = self.cost_values[step + 1]
            # [N, 1]
            not_done = 1.0 - self.dones[step].float()
            delta = (
                self.costs[step]
                + not_done * gamma * next_values
                - self.cost_values[step]
            )
            advantage = delta + not_done * gamma * lam * advantage
            self.cost_returns[step] = advantage + self.cost_values[step]

        # 归一化 advantages（提升训练稳定性）
        self.cost_advantages = self.cost_returns - self.cost_values
        self.cost_advantages = (
            (self.cost_advantages - self.cost_advantages.mean())
            / (self.cost_advantages.std() + 1e-8)
        )

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 1):
        """
        生成 mini-batch，纯 GPU Tensor 操作，无 for 循环遍历 envs。

        Yields:
            (obs_b, forces_b, costs_b, cost_values_b, cost_advantages_b,
             cost_returns_b, log_prob_b, mu_b, sigma_b)
        """
        # 展平 [T, N, ...] → [T*N, ...]
        # T*N = num_transitions_per_env * num_envs（batch 总大小）
        obs_flat = self.disturber_obs.flatten(0, 1)          # [T*N, obs_dim]
        forces_flat = self.forces.flatten(0, 1)              # [T*N, 3]
        costs_flat = self.costs.flatten(0, 1)                # [T*N, 1]
        values_flat = self.cost_values.flatten(0, 1)         # [T*N, 1]
        adv_flat = self.cost_advantages.flatten(0, 1)        # [T*N, 1]
        ret_flat = self.cost_returns.flatten(0, 1)           # [T*N, 1]
        logp_flat = self.actions_log_prob.flatten(0, 1)      # [T*N, 1]
        mu_flat = self.action_mean.flatten(0, 1)             # [T*N, 3]
        sigma_flat = self.action_sigma.flatten(0, 1)         # [T*N, 3]

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for _ in range(num_epochs):
            # 随机打乱索引（GPU 上操作）
            indices = torch.randperm(batch_size, device=self.device)
            for i in range(num_mini_batches):
                idx = indices[i * mini_batch_size: (i + 1) * mini_batch_size]
                yield (
                    obs_flat[idx],      # [B, obs_dim]
                    forces_flat[idx],   # [B, 3]
                    costs_flat[idx],    # [B, 1]
                    values_flat[idx],   # [B, 1]
                    adv_flat[idx],      # [B, 1]
                    ret_flat[idx],      # [B, 1]
                    logp_flat[idx],     # [B, 1]
                    mu_flat[idx],       # [B, 3]
                    sigma_flat[idx],    # [B, 3]
                )

    def clear(self):
        self.step = 0


# ===========================================================================
# 5. HInfDisturberPlugin 主类
# ===========================================================================

class HInfDisturberPlugin:
    """
    H-Infinity Disturber 插件。

    职责:
      1. 在 rollout 阶段：根据当前物理状态生成对抗力，注入 IsaacLab 环境
      2. 在 update 阶段：执行双梯度下降，交替更新 Disturber 和 Lagrangian 乘子
      3. 提供 `compute_cost()` 接口：计算仅含 tracking error 的 C_t

    使用方式（见 ppo_trainer_patch.py）:
      plugin = HInfDisturberPlugin(cfg, robot_articulation, device)
      # rollout 阶段
      plugin.step(disturber_obs, dones)  # 注入外力
      plugin.record(cost, done, cost_value, ...)
      # update 阶段
      plugin.update(last_disturber_obs)
    """

    def __init__(
        self,
        cfg: HInfDisturberCfg,
        robot,          # isaaclab Articulation 对象
        num_envs: int,
        num_transitions_per_env: int,
        device: str = "cuda",
    ):
        self.cfg = cfg
        self.robot = robot
        self.num_envs = num_envs
        self.device = device

        # --- 网络 ---
        self.actor = DisturberActor(
            obs_dim=cfg.disturber_obs_dim,
            hidden_dims=cfg.actor_hidden_dims,
            force_dim=cfg.force_dim,
        ).to(device)

        self.critic = DisturberCritic(
            obs_dim=cfg.disturber_obs_dim,
            hidden_dims=cfg.critic_hidden_dims,
        ).to(device)

        # --- 优化器 ---
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.disturber_lr,
        )

        # --- Lagrangian 乘子 λ（标量，可学习）---
        # 初始化为正值，通过梯度下降更新
        # 使用 log(λ) 参数化保证 λ > 0
        self._log_lambda = torch.tensor(
            [float(cfg.lambda_init)],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.lambda_optimizer = optim.Adam([self._log_lambda], lr=cfg.lagrangian_lr)

        # --- Rollout Storage ---
        self.storage = DisturberRolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            disturber_obs_dim=cfg.disturber_obs_dim,
            force_dim=cfg.force_dim,
            device=device,
        )

        # --- 当前 step 的中间变量（在 step() 和 record() 之间传递）---
        self._current_forces: Optional[torch.Tensor] = None
        self._current_log_prob: Optional[torch.Tensor] = None
        self._current_mu: Optional[torch.Tensor] = None
        self._current_sigma: Optional[torch.Tensor] = None

        # --- 训练迭代计数 ---
        self.iteration = 0

        # --- 找到施力 body 的索引（在 __init__ 时缓存，避免每步查找）---
        self._body_idx = self._resolve_body_idx()

    def _resolve_body_idx(self) -> int:
        """解析施力 body 在 Articulation 中的索引。"""
        body_names = self.robot.data.body_names
        for i, name in enumerate(body_names):
            if self.cfg.apply_body_name in name:
                return i
        # fallback: 使用 root body (index 0)
        print(
            f"[HInfDisturber] Warning: body '{self.cfg.apply_body_name}' not found, "
            f"using body index 0 (root)."
        )
        return 0

    @property
    def lambda_val(self) -> torch.Tensor:
        """当前 Lagrangian 乘子值（clamp 到合法范围）。"""
        return self._log_lambda.clamp(
            min=self.cfg.lambda_min,
            max=self.cfg.lambda_max,
        )

    def is_active(self) -> bool:
        """是否已过 warmup 阶段，开始启用 Disturber。"""
        return self.iteration >= self.cfg.warmup_iterations

    # -----------------------------------------------------------------------
    # 5.1 Rollout 阶段接口
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def step(self, disturber_obs: torch.Tensor) -> torch.Tensor:
        """
        根据当前物理状态生成对抗力并注入环境。

        [CRITICAL] 梯度阻断：
          disturber_obs 必须是从环境直接获取的纯物理状态，
          绝对不能包含 DreamWaQ VAE 的潜空间特征（code/code_vel）。
          此处额外调用 .detach() 作为双重保险，确保 Disturber 的计算
          图与 DreamWaQ VAE 编码器完全隔离。

        Args:
            disturber_obs: [N, obs_dim]  纯物理状态（不含 VAE 特征）

        Returns:
            forces: [N, 3]  对抗力向量（已注入环境）
        """
        # [CRITICAL] 梯度阻断：切断与 DreamWaQ 计算图的任何连接
        obs_detached = disturber_obs.detach()

        if not self.is_active():
            # warmup 阶段：施加零力，不干扰 Actor 学习
            forces = torch.zeros(
                self.num_envs, self.cfg.force_dim, device=self.device
            )
            self._current_forces = forces
            self._current_log_prob = torch.zeros(self.num_envs, device=self.device)
            self._current_mu = forces.clone()
            self._current_sigma = torch.ones_like(forces) * 1e-6
            self._apply_forces_to_env(forces)
            return forces

        # --- 生成对抗力（使用高斯分布采样，类似 PPO Actor）---
        # 均值由网络输出，标准差固定（或可学习）
        mu = self.actor(obs_detached, self.cfg.eta)  # [N, 3]，已做 L2 截断

        # 使用固定标准差
        sigma = torch.ones_like(mu) * (self.cfg.eta * 0.1)  # 10% of eta

        # 从高斯分布采样（网络驱动的确定性力）
        dist = torch.distributions.Normal(mu, sigma)
        forces_raw = dist.sample()  # [N, 3]

        # 再次做 L2 截断（采样后可能超出范围）
        norms_raw = forces_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [N, 1]
        scale_raw = (self.cfg.eta / norms_raw).clamp(max=1.0)
        forces_net = forces_raw * scale_raw  # [N, 3]，严格满足 ||f|| ≤ η

        # ---------------------------------------------------------------
        # [H-Infinity Fix] ε-Greedy 探索，打破 Disturber 方向性模式坍缩
        # ---------------------------------------------------------------
        eps = getattr(self.cfg, "epsilon_greedy", 0.2)  # 探索概率，默认 0.2

        # Step 1: 生成完全随机的对抗力（3D 球面均匀采样）
        # 先采样各向同性高斯噪声，再归一化为单位向量，保证方向在球面均匀分布
        rand_dir = torch.randn(self.num_envs, self.cfg.force_dim, device=self.device)  # [N, 3]
        rand_dir_norm = rand_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)            # [N, 1]
        rand_dir_unit = rand_dir / rand_dir_norm                                        # [N, 3]，球面均匀单位向量

        # Step 2: 随机力的大小在 [0.5η, η] 之间均匀采样，确保冲击足够致命
        # uniform_() 是 in-place 操作，直接在 GPU 上生成，形状 [N, 1]，broadcast 到 [N, 3]
        rand_magnitude = torch.zeros(self.num_envs, 1, device=self.device).uniform_(
            0.5 * self.cfg.eta, self.cfg.eta
        )  # [N, 1]
        forces_random = rand_dir_unit * rand_magnitude  # [N, 3]，满足 0.5η ≤ ||f|| ≤ η

        # Step 3: 构造 ε-greedy 掩码
        # explore_mask: [N, 1]，dtype=bool
        # True 的位置（概率 ε）→ 使用随机力（探索）
        # False 的位置（概率 1-ε）→ 使用网络力（利用）
        # 每个 env 独立 Bernoulli 采样，纯 GPU Tensor 操作，无 for 循环
        explore_mask = (
            torch.rand(self.num_envs, 1, device=self.device) < eps
        )  # [N, 1]，bool

        # Step 4: 混合 — torch.where 按掩码逐 env 选择最终施加的力
        # explore_mask.expand_as(forces_net) 将 [N, 1] 广播到 [N, 3]，对 xyz 三个分量同步选择
        forces_final = torch.where(
            explore_mask.expand_as(forces_net),  # [N, 3]，广播后的布尔掩码
            forces_random,                        # 探索分支：随机方向 + 随机大小
            forces_net,                           # 利用分支：网络输出的确定性力
        )  # [N, 3]

        # ---------------------------------------------------------------
        # [CRITICAL] log_prob 必须基于 forces_final（最终施加的力）计算
        # 而非 forces_raw，保证 PPO 更新的无偏性：
        #   ratio = π_new(forces_final|s) / π_old(forces_final|s)
        # 对随机探索的 20% 样本，log_prob 反映网络在该随机方向的概率密度，
        # 使 Disturber 能从随机探索的高收益样本中学习，逐步覆盖新方向。
        # ---------------------------------------------------------------
        log_prob = dist.log_prob(forces_final).sum(dim=-1)  # [N]，对 xyz 三维求和

        # 缓存供 record() 使用
        self._current_forces = forces_final
        self._current_log_prob = log_prob
        self._current_mu = mu
        self._current_sigma = sigma

        # 注入环境（施加最终混合力）
        self._apply_forces_to_env(forces_final)

        return forces_final

    def _apply_forces_to_env(self, forces: torch.Tensor):
        """
        使用 IsaacLab API 将外力注入所有环境（纯 GPU Tensor 操作）。

        API: Articulation.set_external_force_and_torque()
        参数说明:
          forces:  [N, num_bodies, 3]  各 body 受到的外力（世界坐标系）
          torques: [N, num_bodies, 3]  各 body 受到的外力矩（世界坐标系）
          body_ids: list[int]          指定施力的 body 索引列表
          env_ids:  Tensor             指定施力的环境索引

        注意: 必须在每个 physics step 前调用，力在下一个 step 生效。
        """
        N = self.num_envs

        # 构造 [N, 1, 3] 的力张量（只对单个 body 施力）
        # 纯 GPU Tensor 操作，无 for 循环
        forces_3d = forces.unsqueeze(1)                          # [N, 1, 3]
        torques_3d = torch.zeros_like(forces_3d)                 # [N, 1, 3]，无力矩

        # 所有环境的索引: [N]
        env_ids = torch.arange(N, device=self.device)

        # IsaacLab API: set_external_force_and_torque
        # body_ids 指定施力的 body（torso_link 的索引）
        self.robot.set_external_force_and_torque(
            forces=forces_3d,                    # [N, 1, 3]
            torques=torques_3d,                  # [N, 1, 3]
            body_ids=[self._body_idx],           # 施力 body 索引列表
            env_ids=env_ids,                     # 所有环境
        )

    def record(
        self,
        disturber_obs: torch.Tensor,   # [N, obs_dim]  当前物理状态
        costs: torch.Tensor,           # [N]           C_t（仅 tracking error）
        dones: torch.Tensor,           # [N]           episode 终止标志
    ):
        """
        记录当前 step 的数据到 Rollout Storage。

        必须在 step() 之后、env.step() 之后调用。

        Args:
            disturber_obs: [N, obs_dim]  当前物理状态（.detach() 保证梯度隔离）
            costs:         [N]           C_t = tracking error（不含辅助奖励）
            dones:         [N]           episode 终止标志
        """
        # [CRITICAL] 梯度阻断
        obs_detached = disturber_obs.detach()

        # 计算 cost value 估计（用于 GAE）
        with torch.no_grad():
            cost_values = self.critic(obs_detached)  # [N, 1]

        self.storage.add(
            disturber_obs=obs_detached,
            forces=self._current_forces,
            costs=costs,
            dones=dones,
            cost_values=cost_values,
            actions_log_prob=self._current_log_prob,
            action_mean=self._current_mu,
            action_sigma=self._current_sigma,
        )

    # -----------------------------------------------------------------------
    # 5.2 Cost 计算接口（仅 tracking error，严格按论文定义）
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_cost(
        cmd_vel: torch.Tensor,          # [N, 3]  速度命令 [vx, vy, ωz]
        root_lin_vel_b: torch.Tensor,   # [N, 3]  机器人线速度（body frame）
        root_ang_vel_w: torch.Tensor,   # [N, 3]  机器人角速度（world frame）
        lin_vel_weight: float = 1.0,
        ang_vel_weight: float = 0.5,
    ) -> torch.Tensor:
        """
        计算 H-Infinity Cost C_t。

        论文定义：C_t 仅包含任务追踪误差，绝不包含扭矩、平滑度等辅助奖励。
        C_t = w_lin * ||v_cmd_xy - v_actual_xy||^2
            + w_ang * ||ω_cmd_z - ω_actual_z||^2

        Args:
            cmd_vel:        [N, 3]  速度命令 [vx, vy, ωz]
            root_lin_vel_b: [N, 3]  机器人线速度（body frame，与命令同坐标系）
            root_ang_vel_w: [N, 3]  机器人角速度（world frame）
            lin_vel_weight: 线速度误差权重
            ang_vel_weight: 角速度误差权重

        Returns:
            cost: [N]  每个环境的 C_t（标量，越大表示追踪越差）
        """
        # 线速度追踪误差（xy 平面）: [N]
        lin_vel_error = torch.sum(
            torch.square(cmd_vel[:, :2] - root_lin_vel_b[:, :2]),
            dim=-1,
        )

        # 角速度追踪误差（z 轴）: [N]
        ang_vel_error = torch.square(cmd_vel[:, 2] - root_ang_vel_w[:, 2])

        # C_t（越大表示追踪越差，Disturber 希望最大化）
        cost = lin_vel_weight * lin_vel_error + ang_vel_weight * ang_vel_error

        return cost  # [N]

    # -----------------------------------------------------------------------
    # 5.3 Update 阶段：双梯度下降（论文公式 11 & 12）
    # -----------------------------------------------------------------------

    def compute_returns(self, last_disturber_obs: torch.Tensor):
        """
        计算 Disturber 的 cost returns 和 advantages（GAE）。

        在 rollout 结束后、update() 之前调用。

        Args:
            last_disturber_obs: [N, obs_dim]  最后一步的物理状态
        """
        # [CRITICAL] 梯度阻断
        obs_detached = last_disturber_obs.detach()
        with torch.no_grad():
            last_cost_values = self.critic(obs_detached)  # [N, 1]

        self.storage.compute_returns(
            last_cost_values=last_cost_values,
            gamma=self.cfg.gamma,
            lam=self.cfg.lam,
        )

    def update(self) -> dict:
        """
        执行双梯度下降更新（论文公式 12）。

        更新顺序（严格按论文）:
          Step 1: 更新 Disturber（最大化 cost，受 η 约束）
          Step 2: 更新 Lagrangian 乘子 λ（调整约束松紧度）

        注意: Actor (PPO) 的更新在 DWAQPPO.update() 中进行，
        本函数只负责 Disturber 和 λ 的更新。

        Returns:
            dict: 包含各损失值的字典（用于 TensorBoard 日志）
        """
        if not self.is_active():
            self.storage.clear()  # [Fix] warmup 期间也必须清空 storage，否则下次 rollout 会 overflow
            self.iteration += 1
            return {
                "disturber/actor_loss": 0.0,
                "disturber/critic_loss": 0.0,
                "disturber/lambda": self.lambda_val.item(),
                "disturber/mean_cost": 0.0,
                "disturber/mean_force_norm": 0.0,
            }

        mean_actor_loss = 0.0
        mean_critic_loss = 0.0
        mean_cost = 0.0
        mean_force_norm = 0.0
        num_updates = 0

        for batch in self.storage.mini_batch_generator(
            self.cfg.num_mini_batches, self.cfg.num_learning_epochs
        ):
            (
                obs_b,       # [B, obs_dim]  纯物理状态
                forces_b,    # [B, 3]        旧的对抗力
                costs_b,     # [B, 1]        C_t
                values_b,    # [B, 1]        旧的 cost value 估计
                adv_b,       # [B, 1]        cost advantages（已归一化）
                ret_b,       # [B, 1]        cost returns
                old_logp_b,  # [B, 1]        旧的 log prob
                old_mu_b,    # [B, 3]        旧的均值
                old_sigma_b, # [B, 3]        旧的标准差
            ) = batch

            # [CRITICAL] 梯度阻断：obs_b 已在 storage 中 detach，此处再次确认
            obs_b = obs_b.detach()

            # ---- Step 1a: 重新计算当前 Disturber 的 log prob ----
            # 重新前向传播（需要梯度，用于 Disturber 更新）
            mu_new = self.actor(obs_b, self.cfg.eta)          # [B, 3]
            sigma_new = torch.ones_like(mu_new) * (self.cfg.eta * 0.1)
            dist_new = torch.distributions.Normal(mu_new, sigma_new)
            new_logp = dist_new.log_prob(forces_b).sum(dim=-1, keepdim=True)  # [B, 1]

            # ---- Step 1b: Disturber PPO Loss（论文公式 12，最大化 cost）----
            # ratio = π_new(d|s) / π_old(d|s)
            ratio = torch.exp(new_logp - old_logp_b)  # [B, 1]

            # Disturber 目标：最大化 cost advantages
            # 等价于最小化 -advantages * ratio（PPO clipping）
            # adv_b 是 cost advantages，Disturber 希望最大化，所以取负号
            disturber_surrogate = -adv_b * ratio                          # [B, 1]
            disturber_surrogate_clipped = -adv_b * torch.clamp(
                ratio,
                1.0 - self.cfg.clip_param,
                1.0 + self.cfg.clip_param,
            )                                                              # [B, 1]
            # 取两者中较小的（PPO 保守更新）
            # 注意：这里 min 对应"最大化 cost 的保守更新"
            disturber_actor_loss = torch.max(
                disturber_surrogate, disturber_surrogate_clipped
            ).mean()

            # ---- Step 1c: Disturber Critic Loss（MSE）----
            cost_value_new = self.critic(obs_b)  # [B, 1]
            disturber_critic_loss = (ret_b - cost_value_new).pow(2).mean()

            # ---- Step 1d: Lagrangian 约束项（论文公式 11）----
            # 约束: E[||d(s_t)||] ≤ η
            # 违约量: E[||d||] - η（正值表示违约）
            # 当前 batch 的平均力范数
            force_norms = forces_b.norm(dim=-1, keepdim=True)  # [B, 1]
            constraint_violation = force_norms.mean() - self.cfg.eta

            # Lagrangian 松弛：L = L_disturber + λ * constraint_violation
            # λ 当前值（detach，不让 λ 的梯度影响 Disturber 网络）
            lam = self.lambda_val.detach()
            total_disturber_loss = (
                disturber_actor_loss
                + self.cfg.value_loss_coef * disturber_critic_loss
                + lam * constraint_violation
            )

            # ---- Step 1e: 更新 Disturber 网络 ----
            self.optimizer.zero_grad()
            total_disturber_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.cfg.max_grad_norm,
            )
            self.optimizer.step()

            # ---- Step 2: 更新 Lagrangian 乘子 λ（论文公式 12）----
            # λ 的更新方向：若约束被违反（force_norm > η），增大 λ；反之减小
            # 梯度下降：λ ← λ - α * ∂L/∂λ = λ - α * (-constraint_violation)
            # 即：λ ← λ + α * constraint_violation
            # 使用 Adam 优化器自动处理学习率
            lambda_loss = -self.lambda_val * constraint_violation.detach()
            self.lambda_optimizer.zero_grad()
            lambda_loss.backward()
            self.lambda_optimizer.step()

            # 统计
            mean_actor_loss += disturber_actor_loss.item()
            mean_critic_loss += disturber_critic_loss.item()
            mean_cost += costs_b.mean().item()
            mean_force_norm += force_norms.mean().item()
            num_updates += 1

        # 清空 storage
        self.storage.clear()
        self.iteration += 1

        n = max(num_updates, 1)
        return {
            "disturber/actor_loss": mean_actor_loss / n,
            "disturber/critic_loss": mean_critic_loss / n,
            "disturber/lambda": self.lambda_val.item(),
            "disturber/mean_cost": mean_cost / n,
            "disturber/mean_force_norm": mean_force_norm / n,
        }

    # -----------------------------------------------------------------------
    # 5.4 工具方法
    # -----------------------------------------------------------------------

    def state_dict(self) -> dict:
        """保存 Disturber 状态（用于 checkpoint）。"""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "log_lambda": self._log_lambda.data,
            "lambda_optimizer": self.lambda_optimizer.state_dict(),
            "iteration": self.iteration,
        }

    def load_state_dict(self, state: dict):
        """加载 Disturber 状态。"""
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._log_lambda.data.copy_(state["log_lambda"])
        self.lambda_optimizer.load_state_dict(state["lambda_optimizer"])
        self.iteration = state["iteration"]

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
