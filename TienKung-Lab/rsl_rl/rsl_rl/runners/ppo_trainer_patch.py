"""
PPO Trainer Patch：将 HInfDisturberPlugin 接入 DWAQOnPolicyRunner。

本文件展示如何最小化修改现有 DWAQOnPolicyRunner，
以集成 H-Infinity 双梯度对抗训练。

修改策略：
  - 不修改 DWAQPPO.update() 内部逻辑（保持原有 VAE 训练不变）
  - 在 Runner 的 learn() 循环中插入 Disturber 的 step/record/update
  - 梯度完全隔离：Disturber 使用独立 optimizer，不共享 policy 参数

使用方式：
  将本文件中的 DWAQHInfRunner 替换 DWAQOnPolicyRunner 即可，
  或直接参考 learn() 中的注释，手动 patch 现有 runner。

论文公式对应：
  - 公式(11): Actor PPO Loss + λ * Lagrangian_constraint
  - 公式(12): 交替更新顺序 → π (DWAQPPO.update) → d (plugin.update) → λ (plugin.update)
"""

from __future__ import annotations

import os
import time
import statistics
from collections import deque

import torch

from rsl_rl.algorithms import DWAQPPO
from rsl_rl.algorithms.hinf_disturber import HInfDisturberPlugin, HInfDisturberCfg
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic_DWAQ, EmpiricalNormalization
from rsl_rl.runners.dwaq_on_policy_runner import DWAQOnPolicyRunner


class DWAQHInfRunner(DWAQOnPolicyRunner):
    """
    继承 DWAQOnPolicyRunner，集成 H-Infinity Disturber。

    新增内容：
      1. __init__: 初始化 HInfDisturberPlugin
      2. learn():  在 rollout 循环中注入外力，在 update 阶段执行双梯度下降
      3. save/load: 保存/加载 Disturber checkpoint

    其余逻辑（VAE 训练、PPO 更新、日志）完全继承自父类，不做修改。
    """

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ):
        # 先调用父类初始化（完整的 DWAQ PPO 设置）
        super().__init__(env, train_cfg, log_dir, device)

        # ---- 初始化 Disturber ----
        disturber_cfg_dict = train_cfg.get("disturber", {})
        disturber_cfg = HInfDisturberCfg(**disturber_cfg_dict)

        # 从环境获取 robot Articulation 对象
        # G1DwaqEnv 中 robot 是 self.env.robot
        robot = self.env.robot

        self.disturber = HInfDisturberPlugin(
            cfg=disturber_cfg,
            robot=robot,
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            device=self.device,
        )

        # 缓存 disturber_cfg 供 learn() 使用
        self._disturber_cfg = disturber_cfg

    def _extract_disturber_obs(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        从环境观测中提取纯物理状态，用于 Disturber 输入。

        [CRITICAL] 绝对不能包含 DreamWaQ VAE 潜空间特征。
        obs 是 actor_obs（纯物理状态），不含 latent code，可直接使用。

        G1 DWAQ actor_obs 结构（开启 gait_phase 时 82 维）:
          [0:3]   ang_vel (3)
          [3:6]   projected_gravity (3)
          [6:9]   command (3)
          [9:32]  joint_pos (23)
          [32:55] joint_vel (23)
          [55:78] action (23)
          [78:82] gait_phase sin/cos (4)  ← 若开启

        Disturber 使用完整 actor_obs（不含 VAE latent），
        这与论文要求一致：Disturber 只能观测当前物理状态。

        Args:
            obs:        [N, num_obs]           actor 观测（纯物理状态）
            critic_obs: [N, num_privileged_obs] 特权观测（含速度，供 cost 计算）

        Returns:
            disturber_obs: [N, disturber_obs_dim]  纯物理状态（已 detach）
        """
        # [CRITICAL] .detach() 切断与 DreamWaQ 计算图的连接
        # obs 本身不含 VAE 特征，但仍需 detach 防止意外梯度流
        disturber_obs = obs.detach()

        # 若 Disturber obs_dim 与 actor obs_dim 不一致，截取前 disturber_obs_dim 维
        # （例如 Disturber 不需要 gait_phase 信息）
        expected_dim = self._disturber_cfg.disturber_obs_dim
        if disturber_obs.shape[-1] > expected_dim:
            disturber_obs = disturber_obs[:, :expected_dim]
        elif disturber_obs.shape[-1] < expected_dim:
            # 用零填充（不应发生，配置时需确保维度匹配）
            pad = torch.zeros(
                disturber_obs.shape[0],
                expected_dim - disturber_obs.shape[-1],
                device=self.device,
            )
            disturber_obs = torch.cat([disturber_obs, pad], dim=-1)

        return disturber_obs  # [N, disturber_obs_dim]，已 detach

    def _compute_cost_from_env(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        从环境状态计算 C_t（仅 tracking error，严格按论文定义）。

        C_t = w_lin * ||v_cmd_xy - v_actual_xy||^2
            + w_ang * ||ω_cmd_z - ω_actual_z||^2

        数据来源：
          - 速度命令: env.command_generator.command[:, :3]  [vx, vy, ωz]
          - 实际线速度: robot.data.root_lin_vel_b  (body frame，与命令同坐标系)
          - 实际角速度: robot.data.root_ang_vel_w  (world frame)

        [CRITICAL] 不包含扭矩、平滑度、接触等辅助奖励项。

        Args:
            obs:        [N, num_obs]           actor 观测（含 ang_vel 等）
            critic_obs: [N, num_privileged_obs] 特权观测（含真实速度）

        Returns:
            cost: [N]  每个环境的 C_t
        """
        # 从环境直接获取（比从 obs 解析更准确）
        cmd_vel = self.env.command_generator.command[:, :3].detach()  # [N, 3]
        root_lin_vel_b = self.env.robot.data.root_lin_vel_b.detach()  # [N, 3]
        root_ang_vel_w = self.env.robot.data.root_ang_vel_w.detach()  # [N, 3]

        cost = HInfDisturberPlugin.compute_cost(
            cmd_vel=cmd_vel,
            root_lin_vel_b=root_lin_vel_b,
            root_ang_vel_w=root_ang_vel_w,
            lin_vel_weight=self._disturber_cfg.cost_lin_vel_weight,
            ang_vel_weight=self._disturber_cfg.cost_ang_vel_weight,
        )  # [N]

        return cost

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """
        H-Infinity 双梯度对抗训练主循环。

        相比父类 learn()，新增以下步骤：
          Rollout 阶段（每个 physics step）:
            1. [新增] 提取纯物理状态 disturber_obs（不含 VAE 特征）
            2. [新增] disturber.step(disturber_obs) → 生成并注入对抗力
            3. 原有: alg.act() → env.step() → alg.process_env_step()
            4. [新增] 计算 C_t（仅 tracking error）
            5. [新增] disturber.record(disturber_obs, cost, done)

          Update 阶段（每个 iteration）:
            6. 原有: alg.compute_returns() → alg.update()  ← Actor PPO 更新
            7. [新增] disturber.compute_returns()           ← Disturber GAE
            8. [新增] disturber.update()                    ← Disturber + λ 更新

        论文公式(12) 更新顺序：π → d → λ（步骤 6 → 8 严格对应）
        """
        # ---- 初始化（复用父类逻辑）----
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self._init_logger()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, obs_hist = self.env.get_observations()
        privileged_obs, prev_critic_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs

        obs = obs.to(self.device)
        critic_obs = critic_obs.to(self.device)
        prev_critic_obs = prev_critic_obs.to(self.device)
        obs_hist = obs_hist.to(self.device)

        self.train_mode()
        self.disturber.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            # ================================================================
            # Rollout 阶段
            # ================================================================
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):

                    # [新增 Step 1] 提取纯物理状态（不含 VAE 特征）
                    # obs 是 actor_obs，本身不含 latent code，可直接使用
                    # .detach() 在 _extract_disturber_obs 内部已处理
                    disturber_obs = self._extract_disturber_obs(obs, critic_obs)
                    # disturber_obs: [N, disturber_obs_dim]，已 detach

                    # [新增 Step 2] Disturber 生成并注入对抗力
                    # 注意：必须在 alg.act() 之前调用，确保外力在本 step 生效
                    # step() 内部调用 robot.set_external_force_and_torque()
                    self.disturber.step(disturber_obs)

                    # [原有 Step 3a] Actor 采样动作
                    actions = self.alg.act(obs, critic_obs, prev_critic_obs, obs_hist)

                    # [原有 Step 3b] 环境 step（此时对抗力已注入，会影响物理仿真）
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                    obs_dict = extras.get("observations", {})
                    privileged_obs = obs_dict.get("critic", None)
                    obs_hist = obs_dict.get("obs_hist", obs_hist)
                    prev_critic_obs = obs_dict.get("prev_critic_obs", prev_critic_obs)
                    critic_obs = privileged_obs if privileged_obs is not None else obs

                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    prev_critic_obs = prev_critic_obs.to(self.device)
                    obs_hist = obs_hist.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # [原有 Step 3c] PPO 记录 transition
                    self.alg.process_env_step(rewards, dones, extras)

                    # [新增 Step 4] 计算 C_t（仅 tracking error，不含辅助奖励）
                    # 必须在 env.step() 之后调用（获取最新物理状态）
                    cost = self._compute_cost_from_env(obs, critic_obs)  # [N]

                    # [新增 Step 5] Disturber 记录数据
                    # 使用 env.step() 后的新 obs 作为 disturber_obs（s_{t+1}）
                    # 注意：这里传入的是 step 后的新状态，用于 GAE bootstrap
                    new_disturber_obs = self._extract_disturber_obs(obs, critic_obs)
                    self.disturber.record(
                        disturber_obs=new_disturber_obs,
                        costs=cost,
                        dones=dones,
                    )

                    # 日志
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start
            start = stop

            # ================================================================
            # Update 阶段（双梯度下降，严格按论文公式 12 的顺序）
            # ================================================================

            # [Step 6] Actor PPO 更新（论文公式 12 第一步：更新 π）
            # 包含：surrogate loss + value loss + VAE autoencoder loss
            # 这一步完全不涉及 Disturber，梯度不会流向 Disturber 网络
            self.alg.compute_returns(critic_obs.clone())
            loss_dict = self.alg.update()

            # [Step 7] Disturber GAE 计算
            last_disturber_obs = self._extract_disturber_obs(critic_obs, critic_obs)
            self.disturber.compute_returns(last_disturber_obs)

            # [Step 8] Disturber + λ 更新（论文公式 12 第二步：更新 d 和 λ）
            # 内部顺序：先更新 Disturber 网络，再更新 λ
            disturber_loss_dict = self.disturber.update()

            stop = time.time()
            learn_time = stop - start

            self.current_learning_iteration = it

            # ================================================================
            # 日志
            # ================================================================
            if self.log_dir is not None and not self.disable_logs:
                self._log_hinf(locals(), loss_dict, disturber_loss_dict)
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

            if it == start_iter and not self.disable_logs:
                from rsl_rl.utils import store_code_state
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _log_hinf(self, locs: dict, loss_dict: dict, disturber_loss_dict: dict):
        """记录 H-Infinity 训练日志。

        父类 log() 需要 locs 中包含 mean_value_loss / mean_surrogate_loss / mean_autoenc_loss，
        这些值来自 loss_dict，需要在调用前注入到 locs 中。
        """
        # 将 loss_dict 的值注入 locs，供父类 log() 使用
        locs["mean_value_loss"] = loss_dict["value_function"]
        locs["mean_surrogate_loss"] = loss_dict["surrogate"]
        locs["mean_autoenc_loss"] = loss_dict["autoencoder"]

        # 调用父类日志（记录 PPO 损失、奖励、FPS 等）
        self.log(locs)

        # 额外记录 Disturber 指标到 TensorBoard
        it = locs["it"]
        for key, val in disturber_loss_dict.items():
            self.writer.add_scalar(f"HInf/{key}", val, it)

    def save(self, path: str, infos=None):
        """保存 checkpoint，包含 Disturber 状态。"""
        # 先调用父类 save（保存 PPO policy）
        super().save(path, infos)

        # 额外保存 Disturber 状态
        disturber_path = path.replace(".pt", "_disturber.pt")
        torch.save(self.disturber.state_dict(), disturber_path)

    def load(self, path: str, load_optimizer: bool = True):
        """加载 checkpoint，包含 Disturber 状态（若存在）。"""
        infos = super().load(path, load_optimizer)

        # 尝试加载 Disturber 状态
        disturber_path = path.replace(".pt", "_disturber.pt")
        if os.path.exists(disturber_path):
            state = torch.load(disturber_path, weights_only=False)
            self.disturber.load_state_dict(state)
            print(f"[HInfRunner] Loaded disturber from {disturber_path}")
        else:
            print(f"[HInfRunner] No disturber checkpoint found at {disturber_path}, starting fresh.")

        return infos

    def train_mode(self):
        super().train_mode()
        # 保护：父类 __init__ 内部会调用 train_mode()，此时 disturber 尚未初始化
        if hasattr(self, "disturber"):
            self.disturber.train()

    def eval_mode(self):
        super().eval_mode()
        if hasattr(self, "disturber"):
            self.disturber.eval()
