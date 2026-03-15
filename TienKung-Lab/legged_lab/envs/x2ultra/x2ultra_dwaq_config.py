# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""
X2Ultra DreamWaQ 环境配置

DreamWaQ (Deep Variational Autoencoder for Walking) 盲行配置。

架构说明：
  - Actor:   当前观测 + VAE 编码器输出的隐变量 → 动作
  - Encoder: 观测历史 (5帧) → 速度估计 (3维) + 隐变量 (16维)
  - Decoder: 隐变量 → 重建观测（VAE 损失）
  - Critic:  特权观测（含高度扫描）→ 价值估计

=============================================================
X2Ultra 观测维度精确计算（DoF = 31）
=============================================================

【单帧 Actor 观测】（启用步态相位）：
  ang_vel:          3
  projected_gravity: 3
  commands:         4  (lin_vel_x, lin_vel_y, ang_vel_z, heading)
  joint_pos:        31 (31 DoF)
  joint_vel:        31 (31 DoF)
  actions:          31 (31 DoF)
  sin(leg_phase):   2  (左腿, 右腿)
  cos(leg_phase):   2  (左腿, 右腿)
  ─────────────────────────────
  单帧合计:         107 维

【Actor 网络输入】：
  actor_obs_history_length = 1
  Actor 输入 = 107 × 1 = 107 维
  + cenet_out_dim = 19 (VAE 编码器输出)
  Actor 实际输入 = 107 + 19 = 126 维

【DWAQ obs_hist（VAE 编码器输入）】：
  dwaq_obs_history_length = 5
  obs_hist = 107 × 5 = 535 维

【单帧 Critic 观测】：
  actor_obs:          107
  root_lin_vel:       3
  feet_contact:       2   (2 脚)
  feet_pos_in_body:   6   (2 脚 × 3)
  feet_vel_in_body:   6   (2 脚 × 3)
  feet_contact_force: 6   (2 脚 × 3)
  root_height:        1
  ─────────────────────────────
  单帧合计:           131 维

【Critic 网络输入】：
  critic_obs_history_length = 1
  基础 Critic 观测 = 131 × 1 = 131 维
  height_scan = 187 维 (1.6m×1.0m @ 0.1m 分辨率 = 17×11 = 187 点)
  Critic 输入 = 131 + 187 = 318 维

=============================================================
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab.envs.mdp import events as isaaclab_events

from legged_lab.assets.X2Ultra import X2ULTRA_CFG
from legged_lab.envs.base.base_env_config import (
    BaseAgentCfg,
    BaseEnvCfg,
    RewardCfg,
)
import legged_lab.mdp as mdp
from legged_lab.terrains import ROUGH_TERRAINS_CFG


# ===========================================================================
# DreamWaQ 奖励配置
# ===========================================================================

@configclass
class X2UltraDwaqRewardCfg(RewardCfg):
    """X2Ultra DreamWaQ 盲行奖励配置。

    在标准奖励基础上增加：
    1. alive 存活奖励（防止过早倒下）
    2. idle_penalty 偷懒惩罚（防止收到移动命令却站立不动）
    3. gait_phase_contact 步态相位接触奖励（鼓励正确的两足步态）
    4. feet_swing_height 摆动腿高度控制

    所有 body_names / joint_names 均严格对应 X2Ultra URDF 命名。
    脚部末端 link：left_ankle_roll_link, right_ankle_roll_link
    """

    # ---------- 速度追踪（增强权重，解决偷懒站立问题）----------
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5}
    )

    # ---------- 运动质量惩罚 ----------
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # ---------- 接触惩罚 ----------
    # 除 ankle_roll_link 以外的所有 body 不应接触地面
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names="(?!.*ankle_roll.*).*",
            ),
            "threshold": 1.0,
        },
    )

    # 两脚同时离地惩罚
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=".*ankle_roll_link"
            ),
            "threshold": 1.0,
        },
    )

    # ---------- 姿态惩罚 ----------
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        weight=-2.0,
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # ---------- 终止惩罚 ----------
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # ---------- 脚部奖励 ----------
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=".*ankle_roll_link"
            ),
            "threshold": 0.4,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=".*ankle_roll_link"
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=".*_ankle_roll_link"
            ),
        },
    )

    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=".*ankle_roll_link"
            ),
            "threshold": 500,
            "max_reward": 400,
        },
    )

    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[".*ankle_roll_link"]
            ),
            "threshold": 0.2,
        },
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=[".*ankle_roll_link"]
            )
        },
    )

    # ---------- 关节限位与偏差惩罚 ----------
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"]
            )
        },
    )

    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
            )
        },
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*waist.*",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_elbow_joint",
                    ".*_wrist.*",
                    "head_yaw_joint",
                    "head_pitch_joint",
                ],
            )
        },
    )

    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_pitch_joint", ".*_knee_joint"],
            )
        },
    )

    # ==========================================================
    # DreamWaQ 核心奖励
    # ==========================================================

    # 存活奖励（鼓励机器人保持站立）
    alive = RewTerm(func=mdp.alive, weight=0.15)

    # 偷懒惩罚：被命令移动但实际静止时惩罚
    # cmd_threshold=0.2: 命令速度 > 0.2 m/s 视为"需要移动"
    # vel_threshold=0.1: 实际速度 < 0.1 m/s 视为"静止"
    idle_penalty = RewTerm(
        func=mdp.idle_when_commanded,
        weight=-2.0,
        params={"cmd_threshold": 0.2, "vel_threshold": 0.1},
    )

    # 步态相位接触奖励（两足交替步态）
    # 注意：body_names 顺序必须是 [左脚, 右脚]，与 leg_phase 顺序一致
    # X2Ultra 左脚：left_ankle_roll_link，右脚：right_ankle_roll_link
    gait_phase_contact = RewTerm(
        func=mdp.gait_phase_contact,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "stance_threshold": 0.55,
        },
    )

    # 摆动腿高度控制（鼓励适当抬腿）
    feet_swing_height = RewTerm(
        func=mdp.feet_swing_height,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=".*ankle_roll_link"
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=".*_ankle_roll_link"
            ),
            "target_height": 0.08,
        },
    )


# ===========================================================================
# DreamWaQ 环境配置
# ===========================================================================

@configclass
class X2UltraDwaqEnvCfg(BaseEnvCfg):
    """X2Ultra DreamWaQ 盲行环境配置。

    关键配置：
    - Actor 盲行（无高度扫描），Critic 有地形特权信息
    - VAE 编码器使用 5 帧观测历史
    - 步态相位启用（2维 sin + 2维 cos = 4维额外观测）
    - 脚部顺序：[左脚(left_ankle_roll_link), 右脚(right_ankle_roll_link)]
    """

    reward = X2UltraDwaqRewardCfg()

    def __post_init__(self):
        super().__post_init__()

        # 机器人资产
        self.scene.robot = X2ULTRA_CFG

        # 地形：崎岖地形（DreamWaQ 训练目标）
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG

        # 高度扫描参考 body
        self.scene.height_scanner.prim_body_name = "torso_link"

        # 终止接触检测：躯干接触地面则终止
        self.robot.terminate_contacts_body_names = ["torso_link"]

        # 脚部接触检测（顺序：[左脚, 右脚]，与 leg_phase 一致）
        self.robot.feet_body_names = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]

        # 域随机化：质量扰动施加在 torso_link
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [
            "torso_link"
        ]

        # 高度扫描：启用，但仅 Critic 可见（Actor 盲行）
        self.scene.height_scanner.enable_height_scan = True
        self.scene.height_scanner.critic_only = True

        # Critic 特权信息
        self.scene.privileged_info.enable_feet_info = True          # 脚部位置+速度 (12维)
        self.scene.privileged_info.enable_feet_contact_force = True  # 接触力 (6维)
        self.scene.privileged_info.enable_root_height = True         # 根部高度 (1维)

        # ==========================================================
        # DWAQ 核心参数
        # ==========================================================

        # VAE 编码器观测历史长度（5帧，与原版 DreamWaQ 一致）
        # obs_hist 维度 = 107 × 5 = 535 维
        self.robot.dwaq_obs_history_length = 5

        # Actor/Critic 历史长度（1帧，编码器负责时序信息）
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1

        # 步态相位配置（两足交替步态）
        # enable=True: 在观测中加入 sin/cos 相位信息（+4维）
        # period=0.8s: 步态周期
        # offset=0.5:  左右腿相位差 50%（交替步态）
        self.robot.gait_phase.enable = True
        self.robot.gait_phase.period = 0.8
        self.robot.gait_phase.offset = 0.5

        # 执行器增益随机化（±20%，模拟电机差异）
        self.domain_rand.events.randomize_actuator_gains = EventTerm(
            func=isaaclab_events.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        )


# ===========================================================================
# DreamWaQ Agent 配置
# ===========================================================================

@configclass
class X2UltraDwaqAgentCfg(BaseAgentCfg):
    """X2Ultra DreamWaQ Agent 配置。

    使用 ActorCritic_DWAQ 策略 + DWAQPPO 算法。

    网络输入维度：
      Actor 输入  = 107 (单帧观测) + 19 (VAE 编码器输出) = 126 维
      Critic 输入 = 318 维 (131 基础特权观测 + 187 高度扫描)
      obs_hist    = 535 维 (107 × 5 帧，VAE 编码器输入)

    VAE 编码器输出：
      cenet_out_dim = 19 = 速度估计(3维) + 隐变量(16维)
    """

    experiment_name: str = "x2ultra_dwaq"
    wandb_project: str = "x2ultra_dwaq"
    runner_class_name: str = "DWAQOnPolicyRunner"

    def __post_init__(self):
        super().__post_init__()

        # 使用 ActorCritic_DWAQ 策略（含 VAE 编码器）
        self.policy.class_name = "ActorCritic_DWAQ"
        self.policy.init_noise_std = 1.0
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

        # DWAQ 编码器输出维度：速度(3) + 隐变量(16) = 19
        self.policy.cenet_out_dim = 19

        # 使用 DWAQPPO 算法（PPO + 自编码器损失）
        self.algorithm.class_name = "DWAQPPO"
        self.algorithm.entropy_coef = 0.01
