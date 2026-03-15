# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""
X2Ultra 环境配置

X2Ultra 关节/Link 命名规范（URDF 原始名称）：
  脚部 link:  left_ankle_roll_link, right_ankle_roll_link
  躯干 link:  torso_link
  高度扫描参考 link: torso_link

观测维度说明（actor_obs_history_length=1, gait_phase 启用）：
  单帧 Actor 观测 = 3(ang_vel) + 3(gravity) + 4(cmd) + 31(jpos) + 31(jvel) + 31(act)
                  + 2(sin_phase) + 2(cos_phase) = 107 维
  Actor 输入 = 107 × 1 = 107 维

  单帧 Critic 观测 = 107(actor_obs) + 3(lin_vel) + 2(feet_contact)
                   + 6(feet_pos) + 6(feet_vel) + 6(feet_force) + 1(root_height) = 131 维
  Critic 输入 = 131 × 1 + 187(height_scan) = 318 维
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.X2Ultra import X2ULTRA_CFG
from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


# ===========================================================================
# 奖励配置
# ===========================================================================

@configclass
class X2UltraRewardCfg(RewardCfg):
    """X2Ultra 标准行走奖励配置。

    所有 body_names / joint_names 的正则表达式均严格对应
    X2Ultra URDF 中的实际命名。
    """

    # ---------- 速度追踪 ----------
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5}
    )

    # ---------- 运动质量惩罚 ----------
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # ---------- 接触惩罚 ----------
    # 除踝关节 roll link 以外的所有 body 不应接触地面
    # X2Ultra 脚部末端为 ankle_roll_link
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names="(?!.*ankle_roll.*).*",  # 排除 ankle_roll_link
            ),
            "threshold": 1.0,
        },
    )

    # 两脚同时离地惩罚（飞行惩罚）
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
    # 躯干朝向惩罚：torso_link 应保持竖直
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        weight=-2.0,
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # ---------- 终止惩罚 ----------
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # ---------- 脚部奖励 ----------
    # 脚部腾空时间奖励（鼓励抬腿行走）
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

    # 脚部滑动惩罚（接触时不应有速度）
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

    # 脚部冲击力惩罚
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

    # 两脚距离过近惩罚（防止交叉步）
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

    # 脚部绊倒惩罚
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=[".*ankle_roll_link"]
            )
        },
    )

    # ---------- 关节限位惩罚 ----------
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

    # ---------- 关节偏差惩罚 ----------
    # 髋关节 yaw/roll 偏差（防止外八字步态）
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"]
            )
        },
    )

    # 手臂/腰部关节偏差（保持自然姿态）
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

    # 腿部关节偏差（hip_pitch, knee, ankle）
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle.*"],
            )
        },
    )


# ===========================================================================
# 平地环境配置
# ===========================================================================

@configclass
class X2UltraFlatEnvCfg(BaseEnvCfg):
    """X2Ultra 平地训练环境配置。"""

    reward = X2UltraRewardCfg()

    def __post_init__(self):
        super().__post_init__()

        # 机器人资产
        self.scene.robot = X2ULTRA_CFG

        # 地形：平地（砾石地形）
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG

        # 高度扫描参考 body（X2Ultra 使用 torso_link）
        self.scene.height_scanner.prim_body_name = "torso_link"

        # 终止接触检测：躯干 link 接触地面则终止
        self.robot.terminate_contacts_body_names = ["torso_link"]

        # 脚部接触检测：左右 ankle_roll_link
        # 顺序：[左脚, 右脚]，与步态相位 leg_phase 顺序一致
        self.robot.feet_body_names = [".*ankle_roll_link"]

        # 域随机化：质量扰动施加在 torso_link
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [
            "torso_link"
        ]


@configclass
class X2UltraFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "x2ultra_flat"
    wandb_project: str = "x2ultra_flat"


# ===========================================================================
# 崎岖地形环境配置
# ===========================================================================

@configclass
class X2UltraRoughEnvCfg(X2UltraFlatEnvCfg):
    """X2Ultra 崎岖地形训练环境配置（非对称 Actor-Critic）。

    Actor：盲行（无高度扫描），可直接部署到真实机器人。
    Critic：具有地形高度扫描特权信息，仅用于训练。
    """

    def __post_init__(self):
        super().__post_init__()

        # 崎岖地形
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG

        # 高度扫描：启用，但仅 Critic 可见（非对称 AC）
        self.scene.height_scanner.enable_height_scan = True
        self.scene.height_scanner.critic_only = True

        # Critic 特权信息
        self.scene.privileged_info.enable_feet_info = True          # 脚部位置+速度 (12维)
        self.scene.privileged_info.enable_feet_contact_force = True  # 接触力 (6维)
        self.scene.privileged_info.enable_root_height = True         # 根部高度 (1维)

        # 观测历史长度（10帧提供时序上下文）
        self.robot.actor_obs_history_length = 10
        self.robot.critic_obs_history_length = 10

        # 动作延迟（sim-to-real 迁移）
        self.domain_rand.action_delay.enable = True

        # 崎岖地形奖励权重调整
        self.reward.feet_air_time.weight = 0.25
        self.reward.track_lin_vel_xy_exp.weight = 1.5
        self.reward.track_ang_vel_z_exp.weight = 1.5
        self.reward.lin_vel_z_l2.weight = -0.25


@configclass
class X2UltraRoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "x2ultra_rough"
    wandb_project: str = "x2ultra_rough"

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCritic"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
