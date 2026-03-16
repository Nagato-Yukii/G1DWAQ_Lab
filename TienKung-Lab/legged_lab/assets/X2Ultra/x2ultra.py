# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""
X2Ultra 人形机器人 ArticulationCfg 配置

X2Ultra URDF 关节命名规范（共 31 个可驱动关节）：
  腿部 (12): left/right_hip_pitch_joint, left/right_hip_roll_joint,
             left/right_hip_yaw_joint, left/right_knee_joint,
             left/right_ankle_pitch_joint, left/right_ankle_roll_joint
  腰部 (3):  waist_yaw_joint, waist_pitch_joint, waist_roll_joint
  手臂 (14): left/right_shoulder_pitch_joint, left/right_shoulder_roll_joint,
             left/right_shoulder_yaw_joint, left/right_elbow_joint,
             left/right_wrist_yaw_joint, left/right_wrist_pitch_joint,
             left/right_wrist_roll_joint
  头部 (2):  head_yaw_joint, head_pitch_joint

脚部 link（用于接触检测）：left_ankle_roll_link, right_ankle_roll_link
躯干 link（用于终止检测）：torso_link

USD 路径说明：
  X2Ultra 目前仅提供 URDF，尚无 USD 文件。
  需要先用 Isaac Lab 的 URDF 转换工具生成 USD，
  然后将生成的 USD 放置于：
    legged_lab/assets/X2Ultra/x2ultra_description/usd/x2ultra.usd
  本配置中的 usd_path 已预设为该路径，请在转换完成后使用。
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR


X2ULTRA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # USD 文件路径：需先从 URDF 转换生成
        # URDF 源文件：assets/X2Ultra/x2ultra_description/x2_ultra.urdf
        usd_path=f"{ISAAC_ASSET_DIR}/X2Ultra/x2ultra_description/usd/x2ultra.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # X2Ultra 站立初始姿态（参考 URDF 关节限位设计）
        # 初始高度约 0.85m（pelvis 离地高度）
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            # ===== 腿部关节（来自 x2ultra_config.json）=====
            "right_hip_pitch_joint": -0.25,
            "right_hip_roll_joint": -0.01,
            "right_hip_yaw_joint": -0.01,
            "right_knee_joint": 0.45,
            "right_ankle_pitch_joint": -0.1,
            "right_ankle_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.25,
            "left_hip_roll_joint": -0.01,
            "left_hip_yaw_joint": -0.01,
            "left_knee_joint": 0.45,
            "left_ankle_pitch_joint": -0.1,
            "left_ankle_roll_joint": 0.0,
            # ===== 腰部关节（来自 x2ultra_config.json）=====
            "waist_yaw_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "waist_roll_joint": 0.0,
            # ===== 手臂关节（来自 x2ultra_config.json）=====
            "right_shoulder_pitch_joint": 0.4,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -1.2,
            "right_wrist_yaw_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.4,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -1.2,
            "left_wrist_yaw_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            # ===== 头部关节（来自 x2ultra_config.json）=====
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        # ===== 腿部主驱动关节 =====
        # 包含髋关节（3轴）、膝关节、腰部关节
        # 参考 URDF: hip/knee effort=120N·m, velocity=11.936rad/s
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*waist.*",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_pitch_joint": 120.0,
                ".*_knee_joint": 120.0,
                ".*waist_yaw_joint": 120.0,
                ".*waist_pitch_joint": 48.0,
                ".*waist_roll_joint": 48.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 11.936,
                ".*_hip_roll_joint": 11.936,
                ".*_hip_pitch_joint": 11.936,
                ".*_knee_joint": 11.936,
                ".*waist_yaw_joint": 11.936,
                ".*waist_pitch_joint": 13.088,
                ".*waist_roll_joint": 13.088,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
                ".*waist_yaw_joint": 40.0,
                ".*waist_pitch_joint": 40.0,
                ".*waist_roll_joint": 40.0,
            },
            damping={
                ".*_hip_yaw_joint": 3.0,
                ".*_hip_roll_joint": 3.0,
                ".*_hip_pitch_joint": 4.0,
                ".*_knee_joint": 5.0,
                ".*waist_yaw_joint": 8.0,
                ".*waist_pitch_joint": 5.0,
                ".*waist_roll_joint": 5.0,
            },
            armature=0.01,
        ),
        # ===== 踝关节 =====
        # ankle_pitch: effort=36N·m, velocity=13.087rad/s
        # ankle_roll:  effort=24N·m, velocity=15.077rad/s
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 36.0,
                ".*_ankle_roll_joint": 24.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 13.087,
                ".*_ankle_roll_joint": 15.077,
            },
            stiffness={".*_ankle_pitch_joint": 40.0, ".*_ankle_roll_joint": 30.0},
            damping={".*_ankle_pitch_joint": 3.0, ".*_ankle_roll_joint": 2.0},
            armature=0.01,
        ),
        # ===== 肩关节（pitch/roll）=====
        # effort=36N·m, velocity=13.088rad/s
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 36.0,
                ".*_shoulder_roll_joint": 36.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 13.088,
                ".*_shoulder_roll_joint": 13.088,
            },
            stiffness={".*_shoulder_pitch_joint": 80.0, ".*_shoulder_roll_joint": 40.0},
            damping={".*_shoulder_pitch_joint": 4.0, ".*_shoulder_roll_joint": 1.0},
            armature=0.01,
        ),
        # ===== 手臂关节（shoulder_yaw / elbow）=====
        # shoulder_yaw: effort=24N·m, velocity=15.077rad/s
        # elbow:        effort=24N·m, velocity=15.077rad/s
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_yaw_joint": 24.0,
                ".*_elbow_joint": 24.0,
            },
            velocity_limit_sim={
                ".*_shoulder_yaw_joint": 15.077,
                ".*_elbow_joint": 15.077,
            },
            stiffness={".*_shoulder_yaw_joint": 40.0, ".*_elbow_joint": 40.0},
            damping={".*_shoulder_yaw_joint": 1.0, ".*_elbow_joint": 1.0},
            armature=0.01,
        ),
        # ===== 腕关节 =====
        # wrist_yaw/pitch: effort=4.8N·m, velocity=4.188rad/s
        # wrist_roll:      effort=4.8N·m, velocity=4.188rad/s
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_yaw_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_wrist_yaw_joint": 4.8,
                ".*_wrist_pitch_joint": 4.8,
                ".*_wrist_roll_joint": 4.8,
            },
            velocity_limit_sim={
                ".*_wrist_yaw_joint": 4.188,
                ".*_wrist_pitch_joint": 4.188,
                ".*_wrist_roll_joint": 4.188,
            },
            stiffness={".*_wrist_yaw_joint": 40.0, ".*_wrist_pitch_joint": 40.0, ".*_wrist_roll_joint": 40.0},
            damping={".*_wrist_yaw_joint": 1.0, ".*_wrist_pitch_joint": 1.0, ".*_wrist_roll_joint": 1.0},
            armature=0.01,
        ),
        # ===== 头部关节 =====
        # head_yaw:   effort=2.6N·m, velocity=6.019rad/s
        # head_pitch: effort=0.6N·m, velocity=6.28rad/s
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_yaw_joint",
                "head_pitch_joint",
            ],
            effort_limit_sim={
                "head_yaw_joint": 2.6,
                "head_pitch_joint": 0.6,
            },
            velocity_limit_sim={
                "head_yaw_joint": 6.019,
                "head_pitch_joint": 6.28,
            },
            stiffness={"head_yaw_joint": 20.0, "head_pitch_joint": 20.0},
            damping={"head_yaw_joint": 1.0, "head_pitch_joint": 1.0},
            armature=0.01,
        ),
    },
)
