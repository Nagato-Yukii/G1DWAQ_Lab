# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""
X2Ultra DreamWaQ 环境（继承自 G1DwaqEnv）

X2Ultra 与 G1 的 DreamWaQ 实现完全一致（均为 31-DoF 两足人形机器人），
直接继承 G1DwaqEnv 并替换配置类型即可。

所有 DWAQ 核心逻辑均在 G1DwaqEnv 中实现：
  - 观测历史缓冲区（VAE 编码器输入）
  - 步态相位更新（两足交替步态）
  - 前方障碍高度计算（自适应抬腿）
  - prev_critic_obs 管理（速度估计监督）
"""

from __future__ import annotations

from legged_lab.envs.g1.g1_dwaq_env import G1DwaqEnv
from legged_lab.envs.x2ultra.x2ultra_dwaq_config import X2UltraDwaqEnvCfg


class X2UltraDwaqEnv(G1DwaqEnv):
    """X2Ultra DreamWaQ 盲行训练环境。

    继承 G1DwaqEnv 的全部实现，仅覆盖类型注解。

    观测维度（DoF=31，步态相位启用）：
      Actor 输入:  107 维（单帧）× 1 帧 = 107 维
      Critic 输入: 131 维（单帧）× 1 帧 + 187 维（高度扫描）= 318 维
      obs_hist:    107 维 × 5 帧 = 535 维（VAE 编码器输入）
    """

    def __init__(
        self,
        cfg: X2UltraDwaqEnvCfg,
        headless: bool,
    ):
        self.cfg: X2UltraDwaqEnvCfg
        super().__init__(cfg=cfg, headless=headless)
