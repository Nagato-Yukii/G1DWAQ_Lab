# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""
X2Ultra 标准环境（继承自 G1Env）

X2Ultra 与 G1 的结构完全一致（均为 31-DoF 人形机器人），
因此直接继承 G1Env 并替换配置类型即可。
"""

from __future__ import annotations

from legged_lab.envs.g1.g1_env import G1Env
from legged_lab.envs.x2ultra.x2ultra_config import (
    X2UltraFlatEnvCfg,
    X2UltraRoughEnvCfg,
)


class X2UltraEnv(G1Env):
    """X2Ultra 人形机器人标准训练环境。

    继承 G1Env 的全部实现，仅覆盖类型注解以便 IDE 提示。
    所有行为（观测计算、重置、步进、地形课程）与 G1Env 完全一致。
    """

    def __init__(
        self,
        cfg: X2UltraFlatEnvCfg | X2UltraRoughEnvCfg,
        headless: bool,
    ):
        self.cfg: X2UltraFlatEnvCfg | X2UltraRoughEnvCfg
        super().__init__(cfg=cfg, headless=headless)
