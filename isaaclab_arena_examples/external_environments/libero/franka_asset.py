# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Franka Panda configuration for LIBERO (self-contained Arena port).

Mirrors ``isaaclab_playground.assets.robots.franka:FRANKA_PANDA_LIBERO_HIGH_PD_CFG``.
It derives from the shared ``isaaclab_assets`` Franka high-PD config (available in Arena)
and applies the LIBERO-specific tweaks: stiffer arm PD, contact sensors, gravity disabled,
LIBERO init joint pose and base offset.
"""

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

FRANKA_PANDA_LIBERO_HIGH_PD_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
# The inherited config points at Robots/FrankaEmika/panda_instanceable.usd, which
# the Isaac 6.0 asset tree no longer publishes after the robot library was
# reorganized; fetching it fails with HTTP 404 during scene construction. The
# Panda now ships under Robots/FrankaRobotics/FrankaPanda, the same path
# isaaclab_assets already uses for FRANKA_ROBOTIQ_GRIPPER_CFG.
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.spawn.usd_path = (
    f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
)
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.spawn.activate_contact_sensors = True
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 8000.0
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 800.0
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 8000.0
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.actuators["panda_forearm"].damping = 800.0
# Reset actuator cfg back to Isaac Lab 2.3: the only actuator-level change from Lab 2.3 to Lab 3.0
# was Lab 3.0 adding ``armature=1e-3`` on the arm joints. Lab 2.3 left armature unset (None), which
# falls back to the value baked into the USD joint prim. Restore that to match Lab 2.3 dynamics.
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.actuators["panda_shoulder"].armature = None
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.actuators["panda_forearm"].armature = None
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.init_state.joint_pos = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint.*": 0.04,
}
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.init_state.pos = (-0.51, 0.0, 0.42)  # for libero living-room tabletop task
"""Configuration of Franka Emika Panda robot with stiffer PD control for LIBERO task."""
