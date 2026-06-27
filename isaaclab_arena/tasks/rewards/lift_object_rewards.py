# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms


def object_root_z(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """World-frame z of the object's root, ``(num_envs,)``."""
    object: RigidObject = env.scene[object_cfg.name]
    return wp.to_torch(object.data.root_pos_w)[:, 2]


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    return torch.where(object_root_z(env, object_cfg) > minimal_height, 1.0, 0.0)


def object_lifted_above_reset(
    env: ManagerBasedRLEnv,
    lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    init_pos_attr: str = "_object_init_pos",
) -> torch.Tensor:
    """1 (per call) while the object is raised >= ``lift_height`` above its resting z.

    Unlike :func:`object_is_lifted` (absolute world height), this measures the lift
    RELATIVE to the object's per-env resting z captured at reset. Pair it with the
    :func:`isaaclab_arena.tasks.events.capture_object_init_pos` reset event (using the
    same ``init_pos_attr``), which caches the per-env ``(N, 3)`` resting position on the
    env; the resting z is read from its last column. The ``RewardTermCfg`` weight sets the
    magnitude. Returns ``(num_envs,)`` float (0/1), falling back to zeros until the
    baseline is set.
    """
    z = object_root_z(env, object_cfg)
    init_pos = getattr(env, init_pos_attr, None)
    if init_pos is None:
        return torch.zeros(env.num_envs, device=env.device)
    return ((z - init_pos[:, 2]) > lift_height).float()


def object_dropped_below(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """1 (per call) while the object has fallen below ``minimum_height`` (world z).

    The directional counterpart of :func:`object_is_lifted`. Use a NEGATIVE
    ``RewardTermCfg`` weight to turn it into a drop penalty. Returns ``(num_envs,)``
    float (0/1).
    """
    return (object_root_z(env, object_cfg) < minimum_height).float()


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        wp.to_torch(robot.data.root_pos_w), wp.to_torch(robot.data.root_quat_w), des_pos_b
    )
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - wp.to_torch(object.data.root_pos_w), dim=1)
    # rewarded if the object is lifted above the threshold
    return (wp.to_torch(object.data.root_pos_w)[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
