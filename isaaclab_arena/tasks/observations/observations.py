# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms


def object_position_in_world_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Observation the position of the object in the world frame."""
    object = env.scene[asset_cfg.name]
    return wp.to_torch(object.data.root_pos_w)


def object_position_in_frame(
    env: ManagerBasedRLEnv,
    root_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    root_frame: RigidObject = env.scene[root_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = wp.to_torch(object.data.root_pos_w)[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        wp.to_torch(root_frame.data.root_pos_w), wp.to_torch(root_frame.data.root_quat_w), object_pos_w
    )
    return object_pos_b


def object_pose_in_static_frame(
    env: ManagerBasedRLEnv,
    frame_pos: tuple[float, float, float],
    frame_quat_wxyz: tuple[float, float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object pose (position + quaternion) expressed in a STATIC scene frame.

    Use this for a reference frame that never moves relative to the environment
    origin (e.g. a fridge shelf the object must be placed on). Such a frame is not
    a simulated rigid body / articulation, so it has no ``data.root_pose_w`` to read
    at runtime; instead its world pose is reconstructed as
    ``env.scene.env_origins + frame_pos`` (Isaac Lab environments differ only by a
    per-env translation) with a constant orientation ``frame_quat_wxyz``.

    Returns a ``(num_envs, 7)`` tensor ``[x, y, z, qw, qx, qy, qz]`` of the object
    pose in that frame: translation/heading-invariant and well-scaled (≈ the
    object→goal offset), which makes it a good asymmetric-critic privileged obs.

    Args:
        frame_pos: Frame position relative to the environment origin (meters).
        frame_quat_wxyz: Constant frame orientation as a ``(w, x, y, z)`` quaternion.
        object_cfg: Scene entity of the (rigid) object whose pose is reported.
    """
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = wp.to_torch(object.data.root_pos_w)[:, :3]
    object_quat_w = wp.to_torch(object.data.root_quat_w)
    device = object_pos_w.device
    dtype = object_pos_w.dtype
    num_envs = object_pos_w.shape[0]
    frame_pos_w = env.scene.env_origins + torch.tensor(frame_pos, device=device, dtype=dtype)
    frame_quat_w = torch.tensor(frame_quat_wxyz, device=device, dtype=dtype).unsqueeze(0).expand(num_envs, 4)
    object_pos_f, object_quat_f = subtract_frame_transforms(frame_pos_w, frame_quat_w, object_pos_w, object_quat_w)
    return torch.cat([object_pos_f, object_quat_f], dim=-1)


def object_position_in_static_frame(
    env: ManagerBasedRLEnv,
    frame_pos: tuple[float, float, float],
    frame_quat_wxyz: tuple[float, float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object POSITION in a STATIC scene frame (e.g. the fridge shelf) — the object→target
    displacement, ``(num_envs, 3)``.

    This is the well-scaled, goal-centric core of the asymmetric-critic privileged obs.
    The orientation quaternion is INTENTIONALLY OMITTED: its double-cover (q ≡ -q) flips
    sign discontinuously as the object rotates, which appears to the critic as input jumps
    and drives TD-target / critic-loss spikes. Position-only is continuous and stable.
    """
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = wp.to_torch(object.data.root_pos_w)[:, :3]
    object_quat_w = wp.to_torch(object.data.root_quat_w)
    device = object_pos_w.device
    dtype = object_pos_w.dtype
    num_envs = object_pos_w.shape[0]
    frame_pos_w = env.scene.env_origins + torch.tensor(frame_pos, device=device, dtype=dtype)
    frame_quat_w = torch.tensor(frame_quat_wxyz, device=device, dtype=dtype).unsqueeze(0).expand(num_envs, 4)
    object_pos_f, _ = subtract_frame_transforms(frame_pos_w, frame_quat_w, object_pos_w, object_quat_w)
    return object_pos_f
