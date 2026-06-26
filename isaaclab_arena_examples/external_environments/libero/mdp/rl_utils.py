# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Goal-satisfaction helpers for single-task LIBERO (self-contained Arena port).

Trimmed from the original ``...libero/mdp/rl_utils.py`` to only the single-task
(non grouped multi-task) goal predicates used by ``libero_goals_reached``:
positional-relationship goals (put on / put in) and articulation-operation goals
(open / close / turn on, etc.).
"""

from __future__ import annotations

import re

import torch
import warp as wp

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv

_ORIENTATION_OFFSETS = {
    "right": (0.0, -0.05, 0.0),
    "left": (0.0, 0.05, 0.0),
    "front": (-0.15, 0.0, 0.0),
}


def _extract_single_joint_position(rigid_object: RigidObject, joint_pattern: str) -> torch.Tensor:
    """Return the joint position tensor (num_envs,) for the given joint pattern in the rigid object."""

    joint_ids, _ = rigid_object.find_joints(joint_pattern)
    if len(joint_ids) != 1:
        raise ValueError(
            f"Expected exactly one joint for pattern '{joint_pattern}' in '{rigid_object.name}', "
            f"but found {len(joint_ids)}."
        )

    joint_index = joint_ids[0]
    if isinstance(joint_index, torch.Tensor):
        joint_index = int(joint_index.item())

    # Newton backend exposes joint_pos as a warp array; convert to torch before indexing.
    joint_pos = wp.to_torch(rigid_object.data.joint_pos)[:, joint_index]
    return joint_pos.squeeze(-1) if joint_pos.ndim > 1 else joint_pos


def _resolve_entity_world_pos(env: ManagerBasedRLEnv, entity_name: str) -> torch.Tensor:
    """Return the world-frame position tensor (num_envs, 3) for an entity or articulated link.

    When no body path is specified (e.g. "robot"), return the articulation root pose.
    If body path is specified (e.g. "flat_stove_1/burnerplate"), return the body COM position.
    """

    # Newton backend exposes pose buffers as warp arrays (shape (num_envs,), dtype vec3); convert
    # to torch (-> (num_envs, 3)) before any torch slicing happens downstream.
    asset_rigid = env.scene.rigid_objects.get(entity_name)
    asset_articulation = env.scene.articulations.get(entity_name)
    if asset_rigid is not None and hasattr(asset_rigid, "data"):
        return wp.to_torch(asset_rigid.data.root_pos_w)
    if asset_articulation is not None and hasattr(asset_articulation, "data"):
        return wp.to_torch(asset_articulation.data.root_pos_w)

    if "/" not in entity_name:
        raise KeyError(f"Entity '{entity_name}' not found in scene and no articulation path provided.")

    base_name, _, body_path = entity_name.partition("/")
    base_asset = env.scene.articulations.get(base_name)
    if base_asset is None:
        raise KeyError(f"Base articulation '{base_name}' not found in scene for entity '{entity_name}'.")
    if not hasattr(base_asset, "find_bodies") or not hasattr(base_asset.data, "body_com_pose_w"):
        raise ValueError(f"Asset '{base_name}' does not expose articulation link data.")

    # When no body path is specified (e.g. "robot"), return the articulation root pose.
    if not body_path:
        if hasattr(base_asset.data, "root_pos_w"):
            return wp.to_torch(base_asset.data.root_pos_w)
        raise ValueError(f"Asset '{base_name}' does not expose root_pos_w for entity '{entity_name}'.")

    search_patterns = [re.escape(body_path), re.escape(entity_name)]
    body_indices: list[int] = []
    for pattern in search_patterns:
        body_indices, _ = base_asset.find_bodies(pattern, preserve_order=True)
        if body_indices:
            break

    if not body_indices:
        raise KeyError(f"Unable to locate body '{entity_name}' within articulation '{base_name}'.")

    body_index = body_indices[0]
    if isinstance(body_index, torch.Tensor):
        body_index = int(body_index.item())

    # body_com_pose_w is a warp array (num_envs, num_bodies) of transforms -> torch (num_envs,
    # num_bodies, 7); take the body's first 3 (position) components.
    return wp.to_torch(base_asset.data.body_com_pose_w)[:, body_index, :3]


def _articulation_operation_goal_satisfied(env: ManagerBasedRLEnv, goal: dict) -> torch.Tensor:
    """Evaluate whether an articulation operation goal is satisfied for all environments."""

    operation = goal["operation"]
    target = goal["target"]
    scene = env.scene

    if target == "flat_stove_1":
        thresholds = {"turnon": (0.5, 2.1), "turnoff": (-0.05, 0.05)}
        if operation not in thresholds:
            raise ValueError(f"Unsupported operation '{operation}' for target '{target}'.")
        stove = scene[target]
        knob_pos = _extract_single_joint_position(stove, "button")
        low, high = thresholds[operation]
        return torch.logical_and(knob_pos > low, knob_pos < high)

    if target == "white_cabinet_1":
        thresholds = {"open": (-0.16, -0.14), "close": (-0.05, 0.05)}
        if operation not in thresholds:
            raise ValueError(f"Unsupported operation '{operation}' for target '{target}'.")
        cabinet = scene[target]
        cabinet_pos = _extract_single_joint_position(cabinet, "bottom_level")
        low, high = thresholds[operation]
        return torch.logical_and(cabinet_pos > low, cabinet_pos < high)

    if target == "microwave_1":
        thresholds = {"open": (-2.094, -1.3), "close": (-0.2, 0.1)}
        if operation not in thresholds:
            raise ValueError(f"Unsupported operation '{operation}' for target '{target}'.")
        microwave = scene[target]
        microwave_pos = _extract_single_joint_position(microwave, "microjoint")
        low, high = thresholds[operation]
        return torch.logical_and(microwave_pos > low, microwave_pos < high)

    if target == "wooden_cabinet_1":
        thresholds = {"open": (-0.20, -0.14), "close": (-0.05, 0.05)}
        if operation not in thresholds:
            raise ValueError(f"Unsupported operation '{operation}' for target '{target}'.")
        layer = goal.get("layer")
        layer_to_pattern = {"top": "top_level", "middle": "middle_level", "bottom": "bottom_level"}
        if layer not in layer_to_pattern:
            raise ValueError(f"Unsupported layer '{layer}' for target '{target}'.")
        cabinet = scene[target]
        cabinet_pos = _extract_single_joint_position(cabinet, layer_to_pattern[layer])
        low, high = thresholds[operation]
        return torch.logical_and(cabinet_pos > low, cabinet_pos < high)

    raise ValueError(f"Unsupported operation target '{target}'.")


def _relationship_goal_satisfied(env: ManagerBasedRLEnv, goal: dict) -> torch.Tensor:
    """Evaluate positional relationship goals across all environments."""

    obj_name = goal["ref_obj"]
    target_name = goal["target"]
    xy_threshold = goal["xy_threshold"]
    height_threshold = goal["height_threshold"]
    height_diff = goal["height_diff"]
    enable_force_threshold = goal["enable_force_threshold"]

    target_pos_w = _resolve_entity_world_pos(env, target_name)
    obj_pos_w = _resolve_entity_world_pos(env, obj_name)

    pos_diff = obj_pos_w - target_pos_w

    orientation = str(goal.get("orientation", "")).lower()
    if orientation:
        offset = _ORIENTATION_OFFSETS.get(orientation)
        if offset is not None:
            pos_diff = pos_diff + pos_diff.new_tensor(offset)

    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    satisfied = torch.logical_and(xy_dist < xy_threshold, torch.abs(height_dist - height_diff) < height_threshold)

    if enable_force_threshold != "True":
        return satisfied

    contact_name = f"contact_{target_name}_{obj_name}"
    if contact_name not in env.scene.keys() or env.scene[contact_name] is None:
        return torch.zeros_like(satisfied, dtype=torch.bool, device=env.device)

    contact_force = wp.to_torch(env.scene[contact_name].data.force_matrix_w).squeeze(2).squeeze(1)
    if contact_force.ndim == 1:
        contact_force = contact_force.unsqueeze(0)
    contact_force_norm = torch.linalg.vector_norm(contact_force, dim=1)

    return torch.logical_and(satisfied, contact_force_norm > goal["force_threshold"])
