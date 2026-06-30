# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import warp as wp
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events import sample_object_poses

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def capture_object_init_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    init_pos_attr: str = "_object_init_pos",
):
    """Reset event: cache the object's resting world position (x, y, z) per env.

    Declare as a task ``mode="reset"`` event so it runs on every ``env.reset()`` (after the
    object is placed at its randomized pose). Two consumers read this per-env baseline:

    * :func:`isaaclab_arena.tasks.rewards.lift_object_rewards.object_lifted_above_reset`
      uses the resting z (last column) so "lifted" is measured relative to wherever the
      (randomized) object started.
    * :func:`isaaclab_arena.tasks.observations.observations.object_tilt_penalty_near_init`
      uses the full position so the tilt penalty is only applied while the object is still
      within its initial pickup region on the table.

    Args:
        object_cfg: Scene entity of the (rigid) object whose resting position is cached.
        init_pos_attr: Name of the per-env attribute set on ``env`` to hold the ``(N, 3)`` pose.
    """
    object = env.scene[object_cfg.name]
    pos = wp.to_torch(object.data.root_pos_w)[:, :3]
    if not hasattr(env, init_pos_attr):
        setattr(env, init_pos_attr, pos.clone())
    getattr(env, init_pos_attr)[env_ids] = pos[env_ids]


def randomize_poses_and_align_auxiliary_assets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
    fixed_asset_cfg: SceneEntityCfg | None = None,
    auxiliary_asset_cfgs: list[SceneEntityCfg] | None = None,
    randomization_mode: Literal["held_and_fixed_only", "held_fixed_and_auxiliary"] = "held_and_fixed_only",
):
    """
    Randomize object poses and update the poses of related assets accordingly.

    Args:
        randomization_mode:
            - "held_and_fixed_only": Randomize only the fixed and held assets independently.
            - "held_fixed_and_auxiliary": Randomize fixed, held, and auxiliary assets, with auxiliary
              assets positioned relative to the fixed asset.
    """
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])

            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )

            if (
                randomization_mode == "held_fixed_and_auxiliary"
                and auxiliary_asset_cfgs is not None
                and fixed_asset_cfg is not None
                and asset_cfg.name == fixed_asset_cfg.name
            ):
                # Place auxiliary assets at exactly the same pose as the fixed asset (zero offset).
                # NOTE: This assumes the asset USD files have base frames defined such that zero offset creates a valid scene.
                # Currently designed for gear mesh task where all gears share the same center point.
                # For other assets, this may cause geometry intersections. Customers need to adjust it accordingly.
                for j in range(len(auxiliary_asset_cfgs)):
                    rel_asset_cfg = auxiliary_asset_cfgs[j]
                    rel_asset = env.scene[rel_asset_cfg.name]
                    rel_asset.write_root_pose_to_sim(
                        torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
                    )
                    rel_asset.write_root_velocity_to_sim(
                        torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
                    )
