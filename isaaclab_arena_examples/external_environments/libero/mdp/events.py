# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""LIBERO reset events (self-contained Arena port).

Trimmed from the original ``...libero/mdp/events.py`` to the two events used by the
single-task RL config:

* ``reset_libero_scene_to_initial_state`` — restore robot + objects from demo HDF5
  ``initial_state`` entries.
* ``reset_articulation_joints`` — sample articulated-object joint positions from a range.

Multi-task / domain-randomization helpers are dropped. The grouped asset-name remap is
kept but degrades gracefully to identity for single-task configs (which expose no
``get_group_index``).
"""

from __future__ import annotations

import glob
import os
import torch
import warp as wp
from typing import TYPE_CHECKING

from ..multitask_utils import decode_task_assignment, encode_task_assignment

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["reset_articulation_joints", "reset_libero_scene_to_initial_state"]


def _resolve_env_ids(env_ids_arg, total_envs: int, device: torch.device) -> torch.Tensor:
    """Normalize environment identifiers to a tensor of global indices."""
    if env_ids_arg is None:
        return torch.arange(total_envs, device=device, dtype=torch.long)
    if isinstance(env_ids_arg, slice):
        start, stop, step = env_ids_arg.indices(total_envs)
        return torch.arange(start, stop, step, device=device, dtype=torch.long)
    if torch.is_tensor(env_ids_arg):
        return env_ids_arg.to(device=device, dtype=torch.long)
    if hasattr(env_ids_arg, "__iter__"):
        return torch.tensor([int(val) for val in env_ids_arg], device=device, dtype=torch.long)
    return torch.tensor([int(env_ids_arg)], device=device, dtype=torch.long)


def _local_env_index(entity, env_idx: int) -> int | None:
    """Resolve the local instance index handled by an entity for a given global environment index."""
    mapping = getattr(entity, "_assigned_env_to_local", None)
    if mapping:
        return mapping.get(env_idx)
    count = getattr(entity, "num_instances", None)
    if count is None:
        return env_idx
    if 0 <= env_idx < count:
        return env_idx
    return None


def _scene_asset_exists(container, name: str) -> bool:
    """Check whether an asset container exposes a given asset name."""
    if container is None or not name:
        return False
    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(name) is not None
    try:
        return name in container
    except TypeError:
        return False


def _resolve_group_asset_name(
    env: "ManagerBasedEnv",
    entity_type: str,
    asset_name: str,
    env_idx: int,
    libero_config=None,
) -> str:
    """Resolve group-aware asset names so dataset keys map to scene assets.

    For single-task configs the scene asset names match the dataset keys directly, so this
    returns ``asset_name`` unchanged (and tolerates the absence of ``get_group_index``).
    """
    scene = getattr(env, "scene", None)
    if scene is None:
        return asset_name
    if entity_type == "articulation":
        collection = getattr(scene, "articulations", None)
    elif entity_type == "rigid_object":
        collection = getattr(scene, "rigid_objects", None)
    else:
        return asset_name

    if _scene_asset_exists(collection, asset_name):
        return asset_name

    if libero_config is None:
        libero_config = getattr(env.cfg, "libero_config", None)
    if libero_config is None:
        return asset_name

    get_group_index = getattr(libero_config, "get_group_index", None)
    if not callable(get_group_index):
        return asset_name
    group_idx = get_group_index(env_idx)
    candidate_bases: list[str] = [asset_name]
    if entity_type == "rigid_object":
        if not asset_name.startswith("object_"):
            candidate_bases.append(f"object_{asset_name}_group_{group_idx}")
    elif entity_type == "articulation":
        if not asset_name.startswith("robot"):
            candidate_bases.append(f"object_{asset_name}_group_{group_idx}")

    for candidate in candidate_bases:
        if _scene_asset_exists(collection, candidate):
            return candidate

    return asset_name


def _sanitize_task_name(task_name: str) -> str:
    """Create a filesystem-friendly slug from the Libero task name."""
    return task_name.strip().replace(" ", "_")


def _lookup_task_info(libero_config, assignment: tuple[str, int]) -> dict | None:
    """Fetch task metadata whether the config stores tuple or string keys."""
    tasks_by_spec = getattr(libero_config, "tasks_by_spec", {})
    task_info = tasks_by_spec.get(assignment)
    if task_info is not None:
        return task_info
    task_key = encode_task_assignment(*assignment)
    task_info = tasks_by_spec.get(task_key)
    if task_info is not None:
        return task_info
    suite_name, task_id = assignment
    return tasks_by_spec.get(f"{suite_name}::{task_id}")


def _resolve_demo_dataset_path(datasets_root: str, libero_config, assignment: tuple[str, int]) -> str:
    """Resolve the HDF5 path for a (suite, task_id) assignment."""
    suite_name, task_id = assignment
    task_info = _lookup_task_info(libero_config, assignment)
    task_name = task_info.get("task_name", "") if task_info else ""
    expected_name = f"{suite_name}_task{task_id}_{_sanitize_task_name(task_name)}_demo.hdf5"
    candidate_path = os.path.join(datasets_root, expected_name)
    if os.path.isfile(candidate_path):
        return candidate_path

    pattern = os.path.join(datasets_root, f"{suite_name}_task{task_id}_*.hdf5")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find Libero demo for {suite_name} task {task_id} under '{datasets_root}'."
        )
    if len(matches) == 1:
        return matches[0]

    filtered = []
    if task_name:
        slug = _sanitize_task_name(task_name)
        filtered = [match for match in matches if slug in os.path.basename(match)]
    if len(filtered) == 1:
        return filtered[0]
    raise FileNotFoundError(
        f"Multiple demos matched '{pattern}'. Please disambiguate by providing the exact file. Matches: {matches}"
    )


def _clone_initial_state_dict(
    state: dict[str, dict[str, dict[str, torch.Tensor]]] | None,
) -> dict[str, dict[str, dict[str, torch.Tensor]]] | None:
    """Deep-copy a nested initial-state dictionary to avoid in-place edits."""
    if state is None:
        return None
    cloned: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for entity_type, entity_data in state.items():
        entity_copy: dict[str, dict[str, torch.Tensor]] = {}
        for entity_name, states in entity_data.items():
            state_copy: dict[str, torch.Tensor] = {}
            for key, value in states.items():
                state_copy[key] = value.clone() if torch.is_tensor(value) else value
            entity_copy[entity_name] = state_copy
        cloned[entity_type] = entity_copy
    return cloned


def _load_demo_initial_states(
    dataset_path: str,
    device: torch.device,
) -> list[dict[str, dict[str, dict[str, torch.Tensor]]] | None]:
    """Load full simulator initial states from a Libero HDF5 dataset."""
    handler = HDF5DatasetFileHandler()
    handler.open(dataset_path)
    states: list[dict[str, dict[str, dict[str, torch.Tensor]]] | None] = []
    try:
        for episode_name in sorted(handler.get_episode_names()):
            episode = handler.load_episode(episode_name, device)
            if episode is None:
                continue
            init_state = episode.get_initial_state() if "initial_state" in episode.data else None
            states.append(_clone_initial_state_dict(init_state))
    finally:
        handler.close()
    return states


def _remap_state_asset_names_for_group(
    env: "ManagerBasedEnv",
    env_idx: int,
    state: dict[str, dict[str, dict[str, torch.Tensor]]] | None,
) -> dict[str, dict[str, dict[str, torch.Tensor]]] | None:
    """Rewrite dataset state asset names to match grouped scene assets for the given environment."""
    if state is None:
        return None

    libero_config = getattr(env.cfg, "libero_config", None)
    if libero_config is None:
        return state

    remapped_state: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for entity_type, entity_states in state.items():
        if entity_type not in {"articulation", "rigid_object"} or not isinstance(entity_states, dict):
            remapped_state[entity_type] = entity_states
            continue

        resolved_entries: dict[str, dict[str, torch.Tensor]] = {}
        for asset_name, asset_state in entity_states.items():
            resolved_name = _resolve_group_asset_name(env, entity_type, asset_name, env_idx, libero_config)
            resolved_entries[resolved_name] = asset_state
        remapped_state[entity_type] = resolved_entries

    return remapped_state


def _apply_reservations_to_envs(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    reservations: dict[int, dict[str, dict[str, dict[str, torch.Tensor]]] | None],
    include_articulations: bool,
    include_rigid_objects: bool,
    robot_joint_noise_std: float = 0.0,
):
    """Resolve reservation states for a batch of environments and apply them to the scene."""
    if env_ids.numel() == 0:
        return

    applied_envs: list[int] = []
    remapped_states: list[dict[str, dict[str, dict[str, torch.Tensor]]]] = []

    for env_idx in env_ids.tolist():
        state = reservations.get(env_idx)
        if state is None:
            continue
        remapped = _remap_state_asset_names_for_group(env, env_idx, state)
        if remapped is None:
            continue
        applied_envs.append(env_idx)
        remapped_states.append(remapped)

    for env_idx, state in zip(applied_envs, remapped_states):
        _apply_initial_state_to_env(
            env,
            env_idx,
            state,
            include_articulations=include_articulations,
            include_rigid_objects=include_rigid_objects,
            robot_joint_noise_std=robot_joint_noise_std,
        )


def _apply_initial_state_to_env(
    env: "ManagerBasedEnv",
    env_idx: int,
    state: dict[str, dict[str, dict[str, torch.Tensor]]],
    include_articulations: bool = False,
    include_rigid_objects: bool = True,
    robot_joint_noise_std: float = 0.0,
):
    """Apply per-env initial state overrides to the simulation assets."""
    env_tensor = torch.tensor([env_idx], dtype=torch.long, device=env.device)
    origin = env.scene.env_origins[env_idx, 0:3].to(device=env.device)

    if include_articulations and "articulation" in state:
        for asset_name, asset_state in state["articulation"].items():
            articulation = env.scene.articulations.get(asset_name)
            if articulation is None:
                continue
            root_pose = asset_state.get("root_pose", None)
            root_velocity = asset_state.get("root_velocity", None)
            joint_position = asset_state.get("joint_position", None)
            joint_velocity = asset_state.get("joint_velocity", None)
            if asset_name == "robot":
                # Keep robot base pose from config/default reset, but apply joint states from demo.
                root_pose = None
                root_velocity = None

            if root_pose is not None:
                root_pose = root_pose.clone()
                root_pose[:, :3] += origin
                articulation.write_root_pose_to_sim(root_pose, env_ids=env_tensor)
            if root_velocity is not None:
                articulation.write_root_velocity_to_sim(root_velocity, env_ids=env_tensor)
            if joint_position is not None and joint_velocity is not None:
                joint_position = joint_position.clone()
                if asset_name == "robot" and robot_joint_noise_std > 0.0:
                    joint_position += torch.randn_like(joint_position) * float(robot_joint_noise_std)
                    local_idx = _local_env_index(articulation, env_idx)
                    if local_idx is not None:
                        # Newton backend: soft_joint_pos_limits is a warp array -> convert to torch.
                        joint_pos_limits = wp.to_torch(articulation.data.soft_joint_pos_limits)[
                            local_idx : local_idx + 1
                        ]
                        joint_position = joint_position.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
                articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_tensor)
                articulation.set_joint_position_target(joint_position, env_ids=env_tensor)
                articulation.set_joint_velocity_target(joint_velocity, env_ids=env_tensor)

    if include_rigid_objects and "rigid_object" in state:
        for asset_name, asset_state in state["rigid_object"].items():
            rigid_object = env.scene.rigid_objects.get(asset_name)
            if rigid_object is None:
                continue
            root_pose = asset_state.get("root_pose", None)
            root_velocity = asset_state.get("root_velocity", None)
            if root_pose is not None:
                root_pose = root_pose.clone()
                root_pose[:, :3] += origin
                rigid_object.write_root_pose_to_sim(root_pose, env_ids=env_tensor)
            if root_velocity is not None:
                rigid_object.write_root_velocity_to_sim(root_velocity, env_ids=env_tensor)


def reset_articulation_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    joint_pos_range: dict[str, list[float]],
    joint_names: list[str],
):
    """Sample joint positions from *joint_pos_range* and write them to the sim.

    Used by single-task RL configs where ``franka_stack_events.randomize_object_pose``
    only handles root pose and ignores ``joint_pos_range``.
    """
    if env_ids is None:
        return
    n_joints = len(joint_names)
    n_envs = len(env_ids)

    jp_min = torch.zeros(n_joints, device=env.device)
    jp_max = torch.zeros(n_joints, device=env.device)
    for jid in range(n_joints):
        r = joint_pos_range.get(str(jid), [0.0, 0.0])
        jp_min[jid] = r[0]
        jp_max[jid] = r[1]

    joint_pos = math_utils.sample_uniform(jp_min, jp_max, (n_envs, n_joints), device=env.device)
    asset = env.scene[asset_cfg.name]
    asset.write_joint_state_to_sim(
        joint_pos,
        torch.zeros(n_envs, n_joints, device=env.device),
        env_ids=env_ids,
    )


def reset_libero_scene_to_initial_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    datasets_root: str,
    include_articulations: bool = False,
    include_rigid_objects: bool = True,
    robot_joint_noise_std: float = 0.0,
):
    """Apply demo-sampled initial states directly from Libero HDF5 files."""
    libero_config = getattr(env.cfg, "libero_config", None)
    if libero_config is None:
        return

    requested_envs = _resolve_env_ids(env_ids, env.scene.num_envs, env.device)
    if requested_envs.numel() == 0:
        return

    if not datasets_root:
        raise ValueError("datasets_root must be provided for demo-based scene reset.")
    datasets_root = os.path.abspath(datasets_root)

    cache = getattr(env, "_libero_scene_demo_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_libero_scene_demo_cache", cache)

    envs_by_assignment: dict[tuple[str, int], list[int]] = {}
    for env_idx in requested_envs.tolist():
        assignment_raw = libero_config.get_task_assignment_for_env(env_idx)
        assignment = decode_task_assignment(assignment_raw)
        envs_by_assignment.setdefault(assignment, []).append(env_idx)

    reservations: dict[int, dict[str, dict[str, dict[str, torch.Tensor]]] | None] = {}
    for assignment, env_list in envs_by_assignment.items():
        cache_key = (datasets_root, assignment)
        states = cache.get(cache_key)
        if states is None:
            dataset_path = _resolve_demo_dataset_path(datasets_root, libero_config, assignment)
            states = _load_demo_initial_states(dataset_path, env.device)
            cache[cache_key] = states
        if not states:
            continue

        demo_indices = torch.randint(0, len(states), (len(env_list),), device=env.device)
        for idx, env_idx in enumerate(env_list):
            state = states[int(demo_indices[idx].item())]
            reservations[env_idx] = _clone_initial_state_dict(state)

    _apply_reservations_to_envs(
        env,
        requested_envs,
        reservations,
        include_articulations=include_articulations,
        include_rigid_objects=include_rigid_objects,
        robot_joint_noise_std=robot_joint_noise_std,
    )
