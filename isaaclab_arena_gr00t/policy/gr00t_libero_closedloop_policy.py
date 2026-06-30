# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Closed-loop GR00T policy for the Franka LIBERO Abs-IK environment.

Unlike :class:`~isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy`
(GR1 / G1 / DROID), whose policy I/O is *joint-space* (joint-name remapping between policy
and sim order), the LIBERO Franka checkpoint is *end-effector pose based*:

- **State** fed to GR00T: ``franka_eef_pose`` (pos[3] + axis-angle rotvec[3], base frame)
  and ``franka_gripper_pos`` (first finger position).
- **Action** produced by GR00T: ``franka_eef_pose`` (ABSOLUTE pos[3] + rotvec[3], the
  relative->absolute decode happens inside the GR00T processor using the supplied state as
  reference) and ``franka_gripper_pos``.
- The Arena ``libero`` env (:class:`AbsIKLiberoRLEnvCfg`) consumes an 8-D Abs-IK action
  ``pos[3] + quat_xyzw[4] + gripper[1]`` (``DifferentialInverseKinematicsActionCfg`` with
  ``use_relative_mode=False`` + ``AbsBinaryJointPositionActionCfg``).

So this policy reimplements the verl ``FrankaLiberoEmbodiment`` state-extract / action-convert
conventions (quaternion xyzw<->wxyz reorder, quat<->axis-angle, gripper pass-through) on top of
the standard :class:`gr00t.policy.gr00t_policy.Gr00tPolicy`, and reuses the shared
:class:`~isaaclab_arena.policy.action_chunking.ActionChunkingState`.

Run via the Arena policy runner with the LIBERO external environment::

    python isaaclab_arena/evaluation/policy_runner.py \\
        --policy_type isaaclab_arena_gr00t.policy.gr00t_libero_closedloop_policy.Gr00tLiberoClosedloopPolicy \\
        --policy_config_yaml_path \\
        isaaclab_arena_gr00t/policy/config/franka_libero_gr00t_closedloop_config.yaml \\
        --num_steps 512 --enable_cameras \\
        --external_environment_class_path \\
        isaaclab_arena_examples.external_environments.libero:LiberoEnvironment \\
        libero --task_suite libero_spatial --task_id 3
"""

from __future__ import annotations

import argparse
import gc
import gymnasium as gym
import numpy as np
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

from isaaclab_arena.policy.action_chunking import ActionChunkingState
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.utils.multiprocess import get_local_rank, get_world_size
from isaaclab_arena_gr00t.policy.config.task_mode import TaskMode
from isaaclab_arena_gr00t.utils.eagle_config_compat import apply_eagle_config_compat
from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml, to_numpy, to_tensor

# The Arena ``libero`` env exposes the end-effector pose as ``[pos(3), quat_xyzw(4)]`` (Isaac
# Lab 3.0 xyzw convention) followed by the 2 finger joints, all concatenated in the ``policy``
# observation group. The EEF pose therefore occupies the first 7 columns.
_EEF_POSE_DIM = 7
# Sim Abs-IK action: pos(3) + quat_xyzw(4) + gripper(1).
_SIM_ACTION_DIM = 8

_ALLOWED_EMBODIMENT_TAGS = ("NEW_EMBODIMENT", "OXE_DROID", "GR1")


# --------------------------------------------------------------------------- #
# Quaternion / axis-angle helpers (numpy)
#
# Mirror verl ``FrankaLiberoEmbodiment._quat2axisangle`` / ``_axisangle2quat`` so this eval
# path matches the training/SAC action convention exactly. GR00T's LIBERO state/action
# modality uses (w, x, y, z); the Arena env uses (x, y, z, w).
# --------------------------------------------------------------------------- #


def _quat_xyzw_to_axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) ``[x, y, z, w]`` to axis-angle rotation vector(s).

    Args:
        quat_xyzw: Array of shape ``(..., 4)`` in (x, y, z, w) order.

    Returns:
        Rotation vectors of shape ``(..., 3)`` (axis * angle, radians).
    """
    w = np.clip(quat_xyzw[..., 3:4], -1.0, 1.0)
    xyz = quat_xyzw[..., 0:3]
    angle = 2.0 * np.arccos(np.abs(w))
    den = np.sqrt(1.0 - w * w)
    small = den < 1e-8
    return np.where(small, np.zeros_like(xyz), xyz / den * angle * np.sign(w))


def _axisangle_to_quat_xyzw(axisangle: np.ndarray) -> np.ndarray:
    """Convert axis-angle rotation vector(s) to quaternion(s) ``[x, y, z, w]``.

    Args:
        axisangle: Array of shape ``(..., 3)`` (axis * angle, radians).

    Returns:
        Quaternions of shape ``(..., 4)`` in (x, y, z, w) order.
    """
    angle = np.linalg.norm(axisangle, axis=-1, keepdims=True)
    angle = np.clip(angle, 1e-8, None)
    axis = axisangle / angle
    half = angle * 0.5
    w = np.cos(half)
    xyz = axis * np.sin(half)
    # (w, x, y, z) -> (x, y, z, w)
    return np.concatenate([xyz, w], axis=-1)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class Gr00tLiberoClosedloopPolicyConfig:
    """Config for the closed-loop GR00T LIBERO (Franka Abs-IK) policy."""

    model_path: str = field(
        default=None, metadata={"description": "Path to the GR00T LIBERO checkpoint directory (or HF repo id)."}
    )
    language_instruction: str = field(
        default="", metadata={"description": "Fallback language instruction if the task does not provide one."}
    )
    embodiment_tag: str = field(
        default="NEW_EMBODIMENT", metadata={"description": "GR00T embodiment tag for the LIBERO checkpoint."}
    )
    action_horizon: int = field(
        default=16, metadata={"description": "Number of actions GR00T predicts per inference (action horizon)."}
    )
    action_chunk_length: int = field(
        default=16, metadata={"description": "Number of actions executed per inference (<= action_horizon)."}
    )
    # --- Arena `libero` env observation layout ---
    state_obs_group_key: str = field(
        default="policy", metadata={"description": "Observation group holding the concatenated eef_pose + gripper."}
    )
    camera_obs_group_key: str = field(
        default="rgb_camera", metadata={"description": "Observation group holding the per-camera RGB images."}
    )
    camera_obs_to_video_key: dict = field(
        default_factory=dict,
        metadata={
            "description": (
                "Optional map {env_camera_obs_term -> gr00t_video_key}. Empty -> identity (the env camera obs term"
                " name equals the gr00t video modality key, as is the case for the LIBERO checkpoint)."
            )
        },
    )
    task_mode_name: str = field(
        default=TaskMode.FRANKA_LIBERO_MANIPULATION.value,
        metadata={"description": "Task mode tag (informational; LIBERO uses eef-pose I/O, no joint configs)."},
    )
    policy_device: str = field(
        default="cuda", metadata={"description": "Device to run the GR00T model on (e.g. 'cuda', 'cpu')."}
    )

    def __post_init__(self):
        assert self.model_path is not None, "model_path must be set in the LIBERO closed-loop policy config"
        is_hf_id = bool(
            self.model_path and "/" in self.model_path and not self.model_path.startswith(("/", "."))
        )
        assert (
            Path(self.model_path).exists() or is_hf_id
        ), f"model_path does not exist and is not a HuggingFace model id: {self.model_path}"
        assert (
            self.action_chunk_length <= self.action_horizon
        ), "action_chunk_length must be <= action_horizon"
        assert self.embodiment_tag in _ALLOWED_EMBODIMENT_TAGS, (
            f"embodiment_tag must be one of {_ALLOWED_EMBODIMENT_TAGS} (LIBERO checkpoint uses NEW_EMBODIMENT),"
            f" got: {self.embodiment_tag}"
        )


@dataclass
class Gr00tLiberoClosedloopPolicyArgs:
    """CLI-facing arguments for :class:`Gr00tLiberoClosedloopPolicy`."""

    policy_config_yaml_path: str
    policy_device: str = "cuda"
    num_envs: int = 1

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> Gr00tLiberoClosedloopPolicyArgs:
        return cls(
            policy_config_yaml_path=args.policy_config_yaml_path,
            policy_device=args.policy_device,
            num_envs=args.num_envs,
        )


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


class Gr00tLiberoClosedloopPolicy(PolicyBase):
    """Closed-loop GR00T policy for the Franka LIBERO Abs-IK Arena environment."""

    name = "gr00t_libero_closedloop"
    config_class = Gr00tLiberoClosedloopPolicyArgs

    def __init__(self, config: Gr00tLiberoClosedloopPolicyArgs):
        super().__init__(config)

        self.policy_config: Gr00tLiberoClosedloopPolicyConfig = create_config_from_yaml(
            config.policy_config_yaml_path, Gr00tLiberoClosedloopPolicyConfig
        )

        self.num_envs = config.num_envs
        self.device = config.policy_device
        world_size = get_world_size()
        if world_size > 1 and "cuda" in self.device:
            self.device = f"cuda:{get_local_rank()}"

        # The GR00T policy holds the processor (normalization + relative->absolute decode) and
        # the modality config (key names + horizons), loaded from the checkpoint.
        apply_eagle_config_compat()
        self.policy: Gr00tPolicy | None = Gr00tPolicy(
            model_path=self.policy_config.model_path,
            embodiment_tag=EmbodimentTag[self.policy_config.embodiment_tag],
            device=self.device,
            strict=True,
        )

        modality_configs = self.policy.modality_configs
        self.video_keys: list[str] = list(modality_configs["video"].modality_keys)
        self.state_keys: list[str] = list(modality_configs["state"].modality_keys)
        self.action_keys: list[str] = list(modality_configs["action"].modality_keys)
        self.language_key: str = modality_configs["language"].modality_keys[0]

        # Identify the gripper vs eef keys by name so we build/decode the right modality groups.
        self.state_gripper_key = self._find_gripper_key(self.state_keys)
        self.state_eef_key = self._find_other_key(self.state_keys, self.state_gripper_key)
        self.action_gripper_key = self._find_gripper_key(self.action_keys)
        self.action_eef_key = self._find_other_key(self.action_keys, self.action_gripper_key)

        self.action_horizon = self.policy_config.action_horizon
        self.action_chunk_length = self.policy_config.action_chunk_length
        self._chunking_state = ActionChunkingState(
            num_envs=self.num_envs,
            action_chunk_length=self.action_chunk_length,
            action_horizon=self.action_horizon,
            action_dim=_SIM_ACTION_DIM,
            device=self.device,
            dtype=torch.float,
        )

        self.task_description: str | None = None

    # ------------------------------------------------------------------ #
    # CLI helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_args(args: argparse.Namespace) -> Gr00tLiberoClosedloopPolicy:
        return Gr00tLiberoClosedloopPolicy(Gr00tLiberoClosedloopPolicyArgs.from_cli_args(args))

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            "Gr00t LIBERO Closedloop Policy", "Arguments for the Franka LIBERO closed-loop GR00T policy"
        )
        group.add_argument(
            "--policy_config_yaml_path",
            type=str,
            required=True,
            help="Path to the GR00T LIBERO closed-loop policy config YAML file",
        )
        group.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Device to use for the policy-related operations (default: cuda)",
        )
        return parser

    # ------------------------------------------------------------------ #
    # Task description
    # ------------------------------------------------------------------ #

    def set_task_description(self, task_description: str | None) -> str:
        if task_description is None:
            task_description = self.policy_config.language_instruction
        if not task_description:
            raise ValueError(
                "No language instruction provided. Set 'language_instruction' in the policy config, pass"
                " --language_instruction on the CLI, or define 'task_description' on the task class."
            )
        self.task_description = task_description
        return self.task_description

    # ------------------------------------------------------------------ #
    # Observation / action conversion
    # ------------------------------------------------------------------ #

    def _build_policy_observations(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Build the nested GR00T observation (video / state / language) from the env obs."""
        assert self.task_description is not None, "Task description is not set"

        # --- state: policy group is [pos(3), quat_xyzw(4), finger1, finger2] (concatenated) ---
        state_group = observation[self.policy_config.state_obs_group_key]
        state_np = to_numpy(state_group).astype(np.float32)
        assert state_np.ndim == 2 and state_np.shape[1] >= _EEF_POSE_DIM + 1, (
            f"Expected '{self.policy_config.state_obs_group_key}' obs of shape (N, >= {_EEF_POSE_DIM + 1}),"
            f" got {state_np.shape}. Is concatenate_terms=True on the eef_pose+gripper group?"
        )
        num_envs = state_np.shape[0]
        eef_pos = state_np[:, :3]
        eef_quat_xyzw = state_np[:, 3:_EEF_POSE_DIM]
        gripper_state = state_np[:, _EEF_POSE_DIM : _EEF_POSE_DIM + 1]  # first finger
        eef_rotvec = _quat_xyzw_to_axisangle(eef_quat_xyzw)
        franka_eef_pose = np.concatenate([eef_pos, eef_rotvec], axis=-1).astype(np.float32)  # (N, 6)

        # --- video: per-camera images from the rgb_camera group (concatenate_terms must be False) ---
        camera_group = observation[self.policy_config.camera_obs_group_key]
        assert isinstance(camera_group, dict), (
            f"Expected '{self.policy_config.camera_obs_group_key}' obs to be a per-camera dict; got"
            f" {type(camera_group)}. Set concatenate_terms=False on the RGB camera obs group."
        )
        video: dict[str, np.ndarray] = {}
        for video_key in self.video_keys:
            cam_term = self.policy_config.camera_obs_to_video_key.get(video_key, video_key)
            assert cam_term in camera_group, (
                f"Camera obs term '{cam_term}' (for video key '{video_key}') not found in"
                f" '{self.policy_config.camera_obs_group_key}' group keys {list(camera_group.keys())}."
            )
            img = to_numpy(camera_group[cam_term]).astype(np.uint8)  # (N, H, W, C)
            assert img.ndim == 4 and img.shape[-1] == 3, f"Camera '{cam_term}' must be (N, H, W, 3), got {img.shape}"
            video[video_key] = img.reshape(num_envs, 1, *img.shape[1:])

        state = {
            self.state_eef_key: franka_eef_pose.reshape(num_envs, 1, -1),
            self.state_gripper_key: gripper_state.reshape(num_envs, 1, -1),
        }
        language = {self.language_key: [[self.task_description] for _ in range(num_envs)]}
        return {"video": video, "state": state, "language": language}

    def _decode_sim_action_chunk(self, action_dict: dict[str, np.ndarray], num_envs: int) -> np.ndarray:
        """Convert GR00T's (absolute) eef-pose action to the 8-D Abs-IK sim action chunk.

        Returns:
            ``(num_envs, action_horizon, 8)`` array: pos[3] + quat_xyzw[4] + gripper[1].
        """
        eef = np.asarray(action_dict[self.action_eef_key], dtype=np.float32)  # (N, H, 6)
        grip = np.asarray(action_dict[self.action_gripper_key], dtype=np.float32)  # (N, H, 1)
        pos = eef[..., :3]
        rotvec = eef[..., 3:6]
        quat_xyzw = _axisangle_to_quat_xyzw(rotvec)
        sim_action = np.concatenate([pos, quat_xyzw, grip[..., 0:1]], axis=-1).astype(np.float32)
        assert sim_action.shape[0] == num_envs and sim_action.shape[-1] == _SIM_ACTION_DIM
        return sim_action

    def get_action_chunk(self, observation: dict[str, Any]) -> torch.Tensor:
        """Query GR00T once and return the full sim action chunk ``(N, horizon, 8)``."""
        assert self.policy is not None, "GR00T policy has been closed"
        policy_observations = self._build_policy_observations(observation)
        num_envs = policy_observations["state"][self.state_eef_key].shape[0]
        action_dict, _ = self.policy.get_action(policy_observations)
        sim_action_np = self._decode_sim_action_chunk(action_dict, num_envs)
        action_tensor = to_tensor(sim_action_np, self.device)
        assert action_tensor.shape[0] == self.num_envs and action_tensor.shape[1] >= self.action_chunk_length
        return action_tensor

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Return the next single action per env from the current chunk (refilling as needed)."""

        def fetch_chunk() -> torch.Tensor:
            return self.get_action_chunk(observation)

        return self._chunking_state.get_action(fetch_chunk)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        assert self.policy is not None, "GR00T policy has been closed"
        self.policy.reset()
        self._chunking_state.reset(env_ids)

    def close(self) -> None:
        gr00t_policy = self.policy
        if gr00t_policy is not None:
            for attr_name in ("model", "processor", "collate_fn", "modality_configs"):
                if hasattr(gr00t_policy, attr_name):
                    setattr(gr00t_policy, attr_name, None)
        self.policy = None
        self._chunking_state = None
        gc.collect()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_gripper_key(keys: list[str]) -> str:
        candidates = [k for k in keys if "grip" in k.lower()]
        assert len(candidates) == 1, f"Expected exactly one gripper modality key in {keys}, found {candidates}"
        return candidates[0]

    @staticmethod
    def _find_other_key(keys: list[str], exclude: str) -> str:
        others = [k for k in keys if k != exclude]
        assert len(others) == 1, f"Expected exactly one non-gripper modality key in {keys}, found {others}"
        return others[0]
