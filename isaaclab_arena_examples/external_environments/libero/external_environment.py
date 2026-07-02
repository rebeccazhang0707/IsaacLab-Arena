# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Self-contained LIBERO single-task environment for Isaac Lab Arena.

This exposes the migrated single-task LIBERO Abs-IK RL config (:class:`AbsIKLiberoRLEnvCfg`,
ported in :mod:`.franka_libero_rl_env_cfg`) as an Arena external environment. Unlike a thin
wrapper, the env config is implemented *natively in this package* and only depends on shared
``isaaclab.*`` / ``isaaclab_tasks.*`` / ``isaaclab_assets.*`` packages — it never imports
``isaaclab_playground`` (whose version conflicts with Arena).

Because the ported config is already a complete ``ManagerBasedRLEnvCfg`` (LIBERO USD scene,
Abs-IK actions, agentview + eye-in-hand cameras, sparse ``libero_goals_reached`` reward), we
hand it to the Arena builder verbatim via ``IsaacLabArenaEnvironment.env_cfg_callback`` and
let the builder's ``NoEmbodiment`` / empty ``Scene`` / no-op ``LiberoTask`` contribute
nothing (all their ``modify_env_cfg`` hooks are no-ops).

This is a *homogeneous* task layout (every parallel env runs the same task selected by
``--task_suite`` / ``--task_id``), so none of the multi-task "per-env-class" machinery is
needed.

Run with the Arena policy runner via ``--external_environment_class_path``::

    python isaaclab_arena/evaluation/policy_runner.py \\
        --policy_type zero_action --num_steps 100 \\
        --external_environment_class_path \\
        isaaclab_arena_examples.external_environments.libero:LiberoEnvironment \\
        libero --task_suite libero_10 --task_id 0
"""

from __future__ import annotations

import argparse
import os

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# Candidate mounts for the ``libero_in_lab`` checkout that holds
# benchmarks/datasets/libero/{config,USD,assembled_hdf5}. Mirrors the resolution used by
# verl's isaac_env.py so the same env works across the root / non-root container layouts.
_LIBERO_ROOT_CANDIDATES = ("/libero_in_lab", "/root/libero_in_lab")

# Colocated LIBERO data embedded inside this package
# (``external_environments/libero/data/{config,USD,assembled_hdf5}``). Resolved relative to
# this module file so a local (non-docker) checkout works out of the box, independent of CWD.
# This is the lowest-precedence fallback (explicit env/CLI and /libero_in_lab mounts win).
_COLOCATED_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _resolve_libero_root() -> str | None:
    """Find the libero_in_lab root: explicit env var, else first candidate that has the config dir."""
    env_root = os.environ.get("LIBERO_IN_LAB_ROOT")
    if env_root:
        return env_root
    for candidate in _LIBERO_ROOT_CANDIDATES:
        if os.path.isdir(os.path.join(candidate, "benchmarks/datasets/libero/config")):
            return candidate
    return None


class LiberoTask(TaskBase):
    """Placeholder task carrying only the description / episode length.

    All config contributions (scene, observations, terminations, rewards, events) come from
    the ported ``AbsIKLiberoRLEnvCfg`` via ``env_cfg_callback``, so every getter here returns
    ``None`` (no contribution to the Arena composition).
    """

    name = "libero"

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return None

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        return None

    def get_metrics(self):
        return []


class LiberoEnvironment(ExampleEnvironmentBase):
    """LIBERO single-task RL environment exposed as an Arena external environment."""

    name: str = "libero"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment
        from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        # --- Configure LIBERO via env vars, which the ported LiberoTaskConfig /
        #     AbsIKLiberoRLEnvCfg read in __post_init__ at instantiation time (i.e. inside
        #     the callback below). ---
        self._configure_libero_env_vars(args_cli)

        task = LiberoTask(
            episode_length_s=getattr(args_cli, "episode_length_s", None),
            task_description=self._get_task_description(),
        )

        # The callback replaces the (empty) composed cfg with the full LIBERO RL cfg. The
        # Arena builder's task/embodiment/scene modify_env_cfg are all no-ops, so this is a
        # clean swap. Any per-run overrides must be applied to the *returned* cfg here.
        episode_length_s = getattr(args_cli, "episode_length_s", None)

        def _build_libero_env_cfg(_composed_cfg):
            from .franka_libero_rl_env_cfg import AbsIKLiberoRLEnvCfg

            cfg = AbsIKLiberoRLEnvCfg()
            if episode_length_s is not None:
                cfg.episode_length_s = episode_length_s
            return cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=NoEmbodiment(),
            scene=Scene(assets=[]),
            task=task,
            env_cfg_callback=_build_libero_env_cfg,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _configure_libero_env_vars(self, args_cli: argparse.Namespace) -> None:
        """Translate CLI args into the env vars LIBERO's config classes read.

        Uses ``setdefault`` semantics for the dataset directories (explicit env config wins)
        and hard-sets the per-run task selection / randomization knobs.
        """
        # Task selection (read by LiberoTaskConfig.__post_init__).
        os.environ["LIBERO_TASK_SUITE"] = str(args_cli.task_suite)
        os.environ["LIBERO_TASK_ID"] = str(args_cli.task_id)

        # Reset / randomization knobs (read by RLEventCfgFrankaPanda.__post_init__).
        os.environ["LIBERO_RANDOMIZE_OBJECT_POSE"] = "True" if args_cli.randomize_object_pose else "False"
        os.environ["ROBOT_INIT_NOISE_STD"] = str(args_cli.robot_init_noise_std)

        # Dataset / asset directories. Precedence (high -> low):
        #   explicit CLI flag > pre-set env var > /libero_in_lab mount > colocated `data/` dir.
        # ``setdefault`` keeps any pre-existing env var, and the colocated default ensures a
        # local (non-docker) checkout resolves the embedded data even without a mount.
        libero_root = self._resolve_libero_dirs(args_cli)
        if args_cli.libero_config_dir:
            os.environ["LIBERO_CONFIG_DIR"] = args_cli.libero_config_dir
        elif libero_root:
            os.environ.setdefault("LIBERO_CONFIG_DIR", os.path.join(libero_root, "benchmarks/datasets/libero/config"))
        else:
            os.environ.setdefault("LIBERO_CONFIG_DIR", os.path.join(_COLOCATED_DATA_DIR, "config"))

        if args_cli.libero_assets_dir:
            os.environ["LIBERO_ASSETS_DATA_DIR"] = args_cli.libero_assets_dir
        elif libero_root:
            os.environ.setdefault(
                "LIBERO_ASSETS_DATA_DIR", os.path.join(libero_root, "benchmarks/datasets/libero/USD")
            )
        else:
            os.environ.setdefault("LIBERO_ASSETS_DATA_DIR", os.path.join(_COLOCATED_DATA_DIR, "USD"))

        if args_cli.libero_assembled_dataset_dir:
            os.environ["LIBERO_ASSEMBLED_DATASET_DIR"] = args_cli.libero_assembled_dataset_dir
        elif libero_root:
            os.environ.setdefault(
                "LIBERO_ASSEMBLED_DATASET_DIR",
                os.path.join(libero_root, "benchmarks/datasets/libero/assembled_hdf5"),
            )
        else:
            os.environ.setdefault(
                "LIBERO_ASSEMBLED_DATASET_DIR", os.path.join(_COLOCATED_DATA_DIR, "assembled_hdf5")
            )

    @staticmethod
    def _resolve_libero_dirs(args_cli: argparse.Namespace) -> str | None:
        if getattr(args_cli, "libero_in_lab_root", None):
            return args_cli.libero_in_lab_root
        return _resolve_libero_root()

    @staticmethod
    def _get_task_description() -> str | None:
        """Read the language instruction for the selected task (best-effort)."""
        try:
            from .franka_libero_base_cfg import LiberoTaskConfig

            return LiberoTaskConfig().task_info.get("language_instruction")
        except Exception:
            return None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--task_suite", type=str, default="libero_10", help="LIBERO task suite name.")
        parser.add_argument("--task_id", type=int, default=0, help="LIBERO task id within the suite.")
        parser.add_argument(
            "--randomize_object_pose",
            action="store_true",
            help="Randomize object poses on reset (otherwise demo-sampled initial states).",
        )
        parser.add_argument(
            "--robot_init_noise_std",
            type=float,
            default=0.0,
            help="Std of Gaussian noise added to the robot joint init pose on reset.",
        )
        parser.add_argument("--episode_length_s", type=float, default=None, help="Override episode length (s).")
        parser.add_argument(
            "--libero_in_lab_root",
            type=str,
            default=None,
            help="Root of the libero_in_lab checkout (overrides auto-detection).",
        )
        parser.add_argument("--libero_config_dir", type=str, default=None, help="Override LIBERO_CONFIG_DIR.")
        parser.add_argument("--libero_assets_dir", type=str, default=None, help="Override LIBERO_ASSETS_DATA_DIR.")
        parser.add_argument(
            "--libero_assembled_dataset_dir",
            type=str,
            default=None,
            help="Override LIBERO_ASSEMBLED_DATASET_DIR (HDF5 demos for state reset).",
        )
