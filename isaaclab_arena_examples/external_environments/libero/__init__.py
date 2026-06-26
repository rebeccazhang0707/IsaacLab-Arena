# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Self-contained LIBERO single-task external environment for Isaac Lab Arena.

The full single-task LIBERO Abs-IK RL env config is implemented natively in this package
(no ``isaaclab_playground`` import). The Arena entry point is :class:`LiberoEnvironment`,
re-exported here so the class path stays
``isaaclab_arena_examples.external_environments.libero:LiberoEnvironment``.
"""

from .external_environment import LiberoEnvironment

__all__ = ["LiberoEnvironment"]
