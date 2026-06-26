# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""LIBERO goal-completion termination (self-contained Arena port).

Trimmed from the original ``...libero/mdp/terminations.py`` to the single-task
``libero_goals_reached`` predicate (the grouped multi-task variants are dropped).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .rl_utils import _articulation_operation_goal_satisfied, _relationship_goal_satisfied

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["libero_goals_reached"]


def libero_goals_reached(
    env: ManagerBasedRLEnv,
    goals: list[dict] | None = None,
):
    """Check whether a list of goals is achieved.

    Supports positional-relationship goals (put on / put in) and articulation-operation
    goals (e.g. close microwave, turn on stove).

    Args:
        env: The environment.
        goals: A list of goals to check.

    Returns:
        A tensor of shape (num_envs,) that is True only when all goals are achieved.
    """

    success = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    goals = goals or []

    for goal in goals:
        if "relationship" in goal:
            success = torch.logical_and(success, _relationship_goal_satisfied(env, goal))
        elif "operation" in goal:
            success = torch.logical_and(success, _articulation_operation_goal_satisfied(env, goal))

    return success
