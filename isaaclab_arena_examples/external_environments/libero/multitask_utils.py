# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""LIBERO task-assignment helpers and articulation metadata (self-contained Arena port).

This is the minimal subset of the original
``isaaclab_playground...config.franka.multitask_utils`` needed by the single-task
Abs-IK LIBERO RL env: the (suite, task_id) encode/decode helpers and the articulation
joint metadata used when resetting articulated scene objects. The heavyweight
``MultiTaskLiberoTaskConfig`` / robot ``ArticulationCfg`` definitions are intentionally
omitted because the single-task RL config does not use them.
"""

from __future__ import annotations

TASK_ASSIGNMENT_DELIMITER = "::"


def encode_task_assignment(suite_name: str, task_id: int) -> str:
    """Create a string key that uniquely identifies a (suite, task_id) pair."""
    return f"{suite_name}{TASK_ASSIGNMENT_DELIMITER}{int(task_id)}"


def decode_task_assignment(assignment: str | tuple[str, int]) -> tuple[str, int]:
    """Decode a task assignment string (or tuple for backward compatibility)."""
    if isinstance(assignment, tuple):
        suite_name, task_id = assignment
        return suite_name, int(task_id)

    if not isinstance(assignment, str):
        raise TypeError(f"Unsupported assignment type: {type(assignment)}.")

    if TASK_ASSIGNMENT_DELIMITER not in assignment:
        raise ValueError(f"Assignment '{assignment}' is missing delimiter '{TASK_ASSIGNMENT_DELIMITER}'.")

    suite_name, _, task_id_str = assignment.rpartition(TASK_ASSIGNMENT_DELIMITER)
    if suite_name == "" or task_id_str == "":
        raise ValueError(f"Assignment '{assignment}' is not in '<suite>{TASK_ASSIGNMENT_DELIMITER}<id>' format.")
    return suite_name, int(task_id_str)


# Articulated LIBERO scene objects and their default joint positions / valid joint set.
ARTICULATION_TYPES = {"flat_stove", "microwave", "white_cabinet", "wooden_cabinet"}
ARTICULATION_DEFAULT_JOINTS = {
    "flat_stove": {"button": 0.0},
    "microwave": {"microjoint": -1.79},
    "white_cabinet": {"top_level": 0.0, "middle_level": 0.0, "bottom_level": -0.1523},
    "wooden_cabinet": {"top_level": 0.0, "middle_level": 0.0, "bottom_level": 0.0},
}
