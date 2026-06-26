# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""MDP functions for the self-contained LIBERO Arena env.

Most building blocks are reused from shared Isaac Lab packages (available in Arena):

* ``isaaclab.envs.mdp`` provides ``image``, ``time_out``, ``reset_scene_to_default``,
  ``last_action`` and the action cfgs (``JointPositionActionCfg``,
  ``BinaryJointPositionActionCfg``, ``AbsBinaryJointPositionActionCfg``).
* ``isaaclab_tasks...stack.mdp`` provides ``ee_frame_pose_in_base_frame`` and ``gripper_pos``.

Only the genuinely LIBERO-specific terms (goal evaluation + demo-based resets) are ported
locally in :mod:`.terminations` and :mod:`.events`.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.manipulation.stack.mdp import *  # noqa: F401, F403

from .events import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
